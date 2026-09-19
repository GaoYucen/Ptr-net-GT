"""训练可断点续训的 Native Original 或 canonical GroupOpt 模型。"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from groupopt.models.am import AttentionModel
from groupopt.objectives import reinforce_loss
from groupopt.problems import BatchedCVRPConstruction, BatchedMCycleCoverConstruction
from groupopt.problems.distributions import generate_cvrp_instances

# 这里只保留论文有效范式及其必要对照，不再接纳历史 Adapter/pilot 名称。
BASE_MODES = (
    "native_original",
    "native_conditional_fixed",
    "native_conditional_free",
    "native_capacity_single_chain",
    "native_forest_fixed",
    "native_free_no_head_summary",
    "native_free_no_path_state",
    "native_free_no_last_head",
    "native_random_tail",
)


def evaluate(
    model: nn.Module,
    coordinates: torch.Tensor,
    base_mode: str,
    generator_seed: int,
) -> float:
    """在固定验证集上执行确定性 greedy 评估。"""
    model.eval()
    action_generator = torch.Generator(device=coordinates.device).manual_seed(generator_seed)
    with torch.inference_mode():
        best_rollout_cost = getattr(model, "best_rollout_cost", None)
        if best_rollout_cost is not None:
            chunk_size = int(getattr(model, "evaluation_batch_size", 1))
            costs = []
            for start in range(0, coordinates.size(0), chunk_size):
                output = model(
                    coordinates[start : start + chunk_size],
                    decode_type="greedy",
                    base_mode=base_mode,
                    generator=action_generator,
                )
                costs.append(best_rollout_cost(output))
            return torch.cat(costs).mean().item()
        output = model(
            coordinates,
            decode_type="greedy",
            base_mode=base_mode,
            generator=action_generator,
        )
    return output.cost.mean().item()


def build_am_model(args: argparse.Namespace) -> nn.Module:
    if args.problem == "tsp":
        input_dim = 2
        construction_process = None
    elif args.problem == "cvrp":
        input_dim = 5
        construction_process = BatchedCVRPConstruction()
    elif args.problem == "min_m_ccp":
        input_dim = 2
        construction_process = BatchedMCycleCoverConstruction(args.cycles)
    else:
        raise ValueError(f"unknown problem: {args.problem}")
    return AttentionModel(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_encoder_layers=args.encoder_layers,
        feed_forward_dim=args.feed_forward_dim,
        normalization=args.normalization,
        construction_process=construction_process,
    )


def generate_instances(
    args: argparse.Namespace,
    sample_count: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """生成训练或验证实例，且不接触全局随机数状态。"""
    if args.problem in {"tsp", "min_m_ccp"}:
        return torch.rand(
            sample_count,
            args.graph_size,
            2,
            device=device,
            generator=generator,
        )
    if args.problem == "cvrp":
        return generate_cvrp_instances(
            sample_count, args.graph_size, args.capacity, generator, device
        )
    raise ValueError(f"unknown problem: {args.problem}")


def run(
    args: argparse.Namespace,
    model_builder: Callable[[argparse.Namespace], nn.Module] = build_am_model,
) -> None:
    """按统一协议训练一个宿主模型，并保存完整的恢复状态。"""
    _validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    config = _experiment_config(args, device)
    _write_or_validate_config(output_dir / "config.json", config, args.resume is not None)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    model = model_builder(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    data_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    validation_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    action_generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    validation = generate_instances(
        args, args.validation_size, device, validation_generator
    )

    start_step = 0
    best_cost = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_cost = float(checkpoint["best_cost"])
        data_generator.set_state(checkpoint["data_generator_state"].cpu())
        action_generator.set_state(checkpoint["action_generator_state"].cpu())
        if "cpu_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["cpu_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_states" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])

    metrics_path = output_dir / "metrics.jsonl"
    started_at = time.monotonic()
    if start_step == 0:
        initial_cost = evaluate(model, validation, args.base_mode, args.seed + 4)
        _append_metric(
            metrics_path,
            {"step": 0, "validation_greedy_cost": initial_cost, "elapsed_seconds": 0.0},
        )
        best_cost = initial_cost
        _save_checkpoint(
            checkpoint_dir / "best.pt",
            model,
            optimizer,
            0,
            best_cost,
            config,
            data_generator,
            action_generator,
        )
        print(f"step=000000 val_greedy_cost={initial_cost:.6f}", flush=True)

    step = start_step
    try:
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            coordinates = generate_instances(
                args, args.batch_size, device, data_generator
            )
            output = model(
                coordinates,
                decode_type="sampling",
                base_mode=args.base_mode,
                generator=action_generator,
            )
            if args.training_scheme == "pomo":
                pomo_policy_loss = getattr(model, "pomo_policy_loss", None)
                if pomo_policy_loss is None:
                    raise RuntimeError("POMO 训练必须使用提供 pomo_policy_loss 的模型")
                policy_loss = pomo_policy_loss(output)
            else:
                policy_loss = reinforce_loss(output.cost, output.log_likelihood)
            tail_entropy_bonus = args.tail_entropy_coefficient * output.tail_entropy.mean()
            loss = policy_loss - tail_entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()

            should_evaluate = step % args.eval_every == 0 or step == args.steps
            if should_evaluate:
                validation_cost = evaluate(model, validation, args.base_mode, args.seed + 4)
                is_new_best = validation_cost < best_cost
                if is_new_best:
                    best_cost = validation_cost
                metric = {
                    "step": step,
                    "train_cost": output.cost.mean().item(),
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "gradient_norm": float(gradient_norm),
                    "validation_greedy_cost": validation_cost,
                    "best_validation_greedy_cost": best_cost,
                    "elapsed_seconds": time.monotonic() - started_at,
                    "peak_gpu_memory_gb": _peak_memory_gb(device),
                    "tail_entropy": output.tail_entropy.mean().item(),
                    "action_entropy": output.action_entropy.mean().item(),
                    "tail_entropy_bonus": tail_entropy_bonus.item(),
                    "effective_batch_size": int(output.cost.shape[0]),
                }
                _append_metric(metrics_path, metric)
                print(
                    f"step={step:06d} "
                    f"train_cost={metric['train_cost']:.6f} "
                    f"loss={metric['loss']:.6f} "
                    f"val_greedy_cost={validation_cost:.6f} "
                    f"best={best_cost:.6f} "
                    f"tail_entropy={metric['tail_entropy']:.4f} "
                    f"action_entropy={metric['action_entropy']:.4f} "
                    f"peak_gb={metric['peak_gpu_memory_gb']:.3f}",
                    flush=True,
                )
                if is_new_best:
                    _save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        step,
                        best_cost,
                        config,
                        data_generator,
                        action_generator,
                    )

            if step % args.checkpoint_every == 0 or step == args.steps:
                payload = _checkpoint_payload(
                    model,
                    optimizer,
                    step,
                    best_cost,
                    config,
                    data_generator,
                    action_generator,
                )
                _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
                _atomic_torch_save(payload, checkpoint_dir / f"step-{step:06d}.pt")
    except KeyboardInterrupt:
        payload = _checkpoint_payload(
            model,
            optimizer,
            step,
            best_cost,
            config,
            data_generator,
            action_generator,
        )
        _atomic_torch_save(payload, checkpoint_dir / "interrupted.pt")
        _atomic_torch_save(payload, checkpoint_dir / "latest.pt")
        raise


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_cost: float,
    config: dict[str, Any],
    data_generator: torch.Generator,
    action_generator: torch.Generator,
) -> dict[str, Any]:
    """保存模型、优化器和所有会影响继续训练的随机状态。"""
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_cost": best_cost,
        "config": config,
        "data_generator_state": data_generator.get_state(),
        "action_generator_state": action_generator.get_state(),
        "cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    return payload


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_cost: float,
    config: dict[str, Any],
    data_generator: torch.Generator,
    action_generator: torch.Generator,
) -> None:
    _atomic_torch_save(
        _checkpoint_payload(
            model,
            optimizer,
            step,
            best_cost,
            config,
            data_generator,
            action_generator,
        ),
        path,
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")


def _write_or_validate_config(path: Path, config: dict[str, Any], resuming: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_without_horizon = {key: value for key, value in existing.items() if key != "steps"}
        config_without_horizon = {key: value for key, value in config.items() if key != "steps"}
        if existing_without_horizon != config_without_horizon:
            raise ValueError("续训配置与已有实验不一致")
        if not resuming:
            raise FileExistsError("输出目录已有实验；如需续训请传入 --resume")
        if int(config["steps"]) < int(existing["steps"]):
            raise ValueError("续训的最终 step 不能小于原实验")
        if config != existing:
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if resuming:
        raise ValueError("续训目录中缺少 config.json")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _experiment_config(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = {
        key: value for key, value in vars(args).items() if key not in {"resume", "output_dir"}
    }
    if config.get("model") != "pomo":
        config.pop("qkv_dim", None)
        config.pop("pomo_size", None)
    if float(config.get("tail_entropy_coefficient", 0.0)) == 0.0:
        config.pop("tail_entropy_coefficient", None)
    if config.get("problem") == "tsp":
        # 保持既有 canonical TSP checkpoint 的配置完全兼容。
        config.pop("problem", None)
        config.pop("cycles", None)
        config.pop("capacity", None)
    elif config.get("problem") == "cvrp":
        config.pop("cycles", None)
    elif config.get("problem") == "min_m_ccp":
        config.pop("capacity", None)
    config["device"] = str(device)
    if args.training_scheme == "pomo":
        config["effective_batch_size"] = int(args.batch_size) * int(args.pomo_size)
    else:
        # 与已经完成的 canonical 主实验配置保持兼容。
        config.pop("training_scheme", None)
    return config


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前环境不可用")
    return device


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.graph_size,
        args.steps,
        args.batch_size,
        args.validation_size,
        args.eval_every,
        args.checkpoint_every,
    )
    if min(positive) < 1:
        raise ValueError("规模、step 和记录间隔必须为正数")
    if args.base_mode not in BASE_MODES:
        raise ValueError(f"base_mode 必须属于 {BASE_MODES}")
    if args.training_scheme == "pomo" and getattr(args, "model", None) != "pomo":
        raise ValueError("POMO 训练方案只能用于 POMO 宿主")
    if args.training_scheme != "pomo" and getattr(args, "model", None) == "pomo":
        raise ValueError("POMO 宿主必须使用 --training-scheme pomo")
    if args.pomo_size < 1 or args.pomo_size > args.graph_size:
        raise ValueError("pomo_size 必须位于 [1, graph_size]")
    if args.tail_entropy_coefficient < 0:
        raise ValueError("tail entropy 系数不能为负")
    if args.problem == "min_m_ccp" and args.graph_size < 3 * args.cycles:
        raise ValueError("Min-m-CCP 至少需要 3m 个节点")
    if args.problem == "cvrp" and args.capacity < 9:
        raise ValueError("CVRP capacity 必须至少容纳最大单点需求 9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-mode", choices=BASE_MODES, required=True)
    parser.add_argument(
        "--problem", choices=("tsp", "cvrp", "min_m_ccp"), default="tsp"
    )
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=40)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--graph-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--training-scheme", choices=("reinforce", "pomo"), default="reinforce"
    )
    parser.add_argument("--validation-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--feed-forward-dim", type=int, default=512)
    parser.add_argument("--qkv-dim", type=int, default=16)
    parser.add_argument("--pomo-size", type=int, default=8)
    parser.add_argument("--normalization", choices=("batch", "layer"), default="batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--tail-entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
