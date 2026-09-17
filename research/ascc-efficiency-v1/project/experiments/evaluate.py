"""在固定的独立 TSP 测试集上评估一个训练 checkpoint。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from groupopt.adapters import build_model
from groupopt.problems.distributions import (
    TSP_DISTRIBUTIONS,
    generate_cvrp_instances,
    generate_tsp_coordinates,
)


def generate_test_instances(
    config: dict[str, Any],
    sample_count: int,
    graph_size: int,
    distribution: str,
    seed: int,
) -> torch.Tensor:
    """按 checkpoint 中冻结的问题定义生成独立 CPU 测试集。"""
    problem = str(config.get("problem", "tsp"))
    if problem in {"tsp", "min_m_ccp"}:
        return generate_tsp_coordinates(sample_count, graph_size, distribution, seed)
    if problem == "cvrp":
        if distribution != "uniform":
            raise ValueError("CVRP pilot currently supports only uniform coordinates")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return generate_cvrp_instances(
            sample_count,
            graph_size,
            int(config["capacity"]),
            generator,
            "cpu",
        )
    raise ValueError(f"unknown problem in checkpoint: {problem}")


def load_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
    evaluation_mode: str,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    # 早期 checkpoint 曾保存两个现已删除、且不参与 native conditional
    # 前向计算的诊断模块。只放行这两个精确前缀，其他未知参数仍视为错误。
    allowed_unexpected_prefixes = ["tail_gate.", "joint_action_scorer."]
    unsafe_missing = list(incompatible.missing_keys)
    unsafe_unexpected = [
        name
        for name in incompatible.unexpected_keys
        if not name.startswith(tuple(allowed_unexpected_prefixes))
    ]
    if unsafe_missing or unsafe_unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={unsafe_missing}, unexpected={unsafe_unexpected}"
        )
    model.eval()
    return model, checkpoint


def evaluate(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_graph_size = int(config["graph_size"])
    if training_graph_size != args.graph_size and not args.allow_size_transfer:
        raise ValueError(
            "test graph size differs from checkpoint training graph size; "
            "pass --allow-size-transfer only for the explicit zero-shot size-transfer protocol"
        )

    evaluation_mode = args.base_mode_override or config["base_mode"]
    device = resolve_device(args.device)
    model, checkpoint = load_model(checkpoint_path, config, device, evaluation_mode)
    coordinates = generate_test_instances(
        config,
        args.test_size,
        args.graph_size,
        args.distribution,
        args.test_seed,
    )

    costs: list[torch.Tensor] = []
    action_generator = torch.Generator(device=device).manual_seed(args.test_seed + 1)
    with torch.no_grad():
        for start in range(0, args.test_size, args.batch_size):
            batch = coordinates[start : start + args.batch_size].to(device)
            output = model(
                batch,
                decode_type="greedy",
                base_mode=evaluation_mode,
                generator=action_generator,
            )
            best_rollout_cost = getattr(model, "best_rollout_cost", None)
            evaluated_cost = (
                best_rollout_cost(output) if best_rollout_cost is not None else output.cost
            )
            costs.append(evaluated_cost.cpu())
    cost_tensor = torch.cat(costs)
    if cost_tensor.shape != (args.test_size,) or not torch.isfinite(cost_tensor).all():
        raise RuntimeError("independent test produced invalid costs")

    standard_deviation = cost_tensor.std(unbiased=True).item()
    summary = {
        "base_mode": evaluation_mode,
        "training_base_mode": config["base_mode"],
        "best_validation_cost": float(checkpoint["best_cost"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "distribution": args.distribution,
        "graph_size": args.graph_size,
        "training_graph_size": training_graph_size,
        "is_size_transfer": training_graph_size != args.graph_size,
        "mean_cost": cost_tensor.mean().item(),
        "model": config.get("model", "am"),
        "problem": config.get("problem", "tsp"),
        "standard_deviation": standard_deviation,
        "standard_error": standard_deviation / math.sqrt(args.test_size),
        "test_seed": args.test_seed,
        "evaluation_action_seed": args.test_seed + 1,
        "test_size": args.test_size,
        "train_seed": int(config["seed"]),
        "training_scheme": config.get("training_scheme", "reinforce"),
    }
    if "cycles" in config:
        summary["cycles"] = int(config["cycles"])
    if "capacity" in config:
        summary["capacity"] = int(config["capacity"])
    atomic_torch_save(
        {
            "costs": cost_tensor,
            "metadata": summary,
        },
        output_dir / "costs.pt",
    )
    atomic_text_write(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        output_dir / "summary.json",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def atomic_text_write(text: str, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--test-seed", type=int, default=20260805)
    parser.add_argument("--distribution", choices=TSP_DISTRIBUTIONS, default="uniform")
    parser.add_argument("--graph-size", type=int, default=50)
    parser.add_argument(
        "--allow-size-transfer",
        action="store_true",
        help="明确允许训练规模与测试规模不同，仅用于 zero-shot 规模迁移实验",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--base-mode-override",
        help="用同一 checkpoint 的另一构造模式评估，供严格嵌套对照使用",
    )
    arguments = parser.parse_args()
    if min(arguments.test_size, arguments.graph_size, arguments.batch_size) < 1:
        parser.error("test size, graph size, and batch size must be positive")
    return arguments


if __name__ == "__main__":
    evaluate(parse_args())
