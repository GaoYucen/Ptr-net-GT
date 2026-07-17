from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from ptrnet_gt.config import apply_overrides, load_config
from ptrnet_gt.models.factory import build_model
from ptrnet_gt.problems import TSP
from ptrnet_gt.utils import evaluate_tour_batch
from ptrnet_gt.utils.data import load_dataset_payload, training_budget_from_config
from ptrnet_gt.group_theory.permutation import inverse_permutation, permute_nodes
from ptrnet_gt.utils.tour_metrics import build_successor, edges_from_pi


def _parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_reference_costs(path: str | None):
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("costs", "ref_costs", "lengths"):
            if key in payload:
                return torch.as_tensor(payload[key], dtype=torch.float32)
    return torch.as_tensor(payload, dtype=torch.float32)


def _evaluate_pi_for_model(model_name: str, batch: torch.Tensor, pi: torch.Tensor, cost: torch.Tensor) -> dict:
    if model_name == "component_merge":
        return evaluate_tour_batch(batch, pi, model_cost=cost)

    recomputed_cost, _ = TSP.get_costs(batch, pi)
    feasible = torch.ones(batch.size(0), dtype=torch.bool, device=batch.device)
    cost_error = (cost - recomputed_cost).abs()
    return {
        "feasible": feasible,
        "cost_error": cost_error,
    }


def _permutation_consistency(model, batch: torch.Tensor) -> float:
    n = batch.size(1)
    perms = torch.stack([torch.randperm(n, device=batch.device) for _ in range(batch.size(0))], dim=0)
    with torch.no_grad():
        _, _, pi = model(batch, return_pi=True)
        permuted = torch.stack([permute_nodes(batch[i], perms[i]) for i in range(batch.size(0))], dim=0)
        _, _, pi_perm = model(permuted, return_pi=True)
    tails, heads = edges_from_pi(pi)
    ptails, pheads = edges_from_pi(pi_perm)
    exact = []
    for i in range(batch.size(0)):
        inv = inverse_permutation(perms[i])
        mapped_t = inv[ptails[i]]
        mapped_h = inv[pheads[i]]
        s1 = build_successor(tails[i:i+1], heads[i:i+1])[0]
        s2 = build_successor(mapped_t.unsqueeze(0), mapped_h.unsqueeze(0))[0]
        exact.append((s1 == s2).all())
    return float(torch.stack(exact).float().mean().item())


def main():
    parser = argparse.ArgumentParser(description="Ptr-net-GT unified evaluation entry")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint saved by scripts/train.py")
    parser.add_argument("--override", action="append", default=[], help="Override config entries")
    parser.add_argument("--dataset", default=None, help="Fixed dataset .pt file")
    parser.add_argument("--reference", default=None, help="Optional reference cost tensor/.pt for gap computation")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.override)
    problem = TSP()
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.get("training", {}).get("use_cuda", True) else "cpu")
    model = build_model(config, problem).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    model.set_decode_type("greedy")
    model_name = config.get("model", {}).get("name", "unknown")

    dataset_path = args.dataset or config.get("evaluation", {}).get("dataset") or config.get("data", {}).get("test_dataset")
    dataset_metadata = {}
    if dataset_path:
        dataset_payload = load_dataset_payload(dataset_path)
        dataset = problem.make_dataset(filename=dataset_path, num_samples=dataset_payload["num_instances"])
        dataset_metadata = {k: v for k, v in dataset_payload.items() if k != "coords"}
    else:
        dataset = problem.make_dataset(
            size=config.get("problem", {}).get("size", 20),
            num_samples=config.get("evaluation", {}).get("num_instances", 128),
            distribution=config.get("problem", {}).get("distribution"),
        )
    dataloader = DataLoader(dataset, batch_size=config.get("evaluation", {}).get("batch_size", 128))

    costs = []
    feasible = []
    cost_errors = []
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    import time
    wall_start = time.perf_counter()
    if start is not None:
        start.record()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            cost, _, pi = model(batch, return_pi=True)
            metrics = _evaluate_pi_for_model(model_name, batch, pi, cost)
            costs.append(cost.cpu())
            feasible.append(metrics["feasible"].cpu())
            cost_errors.append(metrics["cost_error"].cpu())
    if end is not None:
        end.record()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - wall_start

    costs = torch.cat(costs, dim=0)
    feasible = torch.cat(feasible, dim=0)
    cost_errors = torch.cat(cost_errors, dim=0)
    reference_costs = _load_reference_costs(args.reference)
    gap = None
    if reference_costs is not None:
        reference_costs = reference_costs[: costs.size(0)]
        gap = float((((costs - reference_costs) / reference_costs).mean() * 100.0).item())
    budget = payload.get("budget") or training_budget_from_config(config)
    result = {
        "method": model_name,
        "mean_tour_length": float(costs.mean().item()),
        "std_tour_length": float(costs.std().item()),
        "avg_cost": float(costs.mean().item()),
        "std_cost": float(costs.std().item()),
        "optimality_gap_percent": gap,
        "feasible_tour_rate": float(feasible.float().mean().item()),
        "mean_cost_consistency_error": float(cost_errors.mean().item()),
        "max_cost_consistency_error": float(cost_errors.max().item()),
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(costs.numel(), 1)),
        "throughput_instances_per_sec": float(costs.numel() / elapsed) if elapsed > 0 else None,
        "parameter_count": _parameter_count(model),
        "training_updates": budget.get("training_updates"),
        "training_samples": budget.get("training_samples"),
        "seed": budget.get("seed"),
        "num_instances": int(costs.numel()),
        "size": int(config.get("problem", {}).get("size", dataset_metadata.get("size", -1))),
        "dataset": dataset_path,
        "dataset_metadata": dataset_metadata,
        "checkpoint": args.checkpoint,
        "permutation_consistency": None,
    }
    # small consistency probe on first batch
    probe_batch = next(iter(dataloader)).to(device)
    result["permutation_consistency"] = _permutation_consistency(model, probe_batch)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_root = Path(config.get("output", {}).get("root", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "last_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()