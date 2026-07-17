from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ptrnet_gt.baselines import (
    nearest_neighbor_multistart_tour,
    nearest_neighbor_tour,
    nearest_neighbor_two_opt_tour,
    random_tour,
    tour_length,
)
from ptrnet_gt.config import apply_overrides, load_config
from ptrnet_gt.models.factory import build_model
from ptrnet_gt.problems import TSP
from ptrnet_gt.utils import evaluate_tour_batch
from ptrnet_gt.utils.data import load_dataset_payload, training_budget_from_config


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


def _gap_percent(costs: torch.Tensor, reference_costs: torch.Tensor | None):
    if reference_costs is None:
        return None
    reference_costs = reference_costs[: costs.size(0)]
    return float((((costs - reference_costs) / reference_costs).mean() * 100.0).item())


def _set_seed(seed: int):
    torch.manual_seed(seed)


def _evaluate_tour_method(name: str, coords: torch.Tensor, tour: torch.Tensor, elapsed: float, reference_costs: torch.Tensor | None = None) -> dict:
    edge_cost, _ = TSP.get_costs(coords, tour)
    feasible = torch.ones(coords.size(0), dtype=torch.bool)
    return {
        "method": name,
        "mean_tour_length": float(edge_cost.mean().item()),
        "std_tour_length": float(edge_cost.std().item()),
        "avg_cost": float(edge_cost.mean().item()),
        "std_cost": float(edge_cost.std().item()),
        "optimality_gap_percent": _gap_percent(edge_cost, reference_costs),
        "feasible_tour_rate": float(feasible.float().mean().item()),
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(coords.size(0), 1)),
        "throughput_instances_per_sec": float(coords.size(0) / elapsed) if elapsed > 0 else None,
        "parameter_count": None,
        "training_updates": None,
        "training_samples": None,
        "num_instances": int(coords.size(0)),
        "size": int(coords.size(1)),
    }


def _evaluate_component_merge(config_path: str, checkpoint_path: str, overrides: list[str], dataset_path: str | None = None, reference_costs: torch.Tensor | None = None) -> dict:
    config = apply_overrides(load_config(config_path), overrides)
    problem = TSP()
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.get("training", {}).get("use_cuda", True) else "cpu")
    model = build_model(config, problem).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    model.set_decode_type("greedy")

    if dataset_path:
        dataset_payload = load_dataset_payload(dataset_path)
        dataset = problem.make_dataset(filename=dataset_path, num_samples=dataset_payload["num_instances"])
    else:
        dataset = problem.make_dataset(
            size=config.get("problem", {}).get("size", 20),
            num_samples=config.get("evaluation", {}).get("num_instances", 128),
            distribution=config.get("problem", {}).get("distribution"),
        )
    coords = torch.stack([dataset[i] for i in range(len(dataset))], dim=0).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        cost, _, pi = model(coords, return_pi=True)
        metrics = evaluate_tour_batch(coords, pi, model_cost=cost)
    elapsed = time.perf_counter() - start
    budget = payload.get("budget") or training_budget_from_config(config)
    return {
        "method": "component_merge",
        "mean_tour_length": float(cost.mean().item()),
        "std_tour_length": float(cost.std().item()),
        "avg_cost": float(cost.mean().item()),
        "std_cost": float(cost.std().item()),
        "optimality_gap_percent": _gap_percent(cost.detach().cpu(), reference_costs),
        "feasible_tour_rate": float(metrics["feasible"].float().mean().item()),
        "mean_cost_consistency_error": float(metrics["cost_error"].mean().item()),
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(coords.size(0), 1)),
        "throughput_instances_per_sec": float(coords.size(0) / elapsed) if elapsed > 0 else None,
        "parameter_count": _parameter_count(model),
        "training_updates": budget.get("training_updates"),
        "training_samples": budget.get("training_samples"),
        "seed": budget.get("seed"),
        "num_instances": int(coords.size(0)),
        "size": int(coords.size(1)),
        "dataset": dataset_path,
        "checkpoint": checkpoint_path,
    }


def _evaluate_neural_tour_model(method_name: str, config_path: str, checkpoint_path: str, overrides: list[str], dataset_path: str | None = None, reference_costs: torch.Tensor | None = None) -> dict:
    config = apply_overrides(load_config(config_path), overrides)
    problem = TSP()
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.get("training", {}).get("use_cuda", True) else "cpu")
    model = build_model(config, problem).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    model.set_decode_type("greedy")

    if dataset_path:
        dataset_payload = load_dataset_payload(dataset_path)
        dataset = problem.make_dataset(filename=dataset_path, num_samples=dataset_payload["num_instances"])
    else:
        dataset = problem.make_dataset(
            size=config.get("problem", {}).get("size", 20),
            num_samples=config.get("evaluation", {}).get("num_instances", 128),
            distribution=config.get("problem", {}).get("distribution"),
        )
    coords = torch.stack([dataset[i] for i in range(len(dataset))], dim=0).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        cost, _, pi = model(coords, return_pi=True)
    elapsed = time.perf_counter() - start
    budget = payload.get("budget") or training_budget_from_config(config)
    return {
        "method": method_name,
        "mean_tour_length": float(cost.mean().item()),
        "std_tour_length": float(cost.std().item()),
        "avg_cost": float(cost.mean().item()),
        "std_cost": float(cost.std().item()),
        "optimality_gap_percent": _gap_percent(cost.detach().cpu(), reference_costs),
        "feasible_tour_rate": 1.0,
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(coords.size(0), 1)),
        "throughput_instances_per_sec": float(coords.size(0) / elapsed) if elapsed > 0 else None,
        "parameter_count": _parameter_count(model),
        "training_updates": budget.get("training_updates"),
        "training_samples": budget.get("training_samples"),
        "seed": budget.get("seed"),
        "num_instances": int(coords.size(0)),
        "size": int(coords.size(1)),
        "dataset": dataset_path,
        "checkpoint": checkpoint_path,
        "tour_shape": list(pi.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate traditional baselines for TSP-only codebase")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--methods", default="random,nearest_neighbor,nearest_neighbor_multistart,nn_two_opt")
    parser.add_argument("--output", default="outputs/baseline_comparison/tsp_baselines.json")
    parser.add_argument("--component-merge-config", default=None)
    parser.add_argument("--component-merge-checkpoint", default=None)
    parser.add_argument("--pointer-network-config", default=None)
    parser.add_argument("--pointer-network-checkpoint", default=None)
    parser.add_argument("--attention-model-config", default=None)
    parser.add_argument("--attention-model-checkpoint", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    _set_seed(args.seed)
    reference_costs = _load_reference_costs(args.reference)
    if args.dataset:
        dataset_payload = load_dataset_payload(args.dataset)
        coords = dataset_payload["coords"]
    else:
        dataset_payload = None
        problem = TSP()
        dataset = problem.make_dataset(size=args.size, num_samples=args.num_instances)
        coords = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = []

    for method in methods:
        start = time.perf_counter()
        if method == "random":
            tour = random_tour(coords)
        elif method == "nearest_neighbor":
            tour = nearest_neighbor_tour(coords)
        elif method == "nearest_neighbor_multistart":
            tour = nearest_neighbor_multistart_tour(coords)
        elif method == "nn_two_opt":
            tour = nearest_neighbor_two_opt_tour(coords)
        else:
            raise ValueError(f"Unsupported method: {method}")
        elapsed = time.perf_counter() - start
        results.append(_evaluate_tour_method(method, coords, tour, elapsed, reference_costs=reference_costs))

    if args.component_merge_config and args.component_merge_checkpoint:
        results.append(_evaluate_component_merge(args.component_merge_config, args.component_merge_checkpoint, args.override, dataset_path=args.dataset, reference_costs=reference_costs))
    if args.pointer_network_config and args.pointer_network_checkpoint:
        results.append(_evaluate_neural_tour_model("pointer_network", args.pointer_network_config, args.pointer_network_checkpoint, args.override, dataset_path=args.dataset, reference_costs=reference_costs))
    if args.attention_model_config and args.attention_model_checkpoint:
        results.append(_evaluate_neural_tour_model("attention_model", args.attention_model_config, args.attention_model_checkpoint, args.override, dataset_path=args.dataset, reference_costs=reference_costs))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "size": int(coords.size(1)),
        "num_instances": int(coords.size(0)),
        "dataset": args.dataset,
        "dataset_metadata": {k: v for k, v in (dataset_payload or {}).items() if k != "coords"},
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()