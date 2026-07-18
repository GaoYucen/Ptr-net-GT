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
from ptrnet_gt.models import component_merge_beam_search_batched
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


def _edge_matrix_from_pi(pi: torch.Tensor, n_nodes: int) -> torch.Tensor:
    tails, heads = edges_from_pi(pi)
    edges = torch.zeros(pi.size(0), n_nodes, n_nodes, dtype=torch.bool, device=pi.device)
    batch_idx = torch.arange(pi.size(0), device=pi.device)[:, None].expand_as(tails)
    edges[batch_idx, tails, heads] = True
    return edges


def _permute_edge_matrix_back(edge_matrix: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    inv = inverse_permutation(permutation)
    return edge_matrix[inv][:, inv]


def _permute_flat_edge_scores_back(flat_scores: torch.Tensor, permutation: torch.Tensor, n_nodes: int) -> torch.Tensor:
    inv = inverse_permutation(permutation)
    matrix = flat_scores.view(n_nodes, n_nodes)
    return matrix[inv][:, inv].reshape(-1)


def _permutation_metrics(model, batch: torch.Tensor) -> dict:
    n = batch.size(1)
    perms = torch.stack([torch.randperm(n, device=batch.device) for _ in range(batch.size(0))], dim=0)
    with torch.no_grad():
        _, cost, pi = None, None, None
        cost, _, pi = model(batch, return_pi=True)
        permuted = torch.stack([permute_nodes(batch[i], perms[i]) for i in range(batch.size(0))], dim=0)
        cost_perm, _, pi_perm = model(permuted, return_pi=True)

        embeddings = model.encode(batch)
        perm_embeddings = model.encode(permuted)
        state = model.problem.make_state(batch) if hasattr(model.problem, "make_state") else None
    exact = []
    jaccards = []
    recalls = []
    rel_length_diffs = []
    prob_equiv_errors = []
    edge_matrix = _edge_matrix_from_pi(pi, n)
    edge_matrix_perm = _edge_matrix_from_pi(pi_perm, n)
    for i in range(batch.size(0)):
        tails, heads = edges_from_pi(pi[i:i+1])
        ptails, pheads = edges_from_pi(pi_perm[i:i+1])
        mapped_t = perms[i][ptails[0]]
        mapped_h = perms[i][pheads[0]]
        s1 = build_successor(tails, heads)[0]
        s2 = build_successor(mapped_t.unsqueeze(0), mapped_h.unsqueeze(0))[0]
        exact.append((s1 == s2).all().float())

        edges_a = edge_matrix[i]
        edges_b = _permute_edge_matrix_back(edge_matrix_perm[i], perms[i])
        intersection = (edges_a & edges_b).sum().float()
        union = (edges_a | edges_b).sum().float().clamp_min(1.0)
        jaccards.append(intersection / union)
        recalls.append(intersection / float(n))
        rel_length_diffs.append((cost[i] - cost_perm[i]).abs() / cost[i].abs().clamp_min(1e-8))

        if getattr(model, "action_mode", None) == "joint_edge":
            from ptrnet_gt.states import ComponentMergeState
            base_state = ComponentMergeState.initialize(batch[i:i+1])
            perm_state = ComponentMergeState.initialize(permuted[i:i+1])
            log_p = model.get_joint_edge_log_p(base_state, embeddings[i:i+1])[0]
            perm_log_p = model.get_joint_edge_log_p(perm_state, perm_embeddings[i:i+1])[0]
            perm_back = _permute_flat_edge_scores_back(perm_log_p, perms[i], n)
            valid = torch.isfinite(log_p) & torch.isfinite(perm_back)
            if valid.any():
                prob_equiv_errors.append((log_p.exp()[valid] - perm_back.exp()[valid]).abs().mean())

    metrics = {
        "permutation_exact_successor_consistency": float(torch.stack(exact).mean().item()),
        "permutation_edge_jaccard": float(torch.stack(jaccards).mean().item()),
        "permutation_edge_recall": float(torch.stack(recalls).mean().item()),
        "permutation_relative_length_diff": float(torch.stack(rel_length_diffs).mean().item()),
        "permutation_action_prob_equiv_error": float(torch.stack(prob_equiv_errors).mean().item()) if prob_equiv_errors else None,
    }
    metrics["permutation_consistency"] = metrics["permutation_exact_successor_consistency"]
    return metrics


def _maybe_compute_permutation_metrics(model, dataloader, device: torch.device, enabled: bool) -> dict:
    if not enabled:
        return {
            "permutation_exact_successor_consistency": None,
            "permutation_edge_jaccard": None,
            "permutation_edge_recall": None,
            "permutation_relative_length_diff": None,
            "permutation_action_prob_equiv_error": None,
            "permutation_consistency": None,
        }
    probe_batch = next(iter(dataloader)).to(device)
    return _permutation_metrics(model, probe_batch)


def main():
    parser = argparse.ArgumentParser(description="Ptr-net-GT unified evaluation entry")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint saved by scripts/train.py")
    parser.add_argument("--override", action="append", default=[], help="Override config entries")
    parser.add_argument("--dataset", default=None, help="Fixed dataset .pt file")
    parser.add_argument("--reference", default=None, help="Optional reference cost tensor/.pt for gap computation")
    parser.add_argument("--decode", choices=["greedy", "beam"], default="greedy", help="Decode strategy")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam width when --decode beam")
    parser.add_argument("--beam-dedup", action="store_true", help="Deduplicate equivalent selected-edge states during beam expansion")
    parser.add_argument("--skip-permutation-probe", action="store_true", help="Skip permutation consistency probe on the first evaluation batch")
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
    duplicate_edge_state_rates = []
    duplicate_component_state_rates = []
    unique_edge_states = []
    unique_component_states = []
    dedup_retention_rates = []
    mean_expanded_candidates = []
    mean_kept_candidates = []
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    import time
    wall_start = time.perf_counter()
    if start is not None:
        start.record()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if args.decode == "beam":
                if model_name != "component_merge":
                    raise ValueError("Beam decode is currently only supported for component_merge")
                beam_result = component_merge_beam_search_batched(
                    model,
                    batch,
                    beam_size=args.beam_size,
                    dedup=args.beam_dedup,
                )
                cost = beam_result["cost"]
                pi = beam_result["pi"]
                duplicate_edge_state_rates.extend(beam_result["duplicate_edge_state_rates"])
                duplicate_component_state_rates.extend(beam_result["duplicate_component_state_rates"])
                dedup_retention_rates.extend(beam_result["dedup_retention_rates"])
                mean_expanded_candidates.extend(beam_result["mean_expanded_candidates_per_instance"])
                mean_kept_candidates.extend(beam_result["mean_kept_candidates_per_instance"])
                unique_edge_states.extend(beam_result["unique_edge_states_per_step"])
                unique_component_states.extend(beam_result["unique_component_states_per_step"])
            else:
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
        "decode_strategy": args.decode,
        "beam_size": args.beam_size if args.decode == "beam" else None,
        "beam_dedup": bool(args.beam_dedup) if args.decode == "beam" else None,
        "duplicate_edge_state_rate": float(sum(duplicate_edge_state_rates) / max(len(duplicate_edge_state_rates), 1)) if duplicate_edge_state_rates else None,
        "duplicate_component_state_rate": float(sum(duplicate_component_state_rates) / max(len(duplicate_component_state_rates), 1)) if duplicate_component_state_rates else None,
        "beam_dedup_retention_rate": float(sum(dedup_retention_rates) / max(len(dedup_retention_rates), 1)) if dedup_retention_rates else None,
        "mean_expanded_candidates_per_step": float(sum(mean_expanded_candidates) / max(len(mean_expanded_candidates), 1)) if mean_expanded_candidates else None,
        "mean_kept_candidates_per_step": float(sum(mean_kept_candidates) / max(len(mean_kept_candidates), 1)) if mean_kept_candidates else None,
        "mean_unique_edge_states_per_step": float(sum(unique_edge_states) / max(len(unique_edge_states), 1)) if unique_edge_states else None,
        "mean_unique_component_states_per_step": float(sum(unique_component_states) / max(len(unique_component_states), 1)) if unique_component_states else None,
        "permutation_consistency": None,
    }
    result.update(
        _maybe_compute_permutation_metrics(
            model,
            dataloader,
            device,
            enabled=not args.skip_permutation_probe,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_root = Path(config.get("output", {}).get("root", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "last_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()