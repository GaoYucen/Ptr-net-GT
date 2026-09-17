"""汇总 AM TSP50 正式信息消融的三种子同实例配对结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


FULL = "native_conditional_free"
ABLATIONS = (
    "native_random_tail",
    "native_free_no_head_summary",
    "native_free_no_path_state",
)
MODES = (FULL, *ABLATIONS)


def _load(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64)
    if costs.numel() == 0 or not torch.isfinite(costs).all():
        raise ValueError(f"invalid costs: {path}")
    return costs, dict(payload["metadata"])


def _statistics(ablation: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    if ablation.shape != full.shape:
        raise ValueError("paired costs must have identical shapes")
    difference = ablation - full
    mean_difference = difference.mean()
    half_width = 1.96 * difference.std(unbiased=True) / math.sqrt(difference.numel())
    ablation_mean = ablation.mean().item()
    full_mean = full.mean().item()
    return {
        "ablation_mean": ablation_mean,
        "full_mean": full_mean,
        "ablation_minus_full": mean_difference.item(),
        "full_relative_improvement_percent": 100.0
        * (ablation_mean - full_mean)
        / ablation_mean,
        "paired_ci95_low": (mean_difference - half_width).item(),
        "paired_ci95_high": (mean_difference + half_width).item(),
        "full_win_rate": (full < ablation).to(torch.float64).mean().item(),
        "test_size": int(difference.numel()),
    }


def summarize(
    ablation_root: Path,
    reference_root: Path,
    output_dir: Path,
    test_seed: int,
) -> None:
    seeds = (1234, 4321, 2468)
    costs: dict[tuple[str, int], torch.Tensor] = {}
    metadata: dict[tuple[str, int], dict[str, object]] = {}
    for seed in seeds:
        full_path = (
            reference_root
            / "eval"
            / f"iid_tsp50_seed{test_seed}"
            / "am"
            / FULL
            / f"seed{seed}"
            / "costs.pt"
        )
        costs[(FULL, seed)], metadata[(FULL, seed)] = _load(full_path)
        for mode in ABLATIONS:
            path = (
                ablation_root
                / "eval"
                / f"iid_tsp50_seed{test_seed}"
                / "am"
                / mode
                / f"seed{seed}"
                / "costs.pt"
            )
            costs[(mode, seed)], metadata[(mode, seed)] = _load(path)

    identity = ("problem", "graph_size", "test_seed", "test_size")
    reference_metadata = metadata[(FULL, seeds[0])]
    for item in metadata.values():
        if any(item.get(key) != reference_metadata.get(key) for key in identity):
            raise ValueError("evaluation metadata mismatch")

    mode_rows: list[dict[str, object]] = []
    for mode in MODES:
        for seed in seeds:
            mode_rows.append(
                {
                    "mode": mode,
                    "train_seed": seed,
                    "mean_cost": costs[(mode, seed)].mean().item(),
                }
            )

    comparison_rows: list[dict[str, object]] = []
    comparisons: dict[str, object] = {}
    for mode in ABLATIONS:
        per_seed: dict[str, object] = {}
        for seed in seeds:
            stats = _statistics(costs[(mode, seed)], costs[(FULL, seed)])
            per_seed[str(seed)] = stats
            comparison_rows.append(
                {"ablation_mode": mode, "train_seed": seed, **stats}
            )
        pooled_ablation = torch.cat([costs[(mode, seed)] for seed in seeds])
        pooled_full = torch.cat([costs[(FULL, seed)] for seed in seeds])
        comparisons[f"{mode}_vs_full"] = {
            "ablation_mode": mode,
            "full_mode": FULL,
            "all_seeds_favor_full": all(
                float(per_seed[str(seed)]["ablation_minus_full"]) > 0
                for seed in seeds
            ),
            "per_seed": per_seed,
            "pooled_instance_seed_pairs": _statistics(pooled_ablation, pooled_full),
        }

    validation = json.loads(
        (ablation_root / "summary" / "implementation_validation.json").read_text()
    )
    summary = {
        "experiment": "paper_ablation_am_tsp50_v1",
        "problem": "uniform_euclidean_tsp50",
        "test_seed": test_seed,
        "training_seeds": list(seeds),
        "mode_mean_across_seeds": {
            mode: sum(costs[(mode, seed)].mean().item() for seed in seeds) / len(seeds)
            for mode in MODES
        },
        "comparisons": comparisons,
        "implementation_validation_status": validation["status"],
        "sign_convention": "positive ablation-minus-full means Full GroupOpt is better",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mode_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mode_rows[0]))
        writer.writeheader()
        writer.writerows(mode_rows)
    with (output_dir / "paired_comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-seed", type=int, default=20260904)
    arguments = parser.parse_args()
    summarize(
        arguments.ablation_root.resolve(),
        arguments.reference_root.resolve(),
        arguments.output_dir.resolve(),
        arguments.test_seed,
    )
