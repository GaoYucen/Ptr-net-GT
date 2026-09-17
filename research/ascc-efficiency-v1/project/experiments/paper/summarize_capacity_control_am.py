"""汇总 AM TSP50 的 Original、容量匹配单链与 Full GroupOpt。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


MODES = (
    "native_original",
    "native_capacity_single_chain",
    "native_conditional_free",
)
COMPARISONS = {
    "original_vs_capacity": ("native_original", "native_capacity_single_chain"),
    "capacity_vs_full": ("native_capacity_single_chain", "native_conditional_free"),
    "original_vs_full": ("native_original", "native_conditional_free"),
}


def _load(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64)
    if costs.numel() == 0 or not torch.isfinite(costs).all():
        raise ValueError(f"invalid costs: {path}")
    return costs, dict(payload["metadata"])


def _statistics(baseline: torch.Tensor, contender: torch.Tensor) -> dict[str, float]:
    if baseline.shape != contender.shape:
        raise ValueError("paired costs must have identical shapes")
    difference = baseline - contender
    mean_difference = difference.mean()
    half_width = 1.96 * difference.std(unbiased=True) / math.sqrt(difference.numel())
    baseline_mean = baseline.mean().item()
    contender_mean = contender.mean().item()
    return {
        "baseline_mean": baseline_mean,
        "contender_mean": contender_mean,
        "baseline_minus_contender": mean_difference.item(),
        "contender_relative_improvement_percent": 100.0
        * (baseline_mean - contender_mean)
        / baseline_mean,
        "paired_ci95_low": (mean_difference - half_width).item(),
        "paired_ci95_high": (mean_difference + half_width).item(),
        "contender_win_rate": (contender < baseline).to(torch.float64).mean().item(),
        "test_size": int(difference.numel()),
    }


def summarize(
    capacity_root: Path,
    reference_root: Path,
    output_dir: Path,
    test_seed: int,
) -> None:
    seeds = (1234, 4321, 2468)
    costs: dict[tuple[str, int], torch.Tensor] = {}
    metadata: dict[tuple[str, int], dict[str, object]] = {}

    for seed in seeds:
        paths = {
            "native_original": reference_root
            / "eval"
            / f"iid_tsp50_seed{test_seed}"
            / "am"
            / "native_original"
            / f"seed{seed}"
            / "costs.pt",
            "native_conditional_free": reference_root
            / "eval"
            / f"iid_tsp50_seed{test_seed}"
            / "am"
            / "native_conditional_free"
            / f"seed{seed}"
            / "costs.pt",
            "native_capacity_single_chain": capacity_root
            / "eval"
            / f"iid_tsp50_seed{test_seed}"
            / "am"
            / "native_capacity_single_chain"
            / f"seed{seed}"
            / "costs.pt",
        }
        for mode, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(path)
            costs[(mode, seed)], metadata[(mode, seed)] = _load(path)

    identity = ("problem", "graph_size", "test_seed", "test_size")
    reference_metadata = metadata[("native_original", seeds[0])]
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
    comparison_summary: dict[str, object] = {}
    for name, (baseline_mode, contender_mode) in COMPARISONS.items():
        seed_results: dict[str, object] = {}
        for seed in seeds:
            stats = _statistics(
                costs[(baseline_mode, seed)], costs[(contender_mode, seed)]
            )
            seed_results[str(seed)] = stats
            comparison_rows.append(
                {
                    "comparison": name,
                    "baseline_mode": baseline_mode,
                    "contender_mode": contender_mode,
                    "train_seed": seed,
                    **stats,
                }
            )
        pooled_baseline = torch.cat([costs[(baseline_mode, seed)] for seed in seeds])
        pooled_contender = torch.cat([costs[(contender_mode, seed)] for seed in seeds])
        pooled = _statistics(pooled_baseline, pooled_contender)
        comparison_summary[name] = {
            "baseline_mode": baseline_mode,
            "contender_mode": contender_mode,
            "all_seeds_favor_contender": all(
                float(seed_results[str(seed)]["baseline_minus_contender"]) > 0
                for seed in seeds
            ),
            "per_seed": seed_results,
            "pooled_instance_seed_pairs": pooled,
        }

    validation_path = capacity_root / "summary" / "parameter_validation.json"
    validation = json.loads(validation_path.read_text())
    summary = {
        "experiment": "paper_capacity_control_am_tsp50_v1",
        "problem": "uniform_euclidean_tsp50",
        "test_seed": test_seed,
        "training_seeds": list(seeds),
        "mode_mean_across_seeds": {
            mode: sum(costs[(mode, seed)].mean().item() for seed in seeds) / len(seeds)
            for mode in MODES
        },
        "comparisons": comparison_summary,
        "parameter_validation": {
            "status": validation["status"],
            "additional_active_parameters": validation["additional_active_parameters"],
            "original_active_parameters": validation["modes"]["native_original"][
                "active_gradient_parameter_count"
            ],
            "capacity_active_parameters": validation["modes"][
                "native_capacity_single_chain"
            ]["active_gradient_parameter_count"],
            "full_active_parameters": validation["modes"]["native_conditional_free"][
                "active_gradient_parameter_count"
            ],
        },
        "sign_convention": "positive baseline-minus-contender means contender is better",
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
    (output_dir / "capacity_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-seed", type=int, default=20260904)
    arguments = parser.parse_args()
    summarize(
        arguments.capacity_root.resolve(),
        arguments.reference_root.resolve(),
        arguments.output_dir.resolve(),
        arguments.test_seed,
    )
