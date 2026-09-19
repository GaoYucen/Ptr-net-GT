"""汇总论文主表中各宿主 Original 与 Full GroupOpt 的三种子结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def load_costs(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64).flatten()
    metadata = payload["metadata"]
    if costs.numel() == 0 or not torch.isfinite(costs).all():
        raise ValueError(f"非法逐实例 cost：{path}")
    return costs, metadata


def paired(original: torch.Tensor, ours: torch.Tensor) -> dict[str, float | int]:
    if original.shape != ours.shape:
        raise ValueError("Original 与 Full 的逐实例 cost 形状不一致")
    difference = original - ours
    mean_difference = difference.mean()
    standard_error = difference.std(unbiased=True) / math.sqrt(difference.numel())
    half_width = 1.96 * standard_error
    original_mean = original.mean()
    ours_mean = ours.mean()
    return {
        "original_mean": float(original_mean),
        "groupopt_mean": float(ours_mean),
        "original_minus_groupopt": float(mean_difference),
        "relative_improvement_percent": float(100.0 * mean_difference / original_mean),
        "paired_ci95_low": float(mean_difference - half_width),
        "paired_ci95_high": float(mean_difference + half_width),
        "groupopt_win_rate": float((ours < original).to(torch.float64).mean()),
        "test_size": int(difference.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--experiment-id", default="paper_main_tsp50_v1")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    result: dict[str, object] = {
        "experiment": args.experiment_id,
        "test_seed": args.test_seed,
        "lower_cost_is_better": True,
        "families": {},
    }
    families_result = result["families"]
    assert isinstance(families_result, dict)

    for family in args.families:
        seed_result: dict[str, object] = {}
        original_means: list[float] = []
        groupopt_means: list[float] = []
        improvements: list[float] = []
        all_original: list[torch.Tensor] = []
        all_groupopt: list[torch.Tensor] = []

        for seed in args.seeds:
            prefix = args.root / f"iid_tsp50_seed{args.test_seed}" / family
            original, original_metadata = load_costs(
                prefix / "native_original" / f"seed{seed}" / "costs.pt"
            )
            groupopt, groupopt_metadata = load_costs(
                prefix / "native_conditional_free" / f"seed{seed}" / "costs.pt"
            )
            identity_keys = ("test_seed", "test_size", "graph_size", "distribution")
            if any(
                original_metadata.get(key) != groupopt_metadata.get(key) for key in identity_keys
            ):
                raise ValueError(f"{family} seed={seed} 不是同一测试集上的配对结果")

            statistics = paired(original, groupopt)
            seed_result[str(seed)] = statistics
            original_means.append(float(statistics["original_mean"]))
            groupopt_means.append(float(statistics["groupopt_mean"]))
            improvements.append(float(statistics["original_minus_groupopt"]))
            all_original.append(original)
            all_groupopt.append(groupopt)
            rows.append({"family": family, "seed": seed, **statistics})

        seed_tensor = torch.tensor(improvements, dtype=torch.float64)
        pooled = paired(torch.cat(all_original), torch.cat(all_groupopt))
        original_macro = sum(original_means) / len(original_means)
        groupopt_macro = sum(groupopt_means) / len(groupopt_means)
        families_result[family] = {
            "seeds": seed_result,
            "macro": {
                "seed_count": len(args.seeds),
                "original_mean": original_macro,
                "groupopt_mean": groupopt_macro,
                "original_minus_groupopt": original_macro - groupopt_macro,
                "relative_improvement_percent": 100.0
                * (original_macro - groupopt_macro)
                / original_macro,
                "all_seeds_improve": all(value > 0.0 for value in improvements),
                "seed_difference_standard_deviation": float(seed_tensor.std(unbiased=True)),
            },
            "pooled_instance_seed_pairs": pooled,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "main_tsp50_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "main_tsp50_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
