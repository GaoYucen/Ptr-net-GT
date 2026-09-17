"""汇总 AM/POMO 在 TSP20/50/100/200 上分别训练后的配对结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch


def _load(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64).flatten()
    if costs.numel() == 0 or not torch.isfinite(costs).all():
        raise ValueError(f"非法的逐实例 cost：{path}")
    return costs, dict(payload["metadata"])


def _paired(original: torch.Tensor, groupopt: torch.Tensor) -> dict[str, float | int]:
    if original.shape != groupopt.shape:
        raise ValueError("Original 与 GroupOpt 的 cost 形状不一致")
    difference = original - groupopt
    half_width = 1.96 * difference.std(unbiased=True) / math.sqrt(difference.numel())
    original_mean = original.mean()
    groupopt_mean = groupopt.mean()
    mean_difference = difference.mean()
    return {
        "original_mean": float(original_mean),
        "groupopt_mean": float(groupopt_mean),
        "original_minus_groupopt": float(mean_difference),
        "relative_improvement_percent": float(100.0 * mean_difference / original_mean),
        "paired_ci95_low": float(mean_difference - half_width),
        "paired_ci95_high": float(mean_difference + half_width),
        "groupopt_win_rate": float((groupopt < original).to(torch.float64).mean()),
        "test_size": int(difference.numel()),
    }


def _cost_path(
    scale_root: Path,
    tsp50_root: Path,
    size: int,
    test_seed: int,
    family: str,
    mode: str,
    train_seed: int,
) -> Path:
    root = tsp50_root if size == 50 else scale_root
    return (
        root
        / "eval"
        / f"iid_tsp{size}_seed{test_seed}"
        / family
        / mode
        / f"seed{train_seed}"
        / "costs.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-root", type=Path, required=True)
    parser.add_argument("--tsp50-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--families", nargs="+", default=["am", "pomo"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234, 4321, 2468])
    parser.add_argument("--test-seed", type=int, default=20260904)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    result: dict[str, object] = {
        "experiment": "paper_scale_effect_tsp_v1",
        "interpretation": "separately trained at every size; not zero-shot transfer",
        "test_seed": args.test_seed,
        "results": {},
    }
    results = result["results"]
    assert isinstance(results, dict)
    identity_keys = ("test_seed", "test_size", "graph_size", "distribution")

    for size in args.sizes:
        size_result: dict[str, object] = {}
        for family in args.families:
            per_seed: dict[str, object] = {}
            original_means: list[float] = []
            groupopt_means: list[float] = []
            differences: list[float] = []
            all_original: list[torch.Tensor] = []
            all_groupopt: list[torch.Tensor] = []
            for train_seed in args.seeds:
                original, original_meta = _load(
                    _cost_path(
                        args.scale_root,
                        args.tsp50_root,
                        size,
                        args.test_seed,
                        family,
                        "native_original",
                        train_seed,
                    )
                )
                groupopt, groupopt_meta = _load(
                    _cost_path(
                        args.scale_root,
                        args.tsp50_root,
                        size,
                        args.test_seed,
                        family,
                        "native_conditional_free",
                        train_seed,
                    )
                )
                if any(
                    original_meta.get(key) != groupopt_meta.get(key)
                    for key in identity_keys
                ):
                    raise ValueError(
                        f"TSP{size} {family} seed={train_seed} 不是同一测试集"
                    )
                statistics = _paired(original, groupopt)
                per_seed[str(train_seed)] = statistics
                original_means.append(float(statistics["original_mean"]))
                groupopt_means.append(float(statistics["groupopt_mean"]))
                differences.append(float(statistics["original_minus_groupopt"]))
                all_original.append(original)
                all_groupopt.append(groupopt)
                rows.append(
                    {
                        "graph_size": size,
                        "family": family,
                        "train_seed": train_seed,
                        **statistics,
                    }
                )

            original_macro = sum(original_means) / len(original_means)
            groupopt_macro = sum(groupopt_means) / len(groupopt_means)
            size_result[family] = {
                "seeds": per_seed,
                "macro": {
                    "seed_count": len(args.seeds),
                    "original_mean": original_macro,
                    "groupopt_mean": groupopt_macro,
                    "original_minus_groupopt": original_macro - groupopt_macro,
                    "relative_improvement_percent": 100.0
                    * (original_macro - groupopt_macro)
                    / original_macro,
                    "all_seeds_improve": all(value > 0.0 for value in differences),
                    "seed_difference_standard_deviation": float(
                        torch.tensor(differences, dtype=torch.float64).std(unbiased=True)
                    ),
                },
                "pooled_instance_seed_pairs": _paired(
                    torch.cat(all_original), torch.cat(all_groupopt)
                ),
            }
        results[str(size)] = size_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scale_effect_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "scale_effect_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
