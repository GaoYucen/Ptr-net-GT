"""汇总长程训练中 Original 与 GroupOpt 在各 checkpoint 的配对结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

EVALUATION_STEPS = {
    "am": (2000, 5000, 10000),
    "ptrnet": (2000, 5000, 10000),
    "gpn": (2000, 5000, 10000),
    "pomo": (1000, 2500, 5000),
}


def load(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    costs = payload["costs"].to(torch.float64).flatten()
    if costs.numel() == 0 or not torch.isfinite(costs).all():
        raise ValueError(f"非法 costs：{path}")
    return costs, payload["metadata"]


def paired(original: torch.Tensor, groupopt: torch.Tensor) -> dict[str, float | int]:
    if original.shape != groupopt.shape:
        raise ValueError("配对 costs 形状不一致")
    difference = original - groupopt
    half_width = 1.96 * difference.std(unbiased=True) / math.sqrt(difference.numel())
    original_mean = original.mean()
    groupopt_mean = groupopt.mean()
    mean_difference = difference.mean()
    return {
        "original_mean": float(original_mean),
        "groupopt_mean": float(groupopt_mean),
        "original_minus_groupopt": float(mean_difference),
        "relative_improvement_percent": float(100 * mean_difference / original_mean),
        "paired_ci95_low": float(mean_difference - half_width),
        "paired_ci95_high": float(mean_difference + half_width),
        "groupopt_win_rate": float((groupopt < original).to(torch.float64).mean()),
        "test_size": int(difference.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    result: dict[str, object] = {
        "experiment": "paper_convergence_tsp50_v1",
        "test_seed": args.test_seed,
        "families": {},
    }
    families = result["families"]
    assert isinstance(families, dict)
    identity_keys = ("test_seed", "test_size", "graph_size", "distribution")

    for family, steps in EVALUATION_STEPS.items():
        family_result: dict[str, object] = {}
        for step in steps:
            prefix = args.root / family / f"step{step}"
            original, original_metadata = load(prefix / "native_original" / "costs.pt")
            groupopt, groupopt_metadata = load(
                prefix / "native_conditional_free" / "costs.pt"
            )
            if any(
                original_metadata.get(key) != groupopt_metadata.get(key)
                for key in identity_keys
            ):
                raise ValueError(f"{family} step={step} 不是同一测试集")
            statistics = paired(original, groupopt)
            family_result[str(step)] = statistics
            rows.append({"family": family, "step": step, **statistics})
        families[family] = family_result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "convergence_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
