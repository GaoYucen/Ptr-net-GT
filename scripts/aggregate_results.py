from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mean = _mean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser(description="Aggregate baseline result JSON files")
    parser.add_argument("inputs", nargs="+", help="Result json files")
    parser.add_argument("--output-dir", default="outputs/baseline_comparison")
    args = parser.parse_args()

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for path_str in args.inputs:
        path = Path(path_str)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("results", payload if isinstance(payload, list) else [payload])
        if isinstance(records, dict):
            records = [records]
        for record in records:
            grouped[(record["method"], int(record.get("size", -1)))].append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"

    fieldnames = [
        "method", "size", "num_runs", "cost_mean", "cost_std_across_runs", "gap_mean_percent",
        "feasible_rate_mean", "time_per_instance_mean_sec", "throughput_mean", "parameter_count",
        "training_updates", "training_samples",
    ]

    rows = []
    for (method, size), records in sorted(grouped.items()):
        rows.append({
            "method": method,
            "size": size,
            "num_runs": len(records),
            "cost_mean": _mean([float(r.get("mean_tour_length", r.get("avg_cost"))) for r in records]),
            "cost_std_across_runs": _std([float(r.get("mean_tour_length", r.get("avg_cost"))) for r in records]),
            "gap_mean_percent": _mean([float(r["optimality_gap_percent"]) for r in records if r.get("optimality_gap_percent") is not None]),
            "feasible_rate_mean": _mean([float(r.get("feasible_tour_rate", 0.0)) for r in records]),
            "time_per_instance_mean_sec": _mean([float(r.get("time_per_instance_sec", 0.0)) for r in records]),
            "throughput_mean": _mean([float(r.get("throughput_instances_per_sec", 0.0)) for r in records]),
            "parameter_count": records[0].get("parameter_count"),
            "training_updates": records[0].get("training_updates"),
            "training_samples": records[0].get("training_samples"),
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Method | Size | Runs | Cost Mean | Cost Std | Gap % | Feasible % | Time/Inst (s) | Throughput | Params | Updates | Samples |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['size']} | {row['num_runs']} | {row['cost_mean']} | {row['cost_std_across_runs']} | {row['gap_mean_percent']} | {row['feasible_rate_mean']} | {row['time_per_instance_mean_sec']} | {row['throughput_mean']} | {row['parameter_count']} | {row['training_updates']} | {row['training_samples']} |\n"
            )


if __name__ == "__main__":
    main()