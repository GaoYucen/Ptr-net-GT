from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(r"Validation overall avg_cost: ([0-9.]+)")
EPOCH_PATTERN = re.compile(r"Start train epoch (\d+)")


def parse_log(path: Path) -> dict[str, list[float] | list[int]]:
    text = path.read_text(encoding="utf-8")
    epochs = [int(m.group(1)) for m in EPOCH_PATTERN.finditer(text)]
    avg_costs = [float(m.group(1)) for m in LOG_PATTERN.finditer(text)]
    avg_costs = avg_costs[: len(epochs)]
    return {"epochs": epochs, "avg_costs": avg_costs}


def build_summary(series: dict[str, dict[str, list[float] | list[int]]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    checkpoints = [1, 3, 5, 8, 10]
    for name, data in series.items():
        avg_costs = data["avg_costs"]
        best_cost = min(avg_costs)
        best_epoch = avg_costs.index(best_cost)
        checkpoint_stats: dict[str, dict[str, float]] = {}
        for k in checkpoints:
            if k <= len(avg_costs):
                current = avg_costs[k - 1]
                checkpoint_stats[str(k)] = {
                    "avg_cost": current,
                    "gap_to_best_abs": current - best_cost,
                    "gap_to_best_pct": (current - best_cost) / best_cost * 100,
                }
        summary[name] = {
            "best_avg_cost": best_cost,
            "best_epoch": best_epoch,
            "checkpoints": checkpoint_stats,
        }
    return summary


def plot_series(series: dict[str, dict[str, list[float] | list[int]]], output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, data in series.items():
        plt.plot(data["epochs"], data["avg_costs"], marker="o", linewidth=2, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation avg_cost")
    plt.title("Training Convergence on TSP20")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def main() -> None:
    log_dir = Path("outputs/logs")
    figure_dir = Path("figure")
    models = {
        "component_merge": log_dir / "component_merge_tsp20_train.log",
        "attention_model": log_dir / "attention_model_tsp20_train.log",
        "pointer_network": log_dir / "pointer_network_tsp20_train.log",
    }
    series = {name: parse_log(path) for name, path in models.items()}
    summary = build_summary(series)

    plot_path = figure_dir / "training_convergence_tsp20.png"
    plot_series(series, plot_path)

    summary_path = figure_dir / "training_convergence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "plot": str(plot_path),
        "summary": str(summary_path),
        "component_merge": summary["component_merge"],
    }, indent=2))


if __name__ == "__main__":
    main()