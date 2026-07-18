from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "figure"


def _read_validation_csv(path: Path) -> dict[str, list[float]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    rows = [line.split(",") for line in lines[1:]]
    epochs = [int(row[0]) for row in rows]
    avg_costs = [float(row[1]) for row in rows]
    return {"epochs": epochs, "avg_costs": avg_costs}


def _read_cm_log(path: Path) -> dict[str, list[float]]:
    pattern = re.compile(r"Validation overall avg_cost: ([0-9.]+)")
    avg_costs = [float(m.group(1)) for m in pattern.finditer(path.read_text(encoding="utf-8"))][:3]
    return {"epochs": list(range(len(avg_costs))), "avg_costs": avg_costs}


def _read_symnco_smoke(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stdout = payload["train_stdout"]
    match = re.search(r"train_score_list = (\[[^\]]+\])", stdout)
    if not match:
        raise ValueError(f"Could not find train_score_list in {path}")
    avg_costs = [float(x) for x in ast.literal_eval(match.group(1))]
    return {"epochs": list(range(len(avg_costs))), "avg_costs": avg_costs}


def main() -> None:
    series = {
        "Original model": _read_validation_csv(PROJECT_ROOT / "outputs/component_merge_tsp20_ep3/validation_results.txt"),
        "Sym-NCO": _read_symnco_smoke(PROJECT_ROOT / "outputs/ep3_comparison/sym_nco_tsp20_ep3_smoke.json"),
        "Sym-NCO + CM + fixed-order": _read_cm_log(PROJECT_ROOT / "outputs/sym_nco_cm_fixed_tsp20/train_ep3.log"),
        "Sym-NCO + CM + orbit-sum": _read_cm_log(PROJECT_ROOT / "outputs/sym_nco_cm_orbit_tsp20/train_ep3.log"),
    }

    plt.figure(figsize=(8.5, 5.2))
    for name, data in series.items():
        plt.plot(data["epochs"], data["avg_costs"], marker="o", linewidth=2, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation / training tour length")
    plt.title("TSP20 3-epoch training curve comparison")
    plt.xticks([0, 1, 2])
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = FIGURE_DIR / "ep3_training_curves_tsp20_4models.png"
    plt.savefig(plot_path, dpi=220)

    summary = {
        name: {
            "epochs": data["epochs"],
            "avg_costs": data["avg_costs"],
            "best_cost": min(data["avg_costs"]),
            "best_epoch": data["avg_costs"].index(min(data["avg_costs"])),
        }
        for name, data in series.items()
    }
    summary_path = FIGURE_DIR / "ep3_training_curves_tsp20_4models.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"plot": str(plot_path), "summary": str(summary_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()