from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ptrnet_gt.problems import TSP
from ptrnet_gt.utils.data import load_dataset_payload


def _solve_instance_with_concorde(points: torch.Tensor, verbose: bool = False):
    try:
        from concorde.tsp import TSPSolver
    except ImportError as exc:
        raise ImportError(
            "PyConcorde is not installed. Please install pyconcorde before running this script."
        ) from exc

    xs = points[:, 0].detach().cpu().numpy()
    ys = points[:, 1].detach().cpu().numpy()
    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    solution = solver.solve(verbose=verbose)
    if getattr(solution, "tour", None) is None:
        raise RuntimeError("Concorde failed to produce a tour")
    return torch.as_tensor(solution.tour, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Solve a fixed TSP dataset with Concorde")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-large", action="store_true", help="Allow running Concorde on size >= 50")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset_payload(args.dataset)
    coords = dataset["coords"]
    size = int(coords.size(1))
    if size >= 50 and not args.allow_large:
        raise ValueError(
            f"Concorde baseline is restricted to TSP50 below by default, got size={size}. "
            "Use LKH-3 instead, or pass --allow-large explicitly if you really need this run."
        )

    tours = []
    start = time.perf_counter()
    for instance in coords:
        tours.append(_solve_instance_with_concorde(instance, verbose=args.verbose))
    elapsed = time.perf_counter() - start

    tours_tensor = torch.stack(tours, dim=0)
    costs, _ = TSP.get_costs(coords, tours_tensor)
    payload = {
        "method": "concorde",
        "solver": "pyconcorde",
        "dataset": args.dataset,
        "size": size,
        "num_instances": int(coords.size(0)),
        "costs": costs.detach().cpu(),
        "tours": tours_tensor.detach().cpu(),
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(coords.size(0), 1)),
        "notes": (
            "Tours solved by Concorde/EUC_2D; costs recomputed with repository floating Euclidean metric."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(payload)


if __name__ == "__main__":
    main()