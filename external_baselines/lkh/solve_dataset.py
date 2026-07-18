from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ptrnet_gt.problems import TSP
from ptrnet_gt.utils.data import load_dataset_payload


def _write_tsplib(instance_path: Path, coords: torch.Tensor, scale: int = 100000) -> None:
    n = coords.size(0)
    with open(instance_path, "w", encoding="utf-8") as f:
        f.write(f"NAME : {instance_path.stem}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, point in enumerate(coords.tolist(), start=1):
            x = int(round(point[0] * scale))
            y = int(round(point[1] * scale))
            f.write(f"{idx} {x} {y}\n")
        f.write("EOF\n")


def _write_parameter_file(
    parameter_path: Path,
    instance_path: Path,
    output_tour_path: Path,
    runs: int,
    max_trials: int,
    seed: int,
) -> None:
    with open(parameter_path, "w", encoding="utf-8") as f:
        f.write(f"PROBLEM_FILE = {instance_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {output_tour_path}\n")
        f.write(f"RUNS = {runs}\n")
        f.write(f"MAX_TRIALS = {max_trials}\n")
        f.write(f"SEED = {seed}\n")


def _parse_lkh_tour(tour_path: Path, expected_size: int) -> torch.Tensor:
    if not tour_path.exists():
        raise FileNotFoundError(f"LKH did not produce tour file: {tour_path}")
    in_section = False
    nodes: list[int] = []
    with open(tour_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == "TOUR_SECTION":
                in_section = True
                continue
            if not in_section:
                continue
            if line in {"-1", "EOF"}:
                break
            if line:
                nodes.append(int(line) - 1)
    if len(nodes) != expected_size:
        raise ValueError(f"Expected {expected_size} nodes from LKH tour, got {len(nodes)}")
    return torch.as_tensor(nodes, dtype=torch.long)


def _solve_instance(
    coords: torch.Tensor,
    lkh_executable: str,
    runs: int,
    max_trials: int,
    seed: int,
) -> torch.Tensor:
    with tempfile.TemporaryDirectory(prefix="lkh_tsp_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        instance_path = tmpdir_path / "instance.tsp"
        parameter_path = tmpdir_path / "instance.par"
        tour_path = tmpdir_path / "instance.tour"
        _write_tsplib(instance_path, coords)
        _write_parameter_file(parameter_path, instance_path, tour_path, runs=runs, max_trials=max_trials, seed=seed)
        result = subprocess.run(
            [lkh_executable, str(parameter_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LKH failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return _parse_lkh_tour(tour_path, expected_size=int(coords.size(0)))


def main():
    parser = argparse.ArgumentParser(description="Solve a fixed TSP dataset with LKH-3")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--lkh-executable", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    dataset = load_dataset_payload(args.dataset)
    coords = dataset["coords"]
    tours = []
    start = time.perf_counter()
    for idx, instance in enumerate(coords):
        tours.append(
            _solve_instance(
                instance,
                lkh_executable=args.lkh_executable,
                runs=args.runs,
                max_trials=args.max_trials,
                seed=args.seed + idx,
            )
        )
    elapsed = time.perf_counter() - start

    tours_tensor = torch.stack(tours, dim=0)
    costs, _ = TSP.get_costs(coords, tours_tensor)
    payload = {
        "method": "lkh",
        "solver": "LKH-3",
        "dataset": args.dataset,
        "size": int(coords.size(1)),
        "num_instances": int(coords.size(0)),
        "costs": costs.detach().cpu(),
        "tours": tours_tensor.detach().cpu(),
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(coords.size(0), 1)),
        "lkh_executable": args.lkh_executable,
        "runs": int(args.runs),
        "max_trials": int(args.max_trials),
        "notes": "Tours solved by LKH on TSPLIB EUC_2D instance; costs recomputed with repository floating Euclidean metric.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(payload)


if __name__ == "__main__":
    main()