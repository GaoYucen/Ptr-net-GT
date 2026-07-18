from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Unified entrypoint for Concorde/LKH reference solving")
    parser.add_argument("--solver", required=True, choices=["concorde", "lkh"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-large", action="store_true", help="Only relevant for Concorde")
    parser.add_argument("--verbose", action="store_true", help="Only relevant for Concorde")
    parser.add_argument("--lkh-executable", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    python_executable = sys.executable
    if args.solver == "concorde":
        command = [
            python_executable,
            str(PROJECT_ROOT / "external_baselines/concorde/solve_dataset.py"),
            "--dataset",
            args.dataset,
            "--output",
            args.output,
        ]
        if args.allow_large:
            command.append("--allow-large")
        if args.verbose:
            command.append("--verbose")
    else:
        if not args.lkh_executable:
            raise ValueError("--lkh-executable is required when --solver lkh")
        command = [
            python_executable,
            str(PROJECT_ROOT / "external_baselines/lkh/solve_dataset.py"),
            "--dataset",
            args.dataset,
            "--lkh-executable",
            args.lkh_executable,
            "--output",
            args.output,
            "--runs",
            str(args.runs),
            "--max-trials",
            str(args.max_trials),
            "--seed",
            str(args.seed),
        ]
    _run(command)


if __name__ == "__main__":
    main()