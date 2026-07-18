from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/opt/conda/envs/py11/bin/python"


def _run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def _extract_last_json(stdout: str) -> dict:
    lines = stdout.strip().splitlines()
    for start in range(len(lines)):
        candidate = "\n".join(lines[start:])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON object from command stdout")


def _objective_run_name(prefix: str, objective: str, samples: int, seed: int) -> str:
    return f"{prefix}_{objective}_s{samples}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick single-seed fixed_order vs orbit_sum learning-curve runner")
    parser.add_argument("--config", default="configs/component_merge/tsp20.yaml", help="Base config path")
    parser.add_argument("--dataset", default=None, help="Optional evaluation dataset override")
    parser.add_argument("--samples", nargs="+", type=int, default=[2048, 4096, 8192, 16384], help="Training sample budgets")
    parser.add_argument("--seed", type=int, default=1234, help="Single seed for quick validation")
    parser.add_argument("--batch-size", type=int, default=64, help="Training and evaluation batch size")
    parser.add_argument("--val-size", type=int, default=128, help="Validation subset size during training")
    parser.add_argument("--objectives", nargs="+", default=["fixed_order", "orbit_sum"], choices=["fixed_order", "orbit_sum", "equiv_set"], help="Objectives to compare")
    parser.add_argument("--run-prefix", default="component_merge_tsp20_quick_curve", help="Output run name prefix")
    parser.add_argument("--output", default="outputs/orbit_sum_quick_curve/results.json", help="Where to save aggregated results")
    parser.add_argument("--skip-permutation-probe", action="store_true", help="Skip permutation metrics during evaluation to speed up validation")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for samples in args.samples:
        for objective in args.objectives:
            run_name = _objective_run_name(args.run_prefix, objective, samples, args.seed)
            train_cmd = [
                PYTHON_BIN,
                "scripts/train.py",
                "--config",
                args.config,
                "--override",
                f"experiment.name={run_name}",
                "--override",
                f"experiment.seed={args.seed}",
                "--override",
                f"training.objective={objective}",
                "--override",
                "training.baseline=none",
                "--override",
                "training.epochs=1",
                "--override",
                f"training.batch_size={args.batch_size}",
                "--override",
                f"training.epoch_size={samples}",
                "--override",
                f"training.val_size={args.val_size}",
                "--override",
                f"evaluation.batch_size={args.batch_size}",
            ]
            print(f"[train] objective={objective} samples={samples} seed={args.seed}")
            train_stdout = _run_command(train_cmd)

            checkpoint_path = PROJECT_ROOT / "outputs" / run_name / "best_model.pt"
            if not checkpoint_path.exists():
                checkpoint_path = PROJECT_ROOT / "outputs" / run_name / "model.pt"

            eval_cmd = [
                PYTHON_BIN,
                "scripts/evaluate.py",
                "--config",
                args.config,
                "--checkpoint",
                str(checkpoint_path),
                "--override",
                f"evaluation.batch_size={args.batch_size}",
            ]
            if args.dataset:
                eval_cmd.extend(["--dataset", args.dataset])
            if args.skip_permutation_probe:
                eval_cmd.append("--skip-permutation-probe")

            print(f"[eval] objective={objective} samples={samples} seed={args.seed}")
            eval_stdout = _run_command(eval_cmd)
            eval_result = _extract_last_json(eval_stdout)
            eval_result.update(
                {
                    "objective": objective,
                    "train_samples_budget": samples,
                    "seed": args.seed,
                    "run_name": run_name,
                    "checkpoint_used": str(checkpoint_path),
                    "train_stdout": train_stdout,
                }
            )
            all_results.append(eval_result)

    summary = {
        "config": args.config,
        "dataset": args.dataset,
        "samples": args.samples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "val_size": args.val_size,
        "objectives": args.objectives,
        "results": all_results,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()