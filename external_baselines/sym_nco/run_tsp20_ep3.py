from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout + result.stderr


def _latest_result_dir(base_dir: Path, suffix: str) -> Path:
    candidates = sorted([p for p in base_dir.glob(f"*{suffix}") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No result directory found in {base_dir} with suffix {suffix}")
    return candidates[-1]


def _parse_scores(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "greedy_score": r"Greedy SCORE:\s*([0-9.]+)",
        "sampling_score": r"Sampling SCORE:\s*([0-9.]+)",
    }
    out: dict[str, float] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            out[key] = float(matches[-1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sym-NCO POMO TSP20 for 3 epochs and export summary")
    parser.add_argument("--python", default="/opt/conda/envs/py11/bin/python")
    parser.add_argument("--symnco-root", default="external_baselines/sym_nco/upstream")
    parser.add_argument("--test-pkl", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-episodes", type=int, default=65536)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--test-episodes", type=int, default=10000)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--aug-factor", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tsp_dir = (PROJECT_ROOT / args.symnco_root / "Sym-NCO-POMO" / "TSP").resolve()
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["SYMNCO_PROBLEM_SIZE"] = "20"
    env["SYMNCO_POMO_SIZE"] = "20"
    env["SYMNCO_SR_SIZE"] = "2"
    env["SYMNCO_EPOCHS"] = str(args.epochs)
    env["SYMNCO_TRAIN_EPISODES"] = str(args.train_episodes)
    env["SYMNCO_TRAIN_BATCH_SIZE"] = str(args.train_batch_size)
    env["SYMNCO_TEST_EPISODES"] = str(args.test_episodes)
    env["SYMNCO_TEST_BATCH_SIZE"] = str(args.test_batch_size)
    env["SYMNCO_AUG_BATCH_SIZE"] = str(args.test_batch_size)
    env["SYMNCO_AUG_FACTOR"] = str(args.aug_factor)
    env["SYMNCO_USE_CUDA"] = env.get("SYMNCO_USE_CUDA", "1")
    env["SYMNCO_CUDA_DEVICE_NUM"] = env.get("SYMNCO_CUDA_DEVICE_NUM", "0")
    env["SYMNCO_WANDB"] = "0"

    before_train = set(p for p in (tsp_dir / "result").glob("*") if p.is_dir())
    train_proc = subprocess.run(
        [args.python, "train_symnco.py"], cwd=str(tsp_dir), env=env, text=True, capture_output=True, check=False
    )
    if train_proc.returncode != 0:
        raise RuntimeError(f"Training failed\nSTDOUT:\n{train_proc.stdout}\nSTDERR:\n{train_proc.stderr}")

    after_train = set(p for p in (tsp_dir / "result").glob("*") if p.is_dir())
    new_train_dirs = sorted(after_train - before_train, key=lambda p: p.stat().st_mtime)
    train_dir = new_train_dirs[-1] if new_train_dirs else _latest_result_dir(tsp_dir / "result", "train__tsp_n20")

    env["SYMNCO_MODEL_PATH"] = str(train_dir)
    env["SYMNCO_MODEL_EPOCH"] = str(args.epochs)
    env["SYMNCO_TEST_DATASET"] = str(Path(args.test_pkl).resolve())

    before_test = set(p for p in (tsp_dir / "result").glob("*") if p.is_dir())
    test_proc = subprocess.run(
        [args.python, "test_symnco.py"], cwd=str(tsp_dir), env=env, text=True, capture_output=True, check=False
    )
    if test_proc.returncode != 0:
        raise RuntimeError(f"Testing failed\nSTDOUT:\n{test_proc.stdout}\nSTDERR:\n{test_proc.stderr}")

    after_test = set(p for p in (tsp_dir / "result").glob("*") if p.is_dir())
    new_test_dirs = sorted(after_test - before_test, key=lambda p: p.stat().st_mtime)
    test_dir = new_test_dirs[-1] if new_test_dirs else _latest_result_dir(tsp_dir / "result", "test__tsp_n20")

    scores = _parse_scores(test_dir / "run_log")
    payload = {
        "method": "sym_nco",
        "variant": "Sym-NCO-POMO-TSP",
        "problem_size": 20,
        "epochs": args.epochs,
        "train_episodes": args.train_episodes,
        "train_batch_size": args.train_batch_size,
        "test_episodes": args.test_episodes,
        "test_batch_size": args.test_batch_size,
        "aug_factor": args.aug_factor,
        "checkpoint_dir": str(train_dir),
        "checkpoint": str(train_dir / f"checkpoint-{args.epochs}.pt"),
        "test_dataset": str(Path(args.test_pkl).resolve()),
        "train_stdout": train_proc.stdout,
        "train_stderr": train_proc.stderr,
        "test_stdout": test_proc.stdout,
        "test_stderr": test_proc.stderr,
        "scores": scores,
        "avg_cost": scores.get("greedy_score"),
        "raw_result_dir": str(test_dir),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()