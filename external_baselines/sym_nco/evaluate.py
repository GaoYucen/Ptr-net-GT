from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptrnet_gt.utils.data import load_dataset_payload


def _build_command(command_template: str, checkpoint: str, dataset: str, raw_output: str) -> list[str]:
    command = command_template.format(checkpoint=checkpoint, dataset=dataset, raw_output=raw_output)
    return shlex.split(command)


def _load_raw_output(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Sym-NCO raw output must be a JSON object")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Adapt external Sym-NCO evaluation output into unified JSON")
    parser.add_argument("--symnco-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--command", required=True, help="Command template with {checkpoint}, {dataset}, {raw_output}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = load_dataset_payload(args.dataset)
    with tempfile.TemporaryDirectory(prefix="symnco_eval_") as tmpdir:
        raw_output_path = Path(tmpdir) / "symnco_raw.json"
        command = _build_command(args.command, args.checkpoint, args.dataset, str(raw_output_path))
        start = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=args.symnco_root,
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            raise RuntimeError(
                f"Sym-NCO command failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        if not raw_output_path.exists():
            raise FileNotFoundError(
                f"Sym-NCO command completed but did not create raw output file: {raw_output_path}"
            )
        raw_payload = _load_raw_output(raw_output_path)

    output_payload = {
        "method": "sym_nco",
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "size": int(dataset["size"]),
        "num_instances": int(dataset["num_instances"]),
        "avg_cost": raw_payload.get("avg_cost", raw_payload.get("mean_tour_length")),
        "std_cost": raw_payload.get("std_cost", raw_payload.get("std_tour_length")),
        "optimality_gap_percent": raw_payload.get("optimality_gap_percent"),
        "feasible_tour_rate": raw_payload.get("feasible_tour_rate", 1.0),
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(int(dataset["num_instances"]), 1)),
        "raw_output": raw_payload,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(output_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()