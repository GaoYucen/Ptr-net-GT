from __future__ import annotations

import json
from pathlib import Path

import torch


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=torch.device("cpu"))


def load_model_file(load_path):
    payload = torch_load_cpu(load_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {load_path} must be a dict payload")
    return payload


def load_args(run_dir):
    run_dir = Path(run_dir)
    for candidate in (run_dir / "config.json", run_dir / "args.json"):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No config.json or args.json found under {run_dir}")


__all__ = ["torch_load_cpu", "load_model_file", "load_args"]