from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from ptrnet_gt.config import apply_overrides, load_config
from ptrnet_gt.models.factory import build_model
from ptrnet_gt.problems import TSP


def main():
    parser = argparse.ArgumentParser(description="Ptr-net-GT unified evaluation entry")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint saved by scripts/train.py")
    parser.add_argument("--override", action="append", default=[], help="Override config entries")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.override)
    problem = TSP()
    device = torch.device("cuda:0" if torch.cuda.is_available() and config.get("training", {}).get("use_cuda", True) else "cpu")
    model = build_model(config, problem).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    model.set_decode_type("greedy")

    dataset = problem.make_dataset(
        size=config.get("problem", {}).get("size", 20),
        num_samples=config.get("evaluation", {}).get("num_instances", 128),
        distribution=config.get("problem", {}).get("distribution"),
    )
    dataloader = DataLoader(dataset, batch_size=config.get("evaluation", {}).get("batch_size", 128))

    costs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            cost, _ = model(batch)
            costs.append(cost.cpu())

    costs = torch.cat(costs, dim=0)
    result = {
        "avg_cost": float(costs.mean().item()),
        "std_cost": float(costs.std().item()),
        "num_instances": int(costs.numel()),
        "checkpoint": args.checkpoint,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    output_root = Path(config.get("output", {}).get("root", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "last_eval.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()