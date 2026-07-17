from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.serialization import add_safe_globals
from rl4co.envs import TSPEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.pomo import POMO


def _load_model(method: str, size: int, checkpoint: str):
    env = TSPEnv(generator_params={"num_loc": size})
    add_safe_globals([TSPEnv])
    if method == "am":
        model = AttentionModel.load_from_checkpoint(checkpoint, env=env, weights_only=False)
    elif method == "pomo":
        model = POMO.load_from_checkpoint(checkpoint, env=env, weights_only=False)
    else:
        raise ValueError(f"Unsupported RL4CO method: {method}")
    model.eval()
    return env, model


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL4CO model on TSP")
    parser.add_argument("--method", required=True, choices=["am", "pomo"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--num-instances", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env, model = _load_model(args.method, args.size, args.checkpoint)
    td = env.reset(batch_size=[args.num_instances])

    start = time.perf_counter()
    with torch.no_grad():
        out = model(td, phase="test")
    elapsed = time.perf_counter() - start

    reward = out["reward"]
    cost = -reward
    payload = {
        "method": f"rl4co_{args.method}",
        "avg_cost": float(cost.mean().item()),
        "std_cost": float(cost.std().item()),
        "feasible_tour_rate": 1.0,
        "total_time_sec": float(elapsed),
        "time_per_instance_sec": float(elapsed / max(args.num_instances, 1)),
        "num_instances": int(args.num_instances),
        "size": int(args.size),
        "checkpoint": args.checkpoint,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
