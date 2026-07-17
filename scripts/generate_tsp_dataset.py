from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from ptrnet_gt.utils.data import save_dataset_payload


def main():
    parser = argparse.ArgumentParser(description="Generate fixed TSP dataset with unified .pt protocol")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--num-instances", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--distribution", default="uniform_2d")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)
    coords = torch.rand((args.num_instances, args.size, 2), generator=generator) * args.scale
    payload = save_dataset_payload(
        args.output,
        coords,
        seed=args.seed,
        distribution=args.distribution,
        scale=args.scale,
    )
    print(payload)


if __name__ == "__main__":
    main()