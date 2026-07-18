from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptrnet_gt.utils.data import load_dataset_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert repository .pt TSP dataset to Sym-NCO .pkl format")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = load_dataset_payload(args.input)
    coords = payload["coords"].detach().cpu().tolist()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(coords, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        {
            "input": args.input,
            "output": str(output_path),
            "num_instances": int(payload["num_instances"]),
            "size": int(payload["size"]),
        }
    )


if __name__ == "__main__":
    main()