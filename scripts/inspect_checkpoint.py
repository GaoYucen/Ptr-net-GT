import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(type(checkpoint))
    if isinstance(checkpoint, dict):
        print("keys:", sorted(checkpoint.keys()))


if __name__ == "__main__":
    main()