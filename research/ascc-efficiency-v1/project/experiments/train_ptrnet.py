"""训练 Original 或 GroupOpt Pointer Network 实验。"""

from __future__ import annotations

import argparse

from torch import nn
from train import parse_args, run

from groupopt.models.ptrnet import PointerNetwork


def build_ptrnet(args: argparse.Namespace) -> nn.Module:
    return PointerNetwork(
        embedding_dim=args.embedding_dim,
        n_encoder_layers=args.encoder_layers,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "ptrnet"
    run(arguments, model_builder=build_ptrnet)
