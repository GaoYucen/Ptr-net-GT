"""训练 Original 或 GroupOpt Graph Pointer Network 实验。"""

from __future__ import annotations

import argparse

from torch import nn
from train import parse_args, run

from groupopt.models.gpn import GraphPointerNetwork


def build_gpn(args: argparse.Namespace) -> nn.Module:
    return GraphPointerNetwork(
        embedding_dim=args.embedding_dim,
        n_encoder_layers=args.encoder_layers,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "gpn"
    run(arguments, model_builder=build_gpn)
