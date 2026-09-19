"""使用 POMO 多起点 baseline 训练 Original 或 GroupOpt。"""

from __future__ import annotations

import argparse

from torch import nn
from train import parse_args, run

from groupopt.models.pomo import POMOModel


def build_pomo(args: argparse.Namespace) -> nn.Module:
    return POMOModel(
        embedding_dim=args.embedding_dim,
        head_num=args.heads,
        qkv_dim=args.qkv_dim,
        encoder_layers=args.encoder_layers,
        feed_forward_dim=args.feed_forward_dim,
        pomo_size=args.pomo_size,
    )


if __name__ == "__main__":
    arguments = parse_args()
    arguments.model = "pomo"
    if arguments.training_scheme != "pomo":
        raise ValueError("train_pomo.py requires --training-scheme pomo")
    run(arguments, model_builder=build_pomo)
