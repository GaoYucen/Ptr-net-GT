from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from rl4co.envs import TSPEnv
from rl4co.models.zoo.pomo import POMO


def main():
    parser = argparse.ArgumentParser(description="Train RL4CO POMO on TSP")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=2)
    parser.add_argument("--limit-val-batches", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/rl4co_pomo_tsp20_smoke")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    env = TSPEnv(generator_params={"num_loc": args.size})
    model = POMO(env=env)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(dirpath=str(output_dir), save_last=True, save_top_k=1, monitor=None)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[checkpoint_cb],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    trainer.fit(model)

    payload = {
        "method": "rl4co_pomo",
        "size": args.size,
        "max_epochs": args.max_epochs,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "seed": args.seed,
        "best_model_path": checkpoint_cb.best_model_path,
        "last_model_path": checkpoint_cb.last_model_path,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
