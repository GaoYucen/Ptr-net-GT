from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim

from ptrnet_gt.config import apply_overrides, load_config
from ptrnet_gt.models.factory import build_model
from ptrnet_gt.problems import TSP
from ptrnet_gt.training import ExponentialBaseline, NoBaseline, RolloutBaseline, WarmupBaseline, train_epoch, validate


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _make_opts(config: dict, run_dir: Path) -> AttrDict:
    problem_cfg = config.get("problem", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    output_cfg = config.get("output", {})
    use_cuda = torch.cuda.is_available() and train_cfg.get("use_cuda", True)
    return AttrDict(
        graph_size=problem_cfg.get("size", 20),
        batch_size=train_cfg.get("batch_size", 512),
        epoch_size=train_cfg.get("epoch_size", train_cfg.get("batch_size", 512) * 128),
        eval_batch_size=train_cfg.get("eval_batch_size", 1024),
        val_size=train_cfg.get("val_size", 256),
        n_epochs=train_cfg.get("epochs", 1),
        epoch_start=0,
        data_distribution=problem_cfg.get("distribution"),
        device=torch.device("cuda:0" if use_cuda else "cpu"),
        use_cuda=use_cuda,
        no_progress_bar=train_cfg.get("no_progress_bar", False),
        no_tensorboard=True,
        log_step=train_cfg.get("log_step", 50),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        run_name=experiment_cfg.get("name", "component_merge"),
        save_dir=str(run_dir),
        bl_alpha=train_cfg.get("bl_alpha", 0.05),
        bl_warmup_epochs=train_cfg.get("warmup_epochs", 0),
        exp_beta=train_cfg.get("exp_beta", 0.8),
        baseline=train_cfg.get("baseline", "rollout"),
    )


def main():
    parser = argparse.ArgumentParser(description="Ptr-net-GT unified training entry")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--override", action="append", default=[], help="Override config entries, e.g. training.epochs=1")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.override)
    seed = config.get("experiment", {}).get("seed", 1234)
    _set_seed(seed)

    output_root = Path(config.get("output", {}).get("root", "outputs"))
    run_dir = output_root / config.get("experiment", {}).get("name", "component_merge")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    problem = TSP()
    model = build_model(config, problem).to(torch.device("cuda:0" if torch.cuda.is_available() and config.get("training", {}).get("use_cuda", True) else "cpu"))
    opts = _make_opts(config, run_dir)

    baseline_name = config.get("training", {}).get("baseline", "rollout")
    if baseline_name == "rollout":
        baseline = RolloutBaseline(model, problem, opts)
    elif baseline_name == "exponential":
        baseline = ExponentialBaseline(config.get("training", {}).get("exp_beta", 0.8))
    elif baseline_name in (None, "none"):
        baseline = NoBaseline()
    else:
        raise ValueError(f"Unsupported TSP-only baseline for unified train entry: {baseline_name}")
    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    optimizer = optim.Adam(model.parameters(), lr=config.get("training", {}).get("learning_rate", 1e-4))
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: config.get("training", {}).get("lr_decay", 1.0) ** epoch)
    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=config.get("training", {}).get("val_size", 256), distribution=opts.data_distribution)

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, None, opts)

    avg_cost, std_cost = validate(model, val_dataset, opts)
    checkpoint_path = run_dir / "model.pt"
    torch.save({"model": model.state_dict(), "config": config, "metrics": {"avg_cost": avg_cost, "std_cost": std_cost}}, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()