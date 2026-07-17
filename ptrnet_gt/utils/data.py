from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_dataset_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        coords = payload.get("coords")
        metadata = {k: v for k, v in payload.items() if k != "coords"}
    else:
        coords = payload
        metadata = {}
    if coords is None:
        raise ValueError(f"Dataset at {path} does not contain 'coords'")
    coords = torch.as_tensor(coords, dtype=torch.float32)
    if coords.ndim != 3 or coords.size(-1) != 2:
        raise ValueError(f"Expected coords with shape [N, n_nodes, 2], got {tuple(coords.shape)}")
    metadata.setdefault("dataset_path", str(path))
    metadata.setdefault("num_instances", int(coords.size(0)))
    metadata.setdefault("size", int(coords.size(1)))
    return {"coords": coords, **metadata}


def save_dataset_payload(
    path: str | Path,
    coords: torch.Tensor,
    *,
    seed: int,
    distribution: str = "uniform_2d",
    scale: float = 1.0,
) -> dict[str, Any]:
    coords = torch.as_tensor(coords, dtype=torch.float32).cpu()
    if coords.ndim != 3 or coords.size(-1) != 2:
        raise ValueError(f"Expected coords with shape [N, n_nodes, 2], got {tuple(coords.shape)}")
    payload = {
        "coords": coords,
        "seed": int(seed),
        "distribution": distribution,
        "scale": float(scale),
        "size": int(coords.size(1)),
        "num_instances": int(coords.size(0)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def training_budget_from_config(config: dict[str, Any]) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    batch_size = int(training_cfg.get("batch_size", 512))
    epoch_size = int(training_cfg.get("epoch_size", batch_size * 128))
    epochs = int(training_cfg.get("epochs", 1))
    updates_per_epoch = epoch_size // batch_size if batch_size > 0 else 0
    return {
        "training_updates": int(epochs * updates_per_epoch),
        "training_samples": int(epochs * epoch_size),
        "seed": int(experiment_cfg.get("seed", 1234)),
    }


__all__ = ["load_dataset_payload", "save_dataset_payload", "training_budget_from_config"]