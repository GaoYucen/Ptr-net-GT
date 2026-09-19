"""Shared, audit-friendly primitives for Pair-ASCC validation experiments.

The internal methods intentionally expose the difference between the historical
single-edge path and the fixed-source optimized path.  External AM/POMO
baselines require an adapter/checkpoint and are rejected here rather than being
silently substituted with a different model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from groupopt.models.ascc_bopo import SourceFirstASCC
from groupopt.models.pair_ascc import PairASCC

INTERNAL_METHODS = ('single_naive', 'single_opt', 'seq2', 'pair_add', 'pair_interaction')
ALL_METHODS = ('am', 'pomo', *INTERNAL_METHODS)


def sha256_tensor(value: Tensor) -> str:
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def make_dataset(count: int, nodes: int, distribution: str, seed: int) -> Tensor:
    """Generate a frozen synthetic TSP dataset without train/test leakage."""
    if distribution not in ('uniform', 'cluster', 'mixed'):
        raise ValueError(f'unknown distribution: {distribution}')
    generator = torch.Generator().manual_seed(seed)
    if distribution == 'uniform':
        return torch.rand(count, nodes, 2, generator=generator)
    clusters = 5
    centers = .15 + .70 * torch.rand(count, clusters, 2, generator=generator)
    labels = torch.randint(0, clusters, (count, nodes), generator=generator)
    clustered = (centers.gather(1, labels[..., None].expand(-1, -1, 2))
                 + .04 * torch.randn(count, nodes, 2, generator=generator)).clamp(0, 1)
    if distribution == 'cluster':
        return clustered
    uniform = torch.rand(count, nodes, 2, generator=generator)
    selector = torch.rand(count, generator=generator) < .5
    return torch.where(selector[:, None, None], uniform, clustered)


def build_model(method: str, device: torch.device, seed: int = 0):
    if method not in INTERNAL_METHODS:
        raise ValueError(f'{method} needs an explicit external-baseline adapter')
    torch.manual_seed(seed)
    if method in ('single_naive', 'single_opt', 'seq2'):
        return SourceFirstASCC().to(device)
    pair = PairASCC().to(device)
    # Pair-Add/P3 use exactly the same shared parameters as Single-ASCC.
    torch.manual_seed(seed)
    single = SourceFirstASCC().to(device)
    pair.load_state_dict({**pair.state_dict(), **single.state_dict()}, strict=True)
    return pair


def run_model(model, method: str, coordinates: Tensor, trajectories: int,
              decode: str, generator: torch.Generator | None = None):
    if method == 'single_naive':
        return model(coordinates, trajectories, 'fixed', decode, generator)
    if method in ('single_opt', 'seq2'):
        return model(coordinates, trajectories, 'fixed', decode, generator,
                     skip_source_scoring=True)
    if method == 'pair_add':
        return model(coordinates, trajectories, interaction=False, decode=decode,
                     generator=generator)
    if method == 'pair_interaction':
        return model(coordinates, trajectories, interaction=True, decode=decode,
                     generator=generator)
    raise ValueError(method)


def effective_parameter_count(model, method: str) -> int:
    ignored = {'pair_left.weight', 'pair_right.weight', 'pair_scale'} if method == 'pair_add' else set()
    return sum(param.numel() for name, param in model.named_parameters() if name not in ignored)


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class RunSpec:
    method: str
    nodes: int
    batch: int
    trajectories: int
    distribution: str
    seed: int

