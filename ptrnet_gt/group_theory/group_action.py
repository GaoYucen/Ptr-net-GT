from __future__ import annotations

import torch

from .permutation import permute_nodes


def act_on_instance(instance: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return permute_nodes(instance, permutation)


def act_on_state(state, permutation: torch.Tensor):
    if hasattr(state, "coords"):
        return state._replace(coords=permute_nodes(state.coords, permutation))
    return state


def act_on_mask(mask: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return mask[:, permutation][:, :, permutation]


def act_on_logits(logits: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return logits[..., permutation]