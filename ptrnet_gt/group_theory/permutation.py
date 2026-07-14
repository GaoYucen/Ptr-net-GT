from __future__ import annotations

import torch


def inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.size(0), device=permutation.device)
    return inverse


def compose_permutations(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return first[second]


def permute_nodes(x: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return x[..., permutation, :]


def permute_tour(tour: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return permutation[tour]


def permute_edge(edge: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return permutation[edge]