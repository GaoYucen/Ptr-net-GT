from __future__ import annotations

import torch

from ptrnet_gt.baselines.nearest_neighbor import nearest_neighbor_tour


def tour_length(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    return (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1).sum(dim=1) + (ordered[:, 0] - ordered[:, -1]).norm(p=2, dim=-1)


def two_opt_tour(coords: torch.Tensor, tour: torch.Tensor, max_iterations: int | None = None) -> torch.Tensor:
    batch_size, n_nodes = tour.size()
    improved_tour = tour.clone()
    if max_iterations is None:
        max_iterations = n_nodes * n_nodes

    for batch_idx in range(batch_size):
        current_tour = improved_tour[batch_idx].clone()
        current_cost = tour_length(coords[batch_idx : batch_idx + 1], current_tour.unsqueeze(0))[0]
        iterations = 0
        improved = True
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(1, n_nodes - 1):
                for j in range(i + 1, n_nodes):
                    if j - i == 1:
                        continue
                    candidate = current_tour.clone()
                    candidate[i:j] = torch.flip(current_tour[i:j], dims=[0])
                    candidate_cost = tour_length(coords[batch_idx : batch_idx + 1], candidate.unsqueeze(0))[0]
                    if candidate_cost + 1e-12 < current_cost:
                        current_tour = candidate
                        current_cost = candidate_cost
                        improved = True
            improved_tour[batch_idx] = current_tour
    return improved_tour


def nearest_neighbor_two_opt_tour(coords: torch.Tensor, start_node: int = 0, max_iterations: int | None = None) -> torch.Tensor:
    return two_opt_tour(coords, nearest_neighbor_tour(coords, start_node=start_node), max_iterations=max_iterations)


__all__ = ["tour_length", "two_opt_tour", "nearest_neighbor_two_opt_tour"]