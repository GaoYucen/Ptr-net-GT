from __future__ import annotations

import torch


def _pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    return torch.cdist(coords, coords, p=2)


def nearest_neighbor_tour(coords: torch.Tensor, start_node: int = 0) -> torch.Tensor:
    batch_size, n_nodes, _ = coords.size()
    distances = _pairwise_distances(coords)
    device = coords.device
    tours = torch.empty(batch_size, n_nodes, dtype=torch.long, device=device)
    visited = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
    current = torch.full((batch_size,), start_node, dtype=torch.long, device=device)

    for step in range(n_nodes):
        tours[:, step] = current
        visited[torch.arange(batch_size, device=device), current] = True
        if step == n_nodes - 1:
            break
        candidate_dist = distances[torch.arange(batch_size, device=device), current].clone()
        candidate_dist[visited] = float("inf")
        current = candidate_dist.argmin(dim=1)

    return tours


def nearest_neighbor_multistart_tour(coords: torch.Tensor) -> torch.Tensor:
    _, n_nodes, _ = coords.size()
    best_tour = None
    best_cost = None
    for start_node in range(n_nodes):
        candidate = nearest_neighbor_tour(coords, start_node=start_node)
        candidate_cost = _tour_length(coords, candidate)
        if best_tour is None:
            best_tour = candidate
            best_cost = candidate_cost
            continue
        improved = candidate_cost < best_cost
        best_tour = torch.where(improved[:, None], candidate, best_tour)
        best_cost = torch.where(improved, candidate_cost, best_cost)
    return best_tour


def _tour_length(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    ordered = coords.gather(1, tour.unsqueeze(-1).expand(-1, -1, coords.size(-1)))
    return (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1).sum(dim=1) + (ordered[:, 0] - ordered[:, -1]).norm(p=2, dim=-1)


__all__ = ["nearest_neighbor_tour", "nearest_neighbor_multistart_tour"]