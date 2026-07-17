from __future__ import annotations

import torch


def random_tour(coords: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    batch_size, n_nodes, _ = coords.size()
    device = coords.device
    tours = []
    for _ in range(batch_size):
        tours.append(torch.randperm(n_nodes, generator=generator, device=device))
    return torch.stack(tours, dim=0)


__all__ = ["random_tour"]