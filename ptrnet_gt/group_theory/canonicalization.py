from __future__ import annotations

from typing import Iterable

import torch


def _as_sorted_python_ints(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in values))


def canonical_selected_edge_key(state, batch_idx: int = 0) -> tuple[tuple[int, int], ...]:
    edge_matrix = state.selected_edges[batch_idx]
    tails, heads = torch.nonzero(edge_matrix, as_tuple=True)
    edges = [(int(t), int(h)) for t, h in zip(tails.tolist(), heads.tolist())]
    edges.sort()
    return tuple(edges)


def canonicalize_components(state, batch_idx: int = 0) -> tuple[tuple[int, int, int, tuple[int, ...]], ...]:
    component_ids = state.component_id[batch_idx]
    starts = state.component_start[batch_idx]
    ends = state.component_end[batch_idx]
    sizes = state.component_size[batch_idx]

    components = []
    for comp_id in torch.unique(component_ids, sorted=True).tolist():
        node_mask = component_ids == comp_id
        nodes = torch.nonzero(node_mask, as_tuple=False).view(-1).tolist()
        components.append(
            (
                int(starts[comp_id].item()),
                int(ends[comp_id].item()),
                int(sizes[comp_id].item()),
                _as_sorted_python_ints(nodes),
            )
        )
    components.sort()
    return tuple(components)


def canonicalize_state(state, batch_idx: int = 0) -> tuple:
    return (
        canonical_selected_edge_key(state, batch_idx=batch_idx),
        canonicalize_components(state, batch_idx=batch_idx),
        int(state.step.item()),
    )


def canonical_edge_order(state, batch_idx: int = 0):
    return canonical_selected_edge_key(state, batch_idx=batch_idx)


__all__ = [
    "canonical_selected_edge_key",
    "canonicalize_components",
    "canonicalize_state",
    "canonical_edge_order",
]