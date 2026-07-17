from typing import NamedTuple

import torch


class ComponentMergeState(NamedTuple):
    coords: torch.Tensor
    dist: torch.Tensor
    ids: torch.Tensor
    selected_tails: torch.Tensor
    selected_heads: torch.Tensor
    tail_available: torch.Tensor
    head_available: torch.Tensor
    out_degree: torch.Tensor
    in_degree: torch.Tensor
    component_id: torch.Tensor
    component_start: torch.Tensor
    component_end: torch.Tensor
    component_size: torch.Tensor
    selected_edges: torch.Tensor
    lengths: torch.Tensor
    step: torch.Tensor

    @staticmethod
    def initialize(coords: torch.Tensor):
        batch_size, n_nodes, _ = coords.size()
        device = coords.device
        arange_n = torch.arange(n_nodes, device=device, dtype=torch.long)
        return ComponentMergeState(
            coords=coords,
            dist=(coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.long, device=device)[:, None],
            selected_tails=torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=device),
            selected_heads=torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=device),
            tail_available=torch.ones(batch_size, n_nodes, dtype=torch.bool, device=device),
            head_available=torch.ones(batch_size, n_nodes, dtype=torch.bool, device=device),
            out_degree=torch.zeros(batch_size, n_nodes, dtype=torch.long, device=device),
            in_degree=torch.zeros(batch_size, n_nodes, dtype=torch.long, device=device),
            component_id=arange_n[None, :].expand(batch_size, n_nodes).clone(),
            component_start=arange_n[None, :].expand(batch_size, n_nodes).clone(),
            component_end=arange_n[None, :].expand(batch_size, n_nodes).clone(),
            component_size=torch.ones(batch_size, n_nodes, dtype=torch.long, device=device),
            selected_edges=torch.zeros(batch_size, n_nodes, n_nodes, dtype=torch.bool, device=device),
            lengths=torch.zeros(batch_size, 1, device=device),
            step=torch.zeros(1, dtype=torch.long, device=device)
        )

    @property
    def n_nodes(self):
        return self.coords.size(1)

    @property
    def batch_size(self):
        return self.coords.size(0)

    def all_finished(self):
        return self.step.item() >= self.n_nodes

    def _component_node_mask(self, component_ids: torch.Tensor):
        return self.component_id == component_ids[:, None]

    def get_edge_mask(self):
        batch_size, n_nodes = self.batch_size, self.n_nodes
        mask = torch.ones(batch_size, n_nodes, n_nodes, dtype=torch.bool, device=self.coords.device)

        tail_candidates = self.tail_available & (self.out_degree == 0)
        head_candidates = self.head_available & (self.in_degree == 0)

        mask &= ~tail_candidates[:, :, None]
        mask |= ~head_candidates[:, None, :]
        mask |= torch.eye(n_nodes, dtype=torch.bool, device=self.coords.device)[None, :, :]

        tail_comp = self.component_id[:, :, None]
        head_comp = self.component_id[:, None, :]
        same_component = tail_comp == head_comp

        is_last_step = self.step.item() == n_nodes - 1
        if not is_last_step:
            mask |= same_component
        else:
            active_components = self._count_active_components()
            allow_close = same_component & (active_components[:, None, None] == 1)
            mask |= ~allow_close

        return mask

    def _count_active_components(self):
        batch_size, n_nodes = self.batch_size, self.n_nodes
        active = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=self.coords.device)
        active.scatter_(1, self.component_id, True)
        return active.sum(dim=1)

    def get_node_role_features(self):
        device = self.coords.device
        batch_size, n_nodes = self.batch_size, self.n_nodes
        batch_idx = torch.arange(batch_size, device=device)[:, None]
        node_idx = torch.arange(n_nodes, device=device)[None, :].expand(batch_size, n_nodes)

        component_sizes = self.component_size[batch_idx, self.component_id]
        component_starts = self.component_start[batch_idx, self.component_id]
        component_ends = self.component_end[batch_idx, self.component_id]

        is_isolated = (component_sizes == 1).float()
        is_start = ((node_idx == component_starts) & (component_sizes > 1)).float()
        is_end = ((node_idx == component_ends) & (component_sizes > 1)).float()
        is_internal = (1.0 - is_isolated - is_start - is_end).clamp(min=0.0)
        normalized_component_size = component_sizes.float() / float(n_nodes)
        step_ratio = torch.full((batch_size, n_nodes), float(self.step.item()) / float(n_nodes), device=device)
        in_degree = self.in_degree.float()
        out_degree = self.out_degree.float()

        return torch.stack(
            (
                is_isolated,
                is_start,
                is_end,
                is_internal,
                normalized_component_size,
                step_ratio,
                in_degree,
                out_degree,
            ),
            dim=-1,
        )

    def update_edge(self, edge_idx: torch.Tensor):
        n_nodes = self.n_nodes
        tail_idx = torch.div(edge_idx, n_nodes, rounding_mode='floor')
        head_idx = edge_idx % n_nodes
        return self.update(tail_idx, head_idx)

    def update(self, tail_idx: torch.Tensor, head_idx: torch.Tensor):
        mask = self.get_edge_mask()
        invalid = mask[self.ids.squeeze(1), tail_idx, head_idx]
        if invalid.any():
            raise ValueError("Attempted to apply invalid edge in ComponentMergeState.update")

        batch_idx = self.ids.squeeze(1)
        step_idx = self.step.item()

        new_selected_tails = self.selected_tails.clone()
        new_selected_heads = self.selected_heads.clone()
        new_selected_tails[:, step_idx] = tail_idx
        new_selected_heads[:, step_idx] = head_idx

        new_tail_available = self.tail_available.scatter(1, tail_idx[:, None], False)
        new_head_available = self.head_available.scatter(1, head_idx[:, None], False)
        new_out_degree = self.out_degree.scatter_add(1, tail_idx[:, None], torch.ones_like(tail_idx[:, None]))
        new_in_degree = self.in_degree.scatter_add(1, head_idx[:, None], torch.ones_like(head_idx[:, None]))

        new_selected_edges = self.selected_edges.clone()
        new_selected_edges[batch_idx, tail_idx, head_idx] = True

        tail_gid = self.component_id[batch_idx, tail_idx]
        head_gid = self.component_id[batch_idx, head_idx]
        tail_start = self.component_start[batch_idx, tail_gid]
        head_end = self.component_end[batch_idx, head_gid]
        merged_size = self.component_size[batch_idx, tail_gid] + self.component_size[batch_idx, head_gid]

        new_component_id = self.component_id.clone()
        head_group_nodes = new_component_id == head_gid[:, None]
        new_component_id = torch.where(head_group_nodes, tail_gid[:, None], new_component_id)

        new_component_start = self.component_start.clone()
        new_component_end = self.component_end.clone()
        new_component_size = self.component_size.clone()
        new_component_start[batch_idx, tail_gid] = tail_start
        new_component_end[batch_idx, tail_gid] = head_end
        new_component_size[batch_idx, tail_gid] = merged_size

        step_dist = self.dist[batch_idx, tail_idx, head_idx][:, None]
        new_lengths = self.lengths + step_dist

        return self._replace(
            selected_tails=new_selected_tails,
            selected_heads=new_selected_heads,
            tail_available=new_tail_available,
            head_available=new_head_available,
            out_degree=new_out_degree,
            in_degree=new_in_degree,
            component_id=new_component_id,
            component_start=new_component_start,
            component_end=new_component_end,
            component_size=new_component_size,
            selected_edges=new_selected_edges,
            lengths=new_lengths,
            step=self.step + 1,
        )

    def get_final_cost(self):
        assert self.all_finished(), "Final cost is only available after all edges are selected"
        return self.lengths.squeeze(-1)

    def to_tour(self):
        assert self.all_finished(), "Tour reconstruction requires a finished state"
        batch_size, n_nodes = self.batch_size, self.n_nodes
        batch_idx = torch.arange(batch_size, device=self.coords.device)
        succ = self.selected_edges.long().argmax(dim=-1)
        tour = torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=self.coords.device)
        current = torch.zeros(batch_size, dtype=torch.long, device=self.coords.device)
        for step in range(n_nodes):
            tour[:, step] = current
            current = succ[batch_idx, current]
        return tour


__all__ = ["ComponentMergeState"]