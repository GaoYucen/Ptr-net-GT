"""固定环数最小代价 Cycle Cover 的张量化构造状态。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BatchedMCycleCoverState:
    """维护若干开放路径和已经闭合的互不相交环。"""

    coordinates: Tensor
    successor: Tensor
    predecessor: Tensor
    component: Tensor
    closed_vertex: Tensor
    closed_cycles: Tensor
    target_cycles: int
    edges_added: int

    @classmethod
    def initialize(cls, coordinates: Tensor, target_cycles: int) -> BatchedMCycleCoverState:
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        batch_size, node_count, _ = coordinates.shape
        if target_cycles < 1 or node_count < 3 * target_cycles:
            raise ValueError("an undirected m-cycle cover needs at least 3m vertices")
        device = coordinates.device
        return cls(
            coordinates=coordinates,
            successor=torch.full(
                (batch_size, node_count), -1, dtype=torch.long, device=device
            ),
            predecessor=torch.full(
                (batch_size, node_count), -1, dtype=torch.long, device=device
            ),
            component=torch.arange(node_count, device=device)
            .unsqueeze(0)
            .expand(batch_size, node_count)
            .clone(),
            closed_vertex=torch.zeros(
                batch_size, node_count, dtype=torch.bool, device=device
            ),
            closed_cycles=torch.zeros(batch_size, dtype=torch.long, device=device),
            target_cycles=target_cycles,
            edges_added=0,
        )

    @property
    def batch_size(self) -> int:
        return self.coordinates.size(0)

    @property
    def n(self) -> int:
        return self.coordinates.size(1)

    @property
    def terminal(self) -> bool:
        if self.edges_added < self.n:
            return False
        if not torch.all(self.closed_cycles == self.target_cycles):
            raise RuntimeError("terminal cycle cover does not contain exactly m cycles")
        return True

    def _component_size(self) -> Tensor:
        labels = self.component.unsqueeze(-1)
        ones = torch.ones_like(labels)
        size_by_label = torch.zeros_like(ones).scatter_add(1, labels, ones)
        return size_by_label.gather(1, labels).squeeze(-1)

    def edge_action_mask(self, fixed_tail: Tensor | None = None) -> Tensor:
        """返回合并开放路径或合法闭环的联合动作 mask。"""
        if self.terminal:
            return torch.ones(
                self.batch_size,
                self.n,
                self.n,
                dtype=torch.bool,
                device=self.coordinates.device,
            )

        open_tail = self.successor < 0
        open_head = self.predecessor < 0
        same_component = self.component.unsqueeze(2) == self.component.unsqueeze(1)

        remaining_cycles = self.target_cycles - self.closed_cycles
        component_size = self._component_size()

        # 开放分量的规模只有 1、2、>=3 三类。一次合并后，若大分量数不足以
        # 单独形成剩余环，则规模 1/2 的分量必须还能被分组成若干个总规模至少 3
        # 的集合。下面的计数条件对 1/2 元素是充要的，可阻止走入死状态。
        count_one = (open_tail & (component_size == 1)).sum(dim=1)
        count_two = (open_tail & (component_size == 2)).sum(dim=1)
        count_large = (open_tail & (component_size >= 3)).sum(dim=1)

        left_size = component_size.unsqueeze(2)
        right_size = component_size.unsqueeze(1)
        merged_size = left_size + right_size
        merge_one = (
            count_one[:, None, None]
            - (left_size == 1).to(torch.long)
            - (right_size == 1).to(torch.long)
            + (merged_size == 1).to(torch.long)
        )
        merge_two = (
            count_two[:, None, None]
            - (left_size == 2).to(torch.long)
            - (right_size == 2).to(torch.long)
            + (merged_size == 2).to(torch.long)
        )
        merge_large = (
            count_large[:, None, None]
            - (left_size >= 3).to(torch.long)
            - (right_size >= 3).to(torch.long)
            + (merged_size >= 3).to(torch.long)
        )
        merge_feasible = self._completion_feasible(
            merge_one,
            merge_two,
            merge_large,
            remaining_cycles[:, None, None],
        )
        merge_valid = ~same_component & merge_feasible

        close_remaining = remaining_cycles - 1
        close_feasible = self._completion_feasible(
            count_one[:, None, None],
            count_two[:, None, None],
            count_large[:, None, None] - 1,
            close_remaining[:, None, None],
        )
        close_valid = (
            same_component
            & (component_size >= 3).unsqueeze(2)
            & (self.closed_cycles < self.target_cycles)[:, None, None]
            & close_feasible
        )

        valid = (
            open_tail.unsqueeze(2)
            & open_head.unsqueeze(1)
            & ~self.closed_vertex.unsqueeze(2)
            & ~self.closed_vertex.unsqueeze(1)
            & (merge_valid | close_valid)
        )

        if fixed_tail is not None:
            self._validate_action_vector(fixed_tail, "fixed_tail")
            batch = torch.arange(self.batch_size, device=self.coordinates.device)
            selected = torch.zeros_like(open_tail)
            selected[batch, fixed_tail] = True
            valid &= selected.unsqueeze(2)

        if (~valid.flatten(1).any(dim=1)).any():
            raise RuntimeError("nonterminal m-cycle cover state has no legal action")
        return ~valid

    @staticmethod
    def _completion_feasible(
        count_one: Tensor,
        count_two: Tensor,
        count_large: Tensor,
        remaining_cycles: Tensor,
    ) -> Tensor:
        """判断当前不可拆分分量能否继续合并成指定数量的合法环。"""
        no_cycles = remaining_cycles == 0
        no_components = (count_one + count_two + count_large) == 0
        large_enough = count_large >= remaining_cycles
        groups_from_small = (remaining_cycles - count_large).clamp_min(0)
        small_feasible = (
            (count_one + count_two >= 2 * groups_from_small)
            & (count_one + 2 * count_two >= 3 * groups_from_small)
        )
        return torch.where(no_cycles, no_components, large_enough | small_feasible)

    def tail_mask(self) -> Tensor:
        return self.edge_action_mask().all(dim=-1)

    def head_mask(self, selected_tail: Tensor) -> Tensor:
        self._validate_action_vector(selected_tail, "selected_tail")
        batch = torch.arange(self.batch_size, device=self.coordinates.device)
        return self.edge_action_mask()[batch, selected_tail]

    def sequential_tail(self, anchor: int = 0) -> Tensor:
        """优先延长当前非单点路径，闭环后再开始下一条路径。"""
        if self.terminal:
            raise ValueError("a terminal state has no tail")
        if not 0 <= anchor < self.n:
            raise ValueError(f"anchor must be in [0, {self.n})")

        legal_tail = ~self.tail_mask()
        size = self._component_size()
        growing = legal_tail & (size > 1)
        indices = torch.arange(self.n, device=self.coordinates.device).unsqueeze(0)
        sentinel = torch.full_like(indices, self.n)
        growing_choice = torch.where(growing, indices, sentinel).amin(dim=1)
        any_growing = growing.any(dim=1)

        anchor_legal = legal_tail[:, anchor]
        first_legal = torch.where(legal_tail, indices, sentinel).amin(dim=1)
        fresh_choice = torch.where(
            anchor_legal,
            torch.full_like(first_legal, anchor),
            first_legal,
        )
        return torch.where(any_growing, growing_choice, fresh_choice)

    def update(self, selected_tail: Tensor, selected_head: Tensor) -> BatchedMCycleCoverState:
        self._validate_action_vector(selected_tail, "selected_tail")
        self._validate_action_vector(selected_head, "selected_head")
        batch = torch.arange(self.batch_size, device=self.coordinates.device)
        mask = self.head_mask(selected_tail)
        if mask[batch, selected_head].any():
            raise ValueError("selected edge violates the m-cycle cover mask")

        old_component = self.component
        left = old_component.gather(1, selected_tail[:, None])
        right = old_component.gather(1, selected_head[:, None])
        closes_cycle = left.squeeze(1) == right.squeeze(1)

        successor = self.successor.clone()
        predecessor = self.predecessor.clone()
        successor[batch, selected_tail] = selected_head
        predecessor[batch, selected_head] = selected_tail

        merged_label = torch.minimum(left, right)
        in_merged = (old_component == left) | (old_component == right)
        component = torch.where(
            (~closes_cycle)[:, None] & in_merged,
            merged_label,
            old_component,
        )
        newly_closed = closes_cycle[:, None] & (old_component == left)
        closed_vertex = self.closed_vertex | newly_closed
        closed_cycles = self.closed_cycles + closes_cycle.to(torch.long)

        return BatchedMCycleCoverState(
            coordinates=self.coordinates,
            successor=successor,
            predecessor=predecessor,
            component=component,
            closed_vertex=closed_vertex,
            closed_cycles=closed_cycles,
            target_cycles=self.target_cycles,
            edges_added=self.edges_added + 1,
        )

    def forest_decoder_features(self, node_embeddings: Tensor) -> Tensor:
        if node_embeddings.shape[:2] != (self.batch_size, self.n):
            raise ValueError("node_embeddings have an incompatible shape")
        labels = self.component.unsqueeze(-1)
        embedding_labels = labels.expand_as(node_embeddings)
        component_sum_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_labels, node_embeddings
        )
        component_sum = component_sum_by_label.gather(1, embedding_labels)
        ones = torch.ones_like(labels, dtype=node_embeddings.dtype)
        size_by_label = torch.zeros_like(ones).scatter_add(1, labels, ones)
        size = size_by_label.gather(1, labels)
        component_mean = component_sum / size

        start = node_embeddings * (self.predecessor < 0).unsqueeze(-1)
        end = node_embeddings * (self.successor < 0).unsqueeze(-1)
        start_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_labels, start
        )
        end_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_labels, end
        )
        path_start = start_by_label.gather(1, embedding_labels)
        path_end = end_by_label.gather(1, embedding_labels)
        return torch.cat((component_mean, path_start, path_end, size / self.n), dim=-1)

    def path_state_features(self, node_embeddings: Tensor) -> Tensor:
        features = self.forest_decoder_features(node_embeddings)
        embedding_dim = node_embeddings.size(-1)
        return torch.cat((features[:, :, : 2 * embedding_dim], features[:, :, -1:]), dim=-1)

    def edge_cost(self, tails: Tensor, heads: Tensor) -> Tensor:
        gather_shape = (*tails.shape, 2)
        tail_coordinates = self.coordinates.gather(
            1, tails.unsqueeze(-1).expand(gather_shape)
        )
        head_coordinates = self.coordinates.gather(
            1, heads.unsqueeze(-1).expand(gather_shape)
        )
        return (tail_coordinates - head_coordinates).norm(dim=-1).sum(dim=1)

    def _validate_action_vector(self, action: Tensor, name: str) -> None:
        if action.shape != (self.batch_size,) or action.dtype != torch.long:
            raise ValueError(f"{name} must be a long tensor with shape (batch,)")
        if ((action < 0) | (action >= self.n)).any():
            raise ValueError(f"{name} contains an out-of-range vertex")


class BatchedMCycleCoverConstruction:
    """固定目标环数的无参数问题插件。"""

    def __init__(self, target_cycles: int) -> None:
        if target_cycles < 1:
            raise ValueError("target_cycles must be positive")
        self.target_cycles = target_cycles

    def initial_state(self, instance: Tensor) -> BatchedMCycleCoverState:
        return BatchedMCycleCoverState.initialize(instance, self.target_cycles)

    def is_terminal(self, state: BatchedMCycleCoverState) -> bool:
        return state.terminal

    def base_mask(self, state: BatchedMCycleCoverState) -> Tensor:
        return state.tail_mask()

    def representative_mask(
        self, state: BatchedMCycleCoverState, selected_base: Tensor
    ) -> Tensor:
        return state.head_mask(selected_base)

    def action_mask(
        self, state: BatchedMCycleCoverState, fixed_base: Tensor | None = None
    ) -> Tensor:
        return state.edge_action_mask(fixed_base)

    def fixed_base(self, state: BatchedMCycleCoverState, anchor: int = 0) -> Tensor:
        return state.sequential_tail(anchor)

    def transition(
        self,
        state: BatchedMCycleCoverState,
        selected_base: Tensor,
        selected_representative: Tensor,
    ) -> BatchedMCycleCoverState:
        return state.update(selected_base, selected_representative)

    def objective(
        self,
        state: BatchedMCycleCoverState,
        selected_bases: Tensor,
        selected_representatives: Tensor,
    ) -> Tensor:
        if not state.terminal:
            raise ValueError("cannot score an unfinished m-cycle cover")
        return state.edge_cost(selected_bases, selected_representatives)

    def solution(self, state: BatchedMCycleCoverState) -> Tensor:
        if not state.terminal:
            raise ValueError("cannot decode an unfinished m-cycle cover")
        return state.successor
