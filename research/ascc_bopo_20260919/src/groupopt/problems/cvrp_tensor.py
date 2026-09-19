"""容量约束车辆路径问题的路径合并式张量状态。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BatchedCVRPState:
    """维护一组容量可行的客户路径；仓库通过实例特征共享给所有客户。"""

    instance: Tensor
    successor: Tensor
    predecessor: Tensor
    component: Tensor
    transitions: int

    @classmethod
    def initialize(cls, instance: Tensor) -> BatchedCVRPState:
        if instance.ndim != 3 or instance.size(-1) != 5:
            raise ValueError(
                "CVRP instance must have [x, y, demand, depot_x, depot_y] per customer"
            )
        batch_size, customer_count, _ = instance.shape
        if batch_size < 1 or customer_count < 2:
            raise ValueError("CVRP needs a batch and at least two customers")
        demands = instance[..., 2]
        if (demands <= 0).any() or (demands > 1).any():
            raise ValueError("normalized customer demands must be in (0, 1]")
        device = instance.device
        return cls(
            instance=instance,
            successor=torch.full(
                (batch_size, customer_count), -1, dtype=torch.long, device=device
            ),
            predecessor=torch.full(
                (batch_size, customer_count), -1, dtype=torch.long, device=device
            ),
            component=torch.arange(customer_count, device=device)
            .unsqueeze(0)
            .expand(batch_size, customer_count)
            .clone(),
            transitions=0,
        )

    @property
    def coordinates(self) -> Tensor:
        return self.instance[..., :2]

    @property
    def demands(self) -> Tensor:
        return self.instance[..., 2]

    @property
    def depot(self) -> Tensor:
        return self.instance[:, 0, 3:5]

    @property
    def batch_size(self) -> int:
        return self.instance.size(0)

    @property
    def n(self) -> int:
        return self.instance.size(1)

    def _component_load(self) -> Tensor:
        labels = self.component
        load_by_label = torch.zeros_like(self.demands).scatter_add(
            1, labels, self.demands
        )
        return load_by_label.gather(1, labels)

    def _raw_valid_actions(self) -> Tensor:
        open_tail = self.successor < 0
        open_head = self.predecessor < 0
        different = self.component.unsqueeze(2) != self.component.unsqueeze(1)
        load = self._component_load()
        capacity_feasible = load.unsqueeze(2) + load.unsqueeze(1) <= 1.0 + 1e-6
        return (
            open_tail.unsqueeze(2)
            & open_head.unsqueeze(1)
            & different
            & capacity_feasible
        )

    @property
    def active(self) -> Tensor:
        """仍能合并至少两条容量可行路径的样本。"""
        return self._raw_valid_actions().flatten(1).any(dim=1)

    @property
    def terminal(self) -> bool:
        return not bool(self.active.any())

    def edge_action_mask(self, fixed_tail: Tensor | None = None) -> Tensor:
        """完成的样本使用唯一的 0→0 填充动作，以同步批次 rollout。"""
        valid = self._raw_valid_actions()
        inactive = ~valid.flatten(1).any(dim=1)
        if inactive.any():
            valid = valid.clone()
            valid[inactive, 0, 0] = True

        if fixed_tail is not None:
            self._validate_action_vector(fixed_tail, "fixed_tail")
            batch = torch.arange(self.batch_size, device=self.instance.device)
            selected = torch.zeros(
                self.batch_size, self.n, dtype=torch.bool, device=self.instance.device
            )
            selected[batch, fixed_tail] = True
            valid &= selected.unsqueeze(2)

        if (~valid.flatten(1).any(dim=1)).any():
            raise RuntimeError("CVRP state has no legal or padding action")
        return ~valid

    def tail_mask(self) -> Tensor:
        return self.edge_action_mask().all(dim=-1)

    def head_mask(self, selected_tail: Tensor) -> Tensor:
        self._validate_action_vector(selected_tail, "selected_tail")
        batch = torch.arange(self.batch_size, device=self.instance.device)
        return self.edge_action_mask()[batch, selected_tail]

    def sequential_tail(self, anchor: int = 0) -> Tensor:
        """始终扩展编号最小且仍可继续装载的当前路线。"""
        if self.terminal:
            raise ValueError("a terminal state has no tail")
        if not 0 <= anchor < self.n:
            raise ValueError(f"anchor must be in [0, {self.n})")
        legal_tail = ~self.tail_mask()
        indices = torch.arange(self.n, device=self.instance.device).unsqueeze(0)
        # 用分量标签而不是端点编号确定优先级，确保同一条路线持续延长，
        # 直到没有任何容量可行的后继后才开始下一条路线。
        priority = self.component * self.n + indices
        sentinel = torch.full_like(priority, self.n * self.n + self.n)
        selected_priority = torch.where(legal_tail, priority, sentinel).amin(dim=1)
        return selected_priority.remainder(self.n)

    def update(self, selected_tail: Tensor, selected_head: Tensor) -> BatchedCVRPState:
        self._validate_action_vector(selected_tail, "selected_tail")
        self._validate_action_vector(selected_head, "selected_head")
        batch = torch.arange(self.batch_size, device=self.instance.device)
        mask = self.head_mask(selected_tail)
        if mask[batch, selected_head].any():
            raise ValueError("selected edge violates the CVRP capacity mask")

        active = self.active
        active_batch = batch[active]
        successor = self.successor.clone()
        predecessor = self.predecessor.clone()
        successor[active_batch, selected_tail[active]] = selected_head[active]
        predecessor[active_batch, selected_head[active]] = selected_tail[active]

        left = self.component.gather(1, selected_tail[:, None])
        right = self.component.gather(1, selected_head[:, None])
        merged = torch.minimum(left, right)
        in_merged = (self.component == left) | (self.component == right)
        component = torch.where(active[:, None] & in_merged, merged, self.component)

        return BatchedCVRPState(
            instance=self.instance,
            successor=successor,
            predecessor=predecessor,
            component=component,
            transitions=self.transitions + 1,
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

        start_source = node_embeddings * (self.predecessor < 0).unsqueeze(-1)
        end_source = node_embeddings * (self.successor < 0).unsqueeze(-1)
        start_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_labels, start_source
        )
        end_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_labels, end_source
        )
        path_start = start_by_label.gather(1, embedding_labels)
        path_end = end_by_label.gather(1, embedding_labels)

        load = self._component_load().unsqueeze(-1).to(node_embeddings.dtype)
        return torch.cat((component_mean, path_start, path_end, load), dim=-1)

    def path_state_features(self, node_embeddings: Tensor) -> Tensor:
        features = self.forest_decoder_features(node_embeddings)
        embedding_dim = node_embeddings.size(-1)
        return torch.cat((features[:, :, : 2 * embedding_dim], features[:, :, -1:]), dim=-1)

    def route_cost(self) -> Tensor:
        """计算客户路径内部边与每条路径两端到仓库的总距离。"""
        batch = torch.arange(self.batch_size, device=self.instance.device)[:, None]
        valid_successor = self.successor >= 0
        target = self.successor.clamp_min(0)
        target_coordinates = self.coordinates[batch, target]
        internal = (self.coordinates - target_coordinates).norm(dim=-1)
        internal = (internal * valid_successor.to(internal.dtype)).sum(dim=1)

        start_distance = (self.coordinates - self.depot[:, None, :]).norm(dim=-1)
        end_distance = start_distance
        depot_edges = (
            start_distance * (self.predecessor < 0).to(start_distance.dtype)
            + end_distance * (self.successor < 0).to(end_distance.dtype)
        ).sum(dim=1)
        return internal + depot_edges

    def _validate_action_vector(self, action: Tensor, name: str) -> None:
        if action.shape != (self.batch_size,) or action.dtype != torch.long:
            raise ValueError(f"{name} must be a long tensor with shape (batch,)")
        if ((action < 0) | (action >= self.n)).any():
            raise ValueError(f"{name} contains an out-of-range customer")


class BatchedCVRPConstruction:
    """将 CVRP 表示为容量可行客户路径的逐步合并。"""

    def initial_state(self, instance: Tensor) -> BatchedCVRPState:
        return BatchedCVRPState.initialize(instance)

    def is_terminal(self, state: BatchedCVRPState) -> bool:
        return state.terminal

    def base_mask(self, state: BatchedCVRPState) -> Tensor:
        return state.tail_mask()

    def representative_mask(self, state: BatchedCVRPState, selected_base: Tensor) -> Tensor:
        return state.head_mask(selected_base)

    def action_mask(
        self, state: BatchedCVRPState, fixed_base: Tensor | None = None
    ) -> Tensor:
        return state.edge_action_mask(fixed_base)

    def fixed_base(self, state: BatchedCVRPState, anchor: int = 0) -> Tensor:
        return state.sequential_tail(anchor)

    def transition(
        self,
        state: BatchedCVRPState,
        selected_base: Tensor,
        selected_representative: Tensor,
    ) -> BatchedCVRPState:
        return state.update(selected_base, selected_representative)

    def objective(
        self,
        state: BatchedCVRPState,
        selected_bases: Tensor,
        selected_representatives: Tensor,
    ) -> Tensor:
        del selected_bases, selected_representatives
        if not state.terminal:
            raise ValueError("cannot score an unfinished CVRP state")
        return state.route_cost()

    def solution(self, state: BatchedCVRPState) -> Tensor:
        if not state.terminal:
            raise ValueError("cannot decode an unfinished CVRP state")
        return state.successor
