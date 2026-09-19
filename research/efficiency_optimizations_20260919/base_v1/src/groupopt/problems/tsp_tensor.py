"""与参考 TSP 过程语义一致的向量化 PyTorch 状态。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BatchedTSPState:
    """一批同步推进的部分有向 TSP 构造。

    mask 中的 ``True`` 表示非法，与 attention decoder 的约定一致。
    分量标签用于编码弱连通路径分量。
    """

    coordinates: Tensor
    successor: Tensor
    predecessor: Tensor
    component: Tensor
    edges_added: int

    @classmethod
    def initialize(cls, coordinates: Tensor) -> BatchedTSPState:
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        batch_size, n, _ = coordinates.shape
        if batch_size < 1 or n < 2:
            raise ValueError("a batch and at least two TSP vertices are required")

        device = coordinates.device
        return cls(
            coordinates=coordinates,
            successor=torch.full((batch_size, n), -1, dtype=torch.long, device=device),
            predecessor=torch.full((batch_size, n), -1, dtype=torch.long, device=device),
            component=torch.arange(n, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_size, n)
            .clone(),
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
        return self.edges_added == self.n

    def tail_mask(self) -> Tensor:
        """mask 掉出边已经固定的顶点。"""
        if self.terminal:
            return torch.ones_like(self.successor, dtype=torch.bool)
        return self.successor >= 0

    def head_mask(self, selected_tail: Tensor) -> Tensor:
        """在闭合步骤前 mask 掉已使用头点和同分量头点。"""
        self._validate_action_vector(selected_tail, "selected_tail")
        if self.terminal:
            raise ValueError("a terminal state has no head candidates")

        batch = torch.arange(self.batch_size, device=self.coordinates.device)
        if self.tail_mask()[batch, selected_tail].any():
            raise ValueError("selected_tail contains a masked vertex")

        basic_mask = self.predecessor >= 0
        if self.edges_added == self.n - 1:
            return basic_mask

        tail_component = self.component.gather(1, selected_tail[:, None])
        same_component = self.component == tail_component
        return basic_mask | same_component

    def edge_action_mask(self, fixed_tail: Tensor | None = None) -> Tensor:
        """mask 掉非法的联合 ``(tail, head)`` 构造动作。"""
        if self.terminal:
            return torch.ones(
                self.batch_size,
                self.n,
                self.n,
                dtype=torch.bool,
                device=self.coordinates.device,
            )

        tail_mask = self.tail_mask().unsqueeze(2)
        head_mask = (self.predecessor >= 0).unsqueeze(1)
        mask = tail_mask | head_mask
        if self.edges_added < self.n - 1:
            same_component = self.component.unsqueeze(2) == self.component.unsqueeze(1)
            mask = mask | same_component

        if fixed_tail is not None:
            self._validate_action_vector(fixed_tail, "fixed_tail")
            batch = torch.arange(self.batch_size, device=self.coordinates.device)
            if self.tail_mask()[batch, fixed_tail].any():
                raise ValueError("fixed_tail contains a masked vertex")
            allowed_tail = torch.zeros_like(self.tail_mask())
            allowed_tail[batch, fixed_tail] = True
            mask = mask | ~allowed_tail.unsqueeze(2)

        if mask.flatten(1).all(dim=1).any():
            raise ValueError("each nonterminal state must have a legal edge action")
        return mask

    def sequential_base(self, anchor: int = 0) -> Tensor:
        """返回批次中每个样本锚定分量的唯一尾点。"""
        if self.terminal:
            raise ValueError("a terminal state has no sequential base")
        if not 0 <= anchor < self.n:
            raise ValueError(f"anchor must be in [0, {self.n})")

        anchor_component = self.component[:, anchor : anchor + 1]
        candidates = (self.component == anchor_component) & ~self.tail_mask()
        if not torch.all(candidates.sum(dim=1) == 1):
            raise ValueError("each anchored component must have exactly one tail")
        return candidates.to(torch.long).argmax(dim=1)

    def path_state_features(self, node_embeddings: Tensor) -> Tensor:
        """汇总每个顶点所在开放路径的状态。

        结果拼接分量平均表示、唯一路径起点（无前驱的顶点）的表示，以及归一化路径规模。
        这些是供可学习 base 选择使用、且与模型无关的状态信息。
        """
        forest_features = self.forest_decoder_features(node_embeddings)
        embedding_dim = node_embeddings.size(-1)
        return torch.cat(
            (
                forest_features[:, :, : 2 * embedding_dim],
                forest_features[:, :, -1:],
            ),
            dim=-1,
        )

    def forest_decoder_features(self, node_embeddings: Tensor) -> Tensor:
        """返回供 Forest 条件 decoder 使用的双端路径状态。

        每个节点获得其分量平均表示、路径起点、路径终点和归一化规模。合法 tail
        本身是路径终点，合法 head 本身是路径起点；同时提供另一端点，使 decoder
        在比较候选连接时能看到两侧局部路径，而不只看到静态节点编码。
        """
        if node_embeddings.ndim != 3 or node_embeddings.shape[:2] != (
            self.batch_size,
            self.n,
        ):
            raise ValueError("node_embeddings must have shape (batch, nodes, embedding)")

        component_index = self.component.unsqueeze(-1)
        embedding_index = component_index.expand_as(node_embeddings)
        component_sum_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_index, node_embeddings
        )
        component_sum = component_sum_by_label.gather(1, embedding_index)
        ones = torch.ones_like(component_index, dtype=node_embeddings.dtype)
        component_size_by_label = torch.zeros_like(ones).scatter_add(
            1, component_index, ones
        )
        component_size = component_size_by_label.gather(1, component_index)
        component_mean = component_sum / component_size

        is_path_start = self.predecessor < 0
        start_source = node_embeddings * is_path_start.unsqueeze(-1)
        path_start_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_index, start_source
        )
        path_start = path_start_by_label.gather(1, embedding_index)

        is_path_end = self.successor < 0
        end_source = node_embeddings * is_path_end.unsqueeze(-1)
        path_end_by_label = torch.zeros_like(node_embeddings).scatter_add(
            1, embedding_index, end_source
        )
        path_end = path_end_by_label.gather(1, embedding_index)

        normalized_size = component_size / self.n
        return torch.cat(
            (component_mean, path_start, path_end, normalized_size), dim=-1
        )

    def update(self, selected_tail: Tensor, selected_head: Tensor) -> BatchedTSPState:
        """为批次中的每个样本加入一条合法边，并返回新状态。"""
        self._validate_action_vector(selected_tail, "selected_tail")
        self._validate_action_vector(selected_head, "selected_head")
        batch = torch.arange(self.batch_size, device=self.coordinates.device)

        mask = self.head_mask(selected_tail)
        if mask[batch, selected_head].any():
            raise ValueError("selected edge violates the TSP construction mask")

        successor = self.successor.clone()
        predecessor = self.predecessor.clone()
        successor[batch, selected_tail] = selected_head
        predecessor[batch, selected_head] = selected_tail

        component = self.component
        if self.edges_added < self.n - 1:
            left = component.gather(1, selected_tail[:, None])
            right = component.gather(1, selected_head[:, None])
            merged = torch.minimum(left, right)
            in_merged_component = (component == left) | (component == right)
            component = torch.where(in_merged_component, merged, component)

        return BatchedTSPState(
            coordinates=self.coordinates,
            successor=successor,
            predecessor=predecessor,
            component=component,
            edges_added=self.edges_added + 1,
        )

    def edge_cost(self, tails: Tensor, heads: Tensor) -> Tensor:
        """返回一批边序列的 Euclidean 距离代价。"""
        if tails.shape != heads.shape or tails.ndim != 2:
            raise ValueError("tails and heads must both have shape (batch, steps)")
        if tails.size(0) != self.batch_size:
            raise ValueError("edge batch size does not match the state")

        gather_shape = (*tails.shape, self.coordinates.size(-1))
        tail_coordinates = self.coordinates.gather(
            1, tails.unsqueeze(-1).expand(gather_shape)
        )
        head_coordinates = self.coordinates.gather(
            1, heads.unsqueeze(-1).expand(gather_shape)
        )
        return (tail_coordinates - head_coordinates).norm(p=2, dim=-1).sum(dim=1)

    def _validate_action_vector(self, action: Tensor, name: str) -> None:
        if action.shape != (self.batch_size,) or action.dtype != torch.long:
            raise ValueError(f"{name} must be a long tensor with shape (batch,)")
        if ((action < 0) | (action >= self.n)).any():
            raise ValueError(f"{name} contains an out-of-range vertex")


class BatchedTSPConstruction:
    """供 :class:`BatchedConstructionProcess` 使用的张量化 TSP 问题插件。

    该无状态适配器独立负责向神经模型适配器暴露的 TSP 可行性、状态转移、
    固定 base 行为和目标函数语义。
    """

    def initial_state(self, instance: Tensor) -> BatchedTSPState:
        return BatchedTSPState.initialize(instance)

    def is_terminal(self, state: BatchedTSPState) -> bool:
        return state.terminal

    def base_mask(self, state: BatchedTSPState) -> Tensor:
        return state.tail_mask()

    def representative_mask(
        self, state: BatchedTSPState, selected_base: Tensor
    ) -> Tensor:
        return state.head_mask(selected_base)

    def action_mask(
        self, state: BatchedTSPState, fixed_base: Tensor | None = None
    ) -> Tensor:
        return state.edge_action_mask(fixed_base)

    def fixed_base(self, state: BatchedTSPState, anchor: int = 0) -> Tensor:
        return state.sequential_base(anchor)

    def transition(
        self,
        state: BatchedTSPState,
        selected_base: Tensor,
        selected_representative: Tensor,
    ) -> BatchedTSPState:
        return state.update(selected_base, selected_representative)

    def objective(
        self,
        state: BatchedTSPState,
        selected_bases: Tensor,
        selected_representatives: Tensor,
    ) -> Tensor:
        return state.edge_cost(selected_bases, selected_representatives)

    def solution(self, state: BatchedTSPState) -> Tensor:
        if not state.terminal:
            raise ValueError("cannot decode a nonterminal batched TSP state")
        return state.successor
