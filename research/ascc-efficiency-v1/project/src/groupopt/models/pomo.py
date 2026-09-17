"""POMO 原生解码与 GroupOpt Forest-aware 接入。

编码器和原生 decoder 的参数布局对应作者官方 MIT 实现：
https://github.com/yd-kwon/POMO/tree/d7c3d6ea580499a53e874fe9e065f69e799a8551

Copyright (c) 2021 Yeong-Dae Kwon. 官方结构依 MIT License 使用。
GroupOpt 接口、Forest rollout 与中文说明为本项目新增实现。
"""

from __future__ import annotations

from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from groupopt.framework.forest_decoder import (
    CallableForestAwareDecoder,
    ForestHeadProposal,
    decode_forest_edges,
)
from groupopt.framework.neural import BatchedConstructionProcess, ConstructionOutput
from groupopt.models.native_conditional import (
    masked_conditional_log_probabilities,
    native_head_summary,
)
from groupopt.problems.tsp_tensor import BatchedTSPConstruction, BatchedTSPState

DecodeType = Literal["greedy", "sampling"]
BaseMode = Literal[
    "native_original",
    "native_conditional_free",
    "native_capacity_single_chain",
    "native_forest_fixed",
    "native_free_no_head_summary",
    "native_free_no_path_state",
    "native_free_no_last_head",
    "native_random_tail",
]
BASE_MODES = (
    "native_original",
    "native_conditional_free",
    "native_capacity_single_chain",
    "native_forest_fixed",
    "native_free_no_head_summary",
    "native_free_no_path_state",
    "native_free_no_last_head",
    "native_random_tail",
)


class POMOModel(nn.Module):
    """保留多起点训练机制的 POMO，并提供完整 GroupOpt 解码模式。"""

    def __init__(
        self,
        embedding_dim: int = 128,
        head_num: int = 8,
        qkv_dim: int = 16,
        encoder_layers: int = 6,
        feed_forward_dim: int = 512,
        logit_clipping: float = 10.0,
        pomo_size: int = 8,
        construction_process: BatchedConstructionProcess[Tensor, BatchedTSPState] | None = None,
    ) -> None:
        super().__init__()
        if min(embedding_dim, head_num, qkv_dim, encoder_layers, feed_forward_dim) < 1:
            raise ValueError("POMO dimensions must be positive")
        if pomo_size < 1:
            raise ValueError("pomo_size must be positive")
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.pomo_size = pomo_size
        self.evaluation_batch_size = max(1, 64 // pomo_size)
        self.encoder = _POMOEncoder(
            embedding_dim,
            head_num,
            qkv_dim,
            encoder_layers,
            feed_forward_dim,
        )
        self.decoder = _POMODecoder(
            embedding_dim,
            head_num,
            qkv_dim,
            logit_clipping,
        )
        self.project_tail_state = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.project_head_summary = nn.Linear(3, embedding_dim, bias=False)
        self.tail_selector = _POMOTailSelector(
            embedding_dim,
            head_num,
            qkv_dim,
            logit_clipping,
        )
        self.construction_process = construction_process or BatchedTSPConstruction()

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "native_conditional_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> ConstructionOutput:
        """返回展平的 ``batch × pomo_size`` rollout，供 POMO 组内 baseline 使用。"""
        del anchor
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown POMO base mode: {base_mode}")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        batch_size, node_count, _ = coordinates.shape
        if self.pomo_size > node_count:
            raise ValueError("pomo_size cannot exceed the number of nodes")

        node_embeddings = self.encoder(coordinates)
        repeated_coordinates = (
            coordinates[:, None]
            .expand(batch_size, self.pomo_size, node_count, 2)
            .reshape(batch_size * self.pomo_size, node_count, 2)
        )
        repeated_embeddings = (
            node_embeddings[:, None]
            .expand(
                batch_size,
                self.pomo_size,
                node_count,
                self.embedding_dim,
            )
            .reshape(
                batch_size * self.pomo_size,
                node_count,
                self.embedding_dim,
            )
        )
        anchors = (
            torch.arange(self.pomo_size, device=coordinates.device)
            .unsqueeze(0)
            .expand(batch_size, self.pomo_size)
            .reshape(-1)
        )
        if base_mode == "native_original":
            return self._decode_native_original(
                repeated_coordinates,
                repeated_embeddings,
                anchors,
                decode_type,
                temperature,
                generator,
            )
        if base_mode == "native_capacity_single_chain":
            return self._decode_capacity_single_chain(
                repeated_coordinates,
                repeated_embeddings,
                anchors,
                decode_type,
                temperature,
                generator,
            )
        return self._decode_groupopt(
            repeated_coordinates,
            repeated_embeddings,
            anchors,
            decode_type,
            temperature,
            generator,
            fixed_tail=base_mode == "native_forest_fixed",
            use_head_summary=base_mode != "native_free_no_head_summary",
            use_path_state=base_mode != "native_free_no_path_state",
            use_last_head=base_mode != "native_free_no_last_head",
            random_tail=base_mode == "native_random_tail",
        )

    def best_rollout_cost(self, output: ConstructionOutput) -> Tensor:
        """按原始实例返回 POMO 多起点中的最短 tour。"""
        if output.cost.numel() % self.pomo_size != 0:
            raise ValueError("rollout count is not divisible by pomo_size")
        return output.cost.reshape(-1, self.pomo_size).amin(dim=1)

    def pomo_policy_loss(self, output: ConstructionOutput) -> Tensor:
        """使用 POMO 原生的同实例多起点平均 reward 作为 baseline。"""
        costs = output.cost.reshape(-1, self.pomo_size)
        log_likelihood = output.log_likelihood.reshape(-1, self.pomo_size)
        advantage = (costs - costs.mean(dim=1, keepdim=True)).detach()
        return (advantage * log_likelihood).mean()

    def _decode_native_original(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        anchors: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> ConstructionOutput:
        batch_size, node_count, _ = coordinates.shape
        first_embedding = _gather_nodes(node_embeddings, anchors)
        current = anchors
        visited = torch.zeros(batch_size, node_count, dtype=torch.bool, device=coordinates.device)
        visited.scatter_(1, current[:, None], True)
        tour = [current]
        selected_log_probabilities: list[Tensor] = []

        for _ in range(node_count - 1):
            current_embedding = _gather_nodes(node_embeddings, current)
            log_p = self.decoder.log_probabilities(
                node_embeddings,
                first_embedding,
                current_embedding[:, None],
                visited[:, None].clone(),
                temperature,
            ).squeeze(1)
            selected = _select(log_p, decode_type, generator)
            selected_log_probabilities.append(log_p.gather(1, selected[:, None]).squeeze(1))
            tour.append(selected)
            visited.scatter_(1, selected[:, None], True)
            current = selected

        tour_tensor = torch.stack(tour, dim=1)
        return _sequential_output(coordinates, tour_tensor, selected_log_probabilities)

    def _decode_capacity_single_chain(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        anchors: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> ConstructionOutput:
        """激活与 Free 相同参数，但仍严格沿 POMO 原生单链构造。"""
        state = self.construction_process.initial_state(coordinates)
        batch_size, node_count, _ = coordinates.shape
        batch_index = torch.arange(batch_size, device=coordinates.device)
        first_embedding = _gather_nodes(node_embeddings, anchors)
        current = anchors
        tour = [current]
        selected_log_probabilities: list[Tensor] = []
        distances = torch.cdist(coordinates, coordinates)

        for _ in range(node_count - 1):
            pair_mask = self.construction_process.action_mask(state)
            all_head_log_p = self.decoder.log_probabilities(
                node_embeddings,
                first_embedding,
                node_embeddings,
                pair_mask,
                temperature,
            )
            summary = native_head_summary(all_head_log_p, distances).detach()
            capacity_candidates = (
                node_embeddings
                + self.project_tail_state(state.path_state_features(node_embeddings))
                + self.project_head_summary(summary)
            )
            current_embedding = _gather_nodes(node_embeddings, current)
            head_mask = self.construction_process.representative_mask(state, current)
            capacity_log_p = self.tail_selector.log_probabilities(
                first_embedding,
                current_embedding,
                capacity_candidates,
                head_mask,
                temperature,
            )
            native_log_p = all_head_log_p[batch_index, current]
            combined_log_p = torch.log_softmax((native_log_p + capacity_log_p) / sqrt(2.0), dim=-1)
            selected = _select(combined_log_p, decode_type, generator)
            selected_log_probabilities.append(
                combined_log_p.gather(1, selected[:, None]).squeeze(1)
            )
            tour.append(selected)
            state = self.construction_process.transition(state, current, selected)
            current = selected

        return _sequential_output(
            coordinates,
            torch.stack(tour, dim=1),
            selected_log_probabilities,
        )

    def _decode_groupopt(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        anchors: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        *,
        fixed_tail: bool = False,
        use_head_summary: bool = True,
        use_path_state: bool = True,
        use_last_head: bool = True,
        random_tail: bool = False,
    ) -> ConstructionOutput:
        first_embedding = _gather_nodes(node_embeddings, anchors)
        last_head_embedding = first_embedding
        distances = torch.cdist(coordinates, coordinates)

        def score_heads(state: BatchedTSPState, pair_mask: Tensor) -> ForestHeadProposal:
            head_log_p = self.decoder.log_probabilities(
                node_embeddings,
                first_embedding,
                node_embeddings,
                pair_mask,
                temperature,
            )
            summary = native_head_summary(head_log_p, distances).detach()
            return ForestHeadProposal(head_log_p, summary)

        def score_tails(
            state: BatchedTSPState,
            proposal: ForestHeadProposal,
            tail_mask: Tensor,
        ) -> Tensor:
            if state.edges_added == 0:
                forced = node_embeddings.new_full(tail_mask.shape, -torch.inf)
                forced.scatter_(1, anchors[:, None], 0.0)
                return forced
            if random_tail:
                return _random_forest_tail_log_probabilities(
                    tail_mask, node_embeddings.dtype, generator
                )
            if fixed_tail:
                return _deterministic_forest_tail_log_probabilities(
                    state, tail_mask, node_embeddings.dtype
                )
            tail_candidates = node_embeddings
            if use_path_state:
                tail_candidates = tail_candidates + self.project_tail_state(
                    state.path_state_features(node_embeddings)
                )
            if use_head_summary:
                tail_candidates = tail_candidates + self.project_head_summary(proposal.features)
            tail_context = (
                last_head_embedding if use_last_head else torch.zeros_like(last_head_embedding)
            )
            return self.tail_selector.log_probabilities(
                first_embedding,
                tail_context,
                tail_candidates,
                tail_mask,
                temperature,
            )

        def update_context(selected_tail: Tensor, selected_head: Tensor) -> None:
            del selected_tail
            nonlocal last_head_embedding
            last_head_embedding = _gather_nodes(node_embeddings, selected_head)

        decoder = CallableForestAwareDecoder[BatchedTSPState](
            head_scorer=score_heads,
            tail_scorer=score_tails,
            context_updater=update_context,
        )
        return decode_forest_edges(
            coordinates,
            self.construction_process,
            decoder,
            decode_type,
            generator,
        )


class _POMOEncoder(nn.Module):
    """保持官方参数名称的 POMO Transformer encoder。"""

    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        layer_num: int,
        feed_forward_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList(
            _POMOEncoderLayer(
                embedding_dim,
                head_num,
                qkv_dim,
                feed_forward_dim,
            )
            for _ in range(layer_num)
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        values = self.embedding(coordinates)
        for layer in self.layers:
            values = layer(values)
        return values


class _POMOEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        feed_forward_dim: int,
    ) -> None:
        super().__init__()
        self.head_num = head_num
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.addAndNormalization1 = _POMOAddAndNormalization(embedding_dim)
        self.feedForward = _POMOFeedForward(embedding_dim, feed_forward_dim)
        self.addAndNormalization2 = _POMOAddAndNormalization(embedding_dim)

    def forward(self, values: Tensor) -> Tensor:
        query = _reshape_by_heads(self.Wq(values), self.head_num)
        key = _reshape_by_heads(self.Wk(values), self.head_num)
        value = _reshape_by_heads(self.Wv(values), self.head_num)
        attended = _multi_head_attention(query, key, value)
        attended = self.multi_head_combine(attended)
        residual = self.addAndNormalization1(values, attended)
        return self.addAndNormalization2(residual, self.feedForward(residual))


class _POMOAddAndNormalization(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.norm((left + right).transpose(1, 2)).transpose(1, 2)


class _POMOFeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(embedding_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, values: Tensor) -> Tensor:
        return self.W2(functional.relu(self.W1(values)))


class _POMODecoder(nn.Module):
    """保持官方 decoder 参数名，同时支持一次计算所有 tail。"""

    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        logit_clipping: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.logit_clipping = logit_clipping
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def log_probabilities(
        self,
        node_embeddings: Tensor,
        first_embedding: Tensor,
        tail_embeddings: Tensor,
        mask: Tensor,
        temperature: float,
    ) -> Tensor:
        query_first = _reshape_by_heads(self.Wq_first(first_embedding[:, None]), self.head_num)
        query_tail = _reshape_by_heads(self.Wq_last(tail_embeddings), self.head_num)
        query = query_first + query_tail
        key = _reshape_by_heads(self.Wk(node_embeddings), self.head_num)
        value = _reshape_by_heads(self.Wv(node_embeddings), self.head_num)
        glimpse = _multi_head_attention(query, key, value, mask)
        glimpse = self.multi_head_combine(glimpse)
        logits = torch.matmul(glimpse, node_embeddings.transpose(1, 2))
        logits = self.logit_clipping * torch.tanh(logits / sqrt(self.embedding_dim))
        return masked_conditional_log_probabilities(logits, mask, temperature)


class _POMOTailSelector(nn.Module):
    """沿用 POMO 的 first/last query 与多头 glimpse 风格选择 tail。"""

    def __init__(
        self,
        embedding_dim: int,
        head_num: int,
        qkv_dim: int,
        logit_clipping: float,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.logit_clipping = logit_clipping
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def log_probabilities(
        self,
        first_embedding: Tensor,
        last_head_embedding: Tensor,
        candidates: Tensor,
        mask: Tensor,
        temperature: float,
    ) -> Tensor:
        query_first = _reshape_by_heads(self.Wq_first(first_embedding[:, None]), self.head_num)
        query_last = _reshape_by_heads(self.Wq_last(last_head_embedding[:, None]), self.head_num)
        key = _reshape_by_heads(self.Wk(candidates), self.head_num)
        value = _reshape_by_heads(self.Wv(candidates), self.head_num)
        glimpse = _multi_head_attention(
            query_first + query_last,
            key,
            value,
            mask[:, None],
        )
        glimpse = self.multi_head_combine(glimpse).squeeze(1)
        logits = torch.matmul(glimpse[:, None], candidates.transpose(1, 2)).squeeze(1)
        logits = self.logit_clipping * torch.tanh(logits / sqrt(self.embedding_dim))
        logits = logits.masked_fill(mask, -torch.inf)
        return torch.log_softmax(logits / temperature, dim=-1)


def _reshape_by_heads(values: Tensor, head_num: int) -> Tensor:
    batch_size, item_count, _ = values.shape
    return values.reshape(batch_size, item_count, head_num, -1).transpose(1, 2)


def _multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    scores = torch.matmul(query, key.transpose(2, 3)) / sqrt(query.size(-1))
    if mask is not None:
        fully_masked = mask.all(dim=-1, keepdim=True)
        safe_mask = mask & ~fully_masked
        scores = scores.masked_fill(safe_mask[:, None], -torch.inf)
    weights = torch.softmax(scores, dim=-1)
    attended = torch.matmul(weights, value)
    return attended.transpose(1, 2).reshape(
        query.size(0), query.size(2), query.size(1) * query.size(3)
    )


def _gather_nodes(node_embeddings: Tensor, selected: Tensor) -> Tensor:
    return node_embeddings.gather(
        1,
        selected[:, None, None].expand(node_embeddings.size(0), 1, node_embeddings.size(-1)),
    ).squeeze(1)


def _deterministic_forest_tail_log_probabilities(
    state: BatchedTSPState, tail_mask: Tensor, dtype: torch.dtype
) -> Tensor:
    """优先调度最短分量的开放端点，形成可复现的多路径 Forest。"""
    node_count = tail_mask.size(1)
    indices = torch.arange(node_count, device=tail_mask.device).expand_as(tail_mask)
    component_size_by_label = torch.zeros_like(state.component).scatter_add(
        1, state.component, torch.ones_like(state.component)
    )
    component_size = component_size_by_label.gather(1, state.component)
    priority = component_size * (node_count + 1) + indices
    invalid_priority = (node_count + 1) ** 2
    selected = priority.masked_fill(tail_mask, invalid_priority).argmin(dim=1)
    if tail_mask.gather(1, selected[:, None]).any():
        raise ValueError("a nonterminal Forest state has no legal tail")
    log_p = torch.full(tail_mask.shape, -torch.inf, dtype=dtype, device=tail_mask.device)
    log_p.scatter_(1, selected[:, None], 0.0)
    return log_p


def _random_forest_tail_log_probabilities(
    tail_mask: Tensor,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    """从合法开放端点中均匀抽取一个、且使实验随机种子可复现。"""
    legal_weights = (~tail_mask).to(dtype)
    if (legal_weights.sum(dim=1) == 0).any():
        raise ValueError("a nonterminal Forest state has no legal tail")
    selected = torch.multinomial(legal_weights, 1, generator=generator)
    log_p = torch.full(tail_mask.shape, -torch.inf, dtype=dtype, device=tail_mask.device)
    log_p.scatter_(1, selected, 0.0)
    return log_p


def _sequential_output(
    coordinates: Tensor,
    tour: Tensor,
    selected_log_probabilities: list[Tensor],
) -> ConstructionOutput:
    ordered = coordinates.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    cost = (ordered - ordered.roll(shifts=-1, dims=1)).norm(p=2, dim=-1).sum(dim=1)
    heads = tour.roll(shifts=-1, dims=1)
    successor = torch.empty_like(tour)
    successor.scatter_(1, tour, heads)
    zeros = cost.new_zeros(cost.shape)
    return ConstructionOutput(
        cost=cost,
        log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
        tails=tour,
        heads=heads,
        successor=successor,
        tail_entropy=zeros,
        action_entropy=zeros,
    )


def _select(
    log_probabilities: Tensor,
    decode_type: DecodeType,
    generator: torch.Generator | None,
) -> Tensor:
    if decode_type == "greedy":
        return log_probabilities.argmax(dim=1)
    if decode_type == "sampling":
        return torch.multinomial(
            log_probabilities.exp(), num_samples=1, generator=generator
        ).squeeze(1)
    raise ValueError(f"unknown decode type: {decode_type}")
