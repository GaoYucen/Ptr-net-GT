"""用于 Original 与 GroupOpt TSP 构造的 Pointer Network 接入实现。"""

from __future__ import annotations

from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn

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
    "native_conditional_fixed",
    "native_conditional_free",
]
BASE_MODES = ("native_original", "native_conditional_fixed", "native_conditional_free")


PointerNetworkOutput = ConstructionOutput


class _PointerAttention(nn.Module):
    def __init__(self, embedding_dim: int, tanh_clipping: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tanh_clipping = tanh_clipping
        self.project_nodes = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pointer = nn.Parameter(torch.empty(embedding_dim))
        bound = 1.0 / sqrt(embedding_dim)
        nn.init.uniform_(self.pointer, -bound, bound)

    def forward(
        self,
        query: Tensor,
        candidates: Tensor,
        mask: Tensor,
        temperature: float,
    ) -> Tensor:
        logits = self.logits(query, candidates, mask)
        return torch.log_softmax(logits / temperature, dim=-1)

    def logits(self, query: Tensor, candidates: Tensor, mask: Tensor) -> Tensor:
        """返回一个或多个条件 query 对应的原生 pointer logits。"""
        squeeze_query = query.ndim == 2
        if squeeze_query:
            query = query.unsqueeze(1)
            mask = mask.unsqueeze(1)
        if query.ndim != 3 or mask.ndim != 3:
            raise ValueError("query and mask must describe one or more queries")
        if candidates.ndim == 3:
            candidates = candidates.unsqueeze(1)
        if candidates.ndim != 4:
            raise ValueError(
                "candidates must have shape (batch, nodes, dim) or (batch, queries, nodes, dim)"
            )
        compatibility = torch.tanh(
            self.project_nodes(candidates) + self.project_query(query).unsqueeze(2)
        )
        logits = torch.matmul(compatibility, self.pointer)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits.masked_fill(mask, -torch.inf)
        return logits.squeeze(1) if squeeze_query else logits


class PointerNetwork(nn.Module):
    """同时支持 Original 与 GroupOpt 解码模式的 LSTM Pointer Network。"""

    def __init__(
        self,
        embedding_dim: int = 128,
        n_encoder_layers: int = 1,
        tanh_clipping: float = 10.0,
        construction_process: BatchedConstructionProcess[Tensor, BatchedTSPState] | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or n_encoder_layers < 1:
            raise ValueError("embedding_dim and n_encoder_layers must be positive")

        self.embedding_dim = embedding_dim
        self.construction_process = construction_process or BatchedTSPConstruction()
        self.input_projection = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(
            embedding_dim,
            embedding_dim,
            num_layers=n_encoder_layers,
            batch_first=True,
        )
        self.decoder_cell = nn.LSTMCell(embedding_dim, embedding_dim)
        self.project_tail_context = nn.Linear(2 * embedding_dim, embedding_dim)
        self.project_head_context = nn.Linear(3 * embedding_dim, embedding_dim)
        self.project_tail_state = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.project_native_head_summary = nn.Linear(3, embedding_dim, bias=False)
        self.tail_pointer = _PointerAttention(embedding_dim, tanh_clipping)
        self.head_pointer = _PointerAttention(embedding_dim, tanh_clipping)

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "native_conditional_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> PointerNetworkOutput:
        if coordinates.ndim != 3 or coordinates.size(-1) != 2:
            raise ValueError("coordinates must have shape (batch, nodes, 2)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")

        inputs = self.input_projection(coordinates)
        node_embeddings, (encoder_hidden, encoder_cell) = self.encoder(inputs)
        if base_mode == "native_original":
            return self._decode_native_original(
                coordinates,
                node_embeddings,
                encoder_hidden[-1],
                encoder_cell[-1],
                decode_type,
                temperature,
                generator,
                anchor,
            )
        if base_mode == "native_conditional_free":
            return self._decode_native_conditional_free(
                coordinates,
                node_embeddings,
                encoder_hidden[-1],
                encoder_cell[-1],
                decode_type,
                temperature,
                generator,
            )

        # Original：使用锚定的构造顺序，由原生 pointer decoder 选择另一端点。
        decoder_hidden = encoder_hidden[-1]
        decoder_cell = encoder_cell[-1]
        graph_embedding = node_embeddings.mean(dim=1)
        process = self.construction_process
        state = process.initial_state(coordinates)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []

        while not process.is_terminal(state):
            selected_tail = process.fixed_base(state, anchor)

            tail_embedding = _gather_nodes(node_embeddings, selected_tail)
            head_query = self.project_head_context(
                torch.cat((graph_embedding, decoder_hidden, tail_embedding), dim=-1)
            )
            head_log_p = self.head_pointer(
                head_query,
                node_embeddings,
                process.representative_mask(state, selected_tail),
                temperature,
            )
            selected_head = _select(head_log_p, decode_type, generator)
            selected_log_probabilities.append(
                head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )

            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            head_embedding = _gather_nodes(node_embeddings, selected_head)
            decoder_hidden, decoder_cell = self.decoder_cell(
                head_embedding, (decoder_hidden, decoder_cell)
            )

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        zeros = torch.zeros(
            state.batch_size,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        return PointerNetworkOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=zeros,
            action_entropy=zeros,
        )

    def _decode_native_original(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        decoder_hidden: Tensor,
        decoder_cell: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        anchor: int,
    ) -> PointerNetworkOutput:
        """直接执行固定首节点的原生顺序 Pointer Network rollout。"""
        batch_size, node_count, _ = coordinates.shape
        if not 0 <= anchor < node_count:
            raise ValueError("anchor is outside the graph")
        graph_embedding = node_embeddings.mean(dim=1)
        current = torch.full((batch_size,), anchor, dtype=torch.long, device=coordinates.device)
        visited = torch.zeros(batch_size, node_count, dtype=torch.bool, device=coordinates.device)
        visited.scatter_(1, current[:, None], True)
        tour = [current]
        selected_log_probabilities: list[Tensor] = []

        for _ in range(node_count - 1):
            current_embedding = _gather_nodes(node_embeddings, current)
            head_query = self.project_head_context(
                torch.cat((graph_embedding, decoder_hidden, current_embedding), dim=-1)
            )
            head_log_p = self.head_pointer(
                head_query, node_embeddings, visited.clone(), temperature
            )
            selected = _select(head_log_p, decode_type, generator)
            selected_log_probabilities.append(head_log_p.gather(1, selected[:, None]).squeeze(1))
            tour.append(selected)
            visited.scatter_(1, selected[:, None], True)
            selected_embedding = _gather_nodes(node_embeddings, selected)
            decoder_hidden, decoder_cell = self.decoder_cell(
                selected_embedding, (decoder_hidden, decoder_cell)
            )
            current = selected

        tour_tensor = torch.stack(tour, dim=1)
        return _sequential_output(coordinates, tour_tensor, selected_log_probabilities)

    def _decode_native_conditional_free(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        decoder_hidden: Tensor,
        decoder_cell: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
    ) -> PointerNetworkOutput:
        """用 Pointer 风格实现完整的 Forest-aware head/tail 接口。"""
        graph_embedding = node_embeddings.mean(dim=1)
        distances = torch.cdist(coordinates, coordinates)
        process = self.construction_process

        def score_heads(state: BatchedTSPState, pair_mask: Tensor) -> ForestHeadProposal:
            expanded_graph = graph_embedding.unsqueeze(1).expand_as(node_embeddings)
            expanded_hidden = decoder_hidden.unsqueeze(1).expand_as(node_embeddings)
            head_queries = self.project_head_context(
                torch.cat((expanded_graph, expanded_hidden, node_embeddings), dim=-1)
            )
            head_logits = self.head_pointer.logits(head_queries, node_embeddings, pair_mask)
            head_log_p = masked_conditional_log_probabilities(head_logits, pair_mask, temperature)
            summary = native_head_summary(head_log_p, distances).detach()
            return ForestHeadProposal(head_log_p, summary)

        def score_tails(
            state: BatchedTSPState,
            proposal: ForestHeadProposal,
            tail_mask: Tensor,
        ) -> Tensor:
            tail_candidates = (
                node_embeddings
                + self.project_tail_state(state.path_state_features(node_embeddings))
                + self.project_native_head_summary(proposal.features)
            )
            tail_query = self.project_tail_context(
                torch.cat((graph_embedding, decoder_hidden), dim=-1)
            )
            tail_logits = self.tail_pointer.logits(tail_query, tail_candidates, tail_mask)
            return torch.log_softmax(tail_logits / temperature, dim=-1)

        def update_context(selected_tail: Tensor, selected_head: Tensor) -> None:
            del selected_tail
            nonlocal decoder_hidden, decoder_cell
            head_embedding = _gather_nodes(node_embeddings, selected_head)
            decoder_hidden, decoder_cell = self.decoder_cell(
                head_embedding, (decoder_hidden, decoder_cell)
            )

        decoder = CallableForestAwareDecoder[BatchedTSPState](
            head_scorer=score_heads,
            tail_scorer=score_tails,
            context_updater=update_context,
        )
        return decode_forest_edges(
            coordinates,
            process,
            decoder,
            decode_type,
            generator,
        )


def _gather_nodes(node_embeddings: Tensor, selected: Tensor) -> Tensor:
    return node_embeddings.gather(
        1, selected[:, None, None].expand(-1, 1, node_embeddings.size(-1))
    ).squeeze(1)


def _sequential_output(
    coordinates: Tensor,
    tour: Tensor,
    selected_log_probabilities: list[Tensor],
) -> PointerNetworkOutput:
    ordered = coordinates.gather(1, tour.unsqueeze(-1).expand(-1, -1, 2))
    cost = (ordered - ordered.roll(shifts=-1, dims=1)).norm(p=2, dim=-1).sum(dim=1)
    heads = tour.roll(shifts=-1, dims=1)
    successor = torch.empty_like(tour)
    successor.scatter_(1, tour, heads)
    zeros = cost.new_zeros(cost.shape)
    return PointerNetworkOutput(
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
