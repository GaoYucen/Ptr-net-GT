"""用于 Original 与 GroupOpt TSP 构造的 Attention Model 接入实现。

编码器遵循 Kool 等人的图 self-attention 结构。decoder 在每个构造步骤执行两次
attention 决策：先选择 tail（base），再选择合法 head（代表元）。可行性完全由
``BatchedTSPState`` 提供。
"""

from __future__ import annotations

from math import sqrt
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from groupopt.framework.forest_decoder import (
    CallableForestAwareDecoder,
    ForestHeadProposal,
    decode_forest_edges,
)
from groupopt.framework.neural import BatchedConstructionProcess, ConstructionOutput
from groupopt.models.native_conditional import (
    categorical_entropy,
    masked_conditional_log_probabilities,
    native_head_summary,
)
from groupopt.problems.tsp_tensor import BatchedTSPConstruction, BatchedTSPState

DecodeType = Literal["greedy", "sampling"]
BaseMode = Literal[
    "native_original",
    "native_conditional_fixed",
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
    "native_conditional_fixed",
    "native_conditional_free",
    "native_capacity_single_chain",
    "native_forest_fixed",
    "native_free_no_head_summary",
    "native_free_no_path_state",
    "native_free_no_last_head",
    "native_random_tail",
)


AttentionModelOutput = ConstructionOutput


class _Normalization(nn.Module):
    def __init__(self, embedding_dim: int, kind: Literal["batch", "layer"]) -> None:
        super().__init__()
        if kind == "batch":
            self.normalizer: nn.Module = nn.BatchNorm1d(embedding_dim)
        elif kind == "layer":
            self.normalizer = nn.LayerNorm(embedding_dim)
        else:
            raise ValueError(f"unknown normalization: {kind}")

    def forward(self, values: Tensor) -> Tensor:
        if isinstance(self.normalizer, nn.BatchNorm1d):
            batch, nodes, embedding = values.shape
            return self.normalizer(values.reshape(batch * nodes, embedding)).reshape(
                batch, nodes, embedding
            )
        return self.normalizer(values)


class _EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        feed_forward_dim: int,
        normalization: Literal["batch", "layer"],
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True, bias=True)
        self.attention_norm = _Normalization(embedding_dim, normalization)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_dim),
        )
        self.feed_forward_norm = _Normalization(embedding_dim, normalization)

    def forward(self, values: Tensor) -> Tensor:
        attended, _ = self.attention(values, values, values, need_weights=False)
        values = self.attention_norm(values + attended)
        return self.feed_forward_norm(values + self.feed_forward(values))


class _GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        feed_forward_dim: int,
        normalization: Literal["batch", "layer"],
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.layers = nn.ModuleList(
            _EncoderLayer(
                embedding_dim,
                n_heads,
                feed_forward_dim,
                normalization,
            )
            for _ in range(n_layers)
        )

    def forward(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.input_projection(coordinates)
        for layer in self.layers:
            nodes = layer(nodes)
        return nodes, nodes.mean(dim=1)


class AttentionModel(nn.Module):
    """同时支持 Original 与 GroupOpt 解码模式的 AM 风格编码器。"""

    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 3,
        feed_forward_dim: int = 512,
        tanh_clipping: float = 10.0,
        normalization: Literal["batch", "layer"] = "batch",
        construction_process: BatchedConstructionProcess[Tensor, BatchedTSPState] | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.construction_process = construction_process or BatchedTSPConstruction()
        self.encoder = _GraphAttentionEncoder(
            input_dim,
            embedding_dim,
            n_heads,
            n_encoder_layers,
            feed_forward_dim,
            normalization,
        )

        self.project_graph = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_nodes = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_tail_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_head_context = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_tail_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_head_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_tail_state = nn.Linear(2 * embedding_dim + 1, embedding_dim, bias=False)
        self.project_state_tail_nodes = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_native_head_summary = nn.Linear(3, embedding_dim, bias=False)
        self.first_step_context = nn.Parameter(torch.empty(embedding_dim))
        nn.init.uniform_(self.first_step_context, -1.0, 1.0)

    def forward(
        self,
        coordinates: Tensor,
        decode_type: DecodeType = "sampling",
        base_mode: BaseMode = "native_conditional_free",
        anchor: int = 0,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> AttentionModelOutput:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if base_mode not in BASE_MODES:
            raise ValueError(f"unknown base mode: {base_mode}")

        node_embeddings, graph_embedding = self.encoder(coordinates)
        if base_mode in {
            "native_conditional_free",
            "native_forest_fixed",
            "native_free_no_head_summary",
            "native_free_no_path_state",
            "native_free_no_last_head",
            "native_random_tail",
        }:
            return self._decode_native_conditional_free(
                coordinates,
                node_embeddings,
                graph_embedding,
                decode_type,
                temperature,
                generator,
                fixed_tail=base_mode == "native_forest_fixed",
                use_head_summary=base_mode != "native_free_no_head_summary",
                use_path_state=base_mode != "native_free_no_path_state",
                use_last_head=base_mode != "native_free_no_last_head",
                random_tail=base_mode == "native_random_tail",
            )
        if base_mode == "native_capacity_single_chain":
            return self._decode_native_capacity_single_chain(
                coordinates,
                node_embeddings,
                graph_embedding,
                decode_type,
                temperature,
                generator,
                anchor,
            )

        # native_original 与历史 fixed 名称共用同一条单链实现；后者仅为旧 checkpoint 兼容。
        process = self.construction_process
        state = process.initial_state(coordinates)
        graph_context = self.project_graph(graph_embedding)
        key, value, logit_key = self.project_nodes(node_embeddings).chunk(3, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []

        while not process.is_terminal(state):
            selected_tail = process.fixed_base(state, anchor)

            tail_embedding = node_embeddings.gather(
                1,
                selected_tail[:, None, None].expand(state.batch_size, 1, self.embedding_dim),
            ).squeeze(1)
            head_query = self.project_head_context(
                torch.cat((graph_context, tail_embedding), dim=-1)
            )
            head_log_p = self._attention_log_probabilities(
                head_query,
                key,
                value,
                logit_key,
                process.representative_mask(state, selected_tail),
                self.project_head_glimpse,
                temperature,
            )
            selected_head = _select(head_log_p, decode_type, generator)

            selected_log_probabilities.append(
                head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )
            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        log_likelihood = torch.stack(selected_log_probabilities, dim=1).sum(dim=1)
        zeros = torch.zeros(
            state.batch_size,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        output = AttentionModelOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=log_likelihood,
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=zeros,
            action_entropy=zeros,
        )
        return output

    def _decode_native_conditional_free(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        graph_embedding: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        *,
        fixed_tail: bool = False,
        use_head_summary: bool = True,
        use_path_state: bool = True,
        use_last_head: bool = True,
        random_tail: bool = False,
    ) -> AttentionModelOutput:
        """Memory-efficient Forest-aware decoding with unchanged policy semantics.

        The all-tail head distributions are only used through stop-gradient summaries
        by the tail policy. During training we therefore compute those proposals without
        autograd, select a tail, and re-score only the selected tail with gradients for
        the head-policy REINFORCE term. During no-grad inference the already computed
        selected row is reused. Static head queries are projected once per instance.
        """
        process = self.construction_process
        state = process.initial_state(coordinates)
        graph_context = self.project_graph(graph_embedding)
        key, value, logit_key = self.project_nodes(node_embeddings).chunk(3, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)
        distances = torch.cdist(coordinates[..., :2], coordinates[..., :2])
        expanded_graph = graph_context.unsqueeze(1).expand(
            state.batch_size, state.n, self.embedding_dim
        )
        head_queries = self.project_head_context(
            torch.cat((expanded_graph, node_embeddings), dim=-1)
        )

        # The proposal head queries/keys are instance-static. Cache their first
        # attention compatibility once, then only re-apply the changing forest mask.
        # This keeps the dense N x N layout (and therefore the canonical numerical
        # ordering) while avoiding the repeated QK matmul at every construction step.
        with torch.no_grad():
            proposal_head_dim = self.embedding_dim // self.n_heads
            proposal_head_query = head_queries.detach().reshape(
                state.batch_size, state.n, self.n_heads, proposal_head_dim
            ).permute(0, 2, 1, 3)
            proposal_compatibility = torch.matmul(
                proposal_head_query, key.detach().transpose(-2, -1)
            ) / sqrt(proposal_head_dim)
            proposal_value = value.detach()
            proposal_logit_key_t = logit_key.detach().transpose(-2, -1)

        def proposal_logits(pair_mask: Tensor) -> Tensor:
            fully_masked = pair_mask.all(dim=-1, keepdim=True)
            safe_mask = pair_mask & ~fully_masked
            compatibility = proposal_compatibility.masked_fill(
                safe_mask[:, None, :, :], -torch.inf
            )
            attention = torch.softmax(compatibility, dim=-1)
            glimpse = torch.matmul(attention, proposal_value)
            glimpse = glimpse.transpose(1, 2).reshape(
                state.batch_size, state.n, self.embedding_dim
            )
            glimpse = self.project_head_glimpse(glimpse)
            logits = torch.matmul(glimpse, proposal_logit_key_t)
            logits = logits / sqrt(self.embedding_dim)
            if self.tanh_clipping > 0:
                logits = torch.tanh(logits) * self.tanh_clipping
            return logits.masked_fill(pair_mask, -torch.inf)

        last_head_embedding = self.first_step_context.unsqueeze(0).expand(
            state.batch_size, self.embedding_dim
        )
        batch_index = torch.arange(state.batch_size, device=coordinates.device)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        tail_entropies: list[Tensor] = []
        action_entropies: list[Tensor] = []

        while not process.is_terminal(state):
            pair_mask = process.action_mask(state)
            tail_mask = process.base_mask(state)

            # The tail policy consumes head-distribution summaries through an explicit
            # stop-gradient channel. Avoid retaining the O(B*N*N) proposal graph.
            with torch.no_grad():
                all_head_logits = proposal_logits(pair_mask)
                all_head_log_p = masked_conditional_log_probabilities(
                    all_head_logits, pair_mask, temperature
                )
                head_summary = native_head_summary(all_head_log_p, distances)

            if random_tail:
                tail_log_p = _random_forest_tail_log_probabilities(
                    tail_mask, node_embeddings.dtype, generator
                )
            elif fixed_tail:
                tail_log_p = _deterministic_forest_tail_log_probabilities(
                    state, tail_mask, node_embeddings.dtype
                )
            else:
                # The differentiable forest-state/tail-policy branch is repeated for
                # every construction step and is the dominant retained-activation cost.
                # Checkpoint the entire branch: forward values/policy are unchanged,
                # while backward recomputes the branch instead of retaining its large
                # scatter/gather/intermediate tensors across all N steps.
                state_for_tail = state
                tail_mask_for_tail = tail_mask

                def tail_policy(
                    node_embeddings_in: Tensor,
                    graph_context_in: Tensor,
                    last_head_in: Tensor,
                    head_summary_in: Tensor,
                ) -> Tensor:
                    state_tail_nodes = node_embeddings_in
                    if use_path_state:
                        state_tail_nodes = state_tail_nodes + self.project_tail_state(
                            state_for_tail.path_state_features(node_embeddings_in)
                        )
                    if use_head_summary:
                        state_tail_nodes = state_tail_nodes + self.project_native_head_summary(
                            head_summary_in
                        )
                    tail_key, tail_value, tail_logit_key = self.project_state_tail_nodes(
                        state_tail_nodes
                    ).chunk(3, dim=-1)
                    tail_context = (
                        last_head_in if use_last_head else torch.zeros_like(last_head_in)
                    )
                    tail_query = self.project_tail_context(
                        torch.cat((graph_context_in, tail_context), dim=-1)
                    )
                    tail_logits = self._attention_logits(
                        tail_query,
                        self._split_heads(tail_key),
                        self._split_heads(tail_value),
                        tail_logit_key,
                        tail_mask_for_tail,
                        self.project_tail_glimpse,
                    )
                    return torch.log_softmax(tail_logits / temperature, dim=-1)

                if torch.is_grad_enabled():
                    tail_log_p = checkpoint(
                        tail_policy,
                        node_embeddings,
                        graph_context,
                        last_head_embedding,
                        head_summary,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    tail_log_p = tail_policy(
                        node_embeddings, graph_context, last_head_embedding, head_summary
                    )

            selected_tail = _select(tail_log_p, decode_type, generator)

            if torch.is_grad_enabled():
                selected_query = head_queries.gather(
                    1,
                    selected_tail[:, None, None].expand(
                        state.batch_size, 1, self.embedding_dim
                    ),
                ).squeeze(1)
                selected_mask = process.representative_mask(state, selected_tail)
                selected_head_logits = self._attention_logits(
                    selected_query,
                    key,
                    value,
                    logit_key,
                    selected_mask,
                    self.project_head_glimpse,
                )
                selected_head_log_p = torch.log_softmax(
                    selected_head_logits / temperature, dim=-1
                )
            else:
                selected_head_log_p = all_head_log_p[batch_index, selected_tail]

            selected_head = _select(selected_head_log_p, decode_type, generator)
            if pair_mask[batch_index, selected_tail, selected_head].any():
                raise RuntimeError("Forest-aware decoder selected an illegal edge")

            selected_log_probabilities.append(
                tail_log_p.gather(1, selected_tail[:, None]).squeeze(1)
                + selected_head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )
            with torch.no_grad():
                tail_entropy = categorical_entropy(tail_log_p.detach())
                tail_probabilities = torch.where(
                    torch.isfinite(tail_log_p.detach()),
                    tail_log_p.detach().exp(),
                    torch.zeros_like(tail_log_p.detach()),
                )
                head_entropy = categorical_entropy(all_head_log_p)
                action_entropy = tail_entropy + (tail_probabilities * head_entropy).sum(dim=-1)
                tail_entropies.append(tail_entropy)
                action_entropies.append(action_entropy)

            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings.gather(
                1,
                selected_head[:, None, None].expand(
                    state.batch_size, 1, self.embedding_dim
                ),
            ).squeeze(1)

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        return AttentionModelOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=torch.stack(tail_entropies, dim=1).mean(dim=1),
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
        )

    def _decode_native_capacity_single_chain(
        self,
        coordinates: Tensor,
        node_embeddings: Tensor,
        graph_embedding: Tensor,
        decode_type: DecodeType,
        temperature: float,
        generator: torch.Generator | None,
        anchor: int,
    ) -> AttentionModelOutput:
        """激活与 Free 相同的参数，但强制沿一条链顺序构造。

        这组对照把 Tail Selector 的完整 AM 风格注意力改作第二个 head
        评分通道，因此所有可学习参数都参与反向传播；tail 仍严格固定为
        上一步的 head。Free 若优于本组，便不能仅用新增参数或额外状态解释。
        """
        process = self.construction_process
        state = process.initial_state(coordinates)
        graph_context = self.project_graph(graph_embedding)
        key, value, logit_key = self.project_nodes(node_embeddings).chunk(3, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)
        distances = torch.cdist(coordinates[..., :2], coordinates[..., :2])
        last_head_embedding = self.first_step_context.unsqueeze(0).expand(
            state.batch_size, self.embedding_dim
        )
        batch_index = torch.arange(state.batch_size, device=coordinates.device)

        tails: list[Tensor] = []
        heads: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        action_entropies: list[Tensor] = []

        while not process.is_terminal(state):
            selected_tail = process.fixed_base(state, anchor)
            pair_mask = process.action_mask(state)
            expanded_graph = graph_context.unsqueeze(1).expand(
                state.batch_size, state.n, self.embedding_dim
            )
            all_head_queries = self.project_head_context(
                torch.cat((expanded_graph, node_embeddings), dim=-1)
            )
            all_head_logits = self._attention_logits(
                all_head_queries,
                key,
                value,
                logit_key,
                pair_mask,
                self.project_head_glimpse,
            )
            all_head_log_p = masked_conditional_log_probabilities(
                all_head_logits, pair_mask, temperature
            )
            summary = native_head_summary(all_head_log_p, distances).detach()

            capacity_nodes = self._state_aware_tail_embeddings(
                node_embeddings, state
            ) + self.project_native_head_summary(summary)
            capacity_key, capacity_value, capacity_logit_key = self.project_state_tail_nodes(
                capacity_nodes
            ).chunk(3, dim=-1)
            capacity_query = self.project_tail_context(
                torch.cat((graph_context, last_head_embedding), dim=-1)
            )
            selected_mask = process.representative_mask(state, selected_tail)
            capacity_logits = self._attention_logits(
                capacity_query,
                self._split_heads(capacity_key),
                self._split_heads(capacity_value),
                capacity_logit_key,
                selected_mask,
                self.project_tail_glimpse,
            )
            native_logits = all_head_logits[batch_index, selected_tail]
            selected_head_log_p = torch.log_softmax(
                (native_logits + capacity_logits) / (sqrt(2.0) * temperature),
                dim=-1,
            )
            selected_head = _select(selected_head_log_p, decode_type, generator)

            selected_log_probabilities.append(
                selected_head_log_p.gather(1, selected_head[:, None]).squeeze(1)
            )
            action_entropies.append(categorical_entropy(selected_head_log_p))
            tails.append(selected_tail)
            heads.append(selected_head)
            state = process.transition(state, selected_tail, selected_head)
            last_head_embedding = node_embeddings.gather(
                1,
                selected_head[:, None, None].expand(state.batch_size, 1, self.embedding_dim),
            ).squeeze(1)

        tail_tensor = torch.stack(tails, dim=1)
        head_tensor = torch.stack(heads, dim=1)
        zeros = torch.zeros(
            state.batch_size,
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        return AttentionModelOutput(
            cost=process.objective(state, tail_tensor, head_tensor),
            log_likelihood=torch.stack(selected_log_probabilities, dim=1).sum(dim=1),
            tails=tail_tensor,
            heads=head_tensor,
            successor=process.solution(state),
            tail_entropy=zeros,
            action_entropy=torch.stack(action_entropies, dim=1).mean(dim=1),
        )

    def _state_aware_tail_embeddings(
        self,
        node_embeddings: Tensor,
        state: BatchedTSPState,
    ) -> Tensor:
        """为每个候选 tail 加入当前开放路径的结构信息。

        路径由起始端点、路径顶点平均表示和归一化规模概括。它们与候选 tail 自身的
        表示一起，使评分器能够感知两个端点以及当前分量的几何结构。
        """
        return node_embeddings + self.project_tail_state(state.path_state_features(node_embeddings))

    def _split_heads(self, values: Tensor) -> Tensor:
        batch, nodes, _ = values.shape
        head_dim = self.embedding_dim // self.n_heads
        return values.reshape(batch, nodes, self.n_heads, head_dim).permute(0, 2, 1, 3)

    def _attention_log_probabilities(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logit_key: Tensor,
        mask: Tensor,
        glimpse_projection: nn.Linear,
        temperature: float,
    ) -> Tensor:
        logits = self._attention_logits(query, key, value, logit_key, mask, glimpse_projection)
        return torch.log_softmax(logits / temperature, dim=-1)

    def _attention_logits(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logit_key: Tensor,
        mask: Tensor,
        glimpse_projection: nn.Linear,
    ) -> Tensor:
        """返回一个或多个条件 query 对应的原生 AM logits。"""
        squeeze_query = query.ndim == 2
        if squeeze_query:
            query = query.unsqueeze(1)
            mask = mask.unsqueeze(1)
        if query.ndim != 3 or mask.ndim != 3:
            raise ValueError("query and mask must describe one or more queries")
        batch, query_count, _ = query.shape
        if mask.shape != (batch, query_count, logit_key.size(1)):
            raise ValueError("attention mask has an incompatible shape")

        head_dim = self.embedding_dim // self.n_heads
        head_query = query.reshape(batch, query_count, self.n_heads, head_dim).permute(0, 2, 1, 3)
        compatibility = torch.matmul(head_query, key.transpose(-2, -1)) / sqrt(head_dim)
        fully_masked = mask.all(dim=-1, keepdim=True)
        safe_mask = mask & ~fully_masked
        compatibility = compatibility.masked_fill(safe_mask[:, None, :, :], -torch.inf)
        attention = torch.softmax(compatibility, dim=-1)
        glimpse = torch.matmul(attention, value)
        glimpse = glimpse.transpose(1, 2).reshape(batch, query_count, self.embedding_dim)
        glimpse = glimpse_projection(glimpse)

        logits = torch.matmul(glimpse, logit_key.transpose(-2, -1))
        logits = logits / sqrt(self.embedding_dim)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits = logits.masked_fill(mask, -torch.inf)
        return logits.squeeze(1) if squeeze_query else logits


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


def _deterministic_forest_tail_log_probabilities(
    state: BatchedTSPState, tail_mask: Tensor, dtype: torch.dtype
) -> Tensor:
    """优先扩展最短分量，并用节点编号打破平局。"""
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
    """从合法开放端点中均匀抽取一个 tail。

    这是无参数消融：tail 决策本身不参与反向传播，head 仍由宿主
    decoder 训练和选择。评估程序会传入固定随机种子，因而结果可复现。
    """
    legal_weights = (~tail_mask).to(dtype)
    if (legal_weights.sum(dim=1) == 0).any():
        raise ValueError("a nonterminal Forest state has no legal tail")
    selected = torch.multinomial(legal_weights, 1, generator=generator)
    log_p = torch.full(tail_mask.shape, -torch.inf, dtype=dtype, device=tail_mask.device)
    log_p.scatter_(1, selected, 0.0)
    return log_p
