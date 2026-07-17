import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.checkpoint import checkpoint

from ptrnet_gt.models.graph_encoder import GraphAttentionEncoder
from ptrnet_gt.states import ComponentMergeState


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],
            glimpse_val=self.glimpse_val[:, key],
            logit_key=self.logit_key[key],
        )


class ComponentMergeDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalization="batch",
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
        context_mode="cross_step",
        action_mode="tail_head",
        use_dynamic_role_features=False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.context_mode = context_mode
        self.action_mode = action_mode
        self.use_dynamic_role_features = use_dynamic_role_features

        self.init_embed = nn.Linear(2, embedding_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
        )
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)

        tail_context_dim = embedding_dim if context_mode == "none" else 2 * embedding_dim
        self.project_step_context_tail = nn.Linear(tail_context_dim, embedding_dim, bias=False)
        self.project_out_tail = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context_head = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_out_head = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_placeholder_tail = nn.Parameter(torch.Tensor(embedding_dim))
        self.W_placeholder_tail.data.uniform_(-1, 1)
        self.role_feature_dim = 8 if use_dynamic_role_features else 0
        edge_feature_dim = 4 * embedding_dim + 2 * self.role_feature_dim + 1
        self.edge_score = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        embeddings = self.encode(input)

        log_p, pi = self._inner(input, embeddings)
        cost = self._calc_cost(input, pi)
        ll = self._calc_log_likelihood(log_p, pi, None, input.size(1))
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _init_embed(self, input):
        return self.init_embed(input)

    def encode(self, input):
        if self.checkpoint_encoder and self.training:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))
        return embeddings

    def _inner(self, input, embeddings):
        if self.action_mode == "joint_edge":
            return self._inner_joint_edge(input, embeddings)
        if self.action_mode != "tail_head":
            raise ValueError(f"Unknown component merge action_mode: {self.action_mode}")

        return self._inner_tail_head(input, embeddings)

    def _inner_tail_head(self, input, embeddings):
        state = ComponentMergeState.initialize(input)
        outputs = []
        sequences = []
        fixed = self._precompute(embeddings)
        batch_size = state.ids.size(0)
        last_head_embedding = self.W_placeholder_tail[None, None, :].expand(batch_size, 1, self.embedding_dim)

        while not state.all_finished():
            edge_mask = state.get_edge_mask()
            tail_mask = edge_mask.all(dim=-1)[:, None, :]
            if self.context_mode == "none":
                tail_context_input = fixed.context_node_projected
            else:
                tail_context_input = torch.cat((fixed.context_node_projected, last_head_embedding), dim=-1)
            query_tail = self.project_step_context_tail(tail_context_input)
            log_p_tail, _ = self._get_log_p(fixed, query_tail, tail_mask, self.project_out_tail)
            selected_tail = self._select_node(log_p_tail.exp()[:, 0, :], tail_mask[:, 0, :])

            head_mask = edge_mask[torch.arange(batch_size, device=input.device), selected_tail][:, None, :]
            tail_embedding = torch.gather(
                embeddings, 1, selected_tail[:, None, None].expand(batch_size, 1, self.embedding_dim)
            )
            head_context_input = torch.cat((fixed.context_node_projected, tail_embedding), dim=-1)
            query_head = self.project_step_context_head(head_context_input)
            log_p_head, _ = self._get_log_p(fixed, query_head, head_mask, self.project_out_head)
            selected_head = self._select_node(log_p_head.exp()[:, 0, :], head_mask[:, 0, :])

            state = state.update(selected_tail, selected_head)
            last_head_embedding = torch.gather(
                embeddings, 1, selected_head[:, None, None].expand(batch_size, 1, self.embedding_dim)
            )
            outputs.extend([log_p_tail[:, 0, :], log_p_head[:, 0, :]])
            sequences.extend([selected_tail, selected_head])

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _inner_joint_edge(self, input, embeddings):
        state = ComponentMergeState.initialize(input)
        outputs = []
        sequences = []
        batch_size, n_nodes, _ = input.size()

        while not state.all_finished():
            edge_mask = state.get_edge_mask()
            log_p_edge = self._get_log_p_edge(state, embeddings, edge_mask)
            selected_edge = self._select_node(log_p_edge.exp(), edge_mask.view(batch_size, -1))
            tail_idx = torch.div(selected_edge, n_nodes, rounding_mode='floor')
            head_idx = selected_edge % n_nodes

            state = state.update(tail_idx, head_idx)
            outputs.append(log_p_edge)
            sequences.extend([tail_idx, head_idx])

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _get_log_p_edge(self, state, embeddings, edge_mask):
        edge_logits = self._compute_edge_logits(state, embeddings)
        flat_logits = edge_logits.view(edge_logits.size(0), -1)
        flat_mask = edge_mask.view(edge_mask.size(0), -1)
        if self.mask_logits:
            flat_logits[flat_mask] = -math.inf
        return torch.log_softmax(flat_logits / self.temp, dim=-1)

    def get_joint_edge_log_p(self, state, embeddings):
        return self._get_log_p_edge(state, embeddings, state.get_edge_mask())

    def _compute_edge_logits(self, state, embeddings):
        node_i = embeddings[:, :, None, :].expand(-1, -1, state.n_nodes, -1)
        node_j = embeddings[:, None, :, :].expand(-1, state.n_nodes, -1, -1)
        dist = state.dist[:, :, :, None]
        parts = [node_i, node_j, node_i * node_j, torch.abs(node_i - node_j)]
        if self.use_dynamic_role_features:
            role_features = state.get_node_role_features()
            role_i = role_features[:, :, None, :].expand(-1, -1, state.n_nodes, -1)
            role_j = role_features[:, None, :, :].expand(-1, state.n_nodes, -1, -1)
            parts.extend([role_i, role_j])
        parts.append(dist)
        edge_features = torch.cat(parts, dim=-1)
        logits = self.edge_score(edge_features).squeeze(-1)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        return logits

    def _get_log_p(self, fixed, query, mask, project_out_layer):
        log_p, _ = self._one_to_many_logits(
            query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask, project_out_layer
        )
        log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        return log_p, mask

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, project_out_layer):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        glimpse = project_out_layer(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        return logits, glimpse.squeeze(-2)

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                selected = probs.multinomial(1).squeeze(1)
        else:
            raise ValueError("Unknown decode type")
        return selected

    def _precompute(self, embeddings):
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_node_embeddings(
            embeddings[:, None, :, :]
        ).chunk(3, dim=-1)
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps=1),
            self._make_heads(glimpse_val_fixed, num_steps=1),
            logit_key_fixed.contiguous(),
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _make_heads(self, v, num_steps=None):
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )

    def _calc_log_likelihood(self, log_p, a, mask, n_nodes=None):
        if self.action_mode == "joint_edge":
            tails = a[:, 0::2]
            heads = a[:, 1::2]
            edge_count = n_nodes if n_nodes is not None else int(math.sqrt(log_p.size(-1)))
            edge_idx = tails * edge_count + heads
            return log_p.gather(2, edge_idx.unsqueeze(-1)).squeeze(-1).sum(1)
        return log_p.gather(2, a.unsqueeze(-1)).squeeze(-1).sum(1)

    def _calc_cost(self, input, pi):
        tails = pi[:, 0::2]
        heads = pi[:, 1::2]
        batch_size, n_nodes, _ = input.size()
        tail_coords = torch.gather(input, 1, tails[:, :, None].expand(batch_size, n_nodes, 2))
        head_coords = torch.gather(input, 1, heads[:, :, None].expand(batch_size, n_nodes, 2))
        return (tail_coords - head_coords).norm(p=2, dim=-1).sum(1)


__all__ = ["ComponentMergeDecoder", "AttentionModelFixed", "set_decode_type"]