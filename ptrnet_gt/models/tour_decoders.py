from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import DataParallel

from ptrnet_gt.models.graph_encoder import GraphAttentionEncoder
from ptrnet_gt.problems.tsp import TSP


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


class TourDecoderBase(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        tanh_clipping=10.0,
        normalization="batch",
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.problem = problem
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        self.init_embed = nn.Linear(2, embedding_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=n_encode_layers,
            normalization=normalization,
        )

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def _init_embed(self, input):
        return self.init_embed(input)

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        masked_probs = probs.clone()
        masked_probs[mask] = 0
        if self.decode_type == "greedy":
            _, selected = masked_probs.max(1)
        elif self.decode_type == "sampling":
            selected = masked_probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                selected = masked_probs.multinomial(1).squeeze(1)
        else:
            raise ValueError("Unknown decode type")
        return selected

    def _update_mask(self, mask: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        next_mask = mask.clone()
        next_mask[torch.arange(mask.size(0), device=mask.device), selected] = True
        return next_mask

    def _calc_log_likelihood(self, log_p, a):
        return log_p.gather(2, a.unsqueeze(-1)).squeeze(-1).sum(1)

    def _calc_cost(self, input, pi):
        cost, _ = TSP.get_costs(input, pi)
        return cost

    def _make_heads(self, v, num_steps=None):
        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )


class PointerNetworkDecoder(TourDecoderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_cell = nn.GRUCell(self.embedding_dim, self.hidden_dim)
        self.pointer_query = nn.Linear(self.hidden_dim, self.embedding_dim, bias=False)
        self.pointer_key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.step_context = nn.Parameter(torch.zeros(self.embedding_dim))

    def forward(self, input, return_pi=False):
        embeddings, graph_embed = self.embedder(self._init_embed(input))
        log_p, pi = self._inner(embeddings, graph_embed)
        cost = self._calc_cost(input, pi)
        ll = self._calc_log_likelihood(log_p, pi)
        if return_pi:
            return cost, ll, pi
        return cost, ll

    def _inner(self, embeddings, graph_embed):
        batch_size, n_nodes, _ = embeddings.size()
        device = embeddings.device
        mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
        hidden = graph_embed
        step_input = self.step_context[None, :].expand(batch_size, -1)
        outputs = []
        sequences = []

        keys = self.pointer_key(embeddings)
        for _ in range(n_nodes):
            hidden = self.decoder_cell(step_input, hidden)
            query = self.pointer_query(hidden)[:, None, :]
            logits = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / math.sqrt(self.embedding_dim)
            logits[mask] = -math.inf
            log_probs = torch.log_softmax(logits / self.temp, dim=-1)
            selected = self._select_node(log_probs.exp(), mask)
            outputs.append(log_probs)
            sequences.append(selected)
            mask = self._update_mask(mask, selected)
            step_input = embeddings[torch.arange(batch_size, device=device), selected]

        return torch.stack(outputs, dim=1), torch.stack(sequences, dim=1)


class AttentionModelDecoder(TourDecoderBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_node_embeddings = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_step_context = nn.Linear(2 * self.embedding_dim, self.embedding_dim, bias=False)
        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_placeholder = nn.Parameter(torch.Tensor(self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)

    def forward(self, input, return_pi=False):
        embeddings, _ = self.embedder(self._init_embed(input))
        log_p, pi = self._inner(embeddings)
        cost = self._calc_cost(input, pi)
        ll = self._calc_log_likelihood(log_p, pi)
        if return_pi:
            return cost, ll, pi
        return cost, ll

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

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        logits[mask] = -math.inf
        return logits

    def _inner(self, embeddings):
        fixed = self._precompute(embeddings)
        batch_size, n_nodes, _ = embeddings.size()
        device = embeddings.device
        mask = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=device)
        first_step = self.W_placeholder[None, None, :].expand(batch_size, 1, self.embedding_dim)
        prev_embedding = first_step
        outputs = []
        sequences = []

        for _ in range(n_nodes):
            query_input = torch.cat((fixed.context_node_projected, prev_embedding), dim=-1)
            query = self.project_step_context(query_input)
            step_mask = mask[:, None, :]
            logits = self._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, step_mask)
            log_probs = torch.log_softmax(logits[:, 0, :] / self.temp, dim=-1)
            selected = self._select_node(log_probs.exp(), mask)
            outputs.append(log_probs)
            sequences.append(selected)
            mask = self._update_mask(mask, selected)
            prev_embedding = embeddings[torch.arange(batch_size, device=device), selected][:, None, :]

        return torch.stack(outputs, dim=1), torch.stack(sequences, dim=1)


__all__ = [
    "set_decode_type",
    "AttentionModelFixed",
    "TourDecoderBase",
    "PointerNetworkDecoder",
    "AttentionModelDecoder",
]