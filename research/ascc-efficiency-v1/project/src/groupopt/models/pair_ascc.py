"""Atomic two-edge path-forest construction for the registered ASCC screen.

The source pair is deliberately fixed to the two smallest unresolved tails.  This
first implementation isolates atomic endpoint-pair decoding from learned source
ordering.  It uses exactly the endpoint state of :class:`SourceFirstASCC` and
adds an optional low-rank interaction between the two endpoint choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from groupopt.models.ascc_bopo import SourceFirstASCC, gather
from groupopt.models.pomo import _reshape_by_heads


@dataclass
class PairRollout:
    costs: Tensor
    action_logp: Tensor
    successor: Tensor
    tails: Tensor
    heads: Tensor
    source_entropy: Tensor

    @property
    def mean_logp(self) -> Tensor:
        # Match the single-edge implementation's per-edge score scale.
        return self.action_logp.sum(-1) / self.successor.shape[-1]


class PairASCC(SourceFirstASCC):
    """Construct a tour using atomic pairs of edges and an optional interaction."""

    def __init__(self, *args, interaction_rank=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction_rank = interaction_rank
        self.pair_left = nn.Linear(self.h, interaction_rank, bias=False)
        self.pair_right = nn.Linear(self.h, interaction_rank, bias=False)
        self.pair_scale = nn.Parameter(torch.zeros(()))

    def forward(self, coordinates: Tensor, trajectories=8, interaction=False,
                decode='hybrid', generator=None, validate=False) -> PairRollout:
        if decode not in ('hybrid', 'sampling', 'greedy'):
            raise ValueError(decode)
        if trajectories < 1:
            raise ValueError('trajectories must be positive')
        bsz, n, _ = coordinates.shape
        if n < 2:
            raise ValueError('TSP requires at least 2 nodes')
        k, rows = trajectories, bsz * trajectories
        encoded = self.encoder(coordinates)
        enc = encoded[:, None].expand(bsz, k, n, self.d).reshape(rows, n, self.d)
        xy = coordinates[:, None].expand(bsz, k, n, 2).reshape(rows, n, 2)
        device = coordinates.device
        batch = torch.arange(rows, device=device)
        node_ids = torch.arange(n, device=device).expand(rows, n)
        greedy = (torch.arange(rows, device=device) % k == 0
                  if decode == 'hybrid' else torch.zeros(rows, device=device,
                                                         dtype=torch.bool))
        if decode == 'greedy':
            greedy = torch.ones_like(greedy)

        succ = torch.full((rows, n), -1, device=device, dtype=torch.long)
        pred = succ.clone()
        comp = node_ids.clone()
        starts, ends = node_ids.clone(), node_ids.clone()
        sizes = torch.ones((rows, n), device=device, dtype=enc.dtype)

        decoder = self.decoder
        keys = _reshape_by_heads(decoder.Wk(enc), self.head_num)
        values = _reshape_by_heads(decoder.Wv(enc), self.head_num)
        first_queries = decoder.Wq_first(enc)
        hn, he = self.head_node(enc), self.head_end(enc)

        action_lps, tails, heads = [], [], []

        def choose(logp: Tensor) -> Tensor:
            best = logp.argmax(-1)
            if decode == 'greedy':
                return best
            sampled = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(1)
            return torch.where(greedy, best, sampled)

        def endpoint_logits(tail: Tensor, forbid_component=True):
            tail_e = gather(enc, tail)
            path_start = starts[batch, tail]
            qfirst = _reshape_by_heads(gather(first_queries, path_start)[:, None],
                                       self.head_num)
            qlast = _reshape_by_heads(decoder.Wq_last(tail_e[:, None]), self.head_num)
            attention = ((qfirst + qlast) @ keys.transpose(-2, -1)) / sqrt(keys.shape[-1])
            mask = pred >= 0
            if forbid_component:
                mask = mask | (comp == comp[batch, tail, None])
            weights = attention.masked_fill(mask[:, None, None], -torch.inf).softmax(-1)
            glimpse = (weights @ values).transpose(1, 2).reshape(rows, 1, -1)
            glimpse = decoder.multi_head_combine(glimpse)
            native = self.clip * torch.tanh(
                (glimpse @ enc.transpose(1, 2)).squeeze(1) / sqrt(self.d))
            start_e = gather(enc, path_start)
            context = self.head_context(torch.cat((tail_e, start_e), -1))
            features = torch.tanh(hn + gather(he, ends) + context[:, None]
                                  + sizes[..., None] / n * self.head_size)
            logits = native + self.head_readout(features).squeeze(-1)
            return logits, mask, features

        def apply_edge(tail: Tensor, head: Tensor, last: bool):
            nonlocal succ, pred, comp, starts, ends, sizes
            left, right = comp[batch, tail], comp[batch, head]
            new_start, new_end = starts[batch, tail], ends[batch, head]
            new_size = sizes[batch, tail] + sizes[batch, head]
            merge = (comp == left[:, None]) | (comp == right[:, None])
            succ = succ.scatter(1, tail[:, None], head[:, None])
            pred = pred.scatter(1, head[:, None], tail[:, None])
            if not last:
                comp = torch.where(merge, torch.minimum(left, right)[:, None], comp)
                starts = torch.where(merge, new_start[:, None], starts)
                ends = torch.where(merge, new_end[:, None], ends)
                sizes = torch.where(merge, new_size[:, None], sizes)

        edge_count = 0
        while edge_count < n:
            unresolved = succ < 0
            remaining = n - edge_count
            if remaining == 1:
                tail = node_ids.masked_fill(~unresolved, n).argmin(-1)
                logits, mask, _ = endpoint_logits(tail, forbid_component=False)
                logp = torch.log_softmax(logits.masked_fill(mask, -torch.inf), -1)
                head = choose(logp)
                if validate and mask[batch, head].any():
                    raise ValueError('illegal final edge')
                action_lps.append(logp[batch, head])
                tails.append(tail)
                heads.append(head)
                apply_edge(tail, head, last=True)
                edge_count += 1
                continue

            # Registered P2/P3 source control: the two smallest unresolved tails.
            ordered = node_ids.masked_fill(~unresolved, n).topk(2, largest=False).values
            first, second = ordered[:, 0], ordered[:, 1]
            logits_a, mask_a, feat_a = endpoint_logits(first, forbid_component=True)
            logits_b, mask_b, feat_b = endpoint_logits(second, forbid_component=True)
            joint = logits_a[:, :, None] + logits_b[:, None, :]
            if interaction:
                left = self.pair_left(feat_a)
                right = self.pair_right(feat_b)
                joint = joint + self.pair_scale.tanh() * torch.einsum(
                    'bir,bjr->bij', left, right) / sqrt(self.interaction_rank)
            pair_mask = mask_a[:, :, None] | mask_b[:, None, :]
            pair_mask |= torch.eye(n, device=device, dtype=torch.bool)[None]
            # Before the last pair, two reciprocal component links form a
            # premature cycle.  At the last pair they are the Hamilton closure.
            if remaining > 2:
                comp_a = comp[batch, first]
                comp_b = comp[batch, second]
                reciprocal = ((comp[:, :, None] == comp_b[:, None, None]) &
                              (comp[:, None, :] == comp_a[:, None, None]))
                pair_mask |= reciprocal
            flat = joint.masked_fill(pair_mask, -torch.inf).reshape(rows, n * n)
            if validate and (~torch.isfinite(flat)).all(-1).any():
                raise ValueError('source pair has no legal endpoint pair')
            logp = torch.log_softmax(flat, -1)
            selected = choose(logp)
            head_a, head_b = selected // n, selected % n
            if validate and pair_mask[batch, head_a, head_b].any():
                raise ValueError('illegal endpoint pair')
            action_lps.append(logp[batch, selected])
            tails.extend((first, second))
            heads.extend((head_a, head_b))
            apply_edge(first, head_a, last=False)
            apply_edge(second, head_b, last=(remaining == 2))
            edge_count += 2

        tail_tensor, head_tensor = torch.stack(tails, -1), torch.stack(heads, -1)
        costs = (gather(xy, tail_tensor) - gather(xy, head_tensor)).norm(dim=-1).sum(-1)
        shape = lambda x: x.reshape(bsz, k, *x.shape[1:])
        return PairRollout(costs.reshape(bsz, k), shape(torch.stack(action_lps, -1)),
                           shape(succ), shape(tail_tensor), shape(head_tensor),
                           costs.new_zeros(bsz, k, len(action_lps)))
