"""Source-first path-forest policy with a BOPO/POMO-compatible backbone.

The original backbone names are preserved for strict checkpoint transfer. No
source continuation gate or endpoint candidate pruning is used. A construction
trajectory, not a marginal successor-permutation probability, is scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn

from groupopt.models.pomo import _POMOEncoder, _POMODecoder, _reshape_by_heads


def gather(x: Tensor, indices: Tensor) -> Tensor:
    if indices.ndim == 1:
        return x[torch.arange(x.shape[0], device=x.device), indices]
    return x.gather(1, indices[..., None].expand(-1, -1, x.shape[-1]))


def entropy(lp: Tensor) -> Tensor:
    safe = torch.where(torch.isfinite(lp), lp, 0.0)
    return -(lp.exp() * safe).sum(-1)


@dataclass
class ForestRollout:
    costs: Tensor                 # (instances, trajectories)
    source_logp: Tensor           # (instances, trajectories, edges)
    endpoint_logp: Tensor
    tails: Tensor
    heads: Tensor
    successor: Tensor
    source_entropy: Tensor
    voluntary_deviations: Tensor
    eligible_deviations: Tensor

    @property
    def mean_logp(self) -> Tensor:
        return (self.source_logp + self.endpoint_logp).mean(-1)


class SourceFirstASCC(nn.Module):
    def __init__(self, embedding_dim=128, head_num=8, qkv_dim=16,
                 encoder_layers=6, feed_forward_dim=512, source_dim=32,
                 logit_clipping=10.0):
        super().__init__()
        d, h = embedding_dim, source_dim
        self.d, self.h, self.head_num = d, h, head_num
        self.clip = logit_clipping
        self.encoder = _POMOEncoder(d, head_num, qkv_dim, encoder_layers, feed_forward_dim)
        self.decoder = _POMODecoder(d, head_num, qkv_dim, logit_clipping)
        # Cache node projections once; path endpoint indices are updated in O(n).
        self.source_node = nn.Linear(d, h, bias=False)
        self.source_start = nn.Linear(d, h, bias=False)
        self.source_end = nn.Linear(d, h, bias=False)
        self.source_context = nn.Linear(2 * d, h, bias=False)
        self.source_size = nn.Parameter(torch.zeros(h))
        self.head_node = nn.Linear(d, h, bias=False)
        self.head_end = nn.Linear(d, h, bias=False)
        self.head_context = nn.Linear(2 * d, h, bias=False)
        self.head_size = nn.Parameter(torch.zeros(h))
        self.head_readout = nn.Linear(h, 1, bias=False)
        # Exact native route initialization, while allowing forest-aware adaptation.
        nn.init.zeros_(self.head_readout.weight)

    def load_backbone(self, state_dict: dict[str, Tensor]):
        expected = {k: v for k, v in self.state_dict().items()
                    if k.startswith(('encoder.', 'decoder.'))}
        supplied = {k: v for k, v in state_dict.items()
                    if k.startswith(('encoder.', 'decoder.'))}
        if supplied.keys() != expected.keys():
            raise ValueError(f'backbone keys differ: {supplied.keys() ^ expected.keys()}')
        for k, v in supplied.items():
            if v.shape != expected[k].shape:
                raise ValueError(f'backbone shape differs: {k}')
        merged = self.state_dict()
        merged.update(supplied)
        self.load_state_dict(merged, strict=True)

    def forward(self, coordinates: Tensor, trajectories=8, source_policy='learned',
                decode='hybrid', generator=None, anchor_mode='same',
                actions: tuple[Tensor, Tensor] | None = None,
                validate=False, skip_source_scoring=False) -> ForestRollout:
        if source_policy not in ('learned', 'route', 'route_capacity', 'random',
                                 'fixed', 'shortest', 'min_entropy', 'max_margin',
                                 'shortest_edge'):
            raise ValueError(source_policy)
        if decode not in ('hybrid', 'sampling', 'greedy'):
            raise ValueError(decode)
        if anchor_mode not in ('same', 'multi') or trajectories < 1:
            raise ValueError('invalid rollout configuration')
        if skip_source_scoring and source_policy != 'fixed':
            raise ValueError('skip_source_scoring is only valid with source_policy=fixed')
        b, n, _ = coordinates.shape
        if n < 2:
            raise ValueError('TSP requires at least 2 nodes')
        k, r = trajectories, b * trajectories
        encoded = self.encoder(coordinates)
        enc = encoded[:, None].expand(b, k, n, self.d).reshape(r, n, self.d)
        xy = coordinates[:, None].expand(b, k, n, 2).reshape(r, n, 2)
        device = coordinates.device
        batch = torch.arange(r, device=device)
        nodes = torch.arange(n, device=device).expand(r, n)
        anchor = (torch.arange(k, device=device) % n if anchor_mode == 'multi'
                  else torch.zeros(k, dtype=torch.long, device=device))
        anchor = anchor[None].expand(b, k).reshape(r)
        anchor_e = gather(enc, anchor)
        # With a fixed source order, these projections and per-step source logits
        # are dead work.  Keep the historical path as the default and make the
        # optimized path explicit for fair Pair-ASCC latency comparisons.
        need_source_scores = not skip_source_scoring
        global_e = enc.mean(1) if need_source_scores else None
        succ = torch.full((r, n), -1, device=device, dtype=torch.long)
        pred = succ.clone()
        comp = nodes.clone()
        starts, ends = nodes.clone(), nodes.clone()
        sizes = torch.ones((r, n), device=device, dtype=enc.dtype)
        route_tail = anchor
        last_e = anchor_e
        greedy_rows = (torch.arange(r, device=device) % k == 0
                       if decode == 'hybrid' else torch.zeros(r, device=device, dtype=torch.bool))
        if decode == 'greedy':
            greedy_rows = torch.ones_like(greedy_rows)
        decoder = self.decoder
        keys = _reshape_by_heads(decoder.Wk(enc), self.head_num)
        values = _reshape_by_heads(decoder.Wv(enc), self.head_num)
        first_queries = decoder.Wq_first(enc)
        sn = ss = se = None
        if need_source_scores:
            sn, ss, se = self.source_node(enc), self.source_start(enc), self.source_end(enc)
        hn, he = self.head_node(enc), self.head_end(enc)
        slps, hlps, tails, heads, ents = [], [], [], [], []
        voluntary = torch.zeros(r, device=device)
        eligible = torch.zeros_like(voluntary)

        def choose(lp):
            best = lp.argmax(-1)
            if decode == 'greedy':
                return best
            sampled = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1)
            return torch.where(greedy_rows, best, sampled)

        for t in range(n):
            tail_mask = succ >= 0
            source_scores = None
            if need_source_scores:
                source_features = torch.tanh(sn + gather(ss, starts) + gather(se, ends)
                                            + sizes[..., None] / n * self.source_size)
                query = self.source_context(torch.cat((global_e, last_e), -1))
                source_scores = self.clip * torch.tanh(
                    (source_features * query[:, None]).sum(-1) / sqrt(self.h))
            all_head_lp = None
            if source_policy in ('min_entropy', 'max_margin', 'shortest_edge'):
                # Dense diagnostic path: score the endpoint distribution for every
                # unresolved source before choosing the source. This is intentionally
                # kept separate from the efficient selected-source production path.
                all_mask = (pred >= 0)[:, None, :].expand(-1, n, -1).clone()
                if t < n - 1:
                    all_mask |= comp[:, :, None] == comp[:, None, :]
                path_start_e = gather(enc, starts)
                qfirst_raw = gather(first_queries, starts)
                qlast_raw = decoder.Wq_last(enc)
                qfirst = _reshape_by_heads(qfirst_raw.reshape(r * n, 1, self.d),
                                           self.head_num).reshape(r, n, self.head_num, -1)
                qlast = _reshape_by_heads(qlast_raw.reshape(r * n, 1, self.d),
                                          self.head_num).reshape(r, n, self.head_num, -1)
                all_q = qfirst + qlast
                all_attention = torch.einsum('rshq,rhjq->rhsj', all_q, keys)
                all_attention = all_attention / sqrt(keys.shape[-1])
                all_weights = all_attention.masked_fill(
                    all_mask[:, None], -torch.inf).softmax(-1)
                all_glimpse = torch.einsum('rhsj,rhjd->rshd', all_weights, values)
                all_glimpse = decoder.multi_head_combine(all_glimpse.reshape(r, n, -1))
                all_native = self.clip * torch.tanh(
                    all_glimpse @ enc.transpose(1, 2) / sqrt(self.d))
                all_hc = self.head_context(torch.cat((enc, path_start_e), -1))
                candidate = hn + gather(he, ends) + sizes[..., None] / n * self.head_size
                all_hf = torch.tanh(all_hc[:, :, None] + candidate[:, None, :])
                all_logits = all_native + self.head_readout(all_hf).squeeze(-1)
                all_head_lp = torch.log_softmax(
                    all_logits.masked_fill(all_mask, -torch.inf), -1)
                safe = torch.where(torch.isfinite(all_head_lp), all_head_lp, 0.0)
                all_entropy = -(all_head_lp.exp() * safe).sum(-1)
                top2 = all_head_lp.topk(2, dim=-1).values
                all_margin = top2[..., 0] - top2[..., 1]
                all_best_head = all_head_lp.argmax(-1)
                best_xy = xy.gather(1, all_best_head[..., None].expand(-1, -1, 2))
                all_edge = (xy - best_xy).norm(dim=-1)
            if source_policy == 'learned':
                source_lp = torch.log_softmax(source_scores.masked_fill(tail_mask, -torch.inf), -1)
                tail = choose(source_lp) if actions is None else actions[0][:, :, t].reshape(r)
                source_ll = source_lp[batch, tail]
                source_ent = entropy(source_lp)
            elif source_policy == 'random':
                source_lp = torch.log_softmax(torch.zeros_like(source_scores).masked_fill(
                    tail_mask, -torch.inf), -1)
                # Random ordering stays random during endpoint-greedy evaluation.
                tail = (torch.multinomial(source_lp.exp(), 1, generator=generator).squeeze(1)
                        if actions is None else actions[0][:, :, t].reshape(r))
                source_ll, source_ent = source_lp[batch, tail], entropy(source_lp)
            else:
                if source_policy in ('route', 'route_capacity'):
                    tail = route_tail
                elif source_policy == 'fixed':
                    tail = nodes.masked_fill(tail_mask, n).argmin(-1)
                elif source_policy == 'shortest':
                    score = sizes * (n + 1) + nodes / n
                    tail = score.masked_fill(tail_mask, torch.inf).argmin(-1)
                elif source_policy == 'min_entropy':
                    tail = all_entropy.masked_fill(tail_mask, torch.inf).argmin(-1)
                elif source_policy == 'max_margin':
                    tail = all_margin.masked_fill(tail_mask, -torch.inf).argmax(-1)
                else:
                    tail = all_edge.masked_fill(tail_mask, torch.inf).argmin(-1)
                if actions is not None and not torch.equal(tail, actions[0][:, :, t].reshape(r)):
                    raise ValueError('replayed source violates deterministic source policy')
                source_ll = enc.new_zeros(r)
                source_ent = enc.new_zeros(r)
            head_mask = pred >= 0
            if t < n - 1:
                head_mask = head_mask | (comp == comp[batch, tail, None])
            if validate and tail_mask[batch, tail].any():
                raise ValueError('illegal replay/source')
            tail_e = gather(enc, tail)
            # A source belongs to its own path; the initial global anchor may lie
            # in another component. Use this path's actual first node in q_first.
            path_start = starts[batch, tail]
            qfirst = _reshape_by_heads(gather(first_queries, path_start)[:, None], self.head_num)
            q = qfirst + _reshape_by_heads(decoder.Wq_last(tail_e[:, None]), self.head_num)
            attn = (q @ keys.transpose(-2, -1)) / sqrt(keys.shape[-1])
            weights = attn.masked_fill(head_mask[:, None, None], -torch.inf).softmax(-1)
            glimpse = (weights @ values).transpose(1, 2).reshape(r, 1, -1)
            glimpse = decoder.multi_head_combine(glimpse)
            native_logits = self.clip * torch.tanh(
                (glimpse @ enc.transpose(1, 2)).squeeze(1) / sqrt(self.d))
            path_start_e = gather(enc, path_start)
            hc = self.head_context(torch.cat((tail_e, path_start_e), -1))
            hf = torch.tanh(hn + gather(he, ends) + hc[:, None]
                            + sizes[..., None] / n * self.head_size)
            logits = native_logits + self.head_readout(hf).squeeze(-1)
            if source_policy == 'route_capacity':
                # Activate the same source parameters as an extra endpoint scorer.
                logits = logits + source_scores
            head_lp = (all_head_lp[batch, tail] if all_head_lp is not None else
                       torch.log_softmax(logits.masked_fill(head_mask, -torch.inf), -1))
            head = choose(head_lp) if actions is None else actions[1][:, :, t].reshape(r)
            if validate and head_mask[batch, head].any():
                raise ValueError('illegal replay/endpoint')
            if t > 0:
                can_continue = ~tail_mask[batch, route_tail]
                has_choice = (~tail_mask).sum(-1) > 1
                eligible += (can_continue & has_choice).to(enc.dtype)
                voluntary += (can_continue & has_choice & (tail != route_tail)).to(enc.dtype)
            slps.append(source_ll)
            hlps.append(head_lp[batch, head])
            tails.append(tail)
            heads.append(head)
            ents.append(source_ent)
            # Record continuation and opposite endpoints before the merge.
            next_tail = ends[batch, head]
            new_start, new_end = starts[batch, tail], ends[batch, head]
            new_size = sizes[batch, tail] + sizes[batch, head]
            left, right = comp[batch, tail], comp[batch, head]
            merge = (comp == left[:, None]) | (comp == right[:, None])
            succ = succ.scatter(1, tail[:, None], head[:, None])
            pred = pred.scatter(1, head[:, None], tail[:, None])
            if t < n - 1:
                comp = torch.where(merge, torch.minimum(left, right)[:, None], comp)
                starts = torch.where(merge, new_start[:, None], starts)
                ends = torch.where(merge, new_end[:, None], ends)
                sizes = torch.where(merge, new_size[:, None], sizes)
            route_tail = next_tail
            last_e = gather(enc, head)
        ts, hs = torch.stack(tails, -1), torch.stack(heads, -1)
        costs = (gather(xy, ts) - gather(xy, hs)).norm(dim=-1).sum(-1)
        shaped = lambda x: x.reshape(b, k, *x.shape[1:])
        return ForestRollout(costs.reshape(b, k), shaped(torch.stack(slps, -1)),
                             shaped(torch.stack(hlps, -1)), shaped(ts), shaped(hs),
                             shaped(succ), shaped(torch.stack(ents, -1)),
                             voluntary.reshape(b, k), eligible.reshape(b, k))
