"""Exact completion-regret diagnostic for source selection in path forests.

For every source, the fitted endpoint policy greedily chooses one legal head.
The remaining directed path components are then completed exactly by Held--Karp.
Thus the oracle only selects a source; it does not replace the endpoint policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path

import torch

from groupopt.models.ascc_bopo import SourceFirstASCC, _reshape_by_heads


def exact_cycle_cost(coords: torch.Tensor, starts: list[int], ends: list[int]) -> float:
    """Minimum directed cycle through fixed-orientation path components."""
    q = len(starts)
    distance = torch.cdist(coords[ends], coords[starts]).double().tolist()
    if q == 1:
        return distance[0][0]
    # Fix component 0 as the first component; cycle rotation is immaterial.
    dp: dict[tuple[int, int], float] = {(1, 0): 0.0}
    for mask in range(1, 1 << q):
        if not mask & 1:
            continue
        for last in range(q):
            old = dp.get((mask, last))
            if old is None:
                continue
            for nxt in range(1, q):
                if mask >> nxt & 1:
                    continue
                key = (mask | 1 << nxt, nxt)
                candidate = old + distance[last][nxt]
                if candidate < dp.get(key, float('inf')):
                    dp[key] = candidate
    full = (1 << q) - 1
    return min(dp[(full, last)] + distance[last][0] for last in range(1, q))


def brute_cycle_cost(coords: torch.Tensor, starts: list[int], ends: list[int]) -> float:
    from itertools import permutations
    if len(starts) == 1:
        return float(torch.dist(coords[ends[0]], coords[starts[0]]))
    values = []
    for rest in permutations(range(1, len(starts))):
        order = (0,) + rest
        values.append(sum(float(torch.dist(coords[ends[a]], coords[starts[b]]))
                          for a, b in zip(order, order[1:] + order[:1])))
    return min(values)


def reconstruct(n: int, tails: torch.Tensor, heads: torch.Tensor, steps: int):
    succ = torch.full((n,), -1, dtype=torch.long)
    pred = succ.clone()
    comp = torch.arange(n)
    starts = torch.arange(n)
    ends = torch.arange(n)
    sizes = torch.ones(n)
    route_tail = int(tails[0])
    last_head = int(tails[0])
    for t in range(steps):
        tail, head = int(tails[t]), int(heads[t])
        if succ[tail] >= 0 or pred[head] >= 0 or comp[tail] == comp[head]:
            raise AssertionError('invalid saved prefix')
        next_tail = int(ends[head])
        left, right = int(comp[tail]), int(comp[head])
        merge = (comp == left) | (comp == right)
        new_start, new_end = int(starts[tail]), int(ends[head])
        new_size = float(sizes[tail] + sizes[head])
        succ[tail], pred[head] = head, tail
        new_comp = min(left, right)
        comp[merge] = new_comp
        starts[merge] = new_start
        ends[merge] = new_end
        sizes[merge] = new_size
        route_tail, last_head = next_tail, head
    return dict(succ=succ, pred=pred, comp=comp, starts=starts, ends=ends,
                sizes=sizes, route_tail=route_tail, last_head=last_head)


@torch.no_grad()
def score_state(model, coords, encoded, state):
    """Return fitted source scores and greedy endpoint diagnostics for all sources."""
    device, n = encoded.device, len(coords)
    enc = encoded.unsqueeze(0)
    starts = state['starts'].to(device)
    ends = state['ends'].to(device)
    sizes = state['sizes'].to(device)
    unresolved = (state['succ'] < 0).nonzero().flatten().to(device)
    available = state['pred'] < 0
    comp = state['comp']
    decoder = model.decoder
    global_e = enc.mean(1).squeeze(0)
    last_e = encoded[state['last_head']]
    sn = model.source_node(encoded)
    ss = model.source_start(encoded)
    se = model.source_end(encoded)
    sf = torch.tanh(sn + ss[starts] + se[ends] + sizes[:, None] / n * model.source_size)
    query = model.source_context(torch.cat((global_e, last_e), -1))
    source_scores = model.clip * torch.tanh((sf * query).sum(-1) / sqrt(model.h))
    keys = _reshape_by_heads(decoder.Wk(enc), model.head_num)
    values = _reshape_by_heads(decoder.Wv(enc), model.head_num)
    first_queries = decoder.Wq_first(encoded)
    hn, he = model.head_node(encoded), model.head_end(encoded)
    rows = []
    for tail_tensor in unresolved:
        tail = int(tail_tensor)
        mask = ~available.to(device)
        # Different components must be joined until one path remains; the last
        # action closes that path into a cycle.
        if len(torch.unique(comp)) > 1:
            mask = mask | (comp.to(device) == int(comp[tail]))
        path_start = int(starts[tail])
        qfirst = _reshape_by_heads(first_queries[path_start][None, None], model.head_num)
        qlast = _reshape_by_heads(decoder.Wq_last(encoded[tail][None, None]), model.head_num)
        attention = (qfirst + qlast) @ keys.transpose(-2, -1) / sqrt(keys.shape[-1])
        weights = attention.masked_fill(mask[None, None, None], -torch.inf).softmax(-1)
        glimpse = (weights @ values).transpose(1, 2).reshape(1, 1, -1)
        glimpse = decoder.multi_head_combine(glimpse)
        native = model.clip * torch.tanh(
            (glimpse @ enc.transpose(1, 2)).squeeze() / sqrt(model.d))
        hc = model.head_context(torch.cat((encoded[tail], encoded[path_start]), -1))
        hf = torch.tanh(hn + he[ends] + hc + sizes[:, None] / n * model.head_size)
        logits = native + model.head_readout(hf).squeeze(-1)
        lp = torch.log_softmax(logits.masked_fill(mask, -torch.inf), -1)
        head = int(lp.argmax())
        finite = lp[~mask]
        top = finite.topk(min(2, len(finite))).values
        margin = float(top[0] - top[1]) if len(top) > 1 else float('inf')
        entropy = float(-(finite.exp() * finite).sum())
        edge = float(torch.dist(coords[tail], coords[head]))
        rows.append(dict(tail=tail, head=head, source_score=float(source_scores[tail]),
                         entropy=entropy, margin=margin, edge=edge,
                         component_size=float(sizes[tail])))
    return rows


def completion_after_action(coords, state, tail, head):
    comp, starts, ends = state['comp'], state['starts'], state['ends']
    left, right = int(comp[tail]), int(comp[head])
    labels = sorted(set(map(int, comp.tolist())) - {left, right})
    merged_start, merged_end = int(starts[tail]), int(ends[head])
    new_starts = [merged_start] + [int(starts[(comp == label).nonzero()[0]]) for label in labels]
    new_ends = [merged_end] + [int(ends[(comp == label).nonzero()[0]]) for label in labels]
    return float(torch.dist(coords[tail], coords[head])) + exact_cycle_cost(
        coords, new_starts, new_ends)


def select_indices(candidates, state):
    by_tail = {row['tail']: i for i, row in enumerate(candidates)}
    return {
        'learned': max(range(len(candidates)), key=lambda i: candidates[i]['source_score']),
        'route': by_tail[state['route_tail']],
        'fixed': min(range(len(candidates)), key=lambda i: candidates[i]['tail']),
        'shortest_component': min(range(len(candidates)),
                                  key=lambda i: (candidates[i]['component_size'], candidates[i]['tail'])),
        'min_entropy': min(range(len(candidates)), key=lambda i: candidates[i]['entropy']),
        'max_margin': max(range(len(candidates)), key=lambda i: candidates[i]['margin']),
        'shortest_edge': min(range(len(candidates)), key=lambda i: candidates[i]['edge']),
    }


def check_dp():
    g = torch.Generator().manual_seed(73)
    for q in range(1, 8):
        coords = torch.rand(2 * q, 2, generator=g)
        starts, ends = list(range(q)), list(range(q, 2 * q))
        assert abs(exact_cycle_cost(coords, starts, ends) -
                   brute_cycle_cost(coords, starts, ends)) < 1e-9


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=96)
    p.add_argument('--sizes', type=int, nargs='+', default=[20, 50])
    p.add_argument('--components', type=int, nargs='+', default=[4, 6, 8, 10])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    check_dp()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    records = []
    checkpoints = {
        'reinforce': args.screen / 'learned-reinforce-seed1234' / 'best.pt',
        'bopo': args.screen / 'learned-bopo-seed1234' / 'best.pt',
    }
    for objective, path in checkpoints.items():
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = SourceFirstASCC(**payload['config']['dimensions']).to(device).eval()
        model.load_state_dict(payload['model'], strict=True)
        for n in args.sizes:
            allowed = [m for m in args.components if m <= n]
            seed = 2026093000 + n
            coords_all = torch.rand(args.count, n, 2,
                                    generator=torch.Generator().manual_seed(seed))
            for index, coords in enumerate(coords_all):
                gpu_coords = coords[None].to(device)
                encoded = model.encoder(gpu_coords).squeeze(0)
                rollouts = {}
                for origin in ('learned', 'route'):
                    rollouts[origin] = model(gpu_coords, 1, origin, 'greedy', anchor_mode='same')
                for origin, rollout in rollouts.items():
                    tails = rollout.tails[0, 0].cpu()
                    heads = rollout.heads[0, 0].cpu()
                    for remaining in allowed:
                        steps = n - remaining
                        state = reconstruct(n, tails, heads, steps)
                        candidates = score_state(model, coords.to(device), encoded, state)
                        # Verify the extracted scorer against the original next action.
                        expected_tail, expected_head = int(tails[steps]), int(heads[steps])
                        chosen = max(candidates, key=lambda row: row['source_score'])
                        if origin == 'learned' and (chosen['tail'], chosen['head']) != (expected_tail, expected_head):
                            raise AssertionError('state scorer disagrees with learned rollout')
                        if origin == 'route':
                            route_row = next(row for row in candidates if row['tail'] == state['route_tail'])
                            if (route_row['tail'], route_row['head']) != (expected_tail, expected_head):
                                raise AssertionError('state scorer disagrees with route rollout')
                        values = [completion_after_action(coords, state, row['tail'], row['head'])
                                  for row in candidates]
                        oracle = min(values)
                        denominator = max(oracle, 1e-12)
                        picks = select_indices(candidates, state)
                        common = dict(objective=objective, n=n, instance=index,
                                      origin=origin, remaining=remaining,
                                      oracle_completion=oracle,
                                      source_spread_pct=100 * (max(values) - oracle) / denominator,
                                      source_count=len(candidates))
                        for policy, selected in picks.items():
                            records.append(common | dict(policy=policy,
                                regret=values[selected] - oracle,
                                regret_pct=100 * (values[selected] - oracle) / denominator,
                                oracle_hit=int(abs(values[selected] - oracle) < 1e-10)))
                        records.append(common | dict(policy='random_expected',
                            regret=sum(values) / len(values) - oracle,
                            regret_pct=100 * (sum(values) / len(values) - oracle) / denominator,
                            oracle_hit=sum(abs(v - oracle) < 1e-10 for v in values) / len(values)))
            print(json.dumps({'objective': objective, 'n': n,
                              'states_finished': args.count * 2 * len(allowed)}), flush=True)
        del model, payload
        torch.cuda.empty_cache()
    fields = list(records[0])
    with (args.out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    groups = {}
    for row in records:
        key = (row['objective'], row['n'], row['origin'], row['remaining'], row['policy'])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        values = torch.tensor([row['regret_pct'] for row in rows], dtype=torch.double)
        hits = torch.tensor([row['oracle_hit'] for row in rows], dtype=torch.double)
        spreads = torch.tensor([row['source_spread_pct'] for row in rows], dtype=torch.double)
        summary.append(dict(zip(('objective', 'n', 'origin', 'remaining', 'policy'), key)) |
                       dict(count=len(rows), mean_regret_pct=float(values.mean()),
                            se_regret_pct=float(values.std(unbiased=True) / len(values) ** .5),
                            oracle_hit_rate=float(hits.mean()), mean_source_spread_pct=float(spreads.mean())))
    provenance = {'role': 'exact_source_regret_diagnostic', 'count_per_size': args.count,
                  'sizes': args.sizes, 'components': args.components,
                  'checkpoint_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest()
                                        for k, v in checkpoints.items()},
                  'endpoint_rule': 'fitted greedy endpoint, then exact component completion',
                  'source_oracle_scope': 'source only; endpoint is never replaced',
                  'summary': summary}
    (args.out / 'summary.json').write_text(json.dumps(provenance, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
