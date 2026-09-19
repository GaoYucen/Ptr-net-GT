"""Upper bound from perfect source values under the actual downstream policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
import time

import torch

from evaluate_source_regret_oracle import score_state
from groupopt.models.ascc_bopo import SourceFirstASCC, _reshape_by_heads


def initial_state(n, anchor=0):
    nodes = torch.arange(n)
    return dict(succ=torch.full((n,), -1, dtype=torch.long),
                pred=torch.full((n,), -1, dtype=torch.long), comp=nodes.clone(),
                starts=nodes.clone(), ends=nodes.clone(), sizes=torch.ones(n),
                route_tail=anchor, last_head=anchor, edges=0, cost=0.0)


def clone_state(state):
    return {key: value.clone() if torch.is_tensor(value) else value
            for key, value in state.items()}


def components(state):
    return len(torch.unique(state['comp']))


def apply_edge(coords, state, tail, head):
    tail, head = int(tail), int(head)
    if state['succ'][tail] >= 0 or state['pred'][head] >= 0:
        raise AssertionError('used source/image')
    count = components(state)
    if count > 1 and state['comp'][tail] == state['comp'][head]:
        raise AssertionError('premature cycle')
    next_tail = int(state['ends'][head])
    left, right = int(state['comp'][tail]), int(state['comp'][head])
    new_start, new_end = int(state['starts'][tail]), int(state['ends'][head])
    new_size = float(state['sizes'][tail] + state['sizes'][head])
    state['succ'][tail], state['pred'][head] = head, tail
    if count > 1:
        merge = (state['comp'] == left) | (state['comp'] == right)
        state['comp'][merge] = min(left, right)
        state['starts'][merge] = new_start
        state['ends'][merge] = new_end
        state['sizes'][merge] = new_size
    state['route_tail'] = next_tail
    state['last_head'] = head
    state['edges'] += 1
    state['cost'] += float(torch.dist(coords[tail], coords[head]))


@torch.no_grad()
def greedy_endpoint(model, coords, encoded, state, tail):
    """Efficient selected-source endpoint used inside the many rollouts."""
    device, n = encoded.device, len(coords)
    enc = encoded.unsqueeze(0)
    starts, ends = state['starts'].to(device), state['ends'].to(device)
    sizes = state['sizes'].to(device)
    mask = (state['pred'] >= 0).to(device)
    if components(state) > 1:
        mask |= state['comp'].to(device) == int(state['comp'][tail])
    decoder = model.decoder
    keys = _reshape_by_heads(decoder.Wk(enc), model.head_num)
    values = _reshape_by_heads(decoder.Wv(enc), model.head_num)
    path_start = int(starts[tail])
    first_query = decoder.Wq_first(encoded[path_start][None, None])
    last_query = decoder.Wq_last(encoded[tail][None, None])
    query = _reshape_by_heads(first_query + last_query, model.head_num)
    attention = query @ keys.transpose(-2, -1) / sqrt(keys.shape[-1])
    weights = attention.masked_fill(mask[None, None, None], -torch.inf).softmax(-1)
    glimpse = (weights @ values).transpose(1, 2).reshape(1, 1, -1)
    glimpse = decoder.multi_head_combine(glimpse)
    native = model.clip * torch.tanh(
        (glimpse @ enc.transpose(1, 2)).squeeze() / sqrt(model.d))
    hn, he = model.head_node(encoded), model.head_end(encoded)
    hc = model.head_context(torch.cat((encoded[tail], encoded[path_start]), -1))
    residual = model.head_readout(torch.tanh(
        hn + he[ends] + hc + sizes[:, None] / n * model.head_size)).squeeze(-1)
    return int((native + residual).masked_fill(mask, -torch.inf).argmax())


def legal_heads(state, tail):
    mask = state['pred'] < 0
    if components(state) > 1:
        mask &= state['comp'] != state['comp'][tail]
    return mask.nonzero().flatten().tolist()


def continue_route(model, coords, encoded, state):
    while state['edges'] < len(coords):
        tail = state['route_tail']
        head = greedy_endpoint(model, coords, encoded, state, tail)
        apply_edge(coords, state, tail, head)
    return state


def route_prefix(model, coords, encoded, start_components):
    state = initial_state(len(coords))
    while components(state) > start_components:
        tail = state['route_tail']
        apply_edge(coords, state, tail, greedy_endpoint(model, coords, encoded, state, tail))
    return state


def best_source_branch(model, coords, encoded, state):
    rows = score_state(model, coords.to(encoded.device), encoded, state)
    best = None
    for row in rows:
        branch = clone_state(state)
        apply_edge(coords, branch, row['tail'], row['head'])
        continue_route(model, coords, encoded, branch)
        candidate = (branch['cost'], row['tail'], row['head'])
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def best_endpoint_branch(model, coords, encoded, state):
    tail = state['route_tail']
    best = None
    for head in legal_heads(state, tail):
        branch = clone_state(state)
        apply_edge(coords, branch, tail, head)
        continue_route(model, coords, encoded, branch)
        candidate = (branch['cost'], tail, head)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def one_shot(model, coords, encoded, start_components, action):
    state = route_prefix(model, coords, encoded, start_components)
    value, _, _ = (best_source_branch(model, coords, encoded, state) if action == 'source'
                    else best_endpoint_branch(model, coords, encoded, state))
    return value


def repeated(model, coords, encoded, start_components, action):
    state = route_prefix(model, coords, encoded, start_components)
    deviations = 0
    while state['edges'] < len(coords):
        route_tail = state['route_tail']
        route_head = greedy_endpoint(model, coords, encoded, state, route_tail)
        _, tail, head = (best_source_branch(model, coords, encoded, state) if action == 'source'
                         else best_endpoint_branch(model, coords, encoded, state))
        deviations += int(tail != route_tail or head != route_head)
        apply_edge(coords, state, tail, head)
    return state['cost'], deviations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--sizes', type=int, nargs='+', default=[20, 50])
    p.add_argument('--start-components', type=int, default=10)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checkpoint = args.screen / 'learned-reinforce-seed1234' / 'best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SourceFirstASCC(**payload['config']['dimensions']).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    methods = ('route', 'source_once', 'source_repeated',
               'endpoint_once', 'endpoint_repeated')
    rows = []
    for n in args.sizes:
        coords_all = torch.rand(args.count, n, 2,
                                generator=torch.Generator().manual_seed(2026093000 + n))
        begin = time.monotonic()
        for index, coords in enumerate(coords_all):
            encoded = model.encoder(coords[None].cuda()).squeeze(0)
            route = continue_route(model, coords, encoded, initial_state(n))['cost']
            # Guard the custom continuation against the production decoder.
            if index == 0:
                production = float(model(coords[None].cuda(), 1, 'route', 'greedy').costs[0, 0])
                if abs(route - production) > 2e-5:
                    raise AssertionError((route, production))
            source_once = one_shot(model, coords, encoded, args.start_components, 'source')
            endpoint_once = one_shot(model, coords, encoded, args.start_components, 'endpoint')
            source_repeated, source_deviations = repeated(
                model, coords, encoded, args.start_components, 'source')
            endpoint_repeated, endpoint_deviations = repeated(
                model, coords, encoded, args.start_components, 'endpoint')
            values = dict(route=route, source_once=source_once,
                          source_repeated=source_repeated, endpoint_once=endpoint_once,
                          endpoint_repeated=endpoint_repeated)
            for method, value in values.items():
                rows.append(dict(n=n, instance=index, method=method, cost=value,
                                 source_deviations=source_deviations,
                                 endpoint_deviations=endpoint_deviations))
            if (index + 1) % 8 == 0:
                print(json.dumps({'n': n, 'finished': index + 1,
                                  'seconds': time.monotonic() - begin}), flush=True)
    with (args.out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {'role': 'actual_downstream_rollout_value_oracle',
               'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'count_per_size': args.count, 'sizes': args.sizes,
               'start_components': args.start_components,
               'downstream_policy': 'route source with fitted greedy endpoint',
               'methods': []}
    for n in args.sizes:
        route = torch.tensor([r['cost'] for r in rows if r['n'] == n and r['method'] == 'route'])
        for method in methods:
            values = torch.tensor([r['cost'] for r in rows if r['n'] == n and r['method'] == method])
            delta = route - values
            mean, se = float(delta.mean()), float(delta.std(unbiased=True) / len(delta) ** .5)
            summary['methods'].append(dict(n=n, method=method, mean_cost=float(values.mean()),
                route_minus_method=mean, relative_improvement_percent=100 * mean / float(route.mean()),
                conditional_normal_ci95=[mean - 1.96 * se, mean + 1.96 * se]))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
