"""Diagnose ASCC on genuine multi-path reconstruction states.

The experiment starts from a complete route, cuts it into non-trivial directed
path fragments, and asks each policy to reconnect the same fragments.  It
separates the value of the forest search space from the ability of the current
learned policy to exploit that space.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from math import sqrt
from pathlib import Path
import time

import torch

from evaluate_rollout_value_oracle import (
    apply_edge, clone_state, components as component_count, greedy_endpoint,
)
from evaluate_source_regret_oracle import score_state
from groupopt.models.ascc_bopo import SourceFirstASCC


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def route_cost(coords: torch.Tensor, tour: list[int]) -> float:
    order = torch.tensor(tour, dtype=torch.long)
    nxt = order.roll(-1)
    return float((coords[order] - coords[nxt]).norm(dim=-1).sum())


def tour_from_actions(tails: torch.Tensor, heads: torch.Tensor) -> list[int]:
    n = len(tails)
    succ = torch.full((n,), -1, dtype=torch.long)
    succ[tails.cpu()] = heads.cpu()
    tour, node = [], 0
    for _ in range(n):
        if node in tour or node < 0:
            raise AssertionError('actions do not define one Hamiltonian cycle')
        tour.append(node)
        node = int(succ[node])
    if node != tour[0] or sorted(tour) != list(range(n)):
        raise AssertionError('actions do not define one Hamiltonian cycle')
    return tour


def two_opt(coords: torch.Tensor, tour: list[int], max_moves: int) -> list[int]:
    """Best-improvement symmetric 2-opt, used only to strengthen the seed tour."""
    tour = list(tour)
    n = len(tour)
    for _ in range(max_moves):
        best_delta, best_pair = -1e-10, None
        for i in range(n - 2):
            a, b = tour[i], tour[i + 1]
            old_ab = float(torch.dist(coords[a], coords[b]))
            for j in range(i + 2, n - (1 if i == 0 else 0)):
                c, d = tour[j], tour[(j + 1) % n]
                delta = (float(torch.dist(coords[a], coords[c]))
                         + float(torch.dist(coords[b], coords[d]))
                         - old_ab - float(torch.dist(coords[c], coords[d])))
                if delta < best_delta:
                    best_delta, best_pair = delta, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
    return tour


def split_fragments(tour: list[int], count: int, generator: torch.Generator,
                    strategy: str, coords: torch.Tensor) -> list[list[int]]:
    """Cut a cycle into paths of at least two nodes using a declared strategy."""
    n = len(tour)
    if 2 * count > n:
        raise ValueError('need at least two nodes in every path fragment')
    if strategy == 'random':
        rotation = int(torch.randint(n, (), generator=generator))
        rotated = tour[rotation:] + tour[:rotation]
        sizes = torch.full((count,), 2, dtype=torch.long)
        if n > 2 * count:
            bins = torch.randint(count, (n - 2 * count,), generator=generator)
            sizes += torch.bincount(bins, minlength=count)
        fragments, offset = [], 0
        for size in sizes.tolist():
            fragments.append(rotated[offset:offset + size])
            offset += size
        assert offset == n
    elif strategy == 'longest':
        edge_lengths = [float(torch.dist(coords[tour[i]], coords[tour[(i + 1) % n]]))
                        for i in range(n)]

        @lru_cache(None)
        def path_opt(index: int, stop: int, need: int):
            if need == 0:
                return 0.0, ()
            if index >= stop or stop - index < 2 * need - 1:
                return -float('inf'), ()
            skip = path_opt(index + 1, stop, need)
            take_tail = path_opt(index + 2, stop, need - 1)
            take = edge_lengths[index] + take_tail[0], (index,) + take_tail[1]
            return take if take[0] > skip[0] else skip

        without_zero = path_opt(1, n, count)
        with_zero_tail = path_opt(2, n - 1, count - 1)
        with_zero = edge_lengths[0] + with_zero_tail[0], (0,) + with_zero_tail[1]
        _, selected = with_zero if with_zero[0] > without_zero[0] else without_zero
        cuts = list(selected)
        if len(cuts) != count:
            raise AssertionError('could not select optimal spaced longest-edge cuts')
        cut_set = set(cuts)
        start = (min(cuts) + 1) % n
        fragments, path = [], []
        for step in range(n):
            index = (start + step) % n
            path.append(tour[index])
            if index in cut_set:
                fragments.append(path)
                path = []
        if path:
            raise AssertionError('cyclic split did not close at a cut edge')
    else:
        raise ValueError(strategy)
    assert len(fragments) == count and sum(map(len, fragments)) == n
    assert all(len(path) >= 2 for path in fragments)
    return fragments


def internal_cost(coords: torch.Tensor, fragments: list[list[int]]) -> float:
    return sum(float(torch.dist(coords[a], coords[b]))
               for path in fragments for a, b in zip(path, path[1:]))


def connection_cost(coords: torch.Tensor, fragments: list[list[int]]) -> float:
    return sum(float(torch.dist(coords[left[-1]], coords[right[0]]))
               for left, right in zip(fragments, fragments[1:] + fragments[:1]))


def exact_fixed_connection(coords: torch.Tensor,
                           fragments: list[list[int]]) -> float:
    """Held--Karp over fixed-orientation path components."""
    q = len(fragments)
    starts = [path[0] for path in fragments]
    ends = [path[-1] for path in fragments]
    distance = torch.cdist(coords[ends], coords[starts]).double().tolist()
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
                key = mask | 1 << nxt, nxt
                value = old + distance[last][nxt]
                if value < dp.get(key, float('inf')):
                    dp[key] = value
    if q == 1:
        return distance[0][0]
    full = (1 << q) - 1
    return min(dp[(full, last)] + distance[last][0] for last in range(1, q))


def exact_reversible_connection(coords: torch.Tensor,
                                fragments: list[list[int]]) -> float:
    """Held--Karp allowing every undirected fragment to be reversed."""
    q = len(fragments)
    starts = [[path[0], path[-1]] for path in fragments]
    ends = [[path[-1], path[0]] for path in fragments]

    def edge(a: int, oa: int, b: int, ob: int) -> float:
        return float(torch.dist(coords[ends[a][oa]], coords[starts[b][ob]]))

    # Fix component 0 in orientation 0: reversing the complete symmetric cycle
    # maps every solution with orientation 1 for component 0 to an equal one.
    dp: dict[tuple[int, int, int], float] = {(1, 0, 0): 0.0}
    for mask in range(1, 1 << q):
        if not mask & 1:
            continue
        for last in range(q):
            for orientation in range(2):
                old = dp.get((mask, last, orientation))
                if old is None:
                    continue
                for nxt in range(1, q):
                    if mask >> nxt & 1:
                        continue
                    for next_orientation in range(2):
                        key = mask | 1 << nxt, nxt, next_orientation
                        value = old + edge(last, orientation, nxt, next_orientation)
                        if value < dp.get(key, float('inf')):
                            dp[key] = value
    if q == 1:
        return edge(0, 0, 0, 0)
    full = (1 << q) - 1
    return min(dp[(full, last, orientation)] + edge(last, orientation, 0, 0)
               for last in range(1, q) for orientation in range(2))


def greedy_pair_connection(coords: torch.Tensor,
                           fragments: list[list[int]]) -> float:
    paths = [list(path) for path in fragments]
    added = 0.0
    while len(paths) > 1:
        _, left, right = min(
            (float(torch.dist(coords[a[-1]], coords[b[0]])), i, j)
            for i, a in enumerate(paths) for j, b in enumerate(paths) if i != j)
        added += float(torch.dist(coords[paths[left][-1]], coords[paths[right][0]]))
        merged = paths[left] + paths[right]
        paths = [path for index, path in enumerate(paths) if index not in (left, right)]
        paths.append(merged)
    return added + float(torch.dist(coords[paths[0][-1]], coords[paths[0][0]]))


def random_connection_costs(coords: torch.Tensor, fragments: list[list[int]],
                            samples: int, generator: torch.Generator) -> list[float]:
    values = []
    for _ in range(samples):
        order = torch.randperm(len(fragments), generator=generator).tolist()
        arranged = [fragments[index] for index in order]
        values.append(connection_cost(coords, arranged))
    return values


def forest_state(coords: torch.Tensor, fragments: list[list[int]]):
    n = len(coords)
    succ = torch.full((n,), -1, dtype=torch.long)
    pred = succ.clone()
    comp = torch.empty(n, dtype=torch.long)
    starts = torch.empty(n, dtype=torch.long)
    ends = torch.empty(n, dtype=torch.long)
    sizes = torch.empty(n)
    cost = 0.0
    for label, path in enumerate(fragments):
        nodes = torch.tensor(path, dtype=torch.long)
        comp[nodes] = label
        starts[nodes] = path[0]
        ends[nodes] = path[-1]
        sizes[nodes] = len(path)
        for tail, head in zip(path, path[1:]):
            succ[tail], pred[head] = head, tail
            cost += float(torch.dist(coords[tail], coords[head]))
    return dict(succ=succ, pred=pred, comp=comp, starts=starts, ends=ends,
                sizes=sizes, route_tail=fragments[0][-1],
                last_head=fragments[0][-1], edges=n - len(fragments), cost=cost)


@torch.no_grad()
def model_reconnect(model, coords: torch.Tensor, encoded: torch.Tensor,
                    initial, policy: str) -> float:
    state = clone_state(initial)
    while state['edges'] < len(coords):
        if policy == 'route':
            tail = state['route_tail']
            head = greedy_endpoint(model, coords, encoded, state, tail)
        elif policy == 'learned':
            rows = score_state(model, coords.to(encoded.device), encoded, state)
            choice = max(rows, key=lambda row: row['source_score'])
            tail, head = choice['tail'], choice['head']
        else:
            raise ValueError(policy)
        apply_edge(coords, state, tail, head)
    if component_count(state) != 1:
        raise AssertionError('reconnection did not finish with one component')
    return state['cost']


def normal_ci(values: torch.Tensor) -> list[float]:
    mean = values.mean()
    if len(values) < 2:
        return [float(mean), float(mean)]
    half = 1.96 * values.std(unbiased=True) / sqrt(len(values))
    return [float(mean - half), float(mean + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=48)
    parser.add_argument('--sizes', type=int, nargs='+', default=[50, 100])
    parser.add_argument('--components', type=int, nargs='+', default=[4, 8, 12])
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--random-samples', type=int, default=8)
    parser.add_argument('--max-2opt-moves', type=int, default=100)
    parser.add_argument('--destruction', choices=['random', 'longest'], default='random')
    parser.add_argument('--seed', type=int, default=2026100401)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output directory already exists')
    if any(2 * k > n for n in args.sizes for k in args.components):
        parser.error('every requested fragment must contain at least two nodes')
    args.out.mkdir(parents=True)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.22, device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    checkpoint = args.screen / 'learned-reinforce-seed1234' / 'best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SourceFirstASCC(**payload['config']['dimensions']).to(device).eval()
    model.load_state_dict(payload['model'], strict=True)
    records = []
    started = time.monotonic()
    for n in args.sizes:
        coords_all = torch.rand(args.count, n, 2,
                                generator=torch.Generator().manual_seed(args.seed + n))
        selected_tours = []
        action_rng = torch.Generator(device=device).manual_seed(args.seed + 10_000 + n)
        with torch.no_grad():
            for batch in coords_all.split(args.batch):
                rollout = model(batch.to(device), args.rollouts, 'route', 'greedy',
                                generator=action_rng, anchor_mode='multi')
                best = rollout.costs.argmin(1)
                for row in range(len(batch)):
                    index = int(best[row])
                    selected_tours.append(tour_from_actions(
                        rollout.tails[row, index], rollout.heads[row, index]))
        for instance, (coords, raw_tour) in enumerate(zip(coords_all, selected_tours)):
            encoded = model.encoder(coords[None].to(device)).squeeze(0)
            origins = {'raw': raw_tour,
                       'two_opt': two_opt(coords, raw_tour, args.max_2opt_moves)}
            for origin, tour in origins.items():
                baseline = route_cost(coords, tour)
                if origin == 'two_opt' and baseline > route_cost(coords, raw_tour) + 1e-7:
                    raise AssertionError('2-opt worsened the route')
                for k in args.components:
                    forest_rng = torch.Generator().manual_seed(
                        args.seed + n * 100_000 + instance * 101 + k)
                    fragments = split_fragments(tour, k, forest_rng,
                                                args.destruction, coords)
                    fixed = internal_cost(coords, fragments)
                    original = fixed + connection_cost(coords, fragments)
                    if abs(original - baseline) > 2e-5:
                        raise AssertionError('fragmentation changed the seed route cost')
                    state = forest_state(coords, fragments)
                    random_values = random_connection_costs(
                        coords, fragments, args.random_samples, forest_rng)
                    costs = {
                        'original_order': original,
                        'exact_fixed_orientation': fixed + exact_fixed_connection(coords, fragments),
                        'exact_reversible': fixed + exact_reversible_connection(coords, fragments),
                        'greedy_pair': fixed + greedy_pair_connection(coords, fragments),
                        'random_mean': fixed + sum(random_values) / len(random_values),
                        'random_best': fixed + min(random_values),
                        'model_route': model_reconnect(model, coords, encoded, state, 'route'),
                        'model_learned': model_reconnect(model, coords, encoded, state, 'learned'),
                    }
                    if costs['exact_fixed_orientation'] > original + 2e-5:
                        raise AssertionError('exact fixed oracle lost the original ordering')
                    if costs['exact_reversible'] > costs['exact_fixed_orientation'] + 2e-5:
                        raise AssertionError('reversible oracle worse than fixed oracle')
                    for method, cost in costs.items():
                        records.append(dict(
                            n=n, instance=instance, origin=origin,
                            destruction=args.destruction, fragments=k,
                            method=method, baseline_cost=baseline, cost=cost,
                            improvement_pct=100 * (baseline - cost) / baseline))
            print(json.dumps({'n': n, 'instance': instance + 1,
                              'count': args.count,
                              'elapsed_seconds': time.monotonic() - started}), flush=True)

    with (args.out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    groups = {}
    for row in records:
        key = row['n'], row['origin'], row['fragments'], row['method']
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, rows in sorted(groups.items()):
        improvement = torch.tensor([row['improvement_pct'] for row in rows],
                                   dtype=torch.double)
        costs = torch.tensor([row['cost'] for row in rows], dtype=torch.double)
        summary_rows.append(dict(
            n=key[0], origin=key[1], fragments=key[2], method=key[3],
            count=len(rows), mean_cost=float(costs.mean()),
            mean_improvement_pct=float(improvement.mean()),
            ci95_improvement_pct=normal_ci(improvement),
            improved_fraction=float((improvement > 1e-7).double().mean())))
    summary = {
        'role': 'genuine_multi_path_reconnection_diagnostic',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'coordinate_seeds': {str(n): args.seed + n for n in args.sizes},
        'fragment_seed_formula': 'seed + n*100000 + instance*101 + fragments',
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        'orientation': 'fixed for deployed policies; exact_reversible is diagnostic only',
        'summary': summary_rows,
        'elapsed_seconds': time.monotonic() - started,
        'coordinate_sha256': {str(n): tensor_sha(torch.rand(
            args.count, n, 2, generator=torch.Generator().manual_seed(args.seed + n)))
            for n in args.sizes},
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
