"""Exact conditional aliasing on six-component forests from larger TSPs.

The suffix-swap partner is a constructed counterfactual, NOT an on-policy sample.
Only a selected-source conditional policy has the information-theoretic bound.
Run: python counterfactual_stress.py --output counterfactual_stress
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np


def distances(x):
    return np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1))


def nn_twoopt(d, max_moves=100):
    n = len(d)
    tour = [0]
    used = np.zeros(n, bool)
    used[0] = True
    for _ in range(n - 1):
        j = int(np.where(used, np.inf, d[tour[-1]]).argmin())
        tour.append(j)
        used[j] = True
    tour = np.array(tour)
    # Reverse the segment between two nonadjacent edges; leave vertex zero fixed.
    legal = np.triu(np.ones((n, n), bool), 2)
    legal[0, n - 1] = False
    for _ in range(max_moves):
        nxt = np.roll(tour, -1)
        delta = (d[tour[:, None], tour[None, :]] + d[nxt[:, None], nxt[None, :]]
                 - d[tour, nxt][:, None] - d[tour, nxt][None, :])
        delta[~legal] = np.inf
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] >= -1e-12:
            break
        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
    return tour


def split_paths(tour, rng, m=6):
    # Uniform m-1 cut positions on the linearized tour, with singleton paths
    # allowed. Resample only if no two non-source paths can support suffix swaps.
    while True:
        cuts = sorted(rng.choice(np.arange(1, len(tour)), m - 1, replace=False))
        paths = [a.tolist() for a in np.split(tour, cuts)]
        eligible = [i for i in range(1, m) if len(paths[i]) >= 2]
        if len(eligible) >= 2:
            return paths, eligible


def swap_suffixes(paths, eligible, rng):
    i, j = rng.choice(eligible, 2, replace=False)
    a = int(rng.integers(1, len(paths[i])))
    b = int(rng.integers(1, len(paths[j])))
    result = [p.copy() for p in paths]
    result[i] = paths[i][:a] + paths[j][b:]
    result[j] = paths[j][:b] + paths[i][a:]
    return result, dict(components=[int(i), int(j)], cuts=[a, b])


def exact_q(d, paths, permutations):
    h = np.array([p[0] for p in paths])
    t = np.array([p[-1] for p in paths])
    w = d[t[:, None], h[None, :]]
    costs = w[permutations, np.roll(permutations, -1, axis=1)].sum(1)
    q = np.array([costs[permutations[:, 1] == j].min() for j in range(1, len(paths))])
    internal = sum(float(d[p[:-1], p[1:]].sum()) for p in paths)
    return q + internal


def assert_pair(paths, partner, n):
    assert sorted(sum(paths, [])) == list(range(n))
    assert sorted(sum(partner, [])) == list(range(n))
    assert paths[0] == partner[0]
    assert [p[0] for p in paths] == [p[0] for p in partner]
    assert sorted(p[-1] for p in paths) == sorted(p[-1] for p in partner)


def bootstrap_mean(a, rng):
    a = np.asarray(a)
    bs = a[rng.integers(len(a), size=(4000, len(a)))].mean(1)
    return [float(v) for v in np.quantile(bs, [.025, .975])]


def old_witness(root):
    src = root.parent / 'strong_hosts_20260919/evidence/endpoint-aliasing.json'
    witness = json.loads(src.read_text())
    x = np.array(witness['coordinates'])
    d = distances(x)
    n = len(x)
    q_all = []
    for edges in witness['partial_edges']:
        q = np.full(n, np.inf)
        completions = 0
        for p in itertools.permutations(range(1, n)):
            tour = np.array((0,) + p)
            succ = np.empty(n, int)
            succ[tour] = np.roll(tour, -1)
            if all(succ[a] == b for a, b in edges):
                cost = d[tour, np.roll(tour, -1)].sum()
                q[succ[0]] = min(q[succ[0]], cost)
                completions += 1
        assert completions == 24
        old = witness['exact_best_completion_by_head'][len(q_all)]
        for j, v in enumerate(old):
            if v is not None:
                assert abs(q[j] - v) < 1e-12
        q_all.append(q)
    q_all = np.array(q_all)
    legal = np.isfinite(q_all).all(0)
    regret = q_all[:, legal] - q_all[:, legal].min(1, keepdims=True)
    avg = regret.mean(0)
    return dict(source_file=str(src.relative_to(root.parent)), verified_all_720_tours=True,
                valid_completions_per_forest=24, legal_heads=np.where(legal)[0].tolist(),
                completion_q=q_all[:, legal].tolist(), regrets=regret.tolist(),
                optimal_heads=np.where(legal)[0][regret.argmin(1)].tolist(),
                blind_best_shared_head=int(np.where(legal)[0][avg.argmin()]),
                exact_equal_prior_bayes_regret=float(avg.min()))


def run(a):
    root = Path(__file__).resolve().parent
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    perms = np.array([(0,) + p for p in itertools.permutations(range(1, 6))])
    start = time.perf_counter()
    cells = []
    for dist_id, distribution in enumerate(('uniform', 'clustered')):
        for n in (50, 100, 200):
            seed = 91972000 + 10 * n + dist_id
            rng = np.random.default_rng(seed)
            records, coordinates = [], []
            for k in range(a.instances):
                if distribution == 'uniform':
                    x = rng.random((n, 2))
                else:
                    centers = .15 + .7 * rng.random((4, 2))
                    x = np.clip(centers[rng.integers(4, size=n)] + .04 * rng.normal(size=(n, 2)), 0, 1)
                d = distances(x)
                tour = nn_twoopt(d)
                paths, eligible = split_paths(tour, rng)
                partner, swap = swap_suffixes(paths, eligible, rng)
                assert_pair(paths, partner, n)
                qa = exact_q(d, paths, perms)
                qb = exact_q(d, partner, perms)
                q = np.stack((qa, qb))
                regrets = q - q.min(1, keepdims=True)
                alias_floor = float(regrets.mean(0).min())
                source = paths[0][-1]
                head_nodes = [p[0] for p in paths[1:]]
                nearest = int(d[source, head_nodes].argmin())
                # Identical-pair control must have zero irreducible loss.
                control = float((np.stack((qa, qa)) - qa.min()).mean(0).min())
                assert abs(control) < 1e-12
                records.append(dict(index=k, paths=paths, partner_paths=partner,
                                    swap=swap, conditional_q=q.tolist(),
                                    alias_floor=alias_floor,
                                    disjoint_optima=bool(alias_floor > 1e-9),
                                    nearest_mean_regret=float(regrets[:, nearest].mean()),
                                    original_tour_cost=float(d[tour, np.roll(tour, -1)].sum()),
                                    control_floor=control))
                coordinates.append(x)
            coords = np.stack(coordinates)
            tag = f'{distribution}{n}'
            np.savez_compressed(out / f'{tag}-coordinates.npz', coordinates=coords)
            (out / f'{tag}-records.json').write_text(json.dumps(records))
            floors = np.array([r['alias_floor'] for r in records])
            frac = (floors > 1e-9).astype(float)
            stat_rng = np.random.default_rng(seed + 900000)
            cell = dict(distribution=distribution, n=n, m=6, instances=a.instances, seed=seed,
                        coordinates_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
                        mean_bayes_regret=float(floors.mean()),
                        bayes_regret_ci=bootstrap_mean(floors, stat_rng),
                        conflict_fraction=float(frac.mean()),
                        conflict_fraction_ci=bootstrap_mean(frac, stat_rng),
                        nearest_mean_regret=float(np.mean([r['nearest_mean_regret'] for r in records])),
                        max_bayes_regret=float(floors.max()),
                        mean_original_tour_cost=float(np.mean([r['original_tour_cost'] for r in records])),
                        unchanged_pair_control_max=0.)
            cells.append(cell)
            print(json.dumps(cell), flush=True)
    summary = dict(kind='constructed paired-forest counterfactual stress test',
                   not_on_policy=True, fixed_source=True, equal_state_prior=True,
                   protocol='iclr_20260919/PROTOCOL.md', cells=cells,
                   old_witness=old_witness(root), seconds=time.perf_counter() - start,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(done=True, seconds=summary['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='iclr_20260919/counterfactual_stress')
    p.add_argument('--instances', type=int, default=256)
    run(p.parse_args())
