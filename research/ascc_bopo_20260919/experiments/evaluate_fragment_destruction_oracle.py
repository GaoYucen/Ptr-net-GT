"""Best-of-many destruction oracle for path-fragment reconstruction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
import time

import torch

from evaluate_fragment_reconnection import (
    exact_fixed_connection, exact_reversible_connection, internal_cost,
    route_cost, split_fragments, tour_from_actions, two_opt,
)
from groupopt.models.ascc_bopo import SourceFirstASCC


def ci95(values: torch.Tensor):
    mean = values.mean()
    half = (1.96 * values.std(unbiased=True) / sqrt(len(values))
            if len(values) > 1 else torch.tensor(0.0))
    return [float(mean - half), float(mean + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=48)
    parser.add_argument('--sizes', type=int, nargs='+', default=[50, 100])
    parser.add_argument('--components', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--cut-samples', type=int, default=64)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--max-2opt-moves', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026100401)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output directory already exists')
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
            origins = {'raw': raw_tour,
                       'two_opt': two_opt(coords, raw_tour, args.max_2opt_moves)}
            for origin, tour in origins.items():
                baseline = route_cost(coords, tour)
                for k in args.components:
                    fixed_costs, reversible_costs = [], []
                    for sample in range(args.cut_samples):
                        rng = torch.Generator().manual_seed(
                            args.seed + n * 10_000_000 + instance * 100_003
                            + k * 1000 + sample)
                        fragments = split_fragments(tour, k, rng, 'random', coords)
                        inside = internal_cost(coords, fragments)
                        fixed_costs.append(inside + exact_fixed_connection(coords, fragments))
                        reversible_costs.append(
                            inside + exact_reversible_connection(coords, fragments))
                    longest = split_fragments(
                        tour, k, torch.Generator().manual_seed(0), 'longest', coords)
                    inside = internal_cost(coords, longest)
                    longest_fixed = inside + exact_fixed_connection(coords, longest)
                    longest_reversible = inside + exact_reversible_connection(coords, longest)
                    methods = {
                        'single_random_fixed_mean': sum(fixed_costs) / len(fixed_costs),
                        'best_random_fixed': min(fixed_costs),
                        'best_random_reversible': min(reversible_costs),
                        'longest_fixed': longest_fixed,
                        'longest_reversible': longest_reversible,
                        'best_combined_reversible': min(min(reversible_costs),
                                                        longest_reversible),
                    }
                    for method, cost in methods.items():
                        records.append(dict(
                            n=n, instance=instance, origin=origin, fragments=k,
                            cut_samples=args.cut_samples, method=method,
                            baseline_cost=baseline, cost=cost,
                            improvement_pct=100 * (baseline - cost) / baseline))
            print(json.dumps({'n': n, 'instance': instance + 1,
                              'count': args.count,
                              'elapsed_seconds': time.monotonic() - started}), flush=True)

    with (args.out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    groups = {}
    for row in records:
        key = row['n'], row['origin'], row['fragments'], row['method']
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, rows in sorted(groups.items()):
        values = torch.tensor([row['improvement_pct'] for row in rows],
                              dtype=torch.double)
        summary_rows.append(dict(
            n=key[0], origin=key[1], fragments=key[2], method=key[3],
            count=len(rows), mean_improvement_pct=float(values.mean()),
            ci95_improvement_pct=ci95(values),
            improved_fraction_at_0_01pct=float((values > .01).double().mean())))
    summary = {
        'role': 'best_of_many_fragment_destruction_oracle',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        'summary': summary_rows,
        'elapsed_seconds': time.monotonic() - started,
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
