"""Aggregate raw Pair-ASCC runs without treating instances as training repeats."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch


def mean(values):
    return statistics.fmean(values) if values else float('nan')


def paired_interval(deltas):
    if len(deltas) < 2:
        return [float('nan'), float('nan')]
    # Exact t critical values for the planned n=10; otherwise normal fallback.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262}.get(len(deltas), 1.96)
    spread = statistics.stdev(deltas) / math.sqrt(len(deltas))
    center = mean(deltas)
    return [center - critical * spread, center + critical * spread]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True,
                        help='directory containing <method>-seed<seed>/summary.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', default='single_opt')
    parser.add_argument('--noninferiority-percent', type=float, default=.10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(dict)
    all_rows = []
    for summary_path in sorted(args.runs.glob('*/summary.json')):
        payload = json.loads(summary_path.read_text())
        config = json.loads((summary_path.parent / 'config.json').read_text())
        method, seed = payload.get('method', config.get('method')), int(config['seed'])
        if method is None:
            raise ValueError(f'missing method: {summary_path}')
        record = dict(method=method, seed=seed,
                      best_test=payload['best_test']['mean_cost'],
                      final_test=payload['final_test']['mean_cost'],
                      train_seconds=payload['train_seconds'],
                      elapsed_seconds=payload['elapsed_seconds'],
                      best_step=payload['best_step'])
        groups[method][seed] = record
        all_rows.append(record)
    with (args.out / 'quality_by_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]) if all_rows else [])
        if all_rows:
            writer.writeheader(); writer.writerows(all_rows)
    baseline = groups.get(args.baseline, {})
    comparisons = {}
    for method, values in groups.items():
        if method == args.baseline:
            continue
        shared = sorted(set(values) & set(baseline))
        deltas = [values[s]['best_test'] - baseline[s]['best_test'] for s in shared]
        relative = [100 * d / baseline[s]['best_test'] for d, s in zip(deltas, shared)]
        interval = paired_interval(relative)
        comparisons[method] = dict(baseline=args.baseline, shared_seeds=shared,
                                   paired_relative_percent=relative,
                                   mean_relative_percent=mean(relative),
                                   ci95_relative_percent=interval,
                                   noninferior=bool(interval[1] <= args.noninferiority_percent))
    # Per-instance bootstrap is saved separately; this summary intentionally uses
    # seed-level repeats for the primary claim.
    (args.out / 'paired_statistics.json').write_text(json.dumps(dict(
        baseline=args.baseline, noninferiority_percent=args.noninferiority_percent,
        methods={method: dict(seeds=sorted(rows),
                              mean_best_test=mean([x['best_test'] for x in rows.values()]),
                              sd_best_test=(statistics.stdev([x['best_test'] for x in rows.values()])
                                            if len(rows) > 1 else 0.0),
                              mean_train_seconds=mean([x['train_seconds'] for x in rows.values()]))
                 for method, rows in groups.items()}, comparisons=comparisons), indent=2))


if __name__ == '__main__':
    main()
