from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


p = argparse.ArgumentParser()
p.add_argument('--oracle-root', type=Path, required=True)
p.add_argument('--sweep-root', type=Path, required=True)
p.add_argument('--confirm-root', type=Path, required=True)
args = p.parse_args()


def read(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('n', 'instance'):
            if key in row: row[key] = int(row[key])
        if 'stage' in row: row['stage'] = int(row['stage'])
        row['cost'] = float(row['cost'])
    return rows


oracle = read(args.oracle_root / 'records.csv')
sweep = read(args.sweep_root / 'records.csv')
confirm = read(args.confirm_root / 'records.csv')


def paired_improvement(rows, method, *, n=None, stage=None):
    subset = [r for r in rows if (n is None or r.get('n') == n)
              and (stage is None or r.get('stage') == stage)]
    route = {r['instance']: r['cost'] for r in subset if r['method'] == 'route'}
    value = {r['instance']: r['cost'] for r in subset if r['method'] == method}
    scale = np.mean(list(route.values()))
    delta = np.array([100 * (route[i] - value[i]) / scale for i in sorted(route)])
    return delta.mean(), 1.96 * delta.std(ddof=1) / np.sqrt(len(delta)), delta


fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
methods = ['source_once', 'source_repeated', 'endpoint_once', 'endpoint_repeated']
labels = ['Source\nonce', 'Source\nrepeated', 'Endpoint\nonce', 'Endpoint\nrepeated']
colors = ['#7c3aed', '#4c1d95', '#059669', '#065f46']
x = np.arange(len(methods)); width = .36
for offset, n, hatch in ((-.5, 20, ''), (.5, 50, '//')):
    means, errors = [], []
    for method in methods:
        mean, error, _ = paired_improvement(oracle, method, n=n)
        means.append(mean); errors.append(error)
    axes[0].bar(x + offset * width, means, width, yerr=errors, capsize=3,
                color=colors, hatch=hatch, edgecolor='white', label=f'TSP{n}')
axes[0].axhline(0, color='black', linewidth=.8)
axes[0].set_xticks(x, labels)
axes[0].set(title='A. Perfect rollout values can improve the final tour',
            ylabel='Improvement over route (%)')
axes[0].legend(frameon=False)

stages = [4, 6, 8, 10, 15, 20, 30]
for method, label, color, marker in (
        ('source_once', 'Source value oracle', '#7c3aed', 'o'),
        ('endpoint_once', 'Endpoint value oracle', '#059669', 's')):
    means, errors = [], []
    for stage in stages:
        rows = confirm if stage in (20, 30) else sweep
        mean, error, _ = paired_improvement(rows, method, stage=stage)
        means.append(mean); errors.append(error)
    axes[1].errorbar(stages, means, yerr=errors, marker=marker, capsize=3,
                     color=color, label=label)
axes[1].axhline(0, color='black', linewidth=.8)
axes[1].set(title='B. Earlier source decisions contain more value',
            xlabel='Remaining path components at intervention',
            ylabel='TSP50 improvement over route (%)')
axes[1].legend(frameon=False)

_, _, source_delta = paired_improvement(confirm, 'source_once', stage=30)
_, _, endpoint_delta = paired_improvement(confirm, 'endpoint_once', stage=30)
axes[2].plot(np.sort(source_delta), np.linspace(0, 1, len(source_delta), endpoint=True),
             color='#7c3aed', label=f'Source ({np.mean(source_delta > 1e-8):.0%} improved)')
axes[2].plot(np.sort(endpoint_delta), np.linspace(0, 1, len(endpoint_delta), endpoint=True),
             color='#059669', label=f'Endpoint ({np.mean(endpoint_delta > 1e-8):.0%} improved)')
axes[2].axvline(0, color='black', linewidth=.8)
axes[2].set(title='C. TSP50, one intervention at 30 components',
            xlabel='Per-instance improvement over route (%)', ylabel='Empirical CDF')
axes[2].legend(frameon=False)

for ax in axes:
    ax.grid(axis='y', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Rollout-value oracle: upper bound for a perfectly learned decision value', fontsize=12)
out = args.confirm_root / 'rollout_value_oracle.png'
fig.savefig(out, dpi=190)
fig.savefig(args.confirm_root / 'rollout_value_oracle.pdf')
print(out)
