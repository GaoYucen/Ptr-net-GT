import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()

with (args.root / 'records.csv').open() as handle:
    records = list(csv.DictReader(handle))
for row in records:
    row['n'] = int(row['n'])
    row['fragments'] = int(row['fragments'])
    row['improvement_pct'] = float(row['improvement_pct'])


def means(origin, n, method):
    result = []
    for fragments in sorted({row['fragments'] for row in records}):
        values = [row['improvement_pct'] for row in records
                  if row['origin'] == origin and row['n'] == n
                  and row['fragments'] == fragments and row['method'] == method]
        result.append((fragments, sum(values) / len(values)))
    return result


plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
styles = {
    ('raw', 50): ('TSP50, raw route', '#2563eb', 'o', '-'),
    ('raw', 100): ('TSP100, raw route', '#7c3aed', 's', '-'),
    ('two_opt', 50): ('TSP50, 2-opt route', '#059669', 'o', '--'),
    ('two_opt', 100): ('TSP100, 2-opt route', '#d97706', 's', '--'),
}
for setting, (label, color, marker, linestyle) in styles.items():
    origin, n = setting
    values = means(origin, n, 'exact_fixed_orientation')
    axes[0].plot(*zip(*values), label=label, color=color, marker=marker,
                 linestyle=linestyle, linewidth=2)
    values = means(origin, n, 'exact_reversible')
    axes[1].plot(*zip(*values), label=label, color=color, marker=marker,
                 linestyle=linestyle, linewidth=2)
for axis, title in zip(axes, ('A. Fixed directed fragments',
                              'B. Oracle may reverse fragments')):
    axis.axhline(0, color='#475569', linewidth=.8)
    axis.set(title=title, xlabel='Number of path fragments',
             ylabel='Improvement over seed tour (%)')
    axis.set_xticks(sorted({row['fragments'] for row in records}))
axes[0].legend(frameon=True, fontsize=8)
fig.suptitle('Exact reconnection headroom in genuine path forests', fontsize=13)
fig.savefig(args.root / 'fragment_oracle_headroom.png', dpi=180)
fig.savefig(args.root / 'fragment_oracle_headroom.pdf')


fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True,
                         constrained_layout=True)
method_styles = {
    'greedy_pair': ('Greedy global pair', '#0891b2', '^'),
    'model_learned': ('Current learned ASCC', '#dc2626', 'o'),
    'model_route': ('Route continuation', '#64748b', 's'),
    'exact_fixed_orientation': ('Exact fixed oracle', '#15803d', 'D'),
}
for axis, (origin, n) in zip(axes.flat,
                             [('raw', 50), ('raw', 100),
                              ('two_opt', 50), ('two_opt', 100)]):
    for method, (label, color, marker) in method_styles.items():
        values = means(origin, n, method)
        axis.plot(*zip(*values), label=label, color=color, marker=marker,
                  linewidth=2)
    axis.axhline(0, color='#0f172a', linewidth=.8)
    axis.set(title=f'TSP{n}, {"raw model route" if origin == "raw" else "2-opt seed route"}',
             ylabel='Improvement over seed tour (%)')
    axis.set_xticks(sorted({row['fragments'] for row in records}))
for axis in axes[-1]:
    axis.set_xlabel('Number of path fragments')
axes[0, 0].legend(frameon=True, fontsize=8)
fig.suptitle('Can current policies exploit genuine path-forest freedom?', fontsize=13)
fig.savefig(args.root / 'fragment_policy_transfer.png', dpi=180)
fig.savefig(args.root / 'fragment_policy_transfer.pdf')
