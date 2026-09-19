from pathlib import Path
import argparse
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path,
                    default=Path(__file__).resolve().parent / 'evidence' / 'oracle-regret-v1')
ROOT = parser.parse_args().root
POLICIES = ['learned', 'random_expected', 'route', 'min_entropy',
            'max_margin', 'shortest_edge', 'shortest_component', 'fixed']
LABELS = {'learned': 'Learned source', 'random_expected': 'Random source',
          'route': 'Continue route', 'min_entropy': 'Min endpoint entropy',
          'max_margin': 'Max endpoint margin', 'shortest_edge': 'Shortest proposed edge',
          'shortest_component': 'Shortest component', 'fixed': 'Fixed index'}
COLORS = {'learned': '#7c3aed', 'random_expected': '#64748b', 'route': '#2563eb',
          'min_entropy': '#059669', 'max_margin': '#d97706',
          'shortest_edge': '#dc2626', 'shortest_component': '#0891b2', 'fixed': '#94a3b8'}


with (ROOT / 'records.csv').open() as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    for key in ('n', 'instance', 'remaining', 'source_count'):
        row[key] = int(row[key])
    for key in ('oracle_completion', 'source_spread_pct', 'regret', 'regret_pct', 'oracle_hit'):
        row[key] = float(row[key])


def mean_se(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


# Main figure: one objective, readable scientific decision chart.
main = [r for r in rows if r['objective'] == 'reinforce' and r['origin'] == 'learned']
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

# A: opportunity (deduplicated across policy rows).
seen = {}
for r in main:
    seen[(r['n'], r['instance'], r['remaining'])] = r
for n, style in ((20, '-o'), (50, '-s')):
    means, ses = [], []
    for m in (4, 6, 8, 10):
        vals = [r['source_spread_pct'] for (nn, _, mm), r in seen.items() if nn == n and mm == m]
        mean, se = mean_se(vals); means.append(mean); ses.append(se)
    axes[0].errorbar((4, 6, 8, 10), means, yerr=np.asarray(ses) * 1.96,
                     fmt=style, capsize=3, label=f'TSP{n}')
axes[0].set(title='A. Source choice has nonzero opportunity', xlabel='Remaining path components',
            ylabel='Worst − best source\n(% of oracle completion cost)')
axes[0].legend(frameon=False)

# B/C: policy performance, averaged over n and remaining component levels.
x = np.arange(len(POLICIES))
regret_means, regret_ci, hit_means = [], [], []
clustered = {}
for policy in POLICIES:
    subset = [r for r in main if r['policy'] == policy]
    # Four construction stages from one TSP instance are repeated measurements.
    # Aggregate them before computing uncertainty across independent instances.
    units = defaultdict(list)
    hit_units = defaultdict(list)
    for r in subset:
        units[(r['n'], r['instance'])].append(r['regret_pct'])
        hit_units[(r['n'], r['instance'])].append(r['oracle_hit'])
    unit_values = {key: np.mean(value) for key, value in units.items()}
    mean, se = mean_se(list(unit_values.values()))
    regret_means.append(mean); regret_ci.append(1.96 * se)
    hit_means.append(100 * np.mean([np.mean(value) for value in hit_units.values()]))
    clustered[policy] = unit_values
axes[1].bar(x, regret_means, yerr=regret_ci, color=[COLORS[p] for p in POLICIES], capsize=3)
axes[1].set(title='B. Current source policies leave regret', ylabel='Excess over source oracle (%)')
axes[2].bar(x, hit_means, color=[COLORS[p] for p in POLICIES])
axes[2].set(title='C. How often each policy selects an oracle source', ylabel='Oracle-source hit rate (%)')
for ax in axes[1:]:
    ax.set_xticks(x, [LABELS[p] for p in POLICIES], rotation=55, ha='right')
for ax in axes:
    ax.grid(axis='y', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Exact source-regret diagnostic — learned-trajectory forest states, REINFORCE checkpoint',
             fontsize=12)
fig.savefig(ROOT / 'source_regret_main.png', dpi=190)
fig.savefig(ROOT / 'source_regret_main.pdf')


# Robustness heatmap: learned minus random. Negative means learned is better.
objectives, origins, sizes, components = ['reinforce', 'bopo'], ['learned', 'route'], [20, 50], [4, 6, 8, 10]
matrix = np.zeros((len(objectives) * len(origins), len(sizes) * len(components)))
for i, objective in enumerate(objectives):
    for j, origin in enumerate(origins):
        row_index = i * len(origins) + j
        for k, n in enumerate(sizes):
            for z, m in enumerate(components):
                cell = [r for r in rows if r['objective'] == objective and r['origin'] == origin
                        and r['n'] == n and r['remaining'] == m]
                learned = np.mean([r['regret_pct'] for r in cell if r['policy'] == 'learned'])
                random = np.mean([r['regret_pct'] for r in cell if r['policy'] == 'random_expected'])
                matrix[row_index, k * len(components) + z] = learned - random
fig2, ax = plt.subplots(figsize=(11.5, 3.8), constrained_layout=True)
limit = max(abs(matrix.min()), abs(matrix.max()), .01)
image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f'{matrix[i, j]:+.2f}', ha='center', va='center', fontsize=9,
                color='white' if abs(matrix[i, j]) > .55 * limit else 'black')
ax.set_yticks(range(4), [f'{o.upper()} / {g} prefix' for o in objectives for g in origins])
ax.set_xticks(range(8), [f'TSP{n}\nm={m}' for n in sizes for m in components])
ax.set_title('Learned-source regret minus random-source expected regret (percentage points)')
fig2.colorbar(image, ax=ax, label='Negative favors learned')
fig2.savefig(ROOT / 'learned_vs_random_heatmap.png', dpi=190)
fig2.savefig(ROOT / 'learned_vs_random_heatmap.pdf')


report = {
    'main_scope': 'reinforce checkpoint, learned-prefix states, pooled across TSP20/TSP50 and m=4/6/8/10',
    'mean_source_opportunity_pct': float(np.mean([r['source_spread_pct'] for r in seen.values()])),
    'policies': {p: {'mean_regret_pct': float(regret_means[i]),
                     'ci95_halfwidth': float(regret_ci[i]),
                     'oracle_hit_rate': float(hit_means[i] / 100)}
                 for i, p in enumerate(POLICIES)},
    'paired_learned_minus_comparator': {},
    'learned_minus_random_range_pp': [float(matrix.min()), float(matrix.max())],
}
for policy in POLICIES[1:]:
    differences = [clustered['learned'][key] - clustered[policy][key]
                   for key in clustered['learned']]
    mean, se = mean_se(differences)
    report['paired_learned_minus_comparator'][policy] = {
        'mean_percentage_points': float(mean),
        'conditional_normal_ci95': [float(mean - 1.96 * se), float(mean + 1.96 * se)],
        'independent_instance_units': len(differences),
    }
(ROOT / 'figure_summary.json').write_text(json.dumps(report, indent=2))
print(ROOT / 'source_regret_main.png')
