from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--oracle-root', type=Path, required=True)
args = parser.parse_args()
summary = json.loads((args.root / 'summary.json').read_text())
oracle = json.loads((args.oracle_root / 'figure_summary.json').read_text())

policies = ['learned', 'random', 'route', 'min_entropy', 'max_margin',
            'shortest_edge', 'shortest', 'fixed']
labels = {'learned': 'Learned', 'random': 'Random', 'route': 'Route',
          'min_entropy': 'Min entropy', 'max_margin': 'Max margin',
          'shortest_edge': 'Shortest edge', 'shortest': 'Shortest component',
          'fixed': 'Fixed index'}
colors = {'learned': '#7c3aed', 'random': '#64748b', 'route': '#2563eb',
          'min_entropy': '#059669', 'max_margin': '#d97706',
          'shortest_edge': '#dc2626', 'shortest': '#0891b2', 'fixed': '#94a3b8'}


def load(n, policy):
    if policy == 'random':
        values = [torch.load(args.root / f'tsp{n}-random-{seed}-costs.pt', weights_only=True)
                  for seed in (7001, 7002, 7003)]
        return torch.stack(values).mean(0).double()
    return torch.load(args.root / f'tsp{n}-{policy}-0-costs.pt', weights_only=True).double()


fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
improvements = {}
for axis, n in zip(axes[:2], (20, 50)):
    route = load(n, 'route')
    means, errors = [], []
    for policy in policies:
        value = load(n, policy)
        delta_pct = 100 * (route - value) / route.mean()
        means.append(float(delta_pct.mean()))
        errors.append(float(1.96 * delta_pct.std(unbiased=True) / len(delta_pct) ** .5))
        improvements[(n, policy)] = float(delta_pct.mean())
    x = np.arange(len(policies))
    axis.bar(x, means, yerr=errors, color=[colors[p] for p in policies], capsize=3)
    axis.axhline(0, color='black', linewidth=.8)
    axis.set_xticks(x, [labels[p] for p in policies], rotation=55, ha='right')
    axis.set(title=f'TSP{n}: full-tour outcome', ylabel='Improvement over route (%)\npositive is better')

# Local one-step regret versus repeated full construction.
local_key = {'learned': 'learned', 'random': 'random_expected', 'route': 'route',
             'min_entropy': 'min_entropy', 'max_margin': 'max_margin',
             'shortest_edge': 'shortest_edge', 'shortest': 'shortest_component', 'fixed': 'fixed'}
for policy in policies:
    x = oracle['policies'][local_key[policy]]['mean_regret_pct']
    y = np.mean([improvements[(20, policy)], improvements[(50, policy)]])
    axes[2].scatter(x, y, s=65, color=colors[policy])
    axes[2].annotate(labels[policy], (x, y), xytext=(4, 3), textcoords='offset points', fontsize=8)
axes[2].axhline(0, color='black', linewidth=.8)
axes[2].set(title='Good one-step choices need not yield good tours',
            xlabel='Exact one-step source regret (%)\nlower is better',
            ylabel='Mean full-tour improvement over route (%)')
for axis in axes:
    axis.grid(axis='y', alpha=.22)
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('End-to-end source-order intervention — REINFORCE checkpoint, 128 fixed instances/size',
             fontsize=12)
fig.savefig(args.root / 'heuristic_rollout_main.png', dpi=190)
fig.savefig(args.root / 'heuristic_rollout_main.pdf')

output = {'relative_improvement_over_route_percent': {
    str(n): {p: improvements[(n, p)] for p in policies} for n in (20, 50)}}
(args.root / 'figure_summary.json').write_text(json.dumps(output, indent=2))
print(args.root / 'heuristic_rollout_main.png')
