from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
args = p.parse_args()
base = json.loads((args.root / 'summary.json').read_text())
tabular = json.loads((args.root / 'tabular_summary.json').read_text())
route = next(row for row in base['test_results'] if row['method'] == 'route')
route_cost = route['mean_cost']
lookup = {row['method']: row for row in base['test_results']}

items = [
    ('Rollout-value oracle', lookup['oracle'], '#15803d'),
    ('Original learned source', lookup['pretrained_learned'], '#7c3aed'),
    ('Min endpoint entropy', lookup['min_entropy'], '#059669'),
    ('Shortest proposed edge', lookup['shortest_edge'], '#dc2626'),
    ('MLP rollout-value model', lookup['selected_value_model'], '#2563eb'),
    ('Tabular value model', tabular, '#0891b2'),
]
means, lower, upper = [], [], []
for _, row, _ in items:
    mean = row['relative_improvement_percent']
    lo, hi = row['conditional_normal_ci95']
    lo_pct, hi_pct = 100 * lo / route_cost, 100 * hi / route_cost
    means.append(mean); lower.append(mean - lo_pct); upper.append(hi_pct - mean)

fig, ax = plt.subplots(figsize=(10.8, 5.1), constrained_layout=True)
x = range(len(items))
ax.bar(x, means, yerr=[lower, upper], color=[row[2] for row in items], capsize=4)
ax.axhline(0, color='black', linewidth=.9)
ax.set_xticks(list(x), [row[0] for row in items], rotation=28, ha='right')
ax.set(ylabel='TSP50 improvement over route (%)\npositive is better',
       title='The source freedom has value, but current learned selectors do not capture it')
ax.grid(axis='y', alpha=.22); ax.spines[['top', 'right']].set_visible(False)
ax.annotate('0.845% oracle headroom', (0, means[0]), xytext=(8, 10),
            textcoords='offset points', fontsize=9)
out = args.root / 'value_selector_result.png'
fig.savefig(out, dpi=190)
fig.savefig(args.root / 'value_selector_result.pdf')
print(out)
