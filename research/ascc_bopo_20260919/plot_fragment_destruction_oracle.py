import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
rows = json.loads((args.root / 'summary.json').read_text())['summary']

methods = [
    ('longest_reversible', 'Longest-edge cut', '#64748b'),
    ('best_random_fixed', 'Best of 64 cuts, fixed direction', '#2563eb'),
    ('best_random_reversible', 'Best of 64 cuts, reversible', '#16a34a'),
]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True,
                         constrained_layout=True)
for axis, origin in zip(axes, ('raw', 'two_opt')):
    x, labels = [], []
    for n in (50, 100):
        for fragments in (4, 8):
            x.append(len(x))
            labels.append(f'TSP{n}\n{fragments} paths')
    width = .24
    for offset, (method, label, color) in enumerate(methods):
        values = []
        for n in (50, 100):
            for fragments in (4, 8):
                row = next(r for r in rows if r['origin'] == origin and r['n'] == n
                           and r['fragments'] == fragments and r['method'] == method)
                values.append(row['mean_improvement_pct'])
        axis.bar([v + (offset - 1) * width for v in x], values, width,
                 label=label, color=color)
    axis.axhline(0, color='#0f172a', linewidth=.8)
    axis.set_xticks(x, labels)
    axis.set(title='Raw model route' if origin == 'raw' else '2-opt seed route',
             ylabel='Improvement over seed tour (%)')
axes[0].legend(frameon=True, fontsize=8)
fig.suptitle('Does selecting the destruction unlock path-forest headroom?', fontsize=13)
fig.savefig(args.root / 'fragment_destruction_oracle.png', dpi=180)
fig.savefig(args.root / 'fragment_destruction_oracle.pdf')
