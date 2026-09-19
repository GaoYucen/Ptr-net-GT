"""Build paper assets from stored metrics; no hand-entered experimental numbers."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
P = R / 'paper'
F = P / 'figures'
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelsize': 10, 'legend.fontsize': 8, 'figure.dpi': 140,
                     'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans'})
colors = ['#275D8C', '#D77435', '#2B8C80']
stress = json.loads((R / 'counterfactual_stress/summary.json').read_text())
hist = json.loads((R / 'historical_recomputed/summary.json').read_text())

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.45))
for i, dist in enumerate(['uniform', 'clustered']):
    cells = [c for c in stress['cells'] if c['distribution'] == dist]
    x = np.arange(3) + (i - .5) * .30
    for j, metric, ci_key, scale in [(0, 'conflict_fraction', 'conflict_fraction_ci', 100),
                                     (1, 'mean_bayes_regret', 'bayes_regret_ci', 1)]:
        y = np.array([c[metric] for c in cells]) * scale
        ci = np.array([c[ci_key] for c in cells]).T * scale
        ax[j].bar(x, y, width=.27, color=colors[i], label=dist.capitalize())
        ax[j].errorbar(x, y, yerr=np.stack([y-ci[0], ci[1]-y]), fmt='none', color='#222', capsize=3, lw=1)
for a in ax:
    a.set_xticks(range(3), ['50', '100', '200'])
    a.set_xlabel('Original instance size N')
    a.grid(axis='y', alpha=.2)
    a.set_axisbelow(True)
ax[0].set_ylabel('Pairs with conflicting optima (%)')
ax[1].set_ylabel('Minimum shared-action regret')
ax[0].legend(frameon=False, loc='upper left')
fig.tight_layout(w_pad=1.5)
fig.savefig(F / 'stress.pdf')
fig.savefig(F / 'stress.png', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.45), sharey=False)
for h, a in zip(['am', 'icam'], ax):
    rows = [next(r for r in hist['rows'] if r['host'] == h and r['case'] == case)
            for case in ['test200', 'cluster200', 'test500']]
    for j, mode in enumerate(['route', 'random', 'learned']):
        y = [r['means'][mode] for r in rows]
        a.bar(np.arange(3) + (j - 1) * .24, y, width=.22, label=mode.capitalize(), color=colors[j])
    a.set_xticks(range(3), ['U200', 'C200', 'U500'])
    a.set_title(h.upper(), fontsize=11)
    a.set_ylabel('Mean complete-tour cost')
    a.grid(axis='y', alpha=.2)
    a.set_axisbelow(True)
ax[0].legend(frameon=False, fontsize=7, ncol=3, loc='upper left')
fig.tight_layout(w_pad=1.5)
fig.savefig(F / 'historical.pdf')
fig.savefig(F / 'historical.png', dpi=180)
plt.close(fig)

old = json.loads((R.parent / 'strong_hosts_20260919/evidence/endpoint-aliasing.json').read_text())
x = np.array(old['coordinates'])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={'width_ratios':[1,1,1.25]})
for i, a in enumerate(ax[:2]):
    a.scatter(x[:,0], x[:,1], color='#6D7781', s=25, zorder=3)
    for j, z in enumerate(x):
        a.annotate(str(j), z, xytext=(4, 4), textcoords='offset points', fontsize=9)
    for u, v in old['partial_edges'][i]:
        a.annotate('', x[v], x[u], arrowprops=dict(arrowstyle='->', lw=1.7, color=colors[i]))
    a.scatter(*x[0], color=colors[2], s=55, zorder=4)
    a.set_title(f'Forest {chr(65+i)}', fontsize=11)
    a.set_aspect('equal')
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlim(-.07, 1.07)
    a.set_ylim(-.03, .93)
w = stress['old_witness']
r = np.array(w['regrets'])
for i in range(2):
    ax[2].bar(np.arange(4)+(i-.5)*.34, r[i], .32, label=f'Forest {chr(65+i)}', color=colors[i])
ax[2].set_xticks(range(4), w['legal_heads'])
ax[2].set_xlabel('Head selected from source 0')
ax[2].set_ylabel('Exact excess completion cost')
ax[2].legend(frameon=False)
ax[2].grid(axis='y', alpha=.2)
fig.tight_layout(w_pad=1.2)
fig.savefig(F / 'witness.pdf')
fig.savefig(F / 'witness.png', dpi=180)
plt.close(fig)

lines = [r'\begin{tabular}{llrrr}', r'\toprule',
         r'Distribution & $N$ & Conflict (\%) & Regret & 95\% CI \\', r'\midrule']
for c in stress['cells']:
    lo, hi = c['bayes_regret_ci']
    lines.append(f"{c['distribution'].capitalize()} & {c['n']} & {100*c['conflict_fraction']:.1f} & {c['mean_bayes_regret']:.4f} & [{lo:.4f}, {hi:.4f}] \\")
lines += [r'\bottomrule', r'\end{tabular}']
# A TeX row requires two backslashes, including rows assembled via Python strings.
lines = [s+'\\' if s.endswith(' \\') and not s.endswith(' \\\\') else s for s in lines]
(P / 'stress_table.tex').write_text('\n'.join(lines)+'\n')

lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
         r'Host & Test & Route & Random & Learned & Learned excess (\%) \\', r'\midrule']
for r in hist['rows']:
    if r['case'] == 'test200-aug8':
        continue
    case = {'test200':'U200','cluster200':'C200','test500':'U500'}[r['case']]
    m = r['means']
    excess = -r['comparisons']['learned']['improvement_percent']
    lines.append(f"{r['host'].upper()} & {case} & {m['route']:.4f} & {m['random']:.4f} & {m['learned']:.4f} & {excess:.2f} \\")
lines += [r'\bottomrule', r'\end{tabular}']
lines = [s+'\\' if s.endswith(' \\') and not s.endswith(' \\\\') else s for s in lines]
(P / 'historical_table.tex').write_text('\n'.join(lines)+'\n')
print('Built witness, stress and historical figures and tables.')
