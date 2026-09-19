from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent/'evidence'/'screen-v1'
fig, axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
colors={'route-bopo-seed1234':'#2563eb','learned-bopo-seed1234':'#ea580c',
        'route-reinforce-seed1234':'#0891b2','learned-reinforce-seed1234':'#9333ea'}
for directory in sorted(root.glob('*-seed1234')):
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    vals=[x for x in rows if 'validation' in x]
    labels=directory.name.replace('-seed1234','').replace('learned','ASCC')
    axes[0].plot([x['step'] for x in vals],[x['validation']['mean_cost'] for x in vals],
                 marker='o',color=colors[directory.name],label=labels)
    tr=[x for x in rows if 'train_best' in x]
    axes[1].plot([x['step'] for x in tr],[x['source_entropy'] for x in tr],
                 color=colors[directory.name],label=labels)
axes[0].set(ylabel='Validation tour cost (lower is better)',xlabel='Optimizer step',
            title='Matched-data TSP100 screen')
axes[1].set(ylabel='Mean source entropy (nats)',xlabel='Optimizer step',
            title='Source policy activity')
for ax in axes:
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8)
fig.suptitle('One seed, 200 steps: diagnostic evidence, not a paper main result',fontsize=11)
fig.savefig(root/'screen_curves.png',dpi=180)
fig.savefig(root/'screen_curves.pdf')
print(root/'screen_curves.png')
