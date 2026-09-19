import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=json.loads((a.root/'endpoint-aliasing.json').read_text()); x=np.array(r['coordinates'])
fig,axs=plt.subplots(1,2,figsize=(10,5.7))
for i,ax in enumerate(axs):
    ax.scatter(x[:,0],x[:,1],s=140,c=['#c84454']+['#58677c']*6,zorder=3)
    for j,(px,py) in enumerate(x): ax.text(px+.014,py+.02,str(j),fontsize=13)
    for u,v in r['partial_edges'][i]:
        ax.annotate('',xy=x[v],xytext=x[u],arrowprops=dict(arrowstyle='->',lw=2,color='#58677c',shrinkA=8,shrinkB=8))
    v=r['optimal_heads'][i]
    ax.annotate('',xy=x[v],xytext=x[0],arrowprops=dict(arrowstyle='->',lw=2.5,linestyle='--',color='#19835c',shrinkA=8,shrinkB=8))
    ax.set(xlim=(-.06,1.05),ylim=(-.05,.9),aspect='equal',title=f'Forest {"AB"[i]}: optimal next edge 0 -> {v}')
    ax.set_xlabel(f'Exact best final tour: {r["exact_best_completion_by_head"][i][v]:.4f}')
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Same endpoint inputs, different optimal actions',fontsize=15)
fig.text(.5,.025,'Solid = fixed edges; dashed = exact-optimal next edge.\nEndpoint logits are identical in both hosts; source-selector features can distinguish the forests.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.15,1,.95));fig.savefig(a.root/'endpoint_aliasing.png',dpi=180)
