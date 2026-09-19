"""Additional zero-training diagnostic: reveal fixed component partners progressively."""
import argparse,json
from pathlib import Path
import numpy as np
from run import stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');a=p.parse_args();out=Path(a.output)
    summary=dict(protocol='Additional diagnostic specified after main training began. For the same geometry and uniform distribution over all six matchings, reveal the partners of fixed non-singleton heads 1,...,r. Partition the six latent states by those observations. For each observation class choose the action minimizing mean exact regret, then average classes by their probability. These are exact conditional Bayes risks, with no neural training.',datasets={})
    for name in ['iid_n9','cluster_n9']:
        d=dict(np.load(out/'data'/f'{name}_all_pairings.npz'));q=d['q'];r=q-q.min(-1,keepdims=True);tails=d['tails'];g,s,acts=q.shape;k=int(d['k']);curves=[];partitions=[]
        # Enumeration uses identical ordering of all matchings for every geometry.
        assert np.all(tails==tails[0:1])
        for revealed in range(k+1):
            labels=tails[0,:,1:revealed+1];_,inv=np.unique(labels,axis=0,return_inverse=True)
            groups=[np.flatnonzero(inv==label) for label in np.unique(inv)]
            bayes=sum((len(ix)/s)*r[:,ix,:].mean(1).min(-1) for ix in groups)
            curves.append(bayes);partitions.append([ix.tolist() for ix in groups])
        arr=np.stack(curves,1);assert np.all(np.diff(arr,axis=1)<=1e-10);assert np.all(arr[:,k-1:]==0)
        records=[]
        for i in range(k+1):
            record=dict(revealed_fixed_heads=i,observation_groups=len(partitions[i]),bayes_regret=stats(arr[:,i]))
            if i:record['reduction_from_previous']=stats(arr[:,i-1]-arr[:,i])
            records.append(record)
        summary['datasets'][name]=dict(n_geometry=g,matchings_per_geometry=s,curve=records,partitions=partitions)
        np.savez_compressed(out/f'{name}_information_curve.npz',bayes_regret=arr)
    (out/'information_curve.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(5.6,3.4),constrained_layout=True)
    for name,color in [('iid_n9','#1579aa'),('cluster_n9','#c49441')]:
        records=summary['datasets'][name]['curve'];v=np.array([z['bayes_regret']['mean'] for z in records]);ci=np.array([z['bayes_regret']['ci95'] for z in records])
        ax.errorbar(range(4),v*1000,yerr=np.stack([v-ci[:,0],ci[:,1]-v])*1000,marker='o',capsize=3,color=color,label='Uniform' if name=='iid_n9' else 'Clustered')
    ax.set_xticks(range(4));ax.set_xlabel('Number of fixed heads with revealed partners');ax.set_ylabel('Exact conditional Bayes regret (×10⁻³)');ax.legend(frameon=False);ax.text(1.55,8,'Two revealed partners determine\nthe third by exclusion.',fontsize=9)
    fig.savefig(out.parent/'information_curve.pdf',bbox_inches='tight');fig.savefig(out.parent/'information_curve.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
