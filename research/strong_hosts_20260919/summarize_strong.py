import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'native':'#525866','route':'#377eb8','random':'#e6a238','learned':'#ca4775','capacity':'#6c54a3'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
    a.root.mkdir(exist_ok=True,parents=True)
    rows=[]; paired=[]
    cases=['test200','test200-aug8','cluster200','test500']
    for host in ('am','icam'):
        d=a.root/host
        for f in sorted(d.glob('*-seed*/summary.json')):
            if 'interrupted' in str(f): continue
            s=json.loads(f.read_text())
            for case,v in s['evaluation'].items():
                rows.append(dict(host=host,mode=s['mode'],seed=s['seed'],case=case,cost=v['mean'],
                                 seconds=v['seconds'],best_step=s['best_step'],
                                 train_seconds=s['training_seconds'],warm_seconds=s['warmup_seconds']))
        for mode in ('random','learned','capacity'):
            for case in cases:
                differences=[]; bases=[]; seed_gains=[]
                for seed in (1234,4321,2468):
                    base=d/f'route-seed{seed}/{case}-costs.pt'
                    target=d/f'{mode}-seed{seed}/{case}-costs.pt'
                    if not base.exists() or not target.exists(): continue
                    b=torch.load(base,weights_only=True).double().numpy()
                    t=torch.load(target,weights_only=True).double().numpy()
                    differences.append(b-t); bases.append(b)
                    seed_gains.append(100*(b.mean()-t.mean())/b.mean())
                if not differences: continue
                diff=np.mean(differences,axis=0); base=np.mean(bases,axis=0)
                rng=np.random.default_rng(179)
                idx=rng.integers(0,len(diff),(5000,len(diff)))
                boots=100*diff[idx].mean(1)/base[idx].mean(1)
                n=len(seed_gains)
                ci_seed=(float(np.mean(seed_gains)-4.302652729*np.std(seed_gains,ddof=1)/math.sqrt(n)),
                         float(np.mean(seed_gains)+4.302652729*np.std(seed_gains,ddof=1)/math.sqrt(n))) if n==3 else None
                paired.append(dict(host=host,mode=mode,case=case,seeds=n,
                                   improvement_percent=float(100*diff.mean()/base.mean()),
                                   absolute_delta=float(diff.mean()),per_seed=seed_gains,
                                   conditional_instance_bootstrap_ci=np.quantile(boots,[.025,.975]).tolist(),
                                   conditional_training_seed_t_ci=ci_seed))
    (a.root/'aggregated.json').write_text(json.dumps(dict(rows=rows,paired=paired,
        note='Seeds share warmup. Both intervals are conditional, not full retraining uncertainty.'),indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,case in zip(axs.flat,cases):
        positions=[]; labels=[]
        for hi,host in enumerate(('am','icam')):
            for mi,mode in enumerate(('native','route','random','learned')):
                pos=hi*5+mi; values=[]
                if mode=='native':
                    f=a.root/host/'native-summary.json'
                    if f.exists(): values=[json.loads(f.read_text())[case]['mean']]
                else: values=[r['cost'] for r in rows if (r['host'],r['mode'],r['case'])==(host,mode,case)]
                positions.append(pos); labels.append(f'{host.upper()}\n{mode}')
                if values:
                    ax.bar(pos,np.mean(values),color=COLORS[mode],alpha=.8)
                    ax.scatter(np.linspace(pos-.15,pos+.15,len(values)),values,c='black',s=17,zorder=3)
                else: ax.text(pos,0,'pending',rotation=90,va='bottom',ha='center',color='#888')
        ax.set_xticks(positions,labels,fontsize=8)
        ax.set_title(case.replace('test','TSP').replace('cluster','Clustered TSP'))
        ax.set_ylabel('Mean tour length (lower is better)')
    fig.suptitle('Official hosts + shared ASCC adapter | dots = joint training seeds')
    fig.tight_layout(); fig.savefig(a.root/'quality.png',dpi=170); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,host in zip(axs,('am','icam')):
        for mode in ('route','random','learned'):
            labeled=False
            for seed in (1234,4321,2468):
                path=a.root/host/f'{mode}-seed{seed}/history.json'
                if not path.exists(): continue
                h=json.loads(path.read_text())
                ax.plot([q['training_seconds']/60 for q in h],[q['validation']['mean'] for q in h],
                        color=COLORS[mode],alpha=.7,label=mode if not labeled else None)
                labeled=True
        ax.set(title=host.upper(),xlabel='Joint training time (minutes; warmup reported separately)',
               ylabel='TSP200 validation mean length')
        ax.legend()
    fig.suptitle('Learning curves: shared warmup, independent joint seeds; no convergence assumption')
    fig.tight_layout(); fig.savefig(a.root/'learning_curves.png',dpi=170); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,host in zip(axs,('am','icam')):
        vals=[]; errors=[]
        for case in cases:
            entry=next((p for p in paired if p['host']==host and p['mode']=='learned' and p['case']==case),None)
            vals.append(entry['improvement_percent'] if entry else np.nan)
            errors.append(entry['conditional_training_seed_t_ci'] if entry else None)
        ax.bar(range(4),vals,color=['#408a68' if v>0 else '#c75667' for v in vals])
        for i,entry in enumerate(errors):
            if entry is not None:
                # Show interval around mean of per-seed gains (slightly differs from ratio of means).
                center=(entry[0]+entry[1])/2
                ax.errorbar(i,center,yerr=(entry[1]-entry[0])/2,color='black',capsize=4)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(4),['TSP200','TSP200\naug8','Cluster200','TSP500'])
        ax.set(title=host.upper(),ylabel='ASCC improvement over adapted route (%)')
    fig.suptitle('Positive = improvement | intervals: 3 joint seeds, conditional on shared warmup')
    fig.tight_layout(); fig.savefig(a.root/'improvement.png',dpi=170); plt.close(fig)
    effect_cases=['test200','test500','cluster200']
    effect=np.array([[next((q['improvement_percent'] for q in paired if
                           (q['host'],q['mode'],q['case'])==(host,'learned',case)),np.nan)
                      for case in effect_cases] for host in ('am','icam')])
    bound=max(10,float(np.nanmax(np.abs(effect)))) if np.isfinite(effect).any() else 10
    fig,ax=plt.subplots(figsize=(9,3.4));im=ax.imshow(effect,cmap='RdYlGn',vmin=-bound,vmax=bound,aspect='auto')
    for i in range(2):
        for j in range(3):
            ax.text(j,i,f'{effect[i,j]:+.2f}%',ha='center',va='center',fontsize=18,
                    color='white' if abs(effect[i,j])>.72*bound else '#242424')
    ax.set_xticks(range(3),['Uniform TSP200','Uniform TSP500','Clustered TSP200'])
    ax.set_yticks(range(2),['AM + ASCC','ICAM + ASCC'])
    ax.set_title('ASCC improvement over adapted route | positive is better',pad=14)
    fig.colorbar(im,ax=ax,label='Relative improvement (%)')
    fig.text(.5,.02,'Mean effects; seed-level uncertainty is reported separately. Equal training samples, not equal time.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(a.root/'ascc_mean_effect.png',dpi=170);plt.close(fig)
    reference_path=a.root/'reference/summary.json'
    reference=json.loads(reference_path.read_text())['cases'] if reference_path.exists() else {}
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    gap_rows=[]
    for ax,host in zip(axs,('am','icam')):
        native=json.loads((a.root/host/'native-summary.json').read_text())
        for mi,mode in enumerate(('native','route','random','learned')):
            vals=[]
            for case in ('test200','cluster200','test500'):
                costs=([native[case]['mean']] if mode=='native' else
                       [r['cost'] for r in rows if (r['host'],r['mode'],r['case'])==(host,mode,case)])
                gap=100*(np.mean(costs)/reference[case]['mean']-1) if costs and case in reference else np.nan
                vals.append(gap)
                if np.isfinite(gap): gap_rows.append(dict(host=host,mode=mode,case=case,gap_percent=float(gap)))
            ax.bar(np.arange(3)+(mi-1.5)*.2,vals,width=.2,label=mode,color=COLORS[mode])
        ax.set_xticks(range(3),['Uniform200','Clustered200','Uniform500'])
        ax.set(title=host.upper(),ylabel='Gap to LKH reference (%)',yscale='symlog')
        ax.legend(fontsize=8)
    fig.suptitle('Available headroom and adapter outcomes | LKH reference is NOT certified OPT')
    fig.tight_layout(); fig.savefig(a.root/'reference_gaps.png',dpi=170); plt.close(fig)
    (a.root/'reference_gaps.json').write_text(json.dumps(gap_rows,indent=2))
    headroom_cases=['test200','test500','cluster200']
    matrix=np.array([[next((r['gap_percent'] for r in gap_rows if
                            (r['host'],r['mode'],r['case'])==(host,'native',case)),np.nan)
                      for case in headroom_cases] for host in ('am','icam')])
    fig,ax=plt.subplots(figsize=(9,3.5));im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=60,aspect='auto')
    for i in range(2):
        for j in range(3):
            ax.text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',fontsize=19,
                    color='white' if matrix[i,j]>35 else '#242424')
    ax.set_xticks(range(3),['Uniform TSP200','Uniform TSP500','Clustered TSP200'])
    ax.set_yticks(range(2),['Official AM','Official ICAM'])
    ax.set_title('Native greedy gap to LKH reference | lower is better',pad=14)
    fig.colorbar(im,ax=ax,label='Gap (%)');fig.text(.5,.02,'LKH is a feasible reference, not certified OPT. Single-start evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(a.root/'native_headroom.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,host in zip(axs,('am','icam')):
        native=json.loads((a.root/host/'native-summary.json').read_text())
        for mode in ('native','route','random','learned'):
            for case,marker in (('test200','o'),('test200-aug8','s')):
                group=([native[case]] if mode=='native' else
                       [dict(mean=r['cost'],seconds=r['seconds']) for r in rows
                        if (r['host'],r['mode'],r['case'])==(host,mode,case)])
                if group: ax.scatter(np.mean([g['seconds'] for g in group])*1000/256,
                                     np.mean([g['mean'] for g in group]),color=COLORS[mode],marker=marker,
                                     s=60,label=f'{mode}: '+('greedy' if marker=='o' else 'aug8'))
        ax.set(title=host.upper(),xlabel='Evaluation ms / instance (log scale)',xscale='log',
               ylabel='TSP200 mean length (lower is better)')
        ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Evaluation includes validity checks | shared GPUs; not equal-time frontiers')
    fig.tight_layout(); fig.savefig(a.root/'deployment.png',dpi=170); plt.close(fig)
    multi_path=a.root/'icam-multistart/summary.json'
    if multi_path.exists():
        ms=json.loads(multi_path.read_text())['cases'];ordered=['test200','test500','cluster200']
        if all(case in ms for case in ordered):
            matrix=np.array([[100*(ms[c]['greedy_first_start_mean']/ms[c]['reference_mean']-1) for c in ordered],
                             [ms[c]['multistart_gap_percent'] for c in ordered],
                             [ms[c]['multistart_aug8_gap_percent'] for c in ordered]])
            fig,ax=plt.subplots(figsize=(9,4.3));im=ax.imshow(matrix,cmap='YlOrRd',vmin=0,vmax=25,aspect='auto')
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}%',ha='center',va='center',fontsize=18,
                                          color='white' if matrix[i,j]>16 else '#242424')
            ax.set_xticks(range(3),['Uniform TSP200','Uniform TSP500','Clustered TSP200'])
            ax.set_yticks(range(3),['Single start','All N starts','All N starts + aug8'])
            ax.set_title('Official ICAM checkpoint: headroom after decoding-budget calibration',pad=14)
            fig.colorbar(im,ax=ax,label='Gap to LKH reference (%)')
            fig.text(.5,.02,'Same held-out instances. LKH reference is not certified OPT. No further training.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.05,1,1));fig.savefig(a.root/'icam_budget_headroom.png',dpi=170);plt.close(fig)
    print(json.dumps(paired,indent=2))

if __name__=='__main__': main()
