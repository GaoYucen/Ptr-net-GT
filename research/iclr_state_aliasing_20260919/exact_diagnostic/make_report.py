from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import stats

SEEDS=[19091941,19091942,19091943]
NAMES=['iid_n9','cluster_n9','ood_n7','ood_n11','ood_n13']
LABELS={'iid_n9':'Uniform n=9','cluster_n9':'Clustered n=9','ood_n7':'Uniform n=7','ood_n11':'Uniform n=11','ood_n13':'Uniform n=13'}

def combined(a):
    a=np.asarray(a); per_seed=a.mean(1); result=stats(a.mean(0));rng=np.random.default_rng(190919992)
    bb=[]
    for _ in range(1500):
        seed_ix=rng.integers(0,len(a),len(a));geo_ix=rng.integers(0,a.shape[1],a.shape[1]);bb.append(a[seed_ix][:,geo_ix].mean())
    result.update(seed_means=per_seed.tolist(),seed_sd=float(per_seed.std(ddof=1)),hierarchical_ci95=np.quantile(bb,[.025,.975]).tolist(),n_training_seeds=len(a))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');a=p.parse_args();out=Path(a.output);root=out.parent
    raw=json.loads((out/'summary.json').read_text());ext=json.loads((out/'extended_summary.json').read_text());sh=json.loads((out/'shuffled_summary.json').read_text());assign=json.loads((out/'assignment_summary.json').read_text());verify=json.loads((out/'verification.json').read_text())
    report=dict(status='complete',task='Exact fixed-source conditional endpoint diagnostic, not full-tour TSP benchmark',train_geometry=40000,validation_geometry=3000,test_geometry_per_distribution=4500,test_data_seeds=[190919301,190919302,190919303],training_seeds=SEEDS,parameters=raw['models'][0]['parameters'],models=raw['models']+sh['models'],verification=verify,datasets={},all_pairings={},statistical_note='ci95 bootstraps independent geometry, after averaging paired states and training seeds. hierarchical_ci95 additionally resamples the three training seeds; seed_sd is the SD of test means across training seeds. All model-selection uses validation only.',claim_scope=['All source choices are fixed to singleton node 0; the joint source-target policy is outside this bound.','All learned controls have identical architecture/parameter count and common geometry, masks, node roles, component-size information. Only true versus absent versus independent random head-tail association changes.','The pair-aware oracle lower bound knows which two latent pairings form the pair. The all-six Bayes risk conditions only on the enriched blind observation and averages every permitted pairing.','Random-pairing training, all-pairing evaluation, and assignment-relaxation heuristic are supplementary controls designed after pilot validation was observed.','Wrong-pairing evaluation swaps the two pairings at test time, with the model weights fixed. It is an intervention/distribution change, not a separately trained model.','No benchmark SOTA, neural runtime advantage, or submission acceptance claim is supported.'])
    arrays={}
    for name in NAMES:
        r=raw['datasets'][name];entry={k:r[k] for k in ['n','k','n_geometry','n_forests','pair_bayes_lower_bound','conflict_rate','nearest_regret','greedy_completion_regret']};entry['assignment_regret']=assign['datasets'][name]['regret'];entry['assignment_cpu_seconds']=assign['datasets'][name]['cpu_seconds'];entry['models']={};arrays[name]={}
        for method in ['blind','aware','shuffled']:
            arr=np.stack([np.load(out/f'{name}_{method}_s{s}.npz')['regret'].mean(1) for s in SEEDS]);arrays[name][method]=arr;entry['models'][method]=combined(arr)
        wrong=np.stack([np.load(out/f'{name}_wrong_pairing_aware_s{s}.npz')['regret'].mean(1) for s in SEEDS]);arrays[name]['wrong_pairing']=wrong;entry['models']['wrong_pairing']=combined(wrong)
        entry['blind_minus_aware']=combined(arrays[name]['blind']-arrays[name]['aware']);entry['shuffled_minus_aware']=combined(arrays[name]['shuffled']-arrays[name]['aware']);entry['wrong_minus_correct']=combined(wrong-arrays[name]['aware'])
        entry['aware_relative_regret_reduction_vs_blind']=float(1-entry['models']['aware']['mean']/entry['models']['blind']['mean'])
        ds=[dict(np.load(f)) for f in sorted((out/'data').glob(name+'_s*.npz'))];cost=np.concatenate([d['q'].min(-1)+d['internal'] for d in ds],0)
        for method in ['blind','aware','shuffled']:
            arr=np.stack([(100*np.load(out/f'{name}_{method}_s{s}.npz')['regret']/cost).mean(1) for s in SEEDS]);entry['models'][method]['percent_optimal_complete_tour_length']=combined(arr)
        report['datasets'][name]=entry
    for name,x in ext['all_pairings'].items():
        xx=dict(n_geometry=x['n_geometry'],pairings_per_geometry=x['pairings_per_geometry'],exact_blind_conditional_bayes_regret=x['exact_blind_conditional_bayes_regret'],models={})
        b=np.load(out/f'{name}_all_pairings_bayes.npz')['bayes_regret']
        for method in ['blind','aware']:
            arr=np.stack([np.load(out/f'{name}_all_pairings_{method}_s{s}.npz')['regret'].mean(1) for s in SEEDS]);xx['models'][method]=combined(arr)
            if method=='aware':xx['blind_bayes_minus_aware']=combined(b[None,:]-arr)
        report['all_pairings'][name]=xx
    report['information_curve']=json.loads((out/'information_curve.json').read_text())
    report['permutation_equivariance_max_abs_error']=ext['permutation_equivariance_max_abs_error']
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    colors={'blind':'#627184','aware':'#1579aa','shuffled':'#c49441','wrong_pairing':'#b94b54','assignment':'#577b59'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(14,3.9),constrained_layout=True)
    ax=axes[0];methods=['blind','aware'];names=['Blind','True pairing'];vals=[];errs=[]
    for m in methods:
        d=report['all_pairings']['iid_n9']['models'][m];vals.append(d['mean']*1000);lo,hi=d.get('hierarchical_ci95',d['ci95']);errs.append([(d['mean']-lo)*1000,(hi-d['mean'])*1000])
    ax.bar(np.arange(2),vals,color=[colors[m] for m in methods],yerr=np.array(errs).T,capsize=3)
    floor=report['all_pairings']['iid_n9']['exact_blind_conditional_bayes_regret']['mean']*1000;ax.axhline(floor,color='black',ls='--',lw=1.2,label='Blind Bayes risk (all 6 pairings)');ax.set_xticks(np.arange(2),names);ax.set_ylabel('Exact completion regret (×10⁻³)');ax.set_title('(a) All six matchings, uniform n=9');ax.legend(fontsize=8,loc='upper right',frameon=False);ax.set_ylim(0,45)
    ax=axes[1];sizes=[7,9,11,13];keys=['ood_n7','iid_n9','ood_n11','ood_n13']
    for m in ['blind','shuffled','aware','assignment']:
        means=[];err=[]
        for key in keys:
            d=report['datasets'][key]['models'][m] if m!='assignment' else report['datasets'][key]['assignment_regret'];means.append(d['mean']*1000);lo,hi=d.get('hierarchical_ci95',d['ci95']);err.append([(d['mean']-lo)*1000,(hi-d['mean'])*1000])
        ax.errorbar(sizes,means,yerr=np.array(err).T,color=colors[m],marker='o',capsize=2,label={'aware':'True pairing','shuffled':'Random pairing'}.get(m,m.title()))
    ax.set_xticks(sizes);ax.set_xlabel('Number of Euclidean nodes (train: 9)');ax.set_title('(b) Zero-shot size transfer');ax.legend(frameon=False,fontsize=8,ncol=2)
    ax=axes[2];keys=['iid_n9','cluster_n9','ood_n13'];positions=np.arange(3)
    for i,m in enumerate(['aware','wrong_pairing']):
        ds=[report['datasets'][k]['models'][m] for k in keys];ys=[d['mean']*1000 for d in ds];err=np.array([[(d['mean']-d['hierarchical_ci95'][0])*1000,(d['hierarchical_ci95'][1]-d['mean'])*1000] for d in ds]).T
        ax.bar(positions+(i-.5)*.34,ys,width=.34,yerr=err,capsize=2,color=colors[m],label='Correct pairing' if m=='aware' else 'Swapped pairing')
    ax.set_xticks(positions,['Uniform 9','Clustered 9','Uniform 13']);ax.set_title('(c) Fixed-weight association intervention');ax.legend(frameon=False,fontsize=8)
    fig.savefig(root/'diagnostic_results.pdf');fig.savefig(root/'diagnostic_results.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5),constrained_layout=True)
    for method in ['blind','aware','shuffled']:
        for i,s in enumerate(SEEDS):
            log=json.loads((out/f'{method}_s{s}_training.json').read_text());ax.plot([d['step'] for d in log],[d.get('val_regret',d.get('validation_regret')) for d in log],color=colors[method],alpha=.65,label=method if i==0 else None)
    ax.set_ylim(0,.08);ax.set_xlabel('Optimization steps');ax.set_ylabel('Validation exact completion regret');ax.legend(frameon=False);fig.savefig(root/'training_curves.pdf');fig.savefig(root/'training_curves.png',dpi=180);plt.close(fig)
    lines=['# Paired-forest conditional endpoint diagnostic','','This is an exact, reproducible mechanism experiment. It is not a full-tour TSP benchmark.','',f"All learned controls use {report['parameters']:,} parameters, 10,000 AdamW steps, batch 512, 40,000 training geometries, and the same three training seeds. Validation contains 3,000 independent geometries. Each test distribution contains 4,500 geometries from three new data seeds (9,000 forests).",'', 'Mean exact completion regret (unit-square Euclidean length; lower is better). Learned values are means across three seeds.','', '| Test | Pair oracle lower bound | Blind | Random pairing | True pairing | Assignment | Nearest |','|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        e=report['datasets'][name];lines.append('| '+LABELS[name]+' | '+' | '.join(f'{v:.6f}' for v in [e['pair_bayes_lower_bound']['mean'],e['models']['blind']['mean'],e['models']['shuffled']['mean'],e['models']['aware']['mean'],e['assignment_regret']['mean'],e['nearest_regret']['mean']])+' |')
    e=report['all_pairings']['iid_n9'];b=e['exact_blind_conditional_bayes_regret'];aa=e['models']['aware'];d=report['datasets']['iid_n9'];delta=d['blind_minus_aware'];wrong=d['wrong_minus_correct']
    lines+=['',f"Enumerating all six possible pairings for the same 4,500 uniform n=9 geometries gives an exact blind conditional Bayes risk of **{b['mean']:.6f}** (geometry-bootstrap 95% CI {b['ci95'][0]:.6f}–{b['ci95'][1]:.6f}). The true-pairing model obtains **{aa['mean']:.6f}** on this full family. This Bayes risk includes the same component-size side information given to both learned models.",'',f"On paired IID tests, true pairing reduces mean regret by **{100*d['aware_relative_regret_reduction_vs_blind']:.1f}%** versus blind. The paired reduction is {delta['mean']:.6f} (geometry-and-seed bootstrap 95% CI {delta['hierarchical_ci95'][0]:.6f}–{delta['hierarchical_ci95'][1]:.6f}). Swapping the association at test time increases regret by {wrong['mean']:.6f} (95% CI {wrong['hierarchical_ci95'][0]:.6f}–{wrong['hierarchical_ci95'][1]:.6f}).",'',f"Independent Held–Karp DP checked {verify['independent_held_karp_forests_checked']} saved forests against enumeration: maximum absolute discrepancy {verify['max_absolute_label_error']:.2g}. Geometry hashes confirm disjoint train/validation/test sets. The existing Forest implementation verifies identical source context and legality masks in paired states. Blind features and predictions are exactly equal within each pair. Permutation equivariance maximum numerical error is {ext['permutation_equivariance_max_abs_error']:.2g}.",'', 'All confidence intervals and per-seed values are in summary.json. Raw coordinates, pairings, exact Q values, predictions, checkpoints, and training logs are in results/.','', 'Limits and protocol chronology:']
    lines += ['- '+s for s in report['claim_scope']]
    lines+=['','Reproduce:','','```bash','python run.py --output results --device cuda --steps 10000 --validate-every 500','python extend.py --output results --device cuda','python shuffled_control.py --output results --device cuda','python assignment_baseline.py --output results','python information_curve.py --output results','python verify.py --output results','python make_report.py --output results','```','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*') if f.is_file() and f.name!='manifest_sha256.json'};(root/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
    print('\n'.join(lines))
if __name__=='__main__':main()
