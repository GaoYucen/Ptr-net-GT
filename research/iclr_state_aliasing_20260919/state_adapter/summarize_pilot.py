"""Paired instance summaries, conditional on the three fitted training seeds."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    out=a.out; sets=torch.load(out/'final_test_data.pt',weights_only=False)
    folders=sorted(out.glob('*-seed*')); folders=[f for f in folders if (f/'test_summary.json').exists()]
    arms={}; rows=[]; rng=np.random.default_rng(880192)
    for folder in folders:
        info=json.loads((folder/'test_summary.json').read_text()); train=json.loads((folder/'training_summary.json').read_text())
        arm=info['mode']+('-aware' if info['aware'] else '-blind')
        for name,metrics in info['test'].items():
            values=torch.load(folder/f'{name}-costs.pt',weights_only=False).double().numpy()
            arms.setdefault(name,{}).setdefault(arm,[]).append(values)
            rows.append(dict(host=out.name,dataset=name,arm=arm,seed=info['seed'],selected_step=info['selected_step'],
                             mean=float(values.mean()),nll=metrics['conditional']['nll'],accuracy=metrics['conditional']['accuracy'],
                             decoder_seconds=metrics['seconds'],training_seconds=train['training_seconds'],
                             parameters=train['parameters']))
    with (out/'per_run.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={}; comparisons={}
    table=['| Dataset | Arm | Mean cost (seed SD) | NLL | Accuracy |','|---|---|---:|---:|---:|']
    for name,methods in arms.items():
        summary[name]={}; comparisons[name]={}
        for arm,v in methods.items():
            v=np.stack(v); methods[arm]=v
            rr=[r for r in rows if r['dataset']==name and r['arm']==arm]
            stat=dict(mean=float(v.mean()),seed_sd=float(v.mean(1).std(ddof=1)),seeds=len(v),instances=v.shape[1],
                      conditional_nll=float(np.mean([r['nll'] for r in rr])),
                      conditional_accuracy=float(np.mean([r['accuracy'] for r in rr])),
                      selected_steps=[r['selected_step'] for r in rr])
            summary[name][arm]=stat
            table.append(f'| {name} | {arm} | {stat["mean"]:.5f} ({stat["seed_sd"]:.5f}) | {stat["conditional_nll"]:.4f} | {stat["conditional_accuracy"]:.4f} |')
        native=sets[name]['native_cost'].double().numpy()
        table.append(f'| {name} | native | {native.mean():.5f} | — | — |')
        for treatment,baseline in [('random-aware','random-blind'),('random-aware','route-aware'),
                                   ('route-aware','route-blind'),('random-aware','native'),('route-aware','native')]:
            c=methods[treatment]
            base=(np.broadcast_to(native,c.shape) if baseline=='native' else methods[baseline])
            paired=((base-c)/base*100).mean(0)
            idx=rng.integers(0,len(paired),(5000,len(paired)))
            boot=paired[idx].mean(1)
            comparisons[name][treatment+' vs '+baseline]=dict(
                improvement_percent=float(paired.mean()),paired_instance_ci95=np.quantile(boot,[.025,.975]).tolist(),
                win_fraction=float((c.mean(0)<base.mean(0)).mean()),
                per_seed_improvement_percent=((base-c)/base*100).mean(1).tolist(),
                caveat='Instance bootstrap conditional on these three seeds; not a seed-population confidence interval.')
    result=dict(summary=summary,comparisons=comparisons,
                interpretation='Positive improvement means lower cost; references are native greedy or 2-opt teachers, never OPT.')
    (out/'aggregate.json').write_text(json.dumps(result,indent=2))
    table+=['','Confidence intervals below bootstrap paired test instances after averaging the three training seeds. They do not quantify uncertainty over all possible training seeds.','',
            '| Dataset | Comparison | Improvement % [95% CI] | Win fraction |','|---|---|---:|---:|']
    for name,cs in comparisons.items():
        for label,v in cs.items():
            lo,hi=v['paired_instance_ci95']; table.append(f'| {name} | {label} | {v["improvement_percent"]:.3f} [{lo:.3f}, {hi:.3f}] | {v["win_fraction"]:.3f} |')
    (out/'SUMMARY.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
