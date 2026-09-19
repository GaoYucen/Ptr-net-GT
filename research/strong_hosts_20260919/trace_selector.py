"""Post-training descriptive mechanism diagnostics on validation coordinates.

No checkpoint selection or further training. Correlations do not establish
causal reasons for quality changes. The replayed source policy is the actual
trained policy, not a separately implemented approximation.
"""
import argparse,json,math,types
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from strong_adapter import Adapter

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--host',choices=['am','icam'],required=True);a=p.parse_args()
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.16)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
d=a.root/a.host;data=torch.load(d/'datasets.pt',weights_only=True)['val200'][:16].cuda()
results=[]
for seed in (1234,4321,2468):
    for mode in ('random','learned'):
        path=d/f'{mode}-seed{seed}/best.pt'
        if not path.exists():continue
        m=Adapter(a.host).cuda().eval()
        payload=torch.load(path,weights_only=False,map_location='cpu')
        m.load_state_dict(payload['model'])
        # Recreate the same model construction and CUDA reset sequence as training.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed);initial_model=Adapter(a.host).cuda();initial=initial_model.selector
            for module in initial.modules():
                if hasattr(module,'reset_parameters'):module.reset_parameters()
            delta=sum(float((value.double()-initial.state_dict()[key].double()).square().sum())
                      for key,value in m.selector.state_dict().items())**.5
            if mode=='random':assert delta==0,('random selector initialization mismatch',seed,delta)
            if mode=='learned' and payload['best_step']>0:assert delta>0
            del initial,initial_model
        saved=[]; original=m.distributions
        def traced(self,enc,x,starts,features,pair_mask,tail_mask,last,mode):
            slp,hlp,raw=original(enc,x,starts,features,pair_mask,tail_mask,last,mode)
            safe=torch.where(torch.isfinite(hlp),hlp,0.)
            entropy=-(hlp.exp()*safe).sum(-1)
            sent=-(slp.exp()*torch.where(torch.isfinite(slp),slp,0.)).sum(-1)
            if mode=='random': sent=(~tail_mask).sum(-1).float().log()
            saved.append(dict(head_entropy=entropy.cpu(),source_entropy=sent.cpu(),
                              resolved=tail_mask.cpu(),sizes=features[:,:,-1].cpu()))
            return slp,hlp,raw
        m.distributions=types.MethodType(traced,m)
        with torch.no_grad():
            o=m.rollout(data,mode,gen=torch.Generator(device='cuda').manual_seed(995731))
        steps=[];tails=o['tails'].cpu();row=torch.arange(len(data))
        for t,s in enumerate(saved):
            unresolved=~s['resolved'];remaining=unresolved.sum(-1);tail=tails[:,t]
            chosen=s['head_entropy'][row,tail]
            # Midrank among unresolved tails: 0 = most confident endpoint policy.
            ranks=((s['head_entropy']<chosen[:,None]).float()+
                   .5*(s['head_entropy']==chosen[:,None]).float())*unresolved
            steps.append(dict(step=t,remaining=int(remaining[0]),
                source_entropy_normalized=float((s['source_entropy']/remaining.float().clamp_min(2).log()).mean()),
                selected_head_entropy_percentile=float((ranks.sum(-1)/remaining).mean()),
                selected_component_fraction=float(s['sizes'][row,tail].mean()),
                largest_component_fraction=float(s['sizes'].amax(-1).mean())))
        results.append(dict(host=a.host,mode=mode,seed=seed,instances=len(data),
                            mean_cost=float(o['cost'].mean()),source_deviation=float(o['deviation'].mean()),
                            selector_parameter_change_from_seed_init=delta,steps=steps))
        del m;torch.cuda.empty_cache()
(a.root/f'{a.host}-selector-traces.json').write_text(json.dumps(dict(
    note='Descriptive validation traces. Not a causal intervention; not test performance.',results=results),indent=2))
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,field,label in zip(axs,['source_entropy_normalized','selected_head_entropy_percentile','largest_component_fraction'],
                         ['Source entropy / log(remaining)','Chosen tail: endpoint entropy rank','Largest component / total nodes']):
    for mode,color in [('random','#e6a238'),('learned','#ca4775')]:
        curves=[np.array([s[field] for s in r['steps']]) for r in results if r['mode']==mode]
        if not curves:continue
        for c in curves:ax.plot(np.arange(len(c))/len(c),c,color=color,alpha=.18)
        ax.plot(np.arange(len(curves[0]))/len(curves[0]),np.mean(curves,axis=0),color=color,label=mode)
    ax.set(xlabel='Construction fraction',ylabel=label,ylim=(-.03,1.03));ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle(f'{a.host.upper()}: validation trajectories | descriptive correlations, not causal evidence')
fig.tight_layout();fig.savefig(a.root/f'{a.host}-selector-traces.png',dpi=170)
print(json.dumps([dict(host=r['host'],mode=r['mode'],seed=r['seed'],mean_cost=r['mean_cost'],
                       selector_parameter_change_from_seed_init=r['selector_parameter_change_from_seed_init'],
                       source_deviation=r['source_deviation'],
                       entropy_first_half=sum(s['source_entropy_normalized'] for s in r['steps'][:100])/100,
                       confidence_rank_first_half=sum(s['selected_head_entropy_percentile'] for s in r['steps'][:100])/100)
                  for r in results],indent=2))
