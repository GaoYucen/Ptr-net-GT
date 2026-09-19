"""All-pairing exact conditional Bayes risk; paired association ablation."""
import argparse,itertools,json,hashlib,platform,time
from pathlib import Path
import numpy as np
import torch
from run import Model,features,evaluate_model,stats,lower_bounds

def full_data(ds):
    heads=ds['heads']; k=int(ds['k']);x=ds['x'];g,n,_=x.shape
    pp=np.array(list(itertools.permutations(range(k+1,2*k+1))));ns=len(pp);m=len(heads)
    tails=np.tile(heads,(g,ns,1));tails[:,:,1:k+1]=pp[None]
    perms=np.array(list(itertools.permutations(range(1,m))),dtype=np.int64)
    order=np.concatenate([np.zeros((len(perms),1),dtype=int),perms,np.zeros((len(perms),1),dtype=int)],axis=1)
    q=np.empty((g,ns,m-1));internal=np.empty((g,ns))
    for start in range(0,g,128):
        xx=x[start:start+128].astype(float);tt=tails[start:start+128];size=len(xx)
        d=np.linalg.norm(xx[:,:,None]-xx[:,None,:],axis=-1);rr=np.arange(size)[:,None,None]
        w=d[rr[...,None],tt[:,:,:,None],heads[None,None,None,:]]
        costs=w[:,:,order[:,:-1],order[:,1:]].sum(-1)
        q[start:start+size]=costs.reshape(size,ns,m-1,-1).min(-1)
        internal[start:start+size]=d[rr,heads[None,None,:],tt].sum(-1)
    return dict(x=x,heads=heads,tails=tails,q=q,internal=internal,k=np.array(k),n=np.array(n))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');p.add_argument('--device',default='cuda');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();out=Path(a.output)
    torch.set_num_threads(4);summary=json.loads((out/'summary.json').read_text());add={}
    dsets={}
    for name in summary['datasets']:
        ds=[dict(np.load(f)) for f in sorted((out/'data').glob(name+'_s*.npz'))]
        dsets[name]={key:np.concatenate([d[key] for d in ds],0) if key in ('x','tails','q','internal') else ds[0][key] for key in ds[0]}
    for name in ['iid_n9','cluster_n9']:
        path=out/'data'/f'{name}_all_pairings.npz'
        if path.exists():full=dict(np.load(path))
        else:full=full_data(dsets[name]);np.savez_compressed(path,**full)
        r=full['q']-full['q'].min(-1,keepdims=True)
        bayes=r.mean(1).min(-1)
        add[name]=dict(n_geometry=len(r),pairings_per_geometry=r.shape[1],exact_blind_conditional_bayes_regret=stats(bayes),models={})
        np.savez_compressed(out/f'{name}_all_pairings_bayes.npz',regret=r,bayes_regret=bayes)
        if a.prepare_only:continue
        for meta in summary['models']:
            modname=meta['name'];ck=torch.load(out/f'{modname}.pt',map_location=a.device,weights_only=False); model=Model(ck['args']['width'],ck['args']['layers']).to(a.device);model.load_state_dict(ck['model']);aware=meta['aware']
            result=evaluate_model(model,full,aware,a.device);np.savez_compressed(out/f'{name}_all_pairings_{modname}.npz',**result)
            add[name]['models'][modname]=dict(regret=stats(result['regret'].mean(1)),accuracy=stats(result['accuracy'].mean(1)))
    if a.prepare_only:
        (out/'full_bayes_preliminary.json').write_text(json.dumps(add,indent=2));return
    ablation={}
    for name,ds in dsets.items():
        r,_,_=lower_bounds(ds);ablation[name]={}
        for meta in summary['models']:
            if not meta['aware']:continue
            modname=meta['name'];res=np.load(out/f'{name}_{modname}.npz');actions=res['actions'][:,::-1]
            wrong=np.take_along_axis(r,actions[...,None],-1).squeeze(-1)
            diff=wrong.mean(1)-res['regret'].mean(1)
            ablation[name][modname]=dict(wrong_pairing_regret=stats(wrong.mean(1)),wrong_minus_correct=stats(diff))
            np.savez_compressed(out/f'{name}_wrong_pairing_{modname}.npz',regret=wrong,actions=actions,paired_delta=diff)
    ck=torch.load(out/f"{summary['models'][-1]['name']}.pt",map_location=a.device,weights_only=False);model=Model(ck['args']['width'],ck['args']['layers']).to(a.device);model.load_state_dict(ck['model']);model.eval()
    f=torch.tensor(features(dsets['iid_n9'],True)[:32],device=a.device);rng=np.random.default_rng(190919701);perm=rng.permutation(f.shape[1]);inv=np.argsort(perm)
    with torch.no_grad():err=float((model(f)-model(f[:,perm])[:,inv]).abs().max().item())
    assert err<1e-5,err
    result=dict(all_pairings=add,wrong_pairing=ablation,permutation_equivariance_max_abs_error=err,semantics='All-pairing Bayes risk conditions on the geometry, source, head set, and identity of the three nontrivial component heads. Both learned controls receive this component-size side information; only association coordinates distinguish aware from blind.',software=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__))
    (out/'extended_summary.json').write_text(json.dumps(result,indent=2))
    manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file() and f.name!='manifest_sha256.json'}
    (out/'manifest_sha256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
