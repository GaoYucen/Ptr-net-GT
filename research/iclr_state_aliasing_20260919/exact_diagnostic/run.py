"""Paired forest endpoint diagnostic. Run from this directory or pass --output.
Exact enumeration labels are for fixed-source conditional decisions, not full TSP.
"""
from __future__ import annotations
import argparse, itertools, json, math, os, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn

def make_data(n, k, count, seed, distribution='uniform'):
    rng=np.random.default_rng(seed)
    if distribution=='uniform': x=rng.random((count,n,2))
    else:
        centers=rng.uniform(.15,.85,(count,3,2)); ids=rng.integers(0,3,(count,n))
        x=np.clip(np.take_along_axis(centers,ids[...,None],axis=1)+rng.normal(0,.06,(count,n,2)),0,1)
    x=x.astype(np.float32).astype(np.float64)
    heads=np.array([0]+list(range(1,k+1))+list(range(2*k+1,n)),dtype=np.int64)
    m=len(heads); tails=np.tile(heads,(count,2,1))
    for i in range(count):
        p=rng.permutation(np.arange(k+1,2*k+1)); tails[i,0,1:k+1]=p
        p=p.copy(); a,b=rng.choice(k,2,replace=False); p[a],p[b]=p[b],p[a]
        tails[i,1,1:k+1]=p
    perms=np.array(list(itertools.permutations(range(1,m))),dtype=np.int64)
    q=np.empty((count,2,m-1),dtype=np.float64)
    internal=np.empty((count,2),dtype=np.float64)
    for st in range(0,count,128):
        xx=x[st:st+128]; tt=tails[st:st+128]
        d=np.sqrt(((xx[:,:,None]-xx[:,None,:])**2).sum(-1))
        rows=np.arange(len(xx))[:,None,None]
        w=d[rows[...,None],tt[:,:,:,None],heads[None,None,None,:]]
        order=np.concatenate([np.zeros((len(perms),1),dtype=int),perms,np.zeros((len(perms),1),dtype=int)],axis=1)
        cs=w[:,:,order[:,:-1],order[:,1:]].sum(-1)
        q[st:st+len(xx)]=cs.reshape(len(xx),2,m-1,-1).min(-1)
        internal[st:st+len(xx)]=d[rows,heads[None,None,:],tt].sum(-1)
    return dict(x=x.astype(np.float32),heads=heads,tails=tails,q=q,internal=internal,
                seed=np.array(seed),n=np.array(n),k=np.array(k),distribution=np.array(distribution))

def features(d,aware):
    x=d['x']; tails=d['tails']; heads=d['heads']; g,n,_=x.shape; ns=tails.shape[1]
    f=np.zeros((g,ns,n,8),dtype=np.float32)
    f[:,:,:,:2]=x[:,None]-x[:,None,0:1]
    f[:,:,heads,2]=1; f[:,:,0,4]=1
    f[:,:,:,3]=1-f[:,:,:,2]
    # Both models know which nodes belong to non-singleton components.
    # Only the association coordinates differ; this is an enriched blind baseline.
    for state in range(ns):
        for c,h in enumerate(heads):
            t=tails[:,state,c]; rows=np.arange(g)
            f[rows,state,h,7]=(t!=h); f[rows,state,t,7]=(t!=h)
    if aware:
        for s in range(ns):
            for c,h in enumerate(heads):
                t=tails[:,s,c]; rows=np.arange(g)
                f[rows,s,h,5:7]=x[rows,t]-x[:,h]
                f[rows,s,t,5:7]=x[:,h]-x[rows,t]
                f[rows,s,h,7]=(t!=h);f[rows,s,t,7]=(t!=h)
    return f.reshape(-1,n,8)

class Model(nn.Module):
    def __init__(self,width=128,layers=3):
        super().__init__(); self.embed=nn.Linear(8,width)
        block=nn.TransformerEncoderLayer(width,8,width*2,dropout=0,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.out=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
    def forward(self,x):return self.out(self.encoder(self.embed(x))).squeeze(-1)

def greedy(d):
    x=d['x']; tails=d['tails']; heads=d['heads']; g,s,m=tails.shape
    out=np.zeros((g,2,m-1))
    dist=np.sqrt(((x[:,:,None]-x[:,None,:])**2).sum(-1))
    for j in range(m-1):
        for state in range(2):
            rows=np.arange(g); current=np.full(g,j+1); visited=np.zeros((g,m),bool);visited[:,0]=True;visited[:,j+1]=True
            cost=dist[:,0,heads[j+1]].copy()
            for step in range(m-2):
                w=dist[rows[:,None],tails[rows,state,current][:,None],heads[None,:]]
                nxt=np.where(visited,np.inf,w).argmin(-1)
                cost+=w[rows,nxt];visited[rows,nxt]=True;current=nxt
            cost+=dist[rows,tails[rows,state,current],0]
            out[:,state,j]=cost
    return out.argmin(-1)

def stats(a,seed=492190):
    a=np.asarray(a,dtype=np.float64); rng=np.random.default_rng(seed)
    bs=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(1000)])
    return dict(mean=float(a.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),n_geometry=len(a))

def lower_bounds(d):
    r=d['q']-d['q'].min(-1,keepdims=True)
    bound=r.mean(1).min(-1)
    opts=r<1e-9
    conflict=~(opts[:,0]&opts[:,1]).any(-1)
    return r,bound,conflict

def exact_self_check():
    d=make_data(7,2,12,190919902)
    n=7; ps=np.array([(0,)+p for p in itertools.permutations(range(1,n))]);su=np.zeros_like(ps)
    for i,p in enumerate(ps):su[i,p]=np.roll(p,-1)
    err=0.
    for g in range(12):
        x=d['x'][g].astype(float);di=np.linalg.norm(x[:,None]-x[None,:],axis=-1);cs=di[ps,np.roll(ps,-1,axis=1)].sum(-1)
        for s in range(2):
            valid=np.ones(len(ps),bool)
            for h,t in zip(d['heads'],d['tails'][g,s]):
                if h!=t:valid &= su[:,h]==t
            for a,h in enumerate(d['heads'][1:]):
                brute=cs[valid&(su[:,0]==h)].min()
                err=max(err,abs(brute-(d['q'][g,s,a]+d['internal'][g,s])))
    assert err<1e-6,err
    return dict(bruteforce_instances=24,max_abs_difference=err,precision='float64 labels, coordinates stored float32')

def evaluate_model(model,d,aware,device):
    model.eval();ff=features(d,aware); pred=[]
    # Keep every geometry family in one batch so identical inputs also use identical kernels.
    bs=(512//d['tails'].shape[1])*d['tails'].shape[1]
    with torch.no_grad():
        for i in range(0,len(ff),bs):pred.append(model(torch.from_numpy(ff[i:i+bs]).to(device)).cpu().numpy()[:,d['heads'][1:]])
    p=np.concatenate(pred).reshape(len(d['x']),d['tails'].shape[1],-1)
    if not aware:assert np.all(p==p[:,0:1]),np.abs(p-p[:,0:1]).max()
    actions=p.argmin(-1);r,_,_=lower_bounds(d)
    reg=np.take_along_axis(r,actions[...,None],-1).squeeze(-1)
    return dict(regret=reg,actions=actions,accuracy=(reg<1e-8),pred=p)

def train_model(train,val,aware,seed,args,out):
    torch.manual_seed(seed);np.random.seed(seed);device=args.device
    model=Model(args.width,args.layers).to(device)
    ff=torch.from_numpy(features(train,aware)).to(device)
    q=train['q'].reshape(-1,len(train['heads'])-1);q=q-q.min(-1,keepdims=True)
    yy=torch.tensor(q,dtype=torch.float32,device=device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    best=float('inf'); logs=[];t0=time.time(); name=('aware' if aware else 'blind')+f'_s{seed}'
    gen=torch.Generator(device=device).manual_seed(seed)
    for step in range(args.steps):
        model.train();ix=torch.randint(len(ff),(args.batch,),generator=gen,device=device)
        p=model(ff[ix])[:,train['heads'][1:]]
        loss=((p-yy[ix])**2).mean()
        optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
        if (step+1)%args.validate_every==0 or step==0:
            result=evaluate_model(model,val,aware,device); vr=float(result['regret'].mean())
            rec=dict(step=step+1,train_mse=float(loss.item()),val_regret=vr,seconds=time.time()-t0);logs.append(rec)
            if vr<best:
                best=vr;torch.save(dict(model=model.state_dict(),args=vars(args),aware=aware,seed=seed,step=step+1,val_regret=best),out/f'{name}.pt')
            print(name,json.dumps(rec),flush=True)
    ck=torch.load(out/f'{name}.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
    (out/f'{name}_training.json').write_text(json.dumps(logs,indent=2))
    return model,dict(name=name,aware=aware,seed=seed,best_step=ck['step'],validation_regret=best,parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.time()-t0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results');ap.add_argument('--device',default='cuda');ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--train-count',type=int,default=40000);ap.add_argument('--test-count',type=int,default=1500);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--width',type=int,default=128);ap.add_argument('--layers',type=int,default=3);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--validate-every',type=int,default=250);ap.add_argument('--seeds',type=int,nargs='+',default=[19091941,19091942,19091943]);ap.add_argument('--data-only',action='store_true');ap.add_argument('--resume',action='store_true');ap.add_argument('--model-kinds',nargs='+',default=['blind','aware']);args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);data_dir=out/'data';data_dir.mkdir(exist_ok=True)
    protocol=dict(task='fixed-source endpoint completion on paired directed path forests',source=0,paired_measure='uniform over the two forests for each geometry',train_geometry=args.train_count,train_data_seed=190919101,validation_data_seed=190919201,test_data_seeds=[190919301,190919302,190919303],train_shape=[9,3],limitations=['Conditional decision diagnostic, not complete-tour optimization.','The selector may distinguish the two forests; this lower bound does not apply to the joint source-target policy.','Pair oracle knows both complete Q tables; it is an optimistic lower bound for the blind representation, not an attainable learned comparator.'],self_check=exact_self_check(),args=vars(args))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    datasets={}
    def obtain(name,n,k,count,seed,dist='uniform'):
        path=data_dir/f'{name}.npz'
        if path.exists():d=dict(np.load(path))
        else:
            t=time.time();d=make_data(n,k,count,seed,dist);np.savez_compressed(path,**d);print('data',name,count,time.time()-t,flush=True)
        return d
    train=obtain('train',9,3,args.train_count,190919101);val=obtain('validation',9,3,3000,190919201)
    for name,n,k,dist in [('iid_n9',9,3,'uniform'),('cluster_n9',9,3,'cluster'),('ood_n7',7,2,'uniform'),('ood_n11',11,4,'uniform'),('ood_n13',13,5,'uniform')]:
        ds=[obtain(f'{name}_s{s}',n,k,args.test_count,s,dist) for s in protocol['test_data_seeds']]
        datasets[name]={key:np.concatenate([d[key] for d in ds],axis=0) if key in ('x','tails','q','internal') else ds[0][key] for key in ds[0]}
    summary=dict(protocol=protocol,datasets={},models=[])
    for name,d in datasets.items():
        r,b,c=lower_bounds(d); nearest=np.linalg.norm(d['x'][:,d['heads'][1:]]-d['x'][:,0:1],axis=-1).argmin(-1)
        nr=r[np.arange(len(r))[:,None],np.arange(2)[None,:],nearest[:,None]]; ga=greedy(d);gr=np.take_along_axis(r,ga[...,None],-1).squeeze(-1)
        sr=dict(n=int(d['n']),k=int(d['k']),n_geometry=len(r),n_forests=len(r)*2,pair_bayes_lower_bound=stats(b),conflict_rate=stats(c),nearest_regret=stats(nr.mean(1)),greedy_completion_regret=stats(gr.mean(1)),models={})
        summary['datasets'][name]=sr
        np.savez_compressed(out/f'{name}_baselines.npz',pair_lower_bound=b,conflict=c,nearest_regret=nr,greedy_regret=gr,greedy_actions=ga)
        print('baseline',name,json.dumps(sr),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if args.data_only:return
    for seed in args.seeds:
        for aware in [x=='aware' for x in args.model_kinds]:
            name=('aware' if aware else 'blind')+f'_s{seed}'
            if args.resume and (out/f'{name}.pt').exists() and (out/f'{name}_meta.json').exists():
                ck=torch.load(out/f'{name}.pt',map_location=args.device,weights_only=False);model=Model(args.width,args.layers).to(args.device);model.load_state_dict(ck['model']);meta=json.loads((out/f'{name}_meta.json').read_text())
            else:
                model,meta=train_model(train,val,aware,seed,args,out);(out/f'{name}_meta.json').write_text(json.dumps(meta,indent=2))
            summary['models'].append(meta)
            for dsname,d in datasets.items():
                rr=evaluate_model(model,d,aware,args.device);np.savez_compressed(out/f'{dsname}_{name}.npz',**rr)
                summary['datasets'][dsname]['models'][name]=dict(regret=stats(rr['regret'].mean(1)),accuracy=stats(rr['accuracy'].mean(1)))
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print('completed',name,flush=True)
    print('DONE',flush=True)
if __name__=='__main__':main()
