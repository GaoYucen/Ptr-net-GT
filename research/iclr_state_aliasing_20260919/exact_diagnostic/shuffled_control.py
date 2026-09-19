"""Additional mechanism control, specified after pilot validation was observed.
Train the same network with a fresh uniformly random head-tail matching each SGD
batch, independent of the label-generating matching. Roles and capacity match.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from run import Model,features,stats,lower_bounds

def shuffled(f,k,generator):
    x=f.clone();b=len(x);rows=torch.arange(b,device=f.device)
    tt=torch.rand((b,k),generator=generator,device=f.device).argsort(-1)+k+1
    for c in range(k):
        h=c+1;t=tt[:,c];delta=x[rows,t,:2]-x[:,h,:2]
        x[:,h,5:7]=delta;x[rows,t,5:7]=-delta
    return x

def evaluate(model,d,device,seed):
    model.eval();ff=features(d,False);pred=[];gen=torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for st in range(0,len(ff),512):
            x=shuffled(torch.from_numpy(ff[st:st+512]).to(device),int(d['k']),gen)
            pred.append(model(x)[:,d['heads'][1:]].cpu().numpy())
    pred=np.concatenate(pred).reshape(len(d['x']),2,-1);actions=pred.argmin(-1);r,_,_=lower_bounds(d)
    regret=np.take_along_axis(r,actions[...,None],-1).squeeze(-1)
    return dict(regret=regret,actions=actions,pred=pred,accuracy=(regret<1e-8))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results_fair');p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=10000);p.add_argument('--seeds',type=int,nargs='+',default=[19091941,19091942,19091943]);a=p.parse_args();out=Path(a.output);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    train=dict(np.load(out/'data/train.npz'));val=dict(np.load(out/'data/validation.npz'));dsets={}
    for name in ['iid_n9','cluster_n9','ood_n7','ood_n11','ood_n13']:
        ds=[dict(np.load(f)) for f in sorted((out/'data').glob(name+'_s*.npz'))]
        dsets[name]={key:np.concatenate([d[key] for d in ds],0) if key in ('x','tails','q','internal') else ds[0][key] for key in ds[0]}
    summary=dict(protocol='Post-pilot additional mechanism control. Fresh uniformly random matching each batch, independent of ground-truth matching; matched roles and parameter count. Validation and test each use a fixed independent corruption stream; selection on validation only.',models=[],datasets={name:{} for name in dsets})
    for seed in a.seeds:
        torch.manual_seed(seed);model=Model().to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5);gen=torch.Generator(device=a.device).manual_seed(seed);corruption_gen=torch.Generator(device=a.device).manual_seed(seed+9181)
        ff=torch.from_numpy(features(train,False)).to(a.device);q=train['q'].reshape(-1,len(train['heads'])-1).copy();q-=q.min(-1,keepdims=True);yy=torch.tensor(q,dtype=torch.float32,device=a.device)
        best=float('inf');logs=[];start=time.time();name=f'shuffled_s{seed}'
        for step in range(a.steps):
            model.train();ix=torch.randint(len(ff),(512,),generator=gen,device=a.device);x=shuffled(ff[ix],int(train['k']),corruption_gen);pred=model(x)[:,train['heads'][1:]];loss=((pred-yy[ix])**2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            if step==0 or (step+1)%500==0:
                ev=evaluate(model,val,a.device,190919811);score=float(ev['regret'].mean());rec=dict(step=step+1,train_mse=float(loss.item()),validation_regret=score,seconds=time.time()-start);logs.append(rec);print(name,json.dumps(rec),flush=True)
                if score<best:
                    best=score;torch.save(dict(model=model.state_dict(),step=step+1,validation_regret=score,seed=seed),out/f'{name}.pt')
        ck=torch.load(out/f'{name}.pt',map_location=a.device,weights_only=False);model.load_state_dict(ck['model']);meta=dict(name=name,seed=seed,best_step=ck['step'],validation_regret=best,parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.time()-start)
        summary['models'].append(meta);(out/f'{name}_training.json').write_text(json.dumps(logs,indent=2));(out/f'{name}_meta.json').write_text(json.dumps(meta,indent=2))
        for dsname,d in dsets.items():
            rr=evaluate(model,d,a.device,190919812);np.savez_compressed(out/f'{dsname}_{name}.npz',**rr);summary['datasets'][dsname][name]=dict(regret=stats(rr['regret'].mean(1)),accuracy=stats(rr['accuracy'].mean(1)))
        (out/'shuffled_summary.json').write_text(json.dumps(summary,indent=2))
        print('completed',name,flush=True)
    print('DONE',flush=True)
if __name__=='__main__':main()
