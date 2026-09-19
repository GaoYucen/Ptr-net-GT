from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import torch
from component_adapter import Model, teacher_state, State
from strong_adapter import tour_cost
from run_strong import improve
from verify_strong import valid


def save_json(path,x):
    path=Path(path); temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(x,indent=2)); temp.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tsha(x): return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()
def sync(): torch.cuda.synchronize()

def coords(seed,count,n,clustered=False):
    g=torch.Generator().manual_seed(seed)
    if not clustered: return torch.rand(count,n,2,generator=g)
    centers=.1+.8*torch.rand(count,5,2,generator=g)
    ids=torch.randint(5,(count,n),generator=g)
    return (centers.gather(1,ids[...,None].expand(-1,-1,2))+.035*torch.randn(count,n,2,generator=g)).clamp(0,1)

@torch.no_grad()
def encode(m,x): return torch.cat([m.host.encode(y.cuda()).cpu() for y in x.split(32)])

@torch.no_grad()
def teachers(m,x):
    native=torch.cat([m.host.native(y.cuda()).cpu() for y in x.split(32)])
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        pi=torch.tensor(np.stack(list(pool.map(improve,zip(x.numpy(),native.numpy()),chunksize=16))))
    assert (tour_cost(x,pi)<=tour_cost(x,native)+1e-5).all()
    return pi,native

def prepare(out,a):
    out.mkdir(parents=True,exist_ok=False); m=Model(kind=a.host).cuda().eval()
    config=dict(protocol=Path(__file__).with_name('PROTOCOL.md').read_text(),
                sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                checkpoint_sha=sha(m.host.path),torch_version=torch.__version__,device=torch.cuda.get_device_name(),
                residual_parameters=sum(p.numel() for p in m.residual.parameters()),
                train_seeds=[73019101,73019102],development_seeds=[73019103,73019104],
                final_test_seeds=[73019111,73019112,73019113,73019114],steps=a.steps,batch=a.batch,host=a.host)
    save_json(out/'config.json',config)
    train_x=torch.cat([coords(73019101,1024,100),coords(73019102,1024,100,True)])
    val_x=torch.cat([coords(73019103,64,100),coords(73019104,64,100,True)])
    all_x=torch.cat([train_x,val_x]); start=time.monotonic()
    enc=encode(m,all_x); print('ENCODED',time.monotonic()-start,flush=True)
    pi,native=teachers(m,all_x)
    data=dict(train_x=train_x,train_enc=enc[:2048],train_pi=pi[:2048],
              val_x=val_x,val_enc=enc[2048:],val_pi=pi[2048:],
              val_native=tour_cost(val_x,native[2048:]))
    torch.save(data,out/'data.pt')
    save_json(out/'data_manifest.json',dict(hashes={k:tsha(v) for k,v in data.items()},
                                          native_mean=float(tour_cost(all_x,native).mean()),
                                          teacher_mean=float(tour_cost(all_x,pi).mean()),
                                          seconds=time.monotonic()-start))
    print('PREPARED',time.monotonic()-start,flush=True)

@torch.no_grad()
def evaluate(m,x,enc,mode,batch=32):
    g=torch.Generator(device='cuda').manual_seed(9417319)
    out=[]; sync(); start=time.monotonic()
    for i in range(0,len(x),batch):
        cost,succ=m.rollout(x[i:i+batch].cuda(),enc[i:i+batch].cuda(),mode,g)
        valid(succ); out.append(cost.cpu())
    sync(); costs=torch.cat(out)
    return costs,dict(mean=float(costs.double().mean()),seconds=time.monotonic()-start)

@torch.no_grad()
def conditional_eval(m,x,enc,pi,mode):
    g=torch.Generator(device='cuda').manual_seed(9417320)
    losses=[]; accuracies=[]
    for fraction in [.25,.5,.75]:
        for i in range(0,len(x),32):
            loss,acc=m.conditional(enc[i:i+32].cuda(),x[i:i+32].cuda(),pi[i:i+32].cuda(),
                                  int(x.size(1)*fraction),mode,g)
            losses.append(loss.cpu()); accuracies.append(acc.cpu())
    return dict(nll=float(torch.cat(losses).double().mean()),accuracy=float(torch.cat(accuracies).float().mean()))

def smoke(out,a):
    torch.manual_seed(391); m=Model(kind=a.host).cuda().eval(); checks=[]
    x=torch.rand(4,20,2,device='cuda'); enc=m.host.encode(x)
    pi=m.host.native(x); original=torch.empty_like(pi).scatter(1,pi,pi.roll(-1,1))
    cost,succ=m.rollout(x,enc,'route'); assert torch.equal(succ,original); valid(succ)
    checks.append(dict(native_route_exact=True))
    for mode in ['route','random']:
        for depth in [0,7,18]:
            g=torch.Generator(device='cuda').manual_seed(9417320)
            st,tail,target=teacher_state(enc,pi,depth,mode,g)
            for row in range(len(x)):
                for node in range(x.size(1)):
                    members=st.start[row].eq(st.start[row,node])
                    assert torch.allclose(st.mean[row,node],enc[row,members].mean(0),atol=1e-6)
                    assert int(st.size[row,node])==int(members.sum())
                    assert st.pred[row,st.start[row,node]]<0 and st.succ[row,st.end[row,node]]<0
            loss,_=m.conditional(enc,x,pi,depth,mode,torch.Generator(device='cuda').manual_seed(9417320))
            m.zero_grad(); loss.mean().backward(); assert torch.isfinite(loss).all()
            assert all(p.grad is None for p in m.host.parameters())
            checks.append(dict(mode=mode,depth=depth,fragments_valid=True,nll=float(loss.mean())))
    # Verify equivariance with nonzero residual, holding the physical source fixed.
    with torch.no_grad(): m.residual.scale.fill_(1); m.residual.geometry[-1].weight.normal_(0,.1)
    st,tail,target=teacher_state(enc,pi,7,'random',torch.Generator(device='cuda').manual_seed(11))
    perm=torch.randperm(20,device='cuda'); inv=perm.argsort()
    stp=State(enc[:,perm]); stp.start=inv[st.start[:,perm]]; stp.end=inv[st.end[:,perm]]
    stp.mean=st.mean[:,perm]; stp.size=st.size[:,perm]; stp.steps=st.steps
    stp.pred=torch.where(st.pred[:,perm]<0,-1,inv[st.pred[:,perm].clamp_min(0)])
    stp.succ=torch.where(st.succ[:,perm]<0,-1,inv[st.succ[:,perm].clamp_min(0)])
    for aware in [False,True]:
        m.aware=aware
        l=m.logits(enc,x,st,tail)[:,perm]; lp=m.logits(enc[:,perm],x[:,perm],stp,inv[tail])
        err=float((l[l.isfinite()]-lp[l.isfinite()]).abs().max()); assert err<1e-4
        checks.append(dict(aware=aware,permutation_max_error=err))
    x=torch.rand(64,100,2,device='cuda'); enc=m.host.encode(x); pi=m.host.native(x)
    opt=torch.optim.Adam(m.residual.parameters(),lr=3e-4)
    sync(); start=time.monotonic()
    for t in range(100):
        opt.zero_grad(); loss,_=m.conditional(enc,x,pi,t%98,'random',None)
        loss.mean().backward(); opt.step()
    sync(); checks.append(dict(updates=100,seconds=time.monotonic()-start,batch=64,n=100,
                             peak_mib=torch.cuda.max_memory_allocated()/2**20))
    cost,metrics=evaluate(m,x.cpu(),enc.cpu(),'random'); checks.append(dict(decode=metrics))
    save_json(out,checks); print(json.dumps(checks,indent=2),flush=True)

def train(out,a):
    data=torch.load(out/'data.pt',weights_only=False); mode=a.mode
    for seed in a.seeds:
        for aware in [False,True]:
            folder=out/f'{mode}-{"aware" if aware else "blind"}-seed{seed}'
            folder.mkdir(exist_ok=False)
            torch.manual_seed(seed); m=Model(aware,kind=a.host).cuda().eval()
            save_json(folder/'training_manifest.json',dict(host=a.host,
                sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                protocol_sha=sha(Path(__file__).with_name('PROTOCOL.md')),data_sha=sha(out/'data.pt')))
            # Only residual parameters are in the optimizer.
            opt=torch.optim.Adam(m.residual.parameters(),lr=3e-4)
            dg=torch.Generator().manual_seed(seed+500000)
            ag=torch.Generator(device='cuda').manual_seed(seed+600000)
            tx=data['train_x'].cuda(); te=data['train_enc'].cuda(); tp=data['train_pi'].cuda()
            best=float('inf'); best_step=None; history=[]; losses=[]; elapsed=0.
            sync(); all_start=time.monotonic()
            for step in range(a.steps+1):
                if step:
                    ids=torch.randint(len(tx),(a.batch,),generator=dg,device='cpu').cuda()
                    depth=int(torch.randint(0,99,(1,),generator=dg))
                    sync(); begin=time.monotonic(); opt.zero_grad()
                    loss,_=m.conditional(te[ids],tx[ids],tp[ids],depth,mode,ag)
                    loss=loss.mean(); assert torch.isfinite(loss)
                    loss.backward(); grad=torch.nn.utils.clip_grad_norm_(m.residual.parameters(),1.)
                    assert torch.isfinite(grad); opt.step(); sync()
                    elapsed+=time.monotonic()-begin; losses.append(float(loss))
                if step%1000==0 or step==a.steps:
                    costs,metrics=evaluate(m,data['val_x'],data['val_enc'],mode)
                    normalized=float((costs/data['val_native']).mean())
                    condition=conditional_eval(m,data['val_x'],data['val_enc'],data['val_pi'],mode)
                    if normalized<best:
                        best=normalized; best_step=step
                        torch.save(dict(residual=m.residual.state_dict(),step=step,seed=seed,aware=aware,mode=mode),folder/'best.pt')
                        torch.save(costs,folder/'best-development-costs.pt')
                    rec=dict(mode=mode,aware=aware,seed=seed,step=step,validation=metrics,
                             normalized_validation=normalized,conditional=condition,best_step=best_step,
                             loss=float(np.mean(losses)) if losses else None,training_seconds=elapsed,
                             wall_seconds=time.monotonic()-all_start)
                    history.append(rec); save_json(folder/'history.json',history); losses=[]
                    print(json.dumps(rec),flush=True)
            torch.save(dict(residual=m.residual.state_dict(),step=a.steps),folder/'final.pt')
            save_json(folder/'training_summary.json',dict(seed=seed,mode=mode,aware=aware,best_step=best_step,
                      training_seconds=elapsed,wall_seconds=time.monotonic()-all_start,steps=a.steps,
                      parameters=sum(p.numel() for p in m.residual.parameters())))
            del m; torch.cuda.empty_cache()

def finalize(out,a):
    folders=[out/f'{mode}-{aw}-seed{seed}' for mode in ['route','random']
             for aw in ['blind','aware'] for seed in a.seeds]
    assert all((f/'training_summary.json').exists() for f in folders),'All arms must finish before test creation'
    save_json(out/'frozen_checkpoint_manifest.json',dict(checkpoints={f.name:sha(f/'best.pt') for f in folders},
                                                      frozen_unix_time=time.time()))
    assert not (out/'final_test_data.pt').exists(),'Do not rerun final test after inspection'
    m=Model(kind=a.host).cuda().eval(); sets={}; manifests={}
    for name,seed,count,n,clustered in [('uniform100',73019111,256,100,False),('cluster100',73019112,128,100,True),
                                      ('uniform200',73019113,128,200,False),('cluster200',73019114,128,200,True)]:
        x=coords(seed,count,n,clustered); sync(); begin=time.monotonic(); enc=encode(m,x); sync()
        encoding_seconds=time.monotonic()-begin; pi,native=teachers(m,x)
        sets[name]=dict(x=x,enc=enc,pi=pi,native_cost=tour_cost(x,native),teacher_cost=tour_cost(x,pi))
        manifests[name]=dict(seed=seed,count=count,n=n,clustered=clustered,sha=tsha(x),
                             native_mean=float(sets[name]['native_cost'].mean()),teacher_mean=float(sets[name]['teacher_cost'].mean()),
                             encoding_seconds=encoding_seconds)
    torch.save(sets,out/'final_test_data.pt'); save_json(out/'final_test_manifest.json',manifests)
    for folder in folders:
        ck=torch.load(folder/'best.pt',weights_only=False); m.aware=ck['aware']; m.residual.load_state_dict(ck['residual'])
        summary=dict(seed=ck['seed'],aware=ck['aware'],mode=ck['mode'],selected_step=ck['step'],test={})
        for name,s in sets.items():
            c,met=evaluate(m,s['x'],s['enc'],ck['mode']); torch.save(c,folder/f'{name}-costs.pt')
            condition=conditional_eval(m,s['x'],s['enc'],s['pi'],ck['mode'])
            summary['test'][name]=dict(**met,conditional=condition,
                native_relative_percent=float(((c/s['native_cost'])-1).mean()*100),
                teacher_relative_percent=float(((c/s['teacher_cost'])-1).mean()*100))
        save_json(folder/'test_summary.json',summary); print('FINAL '+json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','smoke','train','finalize'])
    p.add_argument('--out',type=Path,required=True); p.add_argument('--mode',choices=['route','random'])
    p.add_argument('--host',choices=['icam','am'],default='icam')
    p.add_argument('--steps',type=int,default=12000); p.add_argument('--batch',type=int,default=64)
    p.add_argument('--seeds',type=int,nargs='+',default=[12031,12037,12041]); a=p.parse_args()
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.28)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if a.action=='smoke': smoke(a.out,a)
    elif a.action=='prepare': prepare(a.out,a)
    elif a.action=='train': train(a.out,a)
    else: finalize(a.out,a)

if __name__=='__main__': main()
