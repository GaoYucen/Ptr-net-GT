"""Pre-registered paired TSP200 adaptation study, isolated output directory."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, subprocess, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import torch
from strong_adapter import Adapter, tour_cost
from verify_strong import valid

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tsha(x): return hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()
def sync(): torch.cuda.synchronize()
def save_json(p,v):
    p=Path(p); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)); tmp.replace(p)
def state(m): return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def coords(seed,count,n,clustered=False):
    g=torch.Generator().manual_seed(seed)
    if not clustered: return torch.rand(count,n,2,generator=g)
    centers=.1+.8*torch.rand(count,5,2,generator=g)
    ids=torch.randint(5,(count,n),generator=g)
    return (centers.gather(1,ids[...,None].expand(-1,-1,2))+.035*torch.randn(count,n,2,generator=g)).clamp(0,1)

def improve(job):
    x,pi=job; pi=pi.copy(); n=len(pi)
    d=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    ii,jj=np.indices((n,n)); legal=(jj>ii+1)&~((ii==0)&(jj==n-1))
    for _ in range(200):
        a=pi; b=np.roll(pi,-1)
        gain=d[a,b][:,None]+d[a,b][None,:]-d[a[:,None],a[None,:]]-d[b[:,None],b[None,:]]
        gain[~legal]=-np.inf
        i,j=np.unravel_index(np.argmax(gain),gain.shape)
        if gain[i,j]<1e-9: break
        pi[i+1:j+1]=pi[i+1:j+1][::-1]
    return pi

@torch.no_grad()
def evaluate(m,data,mode,batch=16,aug=1):
    m.eval(); out=[]; deviations=[]; rng=torch.Generator(device='cuda').manual_seed(995731)
    sync(); begin=time.monotonic()
    for x0 in data.split(batch):
        x=x0.cuda()
        if aug==8:
            a,b=x[...,0],x[...,1]
            x=torch.cat([torch.stack(pair,-1) for pair in
                         [(a,b),(1-a,b),(a,1-b),(1-a,1-b),(b,a),(1-b,a),(b,1-a),(1-b,1-a)]],0)
        if mode=='route':
            pi=m.host.native(x)
            successor=torch.empty_like(pi).scatter(1,pi,pi.roll(-1,1))
            o=dict(cost=tour_cost(x,pi),successor=successor,deviation=torch.zeros(len(x),device=x.device))
        else:
            o=m.rollout(x,mode,gen=rng)
        valid(o['successor'])
        out.append(o['cost'].reshape(aug,len(x0)).amin(0).cpu())
        deviations.append(float(o['deviation'].mean()))
    sync(); values=torch.cat(out)
    return values,dict(mean=float(values.double().mean()),seconds=time.monotonic()-begin,
                       source_deviation=sum(deviations)/len(deviations))

def warmup(m,data,tours,val,out,mode,a):
    opt=torch.optim.Adam(m.host.parameters(),lr=a.warmup_lr)
    g=torch.Generator(device='cuda').manual_seed(73319)
    dg=torch.Generator().manual_seed(63319)
    hist=[]; losses=[]; elapsed=0.; best=float('inf'); best_state=None
    for step in range(a.warmup_steps+1):
        if step:
            ids=torch.randint(len(data),(a.warmup_batch,),generator=dg)
            depth=int(torch.randint(0,data.size(1)-1,(1,),generator=dg))
            sync(); begin=time.monotonic(); opt.zero_grad()
            loss=m.warmup_loss(data[ids].cuda(),tours[ids].cuda(),g,mode,depth=depth)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite warmup loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.host.parameters(),1.); opt.step()
            sync(); elapsed+=time.monotonic()-begin; losses.append(float(loss))
        if step%a.warmup_eval_every==0 or step==a.warmup_steps:
            vg=torch.Generator(device='cuda').manual_seed(411119)
            with torch.no_grad():
                # Held-out tours are generated before any updates.
                vl=sum(float(m.warmup_loss(val[0][i:i+8].cuda(),val[1][i:i+8].cuda(),vg,mode))
                       for i in range(0,len(val[0]),8))/(len(val[0])/8)
            if vl<best: best=vl; best_state=state(m); best_step=step
            record=dict(stage='warmup',mode=mode,step=step,val_endpoint_nll=vl,
                        recent_loss=sum(losses)/max(len(losses),1),training_seconds=elapsed)
            hist.append(record); losses=[]; print(json.dumps(record),flush=True)
            save_json(out/f'warmup-{mode}-history.json',hist)
    torch.save(dict(model=best_state,best_step=best_step,best_val_nll=best,training_seconds=elapsed),
               out/f'warmup-{mode}.pt')
    return best_state,elapsed

def train_arm(base,host,mode,seed,datasets,out,a,warm_seconds):
    folder=out/f'{mode}-seed{seed}'; folder.mkdir()
    torch.manual_seed(seed); m=Adapter(host).cuda(); m.load_state_dict(base)
    # Independent selector initialization; warmup only learns host parameters.
    for module in m.selector.modules():
        if hasattr(module,'reset_parameters'): module.reset_parameters()
    opt=torch.optim.Adam([{'params':m.host.parameters(),'lr':a.host_lr},
                          {'params':m.selector.parameters(),'lr':a.source_lr}])
    dg=torch.Generator().manual_seed(seed+901000)
    ag=torch.Generator(device='cuda').manual_seed(seed+801000)
    best=float('inf'); hist=[]; elapsed=0.; losses=[]; costs=[]
    all_start=time.monotonic()
    for step in range(a.steps+1):
        if step:
            x=torch.rand(a.batch,200,2,generator=dg).repeat_interleave(a.rollouts,0).cuda()
            opt.zero_grad(); sync(); begin=time.monotonic()
            result=m.rollout(x,mode,True,ag,checkpoint_steps=True)
            c=result['cost'].reshape(a.batch,a.rollouts)
            advantage=(c-c.mean(1,keepdim=True))*a.rollouts/(a.rollouts-1)
            loss=(advantage.detach()*result['ll'].reshape_as(c)).mean()
            if not torch.isfinite(loss): raise RuntimeError('nonfinite policy loss')
            loss.backward()
            norms={name:sum(float(p.grad.square().sum()) for k,p in m.named_parameters()
                             if k.startswith(name) and p.grad is not None)**.5 for name in ('host','selector')}
            if not all(np.isfinite(list(norms.values()))): raise RuntimeError('nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
            sync(); elapsed+=time.monotonic()-begin; losses.append(float(loss)); costs.append(float(c.mean()))
        if step%a.eval_every==0 or step==a.steps:
            values,metrics=evaluate(m,datasets['val200'],mode,a.eval_batch)
            if metrics['mean']<best:
                best=metrics['mean']; best_step=step
                torch.save(dict(model=state(m),best_step=step,seed=seed,mode=mode),folder/'best.pt')
            record=dict(stage='joint',host=host,mode=mode,seed=seed,step=step,validation=metrics,
                        training_seconds=elapsed,warmup_seconds=warm_seconds,
                        recent_sample_cost=sum(costs)/max(len(costs),1),
                        recent_loss=sum(losses)/max(len(losses),1),gradient_norm=norms if step else {})
            hist.append(record); costs=[]; losses=[]
            print(json.dumps(record),flush=True); save_json(folder/'history.json',hist)
            save_json(out/'status.json',record)
    torch.save(dict(model=state(m),step=a.steps),folder/'final.pt')
    m.load_state_dict(torch.load(folder/'best.pt',weights_only=False,map_location='cpu')['model'])
    evaluation={}
    for name,data in datasets.items():
        if name.startswith('val'): continue
        v,metrics=evaluate(m,data,mode,a.eval_batch)
        torch.save(v,folder/f'{name}-costs.pt'); evaluation[name]=metrics
    v,metrics=evaluate(m,datasets['test200'],mode,max(1,a.eval_batch//8),aug=8)
    torch.save(v,folder/'test200-aug8-costs.pt'); evaluation['test200-aug8']=metrics
    summary=dict(host=host,mode=mode,seed=seed,best_step=best_step,steps=a.steps,
                 training_seconds=elapsed,warmup_seconds=warm_seconds,
                 wall_seconds=time.monotonic()-all_start,evaluation=evaluation,
                 parameters=sum(p.numel() for p in m.parameters()),
                 shared_warmup_seed=True,official_checkpoint_sha=sha(m.host.path))
    save_json(folder/'summary.json',summary); print('DONE_ARM '+json.dumps(summary),flush=True)
    del m; torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',choices=['am','icam'],required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--warmup-steps',type=int,default=2500); p.add_argument('--warmup-batch',type=int,default=32)
    p.add_argument('--warmup-eval-every',type=int,default=250); p.add_argument('--warmup-lr',type=float,default=1e-5)
    p.add_argument('--teacher-size',type=int,default=2048); p.add_argument('--steps',type=int,default=300)
    p.add_argument('--batch',type=int,default=4); p.add_argument('--rollouts',type=int,default=4)
    p.add_argument('--host-lr',type=float,default=1e-5); p.add_argument('--source-lr',type=float,default=1e-4)
    p.add_argument('--eval-every',type=int,default=50); p.add_argument('--eval-batch',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[1234,4321,2468])
    p.add_argument('--modes',nargs='+',default=['route','random','learned'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(918123)
    torch.cuda.set_per_process_memory_fraction(.32)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if a.out.exists() and not a.resume: raise ValueError('output exists')
    a.out.mkdir(parents=True,exist_ok=True)
    off=100000 if a.smoke else 0
    datasets={'val200':coords(6291901+off,8 if a.smoke else 64,200),
              'test200':coords(6291902+off,8 if a.smoke else 256,200),
              'cluster200':coords(6291903+off,8 if a.smoke else 64,200,True),
              'test500':coords(6291904+off,8 if a.smoke else 64,500)}
    m=Adapter(a.host).cuda().eval(); config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(checkpoint_sha=sha(m.host.path),data_hash={k:tsha(v) for k,v in datasets.items()},
                  source_hash={f.name:sha(f) for f in Path(__file__).parent.glob('*strong*.py')},
                  torch_version=torch.__version__,device=torch.cuda.get_device_name(),
                  protocol='TSP200 official-pretrained joint adaptation; no OPT claims; new heldout seeds',
                  shared_warmup=True,normalization='official eval statistics fixed; weights trainable',
                  GPU=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if not a.resume: save_json(a.out/'config.json',config); torch.save(datasets,a.out/'datasets.pt')
    else:
        previous=json.loads((a.out/'config.json').read_text())
        for k in ('checkpoint_sha','data_hash','warmup_steps','steps','modes','seeds'):
            assert previous[k]==config[k],('resume mismatch',k)
    teacher_path=a.out/'teachers.pt'
    if teacher_path.exists(): teacher=torch.load(teacher_path,weights_only=False)
    else:
        tr=coords(6291910+off,a.teacher_size,200); va=coords(6291911+off,8 if a.smoke else 64,200)
        allx=torch.cat((tr,va)); paths=[]; begin=time.monotonic()
        with torch.no_grad():
            for i,x in enumerate(allx.split(32)):
                paths.append(m.host.native(x.cuda()).cpu())
                if i%16==0: print('TEACHER_NATIVE',i,flush=True)
        native=torch.cat(paths)
        with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
            improved=list(pool.map(improve,zip(allx.numpy(),native.numpy()),chunksize=16))
        pi=torch.tensor(np.stack(improved)); cost=tour_cost(allx,pi); native_cost=tour_cost(allx,native)
        assert (cost<=native_cost+1e-5).all()
        teacher=dict(train_x=tr,train_tour=pi[:len(tr)],val_x=va,val_tour=pi[len(tr):],
                     seconds=time.monotonic()-begin,native_mean=float(native_cost.mean()),
                     teacher_mean=float(cost.mean()),native_paths=native)
        torch.save(teacher,teacher_path)
        print('TEACHERS',teacher['native_mean'],teacher['teacher_mean'],teacher['seconds'],flush=True)
    if not (a.out/'native-summary.json').exists():
        native_eval={}
        for name,data in datasets.items():
            if name.startswith('val'): continue
            v,metrics=evaluate(m,data,'route',a.eval_batch)
            torch.save(v,a.out/f'native-{name}-costs.pt'); native_eval[name]=metrics
        v,metrics=evaluate(m,datasets['test200'],'route',1,8)
        torch.save(v,a.out/'native-test200-aug8-costs.pt'); native_eval['test200-aug8']=metrics
        save_json(a.out/'native-summary.json',native_eval)
    initial=state(m); warm={}
    for mode in ('route','random'):
        path=a.out/f'warmup-{mode}.pt'
        if path.exists():
            ck=torch.load(path,weights_only=False); warm[mode]=(ck['model'],ck['training_seconds'])
        else:
            m.load_state_dict(initial)
            warm[mode]=warmup(m,teacher['train_x'],teacher['train_tour'],
                             (teacher['val_x'],teacher['val_tour']),a.out,mode,a)
    del m; torch.cuda.empty_cache()
    for seed in a.seeds:
        for mode in a.modes:
            folder=a.out/f'{mode}-seed{seed}'
            if (folder/'summary.json').exists(): continue
            if folder.exists():
                # Preserve interrupted run, never mix optimizers or partial logs.
                folder.rename(folder.with_name(folder.name+f'-interrupted-{int(time.time())}'))
            base,ws=warm['route' if mode in ('route','capacity') else 'random']
            train_arm(base,a.host,mode,seed,datasets,a.out,a,ws)
    save_json(a.out/'status.json',dict(stage='complete',host=a.host,seeds=a.seeds,modes=a.modes))

if __name__=='__main__': main()
