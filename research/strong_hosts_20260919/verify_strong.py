import argparse, copy, json, time
from pathlib import Path
import torch
from strong_adapter import Adapter, Host, Forest, tour_cost

def valid(succ):
    n=succ.size(1)
    assert (succ.sort(1).values==torch.arange(n,device=succ.device)).all()
    current=torch.zeros(succ.size(0),device=succ.device,dtype=torch.long)
    seen=torch.zeros_like(succ,dtype=torch.bool)
    rows=torch.arange(succ.size(0),device=succ.device)
    for _ in range(n):
        assert not seen[rows,current].any()
        seen[rows,current]=True
        current=succ[rows,current]
    assert (current==0).all()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--host',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); torch.set_num_threads(2); torch.manual_seed(92301)
    torch.cuda.set_per_process_memory_fraction(.32)
    m=Adapter(a.host).cuda().eval(); checks=[]
    for n in (10,50,100,200):
        x=torch.rand(2,n,2,device='cuda')
        with torch.no_grad():
            native=m.host.native(x)
            out=m.rollout(x,'route')
        cost=tour_cost(x,native)
        expected=torch.empty_like(native).scatter(1,native,native.roll(-1,1))
        assert torch.equal(expected,out['successor']), (a.host,n,'native action mismatch')
        assert torch.allclose(cost,out['cost'],atol=2e-5,rtol=1e-6)
        checks.append(dict(n=n,native_actions_exact=True,cost_max_error=float((cost-out['cost']).abs().max())))
    x=torch.rand(2,8,2,device='cuda')
    for mode in ('route','learned','capacity','random','fixed'):
        with torch.no_grad():
            out=m.rollout(x,mode,True,torch.Generator(device='cuda').manual_seed(192))
            valid(out['successor'])
        replay=m.rollout(x,mode,False,actions=out,checkpoint_steps=False)
        assert torch.allclose(out['ll'],replay['ll'],atol=2e-5,rtol=1e-6)
        m.zero_grad(); replay['ll'].mean().backward()
        g1={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None}
        m.zero_grad()
        replay2=m.rollout(x,mode,False,actions=out,checkpoint_steps=True)
        replay2['ll'].mean().backward()
        err=max(float((p.grad-g1[k]).abs().max()) for k,p in m.named_parameters() if k in g1)
        assert err<1e-4,(mode,err)
        norms={name:sum(float(p.grad.square().sum()) for k,p in m.named_parameters()
                        if k.startswith(name) and p.grad is not None)**.5 for name in ('host','selector')}
        assert norms['host']>0 and (mode not in ('learned','capacity') or norms['selector']>0)
        checks.append(dict(mode=mode,valid_tours=True,replay=True,checkpoint_gradient_max_error=err,gradient_norm=norms))
    for mode in ('route','random'):
        with torch.no_grad(): tours=m.host.native(x)
        m.zero_grad(); loss=m.warmup_loss(x,tours,torch.Generator(device='cuda').manual_seed(12),mode)
        loss.backward(); assert torch.isfinite(loss)
        assert all(p.grad is None for p in m.selector.parameters())
        checks.append(dict(warmup_mode=mode,loss=float(loss),selector_unsupervised=True))
    # Profile representative full joint update before allocating study budget.
    x=torch.rand(4,200,2,device='cuda'); m.zero_grad()
    torch.cuda.synchronize(); begin=time.monotonic()
    out=m.rollout(x,'learned',True,checkpoint_steps=True)
    out['ll'].mean().backward(); torch.cuda.synchronize()
    checks.append(dict(profile_nodes=200,batch=4,full_update_seconds=time.monotonic()-begin,
                       peak_mib=torch.cuda.max_memory_allocated()/2**20))
    Path(a.out).write_text(json.dumps(checks,indent=2)); print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
