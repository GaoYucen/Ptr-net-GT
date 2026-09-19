"""Separate, post-hoc official-budget calibration; no training or model selection."""
import argparse,json,math,time
from pathlib import Path
from types import SimpleNamespace
import torch
from strong_adapter import Host,gather

@torch.no_grad()
def decode(host,x,p):
    net=host.net;b,n,_=x.shape;dist=torch.cdist(x,x)
    net.pre_forward(SimpleNamespace(problems=x,dist=dist,log_scale=math.log2(n)))
    cur=torch.arange(p,device=x.device)[None].expand(b,p)
    net.decoder.set_q1(gather(net.encoded_nodes,cur))
    mask=torch.zeros(b,p,n,device=x.device).scatter_(2,cur[...,None],-torch.inf)
    seq=[cur]
    for _ in range(n-1):
        cur_dist=dist.gather(1,cur[...,None].expand(b,p,n))
        cur,_=net(SimpleNamespace(batch_size=b,pomo_size=p,current_node=cur,ninf_mask=mask),cur_dist)
        seq.append(cur);mask=mask.clone().scatter_(2,cur[...,None],-torch.inf)
    pi=torch.stack(seq,-1)
    assert (pi.sort(-1).values==torch.arange(n,device=x.device)).all()
    ordered=x[:,None].expand(b,p,n,2).gather(2,pi[...,None].expand(b,p,n,2))
    return (ordered-ordered.roll(-1,2)).norm(dim=-1).sum(-1),pi

def augment(x):
    a,b=x[...,0],x[...,1]
    return torch.cat([torch.stack(q,-1) for q in
        [(a,b),(1-a,b),(a,1-b),(1-a,1-b),(b,a),(1-b,a),(b,1-a),(1-b,1-a)]],0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.wait:
        print('WAITING_FOR_ICAM_TRAINING',flush=True)
        while json.loads((a.root/'icam/status.json').read_text()).get('stage')!='complete':time.sleep(30)
        # The trace worker is short and uses the same freed GPU; allow it to finish.
        while not (a.root/'icam-selector-traces.json').exists():time.sleep(30)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.16)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    host=Host('icam').cuda().eval()
    torch.manual_seed(923071)
    for n in (50,200):
        x=torch.rand(2,n,2,device='cuda')
        with torch.no_grad(): native=host.native(x)
        _,pi=decode(host,x,1);assert torch.equal(pi[:,0],native)
    d=torch.load(a.root/'icam/datasets.pt',weights_only=True)
    dest=a.root/'icam-multistart';dest.mkdir(exist_ok=True)
    summary=dict(protocol='Separate official native checkpoint, all N starting nodes and 8 geometric augmentations. Post-hoc evaluation-budget diagnostic, not part of matched training arms.',
                 training_updates=0,batch=1,single_start_helper_verified=True,cases={})
    for case in ('test200','cluster200','test500'):
        original=[];multi=[];auged=[];torch.cuda.synchronize();begin=time.monotonic()
        for i,x0 in enumerate(d[case].split(1)):
            costs,_=decode(host,augment(x0.cuda()),x0.size(1))
            original.append(costs[0,0].cpu());multi.append(costs[0].min().cpu());auged.append(costs.min().cpu())
            if (i+1)%32==0:print(case,i+1,flush=True)
        torch.cuda.synchronize();elapsed=time.monotonic()-begin
        orig=torch.stack(original);ms=torch.stack(multi);aug=torch.stack(auged)
        assert (aug<=ms+1e-6).all() and (ms<=orig+1e-6).all()
        torch.save(ms,dest/f'{case}-multistart-costs.pt');torch.save(aug,dest/f'{case}-multistart-aug8-costs.pt')
        ref=torch.load(a.root/f'reference/{case}-costs.pt',weights_only=True)
        old=torch.load(a.root/f'icam/native-{case}-costs.pt',weights_only=True)
        summary['cases'][case]=dict(instances=len(orig),starts=x0.size(1),
            greedy_first_start_mean=float(orig.double().mean()),multistart_mean=float(ms.double().mean()),
            multistart_aug8_mean=float(aug.double().mean()),reference_mean=float(ref.mean()),
            multistart_gap_percent=float(100*(ms.double().mean()/ref.mean()-1)),
            multistart_aug8_gap_percent=float(100*(aug.double().mean()/ref.mean()-1)),
            first_start_max_cost_difference_from_original_evaluation=float((orig-old).abs().max()),
            total_evaluation_seconds=elapsed)
        (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(summary['cases'][case],flush=True)
    (dest/'status.json').write_text(json.dumps(dict(stage='complete')))
if __name__=='__main__':main()
