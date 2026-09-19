from __future__ import annotations
import copy, importlib.util, json, math, sys, time
from pathlib import Path
import torch
from torch import nn

REPO=Path('/workspace/groupopt-modern-hosts/ICAM')
TSP=REPO/'ICAM_TSP'
OUT=Path('/workspace/groupopt-icam-ascc-h2')
DEVICE=torch.device('cuda')
sys.path.insert(0,str(TSP))
sys.path.insert(0,str(REPO))

spec=importlib.util.spec_from_file_location('icam_model',TSP/'TSPModel_ICAM.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
TSPModel=mod.TSPModel
PARAMS=dict(
    embedding_dim=128,
    sqrt_embedding_dim=128**0.5,
    encoder_layer_num=12,
    logit_clipping=50,
    ff_hidden_dim=512,
    eval_type='greedy',
)

def build_host():
    m=TSPModel(**PARAMS).to(DEVICE)
    ck=torch.load(REPO/'pretrained/icam_tsp.pt',map_location=DEVICE,weights_only=False)
    m.load_state_dict(ck['model_state_dict'],strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m

def make_fixed(seed,count,n=50):
    g=torch.Generator(device='cpu').manual_seed(seed)
    return torch.rand(count,n,2,generator=g)

def repeat_rollouts(coords,k):
    b,n,_=coords.shape
    cr=coords[:,None,:,:].expand(b,k,n,2).reshape(b*k,n,2)
    starts=(torch.arange(k,device=coords.device)%n)[None,:].expand(b,k).reshape(-1)
    return cr,starts

def gather_nodes(x,idx):
    return x.gather(1,idx[:,None,None].expand(-1,1,x.size(-1))).squeeze(1)

def tour_cost(coords,seq):
    ordered=coords.gather(1,seq[:,:,None].expand(-1,-1,2))
    return ((ordered-ordered.roll(-1,1))**2).sum(-1).sqrt().sum(-1)

@torch.no_grad()
def host_encode(host,coords):
    dist=torch.cdist(coords,coords,p=2,compute_mode='donot_use_mm_for_euclid_dist')
    log_scale=math.log2(coords.size(1))
    enc=host.encoder(coords,dist,log_scale)
    return enc,dist,log_scale

@torch.no_grad()
def host_probs(host,enc,dist,log_scale,first_idx,tail_idx,valid_mask):
    # Each rollout is represented as its own batch item with pomo=1.
    dec=host.decoder
    dec.set_kv(enc)
    first_e=gather_nodes(enc,first_idx)[:,None,:]
    tail_e=gather_nodes(enc,tail_idx)[:,None,:]
    dec.set_q1(first_e)
    cur_dist=dist.gather(1,tail_idx[:,None,None].expand(-1,1,dist.size(-1)))
    ninf=torch.full(valid_mask.shape,float('-inf'),device=DEVICE)
    ninf[valid_mask]=0.0
    probs=dec(tail_e,cur_dist,log_scale,ninf[:,None,:]).squeeze(1)
    return probs

@torch.no_grad()
def sequential_rollout(host,coords,starts):
    enc,dist,log_scale=host_encode(host,coords)
    b,n,_=coords.shape
    visited=torch.zeros(b,n,dtype=torch.bool,device=DEVICE)
    visited.scatter_(1,starts[:,None],True)
    seq=[starts]; current=starts
    for _ in range(n-1):
        probs=host_probs(host,enc,dist,log_scale,starts,current,~visited)
        nxt=probs.argmax(-1)
        visited.scatter_(1,nxt[:,None],True)
        seq.append(nxt); current=nxt
    return tour_cost(coords,torch.stack(seq,1))

class SourcePolicy(nn.Module):
    def __init__(self,d=128,h=64,continuation_init=2.5):
        super().__init__()
        self.tail=nn.Linear(d,h,bias=False)
        self.start=nn.Linear(d,h,bias=False)
        self.last=nn.Linear(d,h,bias=False)
        self.glob=nn.Linear(d,h,bias=False)
        self.size=nn.Linear(1,h,bias=True)
        self.out=nn.Linear(h,1,bias=False)
        self.continuation_logit=nn.Parameter(torch.tensor(float(continuation_init)))
        nn.init.normal_(self.out.weight,std=0.001)
    def forward(self,enc,comp_start,last_head,comp_size,tail_mask):
        start_e=enc.gather(1,comp_start[:,:,None].expand(-1,-1,enc.size(-1)))
        last_e=gather_nodes(enc,last_head)
        h=(self.tail(enc)+self.start(start_e)+self.last(last_e)[:,None,:]
           +self.glob(enc.mean(1))[:,None,:]+self.size(comp_size[:,:,None]))
        score=self.out(torch.tanh(h)).squeeze(-1)
        nodes=torch.arange(enc.size(1),device=enc.device)[None,:]
        score=score+self.continuation_logit*(nodes==last_head[:,None]).to(score.dtype)
        return score.masked_fill(~tail_mask,-torch.inf)

class ASCCICAM(nn.Module):
    def __init__(self,host,continuation_init=2.5):
        super().__init__()
        self.host=host
        self.source=SourcePolicy(continuation_init=continuation_init)

    def rollout(self,coords,starts,source_greedy,endpoint_greedy,gen=None,route_follow=False,collect_stats=False):
        enc,dist,log_scale=host_encode(self.host,coords)
        b,n,_=coords.shape
        node_idx=torch.arange(n,device=DEVICE)[None,:].expand(b,n)
        succ=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        pred=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        comp=node_idx.clone()
        last_head=starts
        logp=torch.zeros(b,device=DEVICE)
        dev_num=torch.zeros((),device=DEVICE); dev_den=torch.zeros((),device=DEVICE)

        # N-1 merges produce one Hamiltonian path; cycle closure is implicit.
        for step in range(n-1):
            same=comp[:,:,None].eq(comp[:,None,:])
            is_start=pred.lt(0)
            is_end=succ.lt(0)
            comp_start=(same & is_start[:,None,:]).to(torch.int8).argmax(2).long()
            comp_size=same.sum(2).to(enc.dtype)/float(n)

            if step==0:
                tail=starts
            elif route_follow:
                tail=last_head
            else:
                slogits=self.source(enc,comp_start,last_head,comp_size,is_end)
                sprob=torch.softmax(slogits,dim=-1)
                if source_greedy:
                    tail=sprob.argmax(-1)
                else:
                    tail=torch.multinomial(sprob,1,generator=gen).squeeze(1)
                    logp=logp+sprob.gather(1,tail[:,None]).squeeze(1).clamp_min(1e-9).log()
                dev_num=dev_num+(tail!=last_head).float().sum()
                dev_den=dev_den+torch.tensor(float(b),device=DEVICE)

            ct=comp.gather(1,tail[:,None]).squeeze(1)
            first=comp_start.gather(1,tail[:,None]).squeeze(1)
            valid_head=is_start & comp.ne(ct[:,None])
            probs=host_probs(self.host,enc,dist,log_scale,first,tail,valid_head)

            if endpoint_greedy:
                head=probs.argmax(-1)
            else:
                head=torch.multinomial(probs,1,generator=gen).squeeze(1)

            succ.scatter_(1,tail[:,None],head[:,None])
            pred.scatter_(1,head[:,None],tail[:,None])

            ch=comp.gather(1,head[:,None]).squeeze(1)
            comp=torch.where(comp.eq(ch[:,None]),ct[:,None],comp)
            last_head=head

        # One component remains; close unique end -> unique start.
        same=comp[:,:,None].eq(comp[:,None,:])
        is_start=pred.lt(0); is_end=succ.lt(0)
        if not torch.equal(is_start.sum(1),torch.ones(b,device=DEVICE,dtype=torch.long)):
            raise RuntimeError('closure requires one start')
        if not torch.equal(is_end.sum(1),torch.ones(b,device=DEVICE,dtype=torch.long)):
            raise RuntimeError('closure requires one end')
        first=is_start.to(torch.int8).argmax(1).long()
        tail=is_end.to(torch.int8).argmax(1).long()
        succ.scatter_(1,tail[:,None],first[:,None])

        next_xy=coords.gather(1,succ[:,:,None].expand(-1,-1,2))
        cost=((coords-next_xy)**2).sum(-1).sqrt().sum(1)
        stats=None
        if collect_stats:
            stats={
              'source_deviation_fraction':float((dev_num/dev_den.clamp_min(1)).detach()),
              'continuation_logit':float(self.source.continuation_logit.detach())
            }
        return cost,logp,stats

@torch.inference_mode()
def eval_base(host,data,k=8,batch=64):
    vals=[]
    for st in range(0,len(data),batch):
        c=data[st:st+batch].to(DEVICE); cr,s=repeat_rollouts(c,k)
        cost=sequential_rollout(host,cr,s)
        vals.append(cost.view(c.size(0),k).min(1).values.cpu())
    return torch.cat(vals)

@torch.inference_mode()
def eval_ascc(model,data,k=8,batch=64,route_follow=False):
    vals=[]; ss=[]
    model.eval()
    for st in range(0,len(data),batch):
        c=data[st:st+batch].to(DEVICE); cr,s=repeat_rollouts(c,k)
        cost,_,stats=model.rollout(cr,s,True,True,None,route_follow=route_follow,collect_stats=True)
        vals.append(cost.view(c.size(0),k).min(1).values.cpu()); ss.append(stats)
    avg={k:sum(x[k] for x in ss)/len(ss) for k in ss[0]}
    return torch.cat(vals),avg

def clone_state(m):
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def train_source(model,val,steps=600,batch=24,k=4,seed=71001):
    opt=torch.optim.Adam(model.source.parameters(),lr=7e-5)
    dg=torch.Generator(device='cpu').manual_seed(seed)
    ag=torch.Generator(device=DEVICE).manual_seed(seed+1000)
    hist=[]
    v,vs=eval_ascc(model,val,k=8,batch=64)
    best=float(v.mean()); best_step=0; best_state=clone_state(model)
    hist.append({'step':0,'val':best,**vs}); print('ICAM_ASCC',json.dumps(hist[-1]),flush=True)
    checkpoints={100,200,400,600}
    for step in range(1,steps+1):
        coords=torch.rand(batch,50,2,generator=dg).to(DEVICE)
        cr,starts=repeat_rollouts(coords,k)
        cost,lp,tstats=model.rollout(cr,starts,False,True,ag,route_follow=False,collect_stats=True)
        reward=-cost.view(batch,k)
        adv=reward-reward.mean(1,keepdim=True)
        loss=-(adv.detach()*lp.view(batch,k)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.source.parameters(),1.0); opt.step()
        if step in checkpoints:
            v,vs=eval_ascc(model,val,k=8,batch=64)
            vm=float(v.mean())
            row={'step':step,'train_cost':float(cost.mean()),'loss':float(loss),
                 'sample_source_deviation_fraction':tstats['source_deviation_fraction'],
                 'val':vm,**vs}
            hist.append(row); print('ICAM_ASCC',json.dumps(row),flush=True)
            if vm<best:
                best=vm; best_step=step; best_state=clone_state(model)
    model.load_state_dict(best_state,strict=True)
    return hist,best,best_step

def main():
    torch.manual_seed(20260918); torch.cuda.manual_seed_all(20260918)
    test=make_fixed(20260918,1024,50)
    val=make_fixed(20260919,512,50)
    host=build_host()

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    base=eval_base(host,test,k=8,batch=64)
    torch.cuda.synchronize(); base_t=time.perf_counter()-t
    base_peak=torch.cuda.max_memory_allocated()/1024**3

    model=ASCCICAM(copy.deepcopy(host),continuation_init=2.5).to(DEVICE)
    rf,rfstats=eval_ascc(model,test,k=8,batch=64,route_follow=True)
    max_abs=float((rf-base).abs().max()); mism=int(((rf-base).abs()>1e-6).sum())
    if max_abs>1e-5:
        raise RuntimeError(f'ICAM route-follow sanity failed max_abs={max_abs} mism={mism}')

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    hist,best_val,best_step=train_source(model,val)
    torch.cuda.synchronize(); train_t=time.perf_counter()-t
    train_peak=torch.cuda.max_memory_allocated()/1024**3

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    final,stats=eval_ascc(model,test,k=8,batch=64)
    torch.cuda.synchronize(); eval_t=time.perf_counter()-t
    eval_peak=torch.cuda.max_memory_allocated()/1024**3

    base_mean=float(base.mean()); final_mean=float(final.mean())
    imp=(base_mean-final_mean)/base_mean*100
    assessment='GREEN' if imp>0.10 else ('YELLOW' if imp>=-0.05 else 'RED')
    summary={
      'round_id':'E4-H2-ICAM-PILOT',
      'host':'ICAM official icam_tsp.pt frozen',
      'host_commit':(OUT/'upstream_sha.txt').read_text().strip(),
      'graph_size':50,'test_instances':1024,'validation_instances':512,'eval_rollouts':8,
      'base_mean':base_mean,'base_eval_seconds':base_t,'base_eval_peak_gb':base_peak,
      'route_follow_wrapper_mean':float(rf.mean()),'route_follow_max_abs_diff':max_abs,
      'route_follow_mismatch_gt_1e6':mism,
      'best_val_mean':best_val,'best_step':best_step,
      'ascc_source_mean':final_mean,'improvement_vs_base_percent':imp,
      'eval_seconds':eval_t,'eval_peak_gb':eval_peak,
      'train_seconds':train_t,'train_peak_gb':train_peak,
      'source_parameters':sum(p.numel() for p in model.source.parameters()),
      **stats,'history':hist,'assessment':assessment
    }
    summary['interpretation']=(
      'Frozen ICAM supports a positive adaptive source-order gain.' if assessment=='GREEN'
      else 'Frozen ICAM remains near parity; source-order gain is not established in this bounded pilot.' if assessment=='YELLOW'
      else 'Adaptive source ordering degrades frozen ICAM under this bounded pilot.'
    )
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    torch.save({'model':model.state_dict()},OUT/'ascc_source_best.pt')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
