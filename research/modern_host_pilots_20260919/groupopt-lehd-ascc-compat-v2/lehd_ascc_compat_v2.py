from __future__ import annotations
import copy, importlib.util, json, math, time, os
from pathlib import Path
import torch
from torch import nn

ROOT=Path('/workspace/groupopt-modern-hosts/NCO_code')
MODEL_PY=ROOT/'single_objective/LEHD/TSP/TSPModel.py'
CKPT=ROOT/'single_objective/LEHD/TSP/result/20230509_153705_train/checkpoint-150.pt'
OUT=Path('/workspace/groupopt-lehd-ascc-compat-v2')
DEVICE=torch.device('cuda')

spec=importlib.util.spec_from_file_location('lehd_tsp_model', MODEL_PY)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
TSPModel=mod.TSPModel
PARAMS=dict(mode='test',embedding_dim=128,sqrt_embedding_dim=128**0.5,
            decoder_layer_num=6,qkv_dim=16,head_num=8,ff_hidden_dim=512)

def build_host():
    m=TSPModel(**PARAMS).to(DEVICE)
    ck=torch.load(CKPT,map_location=DEVICE,weights_only=False)
    m.load_state_dict(ck['model_state_dict'],strict=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m

def gather_nodes(x, idx):
    return x.gather(1, idx[:,None,None].expand(-1,1,x.size(-1))).squeeze(1)

def gather_feat(x, idx):
    return x.gather(1, idx[:,:,None].expand(-1,-1,x.size(-1)))

def candidate_indices(mask):
    b,n=mask.shape
    counts=mask.sum(1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise RuntimeError('candidate count differs across trajectories')
    c=int(counts[0].item())
    nodes=torch.arange(n,device=mask.device)[None,:].expand(b,n)
    return torch.where(mask,nodes,torch.full_like(nodes,n)).sort(1).values[:,:c]

@torch.no_grad()
def host_candidate_logits(host, encoded, first_idx, last_idx, mask):
    dec=host.decoder
    cand_idx=candidate_indices(mask)
    b,c=cand_idx.shape; d=encoded.size(-1)
    cand=encoded.gather(1,cand_idx[:,:,None].expand(b,c,d))
    first=gather_nodes(encoded,first_idx)
    last=gather_nodes(encoded,last_idx)
    out=torch.cat((dec.embedding_first_node(first).unsqueeze(1),
                   cand,
                   dec.embedding_last_node(last).unsqueeze(1)),1)
    for layer in dec.layers:
        out=layer(out)
    logits=dec.Linear_final(out).squeeze(-1)[:,1:-1]
    return cand_idx, logits

def probs_from_logits(logits):
    p=torch.softmax(logits,dim=-1).clamp_min(1e-9)
    return p/p.sum(-1,keepdim=True)

def choose(cand_idx, probs, greedy, gen):
    if greedy: pos=probs.argmax(-1)
    else: pos=torch.multinomial(probs,1,generator=gen).squeeze(1)
    chosen=cand_idx.gather(1,pos[:,None]).squeeze(1)
    lp=probs.gather(1,pos[:,None]).squeeze(1).log()
    return chosen,lp

def repeat_rollouts(coords,k):
    b,n,_=coords.shape
    x=coords[:,None,:,:].expand(b,k,n,2).reshape(b*k,n,2)
    starts=(torch.arange(k,device=coords.device)%n)[None,:].expand(b,k).reshape(-1)
    return x,starts

def tour_cost(coords, seq):
    ordered=coords.gather(1,seq[:,:,None].expand(-1,-1,2))
    return ((ordered-ordered.roll(-1,1))**2).sum(-1).sqrt().sum(-1)

@torch.no_grad()
def encode(host,coords):
    return host.encoder(coords)

def sequential_rollout(host, coords, starts, greedy, gen):
    enc=encode(host,coords)
    b,n,_=coords.shape
    visited=torch.zeros((b,n),dtype=torch.bool,device=DEVICE)
    visited.scatter_(1,starts[:,None],True)
    seq=[starts]; current=starts; logp=torch.zeros(b,device=DEVICE)
    for _ in range(1,n):
        cand,logits=host_candidate_logits(host,enc,starts,current,~visited)
        probs=probs_from_logits(logits)
        nxt,lp=choose(cand,probs,greedy,gen)
        logp=logp+lp
        visited.scatter_(1,nxt[:,None],True)
        seq.append(nxt); current=nxt
    return tour_cost(coords,torch.stack(seq,1)),logp

class TailPolicy(nn.Module):
    def __init__(self,d=128,h=64,continuation_init=5.0):
        super().__init__()
        self.tail=nn.Linear(d,h,bias=False)
        self.start=nn.Linear(d,h,bias=False)
        self.last=nn.Linear(d,h,bias=False)
        self.glob=nn.Linear(d,h,bias=False)
        self.size=nn.Linear(1,h,bias=True)
        self.out=nn.Linear(h,1,bias=False)
        self.continuation_logit=nn.Parameter(torch.tensor(float(continuation_init)))
        nn.init.normal_(self.out.weight,std=0.001)
    def forward(self,enc,start_enc,last_enc,comp_size,last_idx,tail_mask):
        h=(self.tail(enc)+self.start(start_enc)+self.last(last_enc)[:,None,:]
           +self.glob(enc.mean(1))[:,None,:]+self.size(comp_size[:,:,None]))
        score=self.out(torch.tanh(h)).squeeze(-1)
        nodes=torch.arange(enc.size(1),device=enc.device)[None,:]
        score=score+self.continuation_logit*(nodes==last_idx[:,None]).to(score.dtype)
        return score.masked_fill(~tail_mask,-torch.inf)

class FragmentEndpointAdapter(nn.Module):
    """Small residual adapter using fragment endpoints rather than pretending a forest is one path."""
    def __init__(self,d=128,h=32):
        super().__init__()
        self.tail_proj=nn.Linear(2*d,h,bias=False)
        self.cand_proj=nn.Linear(2*d,h,bias=False)
        self.scalar=nn.Sequential(nn.Linear(5,32),nn.Tanh(),nn.Linear(32,1,bias=False))
        self.gate_logit=nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.tail_proj.weight,std=0.01)
        nn.init.normal_(self.cand_proj.weight,std=0.01)
        nn.init.zeros_(self.scalar[-1].weight)

    def forward(self, enc, coords, first_idx, tail_idx, cand_idx, cand_end_idx,
                tail_size, cand_size, progress, base_logits):
        first_e=gather_nodes(enc,first_idx)
        tail_e=gather_nodes(enc,tail_idx)
        tq=self.tail_proj(torch.cat([first_e,tail_e],-1))[:,None,:]

        cand_start_e=gather_feat(enc,cand_idx)
        cand_end_e=gather_feat(enc,cand_end_idx)
        ck=self.cand_proj(torch.cat([cand_start_e,cand_end_e],-1))
        bilinear=(tq*ck).sum(-1)/math.sqrt(ck.size(-1))

        tail_xy=coords.gather(1,tail_idx[:,None,None].expand(-1,1,2)).squeeze(1)
        cand_xy=coords.gather(1,cand_idx[:,:,None].expand(-1,-1,2))
        dist=((cand_xy-tail_xy[:,None,:])**2).sum(-1).sqrt()

        c=cand_idx.size(1)
        tail_sz=tail_size[:,None].expand(-1,c)
        prog=torch.full_like(dist,float(progress))
        scal=torch.stack([tail_sz,cand_size,dist,prog,base_logits],-1)
        s=self.scalar(scal).squeeze(-1)
        gate=torch.sigmoid(self.gate_logit)
        return gate*(bilinear+s)

class ASCCModel(nn.Module):
    def __init__(self,host,endpoint_adapter=None,continuation_init=5.0):
        super().__init__()
        self.host=host
        self.tail_policy=TailPolicy(continuation_init=continuation_init)
        self.endpoint_adapter=endpoint_adapter

    def rollout(self,coords,starts,greedy,gen,route_follow=False,collect_stats=False):
        host=self.host; enc=encode(host,coords)
        b,n,_=coords.shape
        node_idx=torch.arange(n,device=DEVICE)[None,:].expand(b,n)
        succ=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        pred=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        comp=node_idx.clone(); last_head=starts
        logp=torch.zeros(b,device=DEVICE)
        dev_num=torch.zeros((),device=DEVICE); dev_den=torch.zeros((),device=DEVICE)
        residual_abs=torch.zeros((),device=DEVICE); residual_count=torch.zeros((),device=DEVICE)

        for step in range(n):
            same=comp[:,:,None].eq(comp[:,None,:])
            is_start=pred.lt(0)
            comp_start=(same & is_start[:,None,:]).to(torch.int8).argmax(2).long()
            is_end=succ.lt(0)
            comp_end=(same & is_end[:,None,:]).to(torch.int8).argmax(2).long()
            comp_size=same.sum(2).to(enc.dtype)/float(n)

            if step==0:
                tail=starts
            elif route_follow:
                tail=last_head
            else:
                start_enc=enc.gather(1,comp_start[:,:,None].expand(-1,-1,enc.size(-1)))
                last_enc=gather_nodes(enc,last_head)
                logits=self.tail_policy(enc,start_enc,last_enc,comp_size,last_head,succ.lt(0))
                probs=torch.softmax(logits,dim=-1)
                if greedy: tail=probs.argmax(-1)
                else: tail=torch.multinomial(probs,1,generator=gen).squeeze(1)
                logp=logp+probs.gather(1,tail[:,None]).squeeze(1).clamp_min(1e-9).log()
                dev_num=dev_num+(tail!=last_head).float().sum()
                dev_den=dev_den+torch.tensor(float(b),device=DEVICE)

            ct=comp.gather(1,tail[:,None]).squeeze(1)
            first=comp_start.gather(1,tail[:,None]).squeeze(1)
            head_mask=(pred.lt(0) & comp.ne(ct[:,None])) if step<n-1 else pred.lt(0)
            cand,base_logits=host_candidate_logits(host,enc,first,tail,head_mask)

            logits=base_logits
            if self.endpoint_adapter is not None:
                cand_end_all=comp_end.gather(1,cand)
                cand_sizes=comp_size.gather(1,cand)
                tail_sz=comp_size.gather(1,tail[:,None]).squeeze(1)
                residual=self.endpoint_adapter(
                    enc,coords,first,tail,cand,cand_end_all,tail_sz,cand_sizes,
                    progress=step/float(max(1,n-1)),base_logits=base_logits)
                logits=base_logits+residual
                residual_abs=residual_abs+residual.abs().sum()
                residual_count=residual_count+torch.tensor(float(residual.numel()),device=DEVICE)

            hp=probs_from_logits(logits)
            head,hlp=choose(cand,hp,greedy,gen)
            logp=logp+hlp
            succ.scatter_(1,tail[:,None],head[:,None])
            pred.scatter_(1,head[:,None],tail[:,None])
            if step<n-1:
                ch=comp.gather(1,head[:,None]).squeeze(1)
                comp=torch.where(comp.eq(ch[:,None]),ct[:,None],comp)
            last_head=head

        if (succ<0).any() or (pred<0).any():
            raise RuntimeError('incomplete ASCC cycle')
        next_xy=coords.gather(1,succ[:,:,None].expand(-1,-1,2))
        cost=((coords-next_xy)**2).sum(-1).sqrt().sum(1)
        stats=None
        if collect_stats:
            stats={
                'source_deviation_fraction': float((dev_num/dev_den.clamp_min(1)).detach()),
                'mean_abs_endpoint_residual': float((residual_abs/residual_count.clamp_min(1)).detach())
            }
        return cost,logp,stats

def make_fixed(seed,count,n=50):
    g=torch.Generator(device='cpu').manual_seed(seed)
    return torch.rand(count,n,2,generator=g)

@torch.inference_mode()
def evaluate_seq(host,data,k=8,batch=64):
    host.eval(); vals=[]
    for st in range(0,len(data),batch):
        c=data[st:st+batch].to(DEVICE); cr,s=repeat_rollouts(c,k)
        cost,_=sequential_rollout(host,cr,s,True,None)
        vals.append(cost.view(c.size(0),k).min(1).values.cpu())
    return torch.cat(vals)

@torch.inference_mode()
def evaluate_ascc(model,data,k=8,batch=64,route_follow=False):
    model.eval(); vals=[]; dev=[]; res=[]
    for st in range(0,len(data),batch):
        c=data[st:st+batch].to(DEVICE); cr,s=repeat_rollouts(c,k)
        cost,_,stats=model.rollout(cr,s,True,None,route_follow=route_follow,collect_stats=True)
        vals.append(cost.view(c.size(0),k).min(1).values.cpu())
        dev.append(stats['source_deviation_fraction']); res.append(stats['mean_abs_endpoint_residual'])
    return torch.cat(vals), {'source_deviation_fraction':sum(dev)/len(dev),
                             'mean_abs_endpoint_residual':sum(res)/len(res)}

def timed_eval(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t=time.perf_counter()
    out=fn()
    torch.cuda.synchronize(); dt=time.perf_counter()-t
    peak=torch.cuda.max_memory_allocated()/1024**3
    return out,dt,peak

def clone_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}

def train_policy(model,steps,batch,k,seed,val,lr_tail=5e-5,lr_adapter=1e-4,label='MODEL'):
    params=[{'params':model.tail_policy.parameters(),'lr':lr_tail}]
    if model.endpoint_adapter is not None:
        params.append({'params':model.endpoint_adapter.parameters(),'lr':lr_adapter})
    opt=torch.optim.Adam(params)
    dg=torch.Generator(device='cpu').manual_seed(seed)
    ag=torch.Generator(device=DEVICE).manual_seed(seed+1000)
    trainables=[p for p in model.parameters() if p.requires_grad]
    hist=[]
    init_val,_=evaluate_ascc(model,val,k=4,batch=64)
    best_val=float(init_val.mean()); best_step=0; best_state=clone_state(model)
    hist.append({'step':0,'val':best_val,'continuation_logit':float(model.tail_policy.continuation_logit.detach())})
    print(label,json.dumps(hist[-1]),flush=True)

    checkpoints={100,200,400,800,steps}
    for step in range(1,steps+1):
        coords=torch.rand(batch,50,2,generator=dg).to(DEVICE)
        cr,starts=repeat_rollouts(coords,k)
        cost,lp,_=model.rollout(cr,starts,False,ag)
        reward=-cost.view(batch,k)
        adv=reward-reward.mean(1,keepdim=True)
        loss=-(adv.detach()*lp.view(batch,k)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables,1.0); opt.step()
        if step in checkpoints:
            val_cost,stats=evaluate_ascc(model,val,k=4,batch=64)
            vm=float(val_cost.mean())
            row={'step':step,'train_cost':float(cost.mean()),'loss':float(loss),
                 'val':vm,'continuation_logit':float(model.tail_policy.continuation_logit.detach()),
                 **stats}
            if model.endpoint_adapter is not None:
                row['endpoint_gate']=float(torch.sigmoid(model.endpoint_adapter.gate_logit).detach())
            hist.append(row); print(label,json.dumps(row),flush=True)
            if vm < best_val:
                best_val=vm; best_step=step; best_state=clone_state(model)
    model.load_state_dict(best_state,strict=True)
    return hist,best_val,best_step

def main():
    torch.manual_seed(20260918); torch.cuda.manual_seed_all(20260918)
    test=make_fixed(20260918,1024)
    val=make_fixed(20260919,512)
    host=build_host()

    (base_cost,base_t,base_peak)=(*timed_eval(lambda:evaluate_seq(host,test)),)
    base_mean=float(base_cost.mean())

    sanity=ASCCModel(copy.deepcopy(host),endpoint_adapter=None,continuation_init=5.0).to(DEVICE)
    (rf_pack,rf_t,rf_peak)=timed_eval(lambda:evaluate_ascc(sanity,test,k=8,batch=64,route_follow=True))
    rf_cost,rf_stats=rf_pack
    max_abs=float((rf_cost-base_cost).abs().max())
    mismatch=int(((rf_cost-base_cost).abs()>1e-6).sum())
    if max_abs>1e-5:
        raise RuntimeError(f'route-follow sanity failed: max_abs={max_abs} mismatch={mismatch}')

    tail=ASCCModel(copy.deepcopy(host),endpoint_adapter=None,continuation_init=5.0).to(DEVICE)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    tail_hist,tail_best_val,tail_best_step=train_policy(
        tail,800,24,4,41001,val,lr_tail=5e-5,label='TAIL_ONLY')
    torch.cuda.synchronize(); tail_train_t=time.perf_counter()-t
    tail_train_peak=torch.cuda.max_memory_allocated()/1024**3
    (tail_pack,tail_eval_t,tail_eval_peak)=timed_eval(lambda:evaluate_ascc(tail,test,k=8,batch=64))
    tail_cost,tail_stats=tail_pack
    tail_mean=float(tail_cost.mean())

    forest=ASCCModel(copy.deepcopy(host),endpoint_adapter=FragmentEndpointAdapter().to(DEVICE),
                     continuation_init=5.0).to(DEVICE)
    forest.tail_policy.load_state_dict(tail.tail_policy.state_dict())
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    forest_hist,forest_best_val,forest_best_step=train_policy(
        forest,1200,24,4,51001,val,lr_tail=3e-5,lr_adapter=1e-4,label='FOREST_AWARE')
    torch.cuda.synchronize(); forest_train_t=time.perf_counter()-t
    forest_train_peak=torch.cuda.max_memory_allocated()/1024**3
    (forest_pack,forest_eval_t,forest_eval_peak)=timed_eval(lambda:evaluate_ascc(forest,test,k=8,batch=64))
    forest_cost,forest_stats=forest_pack
    forest_mean=float(forest_cost.mean())

    tail_imp=(base_mean-tail_mean)/base_mean*100
    forest_imp=(base_mean-forest_mean)/base_mean*100
    forest_vs_tail=(tail_mean-forest_mean)/tail_mean*100
    assessment='GREEN' if forest_imp>0.10 else ('YELLOW' if forest_imp>=-0.05 else 'RED')

    summary={
      'round_id':'E4-H1B-LEHD-COMPAT',
      'host':'LEHD NeurIPS 2023 official checkpoint-150',
      'graph_size':50,'test_instances':1024,'validation_instances':512,'eval_rollouts':8,
      'route_follow_sanity':{
        'base_mean':base_mean,'wrapper_mean':float(rf_cost.mean()),
        'max_abs_instance_cost_diff':max_abs,'mismatch_count_gt_1e-6':mismatch,
        'eval_seconds':rf_t,'eval_peak_gb':rf_peak
      },
      'tail_only_frozen_host':{
        'mean_cost':tail_mean,'improvement_vs_base_percent':tail_imp,
        'best_val_mean':tail_best_val,'best_step':tail_best_step,
        'eval_seconds':tail_eval_t,'eval_peak_gb':tail_eval_peak,
        'train_seconds':tail_train_t,'train_peak_gb':tail_train_peak,
        **tail_stats,'history':tail_hist
      },
      'forest_aware_frozen_host':{
        'mean_cost':forest_mean,'improvement_vs_base_percent':forest_imp,
        'improvement_vs_tail_only_percent':forest_vs_tail,
        'best_val_mean':forest_best_val,'best_step':forest_best_step,
        'eval_seconds':forest_eval_t,'eval_peak_gb':forest_eval_peak,
        'train_seconds':forest_train_t,'train_peak_gb':forest_train_peak,
        'adapter_parameters':sum(p.numel() for p in forest.endpoint_adapter.parameters()),
        'tail_parameters':sum(p.numel() for p in forest.tail_policy.parameters()),
        **forest_stats,'history':forest_hist
      },
      'base_eval_seconds':base_t,'base_eval_peak_gb':base_peak,
      'assessment':assessment,
      'interpretation':(
        'Forest-aware adapter gives >0.10% gain over frozen official LEHD.' if assessment=='GREEN'
        else 'Compatibility repair is near parity; more mechanism diagnosis is needed.' if assessment=='YELLOW'
        else 'Even with route-follow trust initialization and a forest-aware residual endpoint adapter, the frozen LEHD host is worse than base.'
      )
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    torch.save({'tail_only':tail.state_dict()},OUT/'tail_only_best.pt')
    torch.save({'forest_aware':forest.state_dict()},OUT/'forest_aware_best.pt')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
