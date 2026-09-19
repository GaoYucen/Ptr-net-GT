from __future__ import annotations
import copy, importlib.util, json, time
from pathlib import Path
import torch
from torch import nn

ROOT=Path('/workspace/groupopt-modern-hosts/NCO_code')
MODEL_PY=ROOT/'single_objective/LEHD/TSP/TSPModel.py'
CKPT=ROOT/'single_objective/LEHD/TSP/result/20230509_153705_train/checkpoint-150.pt'
OUT=Path('/workspace/groupopt-lehd-ascc-e4h1')
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
    for p in m.encoder.parameters(): p.requires_grad_(False)
    return m

def gather_nodes(x, idx):
    return x.gather(1, idx[:,None,None].expand(-1,1,x.size(-1))).squeeze(1)

def candidate_indices(mask):
    b,n=mask.shape
    counts=mask.sum(1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise RuntimeError('candidate count differs across trajectories')
    c=int(counts[0].item())
    nodes=torch.arange(n,device=mask.device)[None,:].expand(b,n)
    return torch.where(mask,nodes,torch.full_like(nodes,n)).sort(1).values[:,:c]

def host_candidate_probs(host, encoded, first_idx, last_idx, mask):
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
    probs=torch.softmax(logits,dim=-1)
    probs=probs.clamp_min(1e-9)
    probs=probs/probs.sum(-1,keepdim=True)
    return cand_idx, probs

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

def encode(host,coords):
    with torch.no_grad():
        return host.encoder(coords)

def sequential_rollout(host, coords, starts, greedy, gen):
    enc=encode(host,coords)
    b,n,_=coords.shape
    visited=torch.zeros((b,n),dtype=torch.bool,device=DEVICE)
    visited.scatter_(1,starts[:,None],True)
    seq=[starts]; current=starts; logp=torch.zeros(b,device=DEVICE)
    for _ in range(1,n):
        cand,probs=host_candidate_probs(host,enc,starts,current,~visited)
        nxt,lp=choose(cand,probs,greedy,gen)
        logp=logp+lp
        visited.scatter_(1,nxt[:,None],True)
        seq.append(nxt); current=nxt
    return tour_cost(coords,torch.stack(seq,1)),logp

class TailPolicy(nn.Module):
    def __init__(self,d=128,h=64):
        super().__init__()
        self.tail=nn.Linear(d,h,bias=False); self.start=nn.Linear(d,h,bias=False)
        self.last=nn.Linear(d,h,bias=False); self.glob=nn.Linear(d,h,bias=False)
        self.size=nn.Linear(1,h,bias=True); self.out=nn.Linear(h,1,bias=False)
        self.continuation_logit=nn.Parameter(torch.tensor(3.0))
        nn.init.normal_(self.out.weight,std=0.01)
    def forward(self,enc,start_enc,last_enc,comp_size,last_idx,tail_mask):
        h=(self.tail(enc)+self.start(start_enc)+self.last(last_enc)[:,None,:]
           +self.glob(enc.mean(1))[:,None,:]+self.size(comp_size[:,:,None]))
        score=self.out(torch.tanh(h)).squeeze(-1)
        nodes=torch.arange(enc.size(1),device=enc.device)[None,:]
        score=score+self.continuation_logit*(nodes==last_idx[:,None]).to(score.dtype)
        return score.masked_fill(~tail_mask,-torch.inf)

class ASCCModel(nn.Module):
    def __init__(self,host):
        super().__init__(); self.host=host; self.tail_policy=TailPolicy()
    def rollout(self,coords,starts,greedy,gen):
        host=self.host; enc=encode(host,coords)
        b,n,_=coords.shape
        node_idx=torch.arange(n,device=DEVICE)[None,:].expand(b,n)
        succ=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        pred=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        comp=node_idx.clone(); last_head=starts; logp=torch.zeros(b,device=DEVICE)
        for step in range(n):
            same=comp[:,:,None].eq(comp[:,None,:])
            is_start=pred.lt(0)
            comp_start=(same & is_start[:,None,:]).to(torch.int8).argmax(2).long()
            if step==0:
                tail=starts
            else:
                start_enc=enc.gather(1,comp_start[:,:,None].expand(-1,-1,enc.size(-1)))
                last_enc=gather_nodes(enc,last_head)
                comp_size=same.sum(2).to(enc.dtype)/float(n)
                logits=self.tail_policy(enc,start_enc,last_enc,comp_size,last_head,succ.lt(0))
                probs=torch.softmax(logits,dim=-1)
                if greedy: tail=probs.argmax(-1)
                else: tail=torch.multinomial(probs,1,generator=gen).squeeze(1)
                logp=logp+probs.gather(1,tail[:,None]).squeeze(1).clamp_min(1e-9).log()
            ct=comp.gather(1,tail[:,None]).squeeze(1)
            first=comp_start.gather(1,tail[:,None]).squeeze(1)
            head_mask=(pred.lt(0) & comp.ne(ct[:,None])) if step<n-1 else pred.lt(0)
            cand,hp=host_candidate_probs(host,enc,first,tail,head_mask)
            head,hlp=choose(cand,hp,greedy,gen); logp=logp+hlp
            succ.scatter_(1,tail[:,None],head[:,None]); pred.scatter_(1,head[:,None],tail[:,None])
            if step<n-1:
                ch=comp.gather(1,head[:,None]).squeeze(1)
                comp=torch.where(comp.eq(ch[:,None]),ct[:,None],comp)
            last_head=head
        if (succ<0).any() or (pred<0).any(): raise RuntimeError('incomplete ASCC cycle')
        dist=torch.cdist(coords,coords)
        return dist.gather(2,succ[:,:,None]).squeeze(-1).sum(1),logp

def make_fixed(seed,count,n):
    g=torch.Generator(device='cpu').manual_seed(seed)
    return torch.rand(count,n,2,generator=g)

@torch.inference_mode()
def evaluate_seq(host,test,k=8,batch=64):
    host.eval(); vals=[]
    for st in range(0,len(test),batch):
        c=test[st:st+batch].to(DEVICE); cr,s=repeat_rollouts(c,k)
        cost,_=sequential_rollout(host,cr,s,True,None)
        vals.append(cost.view(c.size(0),k).min(1).values.cpu())
    return torch.cat(vals)

@torch.inference_mode()
def evaluate_ascc(model,test,k=8,batch=64):
    model.eval(); vals=[]
    for st in range(0,len(test),batch):
        c=test[st:st+batch].to(DEVICE); cr,s=repeat_rollouts(c,k)
        cost,_=model.rollout(cr,s,True,None)
        vals.append(cost.view(c.size(0),k).min(1).values.cpu())
    return torch.cat(vals)

def train_seq(host,steps,batch,k,seed,test):
    host.train(); opt=torch.optim.Adam(host.decoder.parameters(),lr=1e-5)
    dg=torch.Generator(device='cpu').manual_seed(seed)
    ag=torch.Generator(device=DEVICE).manual_seed(seed+100)
    hist=[]
    for step in range(1,steps+1):
        coords=torch.rand(batch,50,2,generator=dg).to(DEVICE); cr,starts=repeat_rollouts(coords,k)
        cost,lp=sequential_rollout(host,cr,starts,False,ag)
        reward=-cost.view(batch,k); adv=reward-reward.mean(1,keepdim=True)
        loss=-(adv.detach()*lp.view(batch,k)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(host.decoder.parameters(),1.0); opt.step()
        if step in {1,100,200,400,steps}:
            val=evaluate_seq(host,test[:256],k=4,batch=64).mean().item()
            row={'step':step,'train_cost':float(cost.mean()),'loss':float(loss),'val':val}
            print('SEQ',json.dumps(row),flush=True); hist.append(row)
    return hist

def train_ascc(model,steps,batch,k,seed,test):
    model.train()
    opt=torch.optim.Adam([{'params':model.host.decoder.parameters(),'lr':1e-5},
                          {'params':model.tail_policy.parameters(),'lr':1e-4}])
    dg=torch.Generator(device='cpu').manual_seed(seed)
    ag=torch.Generator(device=DEVICE).manual_seed(seed+200)
    hist=[]; params=[p for p in model.parameters() if p.requires_grad]
    for step in range(1,steps+1):
        coords=torch.rand(batch,50,2,generator=dg).to(DEVICE); cr,starts=repeat_rollouts(coords,k)
        cost,lp=model.rollout(cr,starts,False,ag)
        reward=-cost.view(batch,k); adv=reward-reward.mean(1,keepdim=True)
        loss=-(adv.detach()*lp.view(batch,k)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if step in {1,100,200,400,steps}:
            val=evaluate_ascc(model,test[:256],k=4,batch=64).mean().item()
            row={'step':step,'train_cost':float(cost.mean()),'loss':float(loss),'val':val,
                 'continuation_logit':float(model.tail_policy.continuation_logit.detach())}
            print('ASCC',json.dumps(row),flush=True); hist.append(row)
    return hist

def main():
    torch.manual_seed(20260918); torch.cuda.manual_seed_all(20260918)
    test=make_fixed(20260918,1024,50)
    base=build_host(); seq=copy.deepcopy(base); ascc=ASCCModel(copy.deepcopy(base)).to(DEVICE)

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    base_cost=evaluate_seq(base,test); torch.cuda.synchronize(); base_time=time.perf_counter()-t
    base_peak=torch.cuda.max_memory_allocated()/1024**3

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    seq_hist=train_seq(seq,800,24,4,31001,test); torch.cuda.synchronize(); seq_train_time=time.perf_counter()-t
    seq_train_peak=torch.cuda.max_memory_allocated()/1024**3

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    ascc_hist=train_ascc(ascc,800,24,4,31001,test); torch.cuda.synchronize(); ascc_train_time=time.perf_counter()-t
    ascc_train_peak=torch.cuda.max_memory_allocated()/1024**3

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    seq_cost=evaluate_seq(seq,test); torch.cuda.synchronize(); seq_eval_time=time.perf_counter()-t
    seq_eval_peak=torch.cuda.max_memory_allocated()/1024**3

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    ascc_cost=evaluate_ascc(ascc,test); torch.cuda.synchronize(); ascc_eval_time=time.perf_counter()-t
    ascc_eval_peak=torch.cuda.max_memory_allocated()/1024**3

    bm=float(base_cost.mean()); sm=float(seq_cost.mean()); am=float(ascc_cost.mean())
    summary={
      'host':'LEHD NeurIPS 2023 official checkpoint-150','graph_size':50,
      'test_instances':1024,'pomo_rollouts_eval':8,'finetune_steps':800,
      'train_batch_instances':24,'train_rollouts':4,
      'base_mean':bm,'sequential_finetune_mean':sm,'ascc_finetune_mean':am,
      'ascc_vs_base_improvement_percent':(bm-am)/bm*100,
      'ascc_vs_seq_finetune_improvement_percent':(sm-am)/sm*100,
      'seq_finetune_vs_base_improvement_percent':(bm-sm)/bm*100,
      'base_eval_seconds':base_time,'seq_eval_seconds':seq_eval_time,'ascc_eval_seconds':ascc_eval_time,
      'seq_train_seconds':seq_train_time,'ascc_train_seconds':ascc_train_time,
      'base_eval_peak_gb':base_peak,'seq_eval_peak_gb':seq_eval_peak,'ascc_eval_peak_gb':ascc_eval_peak,
      'seq_train_peak_gb':seq_train_peak,'ascc_train_peak_gb':ascc_train_peak,
      'seq_history':seq_hist,'ascc_history':ascc_hist,
      'tail_parameters':sum(p.numel() for p in ascc.tail_policy.parameters()),
      'host_trainable_decoder_parameters':sum(p.numel() for p in ascc.host.decoder.parameters()),
    }
    imp=summary['ascc_vs_seq_finetune_improvement_percent']
    summary['assessment']='GREEN' if imp>0.10 else 'YELLOW' if imp>=-0.10 else 'RED'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    torch.save({'model':seq.state_dict()},OUT/'seq_finetuned.pt')
    torch.save({'model':ascc.state_dict()},OUT/'ascc_finetuned.pt')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
