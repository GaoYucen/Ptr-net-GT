from __future__ import annotations
import importlib.util, json, math, time
from pathlib import Path
import torch
from torch import nn

BASE=Path('/workspace/groupopt-lehd-ascc-coupled-v3/compat_base.py')
OUT=Path('/workspace/groupopt-lehd-ascc-coupled-v3')
DEVICE=torch.device('cuda')
spec=importlib.util.spec_from_file_location('compat_base',BASE)
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)

class PairOpportunity(nn.Module):
    def __init__(self,d=128,h=32):
        super().__init__()
        self.tail_proj=nn.Linear(2*d,h,bias=False)
        self.head_proj=nn.Linear(2*d,h,bias=False)
        self.scalar=nn.Sequential(nn.Linear(4,32),nn.Tanh(),nn.Linear(32,1,bias=False))
        nn.init.normal_(self.tail_proj.weight,std=0.01)
        nn.init.normal_(self.head_proj.weight,std=0.01)
        nn.init.zeros_(self.scalar[-1].weight)

    def all_pairs(self,enc,coords,comp_start,comp_end,comp_size,tail_idx,head_idx,progress):
        d=enc.size(-1); m=tail_idx.size(1)
        tail_start=comp_start.gather(1,tail_idx)
        head_end=comp_end.gather(1,head_idx)
        ts=cb.gather_feat(enc,tail_start); te=cb.gather_feat(enc,tail_idx)
        hs=cb.gather_feat(enc,head_idx); he=cb.gather_feat(enc,head_end)
        q=self.tail_proj(torch.cat([ts,te],-1))
        k=self.head_proj(torch.cat([hs,he],-1))
        bilinear=torch.matmul(q,k.transpose(1,2))/math.sqrt(q.size(-1))

        tail_xy=coords.gather(1,tail_idx[:,:,None].expand(-1,-1,2))
        head_xy=coords.gather(1,head_idx[:,:,None].expand(-1,-1,2))
        dist=((tail_xy[:,:,None,:]-head_xy[:,None,:,:])**2).sum(-1).sqrt()

        tail_sz=comp_size.gather(1,tail_idx)[:,:,None].expand(-1,m,m)
        head_sz=comp_size.gather(1,head_idx)[:,None,:].expand(-1,m,m)
        prog=torch.full_like(dist,float(progress))
        scal=torch.stack([tail_sz,head_sz,dist,prog],-1)
        return bilinear+self.scalar(scal).squeeze(-1)

    def selected_pairs(self,enc,coords,comp_start,comp_end,comp_size,tail_idx,head_idx,progress):
        d=enc.size(-1); c=head_idx.size(1)
        tail_start=comp_start.gather(1,tail_idx[:,None]).squeeze(1)
        head_end=comp_end.gather(1,head_idx)
        ts=cb.gather_nodes(enc,tail_start)
        te=cb.gather_nodes(enc,tail_idx)
        hs=cb.gather_feat(enc,head_idx)
        he=cb.gather_feat(enc,head_end)
        q=self.tail_proj(torch.cat([ts,te],-1))[:,None,:]
        k=self.head_proj(torch.cat([hs,he],-1))
        bilinear=(q*k).sum(-1)/math.sqrt(k.size(-1))

        tail_xy=coords.gather(1,tail_idx[:,None,None].expand(-1,1,2)).squeeze(1)
        head_xy=coords.gather(1,head_idx[:,:,None].expand(-1,-1,2))
        dist=((tail_xy[:,None,:]-head_xy)**2).sum(-1).sqrt()
        tail_sz=comp_size.gather(1,tail_idx[:,None]).expand(-1,c)
        head_sz=comp_size.gather(1,head_idx)
        prog=torch.full_like(dist,float(progress))
        scal=torch.stack([tail_sz,head_sz,dist,prog],-1)
        return bilinear+self.scalar(scal).squeeze(-1)

class CoupledASCC(nn.Module):
    def __init__(self,host):
        super().__init__()
        self.host=host
        self.tail_policy=cb.TailPolicy(continuation_init=4.0)
        self.endpoint_adapter=cb.FragmentEndpointAdapter()
        self.pair_policy=PairOpportunity()
        self.source_pair_gate_logit=nn.Parameter(torch.tensor(-1.0))
        self.endpoint_pair_gate_logit=nn.Parameter(torch.tensor(-2.0))

    def rollout(self,coords,starts,greedy,gen,collect_stats=False):
        enc=cb.encode(self.host,coords)
        b,n,_=coords.shape
        node_idx=torch.arange(n,device=DEVICE)[None,:].expand(b,n)
        succ=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        pred=torch.full((b,n),-1,dtype=torch.long,device=DEVICE)
        comp=node_idx.clone(); last_head=starts
        logp=torch.zeros(b,device=DEVICE)
        dev_num=torch.zeros((),device=DEVICE); dev_den=torch.zeros((),device=DEVICE)
        residual_abs=torch.zeros((),device=DEVICE); residual_count=torch.zeros((),device=DEVICE)
        pair_abs=torch.zeros((),device=DEVICE); pair_count=torch.zeros((),device=DEVICE)

        for step in range(n):
            same=comp[:,:,None].eq(comp[:,None,:])
            is_start=pred.lt(0); is_end=succ.lt(0)
            comp_start=(same & is_start[:,None,:]).to(torch.int8).argmax(2).long()
            comp_end=(same & is_end[:,None,:]).to(torch.int8).argmax(2).long()
            comp_size=same.sum(2).to(enc.dtype)/float(n)

            if step==0:
                tail=starts
            elif step==n-1:
                # One component remains; close the cycle from its unique open end.
                tail_idx=cb.candidate_indices(is_end)
                if tail_idx.size(1)!=1:
                    raise RuntimeError(f'expected one open tail at closure, got {tail_idx.size(1)}')
                tail=tail_idx[:,0]
            else:
                tail_idx=cb.candidate_indices(is_end)
                head_idx=cb.candidate_indices(is_start)
                pair=self.pair_policy.all_pairs(
                    enc,coords,comp_start,comp_end,comp_size,tail_idx,head_idx,
                    step/float(max(1,n-1)))
                tail_comp=comp.gather(1,tail_idx)
                head_comp=comp.gather(1,head_idx)
                valid=tail_comp[:,:,None].ne(head_comp[:,None,:])
                pair_masked=pair.masked_fill(~valid,-torch.inf)
                count=valid.sum(-1).clamp_min(1).to(pair.dtype)
                opportunity=torch.logsumexp(pair_masked,dim=-1)-count.log()

                start_enc=enc.gather(1,comp_start[:,:,None].expand(-1,-1,enc.size(-1)))
                last_enc=cb.gather_nodes(enc,last_head)
                base_tail=self.tail_policy(enc,start_enc,last_enc,comp_size,last_head,is_end)
                base_tail=base_tail.gather(1,tail_idx)
                source_gate=torch.sigmoid(self.source_pair_gate_logit)
                source_logits=base_tail+source_gate*opportunity
                source_probs=torch.softmax(source_logits,dim=-1)
                if greedy:
                    pos=source_probs.argmax(-1)
                else:
                    pos=torch.multinomial(source_probs,1,generator=gen).squeeze(1)
                tail=tail_idx.gather(1,pos[:,None]).squeeze(1)
                logp=logp+source_probs.gather(1,pos[:,None]).squeeze(1).clamp_min(1e-9).log()
                dev_num=dev_num+(tail!=last_head).float().sum()
                dev_den=dev_den+torch.tensor(float(b),device=DEVICE)
                pair_abs=pair_abs+pair[valid].abs().sum()
                pair_count=pair_count+valid.sum().to(pair.dtype)

            ct=comp.gather(1,tail[:,None]).squeeze(1)
            first=comp_start.gather(1,tail[:,None]).squeeze(1)
            head_mask=(is_start & comp.ne(ct[:,None])) if step<n-1 else is_start
            cand,base_logits=cb.host_candidate_logits(self.host,enc,first,tail,head_mask)

            cand_end=comp_end.gather(1,cand)
            cand_sizes=comp_size.gather(1,cand)
            tail_sz=comp_size.gather(1,tail[:,None]).squeeze(1)
            residual=self.endpoint_adapter(
                enc,coords,first,tail,cand,cand_end,tail_sz,cand_sizes,
                step/float(max(1,n-1)),base_logits)
            pair_selected=self.pair_policy.selected_pairs(
                enc,coords,comp_start,comp_end,comp_size,tail,cand,
                step/float(max(1,n-1)))
            logits=(base_logits+residual+
                    torch.sigmoid(self.endpoint_pair_gate_logit)*pair_selected)
            hp=cb.probs_from_logits(logits)
            head,hlp=cb.choose(cand,hp,greedy,gen)
            logp=logp+hlp
            succ.scatter_(1,tail[:,None],head[:,None])
            pred.scatter_(1,head[:,None],tail[:,None])
            if step<n-1:
                ch=comp.gather(1,head[:,None]).squeeze(1)
                comp=torch.where(comp.eq(ch[:,None]),ct[:,None],comp)
            last_head=head

            residual_abs=residual_abs+residual.abs().sum()
            residual_count=residual_count+torch.tensor(float(residual.numel()),device=DEVICE)

        if (succ<0).any() or (pred<0).any():
            raise RuntimeError('incomplete coupled ASCC cycle')
        next_xy=coords.gather(1,succ[:,:,None].expand(-1,-1,2))
        cost=((coords-next_xy)**2).sum(-1).sqrt().sum(1)
        stats=None
        if collect_stats:
            stats={
              'source_deviation_fraction':float((dev_num/dev_den.clamp_min(1)).detach()),
              'mean_abs_endpoint_residual':float((residual_abs/residual_count.clamp_min(1)).detach()),
              'mean_abs_pair_score':float((pair_abs/pair_count.clamp_min(1)).detach()),
              'source_pair_gate':float(torch.sigmoid(self.source_pair_gate_logit).detach()),
              'endpoint_pair_gate':float(torch.sigmoid(self.endpoint_pair_gate_logit).detach()),
              'endpoint_residual_gate':float(torch.sigmoid(self.endpoint_adapter.gate_logit).detach()),
              'continuation_logit':float(self.tail_policy.continuation_logit.detach()),
            }
        return cost,logp,stats

@torch.inference_mode()
def evaluate(model,data,k=8,batch=64):
    model.eval(); vals=[]; stats=[]
    for st in range(0,len(data),batch):
        c=data[st:st+batch].to(DEVICE); cr,s=cb.repeat_rollouts(c,k)
        cost,_,ss=model.rollout(cr,s,True,None,collect_stats=True)
        vals.append(cost.view(c.size(0),k).min(1).values.cpu()); stats.append(ss)
    avg={k:sum(x[k] for x in stats)/len(stats) for k in stats[0]}
    return torch.cat(vals),avg

def clone_state(m):
    return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}

def train(model,steps,batch,k,seed,val):
    groups=[
      {'params':model.tail_policy.parameters(),'lr':5e-5},
      {'params':model.endpoint_adapter.parameters(),'lr':1e-4},
      {'params':model.pair_policy.parameters(),'lr':1e-4},
      {'params':[model.source_pair_gate_logit,model.endpoint_pair_gate_logit],'lr':5e-5},
    ]
    opt=torch.optim.Adam(groups)
    dg=torch.Generator(device='cpu').manual_seed(seed)
    ag=torch.Generator(device=DEVICE).manual_seed(seed+2000)
    params=[p for p in model.parameters() if p.requires_grad]
    hist=[]
    v,vs=evaluate(model,val,k=4,batch=64)
    best=float(v.mean()); best_step=0; best_state=clone_state(model)
    row={'step':0,'val':best,**vs}; hist.append(row); print('COUPLED',json.dumps(row),flush=True)

    checkpoints={100,200,400,800,1200,steps}
    for step in range(1,steps+1):
        coords=torch.rand(batch,50,2,generator=dg).to(DEVICE)
        cr,starts=cb.repeat_rollouts(coords,k)
        cost,lp,tstats=model.rollout(cr,starts,False,ag,collect_stats=True)
        reward=-cost.view(batch,k)
        adv=reward-reward.mean(1,keepdim=True)
        loss=-(adv.detach()*lp.view(batch,k)).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if step in checkpoints:
            v,vs=evaluate(model,val,k=4,batch=64)
            vm=float(v.mean())
            row={'step':step,'train_cost':float(cost.mean()),'loss':float(loss),
                 'sample_source_deviation_fraction':tstats['source_deviation_fraction'],
                 'val':vm,**vs}
            hist.append(row); print('COUPLED',json.dumps(row),flush=True)
            if vm<best:
                best=vm; best_step=step; best_state=clone_state(model)
    model.load_state_dict(best_state,strict=True)
    return hist,best,best_step

def main():
    torch.manual_seed(20260918); torch.cuda.manual_seed_all(20260918)
    test=cb.make_fixed(20260918,1024,50)
    val=cb.make_fixed(20260919,512,50)
    host=cb.build_host()
    base=cb.evaluate_seq(host,test,k=8,batch=64)
    base_mean=float(base.mean())

    model=CoupledASCC(host).to(DEVICE)
    init_test,init_stats=evaluate(model,test,k=8,batch=64)
    init_mean=float(init_test.mean())

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    hist,best_val,best_step=train(model,1200,24,4,61001,val)
    torch.cuda.synchronize(); train_seconds=time.perf_counter()-t
    train_peak=torch.cuda.max_memory_allocated()/1024**3

    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t=time.perf_counter()
    final,stats=evaluate(model,test,k=8,batch=64)
    torch.cuda.synchronize(); eval_seconds=time.perf_counter()-t
    eval_peak=torch.cuda.max_memory_allocated()/1024**3
    mean=float(final.mean())
    imp=(base_mean-mean)/base_mean*100

    assessment='GREEN' if imp>0.10 else ('YELLOW' if imp>=-0.05 else 'RED')
    summary={
      'round_id':'E4-H1C-LEHD-COUPLED',
      'host':'LEHD NeurIPS 2023 official checkpoint-150 frozen',
      'graph_size':50,'test_instances':1024,'validation_instances':512,'eval_rollouts':8,
      'base_mean':base_mean,
      'initial_coupled_mean':init_mean,
      'initial_source_deviation_fraction':init_stats['source_deviation_fraction'],
      'best_val_mean':best_val,'best_step':best_step,
      'coupled_mean':mean,'improvement_vs_base_percent':imp,
      'eval_seconds':eval_seconds,'eval_peak_gb':eval_peak,
      'train_seconds':train_seconds,'train_peak_gb':train_peak,
      'pair_parameters':sum(p.numel() for p in model.pair_policy.parameters()),
      'endpoint_adapter_parameters':sum(p.numel() for p in model.endpoint_adapter.parameters()),
      'tail_parameters':sum(p.numel() for p in model.tail_policy.parameters()),
      **stats,'history':hist,'assessment':assessment
    }
    summary['interpretation']=(
      'Coupled pair-aware source-endpoint policy produces a >0.10% gain over frozen LEHD.'
      if assessment=='GREEN' else
      'Coupled policy remains near the LEHD baseline; construction-order freedom is not yet beneficial under this frozen-host integration.'
      if assessment=='YELLOW' else
      'Coupled pair-aware policy materially degrades frozen LEHD.'
    )
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    torch.save({'model':model.state_dict()},OUT/'coupled_best.pt')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
