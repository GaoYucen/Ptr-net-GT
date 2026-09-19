import argparse, importlib.util, json, math, os, random, time
from pathlib import Path
import torch
import torch.nn as nn

V2_PATH=Path("/workspace/groupopt-bopo-tsp100/full-ascc-v2/train_bopo_tsp100_full_ascc_v2.py")
spec=importlib.util.spec_from_file_location("v2",V2_PATH)
v2=importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)

class GatedASCC(nn.Module):
    def __init__(self, p0=0.02):
        super().__init__()
        self.tail=v2.FullASCCAdapter()
        for p in self.tail.head_tail.parameters(): p.requires_grad_(False)
        for p in self.tail.head_node.parameters(): p.requires_grad_(False)
        self.gate=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(p0/(1-p0)))
        self.p0=float(p0)

    def tail_log_probabilities(self,*args,**kwargs):
        return self.tail.tail_log_probabilities(*args,**kwargs)

    def gate_prob(self,x):
        return torch.sigmoid(self.gate(x).squeeze(-1))

def forest_init(coords,enc,k):
    coords,enc,anchors,anchor_e=v2.expand_rollouts(coords,enc,k)
    r,n,_=coords.shape; device=coords.device
    succ=torch.full((r,n),-1,dtype=torch.long,device=device)
    pred=torch.full((r,n),-1,dtype=torch.long,device=device)
    comp=torch.arange(n,device=device,dtype=torch.long)[None].expand(r,n).clone()
    return coords,enc,anchors,anchor_e,succ,pred,comp

@torch.no_grad()
def route_follow_cost(bopo,coords,enc,k):
    coords,enc,anchors,anchor_e,succ,pred,comp=forest_init(coords,enc,k)
    r,n,_=coords.shape; batch=torch.arange(r,device=coords.device)
    total=torch.zeros(r,device=coords.device); route_tail=None
    for edge in range(n):
        tails,heads,frozen_log_p,_=v2.bopo_all_source_head_proposal(bopo,enc,coords,succ,pred,comp,edge)
        st=anchors if edge==0 else route_tail
        pos=tails.eq(st[:,None]).long().argmax(1)
        hp=frozen_log_p[batch,pos].argmax(-1); sh=heads[batch,hp]
        total += (coords[batch,st]-coords[batch,sh]).norm(dim=-1)
        # Determine route continuation from the PRE-MERGE component of selected head.
        if edge<n-1:
            tail_comps=comp.gather(1,tails)
            head_comp=comp[batch,sh]
            next_pos=tail_comps.eq(head_comp[:,None]).long().argmax(1)
            route_tail=tails[batch,next_pos]
        succ=succ.clone(); pred=pred.clone(); succ[batch,st]=sh; pred[batch,sh]=st
        if edge<n-1:
            left=comp.gather(1,st[:,None]); right=comp.gather(1,sh[:,None]); merged=torch.minimum(left,right)
            comp=torch.where(comp.eq(left)|comp.eq(right),merged,comp)
    return total

def rollout_gated(adapter,bopo,coords,enc,k,sample,generator=None,threshold=0.2):
    coords,enc,anchors,anchor_e,succ,pred,comp=forest_init(coords,enc,k)
    r,n,_=coords.shape; device=coords.device; batch=torch.arange(r,device=device)
    total=torch.zeros(r,device=device); ll=torch.zeros(r,device=device)
    last_head_e=anchor_e; route_tail=None
    p_sum=torch.zeros((),device=device); ent_sum=torch.zeros((),device=device); kl_sum=torch.zeros((),device=device)
    dev_count=0; decision_count=0
    eps=1e-8; p0=adapter.p0
    for edge in range(n):
        tails,heads,frozen_log_p,packed_summary=v2.bopo_all_source_head_proposal(bopo,enc,coords,succ,pred,comp,edge)
        if edge==0:
            st=anchors
        elif tails.size(1)==1:
            st=route_tail
        else:
            sf=torch.zeros((r,n,3),device=device,dtype=enc.dtype)
            sf.scatter_(1,tails[:,:,None].expand(r,tails.size(1),3),packed_summary.detach())
            pstate=v2.path_state_features(enc,pred,comp); mask=succ.ge(0)
            tlp=adapter.tail_log_probabilities(anchor_e,last_head_e,enc,pstate,sf,mask)
            default=route_tail
            alt_logits=tlp.clone(); alt_logits[batch,default]=-torch.inf
            alt_logp=alt_logits-torch.logsumexp(alt_logits,dim=-1,keepdim=True)
            if sample:
                cand=torch.multinomial(alt_logp.exp(),1,generator=generator).squeeze(1)
            else:
                cand=alt_logp.argmax(-1)
            def_pos=tails.eq(default[:,None]).long().argmax(1)
            cand_pos=tails.eq(cand[:,None]).long().argmax(1)
            sdef=packed_summary[batch,def_pos].detach(); scand=packed_summary[batch,cand_pos].detach()
            margin=(tlp[batch,cand]-tlp[batch,default]).unsqueeze(1)
            edge_frac=torch.full((r,1),float(edge)/float(n-1),device=device,dtype=enc.dtype)
            feat=torch.cat((margin,scand-sdef,sdef,edge_frac),dim=1)
            pdev=adapter.gate_prob(feat).clamp(eps,1-eps)
            if sample:
                u=torch.rand(pdev.shape,device=device,generator=generator)
                do=u<pdev
                gate_lp=torch.where(do,pdev.log(),torch.log1p(-pdev))
                alt_lp=alt_logp[batch,cand]
                ll=ll+gate_lp+torch.where(do,alt_lp,torch.zeros_like(alt_lp))
            else:
                do=pdev>threshold
            st=torch.where(do,cand,default)
            p_sum=p_sum+pdev.mean()
            ent_sum=ent_sum+(-(pdev*pdev.log()+(1-pdev)*torch.log1p(-pdev))).mean()
            kl_sum=kl_sum+(pdev*torch.log(pdev/p0)+(1-pdev)*torch.log((1-pdev)/(1-p0))).mean()
            dev_count += int(do.detach().sum()); decision_count += r

        pos=tails.eq(st[:,None]).long().argmax(1)
        hp=frozen_log_p[batch,pos].argmax(-1); sh=heads[batch,hp]
        total += (coords[batch,st]-coords[batch,sh]).norm(dim=-1)
        # Determine route continuation from the PRE-MERGE component of selected head.
        if edge<n-1:
            tail_comps=comp.gather(1,tails)
            head_comp=comp[batch,sh]
            next_pos=tail_comps.eq(head_comp[:,None]).long().argmax(1)
            route_tail=tails[batch,next_pos]
        succ=succ.clone(); pred=pred.clone(); succ[batch,st]=sh; pred[batch,sh]=st
        if edge<n-1:
            left=comp.gather(1,st[:,None]); right=comp.gather(1,sh[:,None]); merged=torch.minimum(left,right)
            comp=torch.where(comp.eq(left)|comp.eq(right),merged,comp)
        last_head_e=enc[batch,sh]
    denom=max(n-2,1)
    return total,ll,p_sum/denom,ent_sum/denom,kl_sum/denom,dev_count/max(decision_count,1)

@torch.no_grad()
def evaluate(adapter,bopo,coords,k,thresholds):
    enc=bopo.encoder(coords)
    base=route_follow_cost(bopo,coords,enc,k).reshape(coords.size(0),k).amin(1).double().mean().item()
    out={}
    for th in thresholds:
        c,_,p,e,kl,rate=rollout_gated(adapter,bopo,coords,enc,k,False,threshold=th)
        val=c.reshape(coords.size(0),k).amin(1).double().mean().item()
        out[str(th)]={"mean":val,"improvement":base-val,"deviation_rate":rate,"mean_pdev":float(p)}
    return base,out

@torch.no_grad()
def official_first100(adapter,bopo,threshold):
    problems,opt=torch.load(v2.BOPO_TSP/"data/tsp_n100_0.pkl",map_location="cuda",weights_only=False)
    parts=[]
    for a in range(0,problems.size(0),4):
        c=problems[a:a+4]; enc=bopo.encoder(c)
        costs,_,_,_,_,_=rollout_gated(adapter,bopo,c,enc,100,False,threshold=threshold)
        parts.append(costs.reshape(c.size(0),100).cpu())
    raw=torch.cat(parts,0).reshape(8,100,100)
    best=raw.amin(2).amin(0).double()
    opt=torch.as_tensor(opt).reshape(-1).double().cpu()
    return float(best.mean()),float(((best/opt)-1).mean()*100)

def save(path,adapter,step,val,threshold,config):
    tmp=path.with_suffix(".tmp")
    torch.save({"adapter":adapter.state_dict(),"step":step,"validation":val,"threshold":threshold,"config":config},tmp)
    os.replace(tmp,path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=1200)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--pomo-size",type=int,default=8)
    ap.add_argument("--validation-size",type=int,default=32)
    ap.add_argument("--eval-every",type=int,default=100)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--kl-coeff",type=float,default=0.002)
    ap.add_argument("--p0",type=float,default=0.02)
    ap.add_argument("--seed",type=int,default=1234)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    thresholds=[0.1,0.2,0.3,0.5]
    device=torch.device("cuda",0); torch.cuda.set_device(0)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); random.seed(args.seed)
    bopo=v2.load_bopo(device); adapter=GatedASCC(args.p0).to(device)
    optim=torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad],lr=args.lr)
    vg=torch.Generator(device="cpu").manual_seed(2026091803)
    validation=torch.rand((args.validation_size,100,2),generator=vg).to(device)
    dg=torch.Generator(device=device).manual_seed(args.seed+11)
    ag=torch.Generator(device=device).manual_seed(args.seed+12)
    base_val,vals=evaluate(adapter,bopo,validation,args.pomo_size,thresholds)
    best_val=base_val; best_step=0; best_thr=0.2
    cfg=vars(args).copy(); cfg["out"]=str(cfg["out"])
    save(args.out/"best.pt",adapter,0,best_val,best_thr,cfg)
    print(json.dumps({"step":0,"route_validation":base_val,"thresholds":vals},sort_keys=True),flush=True)
    metrics=args.out/"metrics.jsonl"
    for step in range(1,args.steps+1):
        adapter.train()
        coords=torch.rand((args.batch_size,100,2),generator=dg,device=device)
        with torch.no_grad():
            enc=bopo.encoder(coords)
            base=route_follow_cost(bopo,coords,enc,args.pomo_size)
        cost,ll,pdev,gent,kl,dr=rollout_gated(adapter,bopo,coords,enc,args.pomo_size,True,ag,0.2)
        advantage=(cost-base).detach()
        pg=(advantage*ll).mean()
        loss=pg+args.kl_coeff*kl
        optim.zero_grad(set_to_none=True); loss.backward()
        gn=torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad],1.0)
        optim.step()
        if step%args.eval_every==0 or step==args.steps:
            adapter.eval(); route_val,vs=evaluate(adapter,bopo,validation,args.pomo_size,thresholds)
            th,val=min(((float(t),d["mean"]) for t,d in vs.items()),key=lambda x:x[1])
            if val < best_val-1e-6:
                best_val=val; best_step=step; best_thr=th
                save(args.out/"best.pt",adapter,step,val,th,cfg)
            rec={"step":step,"loss":float(loss.detach()),"pg":float(pg.detach()),"kl":float(kl.detach()),
                 "grad_norm":float(gn),"train_residual_mean":float(advantage.double().mean()),
                 "train_pdev":float(pdev.detach()),"train_deviation_rate":dr,"gate_entropy":float(gent.detach()),
                 "route_validation":route_val,"validation":vs,"best_val":best_val,"best_step":best_step,"best_threshold":best_thr}
            with metrics.open("a") as f: f.write(json.dumps(rec,sort_keys=True)+"\n")
            print(json.dumps(rec,sort_keys=True),flush=True)
    payload=torch.load(args.out/"best.pt",map_location=device,weights_only=False)
    adapter.load_state_dict(payload["adapter"]); best_thr=float(payload["threshold"])
    mean100,gap100=official_first100(adapter,bopo,best_thr)
    summary={"best_step":int(payload["step"]),"best_validation":float(payload["validation"]),
             "route_validation":base_val,"best_threshold":best_thr,
             "official_first100_mean":mean100,"official_first100_gap_percent":gap100,
             "trainable_parameters":sum(p.numel() for p in adapter.parameters() if p.requires_grad),
             "mechanism":"frozen BOPO head + route-follow default + gated residual tail + paired BOPO residual baseline",
             "p0":args.p0,"kl_coeff":args.kl_coeff}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("=== GATED V3 SUMMARY ==="); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main()
