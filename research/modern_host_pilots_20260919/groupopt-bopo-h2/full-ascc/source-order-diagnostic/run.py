from __future__ import annotations
import json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, "/workspace/groupopt-bopo-h2/full-ascc")
import train_bopo_full_ascc as m

ROOT=Path("/workspace/groupopt-bopo-h2/full-ascc")
OUT=ROOT/"source-order-diagnostic"
device=torch.device("cuda",0)
torch.cuda.set_device(0)
bopo=m.load_bopo(device)
adapter=m.FullASCCAdapter().to(device)
payload=torch.load(ROOT/"seed1234-formal-v1/best.pt",map_location=device,weights_only=False)
adapter.load_state_dict(payload["adapter"],strict=True)
adapter.eval()

def instrument(coords, enc, k):
    coords, enc, anchors, anchor_e = m.expand_rollouts(coords, enc, k)
    r,n,_=coords.shape
    batch=torch.arange(r,device=device)
    succ=torch.full((r,n),-1,dtype=torch.long,device=device)
    pred=torch.full((r,n),-1,dtype=torch.long,device=device)
    comp=torch.arange(n,device=device,dtype=torch.long)[None].expand(r,n).clone()
    last_head=anchors.clone()
    last_head_e=anchor_e
    any_div=torch.zeros(r,dtype=torch.bool,device=device)
    any_forced=torch.zeros(r,dtype=torch.bool,device=device)
    c={
      "decisions":0,"source_ne_prev_head":0,
      "prev_head_legal":0,"choose_other_when_prev_head_legal":0,
      "prev_head_illegal":0,
      "nonterminal_decisions":0,"nonterminal_source_ne_prev_head":0,
      "nonterminal_prev_head_legal":0,"nonterminal_choose_other_when_prev_head_legal":0,
      "nonterminal_prev_head_illegal":0,
    }
    for edge in range(n):
        tails,heads,packed_log_p,packed_summary=m.bopo_all_source_head_proposal(
            bopo,enc,coords,succ,pred,comp,edge
        )
        summary_full=torch.zeros((r,n,3),device=device,dtype=enc.dtype)
        summary_full.scatter_(1,tails[:,:,None].expand(r,tails.size(1),3),packed_summary)
        pstate=m.path_state_features(enc,pred,comp)
        tail_mask=succ.ge(0)
        if edge==0:
            selected_tail=anchors
        else:
            lp=adapter.tail_log_probabilities(anchor_e,last_head_e,enc,pstate,summary_full,tail_mask)
            selected_tail=lp.argmax(-1)
            prev_legal=~tail_mask[batch,last_head]
            diff=selected_tail.ne(last_head)
            c["decisions"]+=r
            c["source_ne_prev_head"]+=int(diff.sum())
            c["prev_head_legal"]+=int(prev_legal.sum())
            c["choose_other_when_prev_head_legal"]+=int((prev_legal & diff).sum())
            c["prev_head_illegal"]+=int((~prev_legal).sum())
            any_div |= diff
            any_forced |= ~prev_legal
            if edge < n-1:
                c["nonterminal_decisions"]+=r
                c["nonterminal_source_ne_prev_head"]+=int(diff.sum())
                c["nonterminal_prev_head_legal"]+=int(prev_legal.sum())
                c["nonterminal_choose_other_when_prev_head_legal"]+=int((prev_legal & diff).sum())
                c["nonterminal_prev_head_illegal"]+=int((~prev_legal).sum())

        source_pos=tails.eq(selected_tail[:,None]).long().argmax(1)
        head_lp=packed_log_p[batch,source_pos]
        selected_head=heads[batch,head_lp.argmax(-1)]
        succ=succ.clone(); pred=pred.clone()
        succ[batch,selected_tail]=selected_head
        pred[batch,selected_head]=selected_tail
        if edge<n-1:
            left=comp.gather(1,selected_tail[:,None])
            right=comp.gather(1,selected_head[:,None])
            merged=torch.minimum(left,right)
            comp=torch.where(comp.eq(left)|comp.eq(right),merged,comp)
        last_head=selected_head
        last_head_e=enc[batch,selected_head]
    return c,int(any_div.sum()),int(any_forced.sum()),r

tot={}; any_div=any_forced=rollouts=0
with torch.inference_mode():
    for step in range(0,1000,10):
        problems,_=torch.load(m.BOPO_TSP/f"data/tsp_n50_{step}.pkl",map_location=device,weights_only=False)
        for a in range(0,problems.size(0),20):
            x=problems[a:a+20]
            enc=m.encode_frozen(bopo,x)
            c,d,f,r=instrument(x,enc,50)
            for k,v in c.items(): tot[k]=tot.get(k,0)+v
            any_div+=d; any_forced+=f; rollouts+=r

def rate(num,den): return float(num/den) if den else 0.0
summary={
 "best_step":int(payload["step"]),
 "official_augmented_instances":8000,
 "pomo":50,
 "rollouts":rollouts,
 "source_decisions_excluding_initial":tot["decisions"],
 "source_ne_previous_endpoint_rate":rate(tot["source_ne_prev_head"],tot["decisions"]),
 "previous_endpoint_illegal_as_next_source_rate":rate(tot["prev_head_illegal"],tot["decisions"]),
 "choose_other_source_given_previous_endpoint_legal_rate":rate(tot["choose_other_when_prev_head_legal"],tot["prev_head_legal"]),
 "nonterminal_source_ne_previous_endpoint_rate":rate(tot["nonterminal_source_ne_prev_head"],tot["nonterminal_decisions"]),
 "nonterminal_previous_endpoint_illegal_as_next_source_rate":rate(tot["nonterminal_prev_head_illegal"],tot["nonterminal_decisions"]),
 "nonterminal_choose_other_source_given_previous_endpoint_legal_rate":rate(tot["nonterminal_choose_other_when_prev_head_legal"],tot["nonterminal_prev_head_legal"]),
 "rollouts_with_any_source_deviation_rate":rate(any_div,rollouts),
 "rollouts_with_any_forced_fragment_jump_rate":rate(any_forced,rollouts),
 "interpretation":"Post-hoc only; not used for training, checkpoint selection, or official-test tuning."
}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
