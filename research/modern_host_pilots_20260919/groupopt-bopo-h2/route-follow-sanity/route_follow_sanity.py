from __future__ import annotations
import json,sys,time
from pathlib import Path
import torch
TSP=Path('/workspace/groupopt-modern-hosts/BOPO/TSP')
BASE=Path('/workspace/groupopt-bopo-h2/official-repro')
OUT=Path('/workspace/groupopt-bopo-h2/route-follow-sanity')
sys.path.insert(0,str(TSP))
from TSPModel import TSPModel,_get_encoding
torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device=torch.device('cuda',0)
params={'start_node':'pomo','embedding_dim':128,'sqrt_embedding_dim':128**0.5,'encoder_layer_num':6,'qkv_dim':16,'head_num':8,'logit_clipping':10,'ff_hidden_dim':512,'eval_type':'argmax'}
model=TSPModel(**params).to(device)
ck=torch.load(TSP/'result/saved_n50/checkpoint-200.pt',map_location=device)
model.load_state_dict(ck['model_state_dict'],strict=True); model.eval()
def route_follow(problems):
    # problems: [A, N, 2], where A includes the official 8-fold augmentations.
    A,N,_=problems.shape; K=N
    enc=model.encoder(problems); model.decoder.set_kv(enc)
    starts=torch.arange(N,device=device,dtype=torch.long)[None,:].expand(A,K)
    succ=torch.full((A,K,N),-1,dtype=torch.long,device=device)
    pred=torch.full((A,K,N),-1,dtype=torch.long,device=device)
    comp=torch.arange(N,device=device,dtype=torch.long)[None,None,:].expand(A,K,N).clone()
    last_head=starts
    seq=[starts]
    aidx=torch.arange(A,device=device)[:,None].expand(A,K)
    kidx=torch.arange(K,device=device)[None,:].expand(A,K)
    for step in range(N-1):
        tail=last_head
        is_start=pred.lt(0)
        comp_id=comp.gather(2,tail[:,:,None]).squeeze(2)
        same=comp.eq(comp_id[:,:,None])
        first=(same & is_start).to(torch.int8).argmax(2).long()
        valid_head=is_start & comp.ne(comp_id[:,:,None])
        model.decoder.set_q1(_get_encoding(enc,first))
        ninf=torch.full((A,K,N),float('-inf'),device=device)
        ninf[valid_head]=0.0
        probs=model.decoder(_get_encoding(enc,tail),ninf)
        head=probs.argmax(2)
        # Forest transition: tail is an end; head is the start of another component.
        if pred[aidx,kidx,head].ge(0).any(): raise RuntimeError('head is not a component start')
        succ[aidx,kidx,tail]=head
        pred[aidx,kidx,head]=tail
        child=comp.gather(2,head[:,:,None]).squeeze(2)
        comp=torch.where(comp.eq(child[:,:,None]),comp_id[:,:,None],comp)
        last_head=head; seq.append(head)
    seq=torch.stack(seq,dim=2)
    xy=problems[:,None,:,:].expand(A,K,N,2)
    ordered=xy.gather(2,seq[:,:,:,None].expand(A,K,N,2))
    cost=((ordered-ordered.roll(-1,2))**2).sum(3).sqrt().sum(2)
    return cost
ref=torch.load(BASE/'official_all_aug_pomo_costs.pt',map_location='cpu',weights_only=False)['costs']
parts=[]; t0=time.perf_counter()
with torch.inference_mode():
    for step in range(0,1000,10):
        problems,_=torch.load(TSP/f'data/tsp_n50_{step}.pkl',map_location=device,weights_only=False)
        c=route_follow(problems).detach().cpu().reshape(8,10,50)
        parts.append(c)
route=torch.cat(parts,dim=1)
elapsed=time.perf_counter()-t0
diff=(route-ref).abs()
ref_best=ref.min(2).values.min(0).values
route_best=route.min(2).values.min(0).values
best_diff=(route_best-ref_best).abs()
summary={
 'round_id':'E4-H2-BOPO-ROUTE-FOLLOW-SANITY',
 'raw_shape':list(route.shape),
 'raw_bitwise_equal':bool(torch.equal(route,ref)),
 'raw_max_abs_diff':float(diff.max()),
 'raw_mismatch_gt_1e7':int((diff>1e-7).sum()),
 'raw_mismatch_gt_1e6':int((diff>1e-6).sum()),
 'instance_bitwise_equal':bool(torch.equal(route_best,ref_best)),
 'instance_max_abs_diff':float(best_diff.max()),
 'instance_mismatch_gt_1e7':int((best_diff>1e-7).sum()),
 'official_mean':float(ref_best.double().mean()),
 'route_follow_mean':float(route_best.double().mean()),
 'eval_seconds':elapsed,
}
torch.save({'costs':route},OUT/'route_follow_all_aug_pomo_costs.pt')
torch.save({'costs':route_best},OUT/'route_follow_instance_aug_best_costs.pt')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if summary['instance_mismatch_gt_1e7'] != 0 or summary['instance_max_abs_diff'] > 1e-7:
    raise SystemExit('BOPO route-follow failed strict per-instance numeric equivalence')
