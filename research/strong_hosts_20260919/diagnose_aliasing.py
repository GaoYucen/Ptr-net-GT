"""CPU exact counterexample to endpoint-state sufficiency, not a benchmark.

Find two legal forests with identical endpoint query inputs but disjoint optimal
heads. The selector can distinguish their features; this does NOT establish
failure of the joint policy or a global ASCC bound.
"""
import itertools
import json
import torch
from strong_adapter import Forest, Host

torch.set_num_threads(2)
n=7; pi=torch.tensor([(0,)+p for p in itertools.permutations(range(1,n))])
succ=torch.empty_like(pi).scatter(1,pi,pi.roll(-1,1))
edges=[[(1,3),(2,4)],[(1,4),(2,3)]]
forests=[]; permitted=[]
for e in edges:
    f=Forest(torch.zeros(1,n,2))
    for a,b in e: f.add(torch.tensor([a]),torch.tensor([b]))
    forests.append(f)
    permitted.append(torch.stack([succ[:,a]==b for a,b in e]).all(0))
assert torch.equal(forests[0].masks()[0][:,0],forests[1].masks()[0][:,0])
g=torch.Generator().manual_seed(90919017)
for trial in range(1000):
    x=torch.rand(1,n,2,generator=g)
    d=torch.cdist(x.double(),x.double())[0]
    costs=d[pi,pi.roll(-1,1)].sum(1)
    q=[]
    for ok in permitted:
        q.append([float(costs[ok&(succ[:,0]==head)].min()) if (ok&(succ[:,0]==head)).any()
                  else None for head in range(n)])
    best=[min((value,head) for head,value in enumerate(v) if value is not None)[1] for v in q]
    if best[0]!=best[1] and all(q[i][best[1-i]]-q[i][best[i]]>.05 for i in range(2)):
        break
else: raise RuntimeError('no counterexample found')
checks={}
with torch.no_grad():
    for kind in ('am','icam'):
        host=Host(kind).eval(); enc=host.encode(x); lp=[]
        for f in forests:
            lp.append(host.logits(enc,x,f.starts[:,0,None],torch.tensor([[0]]),
                                  f.masks()[0][:,0,None]).log_softmax(-1))
        legal=torch.isfinite(lp[0])
        assert torch.equal(lp[0],lp[1])
        assert not torch.equal(forests[0].features(enc),forests[1].features(enc))
        checks[kind]=dict(endpoint_logprobs_exactly_identical=True,
                          selector_features_differ=True,
                          endpoint_probabilities=lp[0].exp()[0,0].tolist())
print(json.dumps(dict(kind='constructed representational counterexample',seed=90919017,
    search_trial=trial,coordinates=x[0].tolist(),partial_edges=edges,source=0,
    exact_best_completion_by_head=q,optimal_heads=best,
    cross_state_wrong_head_regret=[q[i][best[1-i]]-q[i][best[i]] for i in range(2)],
    hosts=checks,limitation='Selected-source endpoint ambiguity only. Joint source policy sees different features; this is not a bound on its attainable cost.'),indent=2))
