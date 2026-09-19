"""Exhaustive n=5 reachability and branch-count check; CPU only."""
import itertools
import json
import math
import torch
from strong_adapter import Forest

torch.set_num_threads(2)
n=5
tours=torch.tensor([(0,)+p for p in itertools.permutations(range(1,n))])
successors=torch.empty_like(tours).scatter(1,tours,tours.roll(-1,1))
orders=torch.tensor(list(itertools.permutations(range(n))))
target=successors.repeat_interleave(len(orders),0)
order=orders.repeat(len(tours),1)
b=len(order); rows=torch.arange(b)
f=Forest(torch.zeros(b,n,2)); checks=[]
for depth in range(n):
    mask,resolved=f.masks()
    m=n-depth
    assert ((~mask).sum(-1)[~resolved]==max(m-1,1)).all()
    # Enumerate every possible Hamiltonian completion of every visited state.
    consistent=((f.succ[:,None,:]<0)|(f.succ[:,None,:]==successors[None,:,:])).all(-1)
    assert (consistent.sum(-1)==math.factorial(max(m-1,0))).all()
    tail=order[:,depth]; head=target[rows,tail]
    assert (~mask[rows,tail,head]).all()
    f.add(tail,head)
    checks.append(dict(depth=depth,components=m,legal_heads=max(m-1,1),
                       completions=math.factorial(max(m-1,0))))
assert torch.equal(f.succ,target)
print(json.dumps(dict(n=n,tours=len(tours),source_orders=len(orders),
                      full_trajectories=b,all_reachable=True,checks=checks),indent=2))
