"""Frozen official ICAM with an equal-capacity endpoint residual pilot.

The blind arm receives only static encodings, source first/last, and legal heads.
The aware arm additionally receives each path's far end, mean encoding and size.
All source choices are exogenous: route continuation or uniform open tail.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from strong_adapter import Host, gather


class State:
    def __init__(self, enc):
        b,n,d=enc.shape
        self.start=torch.arange(n,device=enc.device).expand(b,n).clone()
        self.end=self.start.clone()
        self.mean=enc.clone()
        self.size=enc.new_ones(b,n,1)
        self.succ=torch.full_like(self.start,-1)
        self.pred=torch.full_like(self.start,-1)
        self.steps=0

    def add(self, tail, head):
        rows=torch.arange(len(tail),device=tail.device)
        cs=self.start[rows,tail]; ct=self.start[rows,head]
        end=self.end[rows,head]
        sa=self.size[rows,tail]; sb=self.size[rows,head]
        mean=(self.mean[rows,tail]*sa+self.mean[rows,head]*sb)/(sa+sb)
        merged=self.start.eq(cs[:,None])|self.start.eq(ct[:,None])
        self.start=torch.where(merged,cs[:,None],self.start)
        self.end=torch.where(merged,end[:,None],self.end)
        self.mean=torch.where(merged[:,:,None],mean[:,None],self.mean)
        self.size=torch.where(merged[:,:,None],(sa+sb)[:,None],self.size)
        self.succ[rows,tail]=head; self.pred[rows,head]=tail
        self.steps+=1

    def mask(self, tail):
        rows=torch.arange(len(tail),device=tail.device)
        out=self.pred.ge(0)
        if self.steps<self.start.size(1)-1:
            out=out|self.start.eq(self.start[rows,tail,None])
        return out


def teacher_state(enc,tour,depth,mode,generator):
    """Build independent directed teacher fragments via pointer doubling."""
    b,n,d=enc.shape; rows=torch.arange(b,device=enc.device)
    nodes=torch.arange(n,device=enc.device).expand(b,n)
    succ=torch.empty_like(tour).scatter(1,tour,tour.roll(-1,1))
    pred=torch.empty_like(tour).scatter(1,tour,tour.roll(1,1))
    order=tour if mode=='route' else torch.rand(b,n,device=enc.device,generator=generator).argsort(1)
    chosen=torch.zeros(b,n,dtype=torch.bool,device=enc.device)
    chosen.scatter_(1,order[:,:depth],True)
    start=torch.where(chosen.gather(1,pred),pred,nodes)
    end=torch.where(chosen,succ,nodes)
    for _ in range(math.ceil(math.log2(n))):
        start=start.gather(1,start); end=end.gather(1,end)
    sizes=enc.new_zeros(b,n,1).scatter_add_(1,start[:,:,None],enc.new_ones(b,n,1))
    sums=enc.new_zeros(b,n,d).scatter_add_(1,start[:,:,None].expand(b,n,d),enc)
    state=State(enc); state.start=start; state.end=end
    state.size=gather(sizes,start); state.mean=gather(sums,start)/state.size
    state.succ=torch.where(chosen,succ,-1)
    state.pred=torch.where(chosen.gather(1,pred),pred,-1); state.steps=depth
    tail=order[:,depth]; target=succ[rows,tail]
    assert not state.mask(tail)[rows,target].any()
    return state,tail,target


class Residual(nn.Module):
    def __init__(self,d=128,width=64):
        super().__init__()
        self.source=nn.Sequential(nn.Linear(3*d+1,width),nn.SiLU(),nn.Linear(width,width))
        self.target=nn.Sequential(nn.Linear(3*d+1,width),nn.SiLU(),nn.Linear(width,width))
        self.geometry=nn.Sequential(nn.Linear(7,width),nn.SiLU(),nn.Linear(width,1))
        self.scale=nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.geometry[-1].weight); nn.init.zeros_(self.geometry[-1].bias)

    def forward(self,enc,x,state,tail,aware):
        b,n,d=enc.shape; rows=torch.arange(b,device=enc.device)
        head=state.start[rows,tail]
        start_enc=enc[rows,head]; tail_enc=enc[rows,tail]
        if aware:
            sm=state.mean[rows,tail]; ss=state.size[rows,tail]/n
            target_tail=state.end; tm=state.mean; ts=state.size/n
        else:
            sm=(start_enc+tail_enc)/2; ss=enc.new_ones(b,1)/n
            target_tail=torch.arange(n,device=x.device).expand(b,n)
            tm=enc; ts=enc.new_ones(b,n,1)/n
        q=self.source(torch.cat((start_enc,tail_enc,sm,ss),-1))
        k=self.target(torch.cat((enc,gather(enc,target_tail),tm,ts),-1))
        # Shared component-set context: every legal target component can affect
        # every endpoint score. The blind arm gets the identical operation on
        # its restricted observations. No universality claim follows from this
        # finite-width single attention layer.
        att=((q[:,None]*k).sum(-1)/math.sqrt(k.size(-1))).masked_fill(state.mask(tail),-torch.inf).softmax(-1)
        q=q+(att[:,:,None]*k).sum(1)
        a=x[rows,tail,None]; h=x[rows,head,None]; e=gather(x,target_tail)
        geom=torch.stack(((a-x).norm(dim=-1),(h-e).norm(dim=-1),
                          (a-e).norm(dim=-1),(h-x).norm(dim=-1),
                          (x-e).norm(dim=-1),ts.squeeze(-1),ss.expand(-1,n)), -1)
        return self.scale*(q[:,None]*k).sum(-1)/math.sqrt(k.size(-1))+self.geometry(geom).squeeze(-1)


class Model(nn.Module):
    def __init__(self,aware=True,kind='icam'):
        super().__init__(); self.host=Host(kind); self.residual=Residual(); self.aware=aware
        self.host.requires_grad_(False); self.host.eval()

    def logits(self,enc,x,state,tail):
        rows=torch.arange(len(x),device=x.device); mask=state.mask(tail)
        first=state.start[rows,tail,None]
        native=self.host.logits(enc,x,first,tail[:,None],mask[:,None]).squeeze(1)
        return native+self.residual(enc,x,state,tail,self.aware)

    def conditional(self,enc,x,tour,depth,mode,generator):
        state,tail,target=teacher_state(enc,tour,depth,mode,generator)
        lp=self.logits(enc,x,state,tail).log_softmax(-1)
        return -lp[torch.arange(len(x),device=x.device),target],lp.argmax(1).eq(target)

    @torch.no_grad()
    def rollout(self,x,enc=None,mode='route',generator=None):
        if enc is None: enc=self.host.encode(x)
        b,n,d=enc.shape; state=State(enc); rows=torch.arange(b,device=x.device)
        route_tail=self.host.initial(enc,x)[0]
        for _ in range(n):
            tail=(route_tail if mode=='route' else
                  torch.multinomial(state.succ.lt(0).float(),1,generator=generator).squeeze(1))
            logits=self.logits(enc,x,state,tail); head=logits.argmax(1)
            route_tail=state.end[rows,head]
            state.add(tail,head)
        cost=(x-gather(x,state.succ)).norm(dim=-1).sum(1)
        return cost,state.succ
