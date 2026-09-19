"""Canonical-style forest selector around unmodified official AM / ICAM hosts.

Native host context is preserved. This is an explicitly versioned host port,
not a numerical reimplementation of the older AM-style backbone.
"""
from __future__ import annotations
import importlib.util
import json
import math
import sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path('/workspace/计算群论')

def gather(x, idx):
    return x.gather(1, idx[..., None].expand(*idx.shape, x.size(-1)))

def select(lp, sampling=False, gen=None):
    return torch.multinomial(lp.exp(), 1, generator=gen).squeeze(-1) if sampling else lp.argmax(-1)

def tour_cost(x, pi):
    y = gather(x, pi)
    return (y-y.roll(-1, 1)).norm(dim=-1).sum(-1)

class Host(nn.Module):
    def __init__(self, kind, checkpoint_path=None):
        super().__init__()
        self.kind = kind
        if kind == 'am':
            repo = ROOT/'repos/groupopt-official-am-kool'
            sys.path.insert(0, str(repo))
            from nets.attention_model import AttentionModel
            from problems import TSP
            folder = repo/'pretrained/tsp_100'
            a = json.loads((folder/'args.json').read_text())
            self.net = AttentionModel(a['embedding_dim'], a['hidden_dim'], TSP,
                                     n_encode_layers=a['n_encode_layers'],
                                     normalization=a['normalization'], tanh_clipping=a['tanh_clipping'])
            self.path = Path(checkpoint_path or folder/'epoch-99.pt')
            payload = torch.load(self.path, map_location='cpu', weights_only=False)
            self.net.load_state_dict(payload['model'], strict=True)
            self.net.set_decode_type('greedy')
        elif kind == 'icam':
            repo = ROOT/'repos/groupopt-modern-hosts/ICAM'
            spec = importlib.util.spec_from_file_location('official_icam', repo/'ICAM_TSP/TSPModel_ICAM.py')
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            self.net = mod.TSPModel(embedding_dim=128, sqrt_embedding_dim=math.sqrt(128),
                                   encoder_layer_num=12, logit_clipping=50, ff_hidden_dim=512,
                                   eval_type='greedy')
            self.path = Path(checkpoint_path or repo/'pretrained/icam_tsp.pt')
            payload = torch.load(self.path, map_location='cpu', weights_only=False)
            self.net.load_state_dict(payload['model_state_dict'], strict=True)
        else:
            raise ValueError(kind)

    def encode(self, x):
        return (self.net.embedder(self.net._init_embed(x))[0] if self.kind == 'am'
                else self.net.encoder(x, torch.cdist(x, x), math.log2(x.size(1))))

    def logits(self, enc, x, first, tails, mask, initial=False):
        """B,Q query indices; B,Q,N true=invalid. Returns native logits."""
        b, n, d = enc.shape
        if self.kind == 'am':
            net = self.net
            fixed = net._precompute(enc, num_steps=first.size(1))
            context = (net.W_placeholder[None, None].expand(b, first.size(1), -1) if initial else
                       torch.cat((gather(enc, first), gather(enc, tails)), -1))
            q = fixed.context_node_projected + net.project_step_context(context)
            return net._one_to_many_logits(q, fixed.glimpse_key, fixed.glimpse_val,
                                          fixed.logit_key, mask)[0]
        dec = self.net.decoder
        q = dec.Wq_first(gather(enc, first)) + dec.Wq_last(gather(enc, tails))
        k, v = dec.Wk(enc), dec.Wv(enc)
        dist = (gather(x, tails)[:, :, None]-x[:, None]).norm(dim=-1)
        bias = (-dec.alpha1 * math.log2(n)*dist).masked_fill(mask, -torch.inf)
        a = bias.exp()
        ek = k.exp()
        weighted = (a @ (ek*v))/(a @ ek)
        out = q.sigmoid()*weighted
        score = out @ enc.transpose(1, 2)/math.sqrt(d) - dec.alpha2*math.log2(n)*dist
        return (50*score.tanh()).masked_fill(mask, -torch.inf)

    def initial(self, enc, x, sampling=False, gen=None, anchor=0, forced=None):
        b, n, _ = x.shape
        zero = torch.zeros(b, 1, dtype=torch.long, device=x.device)
        if self.kind == 'am':
            lp = self.logits(enc, x, zero, zero, torch.zeros(b, 1, n, dtype=torch.bool,
                                                         device=x.device), True).squeeze(1).log_softmax(-1)
            a = select(lp, sampling, gen) if forced is None else forced
            return a, lp.gather(1, a[:, None]).squeeze(1)
        return (zero.squeeze(1)+anchor if forced is None else forced), enc.new_zeros(b)

    @torch.no_grad()
    def native(self, x, anchor=0):
        """Calls official forward paths, independently of forest state."""
        self.eval()
        if self.kind == 'am':
            return self.net(x, return_pi=True)[2]
        from types import SimpleNamespace
        b, n, _ = x.shape
        dist = torch.cdist(x, x)
        net = self.net
        net.pre_forward(SimpleNamespace(problems=x, dist=dist, log_scale=math.log2(n)))
        cur = torch.full((b, 1), anchor, device=x.device, dtype=torch.long)
        net.decoder.set_q1(gather(net.encoded_nodes, cur))
        visited = torch.zeros(b, 1, n, device=x.device)
        visited.scatter_(2, cur[..., None], -torch.inf)
        seq = [cur.squeeze(1)]
        for _ in range(n-1):
            state = SimpleNamespace(batch_size=b, pomo_size=1, current_node=cur, ninf_mask=visited)
            cur_dist = dist.gather(1, cur[..., None].expand(b, 1, n))
            cur, _ = net(state, cur_dist)
            seq.append(cur.squeeze(1))
            visited = visited.clone().scatter_(2, cur[..., None], -torch.inf)
        return torch.stack(seq, 1)

class Forest:
    def __init__(self, x):
        b, n, _ = x.shape
        nodes = torch.arange(n, device=x.device).expand(b, n)
        self.comp = nodes.clone()
        self.starts = nodes.clone()
        self.succ = torch.full_like(nodes, -1)
        self.pred = torch.full_like(nodes, -1)
        self.steps = 0

    def masks(self):
        mask = self.pred.ge(0)[:, None].expand(-1, self.comp.size(1), -1).clone()
        if self.steps < self.comp.size(1)-1:
            mask |= self.comp[:, :, None].eq(self.comp[:, None, :])
        mask |= self.succ.ge(0)[:, :, None]
        return mask, self.succ.ge(0)

    def features(self, enc):
        same = self.comp[:, :, None].eq(self.comp[:, None, :]).to(enc.dtype)
        size = same.sum(-1, keepdim=True)
        mean = same @ enc / size
        return torch.cat((gather(enc, self.starts), mean, size/enc.size(1)), -1)

    def add(self, tail, head):
        rows = torch.arange(self.comp.size(0), device=tail.device)
        ct, ch = self.comp[rows, tail], self.comp[rows, head]
        start = self.starts[rows, tail]
        merged = self.comp.eq(ct[:, None]) | self.comp.eq(ch[:, None])
        self.comp = torch.where(self.comp.eq(ch[:, None]), ct[:, None], self.comp)
        self.starts = torch.where(merged, start[:, None], self.starts)
        self.succ = self.succ.clone(); self.succ[rows, tail] = head
        self.pred = self.pred.clone(); self.pred[rows, head] = tail
        self.steps += 1

class CanonicalSelector(nn.Module):
    def __init__(self, d=128, heads=8):
        super().__init__()
        self.d, self.heads = d, heads
        self.graph = nn.Linear(d, d, bias=False)
        self.context = nn.Linear(2*d, d, bias=False)
        self.state = nn.Linear(2*d+1, d, bias=False)
        self.summary = nn.Linear(3, d, bias=False)
        self.nodes = nn.Linear(d, 3*d, bias=False)
        self.glimpse = nn.Linear(d, d, bias=False)

    def forward(self, enc, features, head_lp, distances, last, mask):
        probs = head_lp.exp()
        safe_lp = torch.where(torch.isfinite(head_lp), head_lp, 0.)
        summary = torch.stack((-(probs*safe_lp).sum(-1)/math.log(enc.size(1)),
                               (probs*distances).sum(-1)/math.sqrt(2), probs.amax(-1)), -1).detach()
        nodes = enc + self.state(features) + self.summary(summary)
        k,v,l = self.nodes(nodes).chunk(3, -1)
        b,n,d = enc.shape; h=self.heads
        q = self.context(torch.cat((self.graph(enc.mean(1)), gather(enc,last[:,None]).squeeze(1)), -1))
        q = q.reshape(b,h,1,d//h)
        k = k.reshape(b,n,h,d//h).transpose(1,2)
        v = v.reshape(b,n,h,d//h).transpose(1,2)
        attn = (q @ k.transpose(-1,-2)/math.sqrt(d//h)).masked_fill(mask[:,None,None], -torch.inf).softmax(-1)
        g = self.glimpse((attn@v).reshape(b,d))
        return 10*(torch.einsum('bd,bnd->bn',g,l)/math.sqrt(d)).tanh()

class Adapter(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.host = Host(kind)
        self.selector = CanonicalSelector()

    def distributions(self, enc, x, starts, features, pair_mask, tail_mask, last, mode):
        b,n,_ = x.shape
        nodes = torch.arange(n, device=x.device).expand(b,n)
        # Rows belonging to resolved sources may have no heads; keep those rows
        # numerically safe and exclude them from the source distribution.
        dead = pair_mask.all(-1, keepdim=True)
        safe_mask = pair_mask & ~dead
        logits = self.host.logits(enc,x,starts,nodes,safe_mask)
        head_lp = logits.log_softmax(-1)
        head_lp = head_lp.masked_fill(dead, -torch.inf)
        source_logits = self.selector(enc,features,head_lp,torch.cdist(x,x),last,tail_mask)
        source_lp = source_logits.masked_fill(tail_mask,-torch.inf).log_softmax(-1)
        return source_lp, head_lp, source_logits

    def rollout(self, x, mode='learned', sampling=False, gen=None, actions=None,
                checkpoint_steps=False, anchor=0):
        """Modes route/capacity/random/learned. No hard continuation gate."""
        self.eval()  # preserve official normalization statistics during fine tuning
        b,n,_=x.shape; rows=torch.arange(b,device=x.device)
        enc=self.host.encode(x)
        first, init_ll=self.host.initial(enc,x,sampling,gen,anchor,
                                       None if actions is None else actions['first'])
        state=Forest(x); last=first; route_tail=first
        tails=[]; heads=[]; lls=[]; entropies=[]; dev=[]
        for t in range(n):
            pm,tm=state.masks()
            if mode == 'route':
                tail=route_tail
                head_lp=self.host.logits(enc,x,state.starts[rows,tail,None],tail[:,None],pm[rows,tail,None]).squeeze(1).log_softmax(-1)
                sll=enc.new_zeros(b); sent=sll
            else:
                features=state.features(enc)
                args=(enc,x,state.starts,features,pm,tm,last,mode)
                slp,hlp,slogits=(checkpoint(self.distributions,*args,use_reentrant=False) if checkpoint_steps
                                 else self.distributions(*args))
                if mode=='learned': tail=select(slp,sampling,gen)
                elif mode=='random': tail=torch.multinomial((~tm).float(),1,generator=gen).squeeze(1)
                elif mode=='capacity': tail=route_tail
                elif mode=='fixed': tail=(~tm).long().argmax(1)
                else: raise ValueError(mode)
                if actions is not None: tail=actions['tails'][:,t]
                head_lp=hlp[rows,tail]
                if mode=='capacity':
                    head_lp=(head_lp+slogits).masked_fill(pm[rows,tail],-torch.inf).log_softmax(-1)
                sll=slp[rows,tail] if mode=='learned' else enc.new_zeros(b)
                safe=torch.where(torch.isfinite(slp),slp,0.)
                sent=-(slp.exp()*safe).sum(-1)
            head=select(head_lp,sampling,gen) if actions is None else actions['heads'][:,t]
            lls.append(sll+head_lp[rows,head])
            tails.append(tail); heads.append(head); entropies.append(sent)
            dev.append((tail!=route_tail).float())
            # Following a joined component means continuing at its open tail,
            # rather than blindly setting current=selected head.
            target_component=state.comp[rows,head]
            target_tails=state.comp.eq(target_component[:,None]) & state.succ.lt(0)
            next_tail=target_tails.long().argmax(1)
            state.add(tail,head); last=head; route_tail=next_tail
        costs=(x-gather(x,state.succ)).norm(dim=-1).sum(-1)
        # The AM first-node action remains a shared latent anchor. Its score is
        # included for both modes; ICAM's deterministic anchor has zero score.
        return dict(cost=costs,ll=torch.stack(lls,1).sum(1)+init_ll,
                    first=first,tails=torch.stack(tails,1),heads=torch.stack(heads,1),
                    successor=state.succ,entropy=torch.stack(entropies,1).mean(1),
                    deviation=torch.stack(dev,1).mean(1))

    def warmup_loss(self,x,tours,generator,mode,depth=None):
        """One uniformly sampled partial-state depth per minibatch.

        Endpoint-only teacher supervision over a shared high-quality tour set.
        Route and forest use the same tours/depth/data budget; no arbitrary
        source labels are supervised. Selector remains untouched in warmup.
        """
        self.eval(); b,n,_=x.shape; rows=torch.arange(b,device=x.device)
        succ=torch.empty(b,n,dtype=torch.long,device=x.device)
        succ.scatter_(1,tours,tours.roll(-1,1))
        if depth is None:
            depth=int(torch.randint(0,n-1,(1,),generator=generator,device=x.device))
        state=Forest(x)
        order=(tours if mode=='route' else torch.rand(b,n,device=x.device,generator=generator).argsort(1))
        for t in range(depth):
            tail=order[:,t]; state.add(tail,succ[rows,tail])
        tail=order[:,depth]; target=succ[rows,tail]
        enc=self.host.encode(x)
        pm,_=state.masks()
        lp=self.host.logits(enc,x,state.starts[rows,tail,None],tail[:,None],pm[rows,tail,None]).squeeze(1).log_softmax(-1)
        return -lp[rows,target].mean()
