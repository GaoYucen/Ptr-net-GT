import argparse, sys, torch
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--tag', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--checkpoint', required=True)
a=ap.parse_args()
sys.path.insert(0, str(Path(a.root)/'src'))
from groupopt.models.am import AttentionModel
from groupopt.objectives import reinforce_loss

device=torch.device('cuda')
torch.manual_seed(20260918)
torch.cuda.manual_seed_all(20260918)
model=AttentionModel().to(device)
ckpt=torch.load(a.checkpoint,map_location=device,weights_only=False)
model.load_state_dict(ckpt['model'],strict=True)
model.train()
coord_gen=torch.Generator(device=device).manual_seed(314159)
coords=torch.rand(64,50,2,device=device,generator=coord_gen)
action_gen=torch.Generator(device=device).manual_seed(271828)
out=model(coords,decode_type='sampling',base_mode='native_conditional_free',generator=action_gen)
loss=reinforce_loss(out.cost,out.log_likelihood)
model.zero_grad(set_to_none=True)
loss.backward()
grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
torch.save({
 'tails':out.tails.detach().cpu(),
 'heads':out.heads.detach().cpu(),
 'cost':out.cost.detach().cpu(),
 'll':out.log_likelihood.detach().cpu(),
 'loss':float(loss.detach()),
 'grads':grads,
}, a.out)
print(a.tag, 'loss', float(loss.detach()), 'cost', float(out.cost.mean()), 'grad_params', len(grads))
