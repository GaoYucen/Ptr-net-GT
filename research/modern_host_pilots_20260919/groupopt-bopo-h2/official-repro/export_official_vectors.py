from __future__ import annotations
import json,re,sys
from pathlib import Path
import torch
TSP=Path('/workspace/groupopt-modern-hosts/BOPO/TSP')
OUT=Path('/workspace/groupopt-bopo-h2/official-repro')
sys.path.insert(0,str(TSP))
from TSPEnv import TSPEnv
from TSPModel import TSPModel
torch.cuda.set_device(0)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device=torch.device('cuda',0)
env_params={'problem_size':50,'B':50}
model_params={'start_node':'pomo','embedding_dim':128,'sqrt_embedding_dim':128**0.5,'encoder_layer_num':6,'qkv_dim':16,'head_num':8,'logit_clipping':10,'ff_hidden_dim':512,'eval_type':'argmax'}
env=TSPEnv(**env_params)
model=TSPModel(**model_params).to(device)
ckpt=torch.load(TSP/'result/saved_n50/checkpoint-200.pt',map_location=device)
model.load_state_dict(ckpt['model_state_dict'],strict=True); model.eval()
raw=[]; noaug=[]; augbest=[]
with torch.no_grad():
    for step in range(0,1000,10):
        fp=TSP/f'data/tsp_n50_{step}.pkl'
        env.load_problems(10,8,str(fp),device=device)
        reset,_,_=env.reset(); model.pre_forward(reset)
        state,goal,done=env.pre_step()
        while not done:
            selected,_=model(state); state,goal,done=env.step(selected)
        costs=(-goal).reshape(8,10,50).detach().cpu()
        raw.append(costs)
        per_aug=costs.min(dim=2).values
        noaug.append(per_aug[0])
        augbest.append(per_aug.min(dim=0).values)
raw=torch.cat(raw,dim=1)
noaug=torch.cat(noaug)
augbest=torch.cat(augbest)
torch.save({'costs':raw},OUT/'official_all_aug_pomo_costs.pt')
torch.save({'costs':augbest},OUT/'official_instance_aug_best_costs.pt')
torch.save({'costs':noaug},OUT/'official_instance_noaug_costs.pt')
log=(OUT/'official.log').read_text(errors='replace')
m=re.findall(r'AUGMENTATION SCORE:\s*([0-9.]+)',log)
logged=float(m[-1]) if m else None
summary={'instances':int(augbest.numel()),'augmentations':8,'pomo':50,'checkpoint_epoch':200,'noaug_mean':float(noaug.double().mean()),'aug_mean':float(augbest.double().mean()),'official_logged_aug_score':logged,'logged_vs_vector_abs':None if logged is None else abs(logged-float(augbest.double().mean())),'raw_shape':list(raw.shape)}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
