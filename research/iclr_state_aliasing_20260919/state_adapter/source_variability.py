import argparse, json
from pathlib import Path
import numpy as np
import torch
from component_adapter import Model as FrozenModel
from secondary_component import Model as SecondaryModel
from verify_strong import valid


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--host',choices=['icam','am'],required=True)
    p.add_argument('--decoder',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert (a.out/'aggregate.json').exists()
    sets=torch.load(a.out/'final_test_data.pt',weights_only=False)
    m=(SecondaryModel if a.decoder else FrozenModel)(kind=a.host).cuda().eval(); records=[]
    for folder in sorted(a.out.glob('random-*-seed*')):
        ck=torch.load(folder/'best.pt',weights_only=False);m.aware=ck['aware']
        if a.decoder:m.restore(ck)
        else:m.residual.load_state_dict(ck['residual'])
        for name,s in sets.items():
            vectors=[]
            for rng_seed in [9527319,9637319,9747319]:
                g=torch.Generator(device='cuda').manual_seed(rng_seed);out=[]
                for i in range(0,len(s['x']),32):
                    cost,succ=m.rollout(s['x'][i:i+32].cuda(),s['enc'][i:i+32].cuda(),'random',g)
                    valid(succ);out.append(cost.cpu())
                values=torch.cat(out);vectors.append(values)
                torch.save(values,folder/f'{name}-source-rng{rng_seed}-costs.pt')
            means=[float(v.double().mean()) for v in vectors]
            records.append(dict(training_seed=ck['seed'],aware=ck['aware'],dataset=name,
                                source_seeds=[9527319,9637319,9747319],means=means,
                                average=float(np.mean(means)),source_seed_sd=float(np.std(means,ddof=1))))
    (a.out/'source_rng_variability.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
