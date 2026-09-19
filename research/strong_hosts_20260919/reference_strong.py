"""Classical reference for prespecified held-out instances; never used for training."""
import argparse, json, sys, time, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import torch
ELKAI=Path('/workspace/计算群论/environments/ascc-reference-elkai-2.0.1')
sys.path.insert(0,str(ELKAI))
import elkai

def solve(x):
    x=x.astype(np.float64)
    distances=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    scaled=np.rint(distances*1_000_000).astype(np.int64)
    begin=time.monotonic()
    route=elkai.DistanceMatrix(scaled.tolist()).solve_tsp(runs=3)
    assert route[0]==route[-1] and sorted(route[:-1])==list(range(len(x)))
    pi=np.array(route[:-1]); cost=float(distances[pi,np.roll(pi,-1)].sum())
    return dict(tour=pi.tolist(),cost=cost,seconds=time.monotonic()-begin)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    d=torch.load(a.root/'am/datasets.pt',weights_only=False)
    dest=a.root/'reference';dest.mkdir(exist_ok=True)
    summary={}
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        for case in ('test200','cluster200','test500'):
            records=[]; begin=time.monotonic()
            for row in pool.map(solve,list(d[case].numpy()),chunksize=1):
                records.append(row)
                if len(records)%32==0: print(case,len(records),flush=True)
            costs=torch.tensor([v['cost'] for v in records],dtype=torch.float64)
            torch.save(costs,dest/f'{case}-costs.pt')
            (dest/f'{case}-tours.json').write_text(json.dumps(records))
            summary[case]=dict(mean=float(costs.mean()),wall_seconds=time.monotonic()-begin,
                               summed_solver_seconds=sum(v['seconds'] for v in records),
                               instances=len(records),coords_sha256=hashlib.sha256(d[case].numpy().tobytes()).hexdigest())
            (dest/'summary.json').write_text(json.dumps(dict(solver='elkai 2.0.1 / LKH3, RUNS=3',
                 integer_scale=1000000,rescored='original float64 Euclidean distances',
                 certified_optimal=False,cases=summary),indent=2))
            print('REFERENCE',case,summary[case],flush=True)

if __name__=='__main__': main()
