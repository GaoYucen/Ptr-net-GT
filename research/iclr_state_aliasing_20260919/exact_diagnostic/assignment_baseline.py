"""Post-pilot stronger polynomial-time heuristic: directed cycle-cover relaxation.
For every first action, fix source->candidate and solve the remaining assignment,
forbid self loops, allow nontrivial subtours. Evaluate chosen action by exact Q.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from run import stats,lower_bounds

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');a=p.parse_args();out=Path(a.output);summary={'protocol':'Post-pilot stronger heuristic. For each legal source endpoint, condition that edge and solve the minimum directed cycle cover using Hungarian assignment (self-loops forbidden; other subtours allowed). Choose smallest relaxation value; evaluate exact optimal-completion regret of that first action. CPU latency is informational for these tiny states only, not a neural speed claim.','scipy_version':scipy.__version__,'datasets':{}}
    for name in ['iid_n9','cluster_n9','ood_n7','ood_n11','ood_n13']:
        ds=[dict(np.load(f)) for f in sorted((out/'data').glob(name+'_s*.npz'))]
        d={key:np.concatenate([z[key] for z in ds],0) if key in ('x','tails','q','internal') else ds[0][key] for key in ds[0]}
        x=d['x'].astype(float);heads=d['heads'];tails=d['tails'];g,s,m=tails.shape;estimated=np.empty((g,s,m-1));t0=time.perf_counter()
        dist=np.linalg.norm(x[:,:,None]-x[:,None,:],axis=-1)
        for geom in range(g):
            for state in range(s):
                w=dist[geom][tails[geom,state,:,None],heads[None,:]];np.fill_diagonal(w,np.inf)
                for first in range(1,m):
                    cols=np.delete(np.arange(m),first);sub=w[1:,cols];ri,ci=linear_sum_assignment(sub)
                    estimated[geom,state,first-1]=w[0,first]+sub[ri,ci].sum()
        elapsed=time.perf_counter()-t0;actions=estimated.argmin(-1);r,_,_=lower_bounds(d);regret=np.take_along_axis(r,actions[...,None],-1).squeeze(-1)
        # Independently check every relaxation cost lower-bounds the exact connector Q.
        assert np.all(estimated<=d['q']+1e-8),float((estimated-d['q']).max())
        np.savez_compressed(out/f'{name}_assignment.npz',estimated_completion=estimated,actions=actions,regret=regret)
        summary['datasets'][name]=dict(regret=stats(regret.mean(1)),accuracy=stats((regret<1e-8).mean(1)),cpu_seconds=elapsed,seconds_per_forest=elapsed/(g*s),n_geometry=g)
        print(name,summary['datasets'][name],flush=True)
    (out/'assignment_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
