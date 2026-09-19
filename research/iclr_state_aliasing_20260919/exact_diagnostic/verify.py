"""Independent audit of saved geometry splits, masks, associations and exact labels."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,sys
import numpy as np
import torch
from run import features, Model

def dp_q(x,heads,tails):
    w=np.linalg.norm(x[tails][:,None]-x[heads][None,:],axis=-1);m=len(heads)
    @lru_cache(None)
    def solve(last,remaining):
        if not remaining:return w[last,0]
        return min(w[last,j]+solve(j,remaining^(1<<j)) for j in range(1,m) if remaining&(1<<j))
    allbits=(1<<m)-2
    return np.array([w[0,a]+solve(a,allbits^(1<<a)) for a in range(1,m)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results');a=ap.parse_args();out=Path(a.output)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'strong_hosts_20260919'));from strong_adapter import Forest
    files=[f for f in (out/'data').glob('*.npz') if 'all_pairings' not in f.name]
    sets={};checks=[];maxerr=0.;total=0
    for f in files:
        d=dict(np.load(f));xx=d['x'];g,n,_=xx.shape;h=d['heads'];tt=d['tails'];ns=tt.shape[1]
        hashes={hashlib.sha256(x.tobytes()).hexdigest() for x in xx};assert len(hashes)==g
        split='train' if f.stem=='train' else 'validation' if f.stem=='validation' else 'test'
        for other,hs in sets.items():
            if other!=split:assert not hashes&hs,(f,other)
        sets.setdefault(split,set()).update(hashes)
        count=min(20,g);x=torch.from_numpy(xx[:count]);fs=[]
        for state in range(ns):
            forest=Forest(x)
            for c in range(1,int(d['k'])+1):forest.add(torch.full((count,),int(h[c])),torch.from_numpy(tt[:count,state,c]))
            fs.append(forest)
        assert torch.equal(fs[0].starts[:,0],fs[1].starts[:,0])
        assert torch.equal(fs[0].masks()[0][:,0],fs[1].masks()[0][:,0])
        expected=torch.ones(count,n,dtype=torch.bool);expected[:,h[1:]]=False
        assert torch.equal(fs[0].masks()[0][:,0],expected)
        ff=features({**d,'x':xx[:count],'tails':tt[:count]},False).reshape(count,ns,n,8)
        assert np.array_equal(ff[:,0],ff[:,1])
        for row in range(min(5,g)):
            for state in range(ns):
                qq=dp_q(xx[row].astype(float),h,tt[row,state]);err=float(np.max(np.abs(qq-d['q'][row,state])));assert err<1e-10,err;maxerr=max(maxerr,err);total+=1
        checks.append(dict(file=f.name,n_geometry=g,endpoint_mask_checked=count,dp_forests_checked=min(5,g)*ns))
    result=dict(status='pass',geometry_disjoint_by_split=True,unique_geometry_by_split={k:len(v) for k,v in sets.items()},independent_held_karp_forests_checked=total,max_absolute_label_error=maxerr,source_context_and_mask_equal=True,blind_features_identical=True,files=checks)
    replays=[]
    iid=dict(np.load(out/'data/iid_n9_s190919301.npz'));small={**iid,'x':iid['x'][:16],'tails':iid['tails'][:16]}
    torch.set_num_threads(2)
    for checkpoint in sorted(out.glob('*.pt')):
        if not checkpoint.stem.startswith(('blind_','aware_')):continue
        predfile=out/f'iid_n9_{checkpoint.stem}.npz'
        if not predfile.exists():continue
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);model=Model(ck['args']['width'],ck['args']['layers']);model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():pp=model(torch.tensor(features(small,ck['aware'])))[:,iid['heads'][1:]].numpy().reshape(16,2,-1)
        saved=np.load(predfile)['pred'][:16];error=float(np.abs(pp-saved).max());mismatches=int(np.count_nonzero(pp.argmin(-1)!=saved.argmin(-1)));assert mismatches==0,mismatches
        replays.append(dict(checkpoint=checkpoint.name,cpu_replay_prediction_max_abs_error=error,selected_action_mismatches=mismatches,n_forests=32))
    result['checkpoint_cpu_replays']=replays
    result['cross_runtime_note']='CPU replay uses local PyTorch 2.12.1; stored inference uses CUDA PyTorch 2.4.0. Logits can differ numerically; action agreement, not bitwise cross-platform equality, is checked.'
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
