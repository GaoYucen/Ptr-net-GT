"""CPU provenance and completed-output consistency checks."""
import argparse,hashlib,json,math
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);entries=[];hashes=[]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for host in ('am','icam'):
    root=a.root/host;c=json.loads((root/'config.json').read_text());d=torch.load(root/'datasets.pt',weights_only=True)
    h={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in d.items()}
    assert h==c['data_hash'];hashes.append(h)
    for name in ('strong_adapter.py','run_strong.py','verify_strong.py'):
        assert sha(Path(__file__).parent/name)==c['source_hash'][name],('training code changed',name)
    for seed in (1234,4321,2468):
        for mode in ('route','random','learned'):
            f=root/f'{mode}-seed{seed}';s=json.loads((f/'summary.json').read_text());hist=json.loads((f/'history.json').read_text())
            best_record=min(hist,key=lambda v:v['validation']['mean'])
            best=torch.load(f/'best.pt',weights_only=False,map_location='cpu')
            final=torch.load(f/'final.pt',weights_only=False,map_location='cpu')
            assert best['best_step']==s['best_step']==best_record['step']
            assert best['mode']==mode and best['seed']==seed and final['step']==300
            for case,q in s['evaluation'].items():
                cost=torch.load(f/f'{case}-costs.pt',weights_only=True)
                assert torch.isfinite(cost).all()
                assert len(cost)==(256 if case.startswith('test200') else 64)
                assert abs(float(cost.double().mean())-q['mean'])<1e-10
            gradients=[v['gradient_norm'].get('selector',0) for v in hist if v['step']]
            assert all(math.isfinite(v) for v in gradients)
            assert (max(gradients)>0) if mode=='learned' else (max(gradients)==0)
            distance={prefix:sum(float((best['model'][k].double()-v.double()).square().sum())
                       for k,v in final['model'].items() if k.startswith(prefix))**.5
                      for prefix in ('host','selector')}
            entries.append(dict(host=host,mode=mode,seed=seed,best_step=s['best_step'],
                best_checkpoint_sha256=sha(f/'best.pt'),final_checkpoint_sha256=sha(f/'final.pt'),
                best_final_parameter_distance=distance,selector_gradient_norms=gradients,
                validation_source_deviation_initial=hist[0]['validation']['source_deviation'],
                validation_source_deviation_best=best_record['validation']['source_deviation'],
                evaluation_means_verified=True))
assert hashes[0]==hashes[1]
(a.root/'evidence-integrity.json').write_text(json.dumps(dict(training_code_matches_recorded_hashes=True,
    paired_host_datasets_identical=True,all_18_arms_verified=True,entries=entries),indent=2))
print('All 18 arms: code/data hashes, checkpoint selection, source gradients, and saved cost means verified.')
