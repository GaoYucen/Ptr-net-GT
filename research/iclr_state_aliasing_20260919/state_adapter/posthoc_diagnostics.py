"""Frozen-checkpoint interventions and end-to-end timing; never model selection."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from component_adapter import Model as FrozenModel, teacher_state
from secondary_component import Model as SecondaryModel
from strong_adapter import tour_cost
from verify_strong import valid


@torch.no_grad()
def sensitivity(out,host,decoder=False):
    data=torch.load(out/'final_test_data.pt',weights_only=False); m=(SecondaryModel if decoder else FrozenModel)(kind=host).cuda().eval()
    records=[]
    for folder in sorted(out.glob('random-aware-seed*')):
        ck=torch.load(folder/'best.pt',weights_only=False); m.restore(ck) if decoder else m.residual.load_state_dict(ck['residual'])
        for name,s in data.items():
            g=torch.Generator(device='cuda').manual_seed(752691)
            original=[]; corrupted=[]; changes=[]
            for fraction in [.25,.5,.75]:
                for i in range(0,len(s['x']),32):
                    x=s['x'][i:i+32].cuda(); enc=s['enc'][i:i+32].cuda(); pi=s['pi'][i:i+32].cuda()
                    st,tail,target=teacher_state(enc,pi,int(x.size(1)*fraction),'random',g)
                    lp=m.logits(enc,x,st,tail).log_softmax(-1)
                    corrupt=copy.copy(st); corrupt.end=st.end.clone()
                    # Permute only target-tail associations among legal heads.
                    # Masks, source path, mean embeddings and sizes stay fixed.
                    # This is a representation sensitivity intervention; it is
                    # deliberately not a second valid forest or causal proof.
                    mask=st.mask(tail)
                    for row in range(len(x)):
                        heads=(~mask[row]).nonzero().flatten()
                        perm=torch.randperm(len(heads),generator=g,device='cuda')
                        corrupt.end[row,heads]=st.end[row,heads[perm]]
                    lp2=m.logits(enc,x,corrupt,tail).log_softmax(-1)
                    rows=torch.arange(len(x),device='cuda')
                    original.append(-lp[rows,target].cpu()); corrupted.append(-lp2[rows,target].cpu())
                    changes.append(lp.argmax(1).ne(lp2.argmax(1)).cpu())
            o=torch.cat(original).double(); c=torch.cat(corrupted).double()
            records.append(dict(seed=ck['seed'],dataset=name,correct_nll=float(o.mean()),
                                shuffled_tail_nll=float(c.mean()),nll_increase=float((c-o).mean()),
                                argmax_change_fraction=float(torch.cat(changes).float().mean()),
                                interpretation='Post-selection test-time sensitivity; other component features remain correct.'))
    (out/'pairing_sensitivity.json').write_text(json.dumps(records,indent=2))


@torch.no_grad()
def benchmark(out,host,decoder=False):
    data=torch.load(out/'final_test_data.pt',weights_only=False); m=(SecondaryModel if decoder else FrozenModel)(kind=host).cuda().eval(); records=[]
    native_model=FrozenModel(kind=host).cuda().eval()
    def timing(fn):
        fn(); values=[]
        for _ in range(5):
            torch.cuda.synchronize(); begin=time.perf_counter(); fn(); torch.cuda.synchronize()
            values.append(time.perf_counter()-begin)
        return dict(median_seconds=float(np.median(values)),repetitions_seconds=values)
    for name in ['uniform100','uniform200']:
        x=data[name]['x'][:32].cuda()
        enc=m.host.encode(x)
        records.append(dict(dataset=name,method='native',batch=32,end_to_end=timing(lambda:tour_cost(x,native_model.host.native(x))),
                            encoding=timing(lambda:native_model.host.encode(x))))
        for mode in ['route','random']:
            for aware in [False,True]:
                folder=out/f'{mode}-{"aware" if aware else "blind"}-seed12031'
                ck=torch.load(folder/'best.pt',weights_only=False); m.aware=aware; m.restore(ck) if decoder else m.residual.load_state_dict(ck['residual'])
                g=torch.Generator(device='cuda').manual_seed(416319)
                _,succ=m.rollout(x,enc,mode,g); valid(succ)
                records.append(dict(dataset=name,method=f'{mode}-{"aware" if aware else "blind"}',batch=32,seed=12031,
                    checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),
                    end_to_end=timing(lambda:m.rollout(x,None,mode,g)),
                    cached_encoder=timing(lambda:m.rollout(x,enc,mode,g))))
    (out/'latency_benchmark.json').write_text(json.dumps(dict(records=records,
        scope='Five repeats, one warmup, no tour validation inside timed region. Includes cost computation. No claim of exclusive hardware tenancy.'),indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True);p.add_argument('--host',choices=['icam','am'],required=True)
    p.add_argument('--decoder',action='store_true'); a=p.parse_args(); torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    assert (a.out/'aggregate.json').exists(),'Only run after frozen final-test reporting'
    sensitivity(a.out,a.host,a.decoder); benchmark(a.out,a.host,a.decoder)

if __name__=='__main__': main()
