"""Replay frozen conditional evaluation to retain independently checkable vectors."""
import argparse, json, time
from pathlib import Path
import torch
from component_adapter import Model as FrozenModel
from secondary_component import Model as SecondaryModel


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--host',choices=['icam','am'],required=True)
    p.add_argument('--decoder',action='store_true');p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.wait:
        for _ in range(360):
            if (a.out/'aggregate.json').exists():break
            time.sleep(10)
    assert (a.out/'aggregate.json').exists()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sets=torch.load(a.out/'final_test_data.pt',weights_only=False)
    m=(SecondaryModel if a.decoder else FrozenModel)(kind=a.host).cuda().eval();records=[]
    for folder in sorted(a.out.glob('*-seed*')):
        ck=torch.load(folder/'best.pt',weights_only=False);m.aware=ck['aware']
        if a.decoder:m.restore(ck)
        else:m.residual.load_state_dict(ck['residual'])
        expected=json.loads((folder/'test_summary.json').read_text())['test']
        for name,s in sets.items():
            g=torch.Generator(device='cuda').manual_seed(9417320);losses=[];accuracies=[]
            for fraction in [.25,.5,.75]:
                ll=[];aa=[]
                for i in range(0,len(s['x']),32):
                    loss,acc=m.conditional(s['enc'][i:i+32].cuda(),s['x'][i:i+32].cuda(),s['pi'][i:i+32].cuda(),
                                          int(s['x'].size(1)*fraction),ck['mode'],g)
                    ll.append(loss.cpu());aa.append(acc.cpu())
                losses.append(torch.cat(ll));accuracies.append(torch.cat(aa))
            loss=torch.stack(losses);acc=torch.stack(accuracies)
            nll_error=abs(float(loss.double().mean())-expected[name]['conditional']['nll'])
            accuracy_error=abs(float(acc.float().mean())-expected[name]['conditional']['accuracy'])
            assert nll_error<1e-7 and accuracy_error<1e-7,(folder.name,name,nll_error,accuracy_error)
            torch.save(dict(nll=loss,correct=acc,depths=[int(s['x'].size(1)*f) for f in [.25,.5,.75]],
                            generator_seed=9417320,mode=ck['mode']),folder/f'{name}-conditional.pt')
            records.append(dict(arm=folder.name,dataset=name,nll_error=nll_error,accuracy_error=accuracy_error))
    (a.out/'conditional_replay_verification.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
