"""Export compact test provenance, excluding the bulky cached encodings."""
import argparse
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args()
for name in ['results','results_am','results_decoder_icam','results_decoder_am']:
    out=a.root/name
    if (out/'aggregate.json').exists():
        data=torch.load(out/'final_test_data.pt',weights_only=False)
        torch.save({case:{k:v for k,v in item.items() if k!='enc'} for case,item in data.items()},out/'final_test_audit.pt')
        print(name,flush=True)
