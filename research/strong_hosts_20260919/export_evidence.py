"""Export small, reproducible study evidence without full model weights."""
import argparse,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
with tarfile.open(a.root/'evidence.tar.gz','w:gz') as tar:
    for f in sorted(a.root.rglob('*')):
        relative=f.relative_to(a.root)
        if not f.is_file() or any(p.startswith('smoke-') for p in relative.parts):continue
        if f.suffix in ('.json','.png','.md','.log') or f.name.endswith('-costs.pt') or f.name=='datasets.pt':
            tar.add(f,arcname=str(relative))
print(a.root/'evidence.tar.gz')
