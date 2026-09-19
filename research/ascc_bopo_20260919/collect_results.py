# Run remotely; stream small result/provenance files, never full model checkpoints.
from pathlib import Path
import sys,tarfile
root=Path('/workspace/计算群论/results/ascc-bopo-joint-20260919')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for p in root.rglob('*'):
  if not p.is_file():continue
  if p.suffix in ['.json','.jsonl','.log','.md'] or p.name.endswith(('costs.pt','instances.pt')):
   t.add(p,arcname=str(p.relative_to(root)))
