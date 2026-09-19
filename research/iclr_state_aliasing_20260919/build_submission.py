"""Create an explicit, inspectable supplemental archive without runtime caches.

Run after all experiment suites and the paper are finalized. This does not
upload or submit anything. The review copy remains in the workspace.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

R = Path(__file__).resolve().parent
WORK = R.parent
OUT = R / 'output'
STAGE = OUT / 'supplement'
STAGE.mkdir(parents=True, exist_ok=True)
manifest = {}


def copy(source, target):
    target = STAGE / target
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    target.write_bytes(data)
    manifest[str(target.relative_to(STAGE))] = hashlib.sha256(data).hexdigest()


root_files = ['README.md', 'PROTOCOL.md', 'requirements.txt', 'references.bib',
              'counterfactual_stress.py', 'recompute_historical.py',
              'build_paper_assets.py', 'build_new_results.py',
              'verify_witness_interval.py', 'verify_artifacts.py',
              'witness_interval_certificate.json', 'artifact_verification.json']
for name in root_files:
    p = R / name
    if p.exists():
        copy(p, Path('iclr_20260919') / name)

for folder in ['counterfactual_stress', 'historical_recomputed', 'exact_diagnostic', 'state_adapter', 'paper']:
    for p in sorted((R / folder).rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        if p.suffix not in {'.py', '.json', '.npz', '.pt', '.md', '.tex', '.bib', '.bst', '.sty', '.pdf', '.png'}:
            continue
        # Large cached embeddings and redundant final checkpoints are regenerable;
        # selected checkpoints, raw test costs, exact labels and predictions stay.
        if p.name in {'data.pt', 'final_test_data.pt', 'final.pt', 'manifest_sha256.json'}:
            continue
        if p.name == 'run_manifest.json':
            continue  # execution hosts/SSH bookkeeping are not research inputs
        copy(p, Path('iclr_20260919') / p.relative_to(R))

historical = WORK / 'strong_hosts_20260919/evidence'
for p in sorted(historical.rglob('*')):
    if p.is_file() and (p.name.endswith('-costs.pt') or p.name == 'endpoint-aliasing.json'):
        copy(p, p.relative_to(WORK))

suspects = []
for rel in manifest:
    p = STAGE / rel
    if p.suffix in {'.py', '.json', '.md', '.tex', '.bib', '.sh'}:
        text = p.read_text(errors='replace')
        for pattern in [r'/Users/[^/\s]+', r'219\.216\.\d+\.\d+',
                        r'BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY', r'(?i)api[_-]?key\s*[:=]\s*[\"\'][^\"\']{12,}']:
            match = re.search(pattern, text)
            if match:
                suspects.append({'file': rel, 'pattern': pattern})
scan = dict(files=len(manifest), checked_for='local user paths, known host addresses, private keys and key-like fields',
            matches=suspects, limit='This is an automated privacy scan, not a guarantee of double-blind compliance.')
(OUT / 'supplement_privacy_scan.json').write_text(json.dumps(scan, indent=2))
if suspects:
    raise SystemExit('Review potentially identifying entries in supplement_privacy_scan.json before release.')
(STAGE / 'MANIFEST.sha256.json').write_text(json.dumps(manifest, indent=2))
zip_path = OUT / 'iclr_research_supplement.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(STAGE.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(STAGE))
print(json.dumps({'archive': str(zip_path), 'bytes': zip_path.stat().st_size,
                  'sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest(), **scan}, indent=2))
