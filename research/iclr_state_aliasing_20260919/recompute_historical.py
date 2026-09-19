"""Recompute archived strong-host results from per-instance cost tensors.

No historical training is rerun. Bootstrap units are test geometries, averaged
over the three joint-training seeds that share the original host warmup.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / 'strong_hosts_20260919/evidence'
OUT = ROOT / 'historical_recomputed'
OUT.mkdir(exist_ok=True)
SEEDS = [1234, 2468, 4321]
CASES = ['test200', 'test200-aug8', 'cluster200', 'test500']
rows = []
manifest = {}
rng = np.random.default_rng(91975001)


def read_cost(path):
    manifest[str(path.relative_to(ROOT.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert isinstance(payload, torch.Tensor) and payload.ndim == 1
    a = payload.double().numpy()
    assert np.isfinite(a).all() and (a > 0).all()
    return a


for host in ['am', 'icam']:
    for case in CASES:
        costs = {mode: np.stack([read_cost(EVIDENCE / host / f'{mode}-seed{seed}' / f'{case}-costs.pt')
                                for seed in SEEDS]) for mode in ['route', 'random', 'learned']}
        native = read_cost(EVIDENCE / host / f'native-{case}-costs.pt')
        ref = costs['route']
        idx = rng.integers(ref.shape[1], size=(10000, ref.shape[1]))
        r = dict(host=host, case=case, test_instances=ref.shape[1], training_seeds=SEEDS,
                 historical=True, shared_warmup=True, native_mean=float(native.mean()),
                 means={mode: float(v.mean()) for mode, v in costs.items()},
                 per_seed_means={mode: v.mean(1).tolist() for mode, v in costs.items()},
                 comparisons={})
        for mode in ['random', 'learned']:
            improvement = ref.mean(0) - costs[mode].mean(0)
            gains = 100 * improvement[idx].mean(1) / ref.mean(0)[idx].mean(1)
            r['comparisons'][mode] = dict(improvement_percent=float(100 * improvement.mean() / ref.mean()),
                                          paired_instance_ci=np.quantile(gains, [.025, .975]).tolist(),
                                          per_seed_improvement_percent=(100 * (ref.mean(1) - costs[mode].mean(1)) / ref.mean(1)).tolist())
        rows.append(r)
summary = dict(source='Archived official AM/ICAM pilot; no new training', rows=rows,
               bootstrap_samples=10000, bootstrap_seed=91975001,
               input_sha256=manifest,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
for r in rows:
    print(r['host'], r['case'], r['means'], r['comparisons']['learned']['improvement_percent'])
