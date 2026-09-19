"""Small, prespecified size/distribution diagnostic; never used to pick checkpoints.

These are development diagnostics, not new held-out paper evidence. No OPT labels
are assumed. Checkpoints must already have been selected on TSP100 validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from groupopt.models.ascc_bopo import SourceFirstASCC


def dataset(n, count, distribution, seed):
    g = torch.Generator().manual_seed(seed)
    if distribution == 'uniform':
        return torch.rand(count, n, 2, generator=g)
    centers = 0.15 + 0.7 * torch.rand(count, 5, 2, generator=g)
    labels = torch.randint(0, 5, (count, n), generator=g)
    noise = 0.04 * torch.randn(count, n, 2, generator=g)
    return (centers.gather(1, labels[..., None].expand(-1, -1, 2)) + noise).clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--rollouts', type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    results = {'role': 'one_seed_size_distribution_development_diagnostic', 'cells': [],
               'no_opt_labels': True, 'checkpoint_selection': 'TSP100 validation only',
               'count_per_cell': args.count, 'rollouts': args.rollouts,
               'augmentation': False, 'postprocessing': False}
    for n in (200, 500, 1000):
        for distribution in ('uniform', 'clustered'):
            seed = 2026091910 + n + (1 if distribution == 'clustered' else 0)
            coords = dataset(n, args.count, distribution, seed)
            tag = f'{distribution}-{n}'
            torch.save(coords, args.out / f'{tag}-instances.pt')
            cell = {'nodes': n, 'distribution': distribution, 'seed': seed,
                    'coordinate_sha256': hashlib.sha256(coords.numpy().tobytes()).hexdigest(),
                    'models': {}}
            costs = {}
            for source in ('route', 'learned'):
                path = args.screen / f'{source}-bopo-seed1234' / 'best.pt'
                payload = torch.load(path, map_location='cpu', weights_only=False)
                model = SourceFirstASCC(**payload['config']['dimensions']).cuda().eval()
                model.load_state_dict(payload['model'], strict=True)
                pieces = []
                torch.cuda.synchronize()
                begin = time.monotonic()
                with torch.no_grad():
                    for batch in coords.split(args.batch):
                        result = model(batch.cuda(), args.rollouts, source, 'greedy',
                                       anchor_mode='multi')
                        pieces.append(result.costs.amin(1).cpu())
                torch.cuda.synchronize()
                costs[source] = torch.cat(pieces).double()
                torch.save(costs[source], args.out / f'{tag}-{source}-costs.pt')
                cell['models'][source] = {'mean_cost': float(costs[source].mean()),
                    'seconds': time.monotonic()-begin, 'checkpoint_step': payload['step'],
                    'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                del result, model, payload
                torch.cuda.empty_cache()
            delta = costs['route'] - costs['learned']
            se = float(delta.std(unbiased=True) / len(delta)**.5)
            mean = float(delta.mean())
            cell.update(route_minus_ascc=mean, relative_improvement_percent=
                        100*mean/float(costs['route'].mean()),
                        conditional_normal_ci95=[mean-1.96*se, mean+1.96*se])
            results['cells'].append(cell)
            tmp = args.out / 'summary.tmp'
            tmp.write_text(json.dumps(results, indent=2, allow_nan=False))
            tmp.replace(args.out / 'summary.json')
            print(json.dumps(cell), flush=True)


if __name__ == '__main__':
    main()
