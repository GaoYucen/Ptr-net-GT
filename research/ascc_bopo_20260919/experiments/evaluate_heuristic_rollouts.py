"""End-to-end tour test for source heuristics discovered by regret diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from groupopt.models.ascc_bopo import SourceFirstASCC


def tensor_sha(value):
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--sizes', type=int, nargs='+', default=[20, 50])
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--rollouts', type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    path = args.screen / 'learned-reinforce-seed1234' / 'best.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = SourceFirstASCC(**payload['config']['dimensions']).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    settings = [(p, 0) for p in ('learned', 'route', 'min_entropy', 'max_margin',
                                  'shortest_edge', 'shortest', 'fixed')]
    settings += [('random', seed) for seed in (7001, 7002, 7003)]
    result = {'role': 'end_to_end_source_heuristic_diagnostic',
              'checkpoint': 'learned-reinforce-seed1234/best.pt',
              'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'count': args.count, 'rollouts': args.rollouts,
              'decode': 'best of multi-anchor greedy endpoint rollouts',
              'augmentation': False, 'local_search': False, 'sizes': []}
    for n in args.sizes:
        coords = torch.rand(args.count, n, 2,
                            generator=torch.Generator().manual_seed(2026093000 + n))
        size_result = {'n': n, 'seed': 2026093000 + n,
                       'coordinate_sha256': tensor_sha(coords), 'policies': {}}
        costs = {}
        for policy, seed in settings:
            generator = torch.Generator(device='cuda').manual_seed(seed)
            pieces = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); begin = time.monotonic()
            with torch.no_grad():
                for batch in coords.split(args.batch):
                    rollout = model(batch.cuda(), args.rollouts, policy, 'greedy',
                                    generator=generator, anchor_mode='multi')
                    pieces.append(rollout.costs.amin(1).cpu())
            torch.cuda.synchronize()
            values = torch.cat(pieces).double()
            tag = f'{policy}-{seed}'
            costs[tag] = values
            torch.save(values, args.out / f'tsp{n}-{tag}-costs.pt')
            size_result['policies'][tag] = {
                'mean_cost': float(values.mean()), 'seconds': time.monotonic() - begin,
                'peak_memory_gb': torch.cuda.max_memory_allocated() / 2 ** 30}
            print(json.dumps({'n': n, 'policy': tag} |
                             size_result['policies'][tag]), flush=True)
        baseline = costs['route-0']
        for tag, values in costs.items():
            delta = baseline - values
            mean, se = float(delta.mean()), float(delta.std(unbiased=True) / len(delta) ** .5)
            size_result['policies'][tag].update(
                route_minus_policy=mean,
                relative_improvement_percent=100 * mean / float(baseline.mean()),
                conditional_normal_ci95=[mean - 1.96 * se, mean + 1.96 * se])
        result['sizes'].append(size_result)
        temporary = args.out / 'summary.tmp'
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(args.out / 'summary.json')


if __name__ == '__main__':
    main()
