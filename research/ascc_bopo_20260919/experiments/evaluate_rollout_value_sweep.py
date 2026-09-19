"""Sweep the stage of one perfect source/endpoint intervention."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import torch

from evaluate_rollout_value_oracle import (
    SourceFirstASCC, best_endpoint_branch, best_source_branch, continue_route,
    initial_state, route_prefix,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--count', type=int, default=32)
    p.add_argument('--size', type=int, default=50)
    p.add_argument('--stages', type=int, nargs='+', default=[4, 6, 8, 10, 15, 20, 30])
    args = p.parse_args()
    if max(args.stages) >= args.size:
        p.error('stages must be smaller than problem size to exclude pure start-node selection')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checkpoint = args.screen / 'learned-reinforce-seed1234' / 'best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SourceFirstASCC(**payload['config']['dimensions']).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    coords_all = torch.rand(args.count, args.size, 2,
                            generator=torch.Generator().manual_seed(2026093000 + args.size))
    rows, begin = [], time.monotonic()
    for index, coords in enumerate(coords_all):
        encoded = model.encoder(coords[None].cuda()).squeeze(0)
        route = continue_route(model, coords, encoded, initial_state(args.size))['cost']
        for stage in args.stages:
            prefix = route_prefix(model, coords, encoded, stage)
            source, _, _ = best_source_branch(model, coords, encoded, prefix)
            endpoint, _, _ = best_endpoint_branch(model, coords, encoded, prefix)
            rows.extend([dict(instance=index, stage=stage, method='route', cost=route),
                         dict(instance=index, stage=stage, method='source_once', cost=source),
                         dict(instance=index, stage=stage, method='endpoint_once', cost=endpoint)])
        if (index + 1) % 4 == 0:
            print(json.dumps({'finished': index + 1,
                              'seconds': time.monotonic() - begin}), flush=True)
    with (args.out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output = {'role': 'one_shot_rollout_value_stage_sweep', 'n': args.size,
              'count': args.count, 'stages': args.stages,
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'results': []}
    for stage in args.stages:
        route = torch.tensor([r['cost'] for r in rows if r['stage'] == stage and r['method'] == 'route'])
        for method in ('source_once', 'endpoint_once'):
            value = torch.tensor([r['cost'] for r in rows if r['stage'] == stage and r['method'] == method])
            delta = route - value
            mean, se = float(delta.mean()), float(delta.std(unbiased=True) / len(delta) ** .5)
            output['results'].append(dict(stage=stage, method=method, mean_cost=float(value.mean()),
                relative_improvement_percent=100 * mean / float(route.mean()),
                conditional_normal_ci95=[mean - 1.96 * se, mean + 1.96 * se]))
    (args.out / 'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
