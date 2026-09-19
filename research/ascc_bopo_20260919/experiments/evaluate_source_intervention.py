"""Post-hoc source intervention; this is not a separately trained ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from groupopt.models.ascc_bopo import SourceFirstASCC


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    path = args.run / 'best.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = SourceFirstASCC(**payload['config']['dimensions']).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    coords = torch.load(args.run / 'evaluation_instances.pt', weights_only=True)['test']
    summary = {'role': 'post_hoc_policy_intervention_not_retrained_ablation',
               'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'test_sha256': hashlib.sha256(coords.numpy().tobytes()).hexdigest(),
               'rollouts': 8, 'count': len(coords), 'results': []}
    costs = {}
    # Each random repeat has exactly the same best-of-8 budget; never pool them.
    settings = [('learned', 0), ('route', 0), ('fixed', 0), ('shortest', 0)]
    settings += [('random', seed) for seed in (7001, 7002, 7003)]
    for policy, seed in settings:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        torch.cuda.synchronize()
        begin = time.monotonic()
        pieces = []
        with torch.no_grad():
            for batch in coords.split(8):
                result = model(batch.cuda(), 8, policy, 'greedy',
                               generator=generator, anchor_mode='multi')
                pieces.append(result.costs.amin(1).cpu())
        torch.cuda.synchronize()
        value = torch.cat(pieces).double()
        tag = f'{policy}-{seed}'
        costs[tag] = value
        torch.save(value, args.out / f'{tag}_costs.pt')
        row = {'policy': policy, 'action_seed': seed, 'mean_cost': float(value.mean()),
               'seconds': time.monotonic() - begin}
        summary['results'].append(row)
        print(json.dumps(row), flush=True)
    learned = costs['learned-0']
    for row in summary['results']:
        value = costs[f"{row['policy']}-{row['action_seed']}"]
        delta = value - learned
        mean = float(delta.mean())
        se = float(delta.std(unbiased=True) / len(delta)**.5)
        row.update(cost_minus_learned=mean,
                   conditional_normal_ci95=[mean - 1.96 * se, mean + 1.96 * se])
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
