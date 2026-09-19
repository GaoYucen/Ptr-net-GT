"""Train a small source-value selector from actual downstream rollout costs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from evaluate_rollout_value_oracle import (
    SourceFirstASCC, apply_edge, clone_state, continue_route, route_prefix,
)
from evaluate_source_regret_oracle import score_state


FEATURES = [
    'source_score', 'endpoint_entropy', 'endpoint_margin', 'greedy_edge',
    'source_size', 'target_size', 'source_chord', 'target_chord',
    'tail_to_centroid', 'head_to_centroid', 'edge_over_min', 'is_route_tail',
    'source_start_to_head', 'tail_to_target_end',
]


def candidate_features(coords, state, rows):
    center = coords.mean(0)
    minimum_edge = max(min(r['edge'] for r in rows), 1e-8)
    result = []
    for row in rows:
        tail, head = row['tail'], row['head']
        source_start = int(state['starts'][tail])
        target_end = int(state['ends'][head])
        result.append([
            row['source_score'], row['entropy'], row['margin'], row['edge'],
            float(state['sizes'][tail]) / len(coords),
            float(state['sizes'][head]) / len(coords),
            float(torch.dist(coords[source_start], coords[tail])),
            float(torch.dist(coords[head], coords[target_end])),
            float(torch.dist(coords[tail], center)),
            float(torch.dist(coords[head], center)),
            row['edge'] / minimum_edge,
            float(tail == state['route_tail']),
            float(torch.dist(coords[source_start], coords[head])),
            float(torch.dist(coords[tail], coords[target_end])),
        ])
    return torch.tensor(result, dtype=torch.float32)


@torch.no_grad()
def make_split(model, count, seed, n, stage):
    coords_all = torch.rand(count, n, 2, generator=torch.Generator().manual_seed(seed))
    feature_rows, cost_rows, route_rows = [], [], []
    begin = time.monotonic()
    for index, coords in enumerate(coords_all):
        encoded = model.encoder(coords[None].cuda()).squeeze(0)
        state = route_prefix(model, coords, encoded, stage)
        fitted = score_state(model, coords.cuda(), encoded, state)
        features = candidate_features(coords, state, fitted)
        costs = []
        for row in fitted:
            branch = clone_state(state)
            apply_edge(coords, branch, row['tail'], row['head'])
            continue_route(model, coords, encoded, branch)
            costs.append(branch['cost'])
        route_branch = clone_state(state)
        continue_route(model, coords, encoded, route_branch)
        feature_rows.append(features)
        cost_rows.append(torch.tensor(costs, dtype=torch.float32))
        route_rows.append(route_branch['cost'])
        if (index + 1) % 8 == 0:
            print(json.dumps({'seed': seed, 'finished': index + 1,
                              'seconds': time.monotonic() - begin}), flush=True)
    return {'features': torch.stack(feature_rows), 'costs': torch.stack(cost_rows),
            'route': torch.tensor(route_rows, dtype=torch.float32), 'seed': seed}


class Scorer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = (nn.Linear(len(FEATURES), 1) if width == 0 else nn.Sequential(
            nn.Linear(len(FEATURES), width), nn.SiLU(), nn.Linear(width, width),
            nn.SiLU(), nn.Linear(width, 1)))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def evaluate(model, split, mean, scale, objective):
    with torch.no_grad():
        prediction = model((split['features'] - mean) / scale)
        selected = prediction.argmax(-1) if objective == 'classification' else prediction.argmin(-1)
        values = split['costs'].gather(1, selected[:, None]).squeeze(1)
    return values


def fit(train, validation, width, objective, seed, mean, scale):
    torch.manual_seed(seed)
    model = Scorer(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed + 9000)
    best = None
    for epoch in range(300):
        order = torch.randperm(len(train['features']), generator=generator)
        for indices in order.split(16):
            feature = (train['features'][indices] - mean) / scale
            prediction = model(feature)
            costs = train['costs'][indices]
            if objective == 'classification':
                optimum = costs.amin(-1, keepdim=True)
                target = (costs <= optimum + 1e-7).float()
                target /= target.sum(-1, keepdim=True)
                loss = -(target * prediction.log_softmax(-1)).sum(-1).mean()
            else:
                target = 100 * (costs - costs.amin(-1, keepdim=True)) / costs.amin(-1, keepdim=True)
                loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 10 == 0 or epoch == 299:
            values = evaluate(model, validation, mean, scale, objective)
            score = float(values.mean())
            if best is None or score < best[0]:
                best = (score, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model, best[0], best[1]


def controls(split):
    f, q = split['features'], split['costs']
    return {
        'route': split['route'], 'oracle': q.amin(-1), 'random_expected': q.mean(-1),
        'pretrained_learned': q.gather(1, f[..., 0].argmax(-1)[:, None]).squeeze(1),
        'min_entropy': q.gather(1, f[..., 1].argmin(-1)[:, None]).squeeze(1),
        'max_margin': q.gather(1, f[..., 2].argmax(-1)[:, None]).squeeze(1),
        'shortest_edge': q.gather(1, f[..., 3].argmin(-1)[:, None]).squeeze(1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--screen', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-count', type=int, default=80)
    p.add_argument('--validation-count', type=int, default=24)
    p.add_argument('--test-count', type=int, default=48)
    p.add_argument('--nodes', type=int, default=50)
    p.add_argument('--stage', type=int, default=30)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    checkpoint = args.screen / 'learned-reinforce-seed1234' / 'best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    backbone = SourceFirstASCC(**payload['config']['dimensions']).cuda().eval()
    backbone.load_state_dict(payload['model'], strict=True)
    splits = {
        'train': make_split(backbone, args.train_count, 2026100101, args.nodes, args.stage),
        'validation': make_split(backbone, args.validation_count, 2026100102, args.nodes, args.stage),
        'test': make_split(backbone, args.test_count, 2026100103, args.nodes, args.stage),
    }
    torch.save(splits, args.out / 'candidate_dataset.pt')
    mean = splits['train']['features'].mean((0, 1), keepdim=True)
    scale = splits['train']['features'].std((0, 1), keepdim=True).clamp_min(1e-6)
    candidates = []
    for width in (0, 32, 64):
        for objective in ('classification', 'regression'):
            for seed in (11, 22, 33):
                model, validation_cost, epoch = fit(
                    splits['train'], splits['validation'], width, objective, seed, mean, scale)
                candidates.append((validation_cost, width, objective, seed, epoch, model))
    chosen = min(candidates, key=lambda item: item[0])
    _, width, objective, seed, epoch, selector = chosen
    test_values = evaluate(selector, splits['test'], mean, scale, objective)
    methods = controls(splits['test']) | {'selected_value_model': test_values}
    route = methods['route']
    results = []
    for name, values in methods.items():
        delta = route - values
        mean_delta = float(delta.mean())
        se = float(delta.std(unbiased=True) / len(delta) ** .5)
        results.append({'method': name, 'mean_cost': float(values.mean()),
                        'relative_improvement_percent': 100 * mean_delta / float(route.mean()),
                        'conditional_normal_ci95': [mean_delta - 1.96 * se,
                                                    mean_delta + 1.96 * se],
                        'improved_fraction': float((delta > 1e-7).float().mean())})
    result = {'role': 'held_out_rollout_value_selector_pilot', 'features': FEATURES,
              'nodes': args.nodes, 'stage': args.stage,
              'counts': {k: len(v['features']) for k, v in splits.items()},
              'seeds': {k: v['seed'] for k, v in splits.items()},
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'selected_on_validation': {'width': width, 'objective': objective,
                                         'seed': seed, 'epoch': epoch,
                                         'validation_cost': chosen[0]},
              'candidate_configurations': len(candidates), 'test_results': results}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    torch.save({'model': selector.state_dict(), 'mean': mean, 'scale': scale,
                'config': result['selected_on_validation'], 'features': FEATURES},
               args.out / 'selector.pt')


if __name__ == '__main__':
    main()
