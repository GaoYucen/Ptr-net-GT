"""Validation-selected conservative gate for a fitted rollout-value selector."""
import argparse
import json
import math
from pathlib import Path

import torch

from train_rollout_value_selector import Scorer


def predictions(model, split, mean, scale, objective):
    with torch.no_grad():
        raw = model((split['features'] - mean) / scale)
    best = raw.argmax(-1) if objective == 'classification' else raw.argmin(-1)
    route = split['features'][..., 11].argmax(-1)
    row = torch.arange(len(raw))
    margin = (raw[row, best] - raw[row, route] if objective == 'classification'
              else raw[row, route] - raw[row, best])
    return raw, best, route, margin


def selected_values(split, best, route, margin, threshold):
    selected = torch.where(margin > threshold, best, route)
    row = torch.arange(len(selected))
    return split['costs'][row, selected], selected != route


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    data = torch.load(args.run / 'candidate_dataset.pt', weights_only=False)
    payload = torch.load(args.run / 'selector.pt', weights_only=True)
    config = payload['config']
    model = Scorer(config['width'])
    model.load_state_dict(payload['model'])
    objective = config['objective']
    _, vb, vr, vm = predictions(model, data['validation'], payload['mean'],
                                payload['scale'], objective)
    quantiles = torch.linspace(0, 1, 101)
    candidates = torch.unique(torch.cat((torch.tensor([-torch.inf, torch.inf]),
                                         torch.quantile(vm, quantiles)))).tolist()
    scored = []
    for threshold in candidates:
        values, deviations = selected_values(data['validation'], vb, vr, vm, threshold)
        scored.append((float(values.mean()), -float(threshold), float(threshold),
                       float(deviations.float().mean())))
    # Minimize validation cost; on exact ties prefer the more conservative threshold.
    _, _, threshold, validation_deviation = min(scored)
    _, tb, tr, tm = predictions(model, data['test'], payload['mean'],
                                payload['scale'], objective)
    values, deviations = selected_values(data['test'], tb, tr, tm, threshold)
    route = data['test']['route']
    delta = route - values
    mean, se = float(delta.mean()), float(delta.std(unbiased=True) / len(delta) ** .5)
    oracle = data['test']['costs'].amin(-1)
    result = {'role': 'validation_selected_conservative_value_selector',
              'threshold': threshold if math.isfinite(threshold) else str(threshold),
              'validation_deviation_rate': validation_deviation,
              'test_deviation_rate': float(deviations.float().mean()),
              'mean_cost': float(values.mean()),
              'relative_improvement_percent': 100 * mean / float(route.mean()),
              'conditional_normal_ci95': [mean - 1.96 * se, mean + 1.96 * se],
              'improved_fraction': float((delta > 1e-7).float().mean()),
              'oracle_hit_rate': float((values <= oracle + 1e-7).float().mean()),
              'threshold_candidates': len(candidates),
              'test_not_used_for_threshold': True}
    (args.run / 'abstention_summary.json').write_text(json.dumps(result, indent=2,
                                                                  allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
