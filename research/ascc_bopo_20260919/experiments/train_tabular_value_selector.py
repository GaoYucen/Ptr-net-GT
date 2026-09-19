"""Small-data tabular alternatives for the fixed rollout-value dataset."""
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)


def flatten(split):
    x = split['features'].numpy()
    q = split['costs'].numpy()
    regret = 100 * (q - q.min(1, keepdims=True)) / q.min(1, keepdims=True)
    label = q <= q.min(1, keepdims=True) + 1e-7
    return x, q, regret, label


def predicted_scores(model, kind, x):
    flat = x.reshape(-1, x.shape[-1])
    if kind == 'classifier':
        prediction = model.predict_proba(flat)[:, 1]
        return -prediction.reshape(x.shape[:2])  # lower is preferred
    return model.predict(flat).reshape(x.shape[:2])


def select_with_gate(scores, features, threshold):
    best = scores.argmin(1)
    route = features[..., 11].argmax(1)
    row = np.arange(len(scores))
    margin = scores[row, route] - scores[row, best]
    selected = np.where(margin > threshold, best, route)
    return selected, margin, route


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    data = torch.load(args.run / 'candidate_dataset.pt', weights_only=False)
    tx, tq, ty, tl = flatten(data['train'])
    vx, vq, _, _ = flatten(data['validation'])
    candidates = []
    for leaf in (2, 5, 10, 20):
        candidates.append((f'extra-reg-leaf{leaf}', 'regressor', ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=leaf, max_features=1.0,
            random_state=501, n_jobs=2)))
        candidates.append((f'random-reg-leaf{leaf}', 'regressor', RandomForestRegressor(
            n_estimators=400, min_samples_leaf=leaf, max_features=.8,
            random_state=502, n_jobs=2)))
        candidates.append((f'extra-cls-leaf{leaf}', 'classifier', ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=leaf, max_features=1.0,
            class_weight='balanced', random_state=503, n_jobs=2)))
    for leaves in (7, 15, 31):
        candidates.append((f'hist-{leaves}', 'regressor', HistGradientBoostingRegressor(
            max_iter=250, max_leaf_nodes=leaves, l2_regularization=1., random_state=504)))
    fitted = []
    for name, kind, model in candidates:
        target = tl.reshape(-1).astype(int) if kind == 'classifier' else ty.reshape(-1)
        model.fit(tx.reshape(-1, tx.shape[-1]), target)
        scores = predicted_scores(model, kind, vx)
        _, margin, _ = select_with_gate(scores, vx, -np.inf)
        thresholds = np.unique(np.r_[[-np.inf, np.inf], np.quantile(margin, np.linspace(0, 1, 101))])
        for threshold in thresholds:
            selected, _, _ = select_with_gate(scores, vx, threshold)
            cost = vq[np.arange(len(vq)), selected].mean()
            # On exact ties prefer a more conservative threshold.
            fitted.append((float(cost), -float(threshold), name, kind,
                           float(threshold), model))
    validation_cost, _, name, kind, threshold, model = min(fitted, key=lambda row: row[:2])
    x, q, _, _ = flatten(data['test'])
    scores = predicted_scores(model, kind, x)
    selected, _, route_index = select_with_gate(scores, x, threshold)
    row = np.arange(len(q))
    values, route = q[row, selected], data['test']['route'].numpy()
    oracle = q.min(1)
    delta = route - values
    mean, se = float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))
    result = {'role': 'held_out_tabular_rollout_value_selector',
              'selected_on_validation': {'model': name, 'kind': kind,
                  'threshold': threshold if math.isfinite(threshold) else str(threshold),
                  'validation_cost': validation_cost},
              'candidate_model_families': len(candidates),
              'validation_model_threshold_combinations': len(fitted),
              'test_not_used_for_selection': True,
              'test_deviation_rate': float(np.mean(selected != route_index)),
              'mean_cost': float(values.mean()),
              'relative_improvement_percent': 100 * mean / float(route.mean()),
              'conditional_normal_ci95': [mean - 1.96 * se, mean + 1.96 * se],
              'improved_fraction': float(np.mean(delta > 1e-7)),
              'oracle_hit_rate': float(np.mean(values <= oracle + 1e-7))}
    (args.run / 'tabular_summary.json').write_text(json.dumps(result, indent=2,
                                                               allow_nan=False))
    joblib.dump(model, args.run / 'tabular_selector.joblib')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
