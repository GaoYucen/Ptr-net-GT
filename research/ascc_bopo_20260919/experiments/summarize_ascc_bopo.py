"""Report all arms, step-zero selections and fixed-model paired uncertainty."""
import json
from pathlib import Path
import sys

import torch


def main():
    root = Path(sys.argv[1])
    records, vectors, checks = {}, {}, []
    for directory in sorted(root.glob('*-seed*')):
        if not (directory / 'summary.json').exists():
            continue
        summary = json.loads((directory / 'summary.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        records[directory.name] = dict(summary=summary, config=config)
        vectors[directory.name] = torch.load(directory / 'best_test_costs.pt',
                                            map_location='cpu', weights_only=True).double()
    if len(records) != 4:
        raise RuntimeError(f'Expected all four primary arms, found {len(records)}')
    hashes = {v['config']['test_sha256'] for v in records.values()}
    if len(hashes) != 1:
        raise RuntimeError('Evaluation instances do not match')
    for objective in ('bopo', 'reinforce'):
        a, b = f'route-{objective}-seed1234', f'learned-{objective}-seed1234'
        differences = vectors[a] - vectors[b]
        se = float(differences.std(unbiased=True) / differences.numel()**0.5)
        mean = float(differences.mean())
        checks.append(dict(objective=objective, baseline=a, ascc=b, paired_difference=mean,
                           relative_improvement_percent=100*mean/float(vectors[a].mean()),
                           conditional_normal_ci95=[mean-1.96*se, mean+1.96*se],
                           instance_count=differences.numel(), training_seeds=1,
                           ascc_best_step=records[b]['summary']['best_step']))
    (root / 'comparison.json').write_text(json.dumps({'arms': records, 'comparisons': checks},
                                                    indent=2, allow_nan=False))
    lines = ['# ASCC × BOPO: one-seed screening results', '',
             '200-step pilot, not a converged or cross-seed result. Positive delta favors ASCC.', '',
             '| Source | Objective | Initial val | Best step | Best test | Final test | Train s |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, data in records.items():
        s, c = data['summary'], data['config']
        lines.append(f"| {c['source']} | {c['objective']} | {s['initial_validation']['mean_cost']:.6f} | "
                     f"{s['best_step']} | {s['best_test']['mean_cost']:.6f} | "
                     f"{s['final_test']['mean_cost']:.6f} | {s['train_seconds']:.1f} |")
    lines += ['', '| Objective | Route−ASCC | Relative improvement | Conditional instance CI |',
              '| --- | ---: | ---: | --- |']
    for c in checks:
        lo, hi = c['conditional_normal_ci95']
        lines.append(f"| {c['objective']} | {c['paired_difference']:+.6f} | "
                     f"{c['relative_improvement_percent']:+.3f}% | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', 'The interval conditions on these fitted models. It is not training-seed uncertainty.',
              'Best step0 means the selected model is the initialization; do not call it learned gain.',
              'All checkpoints, final results, configs and histories remain available.']
    (root / 'SCREEN_RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
