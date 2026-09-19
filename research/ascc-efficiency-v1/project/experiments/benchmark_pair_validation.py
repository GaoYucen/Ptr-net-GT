"""Raw latency/memory benchmark for the registered Pair-ASCC comparison."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from pair_validation import ALL_METHODS, INTERNAL_METHODS, build_model, effective_parameter_count, make_dataset, run_model, sync


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=ALL_METHODS,
                        default=['single_naive', 'single_opt', 'seq2', 'pair_add', 'pair_interaction'])
    parser.add_argument('--nodes', nargs='+', type=int, default=[50, 100, 200, 500])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 16, 128])
    parser.add_argument('--trajectories', nargs='+', type=int, default=[1, 8, 128])
    parser.add_argument('--distribution', choices=['uniform', 'cluster', 'mixed'], default='uniform')
    parser.add_argument('--seed', type=int, default=2026091901)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeats', type=int, default=200)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-fraction', type=float, default=.70)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows = []
    for method in args.methods:
        if method not in INTERNAL_METHODS:
            rows.append({'method': method, 'status': 'missing_external_adapter'})
            continue
        for nodes in args.nodes:
            for batch in args.batches:
                coordinates = make_dataset(batch, nodes, args.distribution,
                                           args.seed + nodes * 1000 + batch)
                for trajectories in args.trajectories:
                    # Greedy is only meaningful at one trajectory.  Higher values
                    # are explicit sampling/search budgets.
                    decode = 'greedy' if trajectories == 1 else 'sampling'
                    for run in range(args.runs):
                        model = build_model(method, device, args.seed + run).eval()
                        coords = coordinates.to(device)
                        try:
                            with torch.inference_mode():
                                for _ in range(args.warmup):
                                    run_model(model, method, coords, trajectories, decode)
                                sync(device)
                                if device.type == 'cuda':
                                    torch.cuda.reset_peak_memory_stats(device)
                                for repeat in range(args.repeats):
                                    sync(device)
                                    wall_start = time.perf_counter()
                                    if device.type == 'cuda':
                                        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                                        start.record()
                                    output = run_model(model, method, coords, trajectories, decode)
                                    if device.type == 'cuda':
                                        end.record()
                                    sync(device)
                                    wall_seconds = time.perf_counter() - wall_start
                                    cuda_seconds = start.elapsed_time(end) / 1000 if device.type == 'cuda' else wall_seconds
                                    rows.append(dict(method=method, nodes=nodes, batch=batch,
                                                     trajectories=trajectories, decode=decode, run=run,
                                                     repeat=repeat, status='ok', wall_seconds=wall_seconds,
                                                     cuda_seconds=cuda_seconds,
                                                     latency_per_instance=wall_seconds / batch,
                                                     tours_per_second=batch * trajectories / wall_seconds,
                                                     allocated_gb=(torch.cuda.max_memory_allocated(device) / 2**30
                                                                   if device.type == 'cuda' else None),
                                                     reserved_gb=(torch.cuda.max_memory_reserved(device) / 2**30
                                                                  if device.type == 'cuda' else None),
                                                     parameters=effective_parameter_count(model, method),
                                                     mean_cost=float(output.costs.double().mean())))
                        except torch.cuda.OutOfMemoryError:
                            rows.append(dict(method=method, nodes=nodes, batch=batch,
                                             trajectories=trajectories, decode=decode, run=run,
                                             status='oom'))
                            torch.cuda.empty_cache()
                        finally:
                            del model
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
    fields = sorted({key for row in rows for key in row})
    with (args.out / 'latency_raw.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    memory_fields = ['method', 'nodes', 'batch', 'trajectories', 'decode', 'run', 'repeat',
                     'status', 'allocated_gb', 'reserved_gb', 'parameters']
    with (args.out / 'memory_raw.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=memory_fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for row in rows:
        if row.get('status') != 'ok':
            continue
        key = '|'.join(str(row[x]) for x in ('method', 'nodes', 'batch', 'trajectories', 'decode'))
        bucket = summary.setdefault(key, dict(method=row['method'], nodes=row['nodes'],
                                              batch=row['batch'], trajectories=row['trajectories'],
                                              decode=row['decode'], latency_seconds=[],
                                              allocated_gb=[], reserved_gb=[]))
        bucket['latency_seconds'].append(row['latency_per_instance'])
        if row['allocated_gb'] is not None:
            bucket['allocated_gb'].append(row['allocated_gb'])
            bucket['reserved_gb'].append(row['reserved_gb'])
    for bucket in summary.values():
        bucket['mean_latency_per_instance'] = statistics.fmean(bucket.pop('latency_seconds'))
        allocated, reserved = bucket.pop('allocated_gb'), bucket.pop('reserved_gb')
        bucket['peak_allocated_gb'] = max(allocated) if allocated else None
        bucket['peak_reserved_gb'] = max(reserved) if reserved else None
    (args.out / 'benchmark_summary.json').write_text(json.dumps(list(summary.values()), indent=2))
    (args.out / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2))


if __name__ == '__main__':
    main()
