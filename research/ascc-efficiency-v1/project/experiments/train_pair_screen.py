"""Bounded three-arm TSP50 screen for registered atomic-pair ASCC hypotheses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from groupopt.training.preference import matched_reinforce_loss
from pair_validation import ALL_METHODS, build_model, make_dataset, run_model


def tensor_sha(value):
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def run(model, method, coords, trajectories, decode, generator):
    return run_model(model, method, coords, trajectories, decode, generator)


@torch.no_grad()
def evaluate(model, arm, data, trajectories, batch_size, device):
    model.eval()
    values = []
    sync(device)
    started = time.monotonic()
    for coords in data.split(batch_size):
        out = run(model, arm, coords.to(device), trajectories, 'greedy', None)
        values.append(out.costs.amin(1).cpu())
    sync(device)
    costs = torch.cat(values)
    return costs, dict(mean_cost=float(costs.double().mean()),
                       seconds=time.monotonic() - started)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=ALL_METHODS)
    # Kept for the completed formal-v1 queue; new runs should use --method.
    p.add_argument('--arm', choices=['single', 'pair_add', 'pair_interaction'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--nodes', type=int, default=50)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--rollouts', type=int, default=8)
    p.add_argument('--eval-rollouts', type=int, default=1)
    p.add_argument('--eval-batch', type=int, default=16)
    p.add_argument('--validation-size', type=int, default=512)
    p.add_argument('--test-size', type=int, default=10000)
    p.add_argument('--distribution', choices=['uniform', 'cluster', 'mixed'], default='uniform')
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-seconds', type=int, default=14400)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--memory-fraction', type=float, default=.28)
    args = p.parse_args()
    legacy = {'single': 'single_naive', 'pair_add': 'pair_add',
              'pair_interaction': 'pair_interaction'}
    if args.method is None and args.arm is None:
        p.error('one of --method or --arm is required')
    if args.method is not None and args.arm is not None:
        p.error('use --method or --arm, not both')
    args.method = args.method or legacy[args.arm]
    if args.method in ('am', 'pomo'):
        p.error('AM/POMO require their explicit external adapter and checkpoint')
    if args.out.exists():
        p.error('output exists')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = build_model(args.method, device, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_rng = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    action_rng = torch.Generator(device=device).manual_seed(args.seed + 2)
    validation = make_dataset(args.validation_size, args.nodes, args.distribution, 73019301)
    test = make_dataset(args.test_size, args.nodes, args.distribution, 73019302)
    args.out.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(protocol='registered_pair_tsp50_screen_v1',
                  validation_sha256=tensor_sha(validation), test_sha256=tensor_sha(test),
                  parameters=sum(param.numel() for param in model.parameters()),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    atomic_json(args.out / 'config.json', config)
    torch.save(dict(validation=validation, test=test), args.out / 'data.pt')
    history = []
    best, best_step = float('inf'), 0

    def save(name, step):
        temp = args.out / (name + '.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        config=config, step=step, best=best, best_step=best_step), temp)
        temp.replace(args.out / name)

    initial_costs, initial = evaluate(model, args.method, validation, args.eval_rollouts,
                                      args.eval_batch, device)
    best = initial['mean_cost']
    save('step0.pt', 0)
    save('best.pt', 0)
    torch.save(initial_costs, args.out / 'step0_validation_costs.pt')
    history.append(dict(step=0, validation=initial))
    started, train_seconds = time.monotonic(), 0.
    reason, step = 'completed_steps', 0
    try:
        for step in range(1, args.steps + 1):
            if time.monotonic() - started >= args.max_seconds:
                reason = 'time_budget'
                step -= 1
                break
            model.train()
            coords = torch.rand(args.batch, args.nodes, 2, generator=train_rng).to(device)
            sync(device)
            tick = time.monotonic()
            out = run(model, args.method, coords, args.rollouts, 'hybrid', action_rng)
            loss, extra = matched_reinforce_loss(out.costs, out.mean_logp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            sync(device)
            train_seconds += time.monotonic() - tick
            if step == 1 or step % args.log_every == 0:
                row = dict(step=step, loss=float(loss.detach()),
                           train_mean=float(out.costs.detach().mean()),
                           train_best=float(out.costs.detach().amin(1).mean()),
                           grad_norm=float(grad), train_seconds=train_seconds,
                           elapsed_seconds=time.monotonic() - started,
                           peak_memory_gb=(torch.cuda.max_memory_allocated(device) / 2**30
                                           if device.type == 'cuda' else None),
                           **extra)
                history.append(row)
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                _, metrics = evaluate(model, args.method, validation, args.eval_rollouts,
                                      args.eval_batch, device)
                if metrics['mean_cost'] < best:
                    best, best_step = metrics['mean_cost'], step
                    save('best.pt', step)
                history.append(dict(step=step, validation=metrics, best_step=best_step))
                atomic_json(args.out / 'history.json', history)
        save('final.pt', step)
        final_costs, final_metrics = evaluate(model, args.method, test, args.eval_rollouts,
                                              args.eval_batch, device)
        torch.save(final_costs, args.out / 'final_test_costs.pt')
        chosen = torch.load(args.out / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(chosen['model'], strict=True)
        best_costs, best_metrics = evaluate(model, args.method, test, args.eval_rollouts,
                                            args.eval_batch, device)
        torch.save(best_costs, args.out / 'best_test_costs.pt')
        atomic_json(args.out / 'history.json', history)
        summary = dict(status=reason, method=args.method, steps=step, best_step=best_step,
                       initial_validation=initial, best_validation=best,
                       final_test=final_metrics, best_test=best_metrics,
                       train_seconds=train_seconds, elapsed_seconds=time.monotonic() - started)
        atomic_json(args.out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        atomic_json(args.out / 'failure.json', dict(type=type(exc).__name__, error=str(exc),
                                                    step=step,
                                                    elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
