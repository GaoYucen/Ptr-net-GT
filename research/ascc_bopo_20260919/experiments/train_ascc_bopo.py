"""Bounded, provenance-recorded factorial experiments. No implicit resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from groupopt.models.ascc_bopo import SourceFirstASCC
from groupopt.training.preference import bopo_loss, matched_reinforce_loss


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, coords, args, device):
    model.eval()
    chunks, deviations, eligible = [], 0., 0.
    rng = torch.Generator(device=device).manual_seed(args.seed + 70001)
    synchronize(device)
    begin = time.monotonic()
    for x in coords.split(args.eval_batch):
        out = model(x.to(device), args.eval_rollouts, args.source, 'greedy', rng,
                    anchor_mode='multi')
        chunks.append(out.costs.amin(1).cpu())
        deviations += float(out.voluntary_deviations.sum())
        eligible += float(out.eligible_deviations.sum())
    synchronize(device)
    costs = torch.cat(chunks)
    return costs, {'mean_cost': float(costs.double().mean()),
                   'seconds': time.monotonic() - begin,
                   'voluntary_source_deviation': deviations / max(eligible, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--objective', choices=['bopo', 'reinforce'], required=True)
    p.add_argument('--source', choices=['route', 'learned', 'route_capacity', 'random',
                                      'fixed', 'shortest'], required=True)
    p.add_argument('--backbone', type=Path)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--nodes', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--rollouts', type=int, default=16)
    p.add_argument('--filtered', type=int, default=8)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--validation-size', type=int, default=128)
    p.add_argument('--test-size', type=int, default=256)
    p.add_argument('--eval-rollouts', type=int, default=8)
    p.add_argument('--eval-batch', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--max-seconds', type=int, default=900)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--memory-fraction', type=float, default=0.22)
    p.add_argument('--tiny', action='store_true')
    args = p.parse_args()
    if args.steps < 1 or args.nodes < 2 or not 2 <= args.filtered <= args.rollouts:
        p.error('invalid steps/nodes/rollout/filter budget')
    if args.out.exists():
        p.error('output directory already exists; choose a new run id')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no automatic CPU fallback for training')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dimensions = dict(embedding_dim=16, head_num=2, qkv_dim=8, encoder_layers=1,
                      feed_forward_dim=32, source_dim=8) if args.tiny else {}
    model = SourceFirstASCC(**dimensions).to(device)
    if args.backbone:
        payload = torch.load(args.backbone, map_location='cpu', weights_only=False)
        model.load_backbone(payload.get('model_state_dict', payload))
    backbone, adapter = [], []
    for name, param in model.named_parameters():
        (backbone if name.startswith(('encoder.', 'decoder.')) else adapter).append(param)
    optimizer = torch.optim.Adam([
        {'params': backbone, 'lr': args.backbone_lr if args.backbone else args.lr},
        {'params': adapter, 'lr': args.lr}])
    args.out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), root / 'src/groupopt/models/ascc_bopo.py',
                    root / 'src/groupopt/training/preference.py',
                    root / 'src/groupopt/models/pomo.py']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(dimensions=dimensions, source_hashes={str(f.relative_to(root)): sha(f)
                  for f in source_files},
                  git_head=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                                   text=True).strip(),
                  backbone_sha256=sha(args.backbone) if args.backbone else None,
                  torch_version=torch.__version__,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  ld_library_path=os.environ.get('LD_LIBRARY_PATH'),
                  total_parameters=sum(q.numel() for q in model.parameters()),
                  all_backbone_parameters_trainable=all(q.requires_grad for q in backbone),
                  protocol='screening_v1_complete_tour_joint_policy_no_source_gate',
                  train_sampling='B-1 stochastic + one explicit greedy; same initial anchor context',
                  evaluation='multi anchor context; best of K; no augmentation; no local search')
    (args.out / 'source').mkdir()
    for f in source_files:
        dest = args.out / 'source' / f.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    train_rng = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    action_rng = torch.Generator(device=device).manual_seed(args.seed + 2)
    validation = torch.rand(args.validation_size, args.nodes, 2,
                            generator=torch.Generator().manual_seed(2026091903))
    test = torch.rand(args.test_size, args.nodes, 2,
                     generator=torch.Generator().manual_seed(2026091904))
    config['validation_sha256'], config['test_sha256'] = tensor_sha(validation), tensor_sha(test)
    atomic_json(args.out / 'config.json', config)
    torch.save({'validation': validation, 'test': test}, args.out / 'evaluation_instances.pt')
    best, best_step = float('inf'), 0

    def save(name, step):
        temp = args.out / (name + '.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': config, 'step': step, 'best_validation': best,
                    'best_step': best_step, 'train_rng': train_rng.get_state(),
                    'action_rng': action_rng.get_state(), 'torch_rng': torch.get_rng_state()}, temp)
        temp.replace(args.out / name)

    def record(row):
        with (args.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    initial_costs, initial = evaluate(model, validation, args, device)
    best = initial['mean_cost']
    save('initial.pt', 0)
    save('best.pt', 0)
    torch.save(initial_costs, args.out / 'initial_validation_costs.pt')
    record(dict(step=0, validation=initial, checkpoint='initial'))
    started = time.monotonic()
    train_seconds = 0.0
    reason = 'completed_steps'
    try:
        for step in range(1, args.steps + 1):
            if time.monotonic() - started >= args.max_seconds:
                reason = 'time_budget'
                step -= 1
                break
            model.train()
            coords = torch.rand(args.batch, args.nodes, 2, generator=train_rng).to(device)
            synchronize(device)
            tick = time.monotonic()
            out = model(coords, args.rollouts, args.source, 'hybrid', action_rng)
            if args.objective == 'bopo':
                loss, extra = bopo_loss(out.costs, out.mean_logp, args.filtered)
            else:
                loss, extra = matched_reinforce_loss(out.costs, out.mean_logp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                 error_if_nonfinite=True)
            source_grad = sum(float(q.grad.detach().square().sum()) for name, q in
                              model.named_parameters() if name.startswith('source_')
                              and q.grad is not None) ** 0.5
            endpoint_grad = float(model.decoder.Wq_last.weight.grad.norm())
            optimizer.step()
            synchronize(device)
            train_seconds += time.monotonic() - tick
            if step % args.log_every == 0 or step == 1 or step == args.steps:
                record(dict(step=step, loss=float(loss.detach()),
                            train_mean=float(out.costs.detach().mean()),
                            train_best=float(out.costs.detach().amin(1).mean()),
                            source_entropy=float(out.source_entropy.detach().mean()),
                            grad_norm=float(grad), source_grad_norm=source_grad,
                            endpoint_grad_norm=endpoint_grad, train_seconds=train_seconds,
                            elapsed_seconds=time.monotonic() - started,
                            peak_memory_gb=torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == 'cuda' else None, **extra))
            if step % args.eval_every == 0 or step == args.steps:
                _, metrics = evaluate(model, validation, args, device)
                if metrics['mean_cost'] < best:
                    best, best_step = metrics['mean_cost'], step
                    save('best.pt', step)
                record(dict(step=step, validation=metrics, best_step=best_step))
                save('latest.pt', step)
        save('final.pt', step)
        final_costs, final_test = evaluate(model, test, args, device)
        torch.save(final_costs, args.out / 'final_test_costs.pt')
        selected = torch.load(args.out / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(selected['model'], strict=True)
        best_costs, best_test = evaluate(model, test, args, device)
        torch.save(best_costs, args.out / 'best_test_costs.pt')
        summary = dict(status=reason, steps=step, best_step=best_step,
                       initial_validation=initial, best_validation=best,
                       final_test=final_test, best_test=best_test,
                       train_seconds=train_seconds, total_elapsed=time.monotonic() - started,
                       test_sha256=config['test_sha256'],
                       final_checkpoint_sha256=sha(args.out / 'final.pt'),
                       best_checkpoint_sha256=sha(args.out / 'best.pt'))
        atomic_json(args.out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        atomic_json(args.out / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
                                               'elapsed': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
