"""Read-only comparison against the actual upstream BOPO model and checkpoint."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from groupopt.models.ascc_bopo import SourceFirstASCC


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--official-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('upstream_bopo', args.official_root / 'TSPModel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = dict(start_node='pomo', embedding_dim=128, sqrt_embedding_dim=128**0.5,
                  encoder_layer_num=6, qkv_dim=16, head_num=8, logit_clipping=10,
                  ff_hidden_dim=512, eval_type='argmax')
    upstream = module.TSPModel(**params).eval()
    ours = SourceFirstASCC().eval()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    upstream.load_state_dict(payload['model_state_dict'], strict=True)
    ours.load_backbone(payload['model_state_dict'])
    cases = []
    for n in (20, 100):
        coords = torch.rand(1, n, 2, generator=torch.Generator().manual_seed(987 + n))
        bi = torch.zeros(1, n, dtype=torch.long)
        ki = torch.arange(n)[None]
        mask = torch.zeros(1, n, n)
        current, tour = None, []
        with torch.no_grad():
            upstream.pre_forward(SimpleNamespace(problems=coords))
            for _ in range(n):
                current, _ = upstream(SimpleNamespace(BATCH_IDX=bi, B_IDX=ki,
                                        current_node=current, problem_size=n, ninf_mask=mask))
                tour.append(current)
                mask[bi, ki, current] = -torch.inf
            actual = torch.stack(tour, -1)
            out = ours(coords, n, 'route', 'greedy', anchor_mode='multi', validate=True)
        ordered = coords[:, None].expand(1, n, n, 2).gather(
            2, actual[..., None].expand(1, n, n, 2))
        expected_cost = (ordered - ordered.roll(-1, 2)).norm(dim=-1).sum(-1)
        row = dict(nodes=n, starts=n, actions_exact=torch.equal(actual, out.tails),
                   max_cost_difference=float((expected_cost - out.costs).abs().max()))
        assert row['actions_exact'] and row['max_cost_difference'] < 2e-6, row
        cases.append(row)
    result = dict(checkpoint=str(args.checkpoint),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  cases=cases, status='passed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
