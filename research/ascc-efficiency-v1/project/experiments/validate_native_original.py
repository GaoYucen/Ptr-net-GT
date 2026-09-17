"""验证 PtrNet/GPN 原生单链分支与固定起点 Forest 表达严格等价。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from groupopt.models.am import AttentionModel
from groupopt.models.gpn import GraphPointerNetwork
from groupopt.models.ptrnet import PointerNetwork


def _check_model(model: torch.nn.Module, device: torch.device) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for node_count, anchor in ((2, 0), (20, 0), (20, 3)):
        coordinates = torch.rand(8, node_count, 2, device=device)
        for decode_type in ("greedy", "sampling"):
            seed = 20260830 + node_count + anchor
            original = model(
                coordinates,
                decode_type=decode_type,
                base_mode="native_original",
                anchor=anchor,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            fixed = model(
                coordinates,
                decode_type=decode_type,
                base_mode="native_conditional_fixed",
                anchor=anchor,
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            prefix = f"n{node_count}_anchor{anchor}_{decode_type}"
            results[f"{prefix}_tour"] = torch.equal(original.tails, fixed.tails)
            results[f"{prefix}_heads"] = torch.equal(original.heads, fixed.heads)
            results[f"{prefix}_successor"] = torch.equal(original.successor, fixed.successor)
            results[f"{prefix}_cost"] = torch.allclose(
                original.cost, fixed.cost, atol=1e-6, rtol=1e-6
            )
            results[f"{prefix}_log_likelihood"] = torch.allclose(
                original.log_likelihood,
                fixed.log_likelihood,
                atol=1e-6,
                rtol=1e-6,
            )

    model.zero_grad(set_to_none=True)
    coordinates = torch.rand(8, 20, 2, device=device)
    output = model(
        coordinates,
        decode_type="sampling",
        base_mode="native_original",
        generator=torch.Generator(device=device).manual_seed(20260831),
    )
    (-output.log_likelihood.mean()).backward()
    tail_prefixes = (
        "project_tail_context",
        "project_tail_state",
        "project_native_head_summary",
        "tail_pointer",
    )
    results["groupopt_tail_parameters_inactive"] = all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for name, parameter in model.named_parameters()
        if name.startswith(tail_prefixes)
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260830)
    models = {
        "am": AttentionModel(
            embedding_dim=32,
            n_heads=4,
            n_encoder_layers=1,
            feed_forward_dim=64,
        ).to(device),
        "ptrnet": PointerNetwork(embedding_dim=32, n_encoder_layers=1).to(device),
        "gpn": GraphPointerNetwork(embedding_dim=32, n_encoder_layers=2).to(device),
    }
    result = {name: _check_model(model, device) for name, model in models.items()}
    if not all(all(checks.values()) for checks in result.values()):
        raise RuntimeError(f"native Original equivalence failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
