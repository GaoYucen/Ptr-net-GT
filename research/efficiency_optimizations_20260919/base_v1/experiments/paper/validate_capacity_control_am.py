"""验证 AM 容量匹配对照的参数与单链语义。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from groupopt.models.am import AttentionModel


MODES = (
    "native_original",
    "native_capacity_single_chain",
    "native_conditional_free",
)


def _validate_mode(mode: str, device: torch.device) -> dict[str, object]:
    torch.manual_seed(1701)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1701)
    model = AttentionModel(
        embedding_dim=128,
        n_heads=8,
        n_encoder_layers=3,
        feed_forward_dim=512,
        normalization="batch",
    ).to(device)
    coordinates = torch.rand(
        8,
        20,
        2,
        device=device,
        generator=torch.Generator(device=device).manual_seed(1702),
    )
    output = model(
        coordinates,
        decode_type="sampling",
        base_mode=mode,
        generator=torch.Generator(device=device).manual_seed(1703),
    )
    (output.cost.detach() * output.log_likelihood).mean().backward()

    active_names: list[str] = []
    inactive_names: list[str] = []
    active_count = 0
    for name, parameter in model.named_parameters():
        is_active = parameter.grad is not None and bool(
            torch.count_nonzero(parameter.grad).item()
        )
        if is_active:
            active_names.append(name)
            active_count += parameter.numel()
        else:
            inactive_names.append(name)

    successor = output.successor.sort(dim=1).values
    expected = torch.arange(20, device=device).expand(8, 20)
    single_chain = bool(torch.equal(output.tails[:, 1:], output.heads[:, :-1]))
    return {
        "mode": mode,
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
        "active_gradient_parameter_count": active_count,
        "active_parameter_names": active_names,
        "inactive_parameter_names": inactive_names,
        "single_chain": single_chain,
        "hamiltonian": bool(torch.equal(successor, expected)),
        "finite_cost": bool(torch.isfinite(output.cost).all()),
    }


def main(output: Path, device_name: str) -> None:
    device = torch.device(device_name)
    results = {mode: _validate_mode(mode, device) for mode in MODES}
    original = results["native_original"]
    capacity = results["native_capacity_single_chain"]
    full = results["native_conditional_free"]

    assert bool(original["single_chain"])
    assert bool(capacity["single_chain"])
    assert bool(original["hamiltonian"])
    assert bool(capacity["hamiltonian"])
    assert bool(full["hamiltonian"])
    assert int(capacity["total_parameter_count"]) == int(full["total_parameter_count"])
    assert int(capacity["active_gradient_parameter_count"]) == int(
        capacity["total_parameter_count"]
    )
    assert int(full["active_gradient_parameter_count"]) == int(
        full["total_parameter_count"]
    )
    assert int(original["active_gradient_parameter_count"]) < int(
        capacity["active_gradient_parameter_count"]
    )

    payload = {
        "status": "passed",
        "definition": (
            "Capacity-Matched keeps the native single chain while activating exactly "
            "the same parameter set as Full GroupOpt"
        ),
        "additional_active_parameters": int(
            capacity["active_gradient_parameter_count"]
        )
        - int(original["active_gradient_parameter_count"]),
        "modes": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    main(arguments.output.resolve(), arguments.device)
