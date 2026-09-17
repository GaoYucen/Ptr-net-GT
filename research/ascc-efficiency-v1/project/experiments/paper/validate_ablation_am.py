"""验证 AM 正式消融的构造合法性和参数开关。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from groupopt.models.am import AttentionModel


MODES = (
    "native_conditional_free",
    "native_random_tail",
    "native_free_no_head_summary",
    "native_free_no_path_state",
)


def _validate_mode(mode: str, device: torch.device) -> dict[str, object]:
    torch.manual_seed(1801)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1801)
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
        generator=torch.Generator(device=device).manual_seed(1802),
    )
    output = model(
        coordinates,
        decode_type="sampling",
        base_mode=mode,
        generator=torch.Generator(device=device).manual_seed(1803),
    )
    (output.cost.detach() * output.log_likelihood).mean().backward()

    active: list[str] = []
    inactive: list[str] = []
    finite_gradients = True
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            finite_gradients = finite_gradients and bool(
                torch.isfinite(parameter.grad).all()
            )
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
            active.append(name)
        else:
            inactive.append(name)

    expected = torch.arange(20, device=device).expand(8, 20)
    return {
        "mode": mode,
        "hamiltonian": bool(torch.equal(output.successor.sort(dim=1).values, expected)),
        "finite_cost": bool(torch.isfinite(output.cost).all()),
        "finite_gradients": finite_gradients,
        "active_parameter_names": active,
        "inactive_parameter_names": inactive,
    }


def main(output: Path, device_name: str) -> None:
    device = torch.device(device_name)
    results = {mode: _validate_mode(mode, device) for mode in MODES}
    for result in results.values():
        assert result["hamiltonian"]
        assert result["finite_cost"]
        assert result["finite_gradients"]

    full_inactive = set(results["native_conditional_free"]["inactive_parameter_names"])
    no_head_inactive = set(
        results["native_free_no_head_summary"]["inactive_parameter_names"]
    )
    no_path_inactive = set(
        results["native_free_no_path_state"]["inactive_parameter_names"]
    )
    random_inactive = set(results["native_random_tail"]["inactive_parameter_names"])
    assert "project_native_head_summary.weight" not in full_inactive
    assert "project_tail_state.weight" not in full_inactive
    assert "project_native_head_summary.weight" in no_head_inactive
    assert "project_tail_state.weight" in no_path_inactive
    random_prefixes = (
        "project_tail_",
        "project_state_tail_nodes.",
        "project_native_head_summary.",
    )
    expected_random_inactive = {
        name
        for name, _ in AttentionModel().named_parameters()
        if name.startswith(random_prefixes)
    }
    assert expected_random_inactive.issubset(random_inactive)

    # 固定随机种子时 Random Tail 必须可复现，但不能退化为固定编号规则。
    torch.manual_seed(1804)
    model = AttentionModel(
        embedding_dim=128,
        n_heads=8,
        n_encoder_layers=3,
        feed_forward_dim=512,
        normalization="batch",
    ).to(device)
    coordinates = torch.rand(
        16,
        20,
        2,
        device=device,
        generator=torch.Generator(device=device).manual_seed(1805),
    )
    first = model(
        coordinates,
        decode_type="greedy",
        base_mode="native_random_tail",
        generator=torch.Generator(device=device).manual_seed(1806),
    )
    second = model(
        coordinates,
        decode_type="greedy",
        base_mode="native_random_tail",
        generator=torch.Generator(device=device).manual_seed(1806),
    )
    assert torch.equal(first.tails, second.tails)
    assert torch.unique(first.tails[:, 0]).numel() > 1

    payload = {
        "status": "passed",
        "checks": {
            "all_modes_hamiltonian": True,
            "all_costs_and_gradients_finite": True,
            "no_head_summary_disables_dedicated_projection": True,
            "no_path_state_disables_dedicated_projection": True,
            "random_tail_disables_learned_tail_scorer": True,
            "random_tail_seeded_reproducibility": True,
            "random_tail_not_fixed_index": True,
        },
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
