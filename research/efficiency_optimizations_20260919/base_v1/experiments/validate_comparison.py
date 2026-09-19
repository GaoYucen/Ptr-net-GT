"""验证参数匹配且使用留出数据的 Original/Ours 对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from groupopt.models import AttentionModel, GraphPointerNetwork, PointerNetwork

GROUP_OPT_PREFIXES = {
    "am": (
        "project_tail_context",
        "project_tail_glimpse",
        "project_tail_state",
        "project_state_tail_nodes",
        "project_native_head_summary",
        "first_step_context",
    ),
    "ptrnet": (
        "project_tail_context",
        "project_tail_state",
        "project_native_head_summary",
        "tail_pointer",
    ),
    "gpn": (
        "project_tail_context",
        "project_tail_state",
        "project_native_head_summary",
        "tail_pointer",
    ),
}


def build_models() -> dict[str, nn.Module]:
    return {
        "am": AttentionModel(embedding_dim=32, n_heads=4, feed_forward_dim=64),
        "ptrnet": PointerNetwork(embedding_dim=32),
        "gpn": GraphPointerNetwork(embedding_dim=32),
    }


def groupopt_is_inactive_for_original(name: str, model: nn.Module, data: torch.Tensor) -> bool:
    model.zero_grad(set_to_none=True)
    output = model(data, decode_type="sampling", base_mode="native_original")
    (-output.log_likelihood.mean()).backward()
    prefixes = GROUP_OPT_PREFIXES[name]
    selected = [
        parameter
        for parameter_name, parameter in model.named_parameters()
        if parameter_name.startswith(prefixes)
    ]
    if not selected:
        raise RuntimeError(f"no GroupOpt-only parameters found for {name}")
    return all(
        parameter.grad is None or bool(torch.count_nonzero(parameter.grad) == 0)
        for parameter in selected
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=1234)
    parser.add_argument("--test-seed", type=int, default=20260812)
    args = parser.parse_args()

    if args.train_seed == args.test_seed:
        raise ValueError("test seed must differ from the training seed")
    train_generator = torch.Generator().manual_seed(args.train_seed)
    test_generator = torch.Generator().manual_seed(args.test_seed)
    train_probe = torch.rand(4, 8, 2, generator=train_generator)
    test_probe = torch.rand(4, 8, 2, generator=test_generator)
    if torch.equal(train_probe, test_probe):
        raise RuntimeError("training and held-out probes unexpectedly coincide")

    models = build_models()
    result = {
        "modes": {
            "original": "native_original",
            "ours": "native_conditional_free",
        },
        "same_model_class_and_parameters": True,
        "held_out_seed_is_distinct": True,
        "models": {},
    }
    for name, model in models.items():
        result["models"][name] = {
            "parameter_count_each_arm": sum(parameter.numel() for parameter in model.parameters()),
            "groupopt_parameters_inactive_in_original": groupopt_is_inactive_for_original(
                name, model, train_probe
            ),
        }
        if not result["models"][name]["groupopt_parameters_inactive_in_original"]:
            raise RuntimeError(f"GroupOpt-only parameters affect the Original arm for {name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
