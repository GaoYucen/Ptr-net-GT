"""用官方 POMO checkpoint 验证 `native_original` 的逐动作兼容性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from groupopt.models.pomo import POMOModel


def _official_rollout(model: torch.nn.Module, coordinates: torch.Tensor) -> torch.Tensor:
    batch_size, node_count, _ = coordinates.shape
    device = coordinates.device
    model.pre_forward(SimpleNamespace(problems=coordinates))
    batch_index = torch.arange(batch_size, device=device)[:, None].expand(
        batch_size, node_count
    )
    pomo_index = torch.arange(node_count, device=device)[None, :].expand(
        batch_size, node_count
    )
    mask = torch.zeros(batch_size, node_count, node_count, device=device)
    current = None
    selected_nodes = []
    for _ in range(node_count):
        state = SimpleNamespace(
            BATCH_IDX=batch_index,
            POMO_IDX=pomo_index,
            current_node=current,
            ninf_mask=mask,
        )
        selected, _ = model(state)
        selected_nodes.append(selected)
        mask[batch_index, pomo_index, selected] = -torch.inf
        current = selected
    return torch.stack(selected_nodes, dim=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    official_root = args.official_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    sys.path.insert(0, str(official_root))
    from TSPModel import TSPModel  # type: ignore[import-not-found]

    device = torch.device(args.device)
    params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128**0.5,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    official = TSPModel(**params).to(device)
    official.load_state_dict(checkpoint["model_state_dict"], strict=True)
    official.eval()

    ours = POMOModel(pomo_size=50).to(device)
    incompatible = ours.load_state_dict(checkpoint["model_state_dict"], strict=False)
    groupopt_only = ("project_tail_state.", "project_head_summary.", "tail_selector.")
    if incompatible.unexpected_keys or any(
        not key.startswith(groupopt_only) for key in incompatible.missing_keys
    ):
        raise RuntimeError(f"POMO checkpoint 参数布局不兼容：{incompatible}")
    ours.eval()

    coordinates = torch.rand(
        2,
        50,
        2,
        device=device,
        generator=torch.Generator(device=device).manual_seed(20260829),
    )
    with torch.no_grad():
        official_tour = _official_rollout(official, coordinates)
        ours_output = ours(coordinates, decode_type="greedy", base_mode="native_original")
    ours_tour = ours_output.tails.reshape(2, 50, 50)
    ordered = (
        coordinates[:, None]
        .expand(2, 50, 50, 2)
        .gather(2, official_tour.unsqueeze(-1).expand(2, 50, 50, 2))
    )
    official_cost = (ordered - ordered.roll(-1, 2)).norm(dim=-1).sum(dim=2)
    ours_cost = ours_output.cost.reshape(2, 50)

    result = {
        "checkpoint": str(checkpoint_path),
        "state_dict_layout_compatible": True,
        "original_tours_exact": torch.equal(ours_tour, official_tour),
        "original_costs_close": torch.allclose(
            ours_cost, official_cost, atol=1e-6, rtol=1e-6
        ),
        "original_cost_max_abs_diff": float((ours_cost - official_cost).abs().max()),
    }
    if not all(value for value in result.values() if isinstance(value, bool)):
        raise RuntimeError(f"POMO 官方兼容性验证失败：{result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
