"""AM-style canonical GroupOpt 信息消融的接口回归测试。"""

from __future__ import annotations

import torch

from groupopt.models.am import AttentionModel

MODES = (
    "native_forest_fixed",
    "native_free_no_head_summary",
    "native_free_no_path_state",
    "native_free_no_last_head",
    "native_random_tail",
)


def test_capacity_control_matches_full_active_parameters_and_stays_single_chain() -> None:
    coordinates = torch.rand(4, 12, 2, generator=torch.Generator().manual_seed(19))
    active_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    for mode in (
        "native_original",
        "native_capacity_single_chain",
        "native_conditional_free",
    ):
        torch.manual_seed(20)
        model = AttentionModel(
            embedding_dim=32,
            n_heads=4,
            n_encoder_layers=1,
            feed_forward_dim=64,
        )
        output = model(
            coordinates,
            decode_type="sampling",
            base_mode=mode,
            generator=torch.Generator().manual_seed(21),
        )
        (output.cost.detach() * output.log_likelihood).mean().backward()
        total_counts[mode] = sum(parameter.numel() for parameter in model.parameters())
        active_counts[mode] = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad).item())
        )
        if mode != "native_conditional_free":
            assert torch.equal(output.tails[:, 1:], output.heads[:, :-1])

    assert total_counts["native_capacity_single_chain"] == total_counts[
        "native_conditional_free"
    ]
    assert active_counts["native_capacity_single_chain"] == total_counts[
        "native_capacity_single_chain"
    ]
    assert active_counts["native_conditional_free"] == total_counts[
        "native_conditional_free"
    ]
    assert active_counts["native_original"] < active_counts[
        "native_capacity_single_chain"
    ]


def test_am_ablation_modes_are_hamiltonian_and_differentiable() -> None:
    coordinates = torch.rand(4, 12, 2, generator=torch.Generator().manual_seed(21))
    expected = torch.arange(12).expand(4, 12)
    for mode in MODES:
        model = AttentionModel(
            embedding_dim=32,
            n_heads=4,
            n_encoder_layers=1,
            feed_forward_dim=64,
        )
        output = model(
            coordinates,
            decode_type="sampling",
            base_mode=mode,
            generator=torch.Generator().manual_seed(22),
        )
        assert torch.equal(output.successor.sort(dim=1).values, expected)
        assert torch.isfinite(output.cost).all()
        loss = (output.cost.detach() * output.log_likelihood).mean()
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if mode == "native_forest_fixed":
            tail_only_prefixes = (
                "project_tail_context.",
                "project_tail_glimpse.",
                "project_tail_state.",
                "project_state_tail_nodes.",
                "project_native_head_summary.",
            )
            assert all(
                parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                for name, parameter in model.named_parameters()
                if name.startswith(tail_only_prefixes)
            )
        if mode == "native_random_tail":
            assert all(
                parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                for name, parameter in model.named_parameters()
                if name.startswith("project_tail_")
                or name.startswith("project_state_tail_nodes.")
                or name.startswith("project_native_head_summary.")
            )


def test_random_tail_is_reproducible_and_not_a_fixed_index_rule() -> None:
    coordinates = torch.rand(8, 12, 2, generator=torch.Generator().manual_seed(25))
    model = AttentionModel(
        embedding_dim=32,
        n_heads=4,
        n_encoder_layers=1,
        feed_forward_dim=64,
    )
    first = model(
        coordinates,
        decode_type="greedy",
        base_mode="native_random_tail",
        generator=torch.Generator().manual_seed(26),
    )
    second = model(
        coordinates,
        decode_type="greedy",
        base_mode="native_random_tail",
        generator=torch.Generator().manual_seed(26),
    )
    assert torch.equal(first.tails, second.tails)
    assert torch.unique(first.tails[:, 0]).numel() > 1


def test_each_information_ablation_disables_its_dedicated_parameters() -> None:
    coordinates = torch.rand(4, 12, 2, generator=torch.Generator().manual_seed(23))
    checks = {
        "native_free_no_head_summary": "project_native_head_summary.weight",
        "native_free_no_path_state": "project_tail_state.weight",
    }
    for mode, parameter_name in checks.items():
        model = AttentionModel(
            embedding_dim=32,
            n_heads=4,
            n_encoder_layers=1,
            feed_forward_dim=64,
        )
        output = model(
            coordinates,
            decode_type="sampling",
            base_mode=mode,
            generator=torch.Generator().manual_seed(24),
        )
        (output.cost.detach() * output.log_likelihood).mean().backward()
        parameter = dict(model.named_parameters())[parameter_name]
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
