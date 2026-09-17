"""CVRP 与 Min-m-CCP 跨问题接口的最小回归测试。"""

from __future__ import annotations

import torch

from groupopt.models.am import AttentionModel
from groupopt.problems import BatchedCVRPConstruction, BatchedMCycleCoverConstruction
from groupopt.problems.distributions import generate_cvrp_instances


def _cycle_lengths(successor: torch.Tensor) -> list[int]:
    result: list[int] = []
    unseen = set(range(successor.numel()))
    while unseen:
        start = min(unseen)
        current = start
        length = 0
        while current in unseen:
            unseen.remove(current)
            current = int(successor[current])
            length += 1
        assert current == start
        result.append(length)
    return result


def _build_model(input_dim: int, process: object) -> AttentionModel:
    return AttentionModel(
        input_dim=input_dim,
        embedding_dim=32,
        n_heads=4,
        n_encoder_layers=1,
        feed_forward_dim=64,
        construction_process=process,
    )


def test_m_cycle_cover_both_modes_are_exact_and_valid() -> None:
    coordinates = torch.rand(5, 12, 2, generator=torch.Generator().manual_seed(7))
    for mode in ("native_original", "native_conditional_free"):
        model = _build_model(2, BatchedMCycleCoverConstruction(3))
        output = model(coordinates, decode_type="greedy", base_mode=mode)
        assert output.tails.shape == (5, 12)
        assert torch.isfinite(output.cost).all()
        for successor in output.successor:
            lengths = _cycle_lengths(successor)
            assert len(lengths) == 3
            assert min(lengths) >= 3


def test_cvrp_both_modes_cover_every_customer_with_capacity() -> None:
    instance = generate_cvrp_instances(
        6, 20, 30, torch.Generator().manual_seed(9)
    )
    for mode in ("native_original", "native_conditional_free"):
        model = _build_model(5, BatchedCVRPConstruction())
        output = model(instance, decode_type="greedy", base_mode=mode)
        assert output.successor.shape == (6, 20)
        assert torch.isfinite(output.cost).all()
        for sample, successor in zip(instance, output.successor, strict=True):
            predecessor = torch.full_like(successor, -1)
            valid = successor >= 0
            predecessor[successor[valid]] = torch.arange(20)[valid]
            starts = torch.where(predecessor < 0)[0]
            visited: set[int] = set()
            for start in starts.tolist():
                route: list[int] = []
                current = start
                while current >= 0:
                    assert current not in visited
                    visited.add(current)
                    route.append(current)
                    current = int(successor[current])
                assert float(sample[route, 2].sum()) <= 1.0 + 1e-6
            assert visited == set(range(20))


def test_groupopt_extensions_have_finite_policy_gradients() -> None:
    cases = [
        (
            torch.rand(4, 12, 2, generator=torch.Generator().manual_seed(11)),
            _build_model(2, BatchedMCycleCoverConstruction(3)),
        ),
        (
            generate_cvrp_instances(
                4, 20, 30, torch.Generator().manual_seed(12)
            ),
            _build_model(5, BatchedCVRPConstruction()),
        ),
    ]
    for instance, model in cases:
        output = model(
            instance,
            decode_type="sampling",
            base_mode="native_conditional_free",
            generator=torch.Generator().manual_seed(13),
        )
        loss = (output.cost.detach() * output.log_likelihood).mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert model.project_tail_context.weight.grad is not None
