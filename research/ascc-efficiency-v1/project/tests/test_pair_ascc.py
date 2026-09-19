import itertools

import pytest
import torch

from groupopt.models.ascc_bopo import SourceFirstASCC
from groupopt.models.pair_ascc import PairASCC


def small():
    torch.manual_seed(12)
    return PairASCC(embedding_dim=16, head_num=2, qkv_dim=8,
                    encoder_layers=1, feed_forward_dim=32, source_dim=8,
                    interaction_rank=4)


def assert_tours(out, coords):
    for bi, batch in enumerate(out.successor.tolist()):
        for ki, successor in enumerate(batch):
            assert sorted(successor) == list(range(len(successor)))
            seen, cur = [], 0
            for _ in successor:
                assert cur not in seen
                seen.append(cur)
                cur = successor[cur]
            assert cur == 0
            expected = sum(torch.dist(coords[bi, i], coords[bi, j]).item()
                           for i, j in enumerate(successor))
            assert float(out.costs[bi, ki]) == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize('n', range(2, 10))
@pytest.mark.parametrize('interaction', [False, True])
def test_pair_decoder_returns_valid_tours(n, interaction):
    coords = torch.rand(3, n, 2)
    out = small()(coords, 4, interaction=interaction, decode='hybrid',
                  generator=torch.Generator().manual_seed(19), validate=True)
    assert out.tails.shape == (3, 4, n)
    assert out.mean_logp.shape == (3, 4)
    assert torch.isfinite(out.mean_logp).all()
    assert_tours(out, coords)


def test_interaction_zero_initialization_is_exactly_additive():
    model = small().eval()
    coords = torch.rand(2, 8, 2)
    additive = model(coords, 3, interaction=False, decode='greedy', validate=True)
    enabled = model(coords, 3, interaction=True, decode='greedy', validate=True)
    torch.testing.assert_close(additive.costs, enabled.costs, rtol=0, atol=0)
    torch.testing.assert_close(additive.mean_logp, enabled.mean_logp, rtol=0, atol=0)


def test_interaction_receives_gradient():
    model = small()
    model.pair_scale.data.fill_(0.1)
    out = model(torch.rand(2, 8, 2), 6, interaction=True,
                generator=torch.Generator().manual_seed(7))
    (out.costs.detach() * out.mean_logp).mean().backward()
    for name in ('pair_scale', 'pair_left.weight', 'pair_right.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


def test_fixed_source_optimization_is_numerically_exact():
    torch.manual_seed(21)
    model = SourceFirstASCC(embedding_dim=16, head_num=2, qkv_dim=8,
                            encoder_layers=1, feed_forward_dim=32, source_dim=8).eval()
    coords = torch.rand(3, 9, 2)
    historical = model(coords, 1, 'fixed', 'greedy', validate=True)
    optimized = model(coords, 1, 'fixed', 'greedy', validate=True,
                      skip_source_scoring=True)
    torch.testing.assert_close(historical.costs, optimized.costs, rtol=0, atol=0)
    torch.testing.assert_close(historical.successor, optimized.successor, rtol=0, atol=0)


def test_all_four_node_tours_are_representable_by_some_fixed_source_pair_action():
    # For the fixed source order (0,1),(2,3), every anchored Hamilton tour is a
    # legal pair trajectory.  This checks support, independent of network scores.
    n = 4
    for rest in itertools.permutations(range(1, n)):
        tour = (0,) + rest
        succ = {tour[i]: tour[(i + 1) % n] for i in range(n)}
        assert len(set(succ.values())) == n
