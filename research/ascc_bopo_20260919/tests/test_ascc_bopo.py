import itertools
import math

import pytest
import torch

from groupopt.models.ascc_bopo import SourceFirstASCC
from groupopt.models.pomo import POMOModel
from groupopt.training.preference import bopo_loss, matched_reinforce_loss

torch.set_num_threads(2)


def small():
    torch.manual_seed(12)
    return SourceFirstASCC(embedding_dim=16, head_num=2, qkv_dim=8,
                           encoder_layers=1, feed_forward_dim=32, source_dim=8)


def assert_tour(out, coords):
    for b, tours in enumerate(out.successor.tolist()):
        for k, successor in enumerate(tours):
            assert sorted(successor) == list(range(len(successor)))
            visited, cur = [], 0
            for _ in successor:
                assert cur not in visited
                visited.append(cur)
                cur = successor[cur]
            assert cur == 0
            expected = sum(torch.dist(coords[b, i], coords[b, j]).item()
                           for i, j in enumerate(successor))
            assert float(out.costs[b, k]) == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize('policy', ['route', 'route_capacity', 'learned', 'random',
                                  'fixed', 'shortest', 'min_entropy', 'max_margin',
                                  'shortest_edge'])
def test_feasible_tours_and_replayed_joint_likelihood(policy):
    model = small()
    coords = torch.rand(3, 7, 2)
    out = model(coords, 4, policy, generator=torch.Generator().manual_seed(2), validate=True)
    assert_tour(out, coords)
    replay = model(coords, 4, policy, actions=(out.tails, out.heads), validate=True)
    torch.testing.assert_close(out.mean_logp, replay.mean_logp, rtol=0, atol=0)
    torch.testing.assert_close(out.costs, replay.costs, rtol=0, atol=0)


def test_route_exactly_recovers_native_pomo_at_initialization():
    model = small().eval()
    native = POMOModel(embedding_dim=16, head_num=2, qkv_dim=8, encoder_layers=1,
                       feed_forward_dim=32, pomo_size=3).eval()
    state = native.state_dict()
    model.load_backbone(state)
    x = torch.rand(2, 9, 2)
    out = model(x, 3, 'route', 'greedy', anchor_mode='multi', validate=True)
    ref = native(x, decode_type='greedy', base_mode='native_original')
    torch.testing.assert_close(out.costs.reshape(-1), ref.cost, rtol=0, atol=1e-6)
    assert torch.equal(out.tails.reshape(-1, 9), ref.tails)
    assert torch.equal(out.heads.reshape(-1, 9), ref.heads)
    torch.testing.assert_close(out.endpoint_logp.sum(-1).reshape(-1),
                               ref.log_likelihood, rtol=1e-6, atol=1e-6)


def test_hybrid_has_actual_greedy_column_in_training():
    model = small().train()
    x = torch.rand(2, 7, 2)
    greedy = model(x, 1, 'learned', 'greedy')
    one = model(x, 5, 'learned', 'hybrid', torch.Generator().manual_seed(1))
    two = model(x, 5, 'learned', 'hybrid', torch.Generator().manual_seed(9))
    assert torch.equal(greedy.tails[:, 0], one.tails[:, 0])
    assert torch.equal(greedy.heads[:, 0], one.heads[:, 0])
    assert torch.equal(one.heads[:, 0], two.heads[:, 0])
    assert not torch.equal(one.heads[:, 1:], two.heads[:, 1:])


def test_any_tour_reachable_in_adaptive_source_orders():
    # Enumerate every 4-node tour and every source order, score exact trajectories.
    n = 4
    ts, hs = [], []
    for p in itertools.permutations(range(1, n)):
        tour = (0,) + p
        succ = {tour[i]: tour[(i + 1) % n] for i in range(n)}
        for order in itertools.permutations(range(n)):
            ts.append(order)
            hs.append([succ[u] for u in order])
    model = small()
    coords = torch.rand(1, n, 2)
    out = model(coords, len(ts), actions=(torch.tensor([ts]), torch.tensor([hs])),
                validate=True)
    assert torch.isfinite(out.mean_logp).all()
    assert_tour(out, coords)


def test_premature_cycle_rejected():
    model = small()
    with pytest.raises(ValueError, match='illegal replay/endpoint'):
        model(torch.rand(1, 4, 2), 1,
              actions=(torch.tensor([[[0, 1, 2, 3]]]),
                       torch.tensor([[[1, 0, 3, 2]]])), validate=True)


def test_preference_direction_and_tie_mask():
    scores = torch.tensor([[-2., -2., -2., -2.]], requires_grad=True)
    loss, info = bopo_loss(torch.tensor([[2., 2., 3., 4.]]), scores, filtered=4)
    loss.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] == 0
    assert (scores.grad[0, 2:] > 0).all()
    assert info['valid_pairs'] == 2
    all_tied = scores.detach().requires_grad_()
    zero, info = bopo_loss(torch.ones(1, 4), all_tied, 4)
    zero.backward()
    assert zero.item() == 0 and not all_tied.grad.any()


def test_reinforce_does_not_score_greedy_and_uses_other_samples():
    scores = torch.zeros(1, 3, requires_grad=True)
    loss, _ = matched_reinforce_loss(torch.tensor([[1., 2., 4.]]), scores)
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[0., -0.25, 1.25]]))


def test_preference_backpropagates_to_source_and_endpoint():
    model = small()
    out = model(torch.rand(2, 8, 2), 8, generator=torch.Generator().manual_seed(42))
    loss, info = bopo_loss(out.costs, out.mean_logp, 4)
    loss.backward()
    assert info['valid_pairs'] > 0
    for name in ['source_context.weight', 'decoder.Wq_last.weight', 'head_readout.weight',
                 'encoder.embedding.weight']:
        g = dict(model.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.norm() > 0, name
    # Save/load cannot drop the learned branch or change greedy decisions.
    other = small()
    other.load_state_dict(model.state_dict(), strict=True)
    x = torch.rand(1, 8, 2)
    assert torch.equal(model(x, 1, decode='greedy').heads,
                       other(x, 1, decode='greedy').heads)


def test_same_parameter_control_activates_source_parameters():
    model = small()
    out = model(torch.rand(2, 8, 2), 8, 'route_capacity',
                generator=torch.Generator().manual_seed(52))
    loss, _ = bopo_loss(out.costs, out.mean_logp, 4)
    loss.backward()
    for name, p in model.named_parameters():
        if name.startswith('source_'):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
