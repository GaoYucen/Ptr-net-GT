"""Full-trajectory preference optimization and a matched rollout RL control."""
import torch
from torch import Tensor
from torch.nn import functional as F


def bopo_loss(costs: Tensor, mean_logp: Tensor, filtered=8, tie_rtol=1e-6):
    if costs.shape != mean_logp.shape or costs.ndim != 2:
        raise ValueError('costs and trajectory scores must both be (batch, rollouts)')
    b, count = costs.shape
    if not 2 <= filtered <= count:
        raise ValueError('require 2 <= filtered <= rollout count')
    if not torch.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError('BOPO objective-ratio loss requires finite positive costs')
    # Exactly K ranks, also when B is not divisible by K; always include the best.
    ranks = torch.arange(filtered, device=costs.device) * (count // filtered)
    order = costs.detach().argsort(dim=-1, stable=True)[:, ranks]
    c, ll = costs.detach().gather(1, order), mean_logp.gather(1, order)
    scale = c[:, 1:] / c[:, :1]
    valid = (c[:, 1:] - c[:, :1]) > tie_rtol * c[:, :1].abs()
    pair_logits = scale * (ll[:, :1] - ll[:, 1:])
    terms = F.softplus(-pair_logits)
    per_instance = (terms * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    loss = per_instance.mean()
    return loss, {'valid_pairs': int(valid.sum()), 'possible_pairs': b * (filtered - 1),
                  'mean_scale': float(scale.mean()),
                  'pair_accuracy': float(((pair_logits > 0) & valid).sum()
                                         / valid.sum().clamp_min(1))}


def matched_reinforce_loss(costs: Tensor, mean_logp: Tensor):
    """Column0 is greedy; optimize only sampled columns with a LOO baseline.

    The baseline includes the greedy result and the OTHER stochastic trajectories.
    Thus the scored sample is excluded; greedy actions do not receive a fictitious
    on-policy score gradient. All arms generate the same number of trajectories.
    """
    if costs.ndim != 2 or costs.shape != mean_logp.shape or costs.shape[1] < 2:
        raise ValueError('require matching (batch, rollouts>=2) arrays')
    c = costs.detach()
    baseline = (c.sum(-1, keepdim=True) - c[:, 1:]) / (c.shape[1] - 1)
    advantage = c[:, 1:] - baseline
    return (advantage * mean_logp[:, 1:]).mean(), {
        'advantage_abs': float(advantage.abs().mean())}
