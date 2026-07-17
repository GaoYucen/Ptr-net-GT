from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from ptrnet_gt.baselines.nearest_neighbor import nearest_neighbor_multistart_tour
from ptrnet_gt.group_theory.permutation import inverse_permutation, permute_nodes
from ptrnet_gt.states import ComponentMergeState
from ptrnet_gt.utils import move_to


@dataclass
class SupervisedEpochResult:
    avg_loss: float
    avg_cost: float


def _tour_to_edge_pairs(tour: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    heads = torch.roll(tour, shifts=-1, dims=1)
    return tour, heads


def _target_edge_matrix(tails: torch.Tensor, heads: torch.Tensor, n_nodes: int) -> torch.Tensor:
    batch = tails.size(0)
    target = torch.zeros(batch, n_nodes, n_nodes, dtype=torch.bool, device=tails.device)
    batch_idx = torch.arange(batch, device=tails.device)[:, None].expand_as(tails)
    target[batch_idx, tails, heads] = True
    return target


def _permute_edge_logits_back(flat_log_p: torch.Tensor, permutation: torch.Tensor, n_nodes: int) -> torch.Tensor:
    inv = inverse_permutation(permutation)
    matrix = flat_log_p.view(n_nodes, n_nodes)
    return matrix[inv][:, inv].reshape(-1)


def _masked_prob_mse(log_p: torch.Tensor, perm_log_p: torch.Tensor, permutation: torch.Tensor, n_nodes: int) -> torch.Tensor:
    perm_back = _permute_edge_logits_back(perm_log_p, permutation, n_nodes)
    p = log_p.exp()
    q = perm_back.exp()
    valid = torch.isfinite(log_p) & torch.isfinite(perm_back)
    if valid.any():
        p = p * valid.float()
        q = q * valid.float()
        p = p / p.sum().clamp_min(1e-8)
        q = q / q.sum().clamp_min(1e-8)
        return ((p - q) ** 2).sum()
    return torch.zeros((), device=log_p.device)


def train_supervised_epoch(model, optimizer, epoch, problem, opts, objective="fixed_order"):
    model.train()
    model.set_decode_type("greedy")
    dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    loader = DataLoader(dataset, batch_size=opts.batch_size, num_workers=0)
    losses = []
    costs = []
    consistency_weight = float(getattr(opts, "consistency_weight", 0.0))
    for batch in loader:
        x = move_to(batch, opts.device)
        tour = nearest_neighbor_multistart_tour(x)
        tails, heads = _tour_to_edge_pairs(tour)
        target_edges = _target_edge_matrix(tails, heads, x.size(1))
        embeddings = model.encode(x)
        state = ComponentMergeState.initialize(x)
        step_losses = []
        consistency_losses = []
        permuted_x = None
        perm_embeddings = None
        perm_state = None
        perms = None
        if consistency_weight > 0:
            perms = torch.stack([torch.randperm(x.size(1), device=x.device) for _ in range(x.size(0))], dim=0)
            permuted_x = torch.stack([permute_nodes(x[i], perms[i]) for i in range(x.size(0))], dim=0)
            perm_embeddings = model.encode(permuted_x)
            perm_state = ComponentMergeState.initialize(permuted_x)
        for step in range(x.size(1)):
            log_p = model.get_joint_edge_log_p(state, embeddings)
            if objective == "fixed_order":
                edge_idx = tails[:, step] * x.size(1) + heads[:, step]
                step_losses.append(-log_p.gather(1, edge_idx[:, None]).mean())
                state = state.update(tails[:, step], heads[:, step])
                if consistency_weight > 0:
                    perm_tails = perms[torch.arange(x.size(0), device=x.device), tails[:, step]]
                    perm_heads = perms[torch.arange(x.size(0), device=x.device), heads[:, step]]
                    perm_log_p = model.get_joint_edge_log_p(perm_state, perm_embeddings)
                    perm_idx = perm_tails * x.size(1) + perm_heads
                    consistency_losses.append(torch.stack([
                        _masked_prob_mse(log_p[i], perm_log_p[i], perms[i], x.size(1))
                        for i in range(x.size(0))
                    ]).mean())
                    perm_state = perm_state.update(perm_tails, perm_heads)
            elif objective == "equiv_set":
                legal_target = (target_edges & (~state.selected_edges) & (~state.get_edge_mask())).view(x.size(0), -1)
                masked = log_p.masked_fill(~legal_target, float("-inf"))
                step_losses.append(-torch.logsumexp(masked, dim=1).mean())
                next_idx = legal_target.float().argmax(dim=1)
                state = state.update_edge(next_idx)
                if consistency_weight > 0:
                    perm_log_p = model.get_joint_edge_log_p(perm_state, perm_embeddings)
                    consistency_losses.append(torch.stack([
                        _masked_prob_mse(log_p[i], perm_log_p[i], perms[i], x.size(1))
                        for i in range(x.size(0))
                    ]).mean())
                    perm_legal_target = torch.zeros_like(legal_target)
                    for i in range(x.size(0)):
                        perm_matrix = target_edges[i][perms[i]][:, perms[i]]
                        perm_legal = (perm_matrix & (~perm_state.selected_edges[i]) & (~perm_state.get_edge_mask()[i])).view(-1)
                        perm_legal_target[i] = perm_legal
                    perm_next_idx = perm_legal_target.float().argmax(dim=1)
                    perm_state = perm_state.update_edge(perm_next_idx)
            else:
                raise ValueError(f"Unknown supervised objective: {objective}")
        loss = torch.stack(step_losses).mean()
        if consistency_losses:
            loss = loss + consistency_weight * torch.stack(consistency_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        costs.append(state.get_final_cost().detach().mean())
    return SupervisedEpochResult(avg_loss=torch.stack(losses).mean().item(), avg_cost=torch.stack(costs).mean().item())
