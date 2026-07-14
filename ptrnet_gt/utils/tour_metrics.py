import torch


def edges_from_pi(pi: torch.Tensor):
    tails = pi[:, 0::2]
    heads = pi[:, 1::2]
    return tails, heads


def edge_lengths(coords: torch.Tensor, tails: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    batch_size, n_edges = tails.size()
    tail_coords = torch.gather(coords, 1, tails[:, :, None].expand(batch_size, n_edges, coords.size(-1)))
    head_coords = torch.gather(coords, 1, heads[:, :, None].expand(batch_size, n_edges, coords.size(-1)))
    return (tail_coords - head_coords).norm(p=2, dim=-1)


def build_successor(tails: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    batch_size, n_nodes = tails.size()
    succ = torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=tails.device)
    batch_idx = torch.arange(batch_size, device=tails.device)[:, None].expand(batch_size, n_nodes)
    succ[batch_idx, tails] = heads
    return succ


def reconstruct_tour_from_succ(succ: torch.Tensor, start_node: int = 0) -> torch.Tensor:
    batch_size, n_nodes = succ.size()
    batch_idx = torch.arange(batch_size, device=succ.device)
    current = torch.full((batch_size,), start_node, dtype=torch.long, device=succ.device)
    tour = torch.full((batch_size, n_nodes), -1, dtype=torch.long, device=succ.device)
    for step in range(n_nodes):
        tour[:, step] = current
        next_node = succ[batch_idx, current]
        current = torch.where(next_node >= 0, next_node, current)
    return tour


def evaluate_tour_batch(coords: torch.Tensor, pi: torch.Tensor, model_cost: torch.Tensor | None = None) -> dict:
    tails, heads = edges_from_pi(pi)
    batch_size, n_nodes, _ = coords.size()

    out_degree = torch.zeros(batch_size, n_nodes, dtype=torch.long, device=coords.device)
    in_degree = torch.zeros(batch_size, n_nodes, dtype=torch.long, device=coords.device)
    ones = torch.ones_like(tails)
    out_degree.scatter_add_(1, tails, ones)
    in_degree.scatter_add_(1, heads, ones)

    succ = build_successor(tails, heads)
    tour = reconstruct_tour_from_succ(succ, start_node=0)
    visited_once = torch.zeros(batch_size, n_nodes, dtype=torch.bool, device=coords.device)
    visited_once.scatter_(1, tour.clamp_min(0), True)

    unique_visit_count = visited_once.sum(dim=1)
    all_out_one = (out_degree == 1).all(dim=1)
    all_in_one = (in_degree == 1).all(dim=1)
    complete_visit = unique_visit_count == n_nodes
    closes_cycle = succ[torch.arange(batch_size, device=coords.device), tour[:, -1]] == tour[:, 0]
    feasible = all_out_one & all_in_one & complete_visit & closes_cycle

    edge_cost = edge_lengths(coords, tails, heads).sum(dim=1)

    if model_cost is None:
        model_cost = edge_cost
    cost_error = (model_cost - edge_cost).abs()

    return {
        "tails": tails,
        "heads": heads,
        "tour": tour,
        "successor": succ,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "feasible": feasible,
        "feasible_rate": feasible.float().mean(),
        "complete_visit": complete_visit,
        "closes_cycle": closes_cycle,
        "edge_cost": edge_cost,
        "cost_error": cost_error,
        "max_cost_error": cost_error.max(),
        "mean_cost_error": cost_error.mean(),
    }
