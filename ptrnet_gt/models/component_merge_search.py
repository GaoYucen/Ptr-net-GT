from __future__ import annotations

from dataclasses import dataclass

import torch

from ptrnet_gt.group_theory.canonicalization import canonical_selected_edge_key, canonicalize_components
from ptrnet_gt.states import ComponentMergeState


@dataclass
class BeamCandidate:
    state: ComponentMergeState
    log_prob: float
    sequence: list[int]


def _clone_sequence(sequence: list[int], tail_idx: int, head_idx: int) -> list[int]:
    return [*sequence, tail_idx, head_idx]


def _beam_search_single_encoded(model, coords: torch.Tensor, embeddings: torch.Tensor, beam_size: int = 1, dedup: bool = False) -> dict:
    device = coords.device
    batch = coords.unsqueeze(0)
    initial = BeamCandidate(state=ComponentMergeState.initialize(batch), log_prob=0.0, sequence=[])
    beams = [initial]
    duplicate_edge_counts = []
    duplicate_component_counts = []
    unique_edge_states_per_step = []
    unique_component_states_per_step = []

    while not beams[0].state.all_finished():
        expanded = []
        for candidate in beams:
            state = candidate.state
            log_p = model.get_joint_edge_log_p(state, embeddings)[0]
            edge_mask = state.get_edge_mask().view(-1)
            valid_idx = torch.nonzero(torch.isfinite(log_p) & (~edge_mask), as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                continue
            ranked_idx = valid_idx[torch.argsort(log_p[valid_idx], descending=True)]
            n_nodes = batch.size(1)
            branch_count = 0
            for edge_idx in ranked_idx.tolist():
                tail_idx = int(edge_idx // n_nodes)
                head_idx = int(edge_idx % n_nodes)
                if bool(state.get_edge_mask()[0, tail_idx, head_idx].item()):
                    continue
                next_state = state.update(
                    torch.tensor([tail_idx], device=device),
                    torch.tensor([head_idx], device=device),
                )
                expanded.append(
                    BeamCandidate(
                        state=next_state,
                        log_prob=candidate.log_prob + float(log_p[edge_idx].item()),
                        sequence=_clone_sequence(candidate.sequence, tail_idx, head_idx),
                    )
                )
                branch_count += 1
                if branch_count >= beam_size:
                    break

        if not expanded:
            raise RuntimeError("Beam search produced no valid expansions")

        edge_seen = {}
        component_seen = {}
        for cand in expanded:
            edge_key = canonical_selected_edge_key(cand.state)
            component_key = canonicalize_components(cand.state)
            edge_seen.setdefault(edge_key, 0)
            edge_seen[edge_key] += 1
            component_seen.setdefault(component_key, 0)
            component_seen[component_key] += 1
        duplicate_edge_counts.append(sum(v - 1 for v in edge_seen.values() if v > 1))
        duplicate_component_counts.append(sum(v - 1 for v in component_seen.values() if v > 1))
        unique_edge_states_per_step.append(len(edge_seen))
        unique_component_states_per_step.append(len(component_seen))

        if dedup:
            best_by_key = {}
            for cand in expanded:
                key = canonical_selected_edge_key(cand.state)
                prev = best_by_key.get(key)
                if prev is None or cand.log_prob > prev.log_prob:
                    best_by_key[key] = cand
            expanded = list(best_by_key.values())

        expanded.sort(key=lambda cand: cand.log_prob, reverse=True)
        beams = expanded[:beam_size]

    best = beams[0]
    pi = torch.tensor(best.sequence, device=device, dtype=torch.long).unsqueeze(0)
    cost = best.state.get_final_cost()
    return {
        "cost": cost,
        "pi": pi,
        "duplicate_edge_state_rate": float(sum(duplicate_edge_counts) / max(sum(unique_edge_states_per_step) + sum(duplicate_edge_counts), 1)),
        "duplicate_component_state_rate": float(sum(duplicate_component_counts) / max(sum(unique_component_states_per_step) + sum(duplicate_component_counts), 1)),
        "unique_edge_states_per_step": unique_edge_states_per_step,
        "unique_component_states_per_step": unique_component_states_per_step,
    }


def component_merge_beam_search(model, coords: torch.Tensor, beam_size: int = 1, dedup: bool = False) -> dict:
    if coords.dim() != 2:
        raise ValueError(f"Expected single-instance coords with shape [n_nodes, 2], got {tuple(coords.shape)}")
    embeddings = model.encode(coords.unsqueeze(0))
    return _beam_search_single_encoded(model, coords, embeddings, beam_size=beam_size, dedup=dedup)


def component_merge_beam_search_batched(model, coords_batch: torch.Tensor, beam_size: int = 1, dedup: bool = False) -> dict:
    if coords_batch.dim() != 3:
        raise ValueError(f"Expected batched coords with shape [batch, n_nodes, 2], got {tuple(coords_batch.shape)}")
    embeddings_batch = model.encode(coords_batch)
    batch_costs = []
    batch_pis = []
    duplicate_edge_state_rates = []
    duplicate_component_state_rates = []
    unique_edge_states = []
    unique_component_states = []

    for idx in range(coords_batch.size(0)):
        result = _beam_search_single_encoded(
            model,
            coords_batch[idx],
            embeddings_batch[idx:idx + 1],
            beam_size=beam_size,
            dedup=dedup,
        )
        batch_costs.append(result["cost"])
        batch_pis.append(result["pi"])
        duplicate_edge_state_rates.append(result["duplicate_edge_state_rate"])
        duplicate_component_state_rates.append(result["duplicate_component_state_rate"])
        unique_edge_states.extend(result["unique_edge_states_per_step"])
        unique_component_states.extend(result["unique_component_states_per_step"])

    return {
        "cost": torch.cat(batch_costs, dim=0),
        "pi": torch.cat(batch_pis, dim=0),
        "duplicate_edge_state_rates": duplicate_edge_state_rates,
        "duplicate_component_state_rates": duplicate_component_state_rates,
        "unique_edge_states_per_step": unique_edge_states,
        "unique_component_states_per_step": unique_component_states,
    }


__all__ = ["component_merge_beam_search", "component_merge_beam_search_batched"]