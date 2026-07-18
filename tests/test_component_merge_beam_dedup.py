import unittest

import torch

from ptrnet_gt.group_theory.canonicalization import canonical_selected_edge_key, canonicalize_components
from ptrnet_gt.models.component_merge_search import component_merge_beam_search
from ptrnet_gt.states import ComponentMergeState


class _DummyBeamModel:
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def get_joint_edge_log_p(self, state, embeddings):
        n = state.n_nodes
        logits = torch.full((1, n * n), float("-inf"), device=embeddings.device)
        valid = (~state.get_edge_mask()).view(1, -1)
        logits[valid] = -10.0

        preferred_edges = {
            0: [(0, 1), (2, 3)],
            1: [(1, 2), (3, 0)],
            2: [(1, 2), (3, 0)],
            3: [(3, 0)],
        }
        for tail, head in preferred_edges.get(int(state.step.item()), []):
            logits[0, tail * n + head] = 0.0
        return torch.log_softmax(logits, dim=-1)


class TestCanonicalizationAndBeamDedup(unittest.TestCase):
    def setUp(self):
        self.coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=torch.float32)

    def test_canonical_keys_match_for_commuting_edge_orders(self):
        state_a = ComponentMergeState.initialize(self.coords.unsqueeze(0))
        state_a = state_a.update(torch.tensor([0]), torch.tensor([1]))
        state_a = state_a.update(torch.tensor([2]), torch.tensor([3]))

        state_b = ComponentMergeState.initialize(self.coords.unsqueeze(0))
        state_b = state_b.update(torch.tensor([2]), torch.tensor([3]))
        state_b = state_b.update(torch.tensor([0]), torch.tensor([1]))

        self.assertEqual(canonical_selected_edge_key(state_a), canonical_selected_edge_key(state_b))
        self.assertEqual(canonicalize_components(state_a), canonicalize_components(state_b))

    def test_beam_search_runs_and_reports_duplicate_metrics(self):
        model = _DummyBeamModel(n_nodes=4)
        result = component_merge_beam_search(model, self.coords, beam_size=2, dedup=True)
        self.assertEqual(result["pi"].shape, (1, 8))
        self.assertEqual(result["cost"].shape, (1,))
        self.assertIn("duplicate_edge_state_rate", result)
        self.assertIn("dedup_retention_rate", result)
        self.assertIn("expanded_candidates_per_step", result)
        self.assertIn("kept_candidates_per_step", result)
        self.assertIn("unique_edge_states_per_step", result)
        self.assertGreaterEqual(result["duplicate_edge_state_rate"], 0.0)
        self.assertGreaterEqual(result["dedup_retention_rate"], 0.0)
        self.assertLessEqual(result["dedup_retention_rate"], 1.0)
        self.assertEqual(len(result["unique_edge_states_per_step"]), 4)
        self.assertEqual(len(result["expanded_candidates_per_step"]), 4)
        self.assertEqual(len(result["kept_candidates_per_step"]), 4)


if __name__ == "__main__":
    unittest.main()