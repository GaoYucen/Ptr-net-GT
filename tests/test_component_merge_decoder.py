import unittest

import torch

from ptrnet_gt.models.component_merge_decoder import ComponentMergeDecoder
from ptrnet_gt.problems.tsp import TSP
from ptrnet_gt.group_theory.permutation import inverse_permutation, permute_nodes
from ptrnet_gt.states import ComponentMergeState


class TestComponentMergeDecoder(unittest.TestCase):
    def test_forward_shapes(self):
        model = ComponentMergeDecoder(128, 128, TSP(), context_mode="none")
        model.set_decode_type("greedy")
        x = torch.rand(2, 5, 2)
        cost, ll, pi = model(x, return_pi=True)
        self.assertEqual(cost.shape, (2,))
        self.assertEqual(ll.shape, (2,))
        self.assertEqual(pi.shape, (2, 10))

    def test_forward_shapes_joint_edge(self):
        model = ComponentMergeDecoder(128, 128, TSP(), context_mode="none", action_mode="joint_edge")
        model.set_decode_type("greedy")
        x = torch.rand(2, 5, 2)
        cost, ll, pi = model(x, return_pi=True)
        self.assertEqual(cost.shape, (2,))
        self.assertEqual(ll.shape, (2,))
        self.assertEqual(pi.shape, (2, 10))

    def test_joint_edge_produces_valid_tour_edges(self):
        model = ComponentMergeDecoder(128, 128, TSP(), context_mode="none", action_mode="joint_edge")
        model.set_decode_type("greedy")
        x = torch.rand(2, 5, 2)
        _, _, pi = model(x, return_pi=True)
        tails = pi[:, 0::2]
        heads = pi[:, 1::2]
        self.assertTrue(torch.all(torch.sort(tails, dim=1)[0] == torch.arange(5).expand_as(tails)))
        self.assertTrue(torch.all(torch.sort(heads, dim=1)[0] == torch.arange(5).expand_as(heads)))

    def test_joint_edge_role_projection_forward_shapes(self):
        model = ComponentMergeDecoder(
            128,
            128,
            TSP(),
            context_mode="none",
            action_mode="joint_edge",
            use_dynamic_role_features=True,
            use_role_feature_projection=True,
            use_distance_projection=True,
            role_embedding_dim=32,
            distance_embedding_dim=16,
        )
        model.set_decode_type("greedy")
        x = torch.rand(2, 5, 2)
        cost, ll, pi = model(x, return_pi=True)
        self.assertEqual(cost.shape, (2,))
        self.assertEqual(ll.shape, (2,))
        self.assertEqual(pi.shape, (2, 10))

    def test_joint_edge_single_step_log_probs_are_permutation_equivariant(self):
        torch.manual_seed(1234)
        model = ComponentMergeDecoder(
            64,
            64,
            TSP(),
            context_mode="none",
            action_mode="joint_edge",
            use_dynamic_role_features=True,
            use_role_feature_projection=True,
            use_distance_projection=True,
            role_embedding_dim=16,
            distance_embedding_dim=8,
        )
        model.eval()
        x = torch.rand(1, 6, 2)
        perm = torch.randperm(6)
        x_perm = permute_nodes(x[0], perm).unsqueeze(0)
        emb = model.encode(x)
        emb_perm = model.encode(x_perm)
        state = ComponentMergeState.initialize(x)
        state_perm = ComponentMergeState.initialize(x_perm)
        log_p = model.get_joint_edge_log_p(state, emb)[0]
        log_p_perm = model.get_joint_edge_log_p(state_perm, emb_perm)[0]
        inv = inverse_permutation(perm)
        back = log_p_perm.view(6, 6)[inv][:, inv].reshape(-1)
        valid = torch.isfinite(log_p) & torch.isfinite(back)
        self.assertTrue(valid.any())
        self.assertLess((log_p[valid].exp() - back[valid].exp()).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()