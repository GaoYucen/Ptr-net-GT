import unittest

import torch

from ptrnet_gt.models.component_merge_decoder import ComponentMergeDecoder
from ptrnet_gt.problems.tsp import TSP


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


if __name__ == "__main__":
    unittest.main()