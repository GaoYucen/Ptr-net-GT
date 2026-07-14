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


if __name__ == "__main__":
    unittest.main()