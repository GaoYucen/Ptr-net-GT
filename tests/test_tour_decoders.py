import unittest

import torch

from ptrnet_gt.models import AttentionModelDecoder, PointerNetworkDecoder
from ptrnet_gt.problems.tsp import TSP


class TestTourDecoders(unittest.TestCase):
    def test_pointer_network_forward_shapes(self):
        model = PointerNetworkDecoder(128, 128, TSP())
        model.set_decode_type("greedy")
        x = torch.rand(2, 5, 2)
        cost, ll, pi = model(x, return_pi=True)
        self.assertEqual(cost.shape, (2,))
        self.assertEqual(ll.shape, (2,))
        self.assertEqual(pi.shape, (2, 5))

    def test_attention_model_forward_shapes(self):
        model = AttentionModelDecoder(128, 128, TSP())
        model.set_decode_type("greedy")
        x = torch.rand(2, 5, 2)
        cost, ll, pi = model(x, return_pi=True)
        self.assertEqual(cost.shape, (2,))
        self.assertEqual(ll.shape, (2,))
        self.assertEqual(pi.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()