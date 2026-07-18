import unittest

import torch

from ptrnet_gt.models.sym_nco_encoder import SymNCOGraphEncoder


class TestSymNCOEncoder(unittest.TestCase):
    def test_encoder_shape_and_backward(self):
        encoder = SymNCOGraphEncoder(embedding_dim=32, n_heads=4, n_layers=2)
        x = torch.rand(3, 7, 2, requires_grad=True)
        h, g = encoder(x)
        self.assertEqual(h.shape, (3, 7, 32))
        self.assertEqual(g.shape, (3, 32))
        (h.mean() + g.mean()).backward()
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()