import unittest

import torch

from ptrnet_gt.models.component_merge_decoder import ComponentMergeDecoder
from ptrnet_gt.problems.tsp import TSP
from ptrnet_gt.utils import evaluate_tour_batch

class TestCostInvariance(unittest.TestCase):
    def test_model_cost_matches_recomputed_edge_cost(self):
        torch.manual_seed(1234)
        model = ComponentMergeDecoder(128, 128, TSP(), context_mode="none")
        model.set_decode_type("greedy")
        x = torch.rand(4, 6, 2)

        cost, _, pi = model(x, return_pi=True)
        metrics = evaluate_tour_batch(x, pi, model_cost=cost)

        self.assertTrue(metrics["feasible"].all())
        self.assertTrue(torch.allclose(cost, metrics["edge_cost"], atol=1e-6))
        self.assertLessEqual(float(metrics["max_cost_error"].item()), 1e-6)


if __name__ == "__main__":
    unittest.main()