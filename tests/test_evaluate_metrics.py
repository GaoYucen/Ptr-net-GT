import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate import _maybe_compute_permutation_metrics


class _ProbeStubModel:
    def __init__(self):
        self.called = False


class TestEvaluateMetricsHelpers(unittest.TestCase):
    def test_skip_permutation_probe_returns_none_metrics_without_running_model(self):
        dataset = TensorDataset(torch.rand(2, 4, 2))
        dataloader = DataLoader(dataset, batch_size=2)
        model = _ProbeStubModel()

        metrics = _maybe_compute_permutation_metrics(
            model,
            dataloader,
            torch.device("cpu"),
            enabled=False,
        )

        self.assertEqual(
            metrics,
            {
                "permutation_exact_successor_consistency": None,
                "permutation_edge_jaccard": None,
                "permutation_edge_recall": None,
                "permutation_relative_length_diff": None,
                "permutation_action_prob_equiv_error": None,
                "permutation_consistency": None,
            },
        )
        self.assertFalse(model.called)


if __name__ == "__main__":
    unittest.main()