import unittest

import torch

from ptrnet_gt.training.supervised import orbit_logsumexp_loss


class TestSupervisedOrbitLoss(unittest.TestCase):
    def test_orbit_logsumexp_loss_matches_manual_probability_mass(self):
        probs = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.1, 0.2, 0.2],
        ], dtype=torch.float32)
        log_p = probs.log()
        orbit_mask = torch.tensor([
            [False, True, True, False],
            [True, False, False, True],
        ])

        loss = orbit_logsumexp_loss(log_p, orbit_mask)
        expected = -torch.log(torch.tensor([0.5, 0.7], dtype=torch.float32)).mean()

        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_orbit_logsumexp_loss_rejects_empty_orbit(self):
        log_p = torch.log_softmax(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32), dim=1)
        orbit_mask = torch.tensor([[False, False, False]])

        with self.assertRaisesRegex(ValueError, "empty"):
            orbit_logsumexp_loss(log_p, orbit_mask)


if __name__ == "__main__":
    unittest.main()