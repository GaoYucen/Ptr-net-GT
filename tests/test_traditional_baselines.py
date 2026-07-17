import unittest

import torch

from ptrnet_gt.baselines import (
    nearest_neighbor_multistart_tour,
    nearest_neighbor_tour,
    nearest_neighbor_two_opt_tour,
    random_tour,
    tour_length,
    two_opt_tour,
)


class TestTraditionalBaselines(unittest.TestCase):
    def setUp(self):
        self.coords = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            ],
            dtype=torch.float,
        )

    def assert_is_permutation(self, tour):
        n = tour.size(1)
        expected = torch.arange(n, device=tour.device)
        self.assertTrue(torch.equal(tour.sort(dim=1)[0], expected.expand_as(tour)))

    def test_random_tour_is_valid_permutation(self):
        tour = random_tour(self.coords, generator=torch.Generator().manual_seed(1234))
        self.assert_is_permutation(tour)

    def test_nearest_neighbor_is_valid_permutation(self):
        tour = nearest_neighbor_tour(self.coords, start_node=0)
        self.assert_is_permutation(tour)

    def test_multistart_not_worse_than_single_start(self):
        single = nearest_neighbor_tour(self.coords, start_node=0)
        multistart = nearest_neighbor_multistart_tour(self.coords)
        self.assert_is_permutation(multistart)
        self.assertTrue(torch.all(tour_length(self.coords, multistart) <= tour_length(self.coords, single) + 1e-12))

    def test_two_opt_does_not_worsen_tour(self):
        initial = torch.tensor([[0, 2, 1, 3], [0, 2, 1, 3]], dtype=torch.long)
        improved = two_opt_tour(self.coords, initial)
        self.assert_is_permutation(improved)
        self.assertTrue(torch.all(tour_length(self.coords, improved) <= tour_length(self.coords, initial) + 1e-12))

    def test_nn_two_opt_is_valid_permutation(self):
        tour = nearest_neighbor_two_opt_tour(self.coords)
        self.assert_is_permutation(tour)


if __name__ == "__main__":
    unittest.main()