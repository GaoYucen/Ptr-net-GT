import unittest

import torch

from states.component_merge_state import ComponentMergeState


class TestComponentMergeState(unittest.TestCase):
    def setUp(self):
        self.coords = torch.tensor([
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ], dtype=torch.float)

    def test_initialize_mask_disallows_self_loops(self):
        state = ComponentMergeState.initialize(self.coords)
        mask = state.get_edge_mask()
        self.assertTrue(mask[0].diag().all().item())

    def test_tail_cannot_be_reused(self):
        state = ComponentMergeState.initialize(self.coords)
        state = state.update(torch.tensor([0]), torch.tensor([1]))
        mask = state.get_edge_mask()
        self.assertTrue(mask[0, 0].all().item())

    def test_head_cannot_be_reused(self):
        state = ComponentMergeState.initialize(self.coords)
        state = state.update(torch.tensor([0]), torch.tensor([1]))
        mask = state.get_edge_mask()
        self.assertTrue(mask[0, :, 1].all().item())

    def test_no_early_subtour(self):
        state = ComponentMergeState.initialize(self.coords)
        state = state.update(torch.tensor([0]), torch.tensor([1]))
        with self.assertRaises(ValueError):
            state.update(torch.tensor([1]), torch.tensor([0]))

    def test_last_step_closes_single_cycle(self):
        state = ComponentMergeState.initialize(self.coords)
        state = state.update(torch.tensor([0]), torch.tensor([1]))
        state = state.update(torch.tensor([1]), torch.tensor([2]))
        state = state.update(torch.tensor([2]), torch.tensor([3]))
        mask = state.get_edge_mask()
        self.assertFalse(mask[0, 3, 0].item())
        self.assertTrue(mask[0].sum().item() == self.coords.size(1) ** 2 - 1)
        state = state.update(torch.tensor([3]), torch.tensor([0]))
        self.assertTrue(state.all_finished())

    def test_to_tour_and_cost(self):
        state = ComponentMergeState.initialize(self.coords)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for tail, head in edges:
            state = state.update(torch.tensor([tail]), torch.tensor([head]))
        tour = state.to_tour()
        self.assertTrue(torch.equal(tour[0], torch.tensor([0, 1, 2, 3])))
        self.assertAlmostEqual(state.get_final_cost().item(), 4.0, places=5)

    def test_batch_support(self):
        coords = self.coords.repeat(2, 1, 1)
        state = ComponentMergeState.initialize(coords)
        state = state.update(torch.tensor([0, 0]), torch.tensor([1, 2]))
        self.assertEqual(state.step.item(), 1)
        self.assertTrue(state.get_edge_mask()[0, 1, 0].item())
        self.assertTrue(state.get_edge_mask()[1, 2, 0].item())
        self.assertFalse(state.get_edge_mask()[0, 1, 2].item())
        self.assertFalse(state.get_edge_mask()[1, 2, 3].item())


if __name__ == '__main__':
    unittest.main()