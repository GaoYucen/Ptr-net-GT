import unittest

import torch

from states.component_merge_state import ComponentMergeState


class TestComponentMergeMasks(unittest.TestCase):
    def test_only_component_end_can_be_tail_and_start_can_be_head(self):
        coords = torch.tensor([
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        ], dtype=torch.float)
        state = ComponentMergeState.initialize(coords)
        state = state.update(torch.tensor([0]), torch.tensor([1]))
        mask = state.get_edge_mask()
        self.assertTrue(mask[0, 0].all().item())
        self.assertTrue(mask[0, :, 1].all().item())
        self.assertFalse(mask[0, 1, 2].item())


if __name__ == '__main__':
    unittest.main()