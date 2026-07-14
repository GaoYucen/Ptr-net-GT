import unittest

import torch

from ptrnet_gt.group_theory.group_action import act_on_instance


class TestGroupAction(unittest.TestCase):
    def test_act_on_instance(self):
        x = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        perm = torch.tensor([2, 0, 1])
        y = act_on_instance(x, perm)
        self.assertTrue(torch.equal(y[0, 0], x[0, 2]))


if __name__ == "__main__":
    unittest.main()