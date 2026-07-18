import unittest

import torch

from ptrnet_gt.group_theory.geometric_augmentation import augment_xy_data_by_8_fold, augment_xy_data_by_n_fold


class TestGeometricAugmentation(unittest.TestCase):
    def test_augment_8_fold_shape(self):
        x = torch.rand(3, 5, 2)
        y = augment_xy_data_by_8_fold(x)
        self.assertEqual(y.shape, (24, 5, 2))

    def test_augment_n_fold_identity(self):
        x = torch.rand(2, 4, 2)
        y = augment_xy_data_by_n_fold(x, 1)
        self.assertTrue(torch.allclose(x, y))


if __name__ == "__main__":
    unittest.main()