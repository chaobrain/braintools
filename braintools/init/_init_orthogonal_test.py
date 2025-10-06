# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Tests for orthogonal weight initialization strategies.
"""

import unittest

import numpy as np

from braintools.init import (
    Orthogonal,
    DeltaOrthogonal,
    Identity,
)


class TestOrthogonal(unittest.TestCase):
    """Test Orthogonal initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_square_matrix(self):
        init = Orthogonal()
        weights = init((100, 100), rng=self.rng)
        self.assertEqual(weights.shape, (100, 100))

        # Check orthogonality: W @ W.T should be identity
        product = weights @ weights.T
        identity = np.eye(100)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_rectangular_matrix_rows_gt_cols(self):
        init = Orthogonal()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

        # Check semi-orthogonality: W.T @ W should be identity
        product = weights.T @ weights
        identity = np.eye(50)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_rectangular_matrix_cols_gt_rows(self):
        init = Orthogonal()
        weights = init((50, 100), rng=self.rng)
        self.assertEqual(weights.shape, (50, 100))

        # Check semi-orthogonality: W @ W.T should be identity
        product = weights @ weights.T
        identity = np.eye(50)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_gain(self):
        gain = 2.0
        init = Orthogonal(scale=gain)
        weights = init((100, 100), rng=self.rng)

        # Check that norm scaling is correct
        # For orthogonal matrix with scale, W @ W.T = scale^2 * I
        product = weights @ weights.T
        identity = np.eye(100) * (gain ** 2)
        np.testing.assert_allclose(product, identity, atol=1e-5)

    def test_3d_shape(self):
        # Test with 3D shape
        init = Orthogonal()
        weights = init((10, 20, 30), rng=self.rng)
        self.assertEqual(weights.shape, (10, 20, 30))

        # Reshape to 2D for orthogonality check
        weights_2d = weights.reshape(-1, 30)
        product = weights_2d.T @ weights_2d
        identity = np.eye(30)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_invalid_shape(self):
        init = Orthogonal()
        with self.assertRaises(ValueError):
            init(100, rng=self.rng)  # 1D shape should raise error

    def test_repr(self):
        init = Orthogonal(scale=1.5)
        repr_str = repr(init)
        self.assertIn('Orthogonal', repr_str)
        self.assertIn('1.5', repr_str)


class TestDeltaOrthogonal(unittest.TestCase):
    """Test DeltaOrthogonal initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_conv_kernel_shape(self):
        # Test with typical conv kernel shape: (out_channels, in_channels, height, width)
        init = DeltaOrthogonal()
        weights = init((64, 32, 3, 3), rng=self.rng)
        self.assertEqual(weights.shape, (64, 32, 3, 3))

    def test_center_is_orthogonal(self):
        init = DeltaOrthogonal()
        weights = init((64, 32, 3, 3), rng=self.rng)

        # Extract center values (at index [1, 1] for 3x3 kernel)
        center = weights[:, :, 1, 1]

        # Check orthogonality
        if center.shape[0] >= center.shape[1]:
            product = center.T @ center
            identity = np.eye(32)
        else:
            product = center @ center.T
            identity = np.eye(64)

        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_non_center_is_zero(self):
        init = DeltaOrthogonal()
        weights = init((64, 32, 3, 3), rng=self.rng)

        # All values except center should be zero
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue  # Skip center
                np.testing.assert_allclose(weights[:, :, i, j], 0.0, atol=1e-10)

    def test_1d_conv(self):
        # Test with 1D conv kernel
        init = DeltaOrthogonal()
        weights = init((64, 32, 5), rng=self.rng)
        self.assertEqual(weights.shape, (64, 32, 5))

        # Center should be at index 2
        center = weights[:, :, 2]
        if center.shape[0] >= center.shape[1]:
            product = center.T @ center
            identity = np.eye(32)
        else:
            product = center @ center.T
            identity = np.eye(64)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_gain(self):
        gain = 2.0
        init = DeltaOrthogonal(scale=gain)
        weights = init((64, 32, 3, 3), rng=self.rng)

        center = weights[:, :, 1, 1]
        # Check scale scaling
        if center.shape[0] >= center.shape[1]:
            product = center.T @ center
            identity = np.eye(32) * (gain ** 2)
        else:
            product = center @ center.T
            identity = np.eye(64) * (gain ** 2)
        np.testing.assert_allclose(product, identity, atol=1e-5)

    def test_invalid_shape(self):
        init = DeltaOrthogonal()
        with self.assertRaises(ValueError):
            init((64, 32), rng=self.rng)  # 2D shape should raise error

    def test_repr(self):
        init = DeltaOrthogonal(scale=1.5)
        repr_str = repr(init)
        self.assertIn('DeltaOrthogonal', repr_str)
        self.assertIn('1.5', repr_str)


class TestIdentity(unittest.TestCase):
    """Test Identity initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_square_matrix(self):
        init = Identity()
        weights = init((100, 100))
        self.assertEqual(weights.shape, (100, 100))
        np.testing.assert_allclose(weights, np.eye(100), atol=1e-10)

    def test_rectangular_matrix(self):
        init = Identity()
        weights = init((100, 50))
        self.assertEqual(weights.shape, (100, 50))
        expected = np.eye(100, 50)
        np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_gain(self):
        gain = 2.5
        init = Identity(scale=gain)
        weights = init((50, 50))
        expected = np.eye(50) * gain
        np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_1d_shape(self):
        init = Identity(scale=3.0)
        weights = init(100)
        self.assertEqual(weights.shape, (100,))
        expected = np.ones(100) * 3.0
        np.testing.assert_allclose(weights, expected, atol=1e-10)

    def test_3d_shape(self):
        init = Identity()
        weights = init((10, 20, 30))
        self.assertEqual(weights.shape, (10, 20, 30))

        # Check that the last two dimensions have identity-like structure
        for i in range(10):
            expected = np.eye(20, 30)
            np.testing.assert_allclose(weights[i], expected, atol=1e-10)

    def test_repr(self):
        init = Identity(scale=2.0)
        repr_str = repr(init)
        self.assertIn('Identity', repr_str)
        self.assertIn('2.0', repr_str)


if __name__ == '__main__':
    unittest.main()
