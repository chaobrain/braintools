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

from braintools._deprecation import BraintoolsDeprecationWarning, registered_deprecations
from braintools.init import (
    Orthogonal,
    DeltaOrthogonal,
    Identity,
)


def _f64(weights):
    """Materialize weights as float64 for accurate orthogonality checks.

    Matrices are generated in JAX's default float32 (x64 disabled), so the
    matrix entries carry ~1e-7 error. Forming ``W @ W.T`` in float32 then
    *accumulates* rounding to ~1e-4, which spuriously fails a 1e-6 tolerance even
    though the matrix is genuinely orthogonal. Computing the product in float64
    keeps the verification error at the true ~3e-7 level.
    """
    return np.asarray(weights, dtype=np.float64)


class TestOrthogonal(unittest.TestCase):
    """Test Orthogonal initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_square_matrix(self):
        init = Orthogonal()
        weights = init((100, 100), rng=self.rng)
        self.assertEqual(weights.shape, (100, 100))

        # Check orthogonality: W @ W.T should be identity
        w = _f64(weights)
        product = w @ w.T
        identity = np.eye(100)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_rectangular_matrix_rows_gt_cols(self):
        init = Orthogonal()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

        # Check semi-orthogonality: W.T @ W should be identity
        w = _f64(weights)
        product = w.T @ w
        identity = np.eye(50)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_rectangular_matrix_cols_gt_rows(self):
        init = Orthogonal()
        weights = init((50, 100), rng=self.rng)
        self.assertEqual(weights.shape, (50, 100))

        # Check semi-orthogonality: W @ W.T should be identity
        w = _f64(weights)
        product = w @ w.T
        identity = np.eye(50)
        np.testing.assert_allclose(product, identity, atol=1e-6)

    def test_gain(self):
        gain = 2.0
        init = Orthogonal(scale=gain)
        weights = init((100, 100), rng=self.rng)

        # Check that norm scaling is correct
        # For orthogonal matrix with scale, W @ W.T = scale^2 * I
        w = _f64(weights)
        product = w @ w.T
        identity = np.eye(100) * (gain ** 2)
        np.testing.assert_allclose(product, identity, atol=1e-5)

    def test_3d_shape(self):
        # Test with 3D shape
        init = Orthogonal()
        weights = init((10, 20, 30), rng=self.rng)
        self.assertEqual(weights.shape, (10, 20, 30))

        # Reshape to 2D for orthogonality check
        weights_2d = _f64(weights).reshape(-1, 30)
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
    """Test DeltaOrthogonal initialization (channels-last ``(*spatial, in, out)``)."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_conv_kernel_shape(self):
        # brainstate conv kernels are (height, width, in_channels, out_channels)
        init = DeltaOrthogonal()
        weights = init((3, 3, 32, 64), rng=self.rng)
        self.assertEqual(weights.shape, (3, 3, 32, 64))

    def test_center_is_orthogonal(self):
        init = DeltaOrthogonal()
        weights = init((3, 3, 32, 64), rng=self.rng)
        # in <= out: orthonormal rows, so y = x @ W preserves norms
        center = _f64(weights[1, 1])
        np.testing.assert_allclose(center @ center.T, np.eye(32), atol=1e-6)

    def test_center_is_orthogonal_contracting(self):
        init = DeltaOrthogonal()
        weights = init((3, 3, 64, 32), rng=self.rng)
        center = _f64(weights[1, 1])
        np.testing.assert_allclose(center.T @ center, np.eye(32), atol=1e-6)

    def test_non_center_is_zero(self):
        init = DeltaOrthogonal()
        weights = init((3, 3, 32, 64), rng=self.rng)

        # All values except center should be zero
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue  # Skip center
                np.testing.assert_allclose(weights[i, j], 0.0, atol=1e-10)

    def test_1d_conv(self):
        # 1D conv kernel (width, in, out); center at index 2
        init = DeltaOrthogonal()
        weights = init((5, 32, 64), rng=self.rng)
        self.assertEqual(weights.shape, (5, 32, 64))
        center = _f64(weights[2])
        np.testing.assert_allclose(center @ center.T, np.eye(32), atol=1e-6)
        np.testing.assert_allclose(_f64(weights)[[0, 1, 3, 4]], 0.0, atol=1e-10)

    def test_gain(self):
        gain = 2.0
        init = DeltaOrthogonal(scale=gain)
        weights = init((3, 3, 32, 64), rng=self.rng)
        center = _f64(weights[1, 1])
        np.testing.assert_allclose(center @ center.T, np.eye(32) * gain ** 2, atol=1e-5)

    def test_matches_brainstate_conv_kernel(self):
        # The kernel shape brainstate.nn.Conv2d asks its w_init for.
        import brainstate
        conv = brainstate.nn.Conv2d((8, 8, 4), 16, kernel_size=3, w_init=DeltaOrthogonal())
        w = _f64(conv.weight.value['weight'])
        self.assertEqual(w.shape, (3, 3, 4, 16))
        np.testing.assert_allclose(w[1, 1] @ w[1, 1].T, np.eye(4), atol=1e-6)

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


class TestUnitDeprecationWarning(unittest.TestCase):
    """The deprecated ``unit=`` arg must warn at the caller's stack frame (bug D4)."""

    def test_orthogonal_unit_warns_at_caller(self):
        import warnings
        import brainunit as u
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            Orthogonal(unit=u.nS)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, BraintoolsDeprecationWarning)
        # stacklevel=2 attributes the warning to this test module, not the
        # library source file.
        self.assertTrue(caught[0].filename.endswith('_init_orthogonal_test.py'))

    def test_delta_orthogonal_unit_warns_at_caller(self):
        import warnings
        import brainunit as u
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            DeltaOrthogonal(unit=u.nS)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, BraintoolsDeprecationWarning)
        self.assertTrue(caught[0].filename.endswith('_init_orthogonal_test.py'))

    def test_records_registered_and_used(self):
        import warnings
        import brainunit as u
        records = {r.old_name: r for r in registered_deprecations()}
        for cls in (Orthogonal, DeltaOrthogonal):
            with self.subTest(cls=cls.__name__):
                record = records[f'braintools.init.{cls.__name__}(unit=...)']
                self.assertEqual(record.kind, 'argument')
                self.assertEqual(record.since, '0.1.0')
                self.assertEqual(record.removed_in, '1.0.0')
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    init = cls(unit=u.nS)
                self.assertEqual([str(w.message) for w in caught], [record.message])
                self.assertEqual(init.unit, u.nS)

    def test_no_warning_without_unit(self):
        import warnings
        import brainunit as u
        for cls in (Orthogonal, DeltaOrthogonal):
            with self.subTest(cls=cls.__name__):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    init = cls(scale=2.0)
                self.assertEqual(caught, [])
                self.assertEqual(init.unit, u.UNITLESS)

    def test_deprecated_unit_still_applied(self):
        import warnings
        import brainunit as u
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', BraintoolsDeprecationWarning)
            weights = Orthogonal(unit=u.nS)((4, 4), rng=np.random.default_rng(0))
        self.assertEqual(u.get_unit(weights), u.nS)


if __name__ == '__main__':
    unittest.main()
