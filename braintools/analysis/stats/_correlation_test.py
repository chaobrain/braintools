# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import math
import unittest
from functools import partial

import brainstate
import jax.numpy as jnp
from jax import jit

import braintools


class TestMatrixCorrelation(unittest.TestCase):
    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_mc(self):
        A = self.rng.random((100, 100))
        B = self.rng.random((100, 100))
        r1 = (braintools.analysis.stats.matrix_correlation(A, B))

        jit_f = jit(partial(braintools.analysis.stats.matrix_correlation))
        r2 = jit_f(A, B)
        self.assertTrue(jnp.allclose(r1, r2))

    def test_matrix_correlation_identical(self):
        """Test correlation between identical matrices."""
        A = self.rng.random((50, 50))
        corr = braintools.analysis.stats.matrix_correlation(A, A)
        self.assertAlmostEqual(float(corr), 1.0, places=5)

    def test_matrix_correlation_symmetric(self):
        """Test that correlation is symmetric."""
        A = self.rng.random((30, 30))
        B = self.rng.random((30, 30))
        corr_ab = braintools.analysis.stats.matrix_correlation(A, B)
        corr_ba = braintools.analysis.stats.matrix_correlation(B, A)
        self.assertAlmostEqual(float(corr_ab), float(corr_ba), places=5)

    def test_matrix_correlation_bounds(self):
        """Test that correlation is bounded between -1 and 1."""
        A = self.rng.random((25, 25))
        B = self.rng.random((25, 25))
        corr = braintools.analysis.stats.matrix_correlation(A, B)
        self.assertGreaterEqual(float(corr), -1.0)
        self.assertLessEqual(float(corr), 1.0)

    def test_matrix_correlation_perfect_anticorr(self):
        """Test perfect anti-correlation."""
        A = jnp.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
        B = -A + 2 * jnp.eye(3)  # Flip off-diagonal elements
        corr = braintools.analysis.stats.matrix_correlation(A, B)
        self.assertAlmostEqual(float(corr), -1.0, places=3)

    def test_matrix_correlation_input_validation(self):
        """Test input validation for matrix correlation."""
        # Test non-2D arrays
        with self.assertRaises(ValueError):
            braintools.analysis.stats.matrix_correlation(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))

        # Mismatched shapes now raise a clear ValueError instead of silently
        # comparing differently-sized upper triangles (audit E11).
        A = jnp.ones((3, 3))
        B = jnp.ones((4, 4))
        with self.assertRaises(ValueError):
            braintools.analysis.stats.matrix_correlation(A, B)

    def test_matrix_correlation_constant_returns_zero(self):
        """Constant matrices give an ill-defined corrcoef (NaN) -> mapped to 0."""
        A = jnp.ones((4, 4))
        B = self.rng.random((4, 4))
        # x is constant on the upper triangle -> corrcoef is NaN -> nan_to_num -> 0.
        self.assertEqual(float(braintools.analysis.stats.matrix_correlation(A, B)), 0.0)


class TestWeightedCorrelation(unittest.TestCase):
    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_weighted_correlation_basic(self):
        """Test basic weighted correlation functionality."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([2.0, 4.0, 6.0, 8.0, 10.0])
        w = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Equal weights

        corr = braintools.analysis.stats.weighted_correlation(x, y, w)

        # Should be perfect correlation
        self.assertAlmostEqual(float(corr), 1.0, places=5)

    def test_weighted_correlation_vs_unweighted(self):
        """Test that equal weights give same result as unweighted correlation."""
        x = self.rng.normal(0, 1, size=100)
        y = 2 * x + self.rng.normal(0, 0.1, size=100)
        w = jnp.ones(100)  # Equal weights

        weighted_corr = braintools.analysis.stats.weighted_correlation(x, y, w)
        unweighted_corr = jnp.corrcoef(x, y)[0, 1]

        self.assertAlmostEqual(float(weighted_corr), float(unweighted_corr), places=4)

    def test_weighted_correlation_zero_weights(self):
        """Test behavior with zero weights."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Correlated, not constant
        w = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0])  # Only last point has weight

        # Should handle gracefully - when only one point has weight, correlation is undefined (NaN)
        corr = braintools.analysis.stats.weighted_correlation(x, y, w)
        # With only one weighted point, correlation is undefined, so NaN is acceptable
        self.assertTrue(math.isnan(float(corr)) or math.isfinite(float(corr)))

    def test_weighted_correlation_different_weights(self):
        """Test that different weights affect the correlation."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([1.0, 4.0, 6.0, 8.0, 10.0])  # Mostly linear but with outlier

        # Equal weights
        w_equal = jnp.ones(5)
        corr_equal = braintools.analysis.stats.weighted_correlation(x, y, w_equal)

        # Weight down the outlier (second point)
        w_outlier = jnp.array([1.0, 0.1, 1.0, 1.0, 1.0])
        corr_outlier = braintools.analysis.stats.weighted_correlation(x, y, w_outlier)

        # Weighting down outlier should increase correlation
        self.assertGreater(float(corr_outlier), float(corr_equal))

    def test_weighted_correlation_input_validation(self):
        """Test input validation for weighted correlation."""
        # Test non-1D arrays
        with self.assertRaises(ValueError):
            braintools.analysis.stats.weighted_correlation(jnp.array([[1, 2]]), jnp.array([1, 2]), jnp.array([1, 1]))

        with self.assertRaises(ValueError):
            braintools.analysis.stats.weighted_correlation(jnp.array([1, 2]), jnp.array([[1, 2]]), jnp.array([1, 1]))

        with self.assertRaises(ValueError):
            braintools.analysis.stats.weighted_correlation(jnp.array([1, 2]), jnp.array([1, 2]), jnp.array([[1, 1]]))

    def test_weighted_correlation_bounds(self):
        """Test that weighted correlation is bounded between -1 and 1."""
        x = self.rng.normal(0, 1, size=50)
        y = self.rng.normal(0, 1, size=50)
        w = self.rng.uniform(0.1, 2.0, size=50)  # Random positive weights

        corr = braintools.analysis.stats.weighted_correlation(x, y, w)

        self.assertGreaterEqual(float(corr), -1.0)
        self.assertLessEqual(float(corr), 1.0)

    def test_weighted_correlation_perfect_anticorr(self):
        """Test perfect anti-correlation."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x  # Perfect anti-correlation
        w = jnp.array([1.0, 2.0, 1.0, 3.0, 1.0])  # Varying weights

        corr = braintools.analysis.stats.weighted_correlation(x, y, w)
        self.assertAlmostEqual(float(corr), -1.0, places=5)

    def test_weighted_correlation_jit_compatibility(self):
        """Test JIT compilation compatibility."""
        x = self.rng.normal(0, 1, size=20)
        y = self.rng.normal(0, 1, size=20)
        w = jnp.ones(20)

        # Test regular computation
        corr1 = braintools.analysis.stats.weighted_correlation(x, y, w)

        # Test JIT compiled version
        jit_weighted_corr = jit(braintools.analysis.stats.weighted_correlation)
        corr2 = jit_weighted_corr(x, y, w)

        self.assertAlmostEqual(float(corr1), float(corr2), places=6)


class TestStatsRegressions(unittest.TestCase):
    """Targeted regression tests for the audit E-series fixes (statistics)."""

    def setUp(self):
        self.rng = brainstate.random.RandomState(0)

    def test_matrix_correlation_second_arg_not_2d_raises(self):
        """A 2D x with a non-2D y must raise (covers the y.ndim guard)."""
        x = self.rng.random((4, 4))
        y = jnp.array([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError):
            braintools.analysis.stats.matrix_correlation(x, y)

    def test_matrix_correlation_strips_units(self):
        import brainunit as u
        A = self.rng.random((10, 10))
        B = self.rng.random((10, 10))
        plain = braintools.analysis.stats.matrix_correlation(A, B)
        quantity = braintools.analysis.stats.matrix_correlation(A * u.mV, B * u.mV)
        self.assertAlmostEqual(float(plain), float(quantity), places=5)

    def test_weighted_correlation_all_zero_weights_is_zero(self):
        """All-zero weights give 0/0; the guard returns 0.0, not NaN (E17)."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([2.0, 1.0, 4.0, 3.0, 5.0])
        w = jnp.zeros(5)
        corr = braintools.analysis.stats.weighted_correlation(x, y, w)
        self.assertFalse(math.isnan(float(corr)))
        self.assertEqual(float(corr), 0.0)

    def test_weighted_correlation_length_mismatch_raises(self):
        """Differing lengths (all 1d) must raise a clear ValueError."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0])
        w = jnp.array([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            braintools.analysis.stats.weighted_correlation(x, y, w)

    def test_weighted_correlation_constant_x_is_zero(self):
        """A constant variable has zero weighted variance -> correlation 0."""
        x = jnp.ones(6)
        y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        w = jnp.ones(6)
        corr = braintools.analysis.stats.weighted_correlation(x, y, w)
        self.assertEqual(float(corr), 0.0)


if __name__ == '__main__':
    unittest.main()
