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
Tests for basic weight initialization distributions.
"""

import unittest
import warnings

import brainstate
import brainunit as u
import numpy as np

from braintools.init import (
    Constant,
    ZeroInit,
    Uniform,
    Normal,
    LogNormal,
    Gamma,
    Exponential,
    TruncatedNormal,
    Beta,
    Weibull,
)


class TestZeroInit(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_zero_values_with_unit(self):
        """Test that all values are zero with specified unit."""
        init = ZeroInit(u.siemens)
        weights = init(100, rng=self.rng)
        self.assertEqual(weights.shape, (100,))
        self.assertTrue(np.all(weights == 0.0 * u.siemens))
        self.assertEqual(weights.unit, u.siemens)

    def test_zero_values_unitless(self):
        """Test that all values are zero without unit."""
        init = ZeroInit()
        weights = init(100, rng=self.rng)
        self.assertEqual(weights.shape, (100,))
        self.assertTrue(np.all(weights == 0.0))

    def test_zero_with_tuple_size(self):
        """Test zero initialization with multi-dimensional size."""
        init = ZeroInit(u.siemens)
        weights = init((10, 20), rng=self.rng)
        self.assertEqual(weights.shape, (10, 20))
        self.assertTrue(np.all(weights == 0.0 * u.siemens))

    def test_zero_with_different_units(self):
        """Test zero initialization with different units."""
        init = ZeroInit(u.mS)
        weights = init(50, rng=self.rng)
        self.assertEqual(weights.shape, (50,))
        self.assertTrue(np.all(weights == 0.0 * u.mS))
        self.assertEqual(weights.unit, u.mS)

    def test_inheritance_from_constant(self):
        """Test that ZeroInit inherits from Constant correctly."""
        init = ZeroInit(u.siemens)
        self.assertIsInstance(init, Constant)
        self.assertEqual(init.value, 0.0)
        self.assertEqual(init.unit, u.siemens)

    def test_repr(self):
        """Test string representation."""
        init = ZeroInit(u.siemens)
        repr_str = repr(init)
        self.assertIn('ZeroInit', repr_str)
        self.assertIn('S', repr_str)  # siemens is represented as 'S'

        init_unitless = ZeroInit()
        repr_str_unitless = repr(init_unitless)
        self.assertIn('ZeroInit', repr_str_unitless)


class TestConstant(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_constant_value(self):
        init = Constant(0.5 * u.siemens)
        weights = init(100, rng=self.rng)
        self.assertEqual(weights.shape, (100,))
        self.assertTrue(np.all(weights == 0.5 * u.siemens))

    def test_constant_with_tuple_size(self):
        init = Constant(1.0 * u.siemens)
        weights = init((10, 20), rng=self.rng)
        self.assertEqual(weights.shape, (10, 20))
        self.assertTrue(np.all(weights == 1.0 * u.siemens))

    def test_repr(self):
        init = Constant(0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Constant', repr_str)
        self.assertIn('0.5', repr_str)


class TestUniform(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_uniform_distribution(self):
        init = Uniform(0.1 * u.siemens, 1.0 * u.siemens)
        weights = init(10000, rng=self.rng)
        self.assertEqual(weights.shape, (10000,))
        self.assertTrue(np.all(weights >= 0.1 * u.siemens))
        self.assertTrue(np.all(weights < 1.0 * u.siemens))

    def test_uniform_statistics(self):
        init = Uniform(0.0 * u.siemens, 1.0 * u.siemens)
        weights = init(100000, rng=self.rng)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, 0.5, delta=0.01)

    def test_repr(self):
        init = Uniform(0.1 * u.siemens, 1.0 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Uniform', repr_str)


class TestNormal(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_normal_distribution(self):
        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(100000, rng=self.rng)
        self.assertEqual(weights.shape, (100000,))

    def test_normal_statistics(self):
        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(100000, rng=self.rng)
        mean = np.mean(weights.mantissa)
        std = np.std(weights.mantissa)
        self.assertAlmostEqual(mean, 0.5, delta=0.01)
        self.assertAlmostEqual(std, 0.1, delta=0.01)

    def test_repr(self):
        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Normal', repr_str)


class TestLogNormal(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_lognormal_positive(self):
        init = LogNormal(0.5 * u.siemens, 0.2 * u.siemens)
        weights = init(1000, rng=self.rng)
        self.assertTrue(np.all(weights > 0 * u.siemens))

    def test_lognormal_statistics(self):
        init = LogNormal(1.0 * u.siemens, 0.5 * u.siemens)
        weights = init(100000, rng=self.rng)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, 1.0, delta=0.05)

    def test_repr(self):
        init = LogNormal(0.5 * u.siemens, 0.2 * u.siemens)
        repr_str = repr(init)
        self.assertIn('LogNormal', repr_str)


class TestGamma(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_gamma_positive(self):
        init = Gamma(shape=2.0, scale=0.5 * u.siemens)
        weights = init(1000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0 * u.siemens))

    def test_gamma_statistics(self):
        shape = 2.0
        scale = 0.5
        init = Gamma(shape=shape, scale=scale * u.siemens)
        weights = init(100000, rng=self.rng)
        expected_mean = shape * scale
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, expected_mean, delta=0.05)

    def test_repr(self):
        init = Gamma(shape=2.0, scale=0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Gamma', repr_str)


class TestExponential(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_exponential_positive(self):
        init = Exponential(0.5 * u.siemens)
        weights = init(1000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0 * u.siemens))

    def test_exponential_statistics(self):
        scale = 0.5
        init = Exponential(scale * u.siemens)
        weights = init(100000, rng=self.rng)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, scale, delta=0.01)

    def test_repr(self):
        init = Exponential(0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Exponential', repr_str)


class TestTruncatedNormal(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_truncated_normal_bounds(self):
        try:
            init = TruncatedNormal(
                mean=0.5 * u.siemens,
                std=0.2 * u.siemens,
                low=0.0 * u.siemens,
                high=1.0 * u.siemens
            )
            weights = init(1000, rng=self.rng)
            self.assertTrue(np.all(weights >= 0.0 * u.siemens))
            self.assertTrue(np.all(weights <= 1.0 * u.siemens))
        except ImportError:
            self.skipTest("scipy not installed")

    def test_truncated_normal_statistics(self):
        try:
            init = TruncatedNormal(
                mean=0.5 * u.siemens,
                std=0.1 * u.siemens,
                low=0.0 * u.siemens,
                high=1.0 * u.siemens
            )
            weights = init(100000, rng=self.rng)
            mean = np.mean(weights.mantissa)
            self.assertAlmostEqual(mean, 0.5, delta=0.05)
        except ImportError:
            self.skipTest("scipy not installed")

    def test_scipy_import_error(self):
        init = TruncatedNormal(
            mean=0.5 * u.siemens,
            std=0.2 * u.siemens
        )
        try:
            import scipy
            weights = init(100, rng=self.rng)
            self.assertEqual(weights.shape, (100,))
        except ImportError:
            with self.assertRaises(ImportError):
                init(100, rng=self.rng)

    def test_repr(self):
        init = TruncatedNormal(0.5 * u.siemens, 0.2 * u.siemens)
        repr_str = repr(init)
        self.assertIn('TruncatedNormal', repr_str)


class TestBeta(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_beta_bounds(self):
        init = Beta(
            alpha=2.0,
            beta=5.0,
            low=0.0 * u.siemens,
            high=1.0 * u.siemens
        )
        weights = init(1000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_beta_statistics(self):
        alpha, beta = 2.0, 5.0
        init = Beta(
            alpha=alpha,
            beta=beta,
            low=0.0 * u.siemens,
            high=1.0 * u.siemens
        )
        weights = init(100000, rng=self.rng)
        expected_mean = alpha / (alpha + beta)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, expected_mean, delta=0.01)

    def test_repr(self):
        init = Beta(2.0, 5.0, 0.0 * u.siemens, 1.0 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Beta', repr_str)


class TestWeibull(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_weibull_positive(self):
        init = Weibull(shape=1.5, scale=0.5 * u.siemens)
        weights = init(1000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0 * u.siemens))

    def test_repr(self):
        init = Weibull(1.5, 0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Weibull', repr_str)


class TestParameterValidation(unittest.TestCase):
    """Validation of distribution parameters at construction time."""

    def test_normal_negative_std_raises(self):
        """Normal must reject a negative standard deviation (bug H9)."""
        with self.assertRaises(ValueError):
            Normal(0.5 * u.siemens, -0.1 * u.siemens)

    def test_normal_zero_std_allowed(self):
        """A zero std is a degenerate but valid distribution."""
        init = Normal(0.5 * u.siemens, 0.0 * u.siemens)
        weights = init(10, rng=np.random.default_rng(0))
        self.assertTrue(np.allclose(weights.mantissa, 0.5))

    def test_lognormal_negative_std_raises(self):
        """LogNormal must reject a negative standard deviation (bug H9)."""
        with self.assertRaises(ValueError):
            LogNormal(0.5 * u.siemens, -0.2 * u.siemens)

    def test_lognormal_nonpositive_mean_raises(self):
        """LogNormal mean must be strictly positive (bug M1)."""
        with self.assertRaises(ValueError):
            LogNormal(0.0 * u.siemens, 0.2 * u.siemens)
        with self.assertRaises(ValueError):
            LogNormal(-1.0 * u.siemens, 0.2 * u.siemens)

    def test_uniform_low_ge_high_raises(self):
        """Uniform requires low < high."""
        with self.assertRaises(ValueError):
            Uniform(1.0 * u.siemens, 0.5 * u.siemens)

    def test_beta_low_ge_high_raises(self):
        """Beta requires low < high."""
        with self.assertRaises(ValueError):
            Beta(2.0, 5.0, low=1.0 * u.siemens, high=1.0 * u.siemens)

    def test_beta_nonpositive_alpha_raises(self):
        """Beta alpha and beta must be positive."""
        with self.assertRaises(ValueError):
            Beta(-2.0, 5.0, low=0.0 * u.siemens, high=1.0 * u.siemens)
        with self.assertRaises(ValueError):
            Beta(2.0, 0.0, low=0.0 * u.siemens, high=1.0 * u.siemens)

    def test_truncated_normal_low_ge_high_raises(self):
        """TruncatedNormal requires low < high when both are given."""
        with self.assertRaises(ValueError):
            TruncatedNormal(0.5 * u.siemens, 0.2 * u.siemens,
                            low=1.0 * u.siemens, high=0.0 * u.siemens)

    def test_truncated_normal_negative_std_raises(self):
        """TruncatedNormal must reject a negative std."""
        with self.assertRaises(ValueError):
            TruncatedNormal(0.5 * u.siemens, -0.2 * u.siemens)

    def test_gamma_nonpositive_shape_raises(self):
        """Gamma shape must be positive."""
        with self.assertRaises(ValueError):
            Gamma(shape=-1.0, scale=0.5 * u.siemens)

    def test_gamma_dimensionful_shape_raises(self):
        """Gamma shape must be dimensionless."""
        with self.assertRaises(ValueError):
            Gamma(shape=2.0 * u.siemens, scale=0.5 * u.siemens)

    def test_weibull_nonpositive_shape_raises(self):
        """Weibull shape must be positive."""
        with self.assertRaises(ValueError):
            Weibull(shape=0.0, scale=0.5 * u.siemens)

    def test_constant_none_value_raises(self):
        """Constant must reject a None value (bug L8)."""
        with self.assertRaises((ValueError, TypeError)):
            Constant(None)


class TestDeprecatedUnit(unittest.TestCase):
    """Behaviour of the deprecated ``unit=`` argument (bug M2)."""

    def test_deprecated_unit_with_bare_value_warns_and_applies(self):
        """A bare value plus deprecated unit still works, with a warning."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            with self.assertRaises(DeprecationWarning):
                Constant(0.5, unit=u.nS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            init = Constant(0.5, unit=u.nS)
            weights = init(10)
        self.assertEqual(weights.unit, u.nS)
        self.assertTrue(np.allclose(weights.to(u.nS).mantissa, 0.5))

    def test_deprecated_unit_with_united_value_raises(self):
        """Combining a deprecated unit with an already-united value raises."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with self.assertRaises(ValueError):
                Constant(0.5 * u.mS, unit=u.nS)

    def test_weibull_accepts_unit_kwarg(self):
        """Weibull supports the deprecated ``unit=`` like its siblings (bug L6)."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            init = Weibull(shape=1.5, scale=0.5, unit=u.nS)
            weights = init(100, rng=np.random.default_rng(0))
        self.assertEqual(weights.unit, u.nS)


class TestTruncatedNormalBackend(unittest.TestCase):
    """TruncatedNormal must be backend-agnostic (bug H10)."""

    def test_default_rng_works(self):
        """Default rng (brainstate.random) must work without scipy crash."""
        init = TruncatedNormal(mean=0.5 * u.siemens, std=0.2 * u.siemens,
                               low=0.0 * u.siemens, high=1.0 * u.siemens)
        weights = init(1000)
        self.assertEqual(weights.shape, (1000,))
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_brainstate_rng_explicit(self):
        """An explicit brainstate.random backend must not crash (bug H10)."""
        init = TruncatedNormal(mean=0.5 * u.siemens, std=0.2 * u.siemens,
                               low=0.0 * u.siemens, high=1.0 * u.siemens)
        weights = init(1000, rng=brainstate.random)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_numpy_rng_explicit(self):
        """A numpy Generator backend must still work."""
        rng = np.random.default_rng(0)
        init = TruncatedNormal(mean=0.5 * u.siemens, std=0.2 * u.siemens,
                               low=0.0 * u.siemens, high=1.0 * u.siemens)
        weights = init(1000, rng=rng)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_one_sided_bounds(self):
        """Only a lower bound is honoured; upper tail is unbounded."""
        rng = np.random.default_rng(0)
        init = TruncatedNormal(mean=0.0 * u.siemens, std=1.0 * u.siemens,
                               low=0.0 * u.siemens)
        weights = init(10000, rng=rng)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))

    def test_statistics(self):
        """Truncated-normal mean is correct via inverse-CDF sampling."""
        rng = np.random.default_rng(0)
        init = TruncatedNormal(mean=0.5 * u.siemens, std=0.1 * u.siemens,
                               low=0.0 * u.siemens, high=1.0 * u.siemens)
        weights = init(100000, rng=rng)
        self.assertAlmostEqual(float(np.mean(weights.mantissa)), 0.5, delta=0.02)


class TestTruncatedNormalArrayParams(unittest.TestCase):
    """TruncatedNormal must accept array-valued ``mean``/``std`` (bug B1).

    A scalar ``if std == 0`` previously raised "truth value of an array ...
    is ambiguous" for per-element ``std``.
    """

    def test_array_std_does_not_crash(self):
        rng = np.random.default_rng(0)
        init = TruncatedNormal(
            mean=np.array([0.0, 1.0]) * u.mV,
            std=np.array([0.5, 0.2]) * u.mV,
            low=-1.0 * u.mV, high=2.0 * u.mV,
        )
        weights = init(2, rng=rng)
        self.assertEqual(weights.shape, (2,))
        self.assertTrue(np.all(weights >= -1.0 * u.mV))
        self.assertTrue(np.all(weights <= 2.0 * u.mV))

    def test_array_std_with_zero_element_is_degenerate(self):
        """A zero entry in ``std`` collapses that element to the (clamped) mean."""
        rng = np.random.default_rng(0)
        init = TruncatedNormal(
            mean=np.array([0.5, 3.0]) * u.mV,
            std=np.array([0.0, 0.2]) * u.mV,
            low=0.0 * u.mV, high=1.0 * u.mV,
        )
        weights = init(2, rng=rng)
        # std==0 with in-range mean -> exactly the mean.
        self.assertAlmostEqual(float(weights.mantissa[0]), 0.5, places=6)
        # std==0 path must not perturb the second (non-zero std) element's bounds.
        self.assertTrue(0.0 <= float(weights.mantissa[1]) <= 1.0)

    def test_scalar_std_zero_still_degenerate(self):
        """Scalar std==0 keeps the original degenerate behaviour (all == mean)."""
        rng = np.random.default_rng(0)
        init = TruncatedNormal(0.5 * u.mV, 0.0 * u.mV, low=0.0 * u.mV, high=1.0 * u.mV)
        weights = init(5, rng=rng)
        self.assertTrue(np.allclose(weights.mantissa, 0.5))

    def test_scalar_std_zero_mean_out_of_range_is_clamped(self):
        rng = np.random.default_rng(0)
        init = TruncatedNormal(5.0 * u.mV, 0.0 * u.mV, low=0.0 * u.mV, high=1.0 * u.mV)
        weights = init(3, rng=rng)
        self.assertTrue(np.allclose(weights.mantissa, 1.0))

    def test_array_mean_and_std_statistics(self):
        """Per-element parametrization keeps each element near its own mean."""
        rng = np.random.default_rng(0)
        means = np.array([0.2, 0.8]) * u.mV
        init = TruncatedNormal(mean=means, std=np.array([0.05, 0.05]) * u.mV,
                               low=0.0 * u.mV, high=1.0 * u.mV)
        weights = init((100000, 2), rng=rng)
        col_means = np.mean(weights.mantissa, axis=0)
        self.assertAlmostEqual(float(col_means[0]), 0.2, delta=0.02)
        self.assertAlmostEqual(float(col_means[1]), 0.8, delta=0.02)


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_zero_size(self):
        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(0, rng=self.rng)
        self.assertEqual(len(weights), 0)

    def test_large_size(self):
        init = Constant(0.5 * u.siemens)
        weights = init(1000000, rng=self.rng)
        self.assertEqual(len(weights), 1000000)

    def test_tuple_size(self):
        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init((10, 20, 30), rng=self.rng)
        self.assertEqual(weights.shape, (10, 20, 30))

    def test_different_units(self):
        init = Uniform(100.0 * u.uS, 1000.0 * u.uS)
        weights = init(100, rng=self.rng)
        self.assertTrue(np.all(weights >= 100.0 * u.uS))
        self.assertTrue(np.all(weights < 1000.0 * u.uS))

    def test_unit_consistency(self):
        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(100, rng=self.rng)
        self.assertEqual(weights.unit, u.siemens)


if __name__ == '__main__':
    unittest.main()
