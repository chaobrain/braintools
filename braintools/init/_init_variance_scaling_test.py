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
Tests for variance scaling weight initialization strategies.
"""

import unittest

import numpy as np

from braintools.init import (
    VarianceScaling,
    KaimingUniform,
    KaimingNormal,
    XavierUniform,
    XavierNormal,
    LecunUniform,
    LecunNormal,
)


class TestKaimingUniform(unittest.TestCase):
    """Test Kaiming uniform initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_shape(self):
        init = KaimingUniform()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

    def test_variance_fan_in(self):
        # He/Kaiming: variance = 2 / fan_in (the gain for ReLU is sqrt(2), so the
        # variance multiplier is 2). VarianceScaling computes variance = scale/fan,
        # so the effective scale must be 2.0, not sqrt(2). (bug C1)
        init = KaimingUniform(mode='fan_in')
        weights = init((1000, 100), rng=self.rng)
        fan_in = 1000
        expected_var = 2.0 / fan_in
        # For uniform distribution U(-a, a), variance = a^2/3 = scale/fan.
        actual_var = np.var(weights)
        # Allow some tolerance due to finite sample size
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_variance_fan_out(self):
        init = KaimingUniform(mode='fan_out')
        weights = init((100, 1000), rng=self.rng)
        fan_out = 1000
        expected_var = 2.0 / fan_out
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_variance_fan_avg(self):
        init = KaimingUniform(mode='fan_avg')
        weights = init((100, 200), rng=self.rng)
        fan_avg = (100 + 200) / 2
        expected_var = 2.0 / fan_avg
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_leaky_relu(self):
        init = KaimingUniform(nonlinearity='leaky_relu', negative_slope=0.01)
        weights = init((1000, 100), rng=self.rng)
        self.assertEqual(weights.shape, (1000, 100))
        # Leaky-ReLU gain^2 = 2 / (1 + slope^2); variance = gain^2 / fan_in.
        fan_in = 1000
        expected_var = (2.0 / (1 + 0.01 ** 2)) / fan_in
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_repr(self):
        init = KaimingUniform()
        repr_str = repr(init)
        self.assertIn('KaimingUniform', repr_str)


class TestKaimingNormal(unittest.TestCase):
    """Test Kaiming normal initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_shape(self):
        init = KaimingNormal()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

    def test_variance_fan_in(self):
        init = KaimingNormal(mode='fan_in')
        weights = init((1000, 100), rng=self.rng)
        fan_in = 1000
        expected_var = 2.0 / fan_in
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_mean(self):
        init = KaimingNormal()
        weights = init((1000, 100), rng=self.rng)
        # Mean should be close to 0
        self.assertAlmostEqual(np.mean(weights), 0.0, delta=0.01)

    def test_repr(self):
        init = KaimingNormal()
        repr_str = repr(init)
        self.assertIn('KaimingNormal', repr_str)


class TestXavierUniform(unittest.TestCase):
    """Test Xavier uniform initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_shape(self):
        init = XavierUniform()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

    def test_variance(self):
        # Xavier uses fan_avg by default
        init = XavierUniform()
        weights = init((1000, 500), rng=self.rng)
        fan_avg = (1000 + 500) / 2
        expected_var = 1.0 / fan_avg
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.1)

    def test_scale(self):
        init = XavierUniform(scale=2.0)
        weights = init((1000, 500), rng=self.rng)
        fan_avg = (1000 + 500) / 2
        expected_var = 2.0 / fan_avg
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_repr(self):
        init = XavierUniform(scale=1.5)
        repr_str = repr(init)
        self.assertIn('XavierUniform', repr_str)
        self.assertIn('1.5', repr_str)


class TestXavierNormal(unittest.TestCase):
    """Test Xavier normal initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_shape(self):
        init = XavierNormal()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

    def test_variance(self):
        init = XavierNormal()
        weights = init((1000, 500), rng=self.rng)
        fan_avg = (1000 + 500) / 2
        expected_var = 1.0 / fan_avg
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.1)

    def test_mean(self):
        init = XavierNormal()
        weights = init((1000, 500), rng=self.rng)
        self.assertAlmostEqual(np.mean(weights), 0.0, delta=0.01)

    def test_repr(self):
        init = XavierNormal()
        repr_str = repr(init)
        self.assertIn('XavierNormal', repr_str)


class TestLecunUniform(unittest.TestCase):
    """Test LeCun uniform initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_shape(self):
        init = LecunUniform()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

    def test_variance(self):
        # LeCun uses fan_in by default
        init = LecunUniform()
        weights = init((1000, 500), rng=self.rng)
        fan_in = 1000
        expected_var = 1.0 / fan_in
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.1)

    def test_scale(self):
        init = LecunUniform(scale=1.5)
        weights = init((1000, 500), rng=self.rng)
        fan_in = 1000
        expected_var = 1.5 / fan_in
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_repr(self):
        init = LecunUniform()
        repr_str = repr(init)
        self.assertIn('LecunUniform', repr_str)


class TestLecunNormal(unittest.TestCase):
    """Test LeCun normal initialization."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_shape(self):
        init = LecunNormal()
        weights = init((100, 50), rng=self.rng)
        self.assertEqual(weights.shape, (100, 50))

    def test_variance(self):
        init = LecunNormal()
        weights = init((1000, 500), rng=self.rng)
        fan_in = 1000
        expected_var = 1.0 / fan_in
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.1)

    def test_mean(self):
        init = LecunNormal()
        weights = init((1000, 500), rng=self.rng)
        self.assertAlmostEqual(np.mean(weights), 0.0, delta=0.01)

    def test_repr(self):
        init = LecunNormal()
        repr_str = repr(init)
        self.assertIn('LecunNormal', repr_str)


class TestTruncatedNormalDistribution(unittest.TestCase):
    """The truncated_normal distribution must preserve the target variance (bug H11)."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_variance_matches_target(self):
        # A real truncated normal at +-2 stddev loses ~23% variance; the sampler
        # must compensate (divide stddev by 0.8796...) so the achieved variance
        # equals the target scale/fan, not ~0.77x of it.
        init = VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        weights = np.asarray(init((2000, 100), rng=self.rng))
        fan_in = 2000
        expected_var = 2.0 / fan_in
        actual_var = np.var(weights)
        # 200k samples => sampling error well under 1%. The uncorrected
        # clip-at-2-sigma implementation lands ~8% low and must fail this.
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.04)

    def test_samples_are_truncated(self):
        # No sample should fall outside +-2 * stddev_underlying.
        init = VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        weights = np.asarray(init((2000, 100), rng=self.rng))
        variance = 2.0 / 2000
        stddev_underlying = np.sqrt(variance) / 0.8796256610342398
        bound = 2.0 * stddev_underlying
        self.assertTrue(np.all(np.abs(weights) <= bound + 1e-6))

    def test_mean_is_zero(self):
        init = VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        weights = np.asarray(init((2000, 100), rng=self.rng))
        self.assertAlmostEqual(float(np.mean(weights)), 0.0, delta=0.01)


class TestVarianceScalingValidation(unittest.TestCase):
    """mode / distribution must be validated eagerly (bug M6)."""

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            VarianceScaling(mode='not_a_mode')

    def test_invalid_distribution_raises(self):
        with self.assertRaises(ValueError):
            VarianceScaling(distribution='not_a_distribution')

    def test_valid_modes_accepted(self):
        for mode in ('fan_in', 'fan_out', 'fan_avg'):
            VarianceScaling(mode=mode)

    def test_valid_distributions_accepted(self):
        for dist in ('uniform', 'normal', 'truncated_normal'):
            VarianceScaling(distribution=dist)


class TestVarianceScalingFanConventions(unittest.TestCase):
    """The documented dimensionality-dependent fan convention must hold."""

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_scalar_shape_uses_unit_fan(self):
        # A 0-D request degenerates to fan_in = fan_out = 1.
        init = VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
        sample = init((), rng=self.rng)
        self.assertEqual(np.asarray(sample.mantissa if hasattr(sample, 'mantissa') else sample).shape, ())

    def test_1d_shape_fan_in_equals_fan_out(self):
        # For a 1-D shape both fans equal the single dimension, so fan_in and
        # fan_out modes target the same variance (= scale / n).
        n = 4000
        expected_var = 2.0 / n
        var_in = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')(n, rng=self.rng)
        var_out = VarianceScaling(scale=2.0, mode='fan_out', distribution='normal')(n, rng=self.rng)
        self.assertAlmostEqual(float(np.var(var_in)), expected_var, delta=expected_var * 0.2)
        self.assertAlmostEqual(float(np.var(var_out)), expected_var, delta=expected_var * 0.2)

    def test_conv_kernel_fan_uses_receptive_field(self):
        # 3D+ uses the conv convention: fan_in = in_ch * prod(spatial).
        # shape (out, in, kh, kw) = (8, 4, 3, 3): fan_in = 4*9 = 36.
        init = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
        weights = init((8, 4, 3, 3), rng=self.rng)
        expected_var = 2.0 / 36.0
        self.assertAlmostEqual(float(np.var(weights)), expected_var, delta=expected_var * 0.15)

    def test_repr_contains_config(self):
        r = repr(VarianceScaling(scale=2.0, mode='fan_avg', distribution='uniform'))
        self.assertIn('VarianceScaling', r)
        self.assertIn('fan_avg', r)
        self.assertIn('uniform', r)


class TestKaimingNonlinearityValidation(unittest.TestCase):
    """Kaiming initializers must reject unsupported nonlinearities (bug C1 path)."""

    def test_kaiming_uniform_rejects_unknown_nonlinearity(self):
        with self.assertRaises(ValueError):
            KaimingUniform(nonlinearity='gelu')

    def test_kaiming_normal_rejects_unknown_nonlinearity(self):
        with self.assertRaises(ValueError):
            KaimingNormal(nonlinearity='gelu')

    def test_kaiming_normal_leaky_relu_scale(self):
        # leaky_relu scale = 2 / (1 + slope**2); variance ~ scale / fan.
        rng = np.random.default_rng(0)
        slope = 0.2
        fan = 2000
        init = KaimingNormal(mode='fan_in', nonlinearity='leaky_relu', negative_slope=slope)
        weights = init((fan, 1), rng=rng)
        expected_var = (2.0 / (1 + slope ** 2)) / fan
        self.assertAlmostEqual(float(np.var(weights)), expected_var, delta=expected_var * 0.1)


if __name__ == '__main__':
    unittest.main()
