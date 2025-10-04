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
        # For fan_in mode with ReLU, scale = sqrt(2), variance = scale/fan_in = sqrt(2)/fan_in
        init = KaimingUniform(mode='fan_in')
        weights = init((1000, 100), rng=self.rng)
        fan_in = 1000
        scale = np.sqrt(2.0)
        expected_var = scale / fan_in
        # For uniform distribution U(-a, a), variance = a^2/3
        # We have a = sqrt(3 * scale / fan_in)
        actual_var = np.var(weights)
        # Allow some tolerance due to finite sample size
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_variance_fan_out(self):
        init = KaimingUniform(mode='fan_out')
        weights = init((100, 1000), rng=self.rng)
        fan_out = 1000
        scale = np.sqrt(2.0)
        expected_var = scale / fan_out
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_variance_fan_avg(self):
        init = KaimingUniform(mode='fan_avg')
        weights = init((100, 200), rng=self.rng)
        fan_avg = (100 + 200) / 2
        scale = np.sqrt(2.0)
        expected_var = scale / fan_avg
        actual_var = np.var(weights)
        self.assertAlmostEqual(actual_var, expected_var, delta=expected_var * 0.2)

    def test_leaky_relu(self):
        init = KaimingUniform(nonlinearity='leaky_relu', negative_slope=0.01)
        weights = init((1000, 100), rng=self.rng)
        self.assertEqual(weights.shape, (1000, 100))
        # Variance should be different from ReLU
        fan_in = 1000
        scale = np.sqrt(2.0 / (1 + 0.01 ** 2))
        expected_var = scale / fan_in
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
        scale = np.sqrt(2.0)
        expected_var = scale / fan_in
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


if __name__ == '__main__':
    unittest.main()
