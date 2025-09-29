# Copyright 2025 BrainSim Ecosystem Limited. All Rights Reserved.
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
Comprehensive tests for delay initialization classes.

This test suite covers:
- Constant delay initialization
- Uniform delay distribution
- Normal delay distribution
- Gamma delay distribution
"""

import unittest

import brainunit as u
import numpy as np

from braintools.conn import (
    ConstantDelay,
    UniformDelay,
    NormalDelay,
    GammaDelay,
)


class TestDelayInitializations(unittest.TestCase):
    """
    Test delay initialization classes.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import ConstantDelay, UniformDelay

        const_init = ConstantDelay(2.0 * u.ms)
        rng = np.random.default_rng(0)
        delays = const_init(rng, 100)
        assert np.all(delays == 2.0 * u.ms)

        uniform_init = UniformDelay(1.0 * u.ms, 5.0 * u.ms)
        delays = uniform_init(rng, 1000)
        assert np.all((delays >= 1.0 * u.ms) & (delays < 5.0 * u.ms))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_constant_delay(self):
        init = ConstantDelay(2.0 * u.ms)
        delays = init(self.rng, 100)
        self.assertTrue(np.all(delays == 2.0 * u.ms))

    def test_uniform_delay(self):
        init = UniformDelay(1.0 * u.ms, 5.0 * u.ms)
        delays = init(self.rng, 10000)
        self.assertTrue(np.all(delays >= 1.0 * u.ms))
        self.assertTrue(np.all(delays < 5.0 * u.ms))

    def test_normal_delay(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms, min_delay=0.1 * u.ms)
        delays = init(self.rng, 1000)
        self.assertTrue(np.all(delays >= 0.1 * u.ms))

    def test_normal_delay_statistics(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms, min_delay=0.0 * u.ms)
        delays = init(self.rng, 100000)
        mean = np.mean(delays.mantissa)
        self.assertAlmostEqual(mean, 2.0, delta=0.05)

    def test_gamma_delay(self):
        init = GammaDelay(shape=2.0, scale=1.0 * u.ms)
        delays = init(self.rng, 1000)
        self.assertTrue(np.all(delays >= 0 * u.ms))

    def test_delay_repr(self):
        init = ConstantDelay(2.0 * u.ms)
        self.assertIn('ConstantDelay', repr(init))


class TestConstantDelay(unittest.TestCase):
    """
    Test ConstantDelay initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import ConstantDelay

        init = ConstantDelay(2.0 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 100)
        assert np.all(delays == 2.0 * u.ms)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_constant_value(self):
        init = ConstantDelay(2.0 * u.ms)
        delays = init(self.rng, 100)
        self.assertEqual(delays.shape, (100,))
        self.assertTrue(np.all(delays == 2.0 * u.ms))

    def test_constant_with_tuple_size(self):
        init = ConstantDelay(1.5 * u.ms)
        delays = init(self.rng, (10, 20))
        self.assertEqual(delays.shape, (10, 20))
        self.assertTrue(np.all(delays == 1.5 * u.ms))

    def test_repr(self):
        init = ConstantDelay(2.0 * u.ms)
        repr_str = repr(init)
        self.assertIn('ConstantDelay', repr_str)
        self.assertIn('2.0', repr_str)


class TestUniformDelay(unittest.TestCase):
    """
    Test UniformDelay initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import UniformDelay

        init = UniformDelay(1.0 * u.ms, 5.0 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 1000)
        assert np.all((delays >= 1.0 * u.ms) & (delays < 5.0 * u.ms))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_uniform_distribution(self):
        init = UniformDelay(1.0 * u.ms, 5.0 * u.ms)
        delays = init(self.rng, 10000)
        self.assertEqual(delays.shape, (10000,))
        self.assertTrue(np.all(delays >= 1.0 * u.ms))
        self.assertTrue(np.all(delays < 5.0 * u.ms))

    def test_uniform_statistics(self):
        init = UniformDelay(0.0 * u.ms, 10.0 * u.ms)
        delays = init(self.rng, 100000)
        mean = np.mean(delays.mantissa)
        self.assertAlmostEqual(mean, 5.0, delta=0.1)

    def test_repr(self):
        init = UniformDelay(1.0 * u.ms, 5.0 * u.ms)
        repr_str = repr(init)
        self.assertIn('UniformDelay', repr_str)


class TestNormalDelay(unittest.TestCase):
    """
    Test NormalDelay initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import NormalDelay

        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms, min_delay=0.1 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 1000)
        assert np.all(delays >= 0.1 * u.ms)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_normal_distribution(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms)
        delays = init(self.rng, 100000)
        self.assertEqual(delays.shape, (100000,))

    def test_normal_statistics(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms, min_delay=0.0 * u.ms)
        delays = init(self.rng, 100000)
        mean = np.mean(delays.mantissa)
        std = np.std(delays.mantissa)
        self.assertAlmostEqual(mean, 2.0, delta=0.05)
        self.assertAlmostEqual(std, 0.5, delta=0.05)

    def test_min_delay_clipping(self):
        init = NormalDelay(mean=2.0 * u.ms, std=5.0 * u.ms, min_delay=0.5 * u.ms)
        delays = init(self.rng, 10000)
        self.assertTrue(np.all(delays >= 0.5 * u.ms))

    def test_repr(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms)
        repr_str = repr(init)
        self.assertIn('NormalDelay', repr_str)


class TestGammaDelay(unittest.TestCase):
    """
    Test GammaDelay initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import GammaDelay

        init = GammaDelay(shape=2.0, scale=1.0 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 1000)
        assert np.all(delays >= 0 * u.ms)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_gamma_positive(self):
        init = GammaDelay(shape=2.0, scale=1.0 * u.ms)
        delays = init(self.rng, 1000)
        self.assertTrue(np.all(delays >= 0 * u.ms))

    def test_gamma_statistics(self):
        shape = 2.0
        scale = 1.0
        init = GammaDelay(shape=shape, scale=scale * u.ms)
        delays = init(self.rng, 100000)
        expected_mean = shape * scale
        mean = np.mean(delays.mantissa)
        self.assertAlmostEqual(mean, expected_mean, delta=0.05)

    def test_repr(self):
        init = GammaDelay(shape=2.0, scale=1.0 * u.ms)
        repr_str = repr(init)
        self.assertIn('GammaDelay', repr_str)


class TestDelayEdgeCases(unittest.TestCase):
    """
    Test edge cases for delay initializations.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import NormalDelay

        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms)
        rng = np.random.default_rng(0)

        delays_1d = init(rng, 100)
        assert delays_1d.shape == (100,)

        delays_2d = init(rng, (10, 20))
        assert delays_2d.shape == (10, 20)

        delays_zero = init(rng, 0)
        assert len(delays_zero) == 0
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_zero_size(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms)
        delays = init(self.rng, 0)
        self.assertEqual(len(delays), 0)

    def test_large_size(self):
        init = ConstantDelay(2.0 * u.ms)
        delays = init(self.rng, 1000000)
        self.assertEqual(len(delays), 1000000)

    def test_tuple_size(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms)
        delays = init(self.rng, (10, 20, 30))
        self.assertEqual(delays.shape, (10, 20, 30))

    def test_different_units(self):
        init = UniformDelay(1000.0 * u.us, 5000.0 * u.us)
        delays = init(self.rng, 100)
        self.assertTrue(np.all(delays >= 1.0 * u.ms))
        self.assertTrue(np.all(delays < 5.0 * u.ms))

    def test_unit_consistency(self):
        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms)
        delays = init(self.rng, 100)
        self.assertEqual(delays.unit, u.ms)


if __name__ == '__main__':
    unittest.main()
