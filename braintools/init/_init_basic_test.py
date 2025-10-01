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
Tests for basic weight initialization distributions.
"""

import unittest

import brainunit as u
import numpy as np

from braintools.init import (
    Constant,
    Uniform,
    Normal,
    LogNormal,
    Gamma,
    Exponential,
    TruncatedNormal,
    Beta,
    Weibull,
)


class TestConstant(unittest.TestCase):
    """
    Test Constant initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Constant

        init = Constant(0.5 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(100, rng=rng)
        assert np.all(weights == 0.5 * u.siemens)
    """

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
    """
    Test Uniform initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Uniform

        init = Uniform(0.1 * u.siemens, 1.0 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all((weights >= 0.1 * u.siemens) & (weights < 1.0 * u.siemens))
    """

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
    """
    Test Normal initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Normal

        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert abs(np.mean(weights.mantissa) - 0.5) < 0.05
    """

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
    """
    Test LogNormal initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import LogNormal

        init = LogNormal(0.5 * u.siemens, 0.2 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all(weights > 0 * u.siemens)
    """

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
    """
    Test Gamma initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Gamma

        init = Gamma(shape=2.0, scale=0.5 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all(weights >= 0 * u.siemens)
    """

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
    """
    Test Exponential initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Exponential

        init = Exponential(0.5 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all(weights >= 0 * u.siemens)
    """

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
    """
    Test TruncatedNormal initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import TruncatedNormal

        init = TruncatedNormal(
            mean=0.5 * u.siemens,
            std=0.2 * u.siemens,
            low=0.0 * u.siemens,
            high=1.0 * u.siemens
        )
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.0 * u.siemens))
    """

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
    """
    Test Beta initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Beta

        init = Beta(alpha=2.0, beta=5.0, low=0.0 * u.siemens, high=1.0 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.0 * u.siemens))
    """

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
    """
    Test Weibull initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Weibull

        init = Weibull(shape=1.5, scale=0.5 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all(weights >= 0 * u.siemens)
    """

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


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases for basic distributions.
    """

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
