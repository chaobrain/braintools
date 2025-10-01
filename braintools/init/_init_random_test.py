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
Comprehensive tests for weight initialization classes.

This test suite covers:
- Basic weight distributions (Constant, Uniform, Normal, etc.)
- Composite weight distributions (Mixture, Conditional, Scaled, Clipped)
- Edge cases and error handling
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
    ExponentialDecay,
    TruncatedNormal,
    Beta,
    Weibull,
    Mixture,
    Conditional,
    Scaled,
    Clipped,
    init_call,
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


class TestExponentialDecay(unittest.TestCase):
    """
    Test ExponentialDecay initialization.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import ExponentialDecay

        init = ExponentialDecay(
            max_weight=1.0 * u.siemens,
            decay_constant=100.0 * u.um,
            min_weight=0.01 * u.siemens
        )
        rng = np.random.default_rng(0)
        distances = np.array([0, 50, 100, 200]) * u.um
        weights = init(4, distances=distances, rng=rng)
        assert weights[0] == 1.0 * u.siemens
        assert weights[-1] >= 0.01 * u.siemens
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_exponential_decay_without_distances(self):
        init = ExponentialDecay(
            max_weight=1.0 * u.siemens,
            decay_constant=100.0 * u.um
        )
        weights = init(10, rng=self.rng)
        self.assertTrue(np.all(weights == 1.0 * u.siemens))

    def test_exponential_decay_with_distances(self):
        init = ExponentialDecay(
            max_weight=1.0 * u.siemens,
            decay_constant=100.0 * u.um,
            min_weight=0.0 * u.siemens
        )
        distances = np.array([0, 100, 200, 300]) * u.um
        weights = init(4, distances=distances, rng=self.rng)

        self.assertAlmostEqual(weights[0].mantissa, 1.0, delta=0.001)
        self.assertAlmostEqual(weights[1].mantissa, 1.0 / np.e, delta=0.001)
        self.assertTrue(weights[0] > weights[1] > weights[2] > weights[3])

    def test_exponential_decay_min_weight(self):
        init = ExponentialDecay(
            max_weight=1.0 * u.siemens,
            decay_constant=10.0 * u.um,
            min_weight=0.1 * u.siemens
        )
        distances = np.array([0, 100, 1000]) * u.um
        weights = init(3, distances=distances, rng=self.rng)
        self.assertTrue(np.all(weights >= 0.1 * u.siemens))

    def test_repr(self):
        init = ExponentialDecay(1.0 * u.siemens, 100.0 * u.um)
        repr_str = repr(init)
        self.assertIn('ExponentialDecay', repr_str)


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


class TestMixture(unittest.TestCase):
    """
    Test Mixture composite distribution.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Mixture, Normal, Uniform

        init = Mixture(
            distributions=[
                Normal(0.5 * u.siemens, 0.1 * u.siemens),
                Uniform(0.8 * u.siemens, 1.2 * u.siemens)
            ],
            weights=[0.7, 0.3]
        )
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert len(weights) == 1000
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_mixture_basic(self):
        init = Mixture(
            distributions=[
                Constant(0.5 * u.siemens),
                Constant(1.0 * u.siemens)
            ],
            weights=[0.5, 0.5]
        )
        weights = init(10000, rng=self.rng)
        count_low = np.sum(np.abs(weights.mantissa - 0.5) < 0.01)
        count_high = np.sum(np.abs(weights.mantissa - 1.0) < 0.01)
        self.assertAlmostEqual(count_low / 10000, 0.5, delta=0.05)
        self.assertAlmostEqual(count_high / 10000, 0.5, delta=0.05)

    def test_mixture_equal_weights(self):
        init = Mixture(
            distributions=[
                Constant(0.3 * u.siemens),
                Constant(0.6 * u.siemens),
                Constant(0.9 * u.siemens)
            ]
        )
        weights = init(3000, rng=self.rng)
        self.assertEqual(len(weights), 3000)

    def test_repr(self):
        init = Mixture([Constant(0.5 * u.siemens)])
        self.assertIn('Mixture', repr(init))


class TestConditional(unittest.TestCase):
    """
    Test Conditional composite distribution.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Conditional, Constant, Normal

        def is_excitatory(indices):
            return indices < 800

        init = Conditional(
            condition_fn=is_excitatory,
            true_dist=Normal(0.5 * u.siemens, 0.1 * u.siemens),
            false_dist=Normal(-0.3 * u.siemens, 0.05 * u.siemens)
        )
        rng = np.random.default_rng(0)
        weights = init(1000, neuron_indices=np.arange(1000), rng=rng)
        assert len(weights) == 1000
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_conditional_basic(self):
        def is_even(indices):
            return indices % 2 == 0

        init = Conditional(
            condition_fn=is_even,
            true_dist=Constant(0.5 * u.siemens),
            false_dist=Constant(1.0 * u.siemens)
        )
        weights = init(100, neuron_indices=np.arange(100), rng=self.rng)

        for i in range(100):
            if i % 2 == 0:
                self.assertAlmostEqual(weights[i].mantissa, 0.5, delta=0.001)
            else:
                self.assertAlmostEqual(weights[i].mantissa, 1.0, delta=0.001)

    def test_conditional_without_indices(self):
        def all_true(indices):
            return np.ones(len(indices), dtype=bool)

        init = Conditional(
            condition_fn=all_true,
            true_dist=Constant(0.5 * u.siemens),
            false_dist=Constant(1.0 * u.siemens)
        )
        weights = init(50, rng=self.rng)
        self.assertTrue(np.all(np.abs(weights.mantissa - 0.5) < 0.001))

    def test_repr(self):
        def dummy(x):
            return x > 0

        init = Conditional(
            dummy,
            Constant(0.5 * u.siemens),
            Constant(1.0 * u.siemens)
        )
        self.assertIn('Conditional', repr(init))


class TestScaled(unittest.TestCase):
    """
    Test Scaled composite distribution.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Scaled, Normal

        base = Normal(1.0 * u.siemens, 0.2 * u.siemens)
        init = Scaled(base, scale_factor=0.5)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.mean(weights.mantissa) < 1.0
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_scaled_basic(self):
        base = Constant(1.0 * u.siemens)
        init = Scaled(base, scale_factor=0.5)
        weights = init(100, rng=self.rng)
        self.assertTrue(np.all(np.abs(weights.mantissa - 0.5) < 0.001))

    def test_scaled_with_quantity(self):
        base = Constant(1.0 * u.siemens)
        init = Scaled(base, scale_factor=2.0)
        weights = init(100, rng=self.rng)
        self.assertTrue(np.all(np.abs(weights.mantissa - 2.0) < 0.001))

    def test_scaled_statistics(self):
        base = Normal(1.0 * u.siemens, 0.1 * u.siemens)
        init = Scaled(base, scale_factor=2.0)
        weights = init(100000, rng=self.rng)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, 2.0, delta=0.01)

    def test_repr(self):
        base = Constant(1.0 * u.siemens)
        init = Scaled(base, 0.5)
        self.assertIn('Scaled', repr(init))


class TestClipped(unittest.TestCase):
    """
    Test Clipped composite distribution.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Clipped, Normal

        base = Normal(0.5 * u.siemens, 0.3 * u.siemens)
        init = Clipped(base, min_val=0.0 * u.siemens, max_val=1.0 * u.siemens)
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.0 * u.siemens))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_clipped_min(self):
        base = Normal(0.0 * u.siemens, 1.0 * u.siemens)
        init = Clipped(base, min_val=0.0 * u.siemens)
        weights = init(10000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))

    def test_clipped_max(self):
        base = Normal(1.0 * u.siemens, 1.0 * u.siemens)
        init = Clipped(base, max_val=1.0 * u.siemens)
        weights = init(10000, rng=self.rng)
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_clipped_both(self):
        base = Normal(0.5 * u.siemens, 1.0 * u.siemens)
        init = Clipped(base, min_val=0.0 * u.siemens, max_val=1.0 * u.siemens)
        weights = init(10000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_clipped_statistics(self):
        base = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        init = Clipped(base, min_val=0.4 * u.siemens, max_val=0.6 * u.siemens)
        weights = init(100000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0.4 * u.siemens))
        self.assertTrue(np.all(weights <= 0.6 * u.siemens))

    def test_repr(self):
        base = Constant(1.0 * u.siemens)
        init = Clipped(base, min_val=0.0 * u.siemens)
        self.assertIn('Clipped', repr(init))


class TestInitCall(unittest.TestCase):
    """
    Test init_call helper function.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import init_call, Normal

        rng = np.random.default_rng(0)

        weights = init_call(Normal(0.5 * u.siemens, 0.1 * u.siemens), 100, rng=rng)
        assert len(weights) == 100

        scalar_weights = init_call(0.5, 100, rng=rng)
        assert scalar_weights == 0.5

        none_weights = init_call(None, 100, rng=rng)
        assert none_weights is None
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_init_call_with_initialization(self):
        init = Constant(0.5 * u.siemens)
        result = init_call(init, 100, rng=self.rng)
        self.assertEqual(len(result), 100)
        self.assertTrue(np.all(result == 0.5 * u.siemens))

    def test_init_call_with_scalar_float(self):
        result = init_call(0.5, 100, rng=self.rng)
        self.assertEqual(result, 0.5)

    def test_init_call_with_scalar_int(self):
        result = init_call(5, 100, rng=self.rng)
        self.assertEqual(result, 5)

    def test_init_call_with_none(self):
        result = init_call(None, 100, rng=self.rng)
        self.assertIsNone(result)

    def test_init_call_with_quantity_scalar(self):
        result = init_call(0.5 * u.siemens, 100, rng=self.rng)
        self.assertEqual(result, 0.5 * u.siemens)

    def test_init_call_with_quantity_array(self):
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * u.siemens
        result = init_call(arr, 5, rng=self.rng)
        self.assertEqual(len(result), 5)

    def test_init_call_with_invalid_array_size(self):
        arr = np.array([0.1, 0.2, 0.3]) * u.siemens
        with self.assertRaises(ValueError):
            init_call(arr, 100, rng=self.rng)

    def test_init_call_with_invalid_type(self):
        with self.assertRaises(TypeError):
            init_call("invalid", 100, rng=self.rng)


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases and error handling.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Normal

        init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        rng = np.random.default_rng(0)

        weights_1d = init(rng, 100)
        assert weights_1d.shape == (100,)

        weights_2d = init(rng, (10, 20))
        assert weights_2d.shape == (10, 20)

        weights_zero = init(0, rng=rng)
        assert len(weights_zero) == 0
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


class TestCompositeScenarios(unittest.TestCase):
    """
    Test complex composite scenarios combining multiple initialization strategies.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import (
            Clipped, Scaled, Mixture, Normal, Uniform
        )

        init = Clipped(
            Scaled(
                Mixture(
                    distributions=[
                        Normal(0.5 * u.siemens, 0.1 * u.siemens),
                        Uniform(0.3 * u.siemens, 0.7 * u.siemens)
                    ],
                    weights=[0.6, 0.4]
                ),
                scale_factor=2.0
            ),
            min_val=0.0 * u.siemens,
            max_val=1.5 * u.siemens
        )
        rng = np.random.default_rng(0)
        weights = init(1000, rng=rng)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.5 * u.siemens))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_scaled_clipped_combination(self):
        base = Normal(0.5 * u.siemens, 0.2 * u.siemens)
        scaled = Scaled(base, scale_factor=2.0)
        clipped = Clipped(scaled, min_val=0.0 * u.siemens, max_val=1.5 * u.siemens)
        weights = clipped(10000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.5 * u.siemens))

    def test_mixture_of_conditionals(self):
        def is_even(indices):
            return indices % 2 == 0

        cond1 = Conditional(
            is_even,
            Constant(0.3 * u.siemens),
            Constant(0.7 * u.siemens)
        )

        weights = cond1(1000, neuron_indices=np.arange(1000), rng=self.rng)
        self.assertEqual(len(weights), 1000)
        for i in range(100):
            if i % 2 == 0:
                self.assertAlmostEqual(weights[i].mantissa, 0.3, delta=0.001)
            else:
                self.assertAlmostEqual(weights[i].mantissa, 0.7, delta=0.001)

    def test_clipped_mixture_scaled(self):
        mix = Mixture(
            distributions=[
                Normal(0.5 * u.siemens, 0.1 * u.siemens),
                Uniform(0.3 * u.siemens, 0.7 * u.siemens)
            ],
            weights=[0.6, 0.4]
        )
        scaled = Scaled(mix, scale_factor=2.0)
        clipped = Clipped(scaled, min_val=0.0 * u.siemens, max_val=1.5 * u.siemens)
        weights = clipped(10000, rng=self.rng)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.5 * u.siemens))


if __name__ == '__main__':
    unittest.main()
