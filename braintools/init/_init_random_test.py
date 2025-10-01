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

from braintools.conn import (
    ConstantWeight,
    UniformWeight,
    NormalWeight,
    LogNormalWeight,
    GammaWeight,
    ExponentialWeight,
    ExponentialDecayWeight,
    TruncatedNormalWeight,
    BetaWeight,
    WeibullWeight,
    MixtureWeight,
    ConditionalWeight,
    ScaledWeight,
    ClippedWeight,
    init_call,
)


class TestConstantWeight(unittest.TestCase):
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
        weights = init(rng, 100)
        assert np.all(weights == 0.5 * u.siemens)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_constant_value(self):
        init = ConstantWeight(0.5 * u.siemens)
        weights = init(self.rng, 100)
        self.assertEqual(weights.shape, (100,))
        self.assertTrue(np.all(weights == 0.5 * u.siemens))

    def test_constant_with_tuple_size(self):
        init = ConstantWeight(1.0 * u.siemens)
        weights = init(self.rng, (10, 20))
        self.assertEqual(weights.shape, (10, 20))
        self.assertTrue(np.all(weights == 1.0 * u.siemens))

    def test_repr(self):
        init = ConstantWeight(0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Constant', repr_str)
        self.assertIn('0.5', repr_str)


class TestUniformWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all((weights >= 0.1 * u.siemens) & (weights < 1.0 * u.siemens))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_uniform_distribution(self):
        init = UniformWeight(0.1 * u.siemens, 1.0 * u.siemens)
        weights = init(self.rng, 10000)
        self.assertEqual(weights.shape, (10000,))
        self.assertTrue(np.all(weights >= 0.1 * u.siemens))
        self.assertTrue(np.all(weights < 1.0 * u.siemens))

    def test_uniform_statistics(self):
        init = UniformWeight(0.0 * u.siemens, 1.0 * u.siemens)
        weights = init(self.rng, 100000)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, 0.5, delta=0.01)

    def test_repr(self):
        init = UniformWeight(0.1 * u.siemens, 1.0 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Uniform', repr_str)


class TestNormalWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert abs(np.mean(weights.mantissa) - 0.5) < 0.05
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_normal_distribution(self):
        init = NormalWeight(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(self.rng, 100000)
        self.assertEqual(weights.shape, (100000,))

    def test_normal_statistics(self):
        init = NormalWeight(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(self.rng, 100000)
        mean = np.mean(weights.mantissa)
        std = np.std(weights.mantissa)
        self.assertAlmostEqual(mean, 0.5, delta=0.01)
        self.assertAlmostEqual(std, 0.1, delta=0.01)

    def test_repr(self):
        init = NormalWeight(0.5 * u.siemens, 0.1 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Normal', repr_str)


class TestLogNormalWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all(weights > 0 * u.siemens)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_lognormal_positive(self):
        init = LogNormalWeight(0.5 * u.siemens, 0.2 * u.siemens)
        weights = init(self.rng, 1000)
        self.assertTrue(np.all(weights > 0 * u.siemens))

    def test_lognormal_statistics(self):
        init = LogNormalWeight(1.0 * u.siemens, 0.5 * u.siemens)
        weights = init(self.rng, 100000)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, 1.0, delta=0.05)

    def test_repr(self):
        init = LogNormalWeight(0.5 * u.siemens, 0.2 * u.siemens)
        repr_str = repr(init)
        self.assertIn('LogNormal', repr_str)


class TestGammaWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all(weights >= 0 * u.siemens)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_gamma_positive(self):
        init = GammaWeight(shape=2.0, scale=0.5 * u.siemens)
        weights = init(self.rng, 1000)
        self.assertTrue(np.all(weights >= 0 * u.siemens))

    def test_gamma_statistics(self):
        shape = 2.0
        scale = 0.5
        init = GammaWeight(shape=shape, scale=scale * u.siemens)
        weights = init(self.rng, 100000)
        expected_mean = shape * scale
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, expected_mean, delta=0.05)

    def test_repr(self):
        init = GammaWeight(shape=2.0, scale=0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Gamma', repr_str)


class TestExponentialWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all(weights >= 0 * u.siemens)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_exponential_positive(self):
        init = ExponentialWeight(0.5 * u.siemens)
        weights = init(self.rng, 1000)
        self.assertTrue(np.all(weights >= 0 * u.siemens))

    def test_exponential_statistics(self):
        scale = 0.5
        init = ExponentialWeight(scale * u.siemens)
        weights = init(self.rng, 100000)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, scale, delta=0.01)

    def test_repr(self):
        init = ExponentialWeight(0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Exponential', repr_str)


class TestExponentialDecayWeight(unittest.TestCase):
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
        weights = init(rng, 4, distances=distances)
        assert weights[0] == 1.0 * u.siemens
        assert weights[-1] >= 0.01 * u.siemens
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_exponential_decay_without_distances(self):
        init = ExponentialDecayWeight(
            max_weight=1.0 * u.siemens,
            decay_constant=100.0 * u.um
        )
        weights = init(self.rng, 10)
        self.assertTrue(np.all(weights == 1.0 * u.siemens))

    def test_exponential_decay_with_distances(self):
        init = ExponentialDecayWeight(
            max_weight=1.0 * u.siemens,
            decay_constant=100.0 * u.um,
            min_weight=0.0 * u.siemens
        )
        distances = np.array([0, 100, 200, 300]) * u.um
        weights = init(self.rng, 4, distances=distances)

        self.assertAlmostEqual(weights[0].mantissa, 1.0, delta=0.001)
        self.assertAlmostEqual(weights[1].mantissa, 1.0 / np.e, delta=0.001)
        self.assertTrue(weights[0] > weights[1] > weights[2] > weights[3])

    def test_exponential_decay_min_weight(self):
        init = ExponentialDecayWeight(
            max_weight=1.0 * u.siemens,
            decay_constant=10.0 * u.um,
            min_weight=0.1 * u.siemens
        )
        distances = np.array([0, 100, 1000]) * u.um
        weights = init(self.rng, 3, distances=distances)
        self.assertTrue(np.all(weights >= 0.1 * u.siemens))

    def test_repr(self):
        init = ExponentialDecayWeight(1.0 * u.siemens, 100.0 * u.um)
        repr_str = repr(init)
        self.assertIn('ExponentialDecay', repr_str)


class TestTruncatedNormalWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.0 * u.siemens))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_truncated_normal_bounds(self):
        try:
            init = TruncatedNormalWeight(
                mean=0.5 * u.siemens,
                std=0.2 * u.siemens,
                low=0.0 * u.siemens,
                high=1.0 * u.siemens
            )
            weights = init(self.rng, 1000)
            self.assertTrue(np.all(weights >= 0.0 * u.siemens))
            self.assertTrue(np.all(weights <= 1.0 * u.siemens))
        except ImportError:
            self.skipTest("scipy not installed")

    def test_truncated_normal_statistics(self):
        try:
            init = TruncatedNormalWeight(
                mean=0.5 * u.siemens,
                std=0.1 * u.siemens,
                low=0.0 * u.siemens,
                high=1.0 * u.siemens
            )
            weights = init(self.rng, 100000)
            mean = np.mean(weights.mantissa)
            self.assertAlmostEqual(mean, 0.5, delta=0.05)
        except ImportError:
            self.skipTest("scipy not installed")

    def test_scipy_import_error(self):
        init = TruncatedNormalWeight(
            mean=0.5 * u.siemens,
            std=0.2 * u.siemens
        )
        try:
            import scipy
            weights = init(self.rng, 100)
            self.assertEqual(weights.shape, (100,))
        except ImportError:
            with self.assertRaises(ImportError):
                init(self.rng, 100)

    def test_repr(self):
        init = TruncatedNormalWeight(0.5 * u.siemens, 0.2 * u.siemens)
        repr_str = repr(init)
        self.assertIn('TruncatedNormal', repr_str)


class TestBetaWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.0 * u.siemens))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_beta_bounds(self):
        init = BetaWeight(
            alpha=2.0,
            beta=5.0,
            low=0.0 * u.siemens,
            high=1.0 * u.siemens
        )
        weights = init(self.rng, 1000)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_beta_statistics(self):
        alpha, beta = 2.0, 5.0
        init = BetaWeight(
            alpha=alpha,
            beta=beta,
            low=0.0 * u.siemens,
            high=1.0 * u.siemens
        )
        weights = init(self.rng, 100000)
        expected_mean = alpha / (alpha + beta)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, expected_mean, delta=0.01)

    def test_repr(self):
        init = BetaWeight(2.0, 5.0, 0.0 * u.siemens, 1.0 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Beta', repr_str)


class TestWeibullWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all(weights >= 0 * u.siemens)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_weibull_positive(self):
        init = WeibullWeight(shape=1.5, scale=0.5 * u.siemens)
        weights = init(self.rng, 1000)
        self.assertTrue(np.all(weights >= 0 * u.siemens))

    def test_repr(self):
        init = WeibullWeight(1.5, 0.5 * u.siemens)
        repr_str = repr(init)
        self.assertIn('Weibull', repr_str)


class TestMixtureWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert len(weights) == 1000
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_mixture_basic(self):
        init = MixtureWeight(
            distributions=[
                ConstantWeight(0.5 * u.siemens),
                ConstantWeight(1.0 * u.siemens)
            ],
            weights=[0.5, 0.5]
        )
        weights = init(self.rng, 10000)
        count_low = np.sum(np.abs(weights.mantissa - 0.5) < 0.01)
        count_high = np.sum(np.abs(weights.mantissa - 1.0) < 0.01)
        self.assertAlmostEqual(count_low / 10000, 0.5, delta=0.05)
        self.assertAlmostEqual(count_high / 10000, 0.5, delta=0.05)

    def test_mixture_equal_weights(self):
        init = MixtureWeight(
            distributions=[
                ConstantWeight(0.3 * u.siemens),
                ConstantWeight(0.6 * u.siemens),
                ConstantWeight(0.9 * u.siemens)
            ]
        )
        weights = init(self.rng, 3000)
        self.assertEqual(len(weights), 3000)

    def test_repr(self):
        init = MixtureWeight([ConstantWeight(0.5 * u.siemens)])
        self.assertIn('Mixture', repr(init))


class TestConditionalWeight(unittest.TestCase):
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
        weights = init(rng, 1000, neuron_indices=np.arange(1000))
        assert len(weights) == 1000
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_conditional_basic(self):
        def is_even(indices):
            return indices % 2 == 0

        init = ConditionalWeight(
            condition_fn=is_even,
            true_dist=ConstantWeight(0.5 * u.siemens),
            false_dist=ConstantWeight(1.0 * u.siemens)
        )
        weights = init(self.rng, 100, neuron_indices=np.arange(100))

        for i in range(100):
            if i % 2 == 0:
                self.assertAlmostEqual(weights[i].mantissa, 0.5, delta=0.001)
            else:
                self.assertAlmostEqual(weights[i].mantissa, 1.0, delta=0.001)

    def test_conditional_without_indices(self):
        def all_true(indices):
            return np.ones(len(indices), dtype=bool)

        init = ConditionalWeight(
            condition_fn=all_true,
            true_dist=ConstantWeight(0.5 * u.siemens),
            false_dist=ConstantWeight(1.0 * u.siemens)
        )
        weights = init(self.rng, 50)
        self.assertTrue(np.all(np.abs(weights.mantissa - 0.5) < 0.001))

    def test_repr(self):
        def dummy(x):
            return x > 0

        init = ConditionalWeight(
            dummy,
            ConstantWeight(0.5 * u.siemens),
            ConstantWeight(1.0 * u.siemens)
        )
        self.assertIn('Conditional', repr(init))


class TestScaledWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.mean(weights.mantissa) < 1.0
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_scaled_basic(self):
        base = ConstantWeight(1.0 * u.siemens)
        init = ScaledWeight(base, scale_factor=0.5)
        weights = init(self.rng, 100)
        self.assertTrue(np.all(np.abs(weights.mantissa - 0.5) < 0.001))

    def test_scaled_with_quantity(self):
        base = ConstantWeight(1.0 * u.siemens)
        init = ScaledWeight(base, scale_factor=2.0)
        weights = init(self.rng, 100)
        self.assertTrue(np.all(np.abs(weights.mantissa - 2.0) < 0.001))

    def test_scaled_statistics(self):
        base = NormalWeight(1.0 * u.siemens, 0.1 * u.siemens)
        init = ScaledWeight(base, scale_factor=2.0)
        weights = init(self.rng, 100000)
        mean = np.mean(weights.mantissa)
        self.assertAlmostEqual(mean, 2.0, delta=0.01)

    def test_repr(self):
        base = ConstantWeight(1.0 * u.siemens)
        init = ScaledWeight(base, 0.5)
        self.assertIn('Scaled', repr(init))


class TestClippedWeight(unittest.TestCase):
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
        weights = init(rng, 1000)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.0 * u.siemens))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_clipped_min(self):
        base = NormalWeight(0.0 * u.siemens, 1.0 * u.siemens)
        init = ClippedWeight(base, min_val=0.0 * u.siemens)
        weights = init(self.rng, 10000)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))

    def test_clipped_max(self):
        base = NormalWeight(1.0 * u.siemens, 1.0 * u.siemens)
        init = ClippedWeight(base, max_val=1.0 * u.siemens)
        weights = init(self.rng, 10000)
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_clipped_both(self):
        base = NormalWeight(0.5 * u.siemens, 1.0 * u.siemens)
        init = ClippedWeight(base, min_val=0.0 * u.siemens, max_val=1.0 * u.siemens)
        weights = init(self.rng, 10000)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.0 * u.siemens))

    def test_clipped_statistics(self):
        base = NormalWeight(0.5 * u.siemens, 0.1 * u.siemens)
        init = ClippedWeight(base, min_val=0.4 * u.siemens, max_val=0.6 * u.siemens)
        weights = init(self.rng, 100000)
        self.assertTrue(np.all(weights >= 0.4 * u.siemens))
        self.assertTrue(np.all(weights <= 0.6 * u.siemens))

    def test_repr(self):
        base = ConstantWeight(1.0 * u.siemens)
        init = ClippedWeight(base, min_val=0.0 * u.siemens)
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

        weights = init_call(Normal(0.5 * u.siemens, 0.1 * u.siemens), rng, 100)
        assert len(weights) == 100

        scalar_weights = init_call(0.5, rng, 100)
        assert scalar_weights == 0.5

        none_weights = init_call(None, rng, 100)
        assert none_weights is None
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_init_call_with_initialization(self):
        init = ConstantWeight(0.5 * u.siemens)
        result = init_call(init, self.rng, 100)
        self.assertEqual(len(result), 100)
        self.assertTrue(np.all(result == 0.5 * u.siemens))

    def test_init_call_with_scalar_float(self):
        result = init_call(0.5, self.rng, 100)
        self.assertEqual(result, 0.5)

    def test_init_call_with_scalar_int(self):
        result = init_call(5, self.rng, 100)
        self.assertEqual(result, 5)

    def test_init_call_with_none(self):
        result = init_call(None, self.rng, 100)
        self.assertIsNone(result)

    def test_init_call_with_quantity_scalar(self):
        result = init_call(0.5 * u.siemens, self.rng, 100)
        self.assertEqual(result, 0.5 * u.siemens)

    def test_init_call_with_quantity_array(self):
        arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * u.siemens
        result = init_call(arr, self.rng, 5)
        self.assertEqual(len(result), 5)

    def test_init_call_with_invalid_array_size(self):
        arr = np.array([0.1, 0.2, 0.3]) * u.siemens
        with self.assertRaises(ValueError):
            init_call(arr, self.rng, 100)

    def test_init_call_with_invalid_type(self):
        with self.assertRaises(TypeError):
            init_call("invalid", self.rng, 100)


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

        weights_zero = init(rng, 0)
        assert len(weights_zero) == 0
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_zero_size(self):
        init = NormalWeight(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(self.rng, 0)
        self.assertEqual(len(weights), 0)

    def test_large_size(self):
        init = ConstantWeight(0.5 * u.siemens)
        weights = init(self.rng, 1000000)
        self.assertEqual(len(weights), 1000000)

    def test_tuple_size(self):
        init = NormalWeight(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(self.rng, (10, 20, 30))
        self.assertEqual(weights.shape, (10, 20, 30))

    def test_different_units(self):
        init = UniformWeight(100.0 * u.uS, 1000.0 * u.uS)
        weights = init(self.rng, 100)
        self.assertTrue(np.all(weights >= 100.0 * u.uS))
        self.assertTrue(np.all(weights < 1000.0 * u.uS))

    def test_unit_consistency(self):
        init = NormalWeight(0.5 * u.siemens, 0.1 * u.siemens)
        weights = init(self.rng, 100)
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
        weights = init(rng, 1000)
        assert np.all((weights >= 0.0 * u.siemens) & (weights <= 1.5 * u.siemens))
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_scaled_clipped_combination(self):
        base = NormalWeight(0.5 * u.siemens, 0.2 * u.siemens)
        scaled = ScaledWeight(base, scale_factor=2.0)
        clipped = ClippedWeight(scaled, min_val=0.0 * u.siemens, max_val=1.5 * u.siemens)
        weights = clipped(self.rng, 10000)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.5 * u.siemens))

    def test_mixture_of_conditionals(self):
        def is_even(indices):
            return indices % 2 == 0

        cond1 = ConditionalWeight(
            is_even,
            ConstantWeight(0.3 * u.siemens),
            ConstantWeight(0.7 * u.siemens)
        )

        weights = cond1(self.rng, 1000, neuron_indices=np.arange(1000))
        self.assertEqual(len(weights), 1000)
        for i in range(100):
            if i % 2 == 0:
                self.assertAlmostEqual(weights[i].mantissa, 0.3, delta=0.001)
            else:
                self.assertAlmostEqual(weights[i].mantissa, 0.7, delta=0.001)

    def test_clipped_mixture_scaled(self):
        mix = MixtureWeight(
            distributions=[
                NormalWeight(0.5 * u.siemens, 0.1 * u.siemens),
                UniformWeight(0.3 * u.siemens, 0.7 * u.siemens)
            ],
            weights=[0.6, 0.4]
        )
        scaled = ScaledWeight(mix, scale_factor=2.0)
        clipped = ClippedWeight(scaled, min_val=0.0 * u.siemens, max_val=1.5 * u.siemens)
        weights = clipped(self.rng, 10000)
        self.assertTrue(np.all(weights >= 0.0 * u.siemens))
        self.assertTrue(np.all(weights <= 1.5 * u.siemens))


if __name__ == '__main__':
    unittest.main()
