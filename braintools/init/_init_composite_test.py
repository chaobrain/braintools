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
Tests for composite weight initialization distributions.
"""

import unittest

import brainunit as u
import numpy as np

from braintools.init import (
    Constant,
    Normal,
    Uniform,
    Mixture,
    Conditional,
    Scaled,
    Clipped,
)


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
