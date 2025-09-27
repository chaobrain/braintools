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
Comprehensive tests for distance profile classes.

This test suite covers:
- Gaussian distance profile
- Exponential distance profile
- Power-law distance profile
- Linear distance profile
- Step function distance profile
"""

import unittest

import brainunit as u
import numpy as np

from braintools.conn import (
    GaussianProfile,
    ExponentialProfile,
    PowerLawProfile,
    LinearProfile,
    StepProfile,
)


class TestDistanceProfiles(unittest.TestCase):
    """
    Test distance profile classes.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import GaussianProfile, ExponentialProfile

        gaussian = GaussianProfile(sigma=50.0 * u.um, max_distance=200.0 * u.um)
        distances = np.array([0, 25, 50, 100, 200]) * u.um
        probs = gaussian.probability(distances)
        assert probs[0] == 1.0
        assert probs[-1] == 0.0

        exponential = ExponentialProfile(
            decay_constant=100.0 * u.um,
            max_distance=500.0 * u.um
        )
        probs = exponential.probability(distances)
        assert probs[0] == 1.0
    """

    def test_gaussian_profile(self):
        profile = GaussianProfile(sigma=50.0 * u.um, max_distance=200.0 * u.um)
        distances = np.array([0, 25, 50, 100, 200, 250]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3])
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)
        self.assertAlmostEqual(probs[5], 0.0, delta=0.001)

    def test_gaussian_weight_scaling(self):
        profile = GaussianProfile(sigma=50.0 * u.um)
        distances = np.array([0, 50, 100]) * u.um
        weights = profile.weight_scaling(distances)
        self.assertEqual(len(weights), 3)

    def test_exponential_profile(self):
        profile = ExponentialProfile(
            decay_constant=100.0 * u.um,
            max_distance=500.0 * u.um
        )
        distances = np.array([0, 100, 200, 400, 500.1, 600]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertAlmostEqual(probs[1], 1.0 / np.e, delta=0.001)
        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3])
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)
        self.assertAlmostEqual(probs[5], 0.0, delta=0.001)

    def test_power_law_profile(self):
        profile = PowerLawProfile(
            exponent=2.0,
            min_distance=1.0 * u.um,
            max_distance=1000.0 * u.um
        )
        distances = np.array([1, 10, 100, 1000, 2000]) * u.um
        probs = profile.probability(distances)

        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3])
        self.assertEqual(probs[4], 0.0)

    def test_linear_profile(self):
        profile = LinearProfile(max_distance=200.0 * u.um)
        distances = np.array([0, 50, 100, 150, 200, 250]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertAlmostEqual(probs[1], 0.75, delta=0.001)
        self.assertAlmostEqual(probs[2], 0.5, delta=0.001)
        self.assertAlmostEqual(probs[3], 0.25, delta=0.001)
        self.assertEqual(probs[4], 0.0)
        self.assertEqual(probs[5], 0.0)

    def test_step_profile(self):
        profile = StepProfile(
            threshold=100.0 * u.um,
            inside_prob=0.8,
            outside_prob=0.1
        )
        distances = np.array([50, 100, 150]) * u.um
        probs = profile.probability(distances)

        self.assertEqual(probs[0], 0.8)
        self.assertEqual(probs[1], 0.8)
        self.assertEqual(probs[2], 0.1)

    def test_profile_repr(self):
        profile = GaussianProfile(sigma=50.0 * u.um)
        self.assertIn('GaussianProfile', repr(profile))


class TestGaussianProfile(unittest.TestCase):
    """
    Test GaussianProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import GaussianProfile

        profile = GaussianProfile(sigma=50.0 * u.um, max_distance=200.0 * u.um)
        distances = np.array([0, 25, 50, 100, 200]) * u.um
        probs = profile.probability(distances)
        assert probs[0] == 1.0
        assert probs[-1] == 0.0
    """

    def test_gaussian_basic(self):
        profile = GaussianProfile(sigma=50.0 * u.um)
        distances = np.array([0, 50, 100, 150]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3])

    def test_gaussian_with_max_distance(self):
        profile = GaussianProfile(sigma=50.0 * u.um, max_distance=150.0 * u.um)
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertTrue(probs[3] < 0.02)
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)

    def test_gaussian_weight_scaling_matches_probability(self):
        profile = GaussianProfile(sigma=50.0 * u.um)
        distances = np.array([0, 50, 100]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = GaussianProfile(sigma=50.0 * u.um, max_distance=200.0 * u.um)
        repr_str = repr(profile)
        self.assertIn('GaussianProfile', repr_str)
        self.assertIn('50.0', repr_str)


class TestExponentialProfile(unittest.TestCase):
    """
    Test ExponentialProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import ExponentialProfile

        profile = ExponentialProfile(
            decay_constant=100.0 * u.um,
            max_distance=500.0 * u.um
        )
        distances = np.array([0, 50, 100, 200, 500]) * u.um
        probs = profile.probability(distances)
        assert probs[0] == 1.0
    """

    def test_exponential_basic(self):
        profile = ExponentialProfile(decay_constant=100.0 * u.um)
        distances = np.array([0, 100, 200, 300]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertAlmostEqual(probs[1], 1.0 / np.e, delta=0.001)
        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3])

    def test_exponential_with_max_distance(self):
        profile = ExponentialProfile(
            decay_constant=100.0 * u.um,
            max_distance=400.0 * u.um
        )
        distances = np.array([0, 100, 200, 400.1, 500]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertAlmostEqual(probs[3], 0.0, delta=0.001)
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)

    def test_exponential_weight_scaling_matches_probability(self):
        profile = ExponentialProfile(decay_constant=100.0 * u.um)
        distances = np.array([0, 50, 100]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = ExponentialProfile(decay_constant=100.0 * u.um, max_distance=500.0 * u.um)
        repr_str = repr(profile)
        self.assertIn('ExponentialProfile', repr_str)


class TestPowerLawProfile(unittest.TestCase):
    """
    Test PowerLawProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import PowerLawProfile

        profile = PowerLawProfile(
            exponent=2.0,
            min_distance=1.0 * u.um,
            max_distance=1000.0 * u.um
        )
        distances = np.array([1, 10, 100, 1000]) * u.um
        probs = profile.probability(distances)
    """

    def test_power_law_basic(self):
        profile = PowerLawProfile(exponent=2.0, min_distance=1.0 * u.um)
        distances = np.array([1, 10, 100, 1000]) * u.um
        probs = profile.probability(distances)

        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3])

    def test_power_law_with_max_distance(self):
        profile = PowerLawProfile(
            exponent=2.0,
            min_distance=1.0 * u.um,
            max_distance=500.0 * u.um
        )
        distances = np.array([1, 10, 100, 500, 600]) * u.um
        probs = profile.probability(distances)

        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3])
        self.assertEqual(probs[4], 0.0)

    def test_power_law_min_distance_protection(self):
        profile = PowerLawProfile(exponent=2.0)
        distances = np.array([0, 0.1, 1, 10]) * u.um
        probs = profile.probability(distances)

        self.assertTrue(np.all(np.isfinite(probs)))

    def test_repr(self):
        profile = PowerLawProfile(exponent=2.0, min_distance=1.0 * u.um)
        repr_str = repr(profile)
        self.assertIn('PowerLawProfile', repr_str)


class TestLinearProfile(unittest.TestCase):
    """
    Test LinearProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import LinearProfile

        profile = LinearProfile(max_distance=200.0 * u.um)
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)
        assert probs[0] == 1.0
        assert probs[-1] == 0.0
    """

    def test_linear_basic(self):
        profile = LinearProfile(max_distance=200.0 * u.um)
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertAlmostEqual(probs[1], 0.75, delta=0.001)
        self.assertAlmostEqual(probs[2], 0.5, delta=0.001)
        self.assertAlmostEqual(probs[3], 0.25, delta=0.001)
        self.assertEqual(probs[4], 0.0)

    def test_linear_beyond_max(self):
        profile = LinearProfile(max_distance=100.0 * u.um)
        distances = np.array([0, 50, 100, 150]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)
        self.assertEqual(probs[2], 0.0)
        self.assertEqual(probs[3], 0.0)

    def test_linear_weight_scaling_matches_probability(self):
        profile = LinearProfile(max_distance=200.0 * u.um)
        distances = np.array([0, 50, 100]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = LinearProfile(max_distance=200.0 * u.um)
        repr_str = repr(profile)
        self.assertIn('LinearProfile', repr_str)


class TestStepProfile(unittest.TestCase):
    """
    Test StepProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import StepProfile

        profile = StepProfile(
            threshold=100.0 * u.um,
            inside_prob=0.8,
            outside_prob=0.1
        )
        distances = np.array([50, 100, 150]) * u.um
        probs = profile.probability(distances)
    """

    def test_step_basic(self):
        profile = StepProfile(threshold=100.0 * u.um)
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)

        self.assertEqual(probs[0], 1.0)
        self.assertEqual(probs[1], 1.0)
        self.assertEqual(probs[2], 1.0)
        self.assertEqual(probs[3], 0.0)
        self.assertEqual(probs[4], 0.0)

    def test_step_custom_probabilities(self):
        profile = StepProfile(
            threshold=100.0 * u.um,
            inside_prob=0.8,
            outside_prob=0.2
        )
        distances = np.array([50, 100, 150]) * u.um
        probs = profile.probability(distances)

        self.assertEqual(probs[0], 0.8)
        self.assertEqual(probs[1], 0.8)
        self.assertEqual(probs[2], 0.2)

    def test_step_weight_scaling_matches_probability(self):
        profile = StepProfile(threshold=100.0 * u.um, inside_prob=0.7, outside_prob=0.3)
        distances = np.array([50, 100, 150]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = StepProfile(threshold=100.0 * u.um, inside_prob=0.8, outside_prob=0.1)
        repr_str = repr(profile)
        self.assertIn('StepProfile', repr_str)


class TestDistanceProfileEdgeCases(unittest.TestCase):
    """
    Test edge cases for distance profiles.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import GaussianProfile

        profile = GaussianProfile(sigma=50.0 * u.um)

        distances_empty = np.array([]) * u.um
        probs_empty = profile.probability(distances_empty)
        assert len(probs_empty) == 0

        distances_single = np.array([25]) * u.um
        probs_single = profile.probability(distances_single)
        assert len(probs_single) == 1
    """

    def test_empty_distances(self):
        profile = GaussianProfile(sigma=50.0 * u.um)
        distances = np.array([]) * u.um
        probs = profile.probability(distances)
        self.assertEqual(len(probs), 0)

    def test_single_distance(self):
        profile = GaussianProfile(sigma=50.0 * u.um)
        distances = np.array([25]) * u.um
        probs = profile.probability(distances)
        self.assertEqual(len(probs), 1)
        self.assertTrue(0.0 <= probs[0] <= 1.0)

    def test_different_units(self):
        profile = ExponentialProfile(decay_constant=0.1 * u.mm)
        distances = np.array([0, 50, 100]) * u.um
        probs = profile.probability(distances)
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(probs[0], 1.0, delta=0.001)

    def test_large_distances(self):
        profile = LinearProfile(max_distance=100.0 * u.um)
        distances = np.array([1e6, 1e7, 1e8]) * u.um
        probs = profile.probability(distances)
        self.assertTrue(np.all(probs == 0.0))


if __name__ == '__main__':
    unittest.main()
