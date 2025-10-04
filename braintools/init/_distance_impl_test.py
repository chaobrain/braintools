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

from braintools.init import (
    GaussianProfile,
    ExponentialProfile,
    PowerLawProfile,
    LinearProfile,
    StepProfile,
    SigmoidProfile,
    DoGProfile,
    LogisticProfile,
    BimodalProfile,
    MexicanHatProfile,
)


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


class TestSigmoidProfile(unittest.TestCase):
    """
    Test SigmoidProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.init import SigmoidProfile

        profile = SigmoidProfile(
            midpoint=100.0 * u.um,
            slope=0.05,
            max_distance=300.0 * u.um
        )
        distances = np.array([0, 50, 100, 150, 200, 300]) * u.um
        probs = profile.probability(distances)
    """

    def test_sigmoid_basic(self):
        profile = SigmoidProfile(midpoint=100.0 * u.um, slope=0.05)
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)

        # At midpoint, probability should be 0.5
        self.assertAlmostEqual(probs[2], 0.5, delta=0.01)
        # Probability decreases with distance
        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3] > probs[4])

    def test_sigmoid_with_max_distance(self):
        profile = SigmoidProfile(
            midpoint=100.0 * u.um,
            slope=0.05,
            max_distance=200.0 * u.um
        )
        distances = np.array([0, 50, 100, 150, 200, 250]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[2], 0.5, delta=0.01)
        self.assertAlmostEqual(probs[5], 0.0, delta=0.001)

    def test_sigmoid_slope_effect(self):
        profile_steep = SigmoidProfile(midpoint=100.0 * u.um, slope=0.1)
        profile_gentle = SigmoidProfile(midpoint=100.0 * u.um, slope=0.02)
        distances = np.array([80, 100, 120]) * u.um

        probs_steep = profile_steep.probability(distances)
        probs_gentle = profile_gentle.probability(distances)

        # Steeper slope should have larger differences
        steep_diff = probs_steep[0] - probs_steep[2]
        gentle_diff = probs_gentle[0] - probs_gentle[2]
        self.assertTrue(steep_diff > gentle_diff)

    def test_sigmoid_weight_scaling_matches_probability(self):
        profile = SigmoidProfile(midpoint=100.0 * u.um, slope=0.05)
        distances = np.array([0, 100, 200]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = SigmoidProfile(midpoint=100.0 * u.um, slope=0.05, max_distance=300.0 * u.um)
        repr_str = repr(profile)
        self.assertIn('SigmoidProfile', repr_str)
        self.assertIn('100.0', repr_str)


class TestDoGProfile(unittest.TestCase):
    """
    Test DoGProfile (Difference of Gaussians) distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.init import DoGProfile

        profile = DoGProfile(
            sigma_center=30.0 * u.um,
            sigma_surround=90.0 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.5
        )
        distances = np.array([0, 30, 60, 90, 120]) * u.um
        probs = profile.probability(distances)
    """

    def test_dog_basic(self):
        profile = DoGProfile(
            sigma_center=30.0 * u.um,
            sigma_surround=90.0 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.5
        )
        distances = np.array([0, 30, 60, 90, 120]) * u.um
        probs = profile.probability(distances)

        # At distance 0, center dominates
        self.assertTrue(probs[0] > 0)
        # All probabilities should be non-negative
        self.assertTrue(np.all(probs >= 0))

    def test_dog_center_surround_pattern(self):
        profile = DoGProfile(
            sigma_center=20.0 * u.um,
            sigma_surround=60.0 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.8
        )
        distances = np.array([0, 20, 40, 60, 100]) * u.um
        probs = profile.probability(distances)

        # Center should be high
        self.assertTrue(probs[0] > 0.1)
        # Far distances should be low
        self.assertTrue(probs[-1] < probs[0])

    def test_dog_with_max_distance(self):
        profile = DoGProfile(
            sigma_center=30.0 * u.um,
            sigma_surround=90.0 * u.um,
            max_distance=150.0 * u.um
        )
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)

        self.assertTrue(probs[0] > 0)
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)

    def test_dog_weight_scaling_matches_probability(self):
        profile = DoGProfile(
            sigma_center=30.0 * u.um,
            sigma_surround=90.0 * u.um
        )
        distances = np.array([0, 50, 100]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = DoGProfile(
            sigma_center=30.0 * u.um,
            sigma_surround=90.0 * u.um,
            amplitude_center=1.0,
            amplitude_surround=0.5
        )
        repr_str = repr(profile)
        self.assertIn('DoGProfile', repr_str)


class TestLogisticProfile(unittest.TestCase):
    """
    Test LogisticProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.init import LogisticProfile

        profile = LogisticProfile(
            growth_rate=0.05,
            midpoint=100.0 * u.um,
            max_distance=500.0 * u.um
        )
        distances = np.array([0, 50, 100, 200, 500]) * u.um
        probs = profile.probability(distances)
    """

    def test_logistic_basic(self):
        profile = LogisticProfile(growth_rate=0.05, midpoint=100.0 * u.um)
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)

        # At midpoint, probability should be 0.5
        self.assertAlmostEqual(probs[2], 0.5, delta=0.01)
        # Probability decreases with distance
        self.assertTrue(probs[0] > probs[1] > probs[2] > probs[3] > probs[4])

    def test_logistic_with_max_distance(self):
        profile = LogisticProfile(
            growth_rate=0.05,
            midpoint=100.0 * u.um,
            max_distance=250.0 * u.um
        )
        distances = np.array([0, 100, 200, 250, 300]) * u.um
        probs = profile.probability(distances)

        self.assertAlmostEqual(probs[1], 0.5, delta=0.01)
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)

    def test_logistic_growth_rate_effect(self):
        profile_fast = LogisticProfile(growth_rate=0.1, midpoint=100.0 * u.um)
        profile_slow = LogisticProfile(growth_rate=0.02, midpoint=100.0 * u.um)
        distances = np.array([80, 100, 120]) * u.um

        probs_fast = profile_fast.probability(distances)
        probs_slow = profile_slow.probability(distances)

        # Faster growth rate should have steeper transition
        fast_diff = probs_fast[0] - probs_fast[2]
        slow_diff = probs_slow[0] - probs_slow[2]
        self.assertTrue(fast_diff > slow_diff)

    def test_logistic_weight_scaling_matches_probability(self):
        profile = LogisticProfile(growth_rate=0.05, midpoint=100.0 * u.um)
        distances = np.array([0, 100, 200]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = LogisticProfile(growth_rate=0.05, midpoint=100.0 * u.um, max_distance=500.0 * u.um)
        repr_str = repr(profile)
        self.assertIn('LogisticProfile', repr_str)
        self.assertIn('0.05', repr_str)


class TestBimodalProfile(unittest.TestCase):
    """
    Test BimodalProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.init import BimodalProfile

        profile = BimodalProfile(
            sigma1=30.0 * u.um,
            sigma2=50.0 * u.um,
            center1=0.0 * u.um,
            center2=200.0 * u.um,
            amplitude1=1.0,
            amplitude2=0.8
        )
        distances = np.array([0, 50, 100, 200, 300]) * u.um
        probs = profile.probability(distances)
    """

    def test_bimodal_basic(self):
        profile = BimodalProfile(
            sigma1=30.0 * u.um,
            sigma2=50.0 * u.um,
            center1=0.0 * u.um,
            center2=200.0 * u.um,
            amplitude1=1.0,
            amplitude2=0.8
        )
        distances = np.array([0, 50, 100, 200, 300]) * u.um
        probs = profile.probability(distances)

        # Peak at center1 (distance 0)
        self.assertTrue(probs[0] > probs[1])
        # Peak at center2 (distance 200)
        self.assertTrue(probs[3] > probs[2])

    def test_bimodal_two_peaks(self):
        profile = BimodalProfile(
            sigma1=20.0 * u.um,
            sigma2=30.0 * u.um,
            center1=50.0 * u.um,
            center2=150.0 * u.um,
            amplitude1=1.0,
            amplitude2=1.0
        )
        distances = np.array([0, 50, 100, 150, 200]) * u.um
        probs = profile.probability(distances)

        # Both peaks should be higher than points in between
        self.assertTrue(probs[1] > probs[0])  # First peak
        self.assertTrue(probs[3] > probs[4])  # Second peak
        self.assertTrue(probs[1] > probs[2] or probs[3] > probs[2])  # Trough between peaks

    def test_bimodal_different_amplitudes(self):
        profile = BimodalProfile(
            sigma1=30.0 * u.um,
            sigma2=30.0 * u.um,
            center1=0.0 * u.um,
            center2=100.0 * u.um,
            amplitude1=1.0,
            amplitude2=0.5
        )
        distances = np.array([0, 100]) * u.um
        probs = profile.probability(distances)

        # First peak should be higher due to larger amplitude
        # Note: At center, the Gaussian is at its maximum
        self.assertTrue(probs[0] > probs[1])

    def test_bimodal_with_max_distance(self):
        profile = BimodalProfile(
            sigma1=30.0 * u.um,
            sigma2=50.0 * u.um,
            center1=0.0 * u.um,
            center2=200.0 * u.um,
            max_distance=250.0 * u.um
        )
        distances = np.array([0, 100, 200, 250, 300]) * u.um
        probs = profile.probability(distances)

        self.assertTrue(probs[0] > 0)
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)

    def test_bimodal_weight_scaling_matches_probability(self):
        profile = BimodalProfile(
            sigma1=30.0 * u.um,
            sigma2=50.0 * u.um,
            center1=0.0 * u.um,
            center2=200.0 * u.um
        )
        distances = np.array([0, 100, 200]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = BimodalProfile(
            sigma1=30.0 * u.um,
            sigma2=50.0 * u.um,
            center1=0.0 * u.um,
            center2=200.0 * u.um,
            amplitude1=1.0,
            amplitude2=0.8
        )
        repr_str = repr(profile)
        self.assertIn('BimodalProfile', repr_str)


class TestMexicanHatProfile(unittest.TestCase):
    """
    Test MexicanHatProfile distance profile.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.init import MexicanHatProfile

        profile = MexicanHatProfile(
            sigma=50.0 * u.um,
            amplitude=1.0,
            max_distance=300.0 * u.um
        )
        distances = np.array([0, 25, 50, 100, 200]) * u.um
        probs = profile.probability(distances)
    """

    def test_mexican_hat_basic(self):
        profile = MexicanHatProfile(sigma=50.0 * u.um)
        distances = np.array([0, 25, 50, 75, 100, 150]) * u.um
        probs = profile.probability(distances)

        # At distance 0, value should be at maximum
        self.assertTrue(probs[0] > 0)
        # All probabilities should be non-negative (clipped)
        self.assertTrue(np.all(probs >= 0))
        # Peak should be at or near center
        self.assertTrue(probs[0] >= probs[2])

    def test_mexican_hat_center_surround(self):
        profile = MexicanHatProfile(sigma=50.0 * u.um, amplitude=1.0)
        distances = np.array([0, 30, 60, 90, 120, 200]) * u.um
        probs = profile.probability(distances)

        # Center should be positive
        self.assertTrue(probs[0] > 0)
        # Should have a peak at center
        self.assertTrue(probs[0] > probs[3])
        # Far distances should decay to near zero
        self.assertTrue(probs[-1] < probs[0])

    def test_mexican_hat_with_max_distance(self):
        profile = MexicanHatProfile(
            sigma=50.0 * u.um,
            amplitude=1.0,
            max_distance=200.0 * u.um
        )
        distances = np.array([0, 50, 100, 200, 250]) * u.um
        probs = profile.probability(distances)

        self.assertTrue(probs[0] > 0)
        self.assertAlmostEqual(probs[4], 0.0, delta=0.001)

    def test_mexican_hat_amplitude_scaling(self):
        profile1 = MexicanHatProfile(sigma=50.0 * u.um, amplitude=1.0)
        profile2 = MexicanHatProfile(sigma=50.0 * u.um, amplitude=2.0)
        distances = np.array([0, 25, 50, 100]) * u.um

        probs1 = profile1.probability(distances)
        probs2 = profile2.probability(distances)

        # Amplitude should scale the probabilities
        self.assertTrue(np.allclose(probs2, 2.0 * probs1, rtol=0.01))

    def test_mexican_hat_sigma_effect(self):
        profile_narrow = MexicanHatProfile(sigma=30.0 * u.um)
        profile_wide = MexicanHatProfile(sigma=100.0 * u.um)
        distances = np.array([0, 30, 60, 90, 120]) * u.um

        probs_narrow = profile_narrow.probability(distances)
        probs_wide = profile_wide.probability(distances)

        # Wider sigma should have more spread out profile
        # The narrow profile should decay faster with distance
        self.assertTrue(probs_narrow[1] / probs_narrow[0] < probs_wide[1] / probs_wide[0])

    def test_mexican_hat_shape(self):
        profile = MexicanHatProfile(sigma=50.0 * u.um)
        distances = np.linspace(0, 200, 50) * u.um
        probs = profile.probability(distances)

        # Find local maximum (should be at or near d=0)
        max_idx = np.argmax(probs)
        self.assertTrue(max_idx < 5)  # Peak should be near the beginning

        # Check that values are non-negative
        self.assertTrue(np.all(probs >= 0))

    def test_mexican_hat_weight_scaling_matches_probability(self):
        profile = MexicanHatProfile(sigma=50.0 * u.um)
        distances = np.array([0, 50, 100]) * u.um
        probs = profile.probability(distances)
        weights = profile.weight_scaling(distances)
        self.assertTrue(np.allclose(probs, weights))

    def test_repr(self):
        profile = MexicanHatProfile(sigma=50.0 * u.um, amplitude=1.0, max_distance=300.0 * u.um)
        repr_str = repr(profile)
        self.assertIn('MexicanHatProfile', repr_str)
        self.assertIn('50.0', repr_str)
