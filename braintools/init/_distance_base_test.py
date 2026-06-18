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
    DistanceProfile,
    GaussianProfile,
    ExponentialProfile,
    PowerLawProfile,
    LinearProfile,
    StepProfile,
)


class _ProbVsWeightProfile(DistanceProfile):
    """Helper whose probability (1.0) differs from its weight_scaling (2.0)."""

    def probability(self, distances):
        return np.ones_like(np.asarray(u.get_mantissa(distances)), dtype=float)

    def weight_scaling(self, distances):
        return 2.0 * np.ones_like(np.asarray(u.get_mantissa(distances)), dtype=float)


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


class TestComposedProfileMethods(unittest.TestCase):
    """ComposedProfile must keep probability and weight_scaling distinct (bug M5)."""

    def test_composed_probability_uses_probability(self):
        composed = _ProbVsWeightProfile() * 1.0
        d = np.array([10.0, 20.0]) * u.um
        np.testing.assert_allclose(np.asarray(composed.probability(d)), 1.0)

    def test_composed_weight_scaling_uses_weight_scaling(self):
        composed = _ProbVsWeightProfile() * 1.0
        d = np.array([10.0, 20.0]) * u.um
        np.testing.assert_allclose(np.asarray(composed.weight_scaling(d)), 2.0)


class TestPipeProfileRejectsProfile(unittest.TestCase):
    """PipeProfile cannot pipe into a DistanceProfile right operand (bug H4)."""

    def test_pipe_rejects_profile_rhs_probability(self):
        piped = GaussianProfile(50.0 * u.um) | ExponentialProfile(100.0 * u.um)
        d = np.array([10.0]) * u.um
        with self.assertRaises(TypeError):
            piped.probability(d)

    def test_pipe_rejects_profile_rhs_weight_scaling(self):
        piped = GaussianProfile(50.0 * u.um) | ExponentialProfile(100.0 * u.um)
        d = np.array([10.0]) * u.um
        with self.assertRaises(TypeError):
            piped.weight_scaling(d)

    def test_pipe_accepts_callable(self):
        piped = GaussianProfile(50.0 * u.um) | (lambda x: x * 2.0)
        d = np.array([0.0]) * u.um
        np.testing.assert_allclose(np.asarray(piped.probability(d)), 2.0, atol=1e-5)


class TestProfileArithmetic(unittest.TestCase):
    """All overloaded arithmetic operators must compose profiles correctly."""

    def setUp(self):
        self.d = np.array([0.0, 50.0]) * u.um
        self.g = GaussianProfile(sigma=50.0 * u.um)

    def _vals(self, profile):
        return np.asarray(profile.probability(self.d))

    def test_add_and_radd(self):
        base = self._vals(self.g)
        np.testing.assert_allclose(self._vals(self.g + 0.5), base + 0.5, rtol=1e-6)
        np.testing.assert_allclose(self._vals(0.5 + self.g), base + 0.5, rtol=1e-6)

    def test_sub_and_rsub(self):
        base = self._vals(self.g)
        np.testing.assert_allclose(self._vals(self.g - 0.25), base - 0.25, rtol=1e-6)
        np.testing.assert_allclose(self._vals(1.0 - self.g), 1.0 - base, rtol=1e-6)

    def test_mul_and_rmul(self):
        base = self._vals(self.g)
        np.testing.assert_allclose(self._vals(self.g * 2.0), base * 2.0, rtol=1e-6)
        np.testing.assert_allclose(self._vals(2.0 * self.g), base * 2.0, rtol=1e-6)

    def test_truediv_and_rtruediv(self):
        base = self._vals(self.g)
        np.testing.assert_allclose(self._vals(self.g / 2.0), base / 2.0, rtol=1e-6)
        # rtruediv: 1 / profile (base[0] == 1.0 at distance 0, so finite).
        np.testing.assert_allclose(self._vals(1.0 / self.g), 1.0 / base, rtol=1e-6)

    def test_compose_two_profiles_subtraction(self):
        other = ExponentialProfile(decay_constant=50.0 * u.um)
        composed = self.g - other
        np.testing.assert_allclose(
            self._vals(composed),
            self._vals(self.g) - self._vals(other),
            rtol=1e-6,
        )

    def test_evaluate_rejects_bad_operand(self):
        # A non-profile, non-array operand must raise from _evaluate.
        composed = self.g * object()
        with self.assertRaises(TypeError):
            composed.probability(self.d)

    def test_composed_repr(self):
        self.assertIn('+', repr(self.g + 0.5))


class TestClipProfile(unittest.TestCase):
    """ClipProfile must bound both probability and weight_scaling."""

    def setUp(self):
        self.d = np.array([0.0, 50.0, 200.0]) * u.um
        self.profile = GaussianProfile(sigma=50.0 * u.um).clip(0.1, 0.8)

    def test_probability_bounded(self):
        probs = np.asarray(self.profile.probability(self.d))
        self.assertTrue(np.all(probs >= 0.1 - 1e-9))
        self.assertTrue(np.all(probs <= 0.8 + 1e-9))

    def test_weight_scaling_bounded(self):
        ws = np.asarray(self.profile.weight_scaling(self.d))
        self.assertTrue(np.all(ws >= 0.1 - 1e-9))
        self.assertTrue(np.all(ws <= 0.8 + 1e-9))

    def test_clip_only_min(self):
        prof = GaussianProfile(sigma=50.0 * u.um).clip(min_val=0.2)
        probs = np.asarray(prof.probability(self.d))
        self.assertTrue(np.all(probs >= 0.2 - 1e-9))

    def test_clip_only_max(self):
        prof = GaussianProfile(sigma=50.0 * u.um).clip(max_val=0.5)
        probs = np.asarray(prof.probability(self.d))
        self.assertTrue(np.all(probs <= 0.5 + 1e-9))

    def test_repr(self):
        self.assertIn('clip', repr(self.profile))


class TestApplyProfile(unittest.TestCase):
    """ApplyProfile must transform both probability and weight_scaling."""

    def setUp(self):
        self.d = np.array([0.0, 50.0]) * u.um
        self.base = GaussianProfile(sigma=50.0 * u.um)
        self.applied = self.base.apply(lambda x: x ** 2)

    def test_probability_transformed(self):
        np.testing.assert_allclose(
            np.asarray(self.applied.probability(self.d)),
            np.asarray(self.base.probability(self.d)) ** 2,
            rtol=1e-6,
        )

    def test_weight_scaling_transformed(self):
        np.testing.assert_allclose(
            np.asarray(self.applied.weight_scaling(self.d)),
            np.asarray(self.base.weight_scaling(self.d)) ** 2,
            rtol=1e-6,
        )

    def test_repr(self):
        self.assertIn('apply', repr(self.applied))


class TestPipeProfileExtras(unittest.TestCase):
    """PipeProfile weight_scaling, repr, and non-callable rejection."""

    def test_weight_scaling_applies_func(self):
        piped = GaussianProfile(50.0 * u.um) | (lambda x: x * 3.0)
        d = np.array([0.0]) * u.um
        np.testing.assert_allclose(np.asarray(piped.weight_scaling(d)), 3.0, atol=1e-5)

    def test_non_callable_rhs_raises(self):
        piped = GaussianProfile(50.0 * u.um) | 5  # int is neither profile nor callable
        with self.assertRaises(TypeError):
            piped.probability(np.array([0.0]) * u.um)

    def test_repr(self):
        piped = GaussianProfile(50.0 * u.um) | (lambda x: x)
        self.assertIn('|', repr(piped))


if __name__ == '__main__':
    unittest.main()
