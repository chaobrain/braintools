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
Comprehensive tests for DistanceModulated initialization class.

This test suite covers:
- Basic functionality with various distance profiles
- Distance computation from positions
- Integration with different base distributions
- Edge cases and error handling
- Composition with distance profiles
"""

import unittest

import brainunit as u
import numpy as np

from braintools.init import (
    DistanceModulated,
    Normal,
    Uniform,
    Constant,
    GaussianProfile,
    ExponentialProfile,
    LinearProfile,
    StepProfile,
    PowerLawProfile,
)


class TestDistanceModulatedBasic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_gaussian_profile(self):
        """Test DistanceModulated with GaussianProfile."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 50, 100, 150]) * u.um
        weights = init(4, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (4,))
        # At distance 0, weight should be close to base value (1.0 nS)
        self.assertAlmostEqual(weights[0].mantissa, 1.0, delta=0.01)
        # Weights should decay with distance
        self.assertTrue(weights[0] > weights[1] > weights[2] > weights[3])

    def test_basic_exponential_profile(self):
        """Test DistanceModulated with ExponentialProfile."""
        profile = ExponentialProfile(decay_constant=100.0 * u.um)
        base_dist = Constant(2.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 100, 200, 300]) * u.um
        weights = init(4, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (4,))
        # At distance 0, weight should equal base value
        self.assertAlmostEqual(weights[0].mantissa, 2.0, delta=0.01)
        # At decay constant distance, weight should be ~base_value / e
        self.assertAlmostEqual(weights[1].mantissa, 2.0 / np.e, delta=0.01)
        # Weights should decay with distance
        self.assertTrue(weights[0] > weights[1] > weights[2] > weights[3])

    def test_basic_linear_profile(self):
        """Test DistanceModulated with LinearProfile."""
        profile = LinearProfile(max_distance=200.0 * u.um)
        base_dist = Constant(1.5 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 50, 100, 150, 200]) * u.um
        weights = init(5, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (5,))
        # At distance 0, weight should equal base value
        self.assertAlmostEqual(weights[0].mantissa, 1.5, delta=0.01)
        # At max_distance, weight should be 0
        self.assertAlmostEqual(weights[4].mantissa, 0.0, delta=0.01)
        # Weights should decay linearly
        self.assertAlmostEqual(weights[2].mantissa, 0.75, delta=0.01)  # 50% distance

    def test_basic_step_profile(self):
        """Test DistanceModulated with StepProfile."""
        profile = StepProfile(threshold=100.0 * u.um, inside_prob=1.0, outside_prob=0.1)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([50, 100, 150]) * u.um
        weights = init(3, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (3,))
        # Inside threshold
        self.assertAlmostEqual(weights[0].mantissa, 1.0, delta=0.01)
        self.assertAlmostEqual(weights[1].mantissa, 1.0, delta=0.01)
        # Outside threshold
        self.assertAlmostEqual(weights[2].mantissa, 0.1, delta=0.01)

    def test_repr(self):
        """Test string representation."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Normal(1.0 * u.nS, 0.2 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        repr_str = repr(init)
        self.assertIn('DistanceModulated', repr_str)
        self.assertIn('base_dist', repr_str)
        self.assertIn('distance_profile', repr_str)


class TestDistanceModulatedWithPositions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_1d_positions(self):
        """Test with 1D positions."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        # Two neurons at positions 0 and 100
        pre_positions = np.array([[0], [100]]) * u.um
        post_positions = np.array([[0], [100]]) * u.um

        weights = init((2, 2), pre_positions=pre_positions, post_positions=post_positions, rng=self.rng)

        self.assertEqual(weights.shape, (2, 2))
        # Self-connections (distance 0) should have maximum weight
        self.assertAlmostEqual(weights[0, 0].mantissa, 1.0, delta=0.01)
        self.assertAlmostEqual(weights[1, 1].mantissa, 1.0, delta=0.01)
        # Cross-connections (distance 100) should be weaker
        self.assertTrue(weights[0, 0] > weights[0, 1])
        self.assertAlmostEqual(weights[0, 1].mantissa, weights[1, 0].mantissa, delta=0.01)

    def test_2d_positions(self):
        """Test with 2D positions."""
        profile = ExponentialProfile(decay_constant=100.0 * u.um)
        base_dist = Constant(2.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        # Four neurons in a square
        pre_positions = np.array([[0, 0], [100, 0]]) * u.um
        post_positions = np.array([[0, 0], [0, 100]]) * u.um

        weights = init((2, 2), pre_positions=pre_positions, post_positions=post_positions, rng=self.rng)

        self.assertEqual(weights.shape, (2, 2))
        # Distance from [0,0] to [0,0] is 0
        self.assertAlmostEqual(weights[0, 0].mantissa, 2.0, delta=0.01)
        # Distance from [0,0] to [0,100] is 100
        self.assertAlmostEqual(weights[0, 1].mantissa, 2.0 / np.e, delta=0.1)
        # Distance from [100,0] to [0,0] is 100
        self.assertAlmostEqual(weights[1, 0].mantissa, 2.0 / np.e, delta=0.1)

    def test_3d_positions(self):
        """Test with 3D positions."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        # Neurons in 3D space
        pre_positions = np.array([[0, 0, 0], [50, 0, 0]]) * u.um
        post_positions = np.array([[0, 0, 0], [0, 50, 0]]) * u.um

        weights = init((2, 2), pre_positions=pre_positions, post_positions=post_positions, rng=self.rng)

        self.assertEqual(weights.shape, (2, 2))
        # Self-connection at origin
        self.assertAlmostEqual(weights[0, 0].mantissa, 1.0, delta=0.01)
        # All weights should be positive
        self.assertTrue(np.all(weights.mantissa > 0))

    def test_missing_positions_error(self):
        """Test that missing both distances and positions raises an error."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        with self.assertRaises(ValueError) as context:
            init(4, rng=self.rng)

        self.assertIn('distances', str(context.exception))
        self.assertIn('pre_positions', str(context.exception))
        self.assertIn('post_positions', str(context.exception))

    def test_incomplete_positions_error(self):
        """Test that providing only one of pre_positions or post_positions raises an error."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        pre_positions = np.array([[0], [100]]) * u.um

        with self.assertRaises(ValueError):
            init(2, pre_positions=pre_positions, rng=self.rng)


class TestDistanceModulatedWithDistributions(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_with_normal_distribution(self):
        """Test with Normal base distribution."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Normal(1.0 * u.nS, 0.2 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 50, 100]) * u.um
        weights = init(3, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (3,))
        # All weights should be positive (with high probability)
        self.assertTrue(np.all(weights.mantissa > 0))
        # On average, weights should decrease with distance
        # This is a statistical property, so we test with more samples
        distances_large = np.repeat([0, 100], 1000) * u.um
        weights_large = init(2000, distances=distances_large, rng=self.rng)
        weights_near = weights_large[:1000]
        weights_far = weights_large[1000:]
        self.assertTrue(np.mean(weights_near.mantissa) > np.mean(weights_far.mantissa))

    def test_with_uniform_distribution(self):
        """Test with Uniform base distribution."""
        profile = ExponentialProfile(decay_constant=100.0 * u.um)
        base_dist = Uniform(0.5 * u.nS, 1.5 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 100, 200]) * u.um
        weights = init(3, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (3,))
        # All weights should be positive
        self.assertTrue(np.all(weights.mantissa > 0))
        # Weights should be modulated by distance profile
        self.assertTrue(weights[0] > weights[1] > weights[2])

    def test_large_scale(self):
        """Test with large number of connections."""
        profile = LinearProfile(max_distance=500.0 * u.um)
        base_dist = Normal(1.0 * u.nS, 0.1 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.random.uniform(0, 500, size=10000) * u.um
        weights = init(10000, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (10000,))
        # Weights at max distance should be close to 0
        far_indices = np.where(distances.mantissa > 490)
        self.assertTrue(np.mean(weights[far_indices].mantissa) < 0.05)


class TestDistanceModulatedEdgeCases(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_zero_distance(self):
        """Test with all zero distances."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.zeros(5) * u.um
        weights = init(5, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (5,))
        # All weights should be approximately equal to base value
        self.assertTrue(np.allclose(weights.mantissa, 1.0, atol=0.01))

    def test_very_large_distances(self):
        """Test with very large distances."""
        profile = ExponentialProfile(decay_constant=10.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 100, 1000, 10000]) * u.um
        weights = init(4, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (4,))
        # Very large distances should result in near-zero weights
        self.assertTrue(weights[3].mantissa < 1e-10)

    def test_max_distance_cutoff(self):
        """Test profiles with max_distance cutoff."""
        profile = GaussianProfile(sigma=50.0 * u.um, max_distance=150.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 100, 200, 300]) * u.um
        weights = init(4, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (4,))
        # Weights beyond max_distance should be 0
        self.assertAlmostEqual(weights[2].mantissa, 0.0, delta=0.01)
        self.assertAlmostEqual(weights[3].mantissa, 0.0, delta=0.01)

    def test_single_connection(self):
        """Test with single connection."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([50]) * u.um
        weights = init(1, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (1,))
        self.assertTrue(weights[0].mantissa > 0)


class TestDistanceModulatedComposition(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_composed_addition(self):
        """Test with sum of two profiles."""
        profile1 = GaussianProfile(sigma=50.0 * u.um)
        profile2 = ExponentialProfile(decay_constant=100.0 * u.um)
        composed_profile = profile1 * 0.5 + profile2 * 0.5

        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=composed_profile)

        distances = np.array([0, 50, 100]) * u.um
        weights = init(3, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (3,))
        # Weights should be modulated by composed profile
        self.assertTrue(weights[0] > weights[1] > weights[2])

    def test_composed_multiplication(self):
        """Test with product of two profiles."""
        profile1 = GaussianProfile(sigma=50.0 * u.um)
        profile2 = ExponentialProfile(decay_constant=100.0 * u.um)
        composed_profile = profile1 * profile2

        base_dist = Constant(2.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=composed_profile)

        distances = np.array([0, 50, 100]) * u.um
        weights = init(3, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (3,))
        # Weights should decrease faster due to multiplication
        self.assertTrue(weights[0] > weights[1] > weights[2])

    def test_clipped_profile(self):
        """Test with clipped profile."""
        profile = GaussianProfile(sigma=50.0 * u.um).clip(min_val=0.1, max_val=0.8)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 50, 200]) * u.um
        weights = init(3, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (3,))
        # Weight at distance 0 should be clipped to max_val
        self.assertAlmostEqual(weights[0].mantissa, 0.8, delta=0.01)
        # Weight at large distance should be clipped to min_val
        self.assertTrue(weights[2].mantissa >= 0.1)


class TestDistanceModulatedPowerLaw(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_power_law_profile(self):
        """Test with PowerLawProfile."""
        profile = PowerLawProfile(exponent=2.0, min_distance=1.0 * u.um)
        base_dist = Constant(100.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([1, 10, 100]) * u.um
        weights = init(3, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (3,))
        # Power law decay: weight(d) = base * d^(-2)
        self.assertAlmostEqual(weights[0].mantissa, 100.0, delta=1.0)
        self.assertAlmostEqual(weights[1].mantissa, 1.0, delta=0.1)
        self.assertAlmostEqual(weights[2].mantissa, 0.01, delta=0.01)

    def test_power_law_with_max_distance(self):
        """Test PowerLawProfile with max_distance cutoff."""
        profile = PowerLawProfile(exponent=2.0, min_distance=1.0 * u.um, max_distance=500.0 * u.um)
        base_dist = Constant(100.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([1, 100, 500, 1000]) * u.um
        weights = init(4, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (4,))
        # Within max_distance
        self.assertTrue(weights[0].mantissa > 0)
        self.assertTrue(weights[1].mantissa > 0)
        # Beyond max_distance
        self.assertAlmostEqual(weights[3].mantissa, 0.0, delta=0.01)


class TestDistanceModulatedMatrixShape(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_1d_array(self):
        """Test with 1D weight array."""
        profile = GaussianProfile(sigma=50.0 * u.um)
        base_dist = Normal(1.0 * u.nS, 0.1 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.array([0, 25, 50, 75, 100]) * u.um
        weights = init(5, distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (5,))
        self.assertTrue(np.all(weights.mantissa > 0))

    def test_2d_matrix(self):
        """Test with 2D weight matrix."""
        profile = ExponentialProfile(decay_constant=100.0 * u.um)
        base_dist = Constant(1.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.random.uniform(0, 200, size=(10, 20)) * u.um
        weights = init((10, 20), distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (10, 20))
        self.assertTrue(np.all(weights.mantissa >= 0))

    def test_3d_tensor(self):
        """Test with 3D weight tensor."""
        profile = LinearProfile(max_distance=100.0 * u.um)
        base_dist = Constant(2.0 * u.nS)
        init = DistanceModulated(base_dist=base_dist, distance_profile=profile)

        distances = np.random.uniform(0, 150, size=(5, 10, 20)) * u.um
        weights = init((5, 10, 20), distances=distances, rng=self.rng)

        self.assertEqual(weights.shape, (5, 10, 20))
        self.assertTrue(np.all(weights.mantissa >= 0))


if __name__ == '__main__':
    unittest.main()
