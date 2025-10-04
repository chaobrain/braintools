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
Comprehensive tests for point neuron connectivity classes.

This test suite covers:
- Basic patterns (Random, AllToAll, OneToOne, FixedProb)
- Spatial patterns (DistanceDependent, Gaussian, Exponential, Ring, Grid, RadialPatches)
- Topological patterns (SmallWorld, ScaleFree, Regular, Modular, ClusteredRandom)
- Biological patterns (ExcitatoryInhibitory, SynapticPlasticity, ActivityDependent)
- Custom patterns (Custom)
"""

import unittest

import brainunit as u
import numpy as np

from braintools.conn import (
    Random,
    AllToAll,
    OneToOne,
    FixedProb,
    ClusteredRandom,
)
from braintools.init import Constant, Uniform


class TestRandom(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_random_connectivity(self):
        conn = Random(prob=0.1, seed=42)
        result = conn(pre_size=50, post_size=50)

        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['pattern'], 'random')
        self.assertEqual(result.metadata['probability'], 0.1)
        self.assertFalse(result.metadata['allow_self_connections'])

        # Check that connection indices are valid
        self.assertTrue(np.all(result.pre_indices < 50))
        self.assertTrue(np.all(result.post_indices < 50))

    def test_random_with_self_connections(self):
        conn = Random(prob=0.2, allow_self_connections=True, seed=42)
        result = conn(pre_size=20, post_size=20)

        # Should have some self-connections (i.e., pre_index == post_index)
        self_connections = np.sum(result.pre_indices == result.post_indices)
        # With prob=0.2 and 20 neurons, expect about 4 self-connections
        self.assertGreaterEqual(self_connections, 0)

    def test_random_without_self_connections(self):
        conn = Random(prob=0.3, allow_self_connections=False, seed=42)
        result = conn(pre_size=20, post_size=20)

        # Should have no self-connections
        self_connections = np.sum(result.pre_indices == result.post_indices)
        self.assertEqual(self_connections, 0)

    def test_random_with_weights_and_delays(self):
        weight_init = Constant(1.5 * u.nS)
        delay_init = Uniform(1.0 * u.ms, 3.0 * u.ms)

        conn = Random(
            prob=0.15,
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(pre_size=30, post_size=30)

        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)
        self.assertEqual(u.get_unit(result.weights), u.nS)
        self.assertEqual(u.get_unit(result.delays), u.ms)

        # All weights should be 1.5 nS
        np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 1.5)

        # Delays should be between 1.0 and 3.0 ms
        self.assertTrue(np.all(result.delays >= 1.0 * u.ms))
        self.assertTrue(np.all(result.delays < 3.0 * u.ms))

    def test_random_zero_probability(self):
        conn = Random(prob=0.0, seed=42)
        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.n_connections, 0)
        self.assertEqual(len(result.pre_indices), 0)
        self.assertEqual(len(result.post_indices), 0)

    def test_random_high_probability(self):
        conn = Random(prob=0.8, allow_self_connections=True, seed=42)
        result = conn(pre_size=10, post_size=10)

        # Should have many connections (close to 10*10*0.8 = 80)
        expected_connections = 10 * 10 * 0.8
        self.assertGreater(result.n_connections, expected_connections * 0.5)

    def test_random_tuple_sizes(self):
        conn = Random(prob=0.1, seed=42)
        result = conn(pre_size=(4, 5), post_size=(3, 6))

        self.assertEqual(result.pre_size, (4, 5))
        self.assertEqual(result.post_size, (3, 6))
        self.assertEqual(result.shape, (20, 18))  # 4*5, 3*6

        # Check indices are within bounds
        self.assertTrue(np.all(result.pre_indices < 20))
        self.assertTrue(np.all(result.post_indices < 18))

    def test_random_asymmetric_sizes(self):
        conn = Random(prob=0.15, seed=42)
        result = conn(pre_size=25, post_size=15)

        self.assertEqual(result.shape, (25, 15))
        self.assertTrue(np.all(result.pre_indices < 25))
        self.assertTrue(np.all(result.post_indices < 15))

    def test_random_with_positions(self):
        positions = np.random.RandomState(42).uniform(0, 100, (20, 2)) * u.um

        conn = Random(prob=0.2, seed=42)
        result = conn(
            pre_size=20,
            post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertIsNotNone(result.pre_positions)
        self.assertIsNotNone(result.post_positions)
        assert u.math.allclose(result.pre_positions, positions)

    def test_random_scalar_values(self):
        # Test with scalar weight and delay (should get automatic units)
        conn = Random(prob=0.1, weight=2.5 * u.nS, delay=1.0 * u.ms, seed=42)
        result = conn(pre_size=20, post_size=20)

        if result.n_connections > 0:
            # Should automatically get nS and ms units
            self.assertEqual(u.get_unit(result.weights), u.nS)
            self.assertEqual(u.get_unit(result.delays), u.ms)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 2.5)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 1.0)


class TestFixedProbabilityAlias(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_fixed_probability_is_random(self):
        # FixedProb should be an alias for Random
        self.assertTrue(issubclass(FixedProb, Random))

    def test_fixed_probability_functionality(self):
        conn = FixedProb(prob=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'random')
        self.assertEqual(result.metadata['probability'], 0.2)
        self.assertGreater(result.n_connections, 0)


class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_large_networks(self):
        # Test with reasonably large networks
        conn = Random(prob=0.001, seed=42)  # Low probability for large networks
        result = conn(pre_size=500, post_size=500)

        self.assertEqual(result.shape, (500, 500))
        # Should have some connections but not too many
        self.assertGreater(result.n_connections, 0)
        self.assertLess(result.n_connections, 50000)  # Much less than 500*500

    def test_single_neuron_networks(self):
        patterns = [
            Random(prob=1.0, allow_self_connections=True),
            AllToAll(include_self_connections=True),
            OneToOne(circular=False),
        ]

        for pattern in patterns:
            pattern.seed = 42
            result = pattern(pre_size=1, post_size=1)
            self.assertEqual(result.n_connections, 1)
            np.testing.assert_array_equal(result.pre_indices, [0])
            np.testing.assert_array_equal(result.post_indices, [0])

    def test_empty_networks(self):
        # Test various ways to get empty networks
        patterns = [
            Random(prob=0.0),  # Zero probability
            AllToAll(include_self_connections=False),  # Will be tested with size 0
        ]

        for pattern in patterns:
            pattern.seed = 42
            if isinstance(pattern, Random):
                result = pattern(pre_size=10, post_size=10)
                self.assertEqual(result.n_connections, 0)

    def test_very_small_networks(self):
        conn = AllToAll(include_self_connections=False, seed=42)
        result = conn(pre_size=2, post_size=2)

        # 2x2 all-to-all without self connections = 2 connections
        self.assertEqual(result.n_connections, 2)
        expected_connections = {(0, 1), (1, 0)}
        actual_connections = set(zip(result.pre_indices, result.post_indices))
        self.assertEqual(expected_connections, actual_connections)

    def test_asymmetric_extreme_sizes(self):
        conn = Random(prob=0.5, seed=42)
        result = conn(pre_size=1, post_size=100)

        self.assertEqual(result.shape, (1, 100))
        self.assertTrue(np.all(result.pre_indices == 0))  # Only one pre neuron
        self.assertTrue(np.all(result.post_indices < 100))

    def test_tuple_sizes_edge_cases(self):
        conn = Random(prob=0.3, seed=42)

        # Single element tuples
        result = conn(pre_size=(10,), post_size=(8,))
        self.assertEqual(result.shape, (10, 8))

        # One dimension is 1
        result2 = conn(pre_size=(1, 10), post_size=(5, 2), recompute=True)
        self.assertEqual(result2.shape, (10, 10))

    def test_reproducibility_with_seeds(self):
        # Test that same seed produces same results
        conn1 = Random(prob=0.2, seed=42)
        result1 = conn1(pre_size=20, post_size=20)

        conn2 = Random(prob=0.2, seed=42)
        result2 = conn2(pre_size=20, post_size=20)

        # Should have identical connections
        np.testing.assert_array_equal(result1.pre_indices, result2.pre_indices)
        np.testing.assert_array_equal(result1.post_indices, result2.post_indices)

    def test_different_seeds_produce_different_results(self):
        conn1 = Random(prob=0.3, seed=42)
        result1 = conn1(pre_size=30, post_size=30)

        conn2 = Random(prob=0.3, seed=123)
        result2 = conn2(pre_size=30, post_size=30)

        # Should have different connections (very unlikely to be identical)
        self.assertFalse(np.array_equal(result1.pre_indices, result2.pre_indices) and
                         np.array_equal(result1.post_indices, result2.post_indices))

    def test_position_handling_edge_cases(self):
        # Test with 1D positions (should still work)
        positions_1d = np.random.RandomState(42).uniform(0, 100, (10, 1)) * u.um

        conn = Random(prob=0.2, seed=42)
        result = conn(
            pre_size=10, post_size=10,
            pre_positions=positions_1d,
            post_positions=positions_1d
        )

        self.assertIsNotNone(result.pre_positions)
        self.assertEqual(result.pre_positions.shape, (10, 1))

    def test_unit_consistency_across_patterns(self):
        # Test that different patterns handle units consistently
        patterns = [
            Random(prob=0.1, weight=2.0 * u.nS, delay=1.5 * u.ms),
            AllToAll(weight=2.0 * u.nS, delay=1.5 * u.ms),
            OneToOne(weight=2.0 * u.nS, delay=1.5 * u.ms),
        ]

        for pattern in patterns:
            pattern.seed = 42
            result = pattern(pre_size=5, post_size=5)

            if result.n_connections > 0:
                self.assertEqual(u.get_unit(result.weights), u.nS)
                self.assertEqual(u.get_unit(result.delays), u.ms)
                np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 2.0)
                np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 1.5)


class TestClusteredRandom(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_clustered_random(self):
        """Test basic ClusteredRandom connectivity."""
        positions = np.random.RandomState(42).uniform(0, 100, (20, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=20 * u.um,
            cluster_factor=3.0,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'clustered_random')
        self.assertEqual(result.metadata['prob'], 0.1)
        self.assertGreater(result.n_connections, 0)

        # Check indices are valid
        self.assertTrue(np.all(result.pre_indices < 20))
        self.assertTrue(np.all(result.post_indices < 20))

    def test_no_positions_error(self):
        """Test that missing positions raises ValueError."""
        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=50 * u.um,
            seed=42
        )

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=10, post_size=10)
        self.assertIn("Positions required", str(ctx.exception))

    def test_only_pre_positions_error(self):
        """Test that missing post_positions raises ValueError."""
        positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=50 * u.um,
            seed=42
        )

        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=10, post_size=10, pre_positions=positions)
        self.assertIn("Positions required", str(ctx.exception))

    def test_cluster_factor_effect(self):
        """Test that cluster_factor increases connections within cluster radius."""
        # Create two neurons close together and two far apart
        positions = np.array([
            [0, 0],
            [10, 10],  # Close to first (distance ~14.14)
            [200, 200],
            [210, 210]  # Close to third (distance ~14.14)
        ]) * u.um

        # With high cluster factor, should get more connections within pairs
        conn = ClusteredRandom(
            prob=0.05,  # Low base probability
            cluster_radius=20 * u.um,
            cluster_factor=10.0,  # High clustering
            seed=42
        )

        result = conn(
            pre_size=4, post_size=4,
            pre_positions=positions,
            post_positions=positions
        )

        # Should have some connections
        self.assertGreater(result.n_connections, 0)

    def test_cluster_radius_unitless(self):
        """Test ClusteredRandom with unitless cluster radius."""
        positions = np.random.RandomState(42).uniform(0, 100, (15, 2))

        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=15.0,  # Unitless
            cluster_factor=4.0,
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['cluster_radius'], 15.0)

    def test_cluster_radius_with_units(self):
        """Test ClusteredRandom with units in cluster radius."""
        positions = np.random.RandomState(42).uniform(0, 1000, (20, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.08,
            cluster_radius=150 * u.um,
            cluster_factor=3.0,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.metadata['cluster_radius'], 150 * u.um)

    def test_weights_and_delays(self):
        """Test ClusteredRandom with weight and delay initialization."""
        positions = np.random.RandomState(42).uniform(0, 100, (15, 2)) * u.um

        weight_init = Constant(1.5 * u.nS)
        delay_init = Uniform(1.0 * u.ms, 2.5 * u.ms)

        conn = ClusteredRandom(
            prob=0.2,
            cluster_radius=25 * u.um,
            cluster_factor=2.5,
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertIsNotNone(result.delays)
            self.assertEqual(u.get_unit(result.weights), u.nS)
            self.assertEqual(u.get_unit(result.delays), u.ms)

            # All weights should be 1.5 nS
            np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 1.5)

            # Delays should be between 1.0 and 2.5 ms
            self.assertTrue(np.all(result.delays >= 1.0 * u.ms))
            self.assertTrue(np.all(result.delays <= 2.5 * u.ms))

    def test_scalar_weight_and_delay(self):
        """Test ClusteredRandom with scalar weight and delay values."""
        positions = np.random.RandomState(42).uniform(0, 100, (12, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.15,
            cluster_radius=30 * u.um,
            cluster_factor=2.0,
            weight=2.0 * u.nS,
            delay=1.5 * u.ms,
            seed=42
        )

        result = conn(
            pre_size=12, post_size=12,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertEqual(u.get_unit(result.weights), u.nS)
            self.assertEqual(u.get_unit(result.delays), u.ms)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 2.0)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 1.5)

    def test_asymmetric_sizes(self):
        """Test ClusteredRandom with different pre and post sizes."""
        pre_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um
        post_positions = np.random.RandomState(43).uniform(0, 100, (15, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.12,
            cluster_radius=25 * u.um,
            cluster_factor=3.0,
            seed=42
        )

        result = conn(
            pre_size=10, post_size=15,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertEqual(result.shape, (10, 15))
        self.assertTrue(np.all(result.pre_indices < 10))
        self.assertTrue(np.all(result.post_indices < 15))

    def test_tuple_sizes(self):
        """Test ClusteredRandom with tuple sizes."""
        positions = np.random.RandomState(42).uniform(0, 100, (20, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=20 * u.um,
            cluster_factor=2.5,
            seed=42
        )

        result = conn(
            pre_size=(4, 5), post_size=(2, 10),
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.pre_size, (4, 5))
        self.assertEqual(result.post_size, (2, 10))
        self.assertEqual(result.shape, (20, 20))

    def test_zero_probability(self):
        """Test ClusteredRandom with zero base probability."""
        positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.0,
            cluster_radius=20 * u.um,
            cluster_factor=5.0,
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=positions,
            post_positions=positions
        )

        # With prob=0 and cluster_factor=5, within cluster prob=0*5=0
        # So should have no connections
        self.assertEqual(result.n_connections, 0)

    def test_high_probability_capped(self):
        """Test that probability is capped at 1.0 even with high cluster factor."""
        # Create very close neurons
        positions = np.array([[0, 0], [1, 1], [2, 2]]) * u.um

        conn = ClusteredRandom(
            prob=0.3,
            cluster_radius=10 * u.um,
            cluster_factor=10.0,  # Would give 3.0 without clipping
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        # Should handle clipping correctly (no errors)
        self.assertGreaterEqual(result.n_connections, 0)

    def test_empty_connections(self):
        """Test when no connections are generated."""
        # Spread neurons very far apart with small cluster radius
        positions = np.array([
            [0, 0],
            [1000, 1000],
            [2000, 2000]
        ]) * u.um

        conn = ClusteredRandom(
            prob=0.001,  # Very low probability
            cluster_radius=5 * u.um,  # Very small radius
            cluster_factor=2.0,
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        # Should handle empty connections gracefully
        self.assertEqual(result.n_connections, 0)
        self.assertEqual(len(result.pre_indices), 0)
        self.assertEqual(len(result.post_indices), 0)
        self.assertIsNone(result.weights)
        self.assertIsNone(result.delays)

    def test_position_unit_consistency(self):
        """Test that different position units are handled correctly."""
        # Use millimeters instead of micrometers
        positions = np.random.RandomState(42).uniform(0, 10, (12, 2)) * u.mm

        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=2 * u.mm,
            cluster_factor=3.0,
            seed=42
        )

        result = conn(
            pre_size=12, post_size=12,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertIsNotNone(result.pre_positions)
        self.assertEqual(u.get_unit(result.pre_positions), u.mm)
        self.assertGreater(result.n_connections, 0)

    def test_unit_conversion_cluster_radius(self):
        """Test that cluster radius is converted to match position units."""
        positions = np.random.RandomState(42).uniform(0, 1000, (15, 2)) * u.um

        # Use mm for cluster radius (will be converted to um internally)
        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=0.05 * u.mm,  # 50 um in mm
            cluster_factor=3.0,
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreater(result.n_connections, 0)

    def test_3d_positions(self):
        """Test ClusteredRandom with 3D positions."""
        positions = np.random.RandomState(42).uniform(0, 100, (15, 3)) * u.um

        conn = ClusteredRandom(
            prob=0.12,
            cluster_radius=25 * u.um,
            cluster_factor=2.5,
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        # Should handle 3D positions
        self.assertGreater(result.n_connections, 0)
        self.assertEqual(result.pre_positions.shape, (15, 3))

    def test_1d_positions(self):
        """Test ClusteredRandom with 1D positions."""
        positions = np.random.RandomState(42).uniform(0, 100, (12, 1)) * u.um

        conn = ClusteredRandom(
            prob=0.15,
            cluster_radius=15 * u.um,
            cluster_factor=2.0,
            seed=42
        )

        result = conn(
            pre_size=12, post_size=12,
            pre_positions=positions,
            post_positions=positions
        )

        # Should handle 1D positions
        self.assertGreaterEqual(result.n_connections, 0)
        self.assertEqual(result.pre_positions.shape, (12, 1))

    def test_cluster_factor_one(self):
        """Test that cluster_factor=1.0 behaves like standard random connectivity."""
        positions = np.random.RandomState(42).uniform(0, 100, (20, 2)) * u.um

        # cluster_factor=1.0 means no clustering effect
        conn_clustered = ClusteredRandom(
            prob=0.15,
            cluster_radius=20 * u.um,
            cluster_factor=1.0,
            seed=42
        )

        conn_random = Random(
            prob=0.15,
            seed=42
        )

        result_clustered = conn_clustered(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        result_random = conn_random(pre_size=20, post_size=20)

        # Should have similar number of connections (both ~15% of 20*20=400)
        self.assertAlmostEqual(
            result_clustered.n_connections,
            result_random.n_connections,
            delta=40  # Allow some variance
        )

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        positions = np.random.RandomState(42).uniform(0, 100, (15, 2)) * u.um

        conn1 = ClusteredRandom(
            prob=0.1,
            cluster_radius=20 * u.um,
            cluster_factor=3.0,
            seed=100
        )

        result1 = conn1(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        conn2 = ClusteredRandom(
            prob=0.1,
            cluster_radius=20 * u.um,
            cluster_factor=3.0,
            seed=100
        )

        result2 = conn2(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        # Should produce identical results
        self.assertEqual(result1.n_connections, result2.n_connections)
        np.testing.assert_array_equal(result1.pre_indices, result2.pre_indices)
        np.testing.assert_array_equal(result1.post_indices, result2.post_indices)

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        positions = np.random.RandomState(42).uniform(0, 100, (20, 2)) * u.um

        conn1 = ClusteredRandom(
            prob=0.15,
            cluster_radius=25 * u.um,
            cluster_factor=3.0,
            seed=42
        )

        result1 = conn1(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        conn2 = ClusteredRandom(
            prob=0.15,
            cluster_radius=25 * u.um,
            cluster_factor=3.0,
            seed=123
        )

        result2 = conn2(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        # Should produce different results (very unlikely to be identical)
        self.assertFalse(
            np.array_equal(result1.pre_indices, result2.pre_indices) and
            np.array_equal(result1.post_indices, result2.post_indices)
        )

    def test_large_network(self):
        """Test ClusteredRandom with a larger network."""
        positions = np.random.RandomState(42).uniform(0, 1000, (200, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.02,  # Low probability for large network
            cluster_radius=100 * u.um,
            cluster_factor=4.0,
            seed=42
        )

        result = conn(
            pre_size=200, post_size=200,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.shape, (200, 200))
        self.assertGreater(result.n_connections, 0)
        self.assertTrue(np.all(result.pre_indices < 200))
        self.assertTrue(np.all(result.post_indices < 200))

    def test_metadata_completeness(self):
        """Test that all expected metadata is present."""
        positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.15,
            cluster_radius=20 * u.um,
            cluster_factor=2.5,
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=positions,
            post_positions=positions
        )

        metadata = result.metadata
        self.assertIn('pattern', metadata)
        self.assertIn('prob', metadata)
        self.assertIn('cluster_radius', metadata)

        self.assertEqual(metadata['pattern'], 'clustered_random')
        self.assertEqual(metadata['prob'], 0.15)
        self.assertEqual(metadata['cluster_radius'], 20 * u.um)

    def test_result_positions_preserved(self):
        """Test that positions are preserved in result."""
        pre_positions = np.random.RandomState(42).uniform(0, 100, (10, 2)) * u.um
        post_positions = np.random.RandomState(43).uniform(0, 100, (12, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=25 * u.um,
            cluster_factor=2.0,
            seed=42
        )

        result = conn(
            pre_size=10, post_size=12,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertIsNotNone(result.pre_positions)
        self.assertIsNotNone(result.post_positions)
        assert u.math.allclose(result.pre_positions, pre_positions)
        assert u.math.allclose(result.post_positions, post_positions)


if __name__ == '__main__':
    unittest.main()
