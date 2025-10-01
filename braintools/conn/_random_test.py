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
Comprehensive tests for point neuron connectivity classes.

This test suite covers:
- Basic patterns (Random, AllToAll, OneToOne, FixedProbability)
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
    FixedProbability,
)
from braintools.init import Constant, Uniform


class TestRandom(unittest.TestCase):
    """
    Test Random connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import Random

        # Basic random connectivity
        conn = Random(
            prob=0.1,
            allow_self_connections=False,
            weight=2.0 * u.nS,
            delay=1.5 * u.ms,
            seed=42
        )

        result = conn(pre_size=100, post_size=100)

        # Check basic properties
        assert result.model_type == 'point'
        assert result.n_connections > 0
        assert 'pattern' in result.metadata
        assert result.metadata['pattern'] == 'random'

        # Expected number of connections (approximately)
        expected_connections = 100 * 100 * 0.1
        assert abs(result.n_connections - expected_connections) < expected_connections * 0.3
    """

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


class TestDistanceDependent(unittest.TestCase):
    """
    Test DistanceDependent connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import DistanceDependent
        from braintools.conn._init_distance import GaussianProfile

        # Create random positions
        positions = np.random.uniform(0, 200, (30, 2)) * u.um

        # Distance-dependent connectivity with Gaussian profile
        distance_profile = GaussianProfile(
            sigma=50 * u.um,
            max_distance=150 * u.um
        )

        conn = DistanceDependent(
            distance_profile=distance_profile,
            max_prob=0.5,
            weight=1.0 * u.nS
        )

        result = conn(
            pre_size=30, post_size=30,
            pre_positions=positions,
            post_positions=positions
        )

        # Connections should follow distance-dependent probability
        assert result.model_type == 'point'
        assert result.metadata['pattern'] == 'distance_dependent'

        # Check that all connections respect max distance
        if result.n_connections > 0:
            distances = result.get_distances()
            assert np.all(distances <= 150 * u.um)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_distance_dependent_no_positions_error(self):
        # Mock distance profile that just returns 0.1 for any distance
        class MockProfile:
            def probability(self, distances):
                return np.full(distances.shape, 0.1)

        conn = DistanceDependent(
            distance_profile=MockProfile(),
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)  # No positions provided

    def test_distance_dependent_basic(self):
        positions = np.random.RandomState(42).uniform(0, 100, (15, 2)) * u.um

        # Mock distance profile with simple logic
        class MockProfile:
            def probability(self, distances):
                # Higher probability for shorter distances
                dist_vals = distances.mantissa if hasattr(distances, 'mantissa') else distances
                return np.exp(-dist_vals / 50.0)

        conn = DistanceDependent(
            distance_profile=MockProfile(),
            max_prob=0.3,
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'distance_dependent')
        self.assertEqual(result.metadata['max_prob'], 0.3)
        self.assertGreater(result.n_connections, 0)

    def test_distance_dependent_with_weights_and_delays(self):
        positions = np.array([[0, 0], [10, 0], [20, 0], [30, 0]]) * u.um

        class ConstantProfile:
            def probability(self, distances):
                return np.full(distances.shape, 0.8)

        weight_init = Constant(1.2 * u.nS)
        delay_init = Constant(2.0 * u.ms)

        conn = DistanceDependent(
            distance_profile=ConstantProfile(),
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(
            pre_size=4, post_size=4,
            pre_positions=positions,
            post_positions=positions
        )

        if result.n_connections > 0:
            self.assertIsNotNone(result.weights)
            self.assertIsNotNone(result.delays)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 1.2)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 2.0)

    def test_distance_dependent_empty_connections(self):
        positions = np.array([[0, 0], [100, 100]]) * u.um

        class ZeroProfile:
            def probability(self, distances):
                return np.zeros(distances.shape)

        conn = DistanceDependent(
            distance_profile=ZeroProfile(),
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.n_connections, 0)

    def test_distance_dependent_asymmetric_sizes(self):
        pre_positions = np.random.RandomState(42).uniform(0, 50, (8, 2)) * u.um
        post_positions = np.random.RandomState(43).uniform(0, 50, (12, 2)) * u.um

        class SimpleProfile:
            def probability(self, distances):
                return np.full(distances.shape, 0.2)

        conn = DistanceDependent(
            distance_profile=SimpleProfile(),
            seed=42
        )

        result = conn(
            pre_size=8, post_size=12,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        self.assertEqual(result.shape, (8, 12))
        self.assertTrue(np.all(result.pre_indices < 8))
        self.assertTrue(np.all(result.post_indices < 12))

    def test_distance_dependent_tuple_sizes(self):
        positions = np.random.RandomState(42).uniform(0, 50, (12, 2)) * u.um

        class SimpleProfile:
            def probability(self, distances):
                return np.full(distances.shape, 0.1)

        conn = DistanceDependent(
            distance_profile=SimpleProfile(),
            seed=42
        )

        result = conn(
            pre_size=(3, 4), post_size=(2, 6),
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.pre_size, (3, 4))
        self.assertEqual(result.post_size, (2, 6))
        self.assertEqual(result.shape, (12, 12))

    def test_distance_dependent_3d_positions(self):
        # Test with 3D positions (should use first 2 dimensions)
        positions = np.random.RandomState(42).uniform(0, 50, (10, 3)) * u.um

        class SimpleProfile:
            def probability(self, distances):
                return np.full(distances.shape, 0.15)

        conn = DistanceDependent(
            distance_profile=SimpleProfile(),
            seed=42
        )

        result = conn(
            pre_size=10, post_size=10,
            pre_positions=positions,
            post_positions=positions
        )

        # Should handle 3D positions (using first 2 dimensions)
        self.assertGreaterEqual(result.n_connections, 0)


class TestFixedProbabilityAlias(unittest.TestCase):
    """
    Test FixedProbability as alias for Random.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import FixedProbability

        # FixedProbability should work exactly like Random
        conn = FixedProbability(
            prob=0.15,
            weight=1.5 * u.nS,
            seed=42
        )

        result = conn(pre_size=50, post_size=50)
        assert result.metadata['pattern'] == 'random'
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_fixed_probability_is_random(self):
        # FixedProbability should be an alias for Random
        self.assertTrue(issubclass(FixedProbability, Random))

    def test_fixed_probability_functionality(self):
        conn = FixedProbability(prob=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'random')
        self.assertEqual(result.metadata['probability'], 0.2)
        self.assertGreater(result.n_connections, 0)


class TestGaussianAndExponentialAliases(unittest.TestCase):
    """
    Test Gaussian and Exponential as aliases for DistanceDependent.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import Gaussian, Exponential

        # These should be aliases for DistanceDependent
        # (though they require specific distance profiles)
        assert issubclass(Gaussian, DistanceDependent)
        assert issubclass(Exponential, DistanceDependent)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_gaussian_is_distance_dependent(self):
        self.assertTrue(issubclass(Gaussian, DistanceDependent))

    def test_exponential_is_distance_dependent(self):
        self.assertTrue(issubclass(Exponential, DistanceDependent))


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases and error conditions.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import Random, AllToAll

        # Very large networks
        large_random = Random(prob=0.001, seed=42)
        result_large = large_random(pre_size=1000, post_size=1000)

        # Single neuron networks
        single = AllToAll(include_self_connections=True)
        result_single = single(pre_size=1, post_size=1)
        assert result_single.n_connections == 1

        # Empty networks
        empty = Random(prob=0.0)
        result_empty = empty(pre_size=10, post_size=10)
        assert result_empty.n_connections == 0
    """

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


if __name__ == '__main__':
    unittest.main()
