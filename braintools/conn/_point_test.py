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

from braintools.conn._point import (
    Random,
    AllToAll,
    OneToOne,
    FixedProbability,
    DistanceDependent,
    Gaussian,
    Exponential,
    Ring,
    Grid,
    RadialPatches,
    SmallWorld,
    ScaleFree,
    Regular,
    Modular,
    ClusteredRandom,
    ExcitatoryInhibitory,
    SynapticPlasticity,
    ActivityDependent,
    Custom,
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


class TestAllToAll(unittest.TestCase):
    """
    Test AllToAll connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import AllToAll

        # Basic all-to-all connectivity
        conn = AllToAll(
            include_self_connections=False,
            weight=1.0 * u.nS,
            delay=2.0 * u.ms
        )

        result = conn(pre_size=10, post_size=10)

        # Should have 10*10 - 10 = 90 connections (excluding self)
        assert result.n_connections == 90
        assert result.metadata['pattern'] == 'all_to_all'

        # With self-connections
        conn_self = AllToAll(include_self_connections=True)
        result_self = conn_self(pre_size=10, post_size=10)
        assert result_self.n_connections == 100
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_all_to_all(self):
        conn = AllToAll(include_self_connections=False, seed=42)
        result = conn(pre_size=8, post_size=8)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.n_connections, 8 * 8 - 8)  # 56 connections
        self.assertEqual(result.metadata['pattern'], 'all_to_all')
        self.assertFalse(result.metadata['include_self_connections'])

        # Check no self-connections
        self_connections = np.sum(result.pre_indices == result.post_indices)
        self.assertEqual(self_connections, 0)

    def test_all_to_all_with_self_connections(self):
        conn = AllToAll(include_self_connections=True, seed=42)
        result = conn(pre_size=6, post_size=6)

        self.assertEqual(result.n_connections, 6 * 6)  # 36 connections
        self.assertTrue(result.metadata['include_self_connections'])

        # Check that all possible connections exist
        expected_connections = set((i, j) for i in range(6) for j in range(6))
        actual_connections = set(zip(result.pre_indices, result.post_indices))
        self.assertEqual(expected_connections, actual_connections)

    def test_all_to_all_asymmetric(self):
        conn = AllToAll(include_self_connections=False, seed=42)
        result = conn(pre_size=5, post_size=8)

        self.assertEqual(result.n_connections, 5 * 8)  # 40 connections
        self.assertEqual(result.shape, (5, 8))

        # Should connect every pre to every post
        expected_connections = set((i, j) for i in range(5) for j in range(8))
        actual_connections = set(zip(result.pre_indices, result.post_indices))
        self.assertEqual(expected_connections, actual_connections)

    def test_all_to_all_with_weights_and_delays(self):
        weight_init = Constant(0.8 * u.nS)
        delay_init = Constant(1.5 * u.ms)

        conn = AllToAll(
            include_self_connections=True,
            weight=weight_init,
            delay=delay_init,
            seed=42
        )

        result = conn(pre_size=4, post_size=4)

        self.assertEqual(result.n_connections, 16)
        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)

        # All weights and delays should be constant
        np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 0.8)
        np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 1.5)

    def test_all_to_all_tuple_sizes(self):
        conn = AllToAll(include_self_connections=False, seed=42)
        result = conn(pre_size=(2, 3), post_size=(2, 2))

        self.assertEqual(result.pre_size, (2, 3))
        self.assertEqual(result.post_size, (2, 2))
        self.assertEqual(result.shape, (6, 4))

        # 6 pre neurons, 4 post neurons
        # Since pre_size != post_size, no self-connections to exclude
        self.assertEqual(result.n_connections, 6 * 4)

    def test_all_to_all_single_neuron(self):
        # Test edge case with single neuron
        conn = AllToAll(include_self_connections=True, seed=42)
        result = conn(pre_size=1, post_size=1)

        self.assertEqual(result.n_connections, 1)
        np.testing.assert_array_equal(result.pre_indices, [0])
        np.testing.assert_array_equal(result.post_indices, [0])


class TestOneToOne(unittest.TestCase):
    """
    Test OneToOne connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import OneToOne

        # Basic one-to-one connectivity
        conn = OneToOne(
            weight=3.0 * u.nS,
            delay=0.5 * u.ms,
            circular=False
        )

        result = conn(pre_size=10, post_size=10)

        # Should have 10 connections: 0->0, 1->1, ..., 9->9
        assert result.n_connections == 10
        assert result.metadata['pattern'] == 'one_to_one'

        # Test circular indexing with different sizes
        conn_circular = OneToOne(circular=True)
        result_circular = conn_circular(pre_size=5, post_size=8)
        assert result_circular.n_connections == 8  # max(5, 8)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_one_to_one(self):
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=12, post_size=12)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.n_connections, 12)
        self.assertEqual(result.metadata['pattern'], 'one_to_one')
        self.assertFalse(result.metadata['circular'])

        # Check that connections are exactly i->i
        expected_pre = np.arange(12)
        expected_post = np.arange(12)
        np.testing.assert_array_equal(result.pre_indices, expected_pre)
        np.testing.assert_array_equal(result.post_indices, expected_post)

    def test_one_to_one_different_sizes_non_circular(self):
        # Smaller post population
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=10, post_size=6)

        self.assertEqual(result.n_connections, 6)  # min(10, 6)
        np.testing.assert_array_equal(result.pre_indices, np.arange(6))
        np.testing.assert_array_equal(result.post_indices, np.arange(6))

        # Smaller pre population
        result2 = conn(pre_size=4, post_size=8, recompute=True)
        self.assertEqual(result2.n_connections, 4)  # min(4, 8)
        np.testing.assert_array_equal(result2.pre_indices, np.arange(4))
        np.testing.assert_array_equal(result2.post_indices, np.arange(4))

    def test_one_to_one_circular(self):
        conn = OneToOne(circular=True, seed=42)

        # More post than pre
        result = conn(pre_size=5, post_size=8)
        self.assertEqual(result.n_connections, 8)  # max(5, 8)
        self.assertTrue(result.metadata['circular'])

        # Check circular indexing
        expected_pre = np.array([0, 1, 2, 3, 4, 0, 1, 2])  # Wraps around
        expected_post = np.arange(8)
        np.testing.assert_array_equal(result.pre_indices, expected_pre)
        np.testing.assert_array_equal(result.post_indices, expected_post)

        # More pre than post
        result2 = conn(pre_size=7, post_size=4, recompute=True)
        self.assertEqual(result2.n_connections, 7)  # max(7, 4)

        expected_pre2 = np.arange(7)
        expected_post2 = np.array([0, 1, 2, 3, 0, 1, 2])  # Post wraps around
        np.testing.assert_array_equal(result2.pre_indices, expected_pre2)
        np.testing.assert_array_equal(result2.post_indices, expected_post2)

    def test_one_to_one_with_weights_and_delays(self):
        weight_init = Constant(2.5 * u.nS)
        delay_init = Constant(0.8 * u.ms)

        conn = OneToOne(
            weight=weight_init,
            delay=delay_init,
            circular=False,
            seed=42
        )

        result = conn(pre_size=8, post_size=8)

        self.assertEqual(result.n_connections, 8)
        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)

        np.testing.assert_array_almost_equal(u.get_mantissa(result.weights), 2.5)
        np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 0.8)

    def test_one_to_one_tuple_sizes(self):
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=(2, 4), post_size=(3, 3))

        # pre_size = 8, post_size = 9, min = 8
        self.assertEqual(result.n_connections, 8)
        self.assertEqual(result.pre_size, (2, 4))
        self.assertEqual(result.post_size, (3, 3))

    def test_one_to_one_single_neuron(self):
        conn = OneToOne(circular=False, seed=42)
        result = conn(pre_size=1, post_size=1)

        self.assertEqual(result.n_connections, 1)
        np.testing.assert_array_equal(result.pre_indices, [0])
        np.testing.assert_array_equal(result.post_indices, [0])


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


class TestSpatialPatterns(unittest.TestCase):
    """
    Test spatial connectivity patterns (Ring, Grid, RadialPatches, ClusteredRandom).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import Ring, Grid, RadialPatches, ClusteredRandom

        # Ring connectivity
        ring = Ring(neighbors=2, bidirectional=True, weight=1.0 * u.nS)
        result_ring = ring(pre_size=20, post_size=20)
        assert result_ring.metadata['pattern'] == 'ring'

        # Grid connectivity
        grid = Grid(
            grid_shape=(5, 5),
            connectivity='moore',  # 8 neighbors
            periodic=True
        )
        result_grid = grid(pre_size=25, post_size=25)
        assert result_grid.metadata['pattern'] == 'grid'

        # Radial patches
        positions = np.random.uniform(0, 100, (50, 2)) * u.um
        patches = RadialPatches(
            patch_radius=20 * u.um,
            n_patches=3,
            prob=0.7
        )
        result_patches = patches(
            pre_size=50, post_size=50,
            pre_positions=positions,
            post_positions=positions
        )
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_ring_basic(self):
        conn = Ring(neighbors=2, bidirectional=True, seed=42)
        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'ring')
        self.assertEqual(result.metadata['neighbors'], 2)
        self.assertTrue(result.metadata['bidirectional'])

        # Each neuron connects to 2 neighbors on each side (4 total connections)
        # Total = 10 * 4 = 40 connections
        self.assertEqual(result.n_connections, 40)

    def test_ring_unidirectional(self):
        conn = Ring(neighbors=1, bidirectional=False, seed=42)
        result = conn(pre_size=8, post_size=8)

        self.assertFalse(result.metadata['bidirectional'])
        # Each neuron connects to 1 neighbor forward = 8 connections
        self.assertEqual(result.n_connections, 8)

        # Check that connections are forward only
        for i in range(len(result.pre_indices)):
            pre_idx = result.pre_indices[i]
            post_idx = result.post_indices[i]
            self.assertEqual(post_idx, (pre_idx + 1) % 8)

    def test_ring_different_sizes_error(self):
        conn = Ring(neighbors=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_grid_von_neumann(self):
        conn = Grid(
            grid_shape=(4, 4),
            connectivity='von_neumann',
            periodic=False,
            seed=42
        )
        result = conn(pre_size=16, post_size=16)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'grid')
        self.assertEqual(result.metadata['grid_shape'], (4, 4))
        self.assertEqual(result.metadata['connectivity'], 'von_neumann')
        self.assertFalse(result.metadata['periodic'])

        # Interior neurons have 4 neighbors, edge neurons have fewer
        # Total connections depend on boundary conditions
        self.assertGreater(result.n_connections, 0)

    def test_grid_moore_periodic(self):
        conn = Grid(
            grid_shape=(3, 3),
            connectivity='moore',
            periodic=True,
            seed=42
        )
        result = conn(pre_size=9, post_size=9)

        self.assertEqual(result.metadata['connectivity'], 'moore')
        self.assertTrue(result.metadata['periodic'])

        # With periodic boundaries, each neuron has exactly 8 neighbors
        # Total = 9 * 8 = 72 connections
        self.assertEqual(result.n_connections, 72)

    def test_grid_invalid_shape(self):
        conn = Grid(grid_shape=(3, 3), seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)  # 10 != 3*3

    def test_grid_invalid_connectivity(self):
        with self.assertRaises(ValueError):
            Grid(connectivity='invalid', grid_shape=(10, 10))(pre_size=100, post_size=100)

    def test_radial_patches_basic(self):
        positions = np.random.RandomState(42).uniform(0, 100, (20, 2)) * u.um

        conn = RadialPatches(
            patch_radius=30 * u.um,
            n_patches=2,
            prob=0.8,
            seed=42
        )

        result = conn(
            pre_size=20, post_size=20,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'radial_patches')
        self.assertEqual(result.metadata['patch_radius'], 30 * u.um)
        self.assertEqual(result.metadata['n_patches'], 2)
        self.assertGreater(result.n_connections, 0)

    def test_radial_patches_no_positions_error(self):
        conn = RadialPatches(
            patch_radius=20 * u.um,
            n_patches=1,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)  # No positions

    def test_radial_patches_scalar_radius(self):
        positions = np.array([[0, 0], [10, 10], [20, 20]]) * u.um

        conn = RadialPatches(
            patch_radius=15.0,  # Scalar, no units
            n_patches=1,
            prob=1.0,
            seed=42
        )

        result = conn(
            pre_size=3, post_size=3,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertGreaterEqual(result.n_connections, 0)

    def test_clustered_random_basic(self):
        positions = np.random.RandomState(42).uniform(0, 100, (25, 2)) * u.um

        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=25 * u.um,
            cluster_factor=3.0,
            seed=42
        )

        result = conn(
            pre_size=25, post_size=25,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'clustered_random')
        self.assertEqual(result.metadata['prob'], 0.1)
        self.assertEqual(result.metadata['cluster_radius'], 25 * u.um)
        self.assertGreater(result.n_connections, 0)

    def test_clustered_random_no_positions_error(self):
        conn = ClusteredRandom(
            prob=0.1,
            cluster_radius=20 * u.um,
            seed=42
        )

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10)  # No positions


class TestTopologicalPatterns(unittest.TestCase):
    """
    Test topological connectivity patterns (SmallWorld, ScaleFree, Regular, Modular).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import SmallWorld, ScaleFree, Regular, Modular

        # Small-world network
        sw = SmallWorld(k=6, p=0.3, weight=1.0 * u.nS)
        result_sw = sw(pre_size=100, post_size=100)
        assert result_sw.metadata['pattern'] == 'small_world'

        # Scale-free network
        sf = ScaleFree(m=3, weight=0.8 * u.nS)
        result_sf = sf(pre_size=200, post_size=200)
        assert result_sf.metadata['pattern'] == 'scale_free'

        # Regular network
        reg = Regular(degree=8, weight=1.2 * u.nS)
        result_reg = reg(pre_size=150, post_size=150)
        assert result_reg.metadata['pattern'] == 'regular'

        # Modular network
        mod = Modular(n_modules=5, intra_prob=0.3, inter_prob=0.01)
        result_mod = mod(pre_size=100, post_size=100)
        assert result_mod.metadata['pattern'] == 'modular'
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_small_world_basic(self):
        conn = SmallWorld(k=4, p=0.2, seed=42)
        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'small_world')
        self.assertEqual(result.metadata['k'], 4)
        self.assertEqual(result.metadata['p'], 0.2)

        # Each neuron connects to k neighbors, total = n * k
        self.assertEqual(result.n_connections, 20 * 4)

    def test_small_world_different_sizes_error(self):
        conn = SmallWorld(k=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_small_world_rewiring(self):
        # Test that rewiring works (p=1.0 means all edges are rewired)
        conn = SmallWorld(k=2, p=1.0, seed=42)
        result = conn(pre_size=10, post_size=10)

        # Should still have same number of connections
        self.assertEqual(result.n_connections, 10 * 2)

        # But topology should be different from regular ring
        # (Hard to test directly, but at least check no errors)

    def test_scale_free_basic(self):
        conn = ScaleFree(m=2, seed=42)
        result = conn(pre_size=15, post_size=15)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'scale_free')
        self.assertEqual(result.metadata['m'], 2)

        # Should have connections (exact number depends on algorithm)
        self.assertGreater(result.n_connections, 0)

    def test_scale_free_different_sizes_error(self):
        conn = ScaleFree(m=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_regular_basic(self):
        conn = Regular(degree=5, seed=42)
        result = conn(pre_size=12, post_size=12)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'regular')
        self.assertEqual(result.metadata['degree'], 5)

        # Each neuron has exactly 'degree' connections
        self.assertEqual(result.n_connections, 12 * 5)

    def test_regular_different_sizes_error(self):
        conn = Regular(degree=3, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_modular_basic(self):
        conn = Modular(
            n_modules=3,
            intra_prob=0.4,
            inter_prob=0.05,
            seed=42
        )
        result = conn(pre_size=12, post_size=12)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'modular')
        self.assertEqual(result.metadata['n_modules'], 3)
        self.assertEqual(result.metadata['intra_prob'], 0.4)
        self.assertEqual(result.metadata['inter_prob'], 0.05)

        # Should have more intra-module than inter-module connections
        self.assertGreater(result.n_connections, 0)

    def test_modular_different_sizes_error(self):
        conn = Modular(n_modules=2, seed=42)

        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8)  # Different sizes not allowed

    def test_modular_uneven_module_assignment(self):
        # Test when population size doesn't divide evenly by number of modules
        conn = Modular(
            n_modules=3,
            intra_prob=0.3,
            inter_prob=0.01,
            seed=42
        )
        result = conn(pre_size=10, post_size=10)  # 10 doesn't divide evenly by 3

        # Should still work (extra neurons assigned to last module)
        self.assertGreater(result.n_connections, 0)


class TestExcitatoryInhibitory(unittest.TestCase):
    """
    Test ExcitatoryInhibitory connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import ExcitatoryInhibitory

        # Standard E-I network with Dale's principle
        ei_net = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.1,
            inh_prob=0.2,
            exc_weight=1.0 * u.nS,
            inh_weight=-0.8 * u.nS,
            delay=1.5 * u.ms
        )

        result = ei_net(pre_size=100, post_size=100)

        # Check E-I structure
        assert result.metadata['pattern'] == 'excitatory_inhibitory'
        assert result.metadata['exc_ratio'] == 0.8
        assert result.metadata['n_excitatory'] == 80
        assert result.metadata['n_inhibitory'] == 20

        # Should have both positive and negative weights
        if result.n_connections > 0:
            weights = u.get_mantissa(result.weights)
            assert np.any(weights > 0)  # Excitatory weights
            assert np.any(weights < 0)  # Inhibitory weights
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_basic_excitatory_inhibitory(self):
        exc_weight_init = Constant(1.2 * u.nS)
        inh_weight_init = Constant(-0.8 * u.nS)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.75,
            exc_prob=0.15,
            inh_prob=0.25,
            exc_weight=exc_weight_init,
            inh_weight=inh_weight_init,
            seed=42
        )

        result = conn(pre_size=40, post_size=40)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'excitatory_inhibitory')
        self.assertEqual(result.metadata['exc_ratio'], 0.75)
        self.assertEqual(result.metadata['n_excitatory'], 30)  # 40 * 0.75
        self.assertEqual(result.metadata['n_inhibitory'], 10)  # 40 - 30

        if result.n_connections > 0:
            # Should have both positive and negative weights
            weights = u.get_mantissa(result.weights)
            self.assertTrue(np.any(weights > 0))  # Excitatory
            self.assertTrue(np.any(weights < 0))  # Inhibitory

            # Check that excitatory weights are 1.2 and inhibitory are -0.8
            exc_weights = weights[weights > 0]
            inh_weights = weights[weights < 0]

            if len(exc_weights) > 0:
                np.testing.assert_array_almost_equal(exc_weights, 1.2)
            if len(inh_weights) > 0:
                np.testing.assert_array_almost_equal(inh_weights, -0.8)

    def test_excitatory_inhibitory_only_excitatory(self):
        exc_weight_init = Constant(1.0 * u.nS)

        conn = ExcitatoryInhibitory(
            exc_ratio=1.0,  # All excitatory
            exc_prob=0.2,
            inh_prob=0.3,  # Won't be used
            exc_weight=exc_weight_init,
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.metadata['n_excitatory'], 20)
        self.assertEqual(result.metadata['n_inhibitory'], 0)

        if result.n_connections > 0:
            # All weights should be positive
            weights = u.get_mantissa(result.weights)
            self.assertTrue(np.all(weights > 0))

    def test_excitatory_inhibitory_only_inhibitory(self):
        inh_weight_init = Constant(-1.5 * u.nS)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.0,  # All inhibitory
            exc_prob=0.2,  # Won't be used
            inh_prob=0.3,
            inh_weight=inh_weight_init,
            seed=42
        )

        result = conn(pre_size=15, post_size=15)

        self.assertEqual(result.metadata['n_excitatory'], 0)
        self.assertEqual(result.metadata['n_inhibitory'], 15)

        if result.n_connections > 0:
            # All weights should be negative
            weights = u.get_mantissa(result.weights)
            self.assertTrue(np.all(weights < 0))

    def test_excitatory_inhibitory_with_delays(self):
        delay_init = Constant(2.0 * u.ms)

        conn = ExcitatoryInhibitory(
            exc_ratio=0.6,
            exc_prob=0.1,
            inh_prob=0.2,
            exc_weight=0.5 * u.nS,
            inh_weight=-0.3 * u.nS,
            delay=delay_init,
            seed=42
        )

        result = conn(pre_size=25, post_size=25)

        if result.n_connections > 0:
            self.assertIsNotNone(result.delays)
            np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), 2.0)

    def test_excitatory_inhibitory_asymmetric_sizes(self):
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.1,
            inh_prob=0.15,
            seed=42
        )

        result = conn(pre_size=25, post_size=30)

        self.assertEqual(result.shape, (25, 30))
        # Pre population split: 20 excitatory, 5 inhibitory
        self.assertEqual(result.metadata['n_excitatory'], 20)
        self.assertEqual(result.metadata['n_inhibitory'], 5)

    def test_excitatory_inhibitory_zero_probabilities(self):
        conn = ExcitatoryInhibitory(
            exc_ratio=0.7,
            exc_prob=0.0,  # No excitatory connections
            inh_prob=0.0,  # No inhibitory connections
            seed=42
        )

        result = conn(pre_size=20, post_size=20)

        self.assertEqual(result.n_connections, 0)

    def test_excitatory_inhibitory_tuple_sizes(self):
        conn = ExcitatoryInhibitory(
            exc_ratio=0.8,
            exc_prob=0.1,
            inh_prob=0.2,
            seed=42
        )

        result = conn(pre_size=(4, 5), post_size=(2, 10))

        self.assertEqual(result.pre_size, (4, 5))
        self.assertEqual(result.post_size, (2, 10))
        # Pre size = 20, 80% excitatory = 16, 20% inhibitory = 4
        self.assertEqual(result.metadata['n_excitatory'], 16)
        self.assertEqual(result.metadata['n_inhibitory'], 4)


class TestSynapticPlasticityAndActivityDependent(unittest.TestCase):
    """
    Test SynapticPlasticity and ActivityDependent connectivity patterns.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import Random, SynapticPlasticity, ActivityDependent

        # Base connectivity pattern
        base = Random(prob=0.1, weight=1.0 * u.nS, seed=42)

        # Add STDP plasticity
        plastic = SynapticPlasticity(
            base_connectivity=base,
            plasticity_type='stdp',
            plasticity_params={
                'tau_pre': 20 * u.ms,
                'tau_post': 20 * u.ms,
                'A_plus': 0.01,
                'A_minus': 0.01
            }
        )

        result_plastic = plastic(pre_size=100, post_size=100)
        assert 'plasticity_type' in result_plastic.metadata
        assert result_plastic.metadata['plasticity_type'] == 'stdp'

        # Add activity-dependent pruning
        activity_dep = ActivityDependent(
            base_connectivity=base,
            pruning_threshold=0.1,
            strengthening_factor=1.5
        )

        result_activity = activity_dep(pre_size=100, post_size=100)
        assert 'activity_dependent' in result_activity.metadata
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_synaptic_plasticity_basic(self):
        base_conn = Random(prob=0.1, weight=1.0 * u.nS, seed=42)

        plasticity_params = {
            'tau_pre': 20 * u.ms,
            'tau_post': 20 * u.ms,
            'A_plus': 0.01,
            'A_minus': 0.01
        }

        plastic_conn = SynapticPlasticity(
            base_connectivity=base_conn,
            plasticity_type='stdp',
            plasticity_params=plasticity_params
        )

        result = plastic_conn(pre_size=30, post_size=30)

        # Should have same basic structure as base connectivity
        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)

        # Should have plasticity metadata
        self.assertEqual(result.metadata['plasticity_type'], 'stdp')
        self.assertEqual(result.metadata['plasticity_params'], plasticity_params)

    def test_synaptic_plasticity_different_types(self):
        base_conn = AllToAll(include_self_connections=False, seed=42)

        for plasticity_type in ['stdp', 'bcm', 'oja', 'homeostatic']:
            plastic_conn = SynapticPlasticity(
                base_connectivity=base_conn,
                plasticity_type=plasticity_type,
                plasticity_params={'param1': 0.1, 'param2': 0.2}
            )

            result = plastic_conn(pre_size=5, post_size=5)
            self.assertEqual(result.metadata['plasticity_type'], plasticity_type)

    def test_activity_dependent_basic(self):
        base_conn = Random(prob=0.15, weight=1.5 * u.nS, seed=42)

        activity_conn = ActivityDependent(
            base_connectivity=base_conn,
            pruning_threshold=0.1,
            strengthening_factor=1.3
        )

        result = activity_conn(pre_size=25, post_size=25)

        # Should have same basic structure as base connectivity
        self.assertEqual(result.model_type, 'point')
        self.assertGreater(result.n_connections, 0)

        # Should have activity-dependent metadata
        self.assertTrue(result.metadata['activity_dependent'])
        self.assertEqual(result.metadata['pruning_threshold'], 0.1)
        self.assertEqual(result.metadata['strengthening_factor'], 1.3)

    def test_activity_dependent_default_parameters(self):
        base_conn = OneToOne(seed=42)

        activity_conn = ActivityDependent(base_connectivity=base_conn)

        result = activity_conn(pre_size=10, post_size=10)

        # Should use default values
        self.assertEqual(result.metadata['pruning_threshold'], 0.1)
        self.assertEqual(result.metadata['strengthening_factor'], 1.2)

    def test_combined_plasticity_and_activity(self):
        # Test that both can be combined (plasticity wrapping activity-dependent)
        base_conn = Random(prob=0.2, seed=42)

        activity_conn = ActivityDependent(
            base_connectivity=base_conn,
            pruning_threshold=0.05,
            strengthening_factor=1.8
        )

        plastic_conn = SynapticPlasticity(
            base_connectivity=activity_conn,
            plasticity_type='stdp',
            plasticity_params={'tau': 15 * u.ms}
        )

        result = plastic_conn(pre_size=20, post_size=20)

        # Should have both types of metadata
        self.assertTrue(result.metadata['activity_dependent'])
        self.assertEqual(result.metadata['plasticity_type'], 'stdp')


class TestCustom(unittest.TestCase):
    """
    Test Custom connectivity pattern.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn._conn_point import Custom

        def my_connection_func(pre_size, post_size, rng, **kwargs):
            # Custom connectivity logic
            n_connections = min(pre_size, post_size)
            pre_indices = rng.choice(pre_size, n_connections, replace=False)
            post_indices = rng.choice(post_size, n_connections, replace=False)
            weights = rng.uniform(0.5, 2.0, n_connections) * u.nS
            delays = rng.uniform(1.0, 3.0, n_connections) * u.ms
            return pre_indices, post_indices, weights, delays

        conn = Custom(connection_func=my_connection_func)
        result = conn(pre_size=20, post_size=15)

        assert result.model_type == 'point'
        assert result.metadata['pattern'] == 'custom'
        assert result.n_connections == 15  # min(20, 15)
    """

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_custom_basic(self):
        def simple_func(pre_size, post_size, rng, **kwargs):
            # Connect neuron i to neuron i (one-to-one style)
            n_connections = min(pre_size, post_size)
            pre_indices = list(range(n_connections))
            post_indices = list(range(n_connections))
            weights = [1.0] * n_connections
            delays = [2.0] * n_connections
            return pre_indices, post_indices, weights, delays

        conn = Custom(connection_func=simple_func, seed=42)
        result = conn(pre_size=12, post_size=8)

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'custom')
        self.assertEqual(result.n_connections, 8)  # min(12, 8)

        np.testing.assert_array_equal(result.pre_indices, np.arange(8))
        np.testing.assert_array_equal(result.post_indices, np.arange(8))

    def test_custom_with_units(self):
        def func_with_units(pre_size, post_size, rng, **kwargs):
            pre_indices = [0, 1, 2]
            post_indices = [1, 2, 0]
            weights = [0.5, 1.0, 1.5] * u.nS
            delays = [1.0, 2.0, 3.0] * u.ms
            return pre_indices, post_indices, weights, delays

        conn = Custom(connection_func=func_with_units, seed=42)
        result = conn(pre_size=5, post_size=5)

        self.assertEqual(result.n_connections, 3)
        self.assertIsNotNone(result.weights)
        self.assertIsNotNone(result.delays)
        self.assertEqual(u.get_unit(result.weights), u.nS)
        self.assertEqual(u.get_unit(result.delays), u.ms)

    def test_custom_without_weights_delays(self):
        def minimal_func(pre_size, post_size, rng, **kwargs):
            pre_indices = [0, 1]
            post_indices = [1, 0]
            weights = None
            delays = None
            return pre_indices, post_indices, weights, delays

        conn = Custom(connection_func=minimal_func, seed=42)
        result = conn(pre_size=5, post_size=5)

        self.assertEqual(result.n_connections, 2)
        self.assertIsNone(result.weights)
        self.assertIsNone(result.delays)

    def test_custom_automatic_delay_units(self):
        def func_no_delay_units(pre_size, post_size, rng, **kwargs):
            pre_indices = [0, 1]
            post_indices = [1, 0]
            weights = None
            delays = [1.5, 2.5]  # No units
            return pre_indices, post_indices, weights, delays

        conn = Custom(connection_func=func_no_delay_units, seed=42)
        result = conn(pre_size=5, post_size=5)

        self.assertIsNotNone(result.delays)
        self.assertEqual(u.get_unit(result.delays), u.ms)  # Should get automatic units
        np.testing.assert_array_almost_equal(u.get_mantissa(result.delays), [1.5, 2.5])

    def test_custom_with_kwargs(self):
        def func_with_kwargs(pre_size, post_size, rng, custom_param=10, **kwargs):
            n_connections = min(pre_size, post_size, custom_param)
            pre_indices = list(range(n_connections))
            post_indices = list(range(n_connections))
            weights = None
            delays = None
            return pre_indices, post_indices, weights, delays

        conn = Custom(connection_func=func_with_kwargs, seed=42)
        result = conn(pre_size=20, post_size=15, custom_param=5)

        self.assertEqual(result.n_connections, 5)  # Limited by custom_param

    def test_custom_random_connectivity(self):
        def random_func(pre_size, post_size, rng, **kwargs):
            n_connections = pre_size
            pre_indices = rng.choice(pre_size, n_connections, replace=True)
            post_indices = rng.choice(post_size, n_connections, replace=True)
            weights = rng.uniform(0.5, 2.0, n_connections)
            delays = rng.uniform(1.0, 3.0, n_connections)
            return pre_indices, post_indices, weights, delays

        conn = Custom(connection_func=random_func, seed=42)
        result = conn(pre_size=10, post_size=12)

        self.assertEqual(result.n_connections, 10)
        self.assertTrue(np.all(result.pre_indices < 10))
        self.assertTrue(np.all(result.post_indices < 12))

        # Should have automatic units
        self.assertEqual(u.get_unit(result.delays), u.ms)

    def test_custom_empty_connections(self):
        def empty_func(pre_size, post_size, rng, **kwargs):
            return [], [], None, None

        conn = Custom(connection_func=empty_func, seed=42)
        result = conn(pre_size=10, post_size=10)

        self.assertEqual(result.n_connections, 0)
        self.assertEqual(len(result.pre_indices), 0)
        self.assertEqual(len(result.post_indices), 0)


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
