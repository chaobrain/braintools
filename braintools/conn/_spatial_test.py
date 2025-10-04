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

import unittest

import brainunit as u
import numpy as np

from braintools.conn import (
    Ring,
    Grid2d,
    RadialPatches,
    Gaussian,
    DistanceDependent,
    Exponential,
)
from braintools.init import Constant


class TestDistanceDependent(unittest.TestCase):

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
            seed=42
        )

        result = conn(
            pre_size=15, post_size=15,
            pre_positions=positions,
            post_positions=positions
        )

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'distance_dependent')
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


class TestGaussianAndExponentialAliases(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_gaussian_is_distance_dependent(self):
        self.assertTrue(issubclass(Gaussian, DistanceDependent))

    def test_exponential_is_distance_dependent(self):
        self.assertTrue(issubclass(Exponential, DistanceDependent))


class TestSpatialPatterns(unittest.TestCase):
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
        conn = Grid2d(
            connectivity='von_neumann',
            periodic=False,
            seed=42
        )
        result = conn(pre_size=(4, 4), post_size=(4, 4))

        self.assertEqual(result.model_type, 'point')
        self.assertEqual(result.metadata['pattern'], 'grid')
        self.assertEqual(result.metadata['grid_shape'], (4, 4))
        self.assertEqual(result.metadata['connectivity'], 'von_neumann')
        self.assertFalse(result.metadata['periodic'])

        # Interior neurons have 4 neighbors, edge neurons have fewer
        # Total connections depend on boundary conditions
        self.assertGreater(result.n_connections, 0)

    def test_grid_moore_periodic(self):
        conn = Grid2d(
            connectivity='moore',
            periodic=True,
            seed=42
        )
        result = conn(pre_size=(3, 3), post_size=(3, 3))

        self.assertEqual(result.metadata['connectivity'], 'moore')
        self.assertTrue(result.metadata['periodic'])

        # With periodic boundaries, each neuron has exactly 8 neighbors
        # Total = 9 * 8 = 72 connections
        self.assertEqual(result.n_connections, 72)

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
            patch_radius=15.0 * u.um,  # Scalar, no units
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
