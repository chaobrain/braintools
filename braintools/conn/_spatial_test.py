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

import unittest

import brainunit as u
import numpy as np

from braintools.conn import (
    Ring,
    Grid,
    RadialPatches,
    ClusteredRandom,
)


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
