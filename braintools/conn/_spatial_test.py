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
from braintools.init import Constant, GaussianProfile, ExponentialProfile


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
        # Distances use ALL position dimensions (Euclidean over x, y, z), not just
        # the first two. Capture the distances the profile receives and check that a
        # known 3D pair matches sqrt(3^2 + 4^2 + 12^2) = 13.
        positions = np.array([[0, 0, 0], [3, 4, 12]]) * u.um

        captured = {}

        class CaptureProfile:
            def probability(self, distances):
                captured['distances'] = u.get_mantissa(distances)
                return np.full(distances.shape, 0.15)

        conn = DistanceDependent(
            distance_profile=CaptureProfile(),
            seed=42
        )

        result = conn(
            pre_size=2, post_size=2,
            pre_positions=positions,
            post_positions=positions
        )

        # All three dimensions contribute: 0-1 distance must be 13, not 5.
        self.assertAlmostEqual(captured['distances'][0, 1], 13.0)
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


class TestSpatialRegressions(unittest.TestCase):
    """Regression tests for autapse, duplicate-edge and validation fixes."""

    def test_gaussian_recurrent_no_autapses(self):
        # Same positions object -> recurrent net. Distance 0 gives prob ~1, which
        # previously produced self-connections. They must be excluded by default.
        positions = np.random.RandomState(0).uniform(0, 200, (40, 2)) * u.um
        conn = Gaussian(
            distance_profile=GaussianProfile(sigma=100 * u.um),
            seed=0,
        )
        result = conn(pre_size=40, post_size=40,
                      pre_positions=positions, post_positions=positions)
        self.assertEqual(np.sum(result.pre_indices == result.post_indices), 0)

    def test_exponential_recurrent_no_autapses(self):
        positions = np.random.RandomState(1).uniform(0, 200, (40, 2)) * u.um
        conn = Exponential(
            distance_profile=ExponentialProfile(decay_constant=100 * u.um),
            seed=1,
        )
        result = conn(pre_size=40, post_size=40,
                      pre_positions=positions, post_positions=positions)
        self.assertEqual(np.sum(result.pre_indices == result.post_indices), 0)

    def test_allow_self_connections_keeps_autapses(self):
        positions = np.random.RandomState(2).uniform(0, 200, (40, 2)) * u.um
        conn = Gaussian(
            distance_profile=GaussianProfile(sigma=100 * u.um),
            allow_self_connections=True,
            seed=2,
        )
        result = conn(pre_size=40, post_size=40,
                      pre_positions=positions, post_positions=positions)
        # Distance 0 -> prob 1, so every diagonal entry should be present.
        self.assertEqual(np.sum(result.pre_indices == result.post_indices), 40)

    def test_radial_patches_recurrent_no_autapses(self):
        positions = np.random.RandomState(3).uniform(0, 100, (30, 2)) * u.um
        conn = RadialPatches(
            patch_radius=40 * u.um,
            n_patches=3,
            prob=1.0,
            seed=3,
        )
        result = conn(pre_size=30, post_size=30,
                      pre_positions=positions, post_positions=positions)
        self.assertEqual(np.sum(result.pre_indices == result.post_indices), 0)

    def test_distance_dependent_position_count_mismatch(self):
        positions = np.random.RandomState(4).uniform(0, 50, (5, 2)) * u.um
        conn = DistanceDependent(
            distance_profile=GaussianProfile(sigma=20 * u.um),
            seed=4,
        )
        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=10,
                 pre_positions=positions, post_positions=positions)

    def test_gaussian_invalid_sigma_raises(self):
        with self.assertRaises(ValueError):
            Gaussian(distance_profile=GaussianProfile(sigma=0 * u.um))
        with self.assertRaises(ValueError):
            Gaussian(distance_profile=GaussianProfile(sigma=-5 * u.um))

    def test_exponential_invalid_decay_raises(self):
        with self.assertRaises(ValueError):
            Exponential(distance_profile=ExponentialProfile(decay_constant=0 * u.um))
        with self.assertRaises(ValueError):
            Exponential(distance_profile=ExponentialProfile(decay_constant=-5 * u.um))

    def test_radial_patches_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            RadialPatches(patch_radius=0 * u.um)
        with self.assertRaises(ValueError):
            RadialPatches(patch_radius=10 * u.um, prob=1.5)

    def test_ring_no_duplicate_edges_or_self_loops(self):
        # Large neighbors used to wrap and produce duplicate / self edges.
        conn = Ring(neighbors=4, bidirectional=True, seed=5)
        result = conn(pre_size=9, post_size=9)
        edges = list(zip(result.pre_indices.tolist(), result.post_indices.tolist()))
        self.assertEqual(len(edges), len(set(edges)))  # no duplicates
        self.assertEqual(np.sum(result.pre_indices == result.post_indices), 0)  # no self-loops

    def test_ring_invalid_neighbors_raises(self):
        conn = Ring(neighbors=5, seed=6)  # 5 > (9-1)//2 == 4
        with self.assertRaises(ValueError):
            conn(pre_size=9, post_size=9)

    def test_grid2d_periodic_small_no_duplicates(self):
        # 2x2 periodic Moore grid maps multiple offsets onto the same cells.
        conn = Grid2d(connectivity='moore', periodic=True, seed=7)
        result = conn(pre_size=(2, 2), post_size=(2, 2))
        edges = list(zip(result.pre_indices.tolist(), result.post_indices.tolist()))
        self.assertEqual(len(edges), len(set(edges)))  # no duplicates
        self.assertEqual(np.sum(result.pre_indices == result.post_indices), 0)  # no self-loops

    def test_grid2d_non_tuple_size_message(self):
        conn = Grid2d(seed=8)
        with self.assertRaises(ValueError) as ctx:
            conn(pre_size=16, post_size=16)
        self.assertIn('tuple', str(ctx.exception))


def _ring_reference(n, neighbors, bidirectional):
    """The pre-0.7 Python-loop Ring implementation, kept to pin the output."""
    edges = set()
    for i in range(n):
        for offset in range(1, neighbors + 1):
            target = (i + offset) % n
            if target != i:
                edges.add((i, target))
            if bidirectional:
                target = (i - offset) % n
                if target != i:
                    edges.add((i, target))
    edges = sorted(edges)
    return (np.array([e[0] for e in edges], dtype=np.int64),
            np.array([e[1] for e in edges], dtype=np.int64))


def _grid_reference(rows, cols, connectivity, periodic):
    """The pre-0.7 Python-loop Grid2d implementation, kept to pin the output."""
    if connectivity == 'von_neumann':
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    edges = set()
    for i in range(rows):
        for j in range(cols):
            source = i * cols + j
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if periodic:
                    ni, nj = ni % rows, nj % cols
                elif ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    continue
                target = ni * cols + nj
                if target != source:
                    edges.add((source, target))
    edges = sorted(edges)
    return (np.array([e[0] for e in edges], dtype=np.int64),
            np.array([e[1] for e in edges], dtype=np.int64))


class TestRingGridPinnedOutput(unittest.TestCase):
    """Ring and Grid2d are deterministic: edges and their order are pinned."""

    def test_ring_matches_reference(self):
        for n, k, bidir in [(3, 1, True), (5, 2, True), (8, 1, False), (9, 4, True),
                            (10, 3, False), (11, 5, True), (64, 7, True), (101, 13, False)]:
            pre, post = _ring_reference(n, k, bidir)
            result = Ring(neighbors=k, bidirectional=bidir, seed=0)(pre_size=n, post_size=n)
            np.testing.assert_array_equal(result.pre_indices, pre, err_msg=f'{(n, k, bidir)}')
            np.testing.assert_array_equal(result.post_indices, post, err_msg=f'{(n, k, bidir)}')
            self.assertEqual(result.pre_indices.dtype, np.int64)

    def test_ring_tuple_size_matches_reference(self):
        pre, post = _ring_reference(12, 2, True)
        result = Ring(neighbors=2, seed=0)(pre_size=(3, 4), post_size=(3, 4))
        np.testing.assert_array_equal(result.pre_indices, pre)
        np.testing.assert_array_equal(result.post_indices, post)

    def test_ring_literal_small(self):
        result = Ring(neighbors=1, bidirectional=True, seed=0)(pre_size=4, post_size=4)
        np.testing.assert_array_equal(result.pre_indices, [0, 0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(result.post_indices, [1, 3, 0, 2, 1, 3, 0, 2])

    def test_grid_matches_reference(self):
        configs = [(rows, cols, c, p)
                   for rows, cols in [(1, 1), (1, 5), (2, 2), (2, 3), (3, 3), (4, 7), (5, 1), (6, 6)]
                   for c in ('von_neumann', 'moore')
                   for p in (False, True)]
        for rows, cols, c, p in configs:
            pre, post = _grid_reference(rows, cols, c, p)
            result = Grid2d(connectivity=c, periodic=p, seed=0)(pre_size=(rows, cols), post_size=(rows, cols))
            np.testing.assert_array_equal(result.pre_indices, pre, err_msg=f'{(rows, cols, c, p)}')
            np.testing.assert_array_equal(result.post_indices, post, err_msg=f'{(rows, cols, c, p)}')
            self.assertEqual(result.pre_indices.dtype, np.int64)

    def test_grid_literal_small(self):
        result = Grid2d(connectivity='von_neumann', periodic=False, seed=0)(pre_size=(2, 2), post_size=(2, 2))
        np.testing.assert_array_equal(result.pre_indices, [0, 0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_array_equal(result.post_indices, [1, 2, 0, 3, 0, 3, 1, 2])


# ---------------------------------------------------------------------------
# 0.7: chunked / KD-tree DistanceDependent, vectorised RadialPatches
# ---------------------------------------------------------------------------

import tracemalloc
from unittest import mock

from scipy.spatial.distance import cdist

from braintools.conn import _spatial
from braintools.init import LinearProfile, StepProfile, PowerLawProfile


def _legacy_distance_dependent(profile, pre_pos, post_pos, seed, drop_self):
    """The pre-0.7 dense algorithm: one uniform per pair in row-major order."""
    rng = np.random.RandomState(seed)
    pre_val, unit = u.split_mantissa_unit(pre_pos)
    post_val = u.Quantity(post_pos).to(unit).mantissa
    distances = u.maybe_decimal(cdist(pre_val, post_val) * unit)
    probs = np.clip(profile.probability(distances), 0.0, 1.0)
    random_vals = rng.random((len(pre_val), len(post_val)))
    mask = np.asarray((probs > 0) & (random_vals < probs))
    if drop_self:
        np.fill_diagonal(mask, False)
    return np.nonzero(mask)


def _sheet(n, side, seed):
    return np.random.RandomState(seed).uniform(0.0, side, (n, 2)) * u.um


def _assert_sorted_unique(test, result):
    key = result.pre_indices * int(result.shape[1]) + result.post_indices
    test.assertTrue(np.all(np.diff(key) > 0), 'edges must be sorted by (pre, post) and distinct')


class TestDistanceDependentChunked(unittest.TestCase):

    def test_no_cutoff_reproduces_legacy_stream(self):
        positions = _sheet(150, 300.0, 0)
        profile = GaussianProfile(sigma=40 * u.um)
        pre, post = _legacy_distance_dependent(profile, positions, positions, seed=11, drop_self=True)
        result = Gaussian(profile, seed=11)(pre_size=150, post_size=150,
                                            pre_positions=positions, post_positions=positions)
        np.testing.assert_array_equal(result.pre_indices, pre)
        np.testing.assert_array_equal(result.post_indices, post)

    def test_no_cutoff_independent_of_chunk_size(self):
        # A chunk boundary that does not divide N (and N larger than one chunk).
        pre_pos, post_pos = _sheet(97, 200.0, 1), _sheet(61, 200.0, 2)
        profile = ExponentialProfile(decay_constant=30 * u.um)
        pre, post = _legacy_distance_dependent(profile, pre_pos, post_pos, seed=5, drop_self=False)
        with mock.patch.object(_spatial, 'MAX_CHUNK_PAIRS', 61 * 7 + 3):
            result = Exponential(profile, seed=5)(pre_size=97, post_size=61,
                                                  pre_positions=pre_pos, post_positions=post_pos)
        np.testing.assert_array_equal(result.pre_indices, pre)
        np.testing.assert_array_equal(result.post_indices, post)

    def test_precomputed_distances_reproduce_legacy_stream(self):
        pre_pos, post_pos = _sheet(80, 300.0, 3), _sheet(50, 300.0, 4)
        profile = GaussianProfile(sigma=50 * u.um, max_distance=100 * u.um)
        pre, post = _legacy_distance_dependent(profile, pre_pos, post_pos, seed=6, drop_self=False)
        distances = cdist(pre_pos.mantissa, post_pos.mantissa) * u.um
        with mock.patch.object(_spatial, 'MAX_CHUNK_PAIRS', 333):
            result = Gaussian(profile, seed=6)(pre_size=80, post_size=50, distances=distances)
        np.testing.assert_array_equal(result.pre_indices, pre)
        np.testing.assert_array_equal(result.post_indices, post)

    def test_cutoff_edges_within_max_distance_sorted_no_self(self):
        positions = _sheet(400, 400.0, 7)
        conn = Gaussian(GaussianProfile(sigma=20 * u.um, max_distance=50 * u.um), seed=7)
        result = conn(pre_size=400, post_size=400, pre_positions=positions, post_positions=positions)
        self.assertGreater(result.n_connections, 0)
        d = np.linalg.norm(positions.mantissa[result.pre_indices] - positions.mantissa[result.post_indices], axis=1)
        self.assertTrue(np.all(d <= 50.0))
        self.assertEqual(int(np.sum(result.pre_indices == result.post_indices)), 0)
        _assert_sorted_unique(self, result)

    def _check_profile_statistics(self, result, pre_pos, post_pos, profile, drop_self, n_bins=6):
        d = cdist(pre_pos.mantissa, post_pos.mantissa)
        p = np.clip(np.asarray(profile.probability(d * u.um), dtype=float), 0, 1)
        if drop_self:
            np.fill_diagonal(p, 0.0)
        connected = np.zeros(p.shape, dtype=bool)
        connected[result.pre_indices, result.post_indices] = True
        self.assertFalse(np.any(connected & (p == 0)))
        # Total count within 5 sd of its expectation.
        mean, var = p.sum(), (p * (1 - p)).sum()
        self.assertLess(abs(result.n_connections - mean), 5 * np.sqrt(var) + 1)
        # Distance profile: per-bin connection count matches the summed probability.
        inside = p > 0
        edges = np.linspace(0, d[inside].max() + 1e-9, n_bins + 1)
        which = np.digitize(d[inside], edges) - 1
        n_checked = 0
        for b in range(n_bins):
            sel = which == b
            if sel.sum() < 50:
                continue
            pb = p[inside][sel]
            obs = connected[inside][sel].sum()
            self.assertLess(abs(obs - pb.sum()), 5 * np.sqrt((pb * (1 - pb)).sum()) + 1,
                            f'distance bin {b}')
            n_checked += 1
        self.assertGreater(n_checked, 0)

    def test_cutoff_gaussian_matches_profile_statistics(self):
        positions = _sheet(600, 300.0, 8)
        profile = GaussianProfile(sigma=30 * u.um, max_distance=90 * u.um)
        result = Gaussian(profile, seed=8)(pre_size=600, post_size=600,
                                           pre_positions=positions, post_positions=positions)
        self._check_profile_statistics(result, positions, positions, profile, drop_self=True)

    def test_cutoff_exponential_cross_population_statistics(self):
        pre_pos, post_pos = _sheet(500, 300.0, 9), _sheet(450, 300.0, 10)
        profile = ExponentialProfile(decay_constant=25 * u.um, max_distance=80 * u.um)
        result = Exponential(profile, seed=9)(pre_size=500, post_size=450,
                                              pre_positions=pre_pos, post_positions=post_pos)
        self.assertEqual(result.shape, (500, 450))
        self._check_profile_statistics(result, pre_pos, post_pos, profile, drop_self=False)

    def test_cutoff_small_chunks_statistics(self):
        # Many chunks (boundaries not dividing N) give the same distribution.
        positions = _sheet(500, 300.0, 12)
        profile = LinearProfile(max_distance=60 * u.um)
        with mock.patch.object(_spatial, 'MAX_CHUNK_PAIRS', 997):
            result = DistanceDependent(profile, seed=12)(pre_size=500, post_size=500,
                                                         pre_positions=positions, post_positions=positions)
        _assert_sorted_unique(self, result)
        self._check_profile_statistics(result, positions, positions, profile, drop_self=True)

    def test_cutoff_allow_self_connections(self):
        positions = _sheet(200, 300.0, 13)
        conn = Gaussian(GaussianProfile(sigma=30 * u.um, max_distance=60 * u.um),
                        allow_self_connections=True, seed=13)
        result = conn(pre_size=200, post_size=200, pre_positions=positions, post_positions=positions)
        # Distance 0 -> probability 1: every autapse is present.
        self.assertEqual(int(np.sum(result.pre_indices == result.post_indices)), 200)

    def test_cutoff_seed_reproducible_and_seed_dependent(self):
        positions = _sheet(300, 300.0, 14)
        profile = GaussianProfile(sigma=30 * u.um, max_distance=90 * u.um)
        call = dict(pre_size=300, post_size=300, pre_positions=positions, post_positions=positions)
        a = Gaussian(profile, seed=1)(**call)
        b = Gaussian(profile, seed=1)(**call)
        c = Gaussian(profile, seed=2)(**call)
        np.testing.assert_array_equal(a.pre_indices, b.pre_indices)
        np.testing.assert_array_equal(a.post_indices, b.post_indices)
        self.assertFalse(a.n_connections == c.n_connections and np.array_equal(a.post_indices, c.post_indices))

    def test_cutoff_empty_result(self):
        positions = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]]) * u.um
        conn = Gaussian(GaussianProfile(sigma=10 * u.um, max_distance=5 * u.um), seed=0)
        result = conn(pre_size=3, post_size=3, pre_positions=positions, post_positions=positions)
        self.assertEqual(result.n_connections, 0)
        self.assertEqual(result.pre_indices.dtype, np.int64)

    def test_cutoff_unitless_positions(self):
        positions = np.random.RandomState(15).uniform(0, 100, (200, 2))
        conn = Gaussian(GaussianProfile(sigma=10.0, max_distance=30.0), seed=15)
        result = conn(pre_size=200, post_size=200, pre_positions=positions, post_positions=positions)
        d = np.linalg.norm(positions[result.pre_indices] - positions[result.post_indices], axis=1)
        self.assertGreater(result.n_connections, 0)
        self.assertTrue(np.all(d <= 30.0))

    def test_cutoff_in_other_unit_than_positions(self):
        positions = _sheet(200, 300.0, 16)
        conn = Gaussian(GaussianProfile(sigma=0.03 * u.mm, max_distance=0.06 * u.mm), seed=16)
        result = conn(pre_size=200, post_size=200, pre_positions=positions, post_positions=positions)
        d = np.linalg.norm(positions.mantissa[result.pre_indices] - positions.mantissa[result.post_indices], axis=1)
        self.assertTrue(np.all(d <= 60.0 + 1e-9))
        self.assertGreater(result.n_connections, 0)

    def test_weights_receive_edge_distances(self):
        positions = _sheet(150, 200.0, 17)
        captured = {}

        def weight(size, distances=None, **kwargs):
            captured['distances'] = distances
            return np.ones(size)

        for profile in (GaussianProfile(sigma=30 * u.um), GaussianProfile(sigma=30 * u.um, max_distance=60 * u.um)):
            result = Gaussian(profile, weight=weight, seed=17)(
                pre_size=150, post_size=150, pre_positions=positions, post_positions=positions)
            expected = np.linalg.norm(
                positions.mantissa[result.pre_indices] - positions.mantissa[result.post_indices], axis=1)
            self.assertEqual(u.get_unit(captured['distances']), u.um)
            np.testing.assert_allclose(u.get_mantissa(captured['distances']), expected, rtol=1e-10)
            self.assertEqual(u.math.size(result.weights), result.n_connections)

    def test_step_profile_cutoff_path(self):
        positions = _sheet(300, 300.0, 18)
        profile = StepProfile(threshold=40 * u.um, inside_prob=0.5, outside_prob=0.0)
        result = DistanceDependent(profile, seed=18)(pre_size=300, post_size=300,
                                                     pre_positions=positions, post_positions=positions)
        self._check_profile_statistics(result, positions, positions, profile, drop_self=True, n_bins=3)

    def test_profile_cutoff_detection(self):
        self.assertEqual(_spatial._profile_cutoff(GaussianProfile(sigma=1 * u.um, max_distance=3 * u.um)),
                         3 * u.um)
        self.assertIsNone(_spatial._profile_cutoff(GaussianProfile(sigma=1 * u.um)))
        self.assertEqual(_spatial._profile_cutoff(LinearProfile(max_distance=5 * u.um)), 5 * u.um)
        self.assertEqual(_spatial._profile_cutoff(PowerLawProfile(exponent=2.0, max_distance=7 * u.um)),
                         7 * u.um)
        self.assertEqual(_spatial._profile_cutoff(StepProfile(threshold=2 * u.um)), 2 * u.um)
        self.assertIsNone(_spatial._profile_cutoff(StepProfile(threshold=2 * u.um, outside_prob=0.1)))

        class Custom(GaussianProfile):
            def probability(self, distances):
                return np.full(np.shape(distances), 0.5)

        # Subclasses may override probability, so their max_distance is not trusted.
        self.assertIsNone(_spatial._profile_cutoff(Custom(sigma=1 * u.um, max_distance=3 * u.um)))
        self.assertIsNone(_spatial._profile_cutoff(object()))

    def test_chunks_by_counts(self):
        counts = np.array([3, 0, 5, 9, 1, 1, 0, 2])
        chunks = list(_spatial._chunks_by_counts(counts, 6))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], len(counts))
        for (_, b), (c, _) in zip(chunks[:-1], chunks[1:]):
            self.assertEqual(b, c)
        for a, b in chunks:
            self.assertTrue(b - a == 1 or counts[a:b].sum() <= 6)
        self.assertIn((3, 4), chunks)  # a row heavier than the bound forms its own chunk
        self.assertEqual(list(_spatial._chunks_by_counts(np.zeros(0, dtype=int), 6)), [])

    def test_pairs_within_matches_cdist(self):
        rs = np.random.RandomState(19)
        pre, post = rs.uniform(0, 50, (40, 3)), rs.uniform(0, 50, (33, 3))
        got = list(_spatial._pairs_within(pre, post, 12.0, max_pairs=20))
        self.assertGreater(len(got), 1)
        i = np.concatenate([g[0] for g in got])
        j = np.concatenate([g[1] for g in got])
        v = np.concatenate([g[2] for g in got])
        d = cdist(pre, post)
        ei, ej = np.nonzero(d <= 12.0)
        self.assertEqual(set(zip(i.tolist(), j.tolist())), set(zip(ei.tolist(), ej.tolist())))
        self.assertEqual(len(i), len(ei))
        np.testing.assert_allclose(v, d[i, j])
        self.assertEqual(list(_spatial._pairs_within(pre[:0], post, 12.0)), [])

    def test_sorted_unique(self):
        np.testing.assert_array_equal(_spatial._sorted_unique(np.array([5, 1, 5, 3, 1], dtype=np.int64)),
                                      [1, 3, 5])
        self.assertEqual(_spatial._sorted_unique(np.zeros(0, dtype=np.int64)).size, 0)

    def test_memory_bound_2d_sheet_2e4(self):
        # A dense float64 N x N distance matrix alone would be 3.2 GB.
        n = 20000
        positions = _sheet(n, float(np.sqrt(n)) * 10.0, 20)
        conn = Gaussian(GaussianProfile(sigma=15 * u.um, max_distance=45 * u.um), seed=20)
        tracemalloc.start()
        try:
            result = conn(pre_size=n, post_size=n, pre_positions=positions, post_positions=positions)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertGreater(result.n_connections, 5 * n)
        self.assertLess(peak, 200 * 2 ** 20)


class TestRadialPatchesVectorised(unittest.TestCase):

    @staticmethod
    def _balls(post_pos, radius):
        d = cdist(post_pos, post_pos)
        return [frozenset(np.nonzero(row <= radius)[0].tolist()) for row in d]

    def test_sorted_unique_and_in_range(self):
        positions = _sheet(120, 200.0, 0)
        result = RadialPatches(patch_radius=30 * u.um, n_patches=3, prob=0.6, seed=0)(
            pre_size=120, post_size=120, pre_positions=positions, post_positions=positions)
        self.assertGreater(result.n_connections, 0)
        _assert_sorted_unique(self, result)
        self.assertEqual(int(np.sum(result.pre_indices == result.post_indices)), 0)
        self.assertEqual(result.pre_indices.dtype, np.int64)

    def test_single_patch_targets_form_a_ball(self):
        pre_pos, post_pos = _sheet(60, 200.0, 1), _sheet(150, 200.0, 2)
        result = RadialPatches(patch_radius=35 * u.um, n_patches=1, prob=1.0, seed=1)(
            pre_size=60, post_size=150, pre_positions=pre_pos, post_positions=post_pos)
        balls = set(self._balls(post_pos.mantissa, 35.0))
        for i in range(60):
            targets = frozenset(result.post_indices[result.pre_indices == i].tolist())
            self.assertIn(targets, balls)

    def test_mean_out_degree_matches_expectation(self):
        n_pre, n_post, p = 1500, 200, 0.4
        pre_pos, post_pos = _sheet(n_pre, 200.0, 3), _sheet(n_post, 200.0, 4)
        result = RadialPatches(patch_radius=30 * u.um, n_patches=1, prob=p, seed=3)(
            pre_size=n_pre, post_size=n_post, pre_positions=pre_pos, post_positions=post_pos)
        deg = np.array([len(b) for b in self._balls(post_pos.mantissa, 30.0)], dtype=float)
        mean = n_pre * p * deg.mean()
        var = n_pre * (p * (1 - p) * deg.mean() + p ** 2 * deg.var())
        self.assertLess(abs(result.n_connections - mean), 5 * np.sqrt(var))

    def test_overlapping_patches_union_probability(self):
        # Three post neurons all within one radius of each other and two patches:
        # every pair lies in both patches, so P(edge) = 1 - (1 - p)^2.
        n_pre, p = 2000, 0.3
        post_pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) * u.um
        pre_pos = np.zeros((n_pre, 2)) * u.um
        result = RadialPatches(patch_radius=5 * u.um, n_patches=2, prob=p, seed=5)(
            pre_size=n_pre, post_size=3, pre_positions=pre_pos, post_positions=post_pos)
        q = 1 - (1 - p) ** 2
        mean, sd = 3 * n_pre * q, np.sqrt(3 * n_pre * q * (1 - q))
        self.assertLess(abs(result.n_connections - mean), 5 * sd)

    def test_all_centres_full_connectivity(self):
        positions = _sheet(25, 100.0, 6)
        result = RadialPatches(patch_radius=1e-3 * u.um, n_patches=100, prob=1.0, seed=6)(
            pre_size=25, post_size=25, pre_positions=positions, post_positions=positions)
        # n_patches is capped at post_num: every post neuron is the centre of its own ball.
        self.assertEqual(result.n_connections, 25 * 24)

    def test_zero_patches_or_zero_prob_is_empty(self):
        positions = _sheet(30, 100.0, 7)
        call = dict(pre_size=30, post_size=30, pre_positions=positions, post_positions=positions)
        self.assertEqual(RadialPatches(patch_radius=50 * u.um, n_patches=0, seed=7)(**call).n_connections, 0)
        self.assertEqual(
            RadialPatches(patch_radius=50 * u.um, n_patches=2, prob=0.0, seed=7)(**call).n_connections, 0)

    def test_chunking_does_not_change_edges(self):
        # Centres are drawn up front and the per-triple draws are consumed in
        # triple order, so the chunk size does not affect the result.
        positions = _sheet(300, 300.0, 8)
        call = dict(pre_size=300, post_size=300, pre_positions=positions, post_positions=positions)

        def make():
            return RadialPatches(patch_radius=40 * u.um, n_patches=3, prob=0.5, seed=8)

        ref = make()(**call)
        with mock.patch.object(_spatial, 'MAX_CHUNK_PAIRS', 101):
            small = make()(**call)
        self.assertGreater(ref.n_connections, 0)
        np.testing.assert_array_equal(ref.pre_indices, small.pre_indices)
        np.testing.assert_array_equal(ref.post_indices, small.post_indices)

    def test_allow_self_connections_and_weights(self):
        positions = _sheet(80, 100.0, 9)
        result = RadialPatches(patch_radius=200 * u.um, n_patches=1, prob=1.0, allow_self_connections=True,
                               weight=Constant(0.5 * u.nS), delay=Constant(1.0 * u.ms), seed=9)(
            pre_size=80, post_size=80, pre_positions=positions, post_positions=positions)
        # The radius covers the whole sheet, so every pair (autapses included) is connected.
        self.assertEqual(result.n_connections, 80 * 80)
        self.assertEqual(u.get_unit(result.weights), u.nS)
        self.assertEqual(u.get_unit(result.delays), u.ms)

    def test_unitless_positions(self):
        positions = np.random.RandomState(10).uniform(0, 100, (50, 2))
        result = RadialPatches(patch_radius=20.0, n_patches=2, prob=1.0, seed=10)(
            pre_size=50, post_size=50, pre_positions=positions, post_positions=positions)
        self.assertGreater(result.n_connections, 0)
        _assert_sorted_unique(self, result)

    def test_memory_bound_2e4(self):
        n = 20000
        positions = _sheet(n, float(np.sqrt(n)) * 10.0, 11)
        conn = RadialPatches(patch_radius=25 * u.um, n_patches=2, prob=0.5, seed=11)
        tracemalloc.start()
        try:
            result = conn(pre_size=n, post_size=n, pre_positions=positions, post_positions=positions)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertGreater(result.n_connections, n)
        self.assertLess(peak, 200 * 2 ** 20)


class TestSpatialValidation07(unittest.TestCase):

    def test_wrong_profile_type_raises(self):
        with self.assertRaises(TypeError):
            Gaussian(ExponentialProfile(decay_constant=10 * u.um))
        with self.assertRaises(TypeError):
            Exponential(GaussianProfile(sigma=10 * u.um))

    def test_non_positive_scale_on_custom_profile_raises(self):
        class Bad:
            sigma = -1.0 * u.um

            def probability(self, distances):
                return np.zeros(np.shape(distances))

        with self.assertRaises(ValueError):
            DistanceDependent(Bad())

    def test_post_position_count_mismatch(self):
        pre_pos, post_pos = _sheet(10, 50.0, 0), _sheet(7, 50.0, 1)
        conn = DistanceDependent(GaussianProfile(sigma=20 * u.um, max_distance=40 * u.um), seed=0)
        with self.assertRaises(ValueError):
            conn(pre_size=10, post_size=8, pre_positions=pre_pos, post_positions=post_pos)

    def test_one_dimensional_position_vectors(self):
        positions = np.linspace(0.0, 100.0, 60) * u.um  # shape (N,), not (N, 1)
        conn = Gaussian(GaussianProfile(sigma=5 * u.um, max_distance=10 * u.um), seed=0)
        result = conn(pre_size=60, post_size=60, pre_positions=positions, post_positions=positions)
        gaps = np.abs(positions.mantissa[result.pre_indices] - positions.mantissa[result.post_indices])
        self.assertGreater(result.n_connections, 0)
        self.assertTrue(np.all(gaps <= 10.0))

    def test_far_populations_give_empty_chunks(self):
        # Rows without candidates (and chunks made only of such rows) are skipped.
        pre_pos = np.concatenate([np.full((30, 2), 1000.0), np.zeros((30, 2))]) * u.um
        post_pos = np.zeros((20, 2)) * u.um
        with mock.patch.object(_spatial, 'MAX_CHUNK_PAIRS', 15):
            result = Gaussian(GaussianProfile(sigma=5 * u.um, max_distance=10 * u.um), seed=0)(
                pre_size=60, post_size=20, pre_positions=pre_pos, post_positions=post_pos)
        # Distance 0 -> probability 1 for the 30 pre neurons at the origin.
        self.assertEqual(result.n_connections, 30 * 20)
        self.assertTrue(np.all(result.pre_indices >= 30))

    def test_precomputed_square_distances_keep_diagonal(self):
        # Without positions, pre and post are not known to be the same population.
        positions = _sheet(40, 100.0, 2)
        distances = cdist(positions.mantissa, positions.mantissa) * u.um
        result = Gaussian(GaussianProfile(sigma=20 * u.um), seed=2)(pre_size=40, post_size=40, distances=distances)
        self.assertEqual(int(np.sum(result.pre_indices == result.post_indices)), 40)

    def test_grid_validation(self):
        with self.assertRaises(ValueError):
            Grid2d(seed=0)(pre_size=(3, 3), post_size=(3, 4))
        with self.assertRaises(ValueError):
            Grid2d(seed=0)(pre_size=(2, 2, 2), post_size=(2, 2, 2))
        with self.assertRaises(ValueError):
            Grid2d(connectivity='hexagonal', seed=0)(pre_size=(3, 3), post_size=(3, 3))

    def test_grid_empty(self):
        result = Grid2d(seed=0)(pre_size=(0, 4), post_size=(0, 4))
        self.assertEqual(result.n_connections, 0)
        self.assertEqual(result.pre_indices.dtype, np.int64)
