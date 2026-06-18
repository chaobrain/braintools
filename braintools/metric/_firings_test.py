# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import math
import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import braintools

brainstate.environ.set(dt=0.1)


class TestRasterPlot(unittest.TestCase):
    def test_indices_and_times(self):
        spikes = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 0, 0],
        ])
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        idx, st = braintools.metric.raster_plot(spikes, times)
        # Spikes occur (sorted by time-step) at: (t0,n1),(t1,n0),(t2,n2),(t3,n1),(t3,n2)
        np.testing.assert_array_equal(idx, np.array([1, 0, 2, 1, 2]))
        np.testing.assert_allclose(np.asarray(st), np.array([0.0, 0.1, 0.2, 0.3, 0.3]))

    def test_preserves_quantity_units(self):
        spikes = np.zeros((5, 2))
        spikes[2, 1] = 1
        times = np.arange(5) * 0.1 * u.ms
        _, st = braintools.metric.raster_plot(spikes, times)
        self.assertIsInstance(st, u.Quantity)
        self.assertTrue(u.math.allclose(st, jnp.array([0.2]) * u.ms))


class TestFiringRate(unittest.TestCase):
    def test_shape(self):
        spikes = jnp.ones((1000, 10))
        rate = braintools.metric.firing_rate(spikes, 1.0)
        self.assertEqual(rate.shape, (1000,))

    def test_units_are_hz_quantity_vs_float_parity(self):
        # Quantity (ms) and float (seconds) inputs that describe the SAME physical
        # window/step must give the same Hz output.
        spikes = (np.random.RandomState(0).rand(500, 8) < 0.2).astype(float)
        r_q = braintools.metric.firing_rate(spikes, 5 * u.ms, 0.1 * u.ms)
        r_f = braintools.metric.firing_rate(spikes, 0.005, 0.0001)  # seconds
        np.testing.assert_allclose(np.asarray(r_q), np.asarray(r_f), rtol=1e-4)

    def test_constant_rate_value(self):
        # Every neuron spikes every step -> mean fraction 1.0 -> rate = 1/dt Hz.
        spikes = jnp.ones((1000, 4))
        dt = 0.001  # seconds
        rate = braintools.metric.firing_rate(spikes, 0.01, dt)
        # Away from the edges the boxcar average of a constant-1 signal is 1/dt = 1000 Hz.
        np.testing.assert_allclose(float(rate[500]), 1.0 / dt, rtol=1e-5)

    def test_wider_window_is_smoother(self):
        brainstate.random.seed(0)
        spikes = (brainstate.random.random((2000, 20)) < 0.1).astype(float)
        narrow = braintools.metric.firing_rate(spikes, 2 * u.ms, 0.1 * u.ms)
        wide = braintools.metric.firing_rate(spikes, 20 * u.ms, 0.1 * u.ms)
        # Wider smoothing -> smaller temporal variance.
        self.assertLess(float(jnp.std(wide)), float(jnp.std(narrow)))


class TestVictorPurpuraDistance(unittest.TestCase):
    def test_identical_trains(self):
        """Test distance between identical spike trains."""
        spikes = jnp.array([1.0, 2.0, 3.0, 4.0])
        distance = braintools.metric.victor_purpura_distance(spikes, spikes)
        self.assertAlmostEqual(float(distance), 0.0, places=5)

    def test_empty_trains(self):
        """Test distance with empty spike trains."""
        empty = jnp.array([])
        spikes = jnp.array([1.0, 2.0, 3.0])

        # Empty vs non-empty should equal number of spikes
        distance1 = braintools.metric.victor_purpura_distance(empty, spikes)
        self.assertEqual(float(distance1), 3.0)

        # Non-empty vs empty
        distance2 = braintools.metric.victor_purpura_distance(spikes, empty)
        self.assertEqual(float(distance2), 3.0)

        # Empty vs empty
        distance3 = braintools.metric.victor_purpura_distance(empty, empty)
        self.assertEqual(float(distance3), 0.0)

    def test_temporal_shift(self):
        """Test that temporal shifts are penalized correctly."""
        spikes1 = jnp.array([1.0, 2.0, 3.0])
        spikes2 = jnp.array([1.1, 2.1, 3.1])  # Shifted by 0.1

        # With high cost factor, should prefer matching with temporal penalty
        distance_high = braintools.metric.victor_purpura_distance(spikes1, spikes2, cost_factor=10.0)
        # With low cost factor, temporal shifts are cheaper
        distance_low = braintools.metric.victor_purpura_distance(spikes1, spikes2, cost_factor=1.0)

        self.assertGreater(distance_high, distance_low)

    def test_different_lengths(self):
        """Test distance between spike trains of different lengths."""
        spikes1 = jnp.array([1.0, 2.0])
        spikes2 = jnp.array([1.0, 2.0, 3.0, 4.0])

        distance = braintools.metric.victor_purpura_distance(spikes1, spikes2)
        # Should be at least the difference in number of spikes
        self.assertGreaterEqual(float(distance), 2.0)


class TestVanRossumDistance(unittest.TestCase):
    def test_identical_trains(self):
        """Test distance between identical spike trains."""
        spikes = jnp.array([1.0, 3.0, 5.0])
        distance = braintools.metric.van_rossum_distance(spikes, spikes)
        self.assertAlmostEqual(float(distance), 0.0, places=3)

    def test_empty_trains(self):
        """Test distance with empty spike trains."""
        empty = jnp.array([])
        spikes = jnp.array([1.0, 2.0, 3.0])

        distance1 = braintools.metric.van_rossum_distance(empty, spikes)
        distance2 = braintools.metric.van_rossum_distance(spikes, empty)
        distance3 = braintools.metric.van_rossum_distance(empty, empty)

        self.assertGreater(distance1, 0.0)
        self.assertGreater(distance2, 0.0)
        self.assertEqual(distance3, 0.0)

    def test_tau_effect(self):
        """Test that different tau values affect the distance."""
        spikes1 = jnp.array([1.0, 3.0])
        spikes2 = jnp.array([1.5, 3.5])  # Slightly shifted

        # Larger tau should make distance smaller (more temporal smoothing)
        distance_small_tau = braintools.metric.van_rossum_distance(spikes1, spikes2, tau=0.1)
        distance_large_tau = braintools.metric.van_rossum_distance(spikes1, spikes2, tau=1.0)

        self.assertGreater(distance_small_tau, distance_large_tau)

    def test_custom_t_max(self):
        """Test custom t_max parameter."""
        spikes1 = jnp.array([1.0, 2.0])
        spikes2 = jnp.array([1.1, 2.1])

        distance1 = braintools.metric.van_rossum_distance(spikes1, spikes2, t_max=10.0)
        distance2 = braintools.metric.van_rossum_distance(spikes1, spikes2, t_max=20.0)

        # Should be finite and positive
        self.assertGreater(distance1, 0.0)
        self.assertGreater(distance2, 0.0)


class TestSpikeTrainSynchrony(unittest.TestCase):
    def test_perfect_synchrony(self):
        """Test with perfectly synchronized spikes."""
        # All neurons spike at the same times
        spikes = jnp.zeros((100, 5))
        spikes = spikes.at[20, :].set(1)
        spikes = spikes.at[50, :].set(1)
        spikes = spikes.at[80, :].set(1)

        synchrony = braintools.metric.spike_train_synchrony(spikes, window_size=10.0, dt=1.0)
        # Should be close to 1 (perfect synchrony)
        self.assertGreater(float(synchrony), 0.8)

    def test_no_synchrony(self):
        """Test with asynchronous spikes."""
        spikes = jnp.zeros((100, 5))
        # Different neurons spike at different times
        spikes = spikes.at[10, 0].set(1)
        spikes = spikes.at[30, 1].set(1)
        spikes = spikes.at[50, 2].set(1)
        spikes = spikes.at[70, 3].set(1)
        spikes = spikes.at[90, 4].set(1)

        synchrony = braintools.metric.spike_train_synchrony(spikes, window_size=5.0, dt=1.0)
        # Should be low (no synchrony)
        self.assertLess(float(synchrony), 0.5)

    def test_single_neuron(self):
        """Test with single neuron."""
        spikes = jnp.zeros((100, 1))
        spikes = spikes.at[20, 0].set(1)

        synchrony = braintools.metric.spike_train_synchrony(spikes, window_size=10.0, dt=1.0)
        self.assertEqual(float(synchrony), 0.0)

    def test_no_spikes(self):
        """Test with no spikes."""
        spikes = jnp.zeros((100, 5))
        synchrony = braintools.metric.spike_train_synchrony(spikes, window_size=10.0, dt=1.0)
        self.assertEqual(float(synchrony), 0.0)

    def test_bounded_with_asymmetric_counts(self):
        """Regression for C9: many spikes in one train, one in the other.

        The previous ``min(N_i, N_j)`` normalization with one-directional counting
        could exceed 1; the symmetric ratio must stay in [0, 1].
        """
        spikes = jnp.zeros((100, 2))
        spikes = spikes.at[jnp.array([10, 20, 30, 40, 50]), 0].set(1)
        spikes = spikes.at[10, 1].set(1)  # single spike, coincident with one of train 0
        synchrony = braintools.metric.spike_train_synchrony(spikes, window_size=4.0, dt=1.0)
        self.assertGreaterEqual(float(synchrony), 0.0)
        self.assertLessEqual(float(synchrony), 1.0)
        # 1 coincidence in train0 + 1 in train1 over 5+1 spikes = 2/6.
        self.assertAlmostEqual(float(synchrony), 2.0 / 6.0, places=6)


class TestBurstSynchronyIndex(unittest.TestCase):
    def test_synchronized_bursts(self):
        """Test with synchronized bursts across neurons."""
        spikes = jnp.zeros((1000, 5))

        # Add synchronized bursts (5 spikes each)
        for burst_start in [100, 300, 600]:
            for neuron in range(5):
                for spike_offset in range(5):
                    spikes = spikes.at[burst_start + spike_offset, neuron].set(1)

        sync_idx = braintools.metric.burst_synchrony_index(spikes, burst_threshold=3, max_isi=10.0, dt=1.0)
        # Should detect high burst synchrony
        self.assertGreater(float(sync_idx), 0.5)

    def test_no_bursts(self):
        """Test with isolated spikes (no bursts)."""
        spikes = jnp.zeros((1000, 5))

        # Add isolated spikes
        for neuron in range(5):
            spikes = spikes.at[100 + neuron * 50, neuron].set(1)

        sync_idx = braintools.metric.burst_synchrony_index(spikes, burst_threshold=3, max_isi=10.0, dt=1.0)
        self.assertEqual(float(sync_idx), 0.0)

    def test_asynchronous_bursts(self):
        """Test with bursts that don't overlap."""
        spikes = jnp.zeros((1000, 5))

        # Add non-overlapping bursts
        for neuron in range(5):
            burst_start = 100 + neuron * 100
            for spike_offset in range(5):
                spikes = spikes.at[burst_start + spike_offset, neuron].set(1)

        sync_idx = braintools.metric.burst_synchrony_index(spikes, burst_threshold=3, max_isi=10.0, dt=1.0)
        # Should be low since bursts don't overlap
        self.assertLess(float(sync_idx), 0.3)


class TestPhaseLockingValue(unittest.TestCase):
    def test_perfect_phase_locking(self):
        """Test with perfect phase locking to reference frequency."""
        n_time = 1000
        dt = 0.001  # 1 ms
        freq = 10.0  # 10 Hz

        spikes = jnp.zeros((n_time, 3))

        # Add spikes at same phase of each cycle
        for cycle in range(int(freq * n_time * dt)):
            spike_time = int(cycle / freq / dt)
            if spike_time < n_time:
                spikes = spikes.at[spike_time, :].set(1)

        plv = braintools.metric.phase_locking_value(spikes, freq, dt)
        # Should be close to 1 for all neurons
        self.assertTrue(jnp.all(plv > 0.8))

    def test_no_phase_locking(self):
        """Test with random spikes (no phase locking)."""
        spikes = (brainstate.random.random((1000, 5)) < 0.05).astype(float)

        plv = braintools.metric.phase_locking_value(spikes, 10.0, 0.001)
        # Should be low for random spikes
        self.assertTrue(jnp.all(plv < 0.5))

    def test_no_spikes(self):
        """Test with no spikes."""
        spikes = jnp.zeros((1000, 5))

        plv = braintools.metric.phase_locking_value(spikes, 10.0, 0.001)
        # Should be zero for no spikes
        self.assertTrue(jnp.all(plv == 0.0))


class TestSpikeTimeTilingCoefficient(unittest.TestCase):
    def test_perfect_synchrony(self):
        """Test STTC with perfectly synchronized spikes."""
        spikes = jnp.zeros((1000, 3))

        # Add synchronized spikes
        sync_times = [100, 300, 500, 700]
        for t in sync_times:
            spikes = spikes.at[t, :].set(1)

        sttc = braintools.metric.spike_time_tiling_coefficient(spikes, dt=0.001, tau=0.01)

        # Off-diagonal elements should be close to 1
        self.assertTrue(jnp.all(jnp.diag(sttc) == 1.0))  # Diagonal is 1 by definition
        off_diag = sttc[jnp.triu_indices_from(sttc, k=1)]
        self.assertTrue(jnp.all(off_diag > 0.5))

    def test_no_correlation(self):
        """Test STTC with uncorrelated spike trains."""
        spikes = jnp.zeros((1000, 3))

        # Add spikes at different times for each neuron
        spikes = spikes.at[100, 0].set(1)
        spikes = spikes.at[300, 1].set(1)
        spikes = spikes.at[500, 2].set(1)

        sttc = braintools.metric.spike_time_tiling_coefficient(spikes, dt=0.001, tau=0.01)

        # Off-diagonal should be close to 0
        off_diag = sttc[jnp.triu_indices_from(sttc, k=1)]
        self.assertTrue(jnp.all(jnp.abs(off_diag) < 0.3))

    def test_matrix_properties(self):
        """Test that STTC matrix has correct properties."""
        spikes = (brainstate.random.random((500, 4)) < 0.1).astype(float)

        sttc = braintools.metric.spike_time_tiling_coefficient(spikes, dt=0.001, tau=0.005)

        # Should be square matrix
        self.assertEqual(sttc.shape, (4, 4))

        # Should be symmetric
        self.assertTrue(jnp.allclose(sttc, sttc.T))

        # Diagonal should be 1
        self.assertTrue(jnp.allclose(jnp.diag(sttc), 1.0))

        # Values should be bounded between -1 and 1
        self.assertTrue(jnp.all(sttc >= -1.0))
        self.assertTrue(jnp.all(sttc <= 1.0))

    def test_union_time_coverage_not_overcounted(self):
        """Regression for C16: T must use the union of windows.

        With densely packed spikes whose ±tau windows overlap heavily, summing
        per-spike window lengths overcounts (T would far exceed 1 before the old
        ``min(1, .)`` clamp). For two identical dense trains STTC must be exactly 1.
        """
        spikes = jnp.zeros((100, 2))
        dense = jnp.arange(10, 30)  # 20 consecutive spikes -> windows overlap a lot
        spikes = spikes.at[dense, 0].set(1)
        spikes = spikes.at[dense, 1].set(1)
        sttc = braintools.metric.spike_time_tiling_coefficient(spikes, dt=1.0, tau=5.0)
        self.assertAlmostEqual(float(sttc[0, 1]), 1.0, places=6)

    def test_anticorrelated_is_negative(self):
        """Disjoint, non-overlapping trains should give a negative STTC."""
        spikes = jnp.zeros((1000, 2))
        spikes = spikes.at[jnp.arange(0, 500, 20), 0].set(1)
        spikes = spikes.at[jnp.arange(500, 1000, 20), 1].set(1)
        sttc = braintools.metric.spike_time_tiling_coefficient(spikes, dt=1.0, tau=2.0)
        self.assertLess(float(sttc[0, 1]), 0.0)

    def test_empty_train_pair_is_zero(self):
        """A pair involving a silent neuron has STTC 0 (and diagonal stays 1)."""
        spikes = jnp.zeros((100, 2))
        spikes = spikes.at[jnp.array([10, 20, 30]), 0].set(1)  # neuron 1 stays silent
        sttc = braintools.metric.spike_time_tiling_coefficient(spikes, dt=1.0, tau=2.0)
        self.assertEqual(float(sttc[0, 1]), 0.0)
        self.assertEqual(float(sttc[1, 1]), 1.0)


class TestCorrelationIndex(unittest.TestCase):
    def test_perfect_correlation(self):
        """Test with perfectly correlated spike trains."""
        spikes = jnp.zeros((1000, 3))

        # Make all neurons have identical spike patterns
        base_pattern = (brainstate.random.random(1000) < 0.1).astype(float)
        for i in range(3):
            spikes = spikes.at[:, i].set(base_pattern)

        ci = braintools.metric.correlation_index(spikes, window_size=50.0, dt=1.0)
        # Should be close to 1
        self.assertGreater(float(ci), 0.8)

    def test_no_correlation(self):
        """Test with uncorrelated spike trains."""
        # Independent random spike trains
        spikes = (brainstate.random.random((1000, 5)) < 0.05).astype(float)

        ci = braintools.metric.correlation_index(spikes, window_size=50.0, dt=1.0)
        # Should be close to 0
        self.assertLess(jnp.abs(float(ci)), 0.3)

    def test_single_neuron(self):
        """Test with single neuron."""
        spikes = jnp.zeros((1000, 1))
        spikes = spikes.at[::100, 0].set(1)  # Regular spikes

        ci = braintools.metric.correlation_index(spikes, window_size=50.0, dt=1.0)
        self.assertEqual(float(ci), 0.0)

    def test_bounds(self):
        """Test that correlation index is properly bounded."""
        spikes = (brainstate.random.random((500, 8)) < 0.1).astype(float)

        ci = braintools.metric.correlation_index(spikes, window_size=25.0, dt=1.0)
        # Should be between -1 and 1
        self.assertGreaterEqual(float(ci), -1.0)
        self.assertLessEqual(float(ci), 1.0)

    def test_different_window_sizes(self):
        """Test effect of different window sizes."""
        spikes = (brainstate.random.random((1000, 5)) < 0.08).astype(float)

        ci_small = braintools.metric.correlation_index(spikes, window_size=10.0, dt=1.0)
        ci_large = braintools.metric.correlation_index(spikes, window_size=100.0, dt=1.0)

        # Both should be finite
        self.assertFalse(math.isnan(float(ci_small)))
        self.assertFalse(math.isnan(float(ci_large)))

    def test_too_few_bins_returns_zero(self):
        """A window covering (almost) the whole recording yields < 2 bins -> 0.0."""
        spikes = (brainstate.random.random((100, 3)) < 0.1).astype(float)
        ci = braintools.metric.correlation_index(spikes, window_size=100.0, dt=1.0)
        self.assertEqual(float(ci), 0.0)

    def test_constant_neuron_zero_variance(self):
        """A constant (zero-variance) binned train contributes a 0 correlation."""
        spikes = jnp.zeros((1000, 2))
        spikes = spikes.at[:, 0].set(1.0)  # neuron 0 spikes every step -> constant bins
        spikes = spikes.at[jnp.arange(0, 1000, 30), 1].set(1.0)
        ci = braintools.metric.correlation_index(spikes, window_size=50.0, dt=1.0)
        self.assertEqual(float(ci), 0.0)
