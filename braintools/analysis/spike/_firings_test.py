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
from typing import Optional

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy as onp
import pytest

import braintools
from braintools.analysis.spike._firings import (
    _floor_ratio,
    _in_unit,
    _steps,
    burst_synchrony_index,
    correlation_index,
    spike_time_tiling_coefficient,
    spike_train_synchrony,
    van_rossum_distance,
    victor_purpura_distance,
)
from braintools.spike import SpikeTrains

brainstate.environ.set(dt=0.1)


# ----------------------------------------------------------------------------
# Reference copies of the implementations before version 0.5 (Python loops,
# Python ``float`` results). The rewritten functions must reproduce them.
# ----------------------------------------------------------------------------

def _legacy_victor_purpura_distance(
    spike_times_1: brainstate.typing.ArrayLike,
    spike_times_2: brainstate.typing.ArrayLike,
    cost_factor: float = 1.0
):
    spikes1 = onp.asarray(u.get_magnitude(spike_times_1), dtype=float)
    spikes2 = onp.asarray(u.get_magnitude(spike_times_2), dtype=float)
    cost_factor = float(u.get_magnitude(cost_factor))

    n1, n2 = spikes1.shape[0], spikes2.shape[0]

    # Handle empty spike trains
    if n1 == 0:
        return float(n2)
    if n2 == 0:
        return float(n1)

    # Dynamic programming on host (NumPy) arrays with plain Python arithmetic.
    # dp[i, j] = minimum cost to transform spikes1[:i] into spikes2[:j].
    # Using JAX arrays here would force a host<->device round-trip for every
    # ``.at[].set`` and ``dp[...]`` read, making the routine O(n^2) *device*
    # operations; NumPy keeps it a fast in-memory fill.
    dp = onp.empty((n1 + 1, n2 + 1), dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n1 + 1):
        dp[i, 0] = i  # Delete all spikes in train 1
    for j in range(1, n2 + 1):
        dp[0, j] = j  # Insert all spikes in train 2

    for i in range(1, n1 + 1):
        s1 = spikes1[i - 1]
        for j in range(1, n2 + 1):
            # Match spike i with spike j (temporal shift), delete, or insert.
            match_cost = dp[i - 1, j - 1] + cost_factor * abs(s1 - spikes2[j - 1])
            delete_cost = dp[i - 1, j] + 1.0
            insert_cost = dp[i, j - 1] + 1.0
            dp[i, j] = min(match_cost, delete_cost, insert_cost)

    return float(dp[n1, n2])


def _legacy_van_rossum_distance(
    spike_times_1: brainstate.typing.ArrayLike,
    spike_times_2: brainstate.typing.ArrayLike,
    tau: float = 1.0,
    t_max: Optional[float] = None
):
    spikes1 = jnp.asarray(spike_times_1)
    spikes2 = jnp.asarray(spike_times_2)

    # Determine time window (unchanged: default is max spike time + 5*tau)
    if t_max is None:
        all_spikes = jnp.concatenate([spikes1, spikes2])
        if all_spikes.size == 0:
            return 0.0
        t_max = float(jnp.max(all_spikes)) + 5.0 * tau

    def pair_integral(times_a, times_b):
        r"""Exact ``\int_0^{T} f_a(t) f_b(t)\, dt`` for the exponential kernel.

        With :math:`f_a(t) = \frac{1}{\tau}\sum_k e^{-(t-a_k)/\tau} H(t-a_k)`,
        the product of two such responses integrates in closed form to one term
        per spike pair:

        .. math::

            \frac{1}{2\tau}\left[
                e^{(a+b-2m)/\tau} - e^{(a+b-2T)/\tau}
            \right], \qquad m = \max(0, a, b),

        for :math:`m < T` (a pair contributes nothing once :math:`m \ge T`, since
        the integrand vanishes on :math:`[0, T]`).

        This replaces the former left-endpoint rectangle rule on a grid of step
        ``tau / 20``, which biased every result high by ~2.5% (an O(h) error:
        ``h / (1 - e^{-2h/tau})`` vs the exact ``tau/2``, i.e. 1.0508 in the
        squared distance for ``h = tau/20``).
        """
        if times_a.size == 0 or times_b.size == 0:
            return 0.0
        a = times_a[:, None]
        b = times_b[None, :]
        m = jnp.maximum(jnp.maximum(0.0, a), b)
        terms = (
            jnp.exp((a + b - 2.0 * m) / tau)
            - jnp.exp((a + b - 2.0 * t_max) / tau)
        )
        return jnp.sum(jnp.where(m < t_max, terms, 0.0))

    distance_squared = (1.0 / (2.0 * tau)) * (
        pair_integral(spikes1, spikes1)
        + pair_integral(spikes2, spikes2)
        - 2.0 * pair_integral(spikes1, spikes2)
    )
    # Round-off can push an exactly-zero distance marginally negative.
    return float(jnp.sqrt(jnp.maximum(distance_squared, 0.0)))


def _legacy_spike_train_synchrony(
    spike_matrix: brainstate.typing.ArrayLike,
    window_size: float,
    dt: Optional[float] = None
):
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = onp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape

    if n_neurons < 2:
        return 0.0

    # Half-coincidence window in samples (Quantities converted, floats in ms).
    half_steps = _steps(window_size, dt, u.ms) / 2.0

    # Spike sample indices for each neuron.
    spike_idx_list = [onp.where(spikes[:, i] > 0)[0] for i in range(n_neurons)]

    total_synchrony = 0.0
    n_pairs = 0
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            idx_i = spike_idx_list[i]
            idx_j = spike_idx_list[j]
            n_i, n_j = len(idx_i), len(idx_j)
            if n_i == 0 or n_j == 0:
                continue

            # A spike is "coincident" if the *other* train has any spike within the
            # half-window. Counting coincidences symmetrically in both directions and
            # normalizing by the total spike count bounds the ratio to [0, 1] (the
            # earlier ``min(N_i, N_j)`` normalization with one-directional counting
            # could exceed 1).
            within = onp.abs(idx_i[:, None] - idx_j[None, :]) <= half_steps
            coin_i = int(onp.any(within, axis=1).sum())
            coin_j = int(onp.any(within, axis=0).sum())
            total_synchrony += (coin_i + coin_j) / (n_i + n_j)
            n_pairs += 1

    return float(total_synchrony / n_pairs) if n_pairs > 0 else 0.0


def _legacy_burst_synchrony_index(
    spike_matrix: brainstate.typing.ArrayLike,
    burst_threshold: int = 3,
    max_isi: float = 100.0,
    dt: Optional[float] = None
):
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = onp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape
    dt_m = _in_unit(dt, u.ms)
    max_isi_m = _in_unit(max_isi, u.ms)

    def detect_bursts(spike_train):
        """Detect burst events in a single spike train."""
        spike_times = onp.where(spike_train > 0)[0] * dt_m
        if len(spike_times) < burst_threshold:
            return []

        bursts = []
        current_burst = [spike_times[0]]

        for i in range(1, len(spike_times)):
            isi = spike_times[i] - spike_times[i - 1]
            if isi <= max_isi_m:
                current_burst.append(spike_times[i])
            else:
                if len(current_burst) >= burst_threshold:
                    bursts.append((current_burst[0], current_burst[-1]))
                current_burst = [spike_times[i]]

        # Check final burst
        if len(current_burst) >= burst_threshold:
            bursts.append((current_burst[0], current_burst[-1]))

        return bursts

    # Detect bursts for all neurons
    all_bursts = []
    for i in range(n_neurons):
        bursts = detect_bursts(spikes[:, i])
        for start, end in bursts:
            all_bursts.append((i, start, end))

    if len(all_bursts) == 0:
        return 0.0

    # Count synchronized bursts
    synchronous_bursts = 0
    total_bursts = len(all_bursts)

    for i, (neuron1, start1, end1) in enumerate(all_bursts):
        overlapping_neurons = {neuron1}

        for j, (neuron2, start2, end2) in enumerate(all_bursts):
            if i != j and neuron1 != neuron2:
                # Check for temporal overlap. ``>= 0`` (inclusive) counts bursts
                # that touch at a single instant -- the strongest possible
                # synchrony -- which a strict ``> 0`` would wrongly exclude.
                overlap = min(end1, end2) - max(start1, start2)
                if overlap >= 0:
                    overlapping_neurons.add(neuron2)

        # If burst involves multiple neurons, it's synchronous
        if len(overlapping_neurons) > 1:
            synchronous_bursts += 1

    return float(synchronous_bursts / total_bursts) if total_bursts > 0 else 0.0


def _legacy_spike_time_tiling_coefficient(
    spike_matrix: brainstate.typing.ArrayLike,
    dt: Optional[float] = None,
    tau: float = 5.0
):
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = onp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape

    # Coincidence half-window in samples (Quantities converted, floats in ms).
    tau_steps = int(round(_steps(tau, dt, u.ms)))

    def covered_fraction(idx):
        # Fraction of the recording covered by the *union* of [s - tau, s + tau]
        # windows around each spike. Summing per-spike window lengths (the previous
        # behaviour) double-counts overlapping windows and inflates T; the union is
        # the quantity defined by Cutts & Eglen (2014).
        covered = onp.zeros(n_time, dtype=bool)
        for s in idx:
            lo = max(0, s - tau_steps)
            hi = min(n_time, s + tau_steps + 1)
            covered[lo:hi] = True
        return covered.sum() / n_time

    sttc_matrix = onp.eye(n_neurons)  # Diagonal is 1 by definition
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            idx_i = onp.where(spikes[:, i] > 0)[0]
            idx_j = onp.where(spikes[:, j] > 0)[0]
            n_i, n_j = len(idx_i), len(idx_j)

            if n_i == 0 or n_j == 0:
                sttc_matrix[i, j] = sttc_matrix[j, i] = 0.0
                continue

            within = onp.abs(idx_i[:, None] - idx_j[None, :]) <= tau_steps
            P_A = onp.any(within, axis=1).sum() / n_i  # spikes in i near a j spike
            P_B = onp.any(within, axis=0).sum() / n_j  # spikes in j near an i spike
            T_A = covered_fraction(idx_i)
            T_B = covered_fraction(idx_j)

            # Guard each term independently: a single degenerate denominator should
            # zero only its own term, not the whole coefficient.
            term1 = (P_A - T_B) / (1.0 - P_A * T_B) if (1.0 - P_A * T_B) != 0 else 0.0
            term2 = (P_B - T_A) / (1.0 - P_B * T_A) if (1.0 - P_B * T_A) != 0 else 0.0
            sttc_value = 0.5 * (term1 + term2)

            sttc_matrix[i, j] = sttc_matrix[j, i] = sttc_value

    return jnp.asarray(sttc_matrix)


def _legacy_correlation_index(
    spike_matrix: brainstate.typing.ArrayLike,
    window_size: float,
    dt: Optional[float] = None
):
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape

    if n_neurons < 2:
        return 0.0

    # Bin size in samples (``window_size`` and ``dt`` must share a time unit).
    bin_size = max(1, _floor_ratio(_steps(window_size, dt, u.ms)))
    # Only whole bins are used; any trailing samples that do not fill a complete
    # bin are discarded so every bin has the same width.
    n_bins = n_time // bin_size

    if n_bins < 2:
        return 0.0

    # Bin spike counts
    binned_spikes = jnp.zeros((n_bins, n_neurons))
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        binned_spikes = binned_spikes.at[i, :].set(jnp.sum(spikes[start_idx:end_idx, :], axis=0))

    # Calculate pairwise correlations
    correlations = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            # Compute Pearson correlation
            x, y = binned_spikes[:, i], binned_spikes[:, j]

            # Handle case where one or both series have zero variance
            x_var = jnp.var(x)
            y_var = jnp.var(y)

            if x_var == 0 or y_var == 0:
                corr = 0.0
            else:
                corr = jnp.corrcoef(x, y)[0, 1]
                # Handle NaN case
                corr = jnp.where(jnp.isnan(corr), 0.0, corr)

            correlations.append(corr)

    # Clamp to the documented [-1, 1] range (guards against float round-off in
    # ``corrcoef`` pushing the mean marginally outside the interval).
    return float(jnp.clip(jnp.mean(jnp.array(correlations)), -1.0, 1.0))



class TestFiringRate(unittest.TestCase):
    def test_shape(self):
        spikes = jnp.ones((1000, 10))
        rate = braintools.analysis.spike.firing_rate(spikes, 1.0)
        self.assertEqual(rate.shape, (1000,))

    def test_units_are_hz_quantity_vs_float_parity(self):
        # Quantity (ms) and float (seconds) inputs that describe the SAME physical
        # window/step must give the same Hz output.
        spikes = (np.random.RandomState(0).rand(500, 8) < 0.2).astype(float)
        r_q = braintools.analysis.spike.firing_rate(spikes, 5 * u.ms, 0.1 * u.ms)
        r_f = braintools.analysis.spike.firing_rate(spikes, 0.005, 0.0001)  # seconds
        np.testing.assert_allclose(np.asarray(r_q), np.asarray(r_f), rtol=1e-4)

    def test_constant_rate_value(self):
        # Every neuron spikes every step -> mean fraction 1.0 -> rate = 1/dt Hz.
        spikes = jnp.ones((1000, 4))
        dt = 0.001  # seconds
        rate = braintools.analysis.spike.firing_rate(spikes, 0.01, dt)
        # Away from the edges the boxcar average of a constant-1 signal is 1/dt = 1000 Hz.
        np.testing.assert_allclose(float(rate[500]), 1.0 / dt, rtol=1e-5)

    def test_wider_window_is_smoother(self):
        brainstate.random.seed(0)
        spikes = (brainstate.random.random((2000, 20)) < 0.1).astype(float)
        narrow = braintools.analysis.spike.firing_rate(spikes, 2 * u.ms, 0.1 * u.ms)
        wide = braintools.analysis.spike.firing_rate(spikes, 20 * u.ms, 0.1 * u.ms)
        # Wider smoothing -> smaller temporal variance.
        self.assertLess(float(jnp.std(wide)), float(jnp.std(narrow)))


class TestVictorPurpuraDistance(unittest.TestCase):
    def test_identical_trains(self):
        """Test distance between identical spike trains."""
        spikes = jnp.array([1.0, 2.0, 3.0, 4.0])
        distance = braintools.analysis.spike.victor_purpura_distance(spikes, spikes)
        self.assertAlmostEqual(float(distance), 0.0, places=5)

    def test_empty_trains(self):
        """Test distance with empty spike trains."""
        empty = jnp.array([])
        spikes = jnp.array([1.0, 2.0, 3.0])

        # Empty vs non-empty should equal number of spikes
        distance1 = braintools.analysis.spike.victor_purpura_distance(empty, spikes)
        self.assertEqual(float(distance1), 3.0)

        # Non-empty vs empty
        distance2 = braintools.analysis.spike.victor_purpura_distance(spikes, empty)
        self.assertEqual(float(distance2), 3.0)

        # Empty vs empty
        distance3 = braintools.analysis.spike.victor_purpura_distance(empty, empty)
        self.assertEqual(float(distance3), 0.0)

    def test_temporal_shift(self):
        """Test that temporal shifts are penalized correctly."""
        spikes1 = jnp.array([1.0, 2.0, 3.0])
        spikes2 = jnp.array([1.1, 2.1, 3.1])  # Shifted by 0.1

        # With high cost factor, should prefer matching with temporal penalty
        distance_high = braintools.analysis.spike.victor_purpura_distance(spikes1, spikes2, cost_factor=10.0)
        # With low cost factor, temporal shifts are cheaper
        distance_low = braintools.analysis.spike.victor_purpura_distance(spikes1, spikes2, cost_factor=1.0)

        self.assertGreater(distance_high, distance_low)

    def test_different_lengths(self):
        """Test distance between spike trains of different lengths."""
        spikes1 = jnp.array([1.0, 2.0])
        spikes2 = jnp.array([1.0, 2.0, 3.0, 4.0])

        distance = braintools.analysis.spike.victor_purpura_distance(spikes1, spikes2)
        # Should be at least the difference in number of spikes
        self.assertGreaterEqual(float(distance), 2.0)


class TestVanRossumDistance(unittest.TestCase):
    def test_identical_trains(self):
        """Test distance between identical spike trains."""
        spikes = jnp.array([1.0, 3.0, 5.0])
        distance = braintools.analysis.spike.van_rossum_distance(spikes, spikes)
        self.assertAlmostEqual(float(distance), 0.0, places=3)

    def test_empty_trains(self):
        """Test distance with empty spike trains."""
        empty = jnp.array([])
        spikes = jnp.array([1.0, 2.0, 3.0])

        distance1 = braintools.analysis.spike.van_rossum_distance(empty, spikes)
        distance2 = braintools.analysis.spike.van_rossum_distance(spikes, empty)
        distance3 = braintools.analysis.spike.van_rossum_distance(empty, empty)

        self.assertGreater(distance1, 0.0)
        self.assertGreater(distance2, 0.0)
        self.assertEqual(distance3, 0.0)

    def test_tau_effect(self):
        """Test that different tau values affect the distance."""
        spikes1 = jnp.array([1.0, 3.0])
        spikes2 = jnp.array([1.5, 3.5])  # Slightly shifted

        # Larger tau should make distance smaller (more temporal smoothing)
        distance_small_tau = braintools.analysis.spike.van_rossum_distance(spikes1, spikes2, tau=0.1)
        distance_large_tau = braintools.analysis.spike.van_rossum_distance(spikes1, spikes2, tau=1.0)

        self.assertGreater(distance_small_tau, distance_large_tau)

    def test_custom_t_max(self):
        """Test custom t_max parameter."""
        spikes1 = jnp.array([1.0, 2.0])
        spikes2 = jnp.array([1.1, 2.1])

        distance1 = braintools.analysis.spike.van_rossum_distance(spikes1, spikes2, t_max=10.0)
        distance2 = braintools.analysis.spike.van_rossum_distance(spikes1, spikes2, t_max=20.0)

        # Should be finite and positive
        self.assertGreater(distance1, 0.0)
        self.assertGreater(distance2, 0.0)


class TestVanRossumClosedForm(unittest.TestCase):
    """Pin the analytic value of the van Rossum distance.

    The integral is evaluated in closed form, so these must match exactly rather
    than approximately. These are regression tests for the former left-endpoint
    rectangle rule on a grid of step ``tau / 20``, which biased *every* result
    ~2.5% high (e.g. 0.724841 instead of ``sqrt(0.5) = 0.707107`` for a single
    spike against an empty train at ``tau = 1``).
    """

    def test_single_spike_vs_empty_is_sqrt_half_over_tau(self):
        # One spike convolved with (1/tau)exp(-t/tau) squares to 1/(2 tau).
        for tau in (0.5, 1.0, 2.0, 4.0):
            distance = braintools.analysis.spike.van_rossum_distance(
                jnp.array([0.0]), jnp.array([]), tau=tau, t_max=200.0
            )
            self.assertAlmostEqual(
                float(distance), math.sqrt(1.0 / (2.0 * tau)), places=5
            )

    def test_two_spikes_closed_form(self):
        # sqrt((1/tau) * (1 - exp(-gap/tau)))
        for gap in (0.5, 1.0, 2.0, 5.0):
            distance = braintools.analysis.spike.van_rossum_distance(
                jnp.array([0.0]), jnp.array([gap]), tau=1.0, t_max=200.0
            )
            self.assertAlmostEqual(
                float(distance), math.sqrt(1.0 - math.exp(-gap)), places=5
            )

    def test_multi_spike_matches_independent_closed_form(self):
        spikes1, spikes2, tau = [0.1, 0.5, 0.9], [0.2, 0.6], 1.0

        def pair_sum(times_a, times_b):
            if not len(times_a) or not len(times_b):
                return 0.0
            a = np.asarray(times_a, dtype=float)[:, None]
            b = np.asarray(times_b, dtype=float)[None, :]
            return float(np.sum(np.exp(-np.abs(a - b) / tau)))

        expected = math.sqrt(max(
            0.0,
            (pair_sum(spikes1, spikes1) + pair_sum(spikes2, spikes2)
             - 2.0 * pair_sum(spikes1, spikes2)) / (2.0 * tau)
        ))
        distance = braintools.analysis.spike.van_rossum_distance(
            jnp.array(spikes1), jnp.array(spikes2), tau=tau, t_max=500.0
        )
        self.assertAlmostEqual(float(distance), expected, places=5)

    def test_t_max_truncates_the_integral_exactly(self):
        # sqrt((1 - exp(-2 T / tau)) / (2 tau)) for one spike against an empty train.
        tau = 1.0
        for t_max in (2.0, 5.0, 20.0):
            distance = braintools.analysis.spike.van_rossum_distance(
                jnp.array([0.0]), jnp.array([]), tau=tau, t_max=t_max
            )
            expected = math.sqrt((1.0 - math.exp(-2.0 * t_max / tau)) / (2.0 * tau))
            self.assertAlmostEqual(float(distance), expected, places=5)

    def test_default_window_leaves_only_the_five_tau_tail(self):
        # The default t_max = max(spike) + 5*tau drops less than exp(-10) of the kernel,
        # so it must sit very close to the (essentially exact) large-window value.
        default = braintools.analysis.spike.van_rossum_distance(
            jnp.array([0.0]), jnp.array([]), tau=1.0
        )
        large = braintools.analysis.spike.van_rossum_distance(
            jnp.array([0.0]), jnp.array([]), tau=1.0, t_max=1000.0
        )
        self.assertLess(abs(default - large) / large, 1e-4)


class TestSpikeTrainSynchrony(unittest.TestCase):
    def test_perfect_synchrony(self):
        """Test with perfectly synchronized spikes."""
        # All neurons spike at the same times
        spikes = jnp.zeros((100, 5))
        spikes = spikes.at[20, :].set(1)
        spikes = spikes.at[50, :].set(1)
        spikes = spikes.at[80, :].set(1)

        synchrony = braintools.analysis.spike.spike_train_synchrony(spikes, window_size=10.0, dt=1.0)
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

        synchrony = braintools.analysis.spike.spike_train_synchrony(spikes, window_size=5.0, dt=1.0)
        # Should be low (no synchrony)
        self.assertLess(float(synchrony), 0.5)

    def test_single_neuron(self):
        """Test with single neuron."""
        spikes = jnp.zeros((100, 1))
        spikes = spikes.at[20, 0].set(1)

        synchrony = braintools.analysis.spike.spike_train_synchrony(spikes, window_size=10.0, dt=1.0)
        self.assertEqual(float(synchrony), 0.0)

    def test_no_spikes(self):
        """Test with no spikes."""
        spikes = jnp.zeros((100, 5))
        synchrony = braintools.analysis.spike.spike_train_synchrony(spikes, window_size=10.0, dt=1.0)
        self.assertEqual(float(synchrony), 0.0)

    def test_bounded_with_asymmetric_counts(self):
        """Regression for C9: many spikes in one train, one in the other.

        The previous ``min(N_i, N_j)`` normalization with one-directional counting
        could exceed 1; the symmetric ratio must stay in [0, 1].
        """
        spikes = jnp.zeros((100, 2))
        spikes = spikes.at[jnp.array([10, 20, 30, 40, 50]), 0].set(1)
        spikes = spikes.at[10, 1].set(1)  # single spike, coincident with one of train 0
        synchrony = braintools.analysis.spike.spike_train_synchrony(spikes, window_size=4.0, dt=1.0)
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

        sync_idx = braintools.analysis.spike.burst_synchrony_index(spikes, burst_threshold=3, max_isi=10.0, dt=1.0)
        # Should detect high burst synchrony
        self.assertGreater(float(sync_idx), 0.5)

    def test_no_bursts(self):
        """Test with isolated spikes (no bursts)."""
        spikes = jnp.zeros((1000, 5))

        # Add isolated spikes
        for neuron in range(5):
            spikes = spikes.at[100 + neuron * 50, neuron].set(1)

        sync_idx = braintools.analysis.spike.burst_synchrony_index(spikes, burst_threshold=3, max_isi=10.0, dt=1.0)
        self.assertEqual(float(sync_idx), 0.0)

    def test_asynchronous_bursts(self):
        """Test with bursts that don't overlap."""
        spikes = jnp.zeros((1000, 5))

        # Add non-overlapping bursts
        for neuron in range(5):
            burst_start = 100 + neuron * 100
            for spike_offset in range(5):
                spikes = spikes.at[burst_start + spike_offset, neuron].set(1)

        sync_idx = braintools.analysis.spike.burst_synchrony_index(spikes, burst_threshold=3, max_isi=10.0, dt=1.0)
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

        plv = braintools.analysis.spike.phase_locking_value(spikes, freq, dt)
        # Should be close to 1 for all neurons
        self.assertTrue(jnp.all(plv > 0.8))

    def test_no_phase_locking(self):
        """Test with random spikes (no phase locking)."""
        spikes = (brainstate.random.random((1000, 5)) < 0.05).astype(float)

        plv = braintools.analysis.spike.phase_locking_value(spikes, 10.0, 0.001)
        # Should be low for random spikes
        self.assertTrue(jnp.all(plv < 0.5))

    def test_no_spikes(self):
        """Test with no spikes."""
        spikes = jnp.zeros((1000, 5))

        plv = braintools.analysis.spike.phase_locking_value(spikes, 10.0, 0.001)
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

        sttc = braintools.analysis.spike.spike_time_tiling_coefficient(spikes, dt=0.001, tau=0.01)

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

        sttc = braintools.analysis.spike.spike_time_tiling_coefficient(spikes, dt=0.001, tau=0.01)

        # Off-diagonal should be close to 0
        off_diag = sttc[jnp.triu_indices_from(sttc, k=1)]
        self.assertTrue(jnp.all(jnp.abs(off_diag) < 0.3))

    def test_matrix_properties(self):
        """Test that STTC matrix has correct properties."""
        spikes = (brainstate.random.random((500, 4)) < 0.1).astype(float)

        sttc = braintools.analysis.spike.spike_time_tiling_coefficient(spikes, dt=0.001, tau=0.005)

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
        sttc = braintools.analysis.spike.spike_time_tiling_coefficient(spikes, dt=1.0, tau=5.0)
        self.assertAlmostEqual(float(sttc[0, 1]), 1.0, places=6)

    def test_anticorrelated_is_negative(self):
        """Disjoint, non-overlapping trains should give a negative STTC."""
        spikes = jnp.zeros((1000, 2))
        spikes = spikes.at[jnp.arange(0, 500, 20), 0].set(1)
        spikes = spikes.at[jnp.arange(500, 1000, 20), 1].set(1)
        sttc = braintools.analysis.spike.spike_time_tiling_coefficient(spikes, dt=1.0, tau=2.0)
        self.assertLess(float(sttc[0, 1]), 0.0)

    def test_empty_train_pair_is_zero(self):
        """A pair involving a silent neuron has STTC 0 (and diagonal stays 1)."""
        spikes = jnp.zeros((100, 2))
        spikes = spikes.at[jnp.array([10, 20, 30]), 0].set(1)  # neuron 1 stays silent
        sttc = braintools.analysis.spike.spike_time_tiling_coefficient(spikes, dt=1.0, tau=2.0)
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

        ci = braintools.analysis.spike.correlation_index(spikes, window_size=50.0, dt=1.0)
        # Should be close to 1
        self.assertGreater(float(ci), 0.8)

    def test_no_correlation(self):
        """Test with uncorrelated spike trains."""
        # Independent random spike trains
        spikes = (brainstate.random.random((1000, 5)) < 0.05).astype(float)

        ci = braintools.analysis.spike.correlation_index(spikes, window_size=50.0, dt=1.0)
        # Should be close to 0
        self.assertLess(jnp.abs(float(ci)), 0.3)

    def test_single_neuron(self):
        """Test with single neuron."""
        spikes = jnp.zeros((1000, 1))
        spikes = spikes.at[::100, 0].set(1)  # Regular spikes

        ci = braintools.analysis.spike.correlation_index(spikes, window_size=50.0, dt=1.0)
        self.assertEqual(float(ci), 0.0)

    def test_bounds(self):
        """Test that correlation index is properly bounded."""
        spikes = (brainstate.random.random((500, 8)) < 0.1).astype(float)

        ci = braintools.analysis.spike.correlation_index(spikes, window_size=25.0, dt=1.0)
        # Should be between -1 and 1
        self.assertGreaterEqual(float(ci), -1.0)
        self.assertLessEqual(float(ci), 1.0)

    def test_different_window_sizes(self):
        """Test effect of different window sizes."""
        spikes = (brainstate.random.random((1000, 5)) < 0.08).astype(float)

        ci_small = braintools.analysis.spike.correlation_index(spikes, window_size=10.0, dt=1.0)
        ci_large = braintools.analysis.spike.correlation_index(spikes, window_size=100.0, dt=1.0)

        # Both should be finite
        self.assertFalse(math.isnan(float(ci_small)))
        self.assertFalse(math.isnan(float(ci_large)))

    def test_too_few_bins_returns_zero(self):
        """A window covering (almost) the whole recording yields < 2 bins -> 0.0."""
        spikes = (brainstate.random.random((100, 3)) < 0.1).astype(float)
        ci = braintools.analysis.spike.correlation_index(spikes, window_size=100.0, dt=1.0)
        self.assertEqual(float(ci), 0.0)

    def test_constant_neuron_zero_variance(self):
        """A constant (zero-variance) binned train contributes a 0 correlation."""
        spikes = jnp.zeros((1000, 2))
        spikes = spikes.at[:, 0].set(1.0)  # neuron 0 spikes every step -> constant bins
        spikes = spikes.at[jnp.arange(0, 1000, 30), 1].set(1.0)
        ci = braintools.analysis.spike.correlation_index(spikes, window_size=50.0, dt=1.0)
        self.assertEqual(float(ci), 0.0)


class TestTimeUnitHandling(unittest.TestCase):
    """Times given as Quantities in any unit, or as documented plain floats, agree."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.rate_spikes = (rng.random((20000, 30)) < 0.002).astype(float)  # ~20 Hz at dt = 0.1 ms
        s = np.zeros((10000, 2))
        ev = rng.choice(9900, 40, replace=False)
        s[ev, 0] = 1
        s[np.clip(ev + rng.integers(-3, 4, 40), 0, 9999), 1] = 1
        self.paired = s

    def test_firing_rate_mixed_units(self):
        ref = np.asarray(braintools.analysis.spike.firing_rate(self.rate_spikes, 10 * u.ms, 0.1 * u.ms))
        for width, dt in [(0.01 * u.second, 0.1 * u.ms), (0.01, 0.0001), (10 * u.ms, 0.0001)]:
            np.testing.assert_allclose(braintools.analysis.spike.firing_rate(self.rate_spikes, width, dt), ref, rtol=1e-5)
        self.assertAlmostEqual(float(ref.mean()), 20.0, delta=1.0)

    def test_sttc_default_tau_is_5ms(self):
        # The old default (0.005) was a zero-sample window for dt in ms.
        for kw in [dict(dt=1 * u.ms), dict(dt=1.0), dict(dt=0.001 * u.second, tau=5 * u.ms),
                   dict(dt=1 * u.ms, tau=0.005 * u.second)]:
            m = np.asarray(braintools.analysis.spike.spike_time_tiling_coefficient(self.paired, **kw))
            self.assertAlmostEqual(float(m[0, 1]), 1.0, places=5, msg=str(kw))

    def test_window_functions_convert_units(self):
        f = braintools.analysis.spike
        for fn in (f.spike_train_synchrony, f.correlation_index):
            ref = fn(self.paired, window_size=10., dt=1.)
            self.assertAlmostEqual(float(fn(self.paired, window_size=10 * u.ms, dt=0.001 * u.second)), float(ref), places=6)
            self.assertAlmostEqual(float(fn(self.paired, window_size=0.01 * u.second, dt=1 * u.ms)), float(ref), places=6)
        ref = f.burst_synchrony_index(self.paired, max_isi=50., dt=1.)
        self.assertAlmostEqual(float(f.burst_synchrony_index(self.paired, max_isi=0.05 * u.second, dt=1 * u.ms)), float(ref), places=6)


class TestFIR1PhaseLockingQuantityDt(unittest.TestCase):
    def _spikes(self):
        rng = np.random.RandomState(0)
        return jnp.asarray((rng.rand(1000, 5) < 0.1).astype(float))

    def test_quantity_dt_does_not_crash(self):
        # FIR-1: a brainunit.Quantity dt previously raised TypeError in jnp.exp.
        spikes = self._spikes()
        plv = braintools.analysis.spike.phase_locking_value(spikes, 10.0, dt=0.1 * u.ms)
        self.assertEqual(plv.shape, (5,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(plv))))

    def test_quantity_dt_matches_equivalent_seconds(self):
        # FIR-1: a Quantity dt must be interpreted in seconds (reference_freq is Hz),
        # so 0.1 ms must equal the float value 1e-4 s.
        spikes = self._spikes()
        plv_q = braintools.analysis.spike.phase_locking_value(spikes, 10.0, dt=0.1 * u.ms)
        plv_f = braintools.analysis.spike.phase_locking_value(spikes, 10.0, dt=1e-4)
        np.testing.assert_allclose(np.asarray(plv_q), np.asarray(plv_f), rtol=1e-5, atol=1e-6)


class TestFIR3BurstSynchronyEndpointTouch(unittest.TestCase):
    def test_endpoint_touching_bursts_count_as_synchronous(self):
        # FIR-3: bursts that touch at a single instant are the strongest synchrony
        # and must not be excluded by a strict `> 0` overlap test.
        spikes = np.zeros((20, 2))
        spikes[0:5, 0] = 1.0   # neuron 0 burst spans t=0..4
        spikes[4:9, 1] = 1.0   # neuron 1 burst spans t=4..8 (touches at t=4)
        idx = braintools.analysis.spike.burst_synchrony_index(jnp.asarray(spikes), burst_threshold=3,
                                      max_isi=2.0, dt=1.0)
        self.assertGreater(float(idx), 0.0)


class TestFIR2VictorPurpuraCorrectness(unittest.TestCase):
    def test_identical_trains_zero_distance(self):
        s = jnp.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(victor_purpura_distance(s, s, cost_factor=1.0), 0.0, places=5)

    def test_single_extra_spike_costs_one(self):
        s1 = jnp.array([1.0, 2.0])
        s2 = jnp.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(victor_purpura_distance(s1, s2, cost_factor=1.0), 1.0, places=5)

    def test_small_shift_costs_q_times_delta(self):
        s1 = jnp.array([1.0])
        s2 = jnp.array([1.1])
        # cheaper to shift (q*0.1=0.5) than delete+insert (=2)
        self.assertAlmostEqual(victor_purpura_distance(s1, s2, cost_factor=5.0), 0.5, places=5)

    def test_far_shift_prefers_delete_insert(self):
        s1 = jnp.array([1.0])
        s2 = jnp.array([100.0])
        # shifting costs q*99 = 99 >> delete+insert = 2
        self.assertAlmostEqual(victor_purpura_distance(s1, s2, cost_factor=1.0), 2.0, places=5)


class TestFIR6CorrelationIndexBounds(unittest.TestCase):
    def test_result_within_pm_one(self):
        rng = np.random.RandomState(4)
        base = (rng.rand(1000, 1) < 0.1).astype(float)
        spikes = jnp.asarray(np.concatenate([base, base, rng.rand(1000, 1) < 0.1], axis=1).astype(float))
        ci = braintools.analysis.spike.correlation_index(spikes, window_size=50.0, dt=1.0)
        self.assertGreaterEqual(float(ci), -1.0)
        self.assertLessEqual(float(ci), 1.0)


# ----------------------------------------------------------------------------
# Version 0.5: jittable / loop-free rewrites, compared with the previous
# implementations (copied above as ``_legacy_*``) on concrete inputs.
# ----------------------------------------------------------------------------

@pytest.fixture
def x64():
    # Importing braintools resets the flag, so it is enabled here, after import.
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def _random_train(rng, n_max=8, integer=False):
    n = int(rng.integers(0, n_max + 1))
    if integer:  # ties between and within trains
        return np.sort(rng.integers(0, 10, n).astype(float))
    return np.sort(rng.uniform(0., 10., n))


def _random_raster(rng, n_time, n_neurons, p=None, silent=()):
    p = rng.uniform(0.01, 0.2) if p is None else p
    spikes = (rng.random((n_time, n_neurons)) < p).astype(float)
    for k in silent:
        spikes[:, k] = 0.
    return spikes


def _bursty_raster(rng, n_time, n_neurons):
    spikes = np.zeros((n_time, n_neurons))
    for n in range(n_neurons):
        for start in rng.integers(0, n_time - 6, rng.integers(0, 6)):
            length = int(rng.integers(1, 6))
            step = int(rng.integers(1, 3))
            spikes[start:start + length * step:step, n] = 1.
        spikes[rng.random(n_time) < 0.01, n] = 1.
    return spikes


class TestVictorPurpuraJittable:
    def test_matches_legacy_exactly_float64(self, x64):
        rng = np.random.default_rng(0)
        for k in range(40):
            a = _random_train(rng, integer=k % 2 == 0)
            b = _random_train(rng, integer=k % 2 == 0)
            if k % 5 == 0:
                rng.shuffle(a)  # unsorted input follows the same DP
            q = float(rng.choice([0., 0.3, 1., 5., 50.]))
            new = victor_purpura_distance(jnp.asarray(a), jnp.asarray(b), q)
            assert isinstance(new, jax.Array) and new.shape == () and new.dtype == jnp.float64
            assert float(new) == _legacy_victor_purpura_distance(a, b, q)

    def test_matches_legacy_float32(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = _random_train(rng).astype(np.float32)
            b = _random_train(rng).astype(np.float32)
            q = float(rng.choice([0.3, 1., 5.]))
            new = victor_purpura_distance(jnp.asarray(a), jnp.asarray(b), q)
            assert new.dtype == jnp.float32
            np.testing.assert_allclose(float(new), _legacy_victor_purpura_distance(a, b, q), rtol=1e-6)

    def test_empty_trains_return_arrays(self):
        empty = jnp.array([])
        d = victor_purpura_distance(empty, empty)
        assert isinstance(d, jax.Array) and d.shape == () and float(d) == 0.0
        assert float(victor_purpura_distance(empty, jnp.array([1., 2.]))) == 2.0
        assert float(victor_purpura_distance(jnp.array([1., 2., 3.]), empty)) == 3.0

    def test_inf_padding_is_absent(self):
        a = jnp.array([1., 2.5, 4.])
        b = jnp.array([1.2, 3.])
        ref = float(victor_purpura_distance(a, b, 2.))
        pad = jnp.inf
        assert float(victor_purpura_distance(jnp.array([1., 2.5, 4., pad, pad]), b, 2.)) == ref
        assert float(victor_purpura_distance(a, jnp.array([1.2, pad, 3., pad]), 2.)) == ref
        assert float(victor_purpura_distance(jnp.full(3, pad), jnp.full(2, pad), 2.)) == 0.0
        assert float(victor_purpura_distance(jnp.full(3, pad), b, 2.)) == 2.0

    def test_jit_and_vmap_over_padded_rows(self):
        st = SpikeTrains.from_list([[1., 2., 3.], [], [0.5]], t_start=0., t_stop=4.)
        other = SpikeTrains.from_list([[1.1, 2.9], [2.], []], t_start=0., t_stop=4.)
        f = jax.jit(jax.vmap(lambda x, y: victor_purpura_distance(x, y, 2.)))
        got = np.asarray(f(st.times, other.times))
        expected = [_legacy_victor_purpura_distance(r, s, 2.)
                    for r, s in zip(st.to_list(), other.to_list())]
        np.testing.assert_allclose(got, expected, rtol=1e-6)
        traced_q = jax.jit(victor_purpura_distance)(jnp.array([1.]), jnp.array([1.1]), 5.)
        np.testing.assert_allclose(float(traced_q), 0.5, rtol=1e-5)

    def test_grad(self):
        # d = q |s1 - s2| when shifting is cheaper than delete + insert.
        g_q = jax.grad(lambda q: victor_purpura_distance(jnp.array([1.]), jnp.array([1.1]), q))(5.)
        np.testing.assert_allclose(float(g_q), 0.1, rtol=1e-5)
        g_s = jax.grad(lambda s: victor_purpura_distance(s, jnp.array([1.1, jnp.inf]), 5.))(jnp.array([1., jnp.inf]))
        np.testing.assert_allclose(np.asarray(g_s), [-5., 0.], rtol=1e-6)

    @pytest.mark.parametrize('b', [[1.2, 2.5, 3.], [1.2, 2.5, jnp.inf]])
    def test_grad_finite_with_padding(self, x64, b):
        # Regression: q * |s - inf| made d/dq = 0 * inf = nan whenever padding was present.
        a = jnp.array([1., 2., jnp.inf])
        b = jnp.array(b)
        f = lambda q, s, t: victor_purpura_distance(s, t, q)
        np.testing.assert_allclose(float(f(2., a, jnp.array([1.2, 2.5, 3.]))), 2.4, rtol=1e-12)
        g_q, g_a, g_b = jax.grad(f, argnums=(0, 1, 2))(2., a, b)
        for g in (g_q, g_a, g_b):
            assert np.all(np.isfinite(np.asarray(g)))
        h = 1e-6
        np.testing.assert_allclose(float(g_q), (float(f(2. + h, a, b)) - float(f(2. - h, a, b))) / (2 * h),
                                   rtol=1e-6)
        for k in range(2):  # valid spikes of ``a``
            e = jnp.zeros(3).at[k].set(h)
            fd = (float(f(2., a + e, b)) - float(f(2., a - e, b))) / (2 * h)
            np.testing.assert_allclose(float(g_a[k]), fd, rtol=1e-6, atol=1e-9)
        assert float(g_a[2]) == 0.0

    def test_quantity_magnitudes(self):
        a = jnp.array([1., 2.]) * u.ms
        b = jnp.array([1.5]) * u.ms
        assert float(victor_purpura_distance(a, b, 1.)) == float(victor_purpura_distance(a.mantissa, b.mantissa, 1.))

    def test_rejects_2d(self):
        with pytest.raises(ValueError):
            victor_purpura_distance(jnp.ones((2, 2)), jnp.ones(2))


class TestVanRossumJittable:
    @pytest.mark.parametrize('use_x64', [False, True])
    def test_matches_legacy(self, use_x64, request):
        if use_x64:
            request.getfixturevalue('x64')
        rng = np.random.default_rng(2)
        for k in range(25):
            a = jnp.asarray(_random_train(rng))
            b = jnp.asarray(_random_train(rng))
            tau = float(rng.choice([0.2, 1., 3.]))
            t_max = None if k % 2 else float(rng.choice([5., 12., 30.]))
            new = van_rossum_distance(a, b, tau=tau, t_max=t_max)
            old = _legacy_van_rossum_distance(a, b, tau=tau, t_max=t_max)
            assert isinstance(new, jax.Array) and new.shape == ()
            np.testing.assert_allclose(float(new), old, rtol=1e-12 if use_x64 else 1e-6, atol=1e-12)

    def test_empty_returns_zero_array(self):
        d = van_rossum_distance(jnp.array([]), jnp.array([]))
        assert isinstance(d, jax.Array) and float(d) == 0.0
        assert float(van_rossum_distance(jnp.full(3, jnp.inf), jnp.full(2, jnp.inf))) == 0.0

    def test_inf_padding_is_masked(self):
        a = jnp.array([1., 3.])
        b = jnp.array([1.5])
        ref = float(van_rossum_distance(a, b, tau=0.5))
        pad = van_rossum_distance(jnp.array([1., jnp.inf, 3.]), jnp.array([1.5, jnp.inf]), tau=0.5)
        np.testing.assert_allclose(float(pad), ref, rtol=1e-6)
        assert np.isfinite(float(pad))

    def test_jit_requires_t_max(self):
        a, b = jnp.array([1., 3.]), jnp.array([1.5])
        with pytest.raises(ValueError, match='t_max'):
            jax.jit(van_rossum_distance)(a, b)
        with pytest.raises(ValueError, match='t_max'):
            jax.vmap(van_rossum_distance)(a[None], b[None])
        f = jax.jit(lambda x, y: van_rossum_distance(x, y, tau=0.5, t_max=10.))
        np.testing.assert_allclose(float(f(a, b)), float(van_rossum_distance(a, b, tau=0.5, t_max=10.)), rtol=1e-6)

    def test_vmap_over_padded_rows(self):
        st = SpikeTrains.from_list([[1., 2., 3.], [], [0.5]], t_start=0., t_stop=4.)
        other = SpikeTrains.from_list([[1.1, 2.9], [2.], []], t_start=0., t_stop=4.)
        got = jax.vmap(lambda x, y: van_rossum_distance(x, y, tau=1., t_max=20.))(st.times, other.times)
        expected = [_legacy_van_rossum_distance(jnp.asarray(r), jnp.asarray(s), tau=1., t_max=20.)
                    for r, s in zip(st.to_list(), other.to_list())]
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)

    def test_grad(self):
        a = jnp.array([1., 3., jnp.inf])
        b = jnp.array([1.5, jnp.inf])
        f = lambda tau: van_rossum_distance(a, b, tau=tau, t_max=30.)
        g = float(jax.grad(f)(1.))
        h = 1e-3
        np.testing.assert_allclose(g, (float(f(1. + h)) - float(f(1. - h))) / (2 * h), rtol=1e-2)
        g_s = jax.grad(lambda s: van_rossum_distance(s, b, tau=1., t_max=30.))(a)
        assert np.all(np.isfinite(np.asarray(g_s)))
        assert float(g_s[2]) == 0.0


class TestSpikeTrainSynchronyJittable:
    @pytest.mark.parametrize('use_x64', [False, True])
    def test_matches_legacy(self, use_x64, request):
        if use_x64:
            request.getfixturevalue('x64')
        rng = np.random.default_rng(3)
        for k, window in enumerate([0., 1., 3., 4.5, 10., 1e6]):
            spikes = _random_raster(rng, 120, 6, silent=(k % 6,) if k % 2 else ())
            new = spike_train_synchrony(jnp.asarray(spikes), window_size=window, dt=1.)
            old = _legacy_spike_train_synchrony(spikes, window_size=window, dt=1.)
            assert isinstance(new, jax.Array) and new.shape == ()
            np.testing.assert_allclose(float(new), old, rtol=1e-12 if use_x64 else 1e-6)

    def test_degenerate_inputs(self):
        rng = np.random.default_rng(4)
        one = _random_raster(rng, 50, 1)
        silent = np.zeros((50, 4))
        lonely = silent.copy()
        lonely[10, 2] = 1.
        for spikes in (one, silent, lonely):
            new = spike_train_synchrony(jnp.asarray(spikes), window_size=4., dt=1.)
            assert isinstance(new, jax.Array) and float(new) == 0.0
            assert _legacy_spike_train_synchrony(spikes, window_size=4., dt=1.) == 0.0
        # Negative window: no coincidences at all (as before).
        spikes = _random_raster(rng, 50, 3, p=0.3)
        assert float(spike_train_synchrony(jnp.asarray(spikes), window_size=-2., dt=1.)) == \
               _legacy_spike_train_synchrony(spikes, window_size=-2., dt=1.)

    def test_jit_and_vmap_over_trials(self):
        rng = np.random.default_rng(5)
        trials = np.stack([_random_raster(rng, 80, 4, silent=(k,)) for k in range(3)])
        f = jax.jit(lambda s: spike_train_synchrony(s, window_size=6., dt=1.))
        np.testing.assert_allclose(float(f(trials[0])), _legacy_spike_train_synchrony(trials[0], 6., 1.), rtol=1e-6)
        got = jax.vmap(f)(jnp.asarray(trials))
        expected = [_legacy_spike_train_synchrony(t, 6., 1.) for t in trials]
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-6)


class TestLoopFreeRewrites:
    @pytest.mark.parametrize('dt', [1., 0.1, 1. * u.ms])
    def test_burst_synchrony_matches_legacy_exactly(self, dt):
        rng = np.random.default_rng(6)
        for k in range(12):
            spikes = _bursty_raster(rng, 200, 5) if k % 3 else _random_raster(rng, 200, 5, silent=(k % 5,))
            for threshold in (1, 2, 3, 4):
                for max_isi in (0.5, 1., 2., 3., 100.):
                    max_isi = max_isi * u.ms if isinstance(dt, u.Quantity) else max_isi
                    new = burst_synchrony_index(jnp.asarray(spikes), threshold, max_isi, dt)
                    old = _legacy_burst_synchrony_index(jnp.asarray(spikes), threshold, max_isi, dt)
                    assert type(new) is float
                    assert new == old, (k, threshold, max_isi)

    def test_burst_synchrony_degenerate(self):
        assert burst_synchrony_index(np.zeros((30, 3)), dt=1.) == 0.0
        spikes = np.zeros((30, 2))
        spikes[[2, 9, 20], 0] = 1.  # isolated spikes, no burst
        assert burst_synchrony_index(spikes, burst_threshold=2, max_isi=1., dt=1.) == 0.0
        assert _legacy_burst_synchrony_index(spikes, burst_threshold=2, max_isi=1., dt=1.) == 0.0

    @pytest.mark.parametrize('use_x64', [False, True])
    def test_sttc_matches_legacy_exactly(self, use_x64, request):
        if use_x64:
            request.getfixturevalue('x64')
        rng = np.random.default_rng(7)
        for k, tau in enumerate([0., 1., 2.5, 5., 40., 1e4]):
            spikes = _random_raster(rng, 150, 5, silent=(k % 5,) if k % 2 else ())
            new = spike_time_tiling_coefficient(jnp.asarray(spikes), dt=1., tau=tau)
            old = _legacy_spike_time_tiling_coefficient(spikes, dt=1., tau=tau)
            assert new.dtype == old.dtype and new.shape == (5, 5)
            np.testing.assert_array_equal(np.asarray(new), np.asarray(old))

    def test_sttc_negative_tau_covers_nothing(self):
        # The legacy loop sliced ``covered[s + 3:s - 2]``, which wraps around for
        # spikes in the first samples; a negative window now covers nothing.
        spikes = np.zeros((30, 2))
        spikes[[0, 10], 0] = 1.
        spikes[[5, 20], 1] = 1.
        sttc = np.asarray(spike_time_tiling_coefficient(spikes, dt=1., tau=-3.))
        np.testing.assert_array_equal(sttc, np.eye(2))

    def test_sttc_dense_and_degenerate(self):
        # Every sample covered (T_A = T_B = 1, P = 1): denominators vanish -> 0.
        spikes = np.ones((20, 3))
        np.testing.assert_array_equal(np.asarray(spike_time_tiling_coefficient(spikes, dt=1., tau=3.)),
                                      np.asarray(_legacy_spike_time_tiling_coefficient(spikes, dt=1., tau=3.)))
        single = np.asarray(spike_time_tiling_coefficient(np.ones((20, 1)), dt=1., tau=3.))
        np.testing.assert_array_equal(single, [[1.]])

    @pytest.mark.parametrize('use_x64', [False, True])
    def test_correlation_index_matches_legacy(self, use_x64, request):
        if use_x64:
            request.getfixturevalue('x64')
        rng = np.random.default_rng(8)
        for k, window in enumerate([1., 5., 7.5, 20., 50.]):
            spikes = _random_raster(rng, 300, 5, silent=(k % 5,) if k % 2 else ())
            if k == 2:
                spikes[:, 1] = 1.  # constant column: zero variance
            new = correlation_index(jnp.asarray(spikes), window_size=window, dt=1.)
            old = _legacy_correlation_index(jnp.asarray(spikes), window_size=window, dt=1.)
            assert type(new) is float
            np.testing.assert_allclose(new, old, rtol=1e-12 if use_x64 else 1e-6, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
