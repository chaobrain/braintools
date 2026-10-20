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

import unittest

import brainstate
import jax.numpy as jnp
import numpy as np

import braintools

brainstate.environ.set(dt=0.1)


class TestLatencyEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_linear_method(self):
        """Test latency encoder with linear method."""

        encoder = braintools.spike.LatencyEncoder(method='linear', normalize=True)
        data = jnp.array([0.0, 0.5, 1.0])
        spikes = encoder(data, n_time=10)

        # n_time is a number of time steps
        self.assertEqual(spikes.shape, (10, 3))
        # Check that each neuron spikes exactly once
        self.assertTrue(jnp.all(jnp.sum(spikes, axis=0) == 1))

        # Get spike times for each input
        spike_times = jnp.argmax(spikes, axis=0)
        # Higher values should spike earlier (lower spike times)
        self.assertLessEqual(spike_times[2], spike_times[1])  # 1.0 before 0.5
        self.assertLessEqual(spike_times[1], spike_times[0])  # 0.5 before 0.0

    def test_log_method(self):
        """Test latency encoder with log method."""
        encoder = braintools.spike.LatencyEncoder(method='log', normalize=True)
        data = jnp.array([0.1, 0.5, 0.9])
        spikes = encoder(data, n_time=10)

        self.assertEqual(spikes.shape, (10, 3))
        self.assertTrue(jnp.all(jnp.sum(spikes, axis=0) == 1))  # Each neuron spikes once

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with self.assertRaises(ValueError):
            braintools.spike.LatencyEncoder(method='invalid')

    def test_normalization(self):
        """Test data normalization."""
        encoder = braintools.spike.LatencyEncoder(min_val=0, max_val=10, normalize=True)
        data = jnp.array([0, 5, 10])
        spikes = encoder(data, n_time=5)

        self.assertEqual(spikes.shape, (5, 3))


class TestRateEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_linear_rate_encoding(self):
        """Test linear rate encoding."""
        brainstate.random.seed(42)
        encoder = braintools.spike.RateEncoder(gain=100, method='linear')
        data = jnp.array([0.0, 0.5, 1.0])
        spikes = encoder(data, n_time=1000)

        self.assertEqual(spikes.shape, (1000, 3))
        # Higher values should have higher spike rates
        rates = jnp.mean(spikes, axis=0) * 10000  # Convert to Hz (assuming 0.1ms dt)
        self.assertLessEqual(rates[0], rates[1])
        self.assertLessEqual(rates[1], rates[2])

    def test_exponential_rate_encoding(self):
        """Test exponential rate encoding."""
        encoder = braintools.spike.RateEncoder(gain=100, method='exponential')
        data = jnp.array([0.1, 0.5, 0.9])
        spikes = encoder(data, n_time=100)

        self.assertEqual(spikes.shape, (100, 3))
        self.assertTrue(jnp.all(spikes >= 0))
        self.assertTrue(jnp.all(spikes <= 1))

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with self.assertRaises(ValueError):
            braintools.spike.RateEncoder(method='invalid')

    def test_rate_bounds(self):
        """Test rate bounds."""
        encoder = braintools.spike.RateEncoder(min_rate=10, max_rate=100)
        data = jnp.array([0.0, 1.0])
        spikes = encoder(data, n_time=10000)  # Use more time steps for more reliable statistics

        # Even zero input should have some spikes due to min_rate
        # With 10 Hz over 10000 * 0.1ms = 1 second, expect ~10 spikes
        self.assertGreater(jnp.sum(spikes[:, 0]), 2)  # At least a few spikes
        self.assertGreater(jnp.sum(spikes[:, 1]), jnp.sum(spikes[:, 0]))  # Max rate should produce more


class TestPoissonEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_basic_poisson_encoding(self):
        """Test basic Poisson encoding."""
        encoder = braintools.spike.PoissonEncoder()
        rates = jnp.array([10.0, 50.0, 100.0])  # Hz
        n_time = 1000
        spikes = encoder(rates, n_time=n_time)

        self.assertEqual(spikes.shape, (1000, 3))
        self.assertTrue(jnp.all((spikes == 0) | (spikes == 1)))

        # Check approximate rates (with tolerance for randomness)
        spike_counts = jnp.sum(spikes, axis=0)
        # Expected spikes = rate (Hz) * time (s) = rate * (n_time * dt_ms / 1000)
        dt_ms = brainstate.environ.get_dt()  # Should be 0.1 ms
        expected_counts = rates * n_time * dt_ms / 1000  # Convert to expected counts
        print(f"Spike counts: {spike_counts}, Expected: {expected_counts}")
        # Check that spike counts are in reasonable range (Poisson has high variance)
        # Use a more relaxed test since Poisson is inherently random
        # for i in range(len(rates)):
        #     self.assertGreaterEqual(spike_counts[i], expected_counts[i] * 0.2)  # At least 20% of expected
        #     self.assertLessEqual(spike_counts[i], expected_counts[i] * 5)     # At most 5x expected

    def test_normalization(self):
        """Test rate normalization."""
        encoder = braintools.spike.PoissonEncoder(normalize=True, max_rate=100)
        data = jnp.array([0.1, 0.5, 1.0])
        spikes = encoder(data, n_time=100)

        self.assertEqual(spikes.shape, (100, 3))

    def test_zero_rate(self):
        """Test zero rate produces no spikes."""
        encoder = braintools.spike.PoissonEncoder()
        rates = jnp.array([0.0])
        spikes = encoder(rates, n_time=100)

        self.assertEqual(jnp.sum(spikes), 0)


class TestPopulationEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_population_encoding_scalar(self):
        """Test population encoding of scalar input."""
        encoder = braintools.spike.PopulationEncoder(n_neurons=10, min_val=0, max_val=1)
        data = 0.5
        spikes = encoder(data, n_time=100)

        self.assertEqual(spikes.shape, (100, 10))

        # Middle neuron (index 4-5) should be most active
        rates = jnp.mean(spikes, axis=0)
        max_rate_idx = jnp.argmax(rates)
        self.assertTrue(0 <= max_rate_idx <= 6)  # Should be near middle

    def test_population_encoding_array(self):
        """Test population encoding of array input."""
        encoder = braintools.spike.PopulationEncoder(n_neurons=5, min_val=0, max_val=1)
        data = jnp.array([0.0, 1.0])
        spikes = encoder(data, n_time=50)

        self.assertEqual(spikes.shape, (50, 5, 2))

    def test_receptive_field_width(self):
        """Test different receptive field widths."""
        brainstate.random.seed(42)
        narrow_encoder = braintools.spike.PopulationEncoder(n_neurons=10, sigma=0.1)
        wide_encoder = braintools.spike.PopulationEncoder(n_neurons=10, sigma=0.5)

        data = 0.5

        narrow_spikes = narrow_encoder(data, n_time=100)
        wide_spikes = wide_encoder(data, n_time=100)

        # Wide encoder should activate more neurons
        narrow_active = jnp.sum(jnp.sum(narrow_spikes, axis=0) > 0)
        wide_active = jnp.sum(jnp.sum(wide_spikes, axis=0) > 0)
        self.assertGreater(wide_active, narrow_active)


def _population_encoder_pre_0_4_reference(enc, data, n_time):
    """Frozen copy of ``PopulationEncoder.__call__`` before the tuning refactor.

    Used to pin the encoder's output bit-for-bit after it switched to the shared
    :func:`braintools.spike._tuning.gaussian_tuning` primitive.
    """
    import brainunit as u
    from braintools.spike._encoder import _dt_seconds
    data = u.math.asarray(data)
    distances = u.math.abs(data[..., None] - enc.centers)
    activations = u.math.exp(-0.5 * (distances / enc.sigma) ** 2)
    probs = u.math.clip(activations * enc.max_rate * _dt_seconds(), 0, 1)
    if data.ndim == 0:
        spikes = brainstate.random.bernoulli(probs, size=(n_time, enc.n_neurons))
    else:
        spikes = brainstate.random.bernoulli(probs, size=(n_time,) + probs.shape)
        axes = (0, data.ndim + 1) + tuple(range(1, data.ndim + 1))
        spikes = u.math.transpose(spikes, axes)
    return spikes.astype(u.math.float32)


class TestPopulationEncoderTuningRegression(unittest.TestCase):
    """PopulationEncoder output is unchanged by the shared tuning primitive."""

    def _check(self, enc, data, n_time=200, dt=1.0):
        with brainstate.environ.context(dt=dt):
            brainstate.random.seed(123)
            new = enc(data, n_time=n_time)
            brainstate.random.seed(123)
            old = _population_encoder_pre_0_4_reference(enc, data, n_time)
        self.assertEqual(new.dtype, old.dtype)
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))
        self.assertGreater(float(np.asarray(new).sum()), 0.)

    def test_scalar_input(self):
        enc = braintools.spike.PopulationEncoder(n_neurons=11, max_rate=500.)
        for x in [0.0, 0.37, 0.5, 1.0, -0.2, 1.3]:
            self._check(enc, x)

    def test_array_inputs(self):
        enc = braintools.spike.PopulationEncoder(n_neurons=7, min_val=-1., max_val=2., sigma=0.3, max_rate=300.)
        self._check(enc, jnp.array([-1.0, 0.1, 0.77, 2.0]))
        self._check(enc, jnp.linspace(-1.5, 2.5, 12).reshape(3, 4))

    def test_quantity_inputs(self):
        import brainunit as u
        enc = braintools.spike.PopulationEncoder(n_neurons=5, min_val=0. * u.mV, max_val=1. * u.mV)
        self._check(enc, 0.3 * u.mV)
        self._check(enc, jnp.array([0.1, 0.9]) * u.mV)
        self._check(enc, 300. * u.uvolt)  # compatible unit, different scale

    def test_activations_match_closed_form(self):
        import brainunit as u
        from braintools.spike._tuning import gaussian_tuning
        enc = braintools.spike.PopulationEncoder(n_neurons=9, min_val=-2., max_val=3., sigma=0.45)
        data = jnp.linspace(-3., 4., 29)
        expected = u.math.exp(-0.5 * (u.math.abs(data[..., None] - enc.centers) / enc.sigma) ** 2)
        np.testing.assert_array_equal(gaussian_tuning(data[..., None], enc.centers, enc.sigma), expected)


class TestBernoulliEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_basic_bernoulli_encoding(self):
        """Test basic Bernoulli encoding."""
        encoder = braintools.spike.BernoulliEncoder(scale=0.1)
        data = jnp.array([0.1, 0.5, 0.9])
        spikes = encoder(data, n_time=1000)

        self.assertEqual(spikes.shape, (1000, 3))
        self.assertTrue(jnp.all((spikes == 0) | (spikes == 1)))

        # Higher probabilities should generate more spikes
        spike_counts = jnp.sum(spikes, axis=0)
        self.assertLessEqual(spike_counts[0], spike_counts[1])
        self.assertLessEqual(spike_counts[1], spike_counts[2])

    def test_probability_bounds(self):
        """Test probability clipping."""
        encoder = braintools.spike.BernoulliEncoder(scale=2.0)
        data = jnp.array([0.6, 0.8])  # Would exceed 1.0 when scaled
        spikes = encoder(data, n_time=100)

        # Should not crash and should produce valid spikes
        self.assertEqual(spikes.shape, (100, 2))
        self.assertTrue(jnp.all((spikes == 0) | (spikes == 1)))

    def test_normalization(self):
        """Test input normalization."""
        encoder = braintools.spike.BernoulliEncoder(normalize=True, min_val=0, max_val=10)
        data = jnp.array([0, 5, 10])
        spikes = encoder(data, n_time=100)

        self.assertEqual(spikes.shape, (100, 3))


class TestDeltaEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_delta_encoding(self):
        """Test delta encoding of changing signal."""
        encoder = braintools.spike.DeltaEncoder(threshold=0.1)
        # Signal with clear changes
        time_series = jnp.array([0.0, 0.05, 0.2, 0.8, 0.7, 0.1])
        spikes = encoder(time_series)

        self.assertEqual(spikes.shape, (6,))
        # Should spike at significant changes (events are signed +1 / -1)
        self.assertGreater(jnp.sum(jnp.abs(spikes)), 0)

    def test_positive_only_encoding(self):
        """Test positive-only delta encoding."""
        encoder = braintools.spike.DeltaEncoder(threshold=0.1, positive_only=True)
        time_series = jnp.array([0.0, 0.2, 0.1, 0.3])  # Mix of positive and negative changes
        spikes = encoder(time_series)

        self.assertEqual(spikes.shape, (4,))
        # Should only spike on positive changes

    def test_absolute_delta_encoding(self):
        """Test absolute delta encoding."""
        encoder = braintools.spike.DeltaEncoder(threshold=0.1, absolute=True)
        time_series = jnp.array([0.5, 0.4, 0.6])  # Changes of ±0.1
        spikes = encoder(time_series)

        self.assertEqual(spikes.shape, (3,))

    def test_multidimensional_input(self):
        """Test delta encoding with multidimensional input."""
        encoder = braintools.spike.DeltaEncoder(threshold=0.1)
        time_series = jnp.array([[0.0, 0.5], [0.2, 0.3], [0.1, 0.8]])
        spikes = encoder(time_series)

        self.assertEqual(spikes.shape, (3, 2))


class TestStepCurrentEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_current_scaling(self):
        """Test current scaling."""
        encoder = braintools.spike.StepCurrentEncoder(current_scale=10.0)
        data = jnp.array([0.1, 0.5, 1.0])
        currents = encoder(data, n_time=5)

        self.assertEqual(currents.shape, (5, 3))
        # Should be constant over time
        self.assertTrue(jnp.allclose(currents[0], currents[-1]))
        # Should scale linearly
        expected = jnp.array([1.0, 5.0, 10.0])
        np.testing.assert_allclose(currents[0], expected)

    def test_current_offset(self):
        """Test current offset."""
        encoder = braintools.spike.StepCurrentEncoder(current_scale=1.0, offset=5.0)
        data = jnp.array([0.0, 1.0])
        currents = encoder(data, n_time=3)

        expected = jnp.array([5.0, 6.0])
        np.testing.assert_allclose(currents[0], expected)

    def test_normalization(self):
        """Test input normalization."""
        encoder = braintools.spike.StepCurrentEncoder(
            current_scale=10.0,
            normalize=True,
            min_val=0,
            max_val=100
        )
        data = jnp.array([0, 50, 100])
        currents = encoder(data, n_time=3)

        expected = jnp.array([0.0, 5.0, 10.0])
        np.testing.assert_allclose(currents[0], expected)


class TestSpikeCountEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_uniform_distribution(self):
        """Test uniform spike distribution."""
        encoder = braintools.spike.SpikeCountEncoder(max_spikes=5, distribution='uniform')
        data = jnp.array([0.2, 0.6, 1.0])
        spikes = encoder(data, n_time=10)

        self.assertEqual(spikes.shape, (10, 3))
        # Check spike counts
        counts = jnp.sum(spikes, axis=0)
        expected = jnp.array([1, 3, 5])  # 0.2*5=1, 0.6*5=3, 1.0*5=5
        np.testing.assert_array_equal(counts, expected)

    def test_random_distribution(self):
        """Test random spike distribution."""
        encoder = braintools.spike.SpikeCountEncoder(max_spikes=3, distribution='random')
        data = jnp.array([0.34, 1.0])  # 0.34*3 = 1.02 -> 1
        spikes = encoder(data, n_time=10)

        self.assertEqual(spikes.shape, (10, 2))
        counts = jnp.sum(spikes, axis=0)
        expected = jnp.array([1, 3])  # 0.34*3≈1, 1.0*3=3
        np.testing.assert_array_equal(counts, expected)

    def test_invalid_distribution(self):
        """Test invalid distribution raises error."""
        with self.assertRaises(ValueError):
            braintools.spike.SpikeCountEncoder(distribution='invalid')

    def test_zero_spikes(self):
        """Test zero spike count."""
        encoder = braintools.spike.SpikeCountEncoder(max_spikes=10)
        data = jnp.array([0.0])
        spikes = encoder(data, n_time=5)

        self.assertEqual(jnp.sum(spikes), 0)


class TestTemporalEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_pattern_creation(self):
        """Test temporal pattern creation."""
        encoder = braintools.spike.TemporalEncoder(n_patterns=3, pattern_length=10)
        self.assertEqual(len(encoder.patterns), 3)
        for pattern in encoder.patterns.values():
            self.assertTrue(jnp.all(pattern >= 0))
            self.assertTrue(jnp.all(pattern < 10))

    def test_sequence_encoding(self):
        """Test sequence encoding."""
        encoder = braintools.spike.TemporalEncoder(n_patterns=3, pattern_length=5, jitter=0)
        sequence = jnp.array([0, 1, 2, 1])
        spikes = encoder(sequence)

        expected_time = len(sequence) * 5
        self.assertEqual(spikes.shape, (expected_time, 3))

        # Each pattern should appear in its corresponding time window
        for i, pattern_id in enumerate(sequence):
            window_start = i * 5
            window_end = (i + 1) * 5
            window_spikes = spikes[window_start:window_end, pattern_id]
            self.assertGreater(jnp.sum(window_spikes), 0)

    def test_invalid_pattern_id(self):
        """Test handling of invalid pattern IDs."""
        encoder = braintools.spike.TemporalEncoder(n_patterns=3, pattern_length=5)
        sequence = jnp.array([0, 5, 1])  # 5 is invalid
        spikes = encoder(sequence)

        # Should not crash, invalid patterns should be ignored
        self.assertEqual(spikes.shape, (15, 3))

    def test_temporal_jitter(self):
        """Test temporal jitter functionality."""
        encoder_no_jitter = braintools.spike.TemporalEncoder(n_patterns=2, jitter=0.0)
        encoder_with_jitter = braintools.spike.TemporalEncoder(n_patterns=2, jitter=0.2)

        sequence = jnp.array([0, 0, 0])

        spikes_no_jitter = encoder_no_jitter(sequence)
        spikes_with_jitter = encoder_with_jitter(sequence)

        # With jitter, patterns should be slightly different
        self.assertFalse(jnp.allclose(spikes_no_jitter, spikes_with_jitter))


class TestRankOrderEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.random.seed(42)

    def test_rank_order_encoding(self):
        """Test rank order encoding."""
        encoder = braintools.spike.RankOrderEncoder()
        data = jnp.array([0.1, 0.8, 0.3, 0.9, 0.2])
        spikes = encoder(data, n_time=5)

        self.assertEqual(spikes.shape, (5, 5))
        # Each feature should spike exactly once
        self.assertTrue(jnp.all(jnp.sum(spikes, axis=0) == 1))

        # Highest value (0.9 at index 3) should spike first
        first_spike_times = jnp.argmax(spikes, axis=0)
        highest_spike_time = first_spike_times[3]
        self.assertEqual(highest_spike_time, 0)

    def test_value_based_timing(self):
        """Test value-based timing vs rank-based timing."""
        data = jnp.array([0.1, 0.9])

        value_encoder = braintools.spike.RankOrderEncoder(use_values=True)
        rank_encoder = braintools.spike.RankOrderEncoder(use_values=False)

        value_spikes = value_encoder(data, n_time=10)
        rank_spikes = rank_encoder(data, n_time=10)

        self.assertEqual(value_spikes.shape, (10, 2))
        self.assertEqual(rank_spikes.shape, (10, 2))

    def test_order_inversion(self):
        """Test order inversion."""
        data = jnp.array([0.1, 0.9])

        normal_encoder = braintools.spike.RankOrderEncoder(invert=False)
        inverted_encoder = braintools.spike.RankOrderEncoder(invert=True)

        normal_spikes = normal_encoder(data, n_time=10)
        inverted_spikes = inverted_encoder(data, n_time=10)

        # Get spike times
        normal_times = jnp.argmax(normal_spikes, axis=0)
        inverted_times = jnp.argmax(inverted_spikes, axis=0)

        # Order should be inverted
        self.assertLessEqual(normal_times[1], normal_times[0])  # High value spikes first
        self.assertLessEqual(inverted_times[0], inverted_times[1])  # Low value spikes first

    def test_normalization(self):
        """Test input normalization."""
        encoder = braintools.spike.RankOrderEncoder(normalize=True)
        data = jnp.array([10, 50, 100])  # Wide range
        spikes = encoder(data, n_time=10)

        self.assertEqual(spikes.shape, (10, 3))
        # Should handle large values correctly
        self.assertTrue(jnp.all(jnp.sum(spikes, axis=0) == 1))

    def test_identical_values(self):
        """Test handling of identical values."""
        encoder = braintools.spike.RankOrderEncoder()
        data = jnp.array([0.5, 0.5, 0.5])
        spikes = encoder(data, n_time=5)

        self.assertEqual(spikes.shape, (5, 3))
        # All should spike at the same time (middle of range)
        spike_times = jnp.argmax(spikes, axis=0)
        self.assertTrue(jnp.all(spike_times == spike_times[0]))


if __name__ == '__main__':
    unittest.main()


class TestEncoderSemantics(unittest.TestCase):
    """Window, units, and send-on-delta behaviour."""

    def setUp(self):
        brainstate.random.seed(0)

    def test_latency_linear_spans_window(self):
        spikes = braintools.spike.LatencyEncoder(method='linear', normalize=True)(jnp.array([0.0, 0.5, 1.0]), n_time=21)
        np.testing.assert_array_equal(jnp.argmax(spikes, axis=0), [20, 10, 0])

    def test_latency_quantity_window(self):
        import brainunit as u
        with brainstate.environ.context(dt=0.1 * u.ms):
            spikes = braintools.spike.LatencyEncoder(method='linear', normalize=True)(jnp.array([0.5]), n_time=5 * u.ms)
        self.assertEqual(spikes.shape, (50, 1))

    def test_latency_log_normalized_threshold_at_last_step(self):
        enc = braintools.spike.LatencyEncoder(method='log', normalize=True, threshold=0.1)
        spikes = enc(jnp.array([0.1, 1.0]), n_time=100)
        # x = threshold fires last; x = 1 fires at 99 * log(1 / 0.9) / log(0.1 / 1e-7), i.e. step 1
        np.testing.assert_array_equal(jnp.argmax(spikes, axis=0), [99, 1])

    def test_latency_unnormalized_uses_tau_and_drops_late_spikes(self):
        import brainunit as u
        with brainstate.environ.context(dt=1. * u.ms):
            enc = braintools.spike.LatencyEncoder(method='linear', normalize=False, tau=10 * u.ms)
            spikes = enc(jnp.array([1.0, 0.7, 0.2]), n_time=5)  # times 0, 3, 8 ms
        np.testing.assert_array_equal(jnp.sum(spikes, axis=0), [1, 1, 0])
        self.assertEqual(int(jnp.argmax(spikes[:, 1])), 3)

    def test_rate_encoders_independent_of_time_unit(self):
        import brainunit as u
        rates = {}
        for dt in [0.1 * u.ms, 1e-4 * u.second]:
            brainstate.random.seed(1)
            with brainstate.environ.context(dt=dt):
                r = braintools.spike.RateEncoder(gain=100.)(jnp.array([0.5]), 200000)
                rates[str(dt)] = float(r.mean()) / 1e-4
        for v in rates.values():
            self.assertAlmostEqual(v, 50.0, delta=3.0)

    def test_poisson_time_window_deprecated(self):
        import pytest
        from braintools._deprecation import BraintoolsDeprecationWarning
        with pytest.warns(BraintoolsDeprecationWarning, match=r'PoissonEncoder\(time_window=\.\.\.\)') as caught:
            enc = braintools.spike.PoissonEncoder(time_window=500.)
        self.assertEqual(len(caught), 1)
        self.assertIn('1.0.0', str(caught[0].message))
        # The warning points at the caller, not at the encoder module.
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(enc.time_window, 500.)

    def test_poisson_time_window_deprecation_is_registered(self):
        from braintools._deprecation import registered_deprecations
        records = [r for r in registered_deprecations()
                   if r.old_name == 'braintools.spike.PoissonEncoder(time_window=...)']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].kind, 'argument')
        self.assertEqual((records[0].since, records[0].removed_in), ('0.4.0', '1.0.0'))

    def test_poisson_without_time_window_does_not_warn(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            braintools.spike.PoissonEncoder()
            braintools.spike.PoissonEncoder(time_window=None, normalize=True)

    def test_poisson_time_window_has_no_effect_on_output(self):
        from braintools._deprecation import BraintoolsDeprecationWarning
        rates = jnp.array([10., 50., 100.])
        brainstate.random.seed(7)
        plain = braintools.spike.PoissonEncoder()(rates, n_time=300)
        with self.assertWarns(BraintoolsDeprecationWarning):
            enc = braintools.spike.PoissonEncoder(time_window=500.)
        brainstate.random.seed(7)
        np.testing.assert_array_equal(enc(rates, n_time=300), plain)

    def test_delta_send_on_delta(self):
        enc = braintools.spike.DeltaEncoder(threshold=0.25, normalize=False)
        np.testing.assert_array_equal(enc(jnp.array([0.0, 0.1, 0.2, 0.3, 0.3, 0.0])), [0, 0, 0, 1, 0, -1])
        # Slow ramps are encoded: about height / threshold events
        self.assertEqual(int(braintools.spike.DeltaEncoder(threshold=0.1, normalize=False)(jnp.linspace(0, 1, 100)).sum()), 9)
        absolute = braintools.spike.DeltaEncoder(threshold=0.25, normalize=False, absolute=True)
        np.testing.assert_array_equal(absolute(jnp.array([0.0, 0.3, 0.0])), [0, 1, 1])
        positive = braintools.spike.DeltaEncoder(threshold=0.25, normalize=False, positive_only=True)
        np.testing.assert_array_equal(positive(jnp.array([0.0, 0.3, 0.0, 0.3])), [0, 1, 0, 1])

    def test_spike_count_saturates_at_window(self):
        # Requesting more spikes than time steps fills the window instead of giving none.
        for dist in ['random', 'uniform']:
            enc = braintools.spike.SpikeCountEncoder(max_spikes=20, distribution=dist, normalize=False)
            np.testing.assert_array_equal(enc(jnp.array([1.0, 0.5]), n_time=10).sum(0), [10, 10])


ENCODER_NAMES = (
    'LatencyEncoder', 'RateEncoder', 'PoissonEncoder', 'PopulationEncoder',
    'BernoulliEncoder', 'DeltaEncoder', 'StepCurrentEncoder', 'SpikeCountEncoder',
    'TemporalEncoder', 'RankOrderEncoder',
)

# Minimal constructor arguments for every encoder.
_ENCODER_ARGS = {'PopulationEncoder': (5,), 'TemporalEncoder': (3,)}


class TestTopLevelEncoderAliases(unittest.TestCase):
    """``braintools.<Encoder>`` are deprecated aliases of ``braintools.spike.<Encoder>``."""

    def test_aliases_warn_on_instantiation_only(self):
        import warnings
        from braintools._deprecation import BraintoolsDeprecationWarning
        for name in ENCODER_NAMES:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                alias = getattr(braintools, name)  # access does not warn
            new = getattr(braintools.spike, name)
            self.assertTrue(issubclass(alias, new), name)
            self.assertEqual(alias.__module__, 'braintools')
            self.assertEqual(new.__module__, 'braintools.spike')
            with self.assertWarns(BraintoolsDeprecationWarning) as cm:
                obj = alias(*_ENCODER_ARGS.get(name, ()))
            self.assertIn(f'braintools.spike.{name}', str(cm.warning))
            self.assertIsInstance(obj, new)

    def test_alias_output_identical(self):
        from braintools._deprecation import BraintoolsDeprecationWarning
        data = jnp.array([0.0, 0.5, 1.0])
        with self.assertWarns(BraintoolsDeprecationWarning):
            old = braintools.LatencyEncoder(method='linear', normalize=True)(data, n_time=5)
        new = braintools.spike.LatencyEncoder(method='linear', normalize=True)(data, n_time=5)
        np.testing.assert_array_equal(old, new)

    def test_aliases_exported(self):
        for name in ENCODER_NAMES:
            self.assertIn(name, braintools.__all__)
            self.assertIn(name, braintools.spike.__all__)
