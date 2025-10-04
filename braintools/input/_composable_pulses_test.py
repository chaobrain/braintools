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

"""Tests for composable pulse input generators and their docstring examples."""

from unittest import TestCase

import brainstate
import brainunit as u
import numpy as np

from braintools.input import (
    Spike, GaussianPulse, ExponentialDecay,
    DoubleExponential, Burst,
    # For combining examples
    Constant, Ramp, Step,
    WienerProcess, Sinusoidal
)


class TestSpike(TestCase):
    """Test Spike class and its docstring examples."""

    def test_simple_spike_train(self):
        """Test simple spike train with uniform properties."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            spikes = Spike(
                sp_times=[10, 20, 30, 200, 300] * u.ms,
                duration=400 * u.ms,
                sp_lens=1 * u.ms,  # All spikes 1ms long
                sp_sizes=0.5 * u.nA  # All spikes 0.5nA amplitude
            )
            array = spikes()
            self.assertEqual(array.shape[0], 4000)

            # Check spike at t=10ms
            self.assertAlmostEqual(u.get_magnitude(array[100]), 0.5, places=5)
            # Check baseline between spikes
            self.assertAlmostEqual(u.get_magnitude(array[500]), 0, places=5)

    def test_variable_spike_properties(self):
        """Test spikes with different durations and amplitudes."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            spikes = Spike(
                sp_times=[10, 50, 100] * u.ms,
                duration=150 * u.ms,
                sp_lens=[1, 2, 0.5] * u.ms,  # Different durations
                sp_sizes=[0.5, 1.0, 0.3] * u.nA  # Different amplitudes
            )
            array = spikes()
            self.assertEqual(array.shape[0], 1500)

            # Check different amplitudes
            self.assertAlmostEqual(u.get_magnitude(array[100]), 0.5, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[500]), 1.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1000]), 0.3, places=5)

    def test_add_to_background(self):
        """Test adding spikes to background activity."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            spikes = Spike([10, 50, 100, 150] * u.ms, 200 * u.ms, sp_lens=1. * u.ms, sp_sizes=1.0)
            background = Constant([(0.1, 200 * u.ms)])
            combined = spikes + background

            array = combined()
            self.assertEqual(array.shape[0], 2000)

            # Check background level
            self.assertAlmostEqual(u.get_magnitude(array[250]), 0.1, places=5)
            # Check spike on top of background
            self.assertTrue(u.get_magnitude(array[100]) > 1.0)

    def test_high_frequency_burst(self):
        """Test high-frequency burst simulation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            times = np.arange(0, 50, 2) * u.ms  # Every 2ms
            spikes = Spike(
                sp_times=times,
                duration=100 * u.ms,
                sp_lens=0.5 * u.ms,
                sp_sizes=2.0 * u.pA
            )

            array = spikes()
            self.assertEqual(array.shape[0], 1000)

            # Check that we have 25 spikes (every 2ms from 0 to 48ms)
            expected_spike_times = np.arange(0, 50, 2)
            spike_count = 0
            for t in expected_spike_times:
                idx = int(t * 10)  # Convert ms to index (dt=0.1ms)
                # Check if spike is present (spike duration is 0.5ms = 5 samples)
                if idx < 1000 and u.get_magnitude(array[idx]) > 1.0:
                    spike_count += 1
            self.assertGreaterEqual(spike_count, 20)

    def test_combine_with_noise(self):
        """Test combining spikes with noise."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            spikes = Spike([20, 40, 60] * u.ms, 100 * u.ms, sp_lens=1. * u.ms, )
            noise = WienerProcess(100 * u.ms, sigma=0.05, seed=123)
            noisy_spikes = spikes + noise

            array = noisy_spikes()
            self.assertEqual(array.shape[0], 1000)

            # Should have noise throughout
            baseline_std = np.std(u.get_magnitude(array[700:900]))
            self.assertTrue(baseline_std > 0.01)

    def test_increasing_amplitudes(self):
        """Test pattern of increasing amplitudes."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            amplitudes = np.linspace(0.5, 2.0, 10)
            times = np.linspace(10, 190, 10) * u.ms
            increasing_spikes = Spike(
                sp_times=times,
                duration=200 * u.ms,
                sp_sizes=amplitudes * u.nA,
                sp_lens=1 * u.ms  # 1ms spike duration
            )

            array = increasing_spikes()
            self.assertEqual(array.shape[0], 2000)

            # Check that amplitudes increase
            # First spike at t=10ms (index 100)
            early_spike = u.get_magnitude(array[100])
            # Last spike at t=190ms (index 1900)
            late_spike = u.get_magnitude(array[1900])
            self.assertAlmostEqual(early_spike, 0.5, delta=0.1)
            self.assertAlmostEqual(late_spike, 2.0, delta=0.1)

    def test_paired_pulse_facilitation(self):
        """Test paired-pulse facilitation protocol."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            paired = Spike(
                sp_times=[50, 70] * u.ms,  # 20ms interval
                duration=150 * u.ms,
                sp_lens=2 * u.ms,
                sp_sizes=[1.0, 1.5] * u.nA  # Second spike larger
            )

            array = paired()
            self.assertEqual(array.shape[0], 1500)

            # Check that second spike is larger
            # First spike at t=50ms (index 500), duration 2ms
            first_spike = u.get_magnitude(array[500])
            # Second spike at t=70ms (index 700), duration 2ms
            second_spike = u.get_magnitude(array[700])
            self.assertAlmostEqual(first_spike, 1.0, places=5)
            self.assertAlmostEqual(second_spike, 1.5, places=5)


class TestGaussianPulse(TestCase):
    """Test GaussianPulse class and its docstring examples."""

    def test_single_gaussian_pulse(self):
        """Test single Gaussian pulse."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulse = GaussianPulse(
                amplitude=1.0 * u.nA,
                center=100 * u.ms,
                sigma=20 * u.ms,
                duration=200 * u.ms
            )
            array = pulse()
            self.assertEqual(array.shape[0], 2000)

            # Peak should be near center
            peak_idx = np.argmax(u.get_magnitude(array))
            self.assertTrue(900 < peak_idx < 1100)  # Near 100ms

            # Peak value should be close to amplitude
            self.assertAlmostEqual(u.get_magnitude(array[peak_idx]), 1.0, delta=0.1)

    def test_multiple_overlapping_pulses(self):
        """Test multiple overlapping pulses."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulse1 = GaussianPulse(1.0, 100 * u.ms, 20 * u.ms, 500 * u.ms)
            pulse2 = GaussianPulse(0.8, 300 * u.ms, 30 * u.ms, 500 * u.ms)
            double_pulse = pulse1 + pulse2

            array = double_pulse()
            self.assertEqual(array.shape[0], 5000)

            # Should have two peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(u.get_magnitude(array), height=0.5)
            self.assertGreaterEqual(len(peaks), 2)

    def test_amplitude_modulation(self):
        """Test amplitude modulation with Gaussian envelope."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            envelope = GaussianPulse(1.0, 250 * u.ms, 50 * u.ms, 500 * u.ms)
            carrier = Sinusoidal(1.0, 50 * u.Hz, 500 * u.ms)
            modulated = envelope * carrier

            array = modulated()
            self.assertEqual(array.shape[0], 5000)

            # Envelope should modulate amplitude
            early = np.max(np.abs(u.get_magnitude(array[:500])))
            middle = np.max(np.abs(u.get_magnitude(array[2000:3000])))
            late = np.max(np.abs(u.get_magnitude(array[4500:])))

            self.assertTrue(middle > early)
            self.assertTrue(middle > late)

    def test_noisy_gaussian_pulse(self):
        """Test noisy Gaussian pulse."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulse = GaussianPulse(2.0, 100 * u.ms, 15 * u.ms, 200 * u.ms)
            noise = WienerProcess(200 * u.ms, sigma=0.1, seed=42)
            noisy_pulse = pulse + noise

            array = noisy_pulse()
            self.assertEqual(array.shape[0], 2000)

            # Should have noise added
            clean = pulse()
            diff = u.get_magnitude(array - clean)
            print(np.std(diff))
            self.assertTrue(np.std(diff) > 0.01)

    def test_wide_narrow_comparison(self):
        """Test wide and narrow pulses comparison."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            narrow = GaussianPulse(1.0, 100 * u.ms, 5 * u.ms, 200 * u.ms)
            wide = GaussianPulse(1.0, 100 * u.ms, 30 * u.ms, 200 * u.ms)

            narrow_array = narrow()
            wide_array = wide()

            # Narrow should have sharper peak
            narrow_width = np.sum(u.get_magnitude(narrow_array) > 0.5)
            wide_width = np.sum(u.get_magnitude(wide_array) > 0.5)

            self.assertTrue(wide_width > narrow_width)

    def test_inverted_pulse(self):
        """Test inverted (inhibitory) pulse."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            excitatory = GaussianPulse(0.5, 150 * u.ms, 25 * u.ms, 300 * u.ms)
            inhibitory = -excitatory

            exc_array = excitatory()
            inh_array = inhibitory()

            # Should be negated
            np.testing.assert_allclose(
                u.get_magnitude(inh_array),
                -u.get_magnitude(exc_array),
                rtol=1e-5
            )


class TestExponentialDecay(TestCase):
    """Test ExponentialDecay class and its docstring examples."""

    def test_simple_decay(self):
        """Test simple exponential decay."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            decay = ExponentialDecay(
                amplitude=2.0 * u.nA,
                tau=30 * u.ms,
                duration=200 * u.ms
            )
            array = decay()
            self.assertEqual(array.shape[0], 2000)

            # Should start at amplitude
            self.assertAlmostEqual(u.get_magnitude(array[0]), 2.0, delta=0.1)

            # Should decay over time
            self.assertTrue(u.get_magnitude(array[300]) < u.get_magnitude(array[0]))

            # After tau, should be ~37% of initial
            tau_idx = int(30 / 0.1)
            expected = 2.0 / np.e
            self.assertAlmostEqual(u.get_magnitude(array[tau_idx]), expected, delta=0.2)

    def test_delayed_decay(self):
        """Test delayed decay starting at t=50ms."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            decay = ExponentialDecay(
                amplitude=1.5,
                tau=20 * u.ms,
                duration=150 * u.ms,
                t_start=50 * u.ms
            )

            array = decay()
            self.assertEqual(array.shape[0], 1500)

            # Before t_start should be zero
            self.assertAlmostEqual(u.get_magnitude(array[250]), 0, places=5)

            # After t_start should decay
            self.assertTrue(u.get_magnitude(array[500]) > 0)
            self.assertTrue(u.get_magnitude(array[1000]) < u.get_magnitude(array[600]))

    def test_gated_decay(self):
        """Test gated decay with step function."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            decay = ExponentialDecay(2.0, 30 * u.ms, 500 * u.ms, t_start=100 * u.ms)
            step = Step([0, 1], [0, 100] * u.ms, 500 * u.ms)
            gated_decay = decay * step

            array = gated_decay()
            self.assertEqual(array.shape[0], 5000)

            # Before step, should be zero
            self.assertAlmostEqual(u.get_magnitude(array[500]), 0, places=5)

            # After step and decay start, should have signal
            self.assertTrue(u.get_magnitude(array[1500]) > 0)

    def test_multiple_decay_phases(self):
        """Test multiple decay phases (bi-exponential)."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            fast_decay = ExponentialDecay(1.0, 10 * u.ms, 200 * u.ms)
            slow_decay = ExponentialDecay(0.5, 50 * u.ms, 200 * u.ms)
            combined = fast_decay + slow_decay

            array = combined()
            self.assertEqual(array.shape[0], 2000)

            # Initial value should be sum
            self.assertAlmostEqual(u.get_magnitude(array[0]), 1.5, delta=0.1)

            # Should decay with two components
            self.assertTrue(u.get_magnitude(array[500]) > 0)

    def test_adaptation_current(self):
        """Test adaptation current simulation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            trigger = Step([0, 1, 0], [0, 50, 150] * u.ms, 300 * u.ms)
            adaptation = ExponentialDecay(0.3, 40 * u.ms, 300 * u.ms, t_start=50 * u.ms)
            net_current = trigger - adaptation

            array = net_current()
            self.assertEqual(array.shape[0], 3000)

            # Initially should be 1 (trigger) - 0.3 (adaptation start)
            self.assertTrue(u.get_magnitude(array[500]) < 1.0)


class TestDoubleExponential(TestCase):
    """Test DoubleExponential class and its docstring examples."""

    def test_ampa_like_current(self):
        """Test AMPA-like synaptic current."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ampa = DoubleExponential(
                amplitude=1.0 * u.nA,
                tau_rise=0.5 * u.ms,
                tau_decay=5 * u.ms,
                duration=50 * u.ms
            )
            array = ampa()
            self.assertEqual(array.shape[0], 500)

            # Should have a peak
            peak_idx = np.argmax(u.get_magnitude(array))
            peak_value = u.get_magnitude(array[peak_idx])

            # Peak should be close to amplitude
            self.assertAlmostEqual(peak_value, 1.0, delta=0.1)

            # Should rise then decay
            self.assertTrue(u.get_magnitude(array[peak_idx + 50]) < peak_value)

    def test_synaptic_with_noise(self):
        """Test synaptic current with noise."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            alpha = DoubleExponential(1.0, 5 * u.ms, 20 * u.ms, 200 * u.ms)
            noise = WienerProcess(200 * u.ms, sigma=0.05, seed=123)
            synaptic = alpha + noise

            array = synaptic()
            self.assertEqual(array.shape[0], 2000)

            # Should have noise added
            clean = alpha()
            diff = u.get_magnitude(array - clean)
            self.assertTrue(np.std(diff) > 0.01)

    def test_paired_pulse_facilitation(self):
        """Test paired-pulse facilitation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulse1 = DoubleExponential(1.0, 2 * u.ms, 15 * u.ms, 100 * u.ms, t_start=20 * u.ms)
            pulse2 = DoubleExponential(1.5, 2 * u.ms, 15 * u.ms, 100 * u.ms, t_start=40 * u.ms)
            ppf = pulse1 + pulse2

            array = ppf()
            self.assertEqual(array.shape[0], 1000)

            # Should have two peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(u.get_magnitude(array), height=0.5)
            self.assertGreaterEqual(len(peaks), 2)

            # Second peak should be larger
            if len(peaks) >= 2:
                self.assertTrue(u.get_magnitude(array[peaks[1]]) > u.get_magnitude(array[peaks[0]]))

    def test_inhibitory_current(self):
        """Test inhibitory synaptic current."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ipsc = DoubleExponential(
                amplitude=-0.8 * u.nA,  # Negative for inhibition
                tau_rise=1 * u.ms,
                tau_decay=20 * u.ms,
                duration=100 * u.ms
            )

            array = ipsc()
            self.assertEqual(array.shape[0], 1000)

            # Should be negative
            min_idx = np.argmin(u.get_magnitude(array))
            self.assertTrue(u.get_magnitude(array[min_idx]) < -0.7)


class TestBurst(TestCase):
    """Test Burst class and its docstring examples."""

    def test_oscillatory_bursts_50hz(self):
        """Test oscillatory bursts at 50Hz."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            bursts = Burst(
                n_bursts=5,
                burst_amp=1.0 * u.nA,
                burst_freq=50 * u.Hz,  # 50Hz oscillation
                burst_duration=30 * u.ms,
                inter_burst_interval=100 * u.ms,
                duration=500 * u.ms
            )
            array = bursts()
            self.assertEqual(array.shape[0], 5000)

            # Check that we have oscillations during bursts
            # First burst at t=0-30ms (indices 0-300)
            burst_segment = u.get_magnitude(array[:300])
            # Should have oscillations - check for sign changes
            sign_changes = np.sum(np.diff(np.sign(burst_segment)) != 0)
            self.assertTrue(sign_changes > 0)  # Should have oscillations

            # Inter-burst at t=50ms should be quiet
            quiet_period = u.get_magnitude(array[500:900])
            self.assertTrue(np.all(np.abs(quiet_period) < 0.1))

    def test_oscillatory_bursts(self):
        """Test oscillatory bursts (theta-burst stimulation)."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            theta_bursts = Burst(
                n_bursts=10,
                burst_amp=2.0,
                burst_freq=100 * u.Hz,  # 100Hz oscillation within bursts
                burst_duration=40 * u.ms,
                inter_burst_interval=200 * u.ms,  # 5Hz burst rate
                duration=2000 * u.ms
            )

            array = theta_bursts()
            self.assertEqual(array.shape[0], 20000)

            # Should have oscillations during bursts
            # Check first burst (0-40ms = indices 0-400)
            burst_segment = u.get_magnitude(array[50:350])
            # Count zero crossings as proxy for oscillation
            zero_crossings = np.sum(np.diff(np.sign(burst_segment)) != 0)
            self.assertTrue(zero_crossings > 5)  # Should have multiple oscillations

    def test_bursts_with_ramped_amplitude(self):
        """Test bursts with ramped amplitude."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            burst = Burst(5, 1.0, 50 * u.Hz, 30 * u.ms, 100 * u.ms, 500 * u.ms)
            ramp = Ramp(0.5, 1.5, 500 * u.ms)
            modulated_burst = burst * ramp

            array = modulated_burst()
            self.assertEqual(array.shape[0], 5000)

            # The modulation should increase the amplitude over time
            # Check RMS values in early vs late windows
            early_rms = np.sqrt(np.mean(u.get_magnitude(array[:1000]) ** 2))
            late_rms = np.sqrt(np.mean(u.get_magnitude(array[4000:]) ** 2))
            # Late RMS should be larger due to ramp modulation
            self.assertTrue(late_rms > early_rms * 1.5)  # Should be roughly 3x larger

    def test_burst_with_noise(self):
        """Test burst pattern with noise."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            bursts = Burst(4, 1.5, 30 * u.Hz, 50 * u.ms, 150 * u.ms, 600 * u.ms)
            noise = WienerProcess(600 * u.ms, sigma=0.1, seed=456)
            noisy_bursts = bursts + noise

            array = noisy_bursts()
            self.assertEqual(array.shape[0], 6000)

            # Should have noise throughout
            quiet_period = u.get_magnitude(array[700:900])  # Between bursts
            print(np.std(quiet_period))
            self.assertTrue(np.std(quiet_period) > 0.01)

    def test_gamma_in_theta(self):
        """Test gamma bursts in theta rhythm."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            gamma_in_theta = Burst(
                n_bursts=15,
                burst_amp=1.0 * u.nA,
                burst_freq=40 * u.Hz,  # Gamma frequency
                burst_duration=100 * u.ms,
                inter_burst_interval=200 * u.ms,  # Theta rhythm (5Hz)
                duration=3000 * u.ms
            )

            array = gamma_in_theta()
            self.assertEqual(array.shape[0], 30000)

            # Should have gamma oscillations within bursts
            # Check a burst period
            burst_start = 0
            burst_end = 1000  # 100ms
            burst_data = u.get_magnitude(array[burst_start:burst_end])

            # Count oscillations
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(burst_data, height=0.5)
            # Should have roughly 4 peaks in 100ms at 40Hz
            self.assertTrue(2 <= len(peaks) <= 6)
