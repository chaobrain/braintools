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

"""Tests for composable stochastic input classes."""

from unittest import TestCase

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braintools.input import GaussianPulse
from braintools.input import Ramp, Step
from braintools.input import Sinusoidal
from braintools.input import WienerProcess, OUProcess, Poisson

block = False


def show(current, duration, title=''):
    if plt is not None:
        dt_value = u.get_magnitude(brainstate.environ.get_dt())
        duration_value = u.get_magnitude(duration) if hasattr(duration, 'unit') else duration
        ts = np.arange(0, duration_value, dt_value)
        current_value = u.get_magnitude(current) if hasattr(current, 'unit') else current
        if current_value.ndim == 1:
            plt.plot(ts, current_value)
        else:
            # For multi-dimensional inputs, plot each channel
            for i in range(current_value.shape[1]):
                plt.plot(ts, current_value[:, i], label=f'Channel {i + 1}')
            plt.legend()
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestWienerProcess(TestCase):
    """Test WienerProcess class."""

    def test_simple_wiener(self):
        """Test simple Wiener process from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            noise = WienerProcess(duration=100 * u.ms, sigma=0.5)
            signal = noise()
            self.assertEqual(signal.shape[0], 1000)
            show(signal, 100 * u.ms, 'Simple Wiener Process')

    def test_multiple_processes(self):
        """Test multiple independent processes from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            multi_noise = WienerProcess(
                duration=200 * u.ms,
                n=5,  # 5 independent processes
                sigma=1.0
            )
            signal = multi_noise()
            self.assertEqual(signal.shape, (2000, 5))
            show(signal, 200 * u.ms, 'Multiple Wiener Processes')

    def test_windowed_noise(self):
        """Test windowed noise from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            windowed = WienerProcess(
                duration=500 * u.ms,
                sigma=2.0,
                t_start=100 * u.ms,
                t_end=400 * u.ms
            )
            signal = windowed()
            self.assertEqual(signal.shape[0], 5000)

            # Check windowing - should be zero outside window
            signal_mag = u.get_magnitude(signal)
            # Before t_start (first 1000 samples)
            self.assertTrue(np.all(signal_mag[:1000] == 0))
            # After t_end (last 1000 samples)
            self.assertTrue(np.all(signal_mag[4000:] == 0))
            show(signal, 500 * u.ms, 'Windowed Wiener Process')

    def test_noisy_drift(self):
        """Test combining Wiener process with drift from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            drift = Ramp(0, 0.5, 500 * u.ms)
            noise = WienerProcess(500 * u.ms, sigma=0.1)
            drifting_noise = noise + drift
            signal = drifting_noise()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'Wiener Process with Linear Drift')

    def test_modulated_noise(self):
        """Test modulated noise from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            noise = WienerProcess(500 * u.ms, sigma=0.2)
            envelope = Step([0, 1.0], [0 * u.ms, 100 * u.ms], 500 * u.ms)
            modulated = noise * envelope
            signal = modulated()
            self.assertEqual(signal.shape[0], 5000)

            # Check modulation - should be zero before step
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:1000] == 0))
            show(signal, 500 * u.ms, 'Step-Modulated Wiener Process')

    def test_reproducible_noise(self):
        """Test reproducible noise with seed from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            fixed_noise1 = WienerProcess(
                duration=100 * u.ms,
                sigma=0.3,
                seed=42  # Fixed seed for reproducibility
            )
            signal1 = fixed_noise1()

            fixed_noise2 = WienerProcess(
                duration=100 * u.ms,
                sigma=0.3,
                seed=42  # Same seed
            )
            signal2 = fixed_noise2()

            # Should be identical
            assert u.math.allclose(signal1, signal2)
            self.assertEqual(signal1.shape[0], 1000)

    def test_different_parameters(self):
        """Test Wiener process with different parameters."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # High noise
            high_noise = WienerProcess(duration=300 * u.ms, sigma=5.0, n=3)
            signal = high_noise()
            self.assertEqual(signal.shape, (3000, 3))

            # Low noise
            low_noise = WienerProcess(duration=300 * u.ms, sigma=0.01, n=1)
            signal = low_noise()
            self.assertEqual(signal.shape[0], 3000)


class TestOUProcess(TestCase):
    """Test OUProcess class."""

    def test_simple_ou(self):
        """Test simple OU process from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ou = OUProcess(
                mean=0.5,
                sigma=0.2,
                tau=10 * u.ms,
                duration=500 * u.ms
            )
            signal = ou()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'Simple OU Process')

    def test_fast_fluctuations(self):
        """Test fast fluctuations around zero from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            fast_ou = OUProcess(
                mean=0.0,
                sigma=0.5,
                tau=2 * u.ms,  # Fast time constant
                duration=200 * u.ms
            )
            signal = fast_ou()
            self.assertEqual(signal.shape[0], 2000)
            show(signal, 200 * u.ms, 'Fast OU Process')

    def test_slow_fluctuations(self):
        """Test slow fluctuations with drift from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            slow_ou = OUProcess(
                mean=1.0,
                sigma=0.3,
                tau=50 * u.ms,  # Slow time constant
                duration=1000 * u.ms
            )
            signal = slow_ou()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Slow OU Process with Drift')

    def test_multiple_ou_processes(self):
        """Test multiple independent processes from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            multi_ou = OUProcess(
                mean=0.0,
                sigma=0.2,
                tau=5 * u.ms,
                duration=300 * u.ms,
                n=10  # 10 independent processes
            )
            signal = multi_ou()
            self.assertEqual(signal.shape, (3000, 10))
            show(signal, 300 * u.ms, 'Multiple OU Processes')

    def test_windowed_ou(self):
        """Test windowed OU process from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            windowed_ou = OUProcess(
                mean=0.5,
                sigma=0.1,
                tau=20 * u.ms,
                duration=500 * u.ms,
                t_start=100 * u.ms,
                t_end=400 * u.ms
            )
            signal = windowed_ou()
            self.assertEqual(signal.shape[0], 5000)

            # Check windowing
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:1000] == 0))
            self.assertTrue(np.all(signal_mag[4000:] == 0))
            show(signal, 500 * u.ms, 'Windowed OU Process')

    def test_ou_with_time_varying_mean(self):
        """Test OU process with time-varying mean from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ou = OUProcess(mean=0.5, sigma=0.1, tau=20 * u.ms, duration=500 * u.ms)
            sine_mean = Sinusoidal(0.3, 2 * u.Hz, 500 * u.ms)
            modulated_ou = ou + sine_mean
            signal = modulated_ou()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'OU Process with Sinusoidal Mean')

    def test_gated_ou(self):
        """Test gated OU process from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ou = OUProcess(mean=0.5, sigma=0.1, tau=20 * u.ms, duration=500 * u.ms)
            gate = Step([0, 1.0], [0 * u.ms, 50 * u.ms], 500 * u.ms)
            gated_ou = ou * gate
            signal = gated_ou()
            self.assertEqual(signal.shape[0], 5000)

            # Check gating
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:500] == 0))
            show(signal, 500 * u.ms, 'Gated OU Process')

    def test_reproducible_ou(self):
        """Test reproducible OU process from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            fixed_ou1 = OUProcess(
                mean=0.0,
                sigma=0.15,
                tau=15 * u.ms,
                duration=200 * u.ms,
                seed=123  # Fixed seed
            )
            signal1 = fixed_ou1()

            fixed_ou2 = OUProcess(
                mean=0.0,
                sigma=0.15,
                tau=15 * u.ms,
                duration=200 * u.ms,
                seed=123  # Same seed
            )
            signal2 = fixed_ou2()

            assert u.math.allclose(signal1, signal2)
            self.assertEqual(signal1.shape[0], 2000)

    def test_different_tau_values(self):
        """Test OU process with different tau values."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Very fast (approaches white noise)
            fast = OUProcess(mean=0, sigma=0.5, tau=0.5 * u.ms, duration=200 * u.ms)
            fast_signal = fast()

            # Very slow (approaches Wiener process)
            slow = OUProcess(mean=0, sigma=0.5, tau=100 * u.ms, duration=200 * u.ms)
            slow_signal = slow()

            self.assertEqual(fast_signal.shape[0], 2000)
            self.assertEqual(slow_signal.shape[0], 2000)


class TestPoisson(TestCase):
    """Test Poisson class."""

    def test_simple_poisson(self):
        """Test simple Poisson spike train from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            spikes = Poisson(
                rate=10 * u.Hz,
                duration=1000 * u.ms
            )
            signal = spikes()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Simple Poisson Spike Train')

    def test_high_frequency_background(self):
        """Test high-frequency background activity from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            background = Poisson(
                rate=100 * u.Hz,
                duration=500 * u.ms,
                amplitude=0.5  # Smaller amplitude
            )
            signal = background()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'High-Frequency Poisson Background')

    def test_multiple_spike_trains(self):
        """Test multiple independent spike trains from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            multi_spikes = Poisson(
                rate=20 * u.Hz,
                duration=2000 * u.ms,
                n=50,  # 50 independent spike trains
                amplitude=2.0
            )
            signal = multi_spikes()
            self.assertEqual(signal.shape, (20000, 50))
            # Show only first 5 channels for clarity
            show(signal[:, :5], 2000 * u.ms, 'Multiple Poisson Spike Trains (first 5)')

    def test_windowed_spiking(self):
        """Test windowed spiking activity from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            burst = Poisson(
                rate=50 * u.Hz,
                duration=1000 * u.ms,
                t_start=200 * u.ms,
                t_end=800 * u.ms,
                amplitude=1.0
            )
            signal = burst()
            self.assertEqual(signal.shape[0], 10000)

            # Check windowing
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:2000] == 0))
            self.assertTrue(np.all(signal_mag[8000:] == 0))
            show(signal, 1000 * u.ms, 'Windowed Poisson Burst')

    def test_low_rate_spontaneous(self):
        """Test low rate spontaneous activity from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            spontaneous = Poisson(
                rate=1 * u.Hz,
                duration=10000 * u.ms,
                amplitude=5.0
            )
            signal = spontaneous()
            self.assertEqual(signal.shape[0], 100000)

            # Check that there are some spikes but not too many
            n_spikes = np.sum(u.get_magnitude(signal) > 0)
            # Expected around 10 spikes for 1 Hz over 10 seconds
            self.assertTrue(0 < n_spikes < 50)  # Allow for randomness
            show(signal[:10000], 1000 * u.ms, 'Low Rate Spontaneous Activity (first 1s)')

    def test_gaussian_modulated_poisson(self):
        """Test Poisson spikes with Gaussian envelope from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            poisson = Poisson(50 * u.Hz, 1000 * u.ms)
            envelope = GaussianPulse(1.0, 500 * u.ms, 100 * u.ms, 1000 * u.ms)
            modulated = poisson * envelope
            signal = modulated()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Gaussian-Modulated Poisson Spikes')

    def test_rhythmic_modulation(self):
        """Test rhythmic modulation of spike rate from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            poisson = Poisson(50 * u.Hz, 1000 * u.ms)
            rhythm = Sinusoidal(0.5, 5 * u.Hz, 1000 * u.ms)
            rhythmic_spikes = poisson * (1 + rhythm)
            signal = rhythmic_spikes()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Rhythmically Modulated Poisson Spikes')

    def test_reproducible_spikes(self):
        """Test reproducible spike pattern from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            fixed_spikes1 = Poisson(
                rate=30 * u.Hz,
                duration=500 * u.ms,
                seed=456  # Fixed seed for reproducibility
            )
            signal1 = fixed_spikes1()

            fixed_spikes2 = Poisson(
                rate=30 * u.Hz,
                duration=500 * u.ms,
                seed=456  # Same seed
            )
            signal2 = fixed_spikes2()

            assert u.math.allclose(signal1, signal2)
            self.assertEqual(signal1.shape[0], 5000)

    def test_increasing_rate(self):
        """Test inhomogeneous Poisson process with increasing rate from docstring example."""
        from braintools.input._composable_basic import Ramp

        with brainstate.environ.context(dt=0.1 * u.ms):
            # Base Poisson process
            base_poisson = Poisson(10 * u.Hz, 1000 * u.ms)
            # Increasing rate envelope
            ramp = Ramp(0.1, 1.0, 1000 * u.ms)
            increasing_rate = base_poisson * ramp
            signal = increasing_rate()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Poisson with Increasing Rate')

    def test_different_rates(self):
        """Test Poisson input with different rates."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Very low rate
            low_rate = Poisson(rate=0.5 * u.Hz, duration=1000 * u.ms, amplitude=10.0)
            low_signal = low_rate()

            # Very high rate
            high_rate = Poisson(rate=200 * u.Hz, duration=1000 * u.ms, amplitude=0.1)
            high_signal = high_rate()

            self.assertEqual(low_signal.shape[0], 10000)
            self.assertEqual(high_signal.shape[0], 10000)

            # Check spike counts
            low_spikes = np.sum(u.get_magnitude(low_signal) > 0)
            high_spikes = np.sum(u.get_magnitude(high_signal) > 0)

            # High rate should have significantly more spikes
            self.assertTrue(high_spikes > low_spikes * 10)


class TestCombinedStochastic(TestCase):
    """Test combinations of stochastic processes."""

    def test_ou_plus_wiener(self):
        """Test combining OU and Wiener processes."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ou = OUProcess(mean=0.5, sigma=0.1, tau=10 * u.ms, duration=400 * u.ms)
            wiener = WienerProcess(duration=400 * u.ms, sigma=0.2)
            combined = ou + wiener
            signal = combined()
            self.assertEqual(signal.shape[0], 4000)
            show(signal, 400 * u.ms, 'Combined OU + Wiener Process')

    def test_poisson_plus_ou(self):
        """Test combining Poisson spikes with OU background."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            poisson = Poisson(rate=20 * u.Hz, duration=500 * u.ms, amplitude=2.0)
            ou = OUProcess(mean=0.2, sigma=0.05, tau=15 * u.ms, duration=500 * u.ms)
            combined = poisson + ou
            signal = combined()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'Poisson Spikes + OU Background')

    def test_complex_combination(self):
        """Test complex combination of multiple stochastic processes."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Base OU process
            ou = OUProcess(mean=0.3, sigma=0.1, tau=20 * u.ms, duration=1000 * u.ms)

            # Add Wiener noise
            wiener = WienerProcess(duration=1000 * u.ms, sigma=0.05)

            # Add Poisson spikes
            poisson = Poisson(rate=5 * u.Hz, duration=1000 * u.ms, amplitude=1.0)

            # Combine all
            combined = ou + wiener + poisson
            signal = combined()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Complex Stochastic Combination')

    def test_modulated_stochastic(self):
        """Test modulating stochastic processes with deterministic signals."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create stochastic base
            ou = OUProcess(mean=0.0, sigma=0.2, tau=10 * u.ms, duration=800 * u.ms)

            # Create modulation envelope
            gaussian = GaussianPulse(1.0, 400 * u.ms, 100 * u.ms, 800 * u.ms)

            # Apply modulation
            modulated = ou * gaussian
            signal = modulated()
            self.assertEqual(signal.shape[0], 8000)
            show(signal, 800 * u.ms, 'Gaussian-Modulated OU Process')


class TestStochasticStatistics(TestCase):
    """Test statistical properties of stochastic processes."""

    def setUp(self):
        np.random.seed(420)

    def test_wiener_variance_scaling(self):
        """Test that Wiener process variance scales with dt."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            sigma = 1.0
            n_samples = 100

            wiener = WienerProcess(duration=duration, sigma=sigma, n=n_samples)
            signals = wiener()

            # Check variance scaling (approximately)
            dt = u.get_magnitude(brainstate.environ.get_dt())
            expected_var = sigma ** 2 * dt

            # Calculate increments
            if signals.ndim == 1:
                increments = np.diff(u.get_magnitude(signals))
            else:
                increments = np.diff(u.get_magnitude(signals), axis=0)

            actual_var = np.var(increments)
            print(actual_var)
            # Allow 50% tolerance due to randomness
            self.assertTrue(0.5 * expected_var < actual_var < 2.0 * expected_var)

    def test_ou_mean_reversion(self):
        """Test that OU process reverts to mean."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            mean = 0.5
            ou = OUProcess(mean=mean, sigma=0.1, tau=10 * u.ms, duration=5000 * u.ms, n=10)
            signals = ou()

            # Check that average converges to mean (approximately)
            signal_mean = np.mean(u.get_magnitude(signals))
            # Allow 20% tolerance
            self.assertTrue(0.8 * mean < signal_mean < 1.2 * mean)

    def test_poisson_rate(self):
        """Test that Poisson process matches expected rate."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            rate = 50  # Hz
            duration = 10000 * u.ms

            poisson = Poisson(rate=rate * u.Hz, duration=duration, n=10)
            signals = poisson()

            # Count spikes
            n_spikes = np.sum(u.get_magnitude(signals) > 0)
            expected_spikes = rate * 10 * 10  # rate * duration_seconds * n_channels

            # Allow 20% tolerance due to randomness
            self.assertTrue(0.8 * expected_spikes < n_spikes < 1.2 * expected_spikes)
