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

"""Tests for pulse input functions."""

from unittest import TestCase

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braintools.input import (spike, gaussian_pulse, exponential_decay, double_exponential, burst)

block = False


def show(current, duration, title=''):
    if plt is not None:
        dt_value = u.get_magnitude(brainstate.environ.get_dt())
        duration_value = u.get_magnitude(duration) if hasattr(duration, 'unit') else duration
        ts = np.arange(0, duration_value, dt_value)
        current_value = u.get_magnitude(current) if hasattr(current, 'unit') else current
        plt.plot(ts, current_value)
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestPulseInputs(TestCase):
    def test_spike(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test with time values (backward compatibility)
            current = spike(
                sp_times=[10, 20, 30, 200, 300] * u.ms,
                sp_lens=1. * u.ms,
                sp_sizes=0.5 * u.nA,
                duration=400. * u.ms
            )
            show(current, 400 * u.ms, 'Spike Input Example')
            self.assertEqual(current.shape[0], 4000)

    def test_spike_variable_params(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test with variable spike lengths and sizes
            current = spike(
                sp_times=[50, 150, 250] * u.ms,
                sp_lens=[1., 2., 3.] * u.ms,  # Different spike lengths
                sp_sizes=[0.5, 1.0, 0.3] * u.nA,  # Different spike amplitudes
                duration=350. * u.ms
            )
            show(current, 350 * u.ms, 'Spike Input with Variable Parameters')
            self.assertEqual(current.shape[0], 3500)

    def test_spike_from_docstring(self):
        """Test examples from spike docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple spike train with uniform properties
            current = spike(
                sp_times=[10, 20, 30, 200, 300] * u.ms,
                sp_lens=1 * u.ms,  # All spikes 1ms long
                sp_sizes=0.5 * u.nA,  # All spikes 0.5nA amplitude
                duration=400 * u.ms
            )
            self.assertEqual(current.shape[0], 4000)
            self.assertEqual(u.get_unit(current), u.nA)

            # Variable spike properties
            current = spike(
                sp_times=np.array([10, 50, 100]) * u.ms,
                sp_lens=np.array([1, 2, 0.5]) * u.ms,  # Different durations
                sp_sizes=np.array([0.5, 1.0, 0.3]) * u.nA,  # Different amplitudes
                duration=150 * u.ms
            )
            self.assertEqual(current.shape[0], 1500)

            # High-frequency burst
            times = np.arange(0, 50, 2) * u.ms  # Every 2ms
            current = spike(
                sp_times=times,
                sp_lens=0.5 * u.ms,
                sp_sizes=1.0 * u.pA,
                duration=100 * u.ms
            )
            self.assertEqual(current.shape[0], 1000)
            self.assertEqual(u.get_unit(current), u.pA)

    def test_gaussian_pulse(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 500 * u.ms
            current = gaussian_pulse(amplitude=1.0 * u.nA,
                                     center=250 * u.ms,
                                     sigma=30 * u.ms,
                                     duration=duration)
            show(current, duration, 'Gaussian Pulse')
            self.assertEqual(current.shape[0], 5000)

    def test_gaussian_pulse_from_docstring(self):
        """Test examples from gaussian_pulse docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Single Gaussian pulse
            current = gaussian_pulse(
                amplitude=10 * u.pA,
                center=50 * u.ms,
                sigma=10 * u.ms,
                duration=100 * u.ms
            )
            self.assertEqual(current.shape[0], 1000)
            self.assertEqual(u.get_unit(current), u.pA)

            # Multiple identical pulses
            currents = gaussian_pulse(
                amplitude=5 * u.nA,
                center=25 * u.ms,
                sigma=5 * u.ms,
                duration=50 * u.ms,
                n=10  # Generate 10 identical pulses
            )
            self.assertEqual(currents.shape, (500, 10))

            # Narrow pulse (approximating delta function)
            current = gaussian_pulse(
                amplitude=100 * u.pA,
                center=10 * u.ms,
                sigma=0.5 * u.ms,
                duration=20 * u.ms
            )
            self.assertEqual(current.shape[0], 200)

            # Wide pulse (slow activation)
            current = gaussian_pulse(
                amplitude=2 * u.nA,
                center=100 * u.ms,
                sigma=30 * u.ms,
                duration=200 * u.ms
            )
            self.assertEqual(current.shape[0], 2000)

    def test_gaussian_pulse_multiple(self):
        # Test multiple gaussian pulses
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            current = gaussian_pulse(amplitude=0.8 * u.nA,
                                     center=200 * u.ms,
                                     sigma=20 * u.ms,
                                     duration=duration)
            current += gaussian_pulse(amplitude=1.2 * u.nA,
                                      center=600 * u.ms,
                                      sigma=40 * u.ms,
                                      duration=duration)
            show(current, duration, 'Multiple Gaussian Pulses')
            self.assertEqual(current.shape[0], 10000)

    def test_exponential_decay(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 500 * u.ms
            current = exponential_decay(amplitude=2.0 * u.nA,
                                        tau=50 * u.ms,
                                        t_start=50 * u.ms,
                                        t_end=450 * u.ms,
                                        duration=duration)
            show(current, duration, 'Exponential Decay')
            self.assertEqual(current.shape[0], 5000)

    def test_exponential_decay_from_docstring(self):
        """Test examples from exponential_decay docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple exponential decay
            current = exponential_decay(
                amplitude=10 * u.pA,
                tau=20 * u.ms,
                duration=100 * u.ms
            )
            self.assertEqual(current.shape[0], 1000)
            self.assertEqual(u.get_unit(current), u.pA)
            # Check decay behavior
            self.assertTrue(u.get_magnitude(current[0]) > u.get_magnitude(current[200]))

            # Fast decay (mimicking AMPA receptor)
            current = exponential_decay(
                amplitude=1 * u.nA,
                tau=2 * u.ms,
                duration=20 * u.ms
            )
            self.assertEqual(current.shape[0], 200)

            # Slow decay (mimicking NMDA receptor)
            current = exponential_decay(
                amplitude=0.5 * u.nA,
                tau=100 * u.ms,
                duration=500 * u.ms
            )
            self.assertEqual(current.shape[0], 5000)

            # Delayed decay
            current = exponential_decay(
                amplitude=5 * u.pA,
                tau=10 * u.ms,
                duration=100 * u.ms,
                t_start=20 * u.ms,  # Start decay at 20ms
                t_end=80 * u.ms  # End at 80ms
            )
            self.assertEqual(current.shape[0], 1000)
            # Check that decay starts after t_start
            self.assertAlmostEqual(u.get_magnitude(current[0]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(current[199]), 0.0, places=5)
            self.assertTrue(u.get_magnitude(current[200]) > 0)  # At t_start

    def test_exponential_decay_full_duration(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 300 * u.ms
            current = exponential_decay(amplitude=1.5 * u.nA,
                                        tau=30 * u.ms,
                                        duration=duration)
            show(current, duration, 'Exponential Decay (Full Duration)')
            self.assertEqual(current.shape[0], 3000)

    def test_double_exponential(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 600 * u.ms
            current = double_exponential(amplitude=1.0 * u.nA,
                                         tau_rise=10 * u.ms,
                                         tau_decay=50 * u.ms,
                                         t_start=50 * u.ms,
                                         duration=duration)
            show(current, duration, 'Double Exponential')
            self.assertEqual(current.shape[0], 6000)

    def test_double_exponential_from_docstring(self):
        """Test examples from double_exponential docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # AMPA-like synaptic current
            current = double_exponential(
                amplitude=1 * u.nA,
                tau_rise=0.5 * u.ms,
                tau_decay=5 * u.ms,
                duration=30 * u.ms
            )
            self.assertEqual(current.shape[0], 300)
            self.assertEqual(u.get_unit(current), u.nA)

            # NMDA-like synaptic current
            current = double_exponential(
                amplitude=0.5 * u.nA,
                tau_rise=2 * u.ms,
                tau_decay=100 * u.ms,
                duration=500 * u.ms
            )
            self.assertEqual(current.shape[0], 5000)

            # GABA-A like inhibitory current
            current = double_exponential(
                amplitude=-0.8 * u.nA,  # Negative for inhibition
                tau_rise=0.5 * u.ms,
                tau_decay=10 * u.ms,
                duration=50 * u.ms
            )
            self.assertEqual(current.shape[0], 500)
            # Check that it's negative
            min_val = np.min(u.get_magnitude(current))
            self.assertTrue(min_val < 0)

            # Delayed synaptic input
            current = double_exponential(
                amplitude=2 * u.pA,
                tau_rise=1 * u.ms,
                tau_decay=15 * u.ms,
                duration=100 * u.ms,
                t_start=20 * u.ms  # Delay of 20ms
            )
            self.assertEqual(current.shape[0], 1000)
            # Check delay
            self.assertAlmostEqual(u.get_magnitude(current[199]), 0.0, places=5)

    def test_double_exponential_different_taus(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 800 * u.ms
            current = double_exponential(amplitude=0.8 * u.nA,
                                         tau_rise=5 * u.ms,
                                         tau_decay=100 * u.ms,
                                         t_start=100 * u.ms,
                                         duration=duration)
            show(current, duration, 'Double Exponential (Fast Rise, Slow Decay)')
            self.assertEqual(current.shape[0], 8000)

    def test_burst(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            current = burst(
                burst_amp=1.0 * u.nA,
                burst_freq=100 * u.Hz,  # Added frequency parameter
                burst_duration=50 * u.ms,
                inter_burst_interval=150 * u.ms,
                n_bursts=5,
                duration=duration
            )
            show(current, duration, 'Burst Input (5 bursts)')
            self.assertEqual(current.shape[0], 10000)

    def test_burst_from_docstring(self):
        """Test examples from burst docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Theta burst stimulation
            current = burst(
                burst_amp=10 * u.pA,
                burst_freq=100 * u.Hz,  # 100Hz within burst
                burst_duration=50 * u.ms,  # 50ms bursts
                inter_burst_interval=150 * u.ms,  # 150ms between bursts
                n_bursts=5,
                duration=1000 * u.ms
            )
            self.assertEqual(current.shape[0], 10000)
            self.assertEqual(u.get_unit(current), u.pA)

            # Gamma burst pattern
            current = burst(
                burst_amp=5 * u.nA,
                burst_freq=40 * u.Hz,  # Gamma frequency
                burst_duration=100 * u.ms,
                inter_burst_interval=100 * u.ms,
                n_bursts=10,
                duration=2000 * u.ms
            )
            self.assertEqual(current.shape[0], 20000)

            # High-frequency stimulation protocol
            current = burst(
                burst_amp=20 * u.pA,
                burst_freq=200 * u.Hz,
                burst_duration=20 * u.ms,
                inter_burst_interval=80 * u.ms,
                n_bursts=20,
                duration=2000 * u.ms
            )
            self.assertEqual(current.shape[0], 20000)

            # Slow oscillatory bursts
            current = burst(
                burst_amp=1 * u.nA,
                burst_freq=5 * u.Hz,  # Slow oscillation
                burst_duration=500 * u.ms,
                inter_burst_interval=500 * u.ms,
                n_bursts=3,
                duration=3000 * u.ms
            )
            self.assertEqual(current.shape[0], 30000)

    def test_burst_with_frequency(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test burst input with different frequencies
            duration = 1500 * u.ms
            current = burst(
                burst_amp=0.7 * u.nA,
                burst_freq=50 * u.Hz,  # 50 Hz oscillation
                burst_duration=100 * u.ms,
                inter_burst_interval=200 * u.ms,
                n_bursts=3,
                duration=duration
            )
            show(current, duration, 'Burst Input with Frequency')
            self.assertEqual(current.shape[0], 15000)

    def test_combined_pulses(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test combining different pulse types
            duration = 1000 * u.ms

            # Create a complex stimulus with multiple pulse types
            current = gaussian_pulse(amplitude=0.5 * u.nA, center=200 * u.ms,
                                     sigma=30 * u.ms, duration=duration)
            current += exponential_decay(amplitude=0.3 * u.nA, tau=50 * u.ms,
                                         t_start=400 * u.ms, duration=duration)
            current += double_exponential(amplitude=0.4 * u.nA, tau_rise=10 * u.ms,
                                          tau_decay=40 * u.ms, t_start=700 * u.ms,
                                          duration=duration)

            show(current, duration, 'Combined Pulse Types')
            self.assertEqual(current.shape[0], 10000)
