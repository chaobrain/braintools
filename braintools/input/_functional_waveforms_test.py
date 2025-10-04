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

"""Tests for waveform input functions."""

from unittest import TestCase

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braintools.input import (sinusoidal, square, triangular, sawtooth, chirp, noisy_sinusoidal)

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


class TestWaveformInputs(TestCase):
    def test_sinusoidal(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 2000 * u.ms
            current = sinusoidal(amplitude=1. * u.pA,
                                 frequency=2.0 * u.Hz,
                                 duration=duration,
                                 t_start=100. * u.ms)
            show(current, duration, 'Sinusoidal Input')
            self.assertEqual(current.shape[0], 20000)

    def test_sinusoidal_with_bias(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            current = sinusoidal(amplitude=0.5 * u.pA,
                                 frequency=5.0 * u.Hz,
                                 duration=duration,
                                 bias=True)
            show(current, duration, 'Sinusoidal Input with Bias')
            # With bias, the signal should oscillate between 0 and 2*amplitude
            self.assertTrue(np.all(u.get_magnitude(current) >= -0.01))  # Allow small numerical errors
            self.assertEqual(current.shape[0], 10000)

    def test_sinusoidal_from_docstring(self):
        """Test examples from sinusoidal docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple 10 Hz sinusoid
            current = sinusoidal(
                amplitude=5 * u.pA,
                frequency=10 * u.Hz,
                duration=1000 * u.ms
            )
            self.assertEqual(current.shape[0], 10000)
            self.assertEqual(u.get_unit(current), u.pA)

            # High-frequency stimulation
            current = sinusoidal(
                amplitude=2 * u.nA,
                frequency=100 * u.Hz,
                duration=500 * u.ms
            )
            self.assertEqual(current.shape[0], 5000)

            # Sinusoid with positive bias
            current = sinusoidal(
                amplitude=10 * u.pA,
                frequency=5 * u.Hz,
                duration=2000 * u.ms,
                bias=True
            )
            self.assertEqual(current.shape[0], 20000)
            self.assertTrue(np.all(u.get_magnitude(current) >= -0.01))

            # Windowed sinusoid
            current = sinusoidal(
                amplitude=8 * u.pA,
                frequency=20 * u.Hz,
                duration=1000 * u.ms,
                t_start=200 * u.ms,
                t_end=800 * u.ms
            )
            self.assertEqual(current.shape[0], 10000)
            # Check windowing
            self.assertAlmostEqual(u.get_magnitude(current[0]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(current[1999]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(current[8000]), 0.0, places=5)

    def test_square(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 2000 * u.ms
            current = square(amplitude=1. * u.pA,
                             frequency=2.0 * u.Hz,
                             duration=duration,
                             t_start=100 * u.ms)
            show(current, duration, 'Square Input')
            self.assertEqual(current.shape[0], 20000)

    def test_square_with_bias(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            current = square(amplitude=0.5 * u.pA, frequency=3.0 * u.Hz,
                             duration=duration, bias=True)
            show(current, duration, 'Square Input with Bias')
            self.assertEqual(current.shape[0], 10000)

    def test_square_with_duty_cycle(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            current = square(amplitude=1.0 * u.nA, frequency=2.0 * u.Hz,
                             duration=duration, duty_cycle=0.2)
            show(current, duration, 'Square Input with 20% Duty Cycle')
            self.assertEqual(current.shape[0], 10000)

    def test_square_from_docstring(self):
        """Test examples from square docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Symmetric square wave at 2 Hz
            current = square(
                amplitude=10 * u.pA,
                frequency=2 * u.Hz,
                duration=2000 * u.ms
            )
            self.assertEqual(current.shape[0], 20000)

            # High-frequency pulse train with duty cycle
            current = square(
                amplitude=5 * u.nA,
                frequency=50 * u.Hz,
                duration=500 * u.ms,
                duty_cycle=0.2
            )
            self.assertEqual(current.shape[0], 5000)

            # Square wave with positive bias
            current = square(
                amplitude=8 * u.pA,
                frequency=10 * u.Hz,
                duration=1000 * u.ms,
                bias=True
            )
            self.assertEqual(current.shape[0], 10000)

    def test_triangular(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1500 * u.ms
            current = triangular(amplitude=1.0 * u.pA, frequency=1.5 * u.Hz,
                                 duration=duration)
            show(current, duration, 'Triangular Wave Input')
            self.assertEqual(current.shape[0], 15000)

    def test_triangular_with_bias(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            current = triangular(amplitude=0.8 * u.pA, frequency=2.0 * u.Hz,
                                 duration=duration, bias=True,
                                 t_start=50 * u.ms, t_end=950 * u.ms)
            show(current, duration, 'Triangular Wave with Bias')
            self.assertEqual(current.shape[0], 10000)

    def test_triangular_from_docstring(self):
        """Test examples from triangular docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple triangular wave at 5 Hz
            current = triangular(
                amplitude=10 * u.pA,
                frequency=5 * u.Hz,
                duration=1000 * u.ms
            )
            self.assertEqual(current.shape[0], 10000)

            # Slow triangular ramp
            current = triangular(
                amplitude=100 * u.pA,
                frequency=0.5 * u.Hz,
                duration=4000 * u.ms
            )
            self.assertEqual(current.shape[0], 40000)

    def test_sawtooth(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1500 * u.ms
            current = sawtooth(amplitude=1.0 * u.pA, frequency=1.0 * u.Hz,
                               duration=duration)
            show(current, duration, 'Sawtooth Wave Input')
            self.assertEqual(current.shape[0], 15000)

    def test_sawtooth_with_time_window(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 2000 * u.ms
            current = sawtooth(amplitude=0.7 * u.pA, frequency=2.5 * u.Hz,
                               duration=duration,
                               t_start=200 * u.ms, t_end=1800 * u.ms)
            show(current, duration, 'Sawtooth Wave with Time Window')
            self.assertEqual(current.shape[0], 20000)

    def test_sawtooth_from_docstring(self):
        """Test examples from sawtooth docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple sawtooth at 2 Hz
            current = sawtooth(
                amplitude=10 * u.pA,
                frequency=2 * u.Hz,
                duration=2000 * u.ms
            )
            self.assertEqual(current.shape[0], 20000)

            # Sawtooth with positive bias
            current = sawtooth(
                amplitude=5 * u.nA,
                frequency=10 * u.Hz,
                duration=500 * u.ms,
                bias=True
            )
            self.assertEqual(current.shape[0], 5000)

    def test_chirp_linear(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 2000 * u.ms
            current = chirp(amplitude=1.0 * u.pA,
                            f_start=0.5 * u.Hz,
                            f_end=5.0 * u.Hz,
                            duration=duration,
                            method='linear')
            show(current, duration, 'Linear Chirp Input (0.5 Hz to 5 Hz)')
            self.assertEqual(current.shape[0], 20000)

    def test_chirp_logarithmic(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 2000 * u.ms
            current = chirp(amplitude=0.8 * u.pA,
                            f_start=1.0 * u.Hz,
                            f_end=10.0 * u.Hz,
                            duration=duration,
                            method='logarithmic')
            show(current, duration, 'Logarithmic Chirp Input (1 Hz to 10 Hz)')
            self.assertEqual(current.shape[0], 20000)

    def test_chirp_with_bias(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1500 * u.ms
            current = chirp(amplitude=0.5 * u.pA,
                            f_start=2.0 * u.Hz,
                            f_end=8.0 * u.Hz,
                            duration=duration,
                            bias=True,
                            t_start=100 * u.ms,
                            t_end=1400 * u.ms)
            show(current, duration, 'Chirp with Bias and Time Window')
            self.assertEqual(current.shape[0], 15000)

    def test_chirp_from_docstring(self):
        """Test examples from chirp docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Linear frequency sweep
            current = chirp(
                amplitude=5 * u.pA,
                f_start=1 * u.Hz,
                f_end=50 * u.Hz,
                duration=2000 * u.ms,
                method='linear'
            )
            self.assertEqual(current.shape[0], 20000)

            # Logarithmic sweep
            current = chirp(
                amplitude=2 * u.nA,
                f_start=0.1 * u.Hz,
                f_end=100 * u.Hz,
                duration=5000 * u.ms,
                method='logarithmic'
            )
            self.assertEqual(current.shape[0], 50000)

            # Reverse chirp (high to low)
            current = chirp(
                amplitude=3 * u.nA,
                f_start=100 * u.Hz,
                f_end=1 * u.Hz,
                duration=2000 * u.ms
            )
            self.assertEqual(current.shape[0], 20000)

    def test_noisy_sinusoidal(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 1000 * u.ms
            current = noisy_sinusoidal(amplitude=1.0 * u.pA,
                                       frequency=3.0 * u.Hz,
                                       noise_amplitude=0.2 * u.pA,
                                       duration=duration,
                                       seed=42)
            show(current, duration, 'Noisy Sinusoidal Input')
            self.assertEqual(current.shape[0], 10000)

    def test_noisy_sinusoidal_reproducible(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test that seed produces reproducible results
            duration = 500 * u.ms
            current1 = noisy_sinusoidal(amplitude=1.0 * u.nA,
                                        frequency=2.0 * u.Hz,
                                        noise_amplitude=0.3 * u.nA,
                                        duration=duration,
                                        seed=123)
            current2 = noisy_sinusoidal(amplitude=1.0 * u.nA,
                                        frequency=2.0 * u.Hz,
                                        noise_amplitude=0.3 * u.nA,
                                        duration=duration,
                                        seed=123)
            assert u.math.allclose(current1, current2)

    def test_noisy_sinusoidal_from_docstring(self):
        """Test examples from noisy_sinusoidal docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Sinusoid with small noise
            current = noisy_sinusoidal(
                amplitude=10 * u.pA,
                frequency=10 * u.Hz,
                noise_amplitude=1 * u.pA,
                duration=1000 * u.ms
            )
            self.assertEqual(current.shape[0], 10000)

            # High noise for stochastic resonance
            current = noisy_sinusoidal(
                amplitude=5 * u.pA,
                frequency=5 * u.Hz,
                noise_amplitude=10 * u.pA,
                duration=2000 * u.ms
            )
            self.assertEqual(current.shape[0], 20000)

            # Reproducible noisy signal
            current = noisy_sinusoidal(
                amplitude=15 * u.pA,
                frequency=40 * u.Hz,
                noise_amplitude=3 * u.pA,
                duration=500 * u.ms,
                seed=42
            )
            self.assertEqual(current.shape[0], 5000)
            # Test reproducibility
            current2 = noisy_sinusoidal(
                amplitude=15 * u.pA,
                frequency=40 * u.Hz,
                noise_amplitude=3 * u.pA,
                duration=500 * u.ms,
                seed=42
            )
            assert u.math.allclose(current, current2)
