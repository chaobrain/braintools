# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

from braintools.input import (sinusoidal_input, square_input, triangular_input, 
                             sawtooth_input, chirp_input, noisy_sinusoidal)

block = False

brainstate.environ.set(dt=0.1)


def show(current, duration, title=''):
    if plt is not None:
        ts = np.arange(0, duration, brainstate.environ.get_dt())
        plt.plot(ts, current)
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestWaveformInputs(TestCase):
    def test_sinusoidal_input(self):
        duration = 2000 * u.ms
        current = sinusoidal_input(amplitude=1., frequency=2.0 * u.Hz,
                                  duration=duration, t_start=100. * u.ms, dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Sinusoidal Input')
        self.assertEqual(current.shape[0], 20000)

    def test_sinusoidal_input_with_bias(self):
        duration = 1000 * u.ms
        current = sinusoidal_input(amplitude=0.5, frequency=5.0 * u.Hz,
                                  duration=duration, bias=True, dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Sinusoidal Input with Bias')
        # With bias, the signal should oscillate between 0 and 2*amplitude
        self.assertTrue(np.all(current >= -0.01))  # Allow small numerical errors
        self.assertEqual(current.shape[0], 10000)

    def test_square_input(self):
        duration = 2000 * u.ms
        current = square_input(amplitude=1., frequency=2.0 * u.Hz,
                              duration=duration,
                              t_start=100 * u.ms,
                              dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Square Input')
        self.assertEqual(current.shape[0], 20000)

    def test_square_input_with_bias(self):
        duration = 1000 * u.ms
        current = square_input(amplitude=0.5, frequency=3.0 * u.Hz,
                              duration=duration, bias=True, dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Square Input with Bias')
        self.assertEqual(current.shape[0], 10000)

    def test_triangular_input(self):
        duration = 1500 * u.ms
        current = triangular_input(amplitude=1.0, frequency=1.5 * u.Hz,
                                  duration=duration, dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Triangular Wave Input')
        self.assertEqual(current.shape[0], 15000)
        
    def test_triangular_input_with_bias(self):
        duration = 1000 * u.ms
        current = triangular_input(amplitude=0.8, frequency=2.0 * u.Hz,
                                  duration=duration, bias=True,
                                  t_start=50 * u.ms, t_end=950 * u.ms,
                                  dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Triangular Wave with Bias')
        self.assertEqual(current.shape[0], 10000)

    def test_sawtooth_input(self):
        duration = 1500 * u.ms
        current = sawtooth_input(amplitude=1.0, frequency=1.0 * u.Hz,
                                duration=duration, dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Sawtooth Wave Input')
        self.assertEqual(current.shape[0], 15000)
        
    def test_sawtooth_input_with_time_window(self):
        duration = 2000 * u.ms
        current = sawtooth_input(amplitude=0.7, frequency=2.5 * u.Hz,
                                duration=duration,
                                t_start=200 * u.ms, t_end=1800 * u.ms,
                                dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Sawtooth Wave with Time Window')
        self.assertEqual(current.shape[0], 20000)

    def test_chirp_input_linear(self):
        duration = 2000 * u.ms
        current = chirp_input(amplitude=1.0,
                             f_start=0.5 * u.Hz,
                             f_end=5.0 * u.Hz,
                             duration=duration,
                             method='linear',
                             dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Linear Chirp Input (0.5 Hz to 5 Hz)')
        self.assertEqual(current.shape[0], 20000)

    def test_chirp_input_logarithmic(self):
        duration = 2000 * u.ms
        current = chirp_input(amplitude=0.8,
                             f_start=1.0 * u.Hz,
                             f_end=10.0 * u.Hz,
                             duration=duration,
                             method='logarithmic',
                             dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Logarithmic Chirp Input (1 Hz to 10 Hz)')
        self.assertEqual(current.shape[0], 20000)

    def test_chirp_input_with_bias(self):
        duration = 1500 * u.ms
        current = chirp_input(amplitude=0.5,
                             f_start=2.0 * u.Hz,
                             f_end=8.0 * u.Hz,
                             duration=duration,
                             bias=True,
                             t_start=100 * u.ms,
                             t_end=1400 * u.ms,
                             dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Chirp with Bias and Time Window')
        self.assertEqual(current.shape[0], 15000)

    def test_noisy_sinusoidal(self):
        duration = 1000 * u.ms
        current = noisy_sinusoidal(amplitude=1.0,
                                   frequency=3.0 * u.Hz,
                                   noise_amplitude=0.2,
                                   duration=duration,
                                   dt=0.1 * u.ms,
                                   seed=42)
        show(current, duration / u.ms, 'Noisy Sinusoidal Input')
        self.assertEqual(current.shape[0], 10000)
        
    def test_noisy_sinusoidal_reproducible(self):
        # Test that seed produces reproducible results
        duration = 500 * u.ms
        current1 = noisy_sinusoidal(amplitude=1.0,
                                    frequency=2.0 * u.Hz,
                                    noise_amplitude=0.3,
                                    duration=duration,
                                    seed=123,
                                    dt=0.1 * u.ms)
        current2 = noisy_sinusoidal(amplitude=1.0,
                                    frequency=2.0 * u.Hz,
                                    noise_amplitude=0.3,
                                    duration=duration,
                                    seed=123,
                                    dt=0.1 * u.ms)
        np.testing.assert_array_equal(current1, current2)