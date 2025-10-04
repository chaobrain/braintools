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

"""Tests for stochastic input functions."""

from unittest import TestCase

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braintools.input import wiener_process, ou_process, poisson

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


class TestStochasticInputs(TestCase):
    def test_wiener_process(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 200 * u.ms
            current = wiener_process(duration, sigma=0.5 * u.pA, n=2, t_start=10. * u.ms, t_end=180. * u.ms)
            show(current, duration, 'Wiener Process')
            self.assertEqual(current.shape, (2000, 2))

    def test_wiener_process_single_channel(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 300 * u.ms
            current = wiener_process(duration, sigma=1.0 * u.nA, n=1)
            show(current, duration, 'Wiener Process (Single Channel)')
            self.assertEqual(current.shape[0], 3000)

    def test_wiener_process_with_seed(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 100 * u.ms
            # Test reproducibility with seed
            current1 = wiener_process(duration, sigma=0.3 * u.nA, n=3, seed=42)
            current2 = wiener_process(duration, sigma=0.3 * u.nA, n=3, seed=42)
            assert u.math.allclose(current1, current2)
            self.assertEqual(current1.shape, (1000, 3))

    def test_wiener_process_from_docstring(self):
        """Test examples from wiener_process docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple Wiener process
            noise = wiener_process(
                duration=100 * u.ms,
                sigma=0.5 * u.pA
            )
            self.assertEqual(noise.shape[0], 1000)
            self.assertEqual(u.get_unit(noise), u.pA)

            # Multiple independent processes
            noises = wiener_process(
                duration=200 * u.ms,
                sigma=1.0 * u.nA,
                n=10  # 10 independent processes
            )
            self.assertEqual(noises.shape, (2000, 10))
            self.assertEqual(u.get_unit(noises), u.nA)

            # Windowed noise
            noise = wiener_process(
                duration=500 * u.ms,
                sigma=2.0 * u.pA,
                t_start=100 * u.ms,
                t_end=400 * u.ms
            )
            self.assertEqual(noise.shape[0], 5000)
            # Check that noise is zero outside window
            self.assertAlmostEqual(u.get_magnitude(noise[0]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(noise[999]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(noise[4000]), 0.0, places=5)

            # Reproducible noise
            noise = wiener_process(
                duration=100 * u.ms,
                sigma=0.3 * u.nA,
                seed=42  # Fixed seed for reproducibility
            )
            self.assertEqual(noise.shape[0], 1000)
            # Test reproducibility
            noise2 = wiener_process(
                duration=100 * u.ms,
                sigma=0.3 * u.nA,
                seed=42
            )
            assert u.math.allclose(noise, noise2)

    def test_ou_process(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 200 * u.ms
            current = ou_process(mean=1. * u.nA, sigma=0.1 * u.nA, tau=10. * u.ms,
                                 duration=duration, n=2,
                                 t_start=10. * u.ms, t_end=180. * u.ms)
            show(current, duration, 'Ornstein-Uhlenbeck Process')
            self.assertEqual(current.shape, (2000, 2))

    def test_ou_process_different_params(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 400 * u.ms
            # Test with different OU parameters
            current = ou_process(mean=0.5 * u.nA, sigma=0.2 * u.nA, tau=20. * u.ms,
                                 duration=duration, n=3)
            show(current, duration, 'OU Process (Different Parameters)')
            self.assertEqual(current.shape, (4000, 3))

    def test_ou_process_with_seed(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 150 * u.ms
            # Test reproducibility
            current1 = ou_process(mean=0. * u.pA, sigma=0.15 * u.pA, tau=15. * u.ms,
                                  duration=duration, n=2, seed=123)
            current2 = ou_process(mean=0. * u.pA, sigma=0.15 * u.pA, tau=15. * u.ms,
                                  duration=duration, n=2, seed=123)
            assert u.math.allclose(current1, current2)

    def test_ou_process_from_docstring(self):
        """Test examples from ou_process docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple OU process
            current = ou_process(
                mean=0.5 * u.nA,
                sigma=0.2 * u.nA,
                tau=10 * u.ms,
                duration=500 * u.ms
            )
            self.assertEqual(current.shape[0], 5000)
            self.assertEqual(u.get_unit(current), u.nA)

            # Fast fluctuations around zero
            current = ou_process(
                mean=0 * u.pA,
                sigma=5 * u.pA,
                tau=2 * u.ms,  # Fast time constant
                duration=200 * u.ms
            )
            self.assertEqual(current.shape[0], 2000)
            self.assertEqual(u.get_unit(current), u.pA)

            # Slow fluctuations with drift
            current = ou_process(
                mean=1.0 * u.nA,
                sigma=0.3 * u.nA,
                tau=50 * u.ms,  # Slow time constant
                duration=1000 * u.ms
            )
            self.assertEqual(current.shape[0], 10000)

            # Multiple independent processes
            currents = ou_process(
                mean=0 * u.pA,
                sigma=2 * u.pA,
                tau=5 * u.ms,
                duration=300 * u.ms,
                n=10  # 10 independent processes
            )
            self.assertEqual(currents.shape, (3000, 10))

            # Windowed OU process
            current = ou_process(
                mean=0.5 * u.nA,
                sigma=0.1 * u.nA,
                tau=20 * u.ms,
                duration=500 * u.ms,
                t_start=100 * u.ms,
                t_end=400 * u.ms
            )
            self.assertEqual(current.shape[0], 5000)
            # Check windowing
            self.assertAlmostEqual(u.get_magnitude(current[0]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(current[999]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(current[4000]), 0.0, places=5)

    def test_poisson(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 500 * u.ms
            current = poisson(rate=20 * u.Hz,
                              duration=duration,
                              amplitude=1 * u.pA,
                              n=3)
            show(current, duration, 'Poisson Input (20 Hz)')
            self.assertEqual(current.shape, (5000, 3))

    def test_poisson_high_rate(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 300 * u.ms
            current = poisson(rate=100 * u.Hz,
                              duration=duration,
                              amplitude=0.5 * u.nA,
                              n=2)
            show(current, duration, 'Poisson Input (High Rate: 100 Hz)')
            self.assertEqual(current.shape, (3000, 2))

    def test_poisson_with_time_window(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 600 * u.ms
            current = poisson(rate=30 * u.Hz,
                              duration=duration,
                              amplitude=2 * u.pA,
                              n=1,
                              t_start=100 * u.ms,
                              t_end=500 * u.ms)
            show(current, duration, 'Poisson Input with Time Window')
            self.assertEqual(current.shape[0], 6000)

            # Check that spikes only occur in the specified time window
            before_window = u.get_magnitude(current[:1000])  # Before t_start
            after_window = u.get_magnitude(current[5000:])  # After t_end
            self.assertTrue(np.all(before_window == 0))
            self.assertTrue(np.all(after_window == 0))

    def test_poisson_with_seed(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 200 * u.ms
            # Test reproducibility
            current1 = poisson(rate=50 * u.Hz,
                               duration=duration,
                               amplitude=1 * u.nA,
                               n=4,
                               seed=456)
            current2 = poisson(rate=50 * u.Hz,
                               duration=duration,
                               amplitude=1 * u.nA,
                               n=4,
                               seed=456)
            assert u.math.allclose(current1, current2)

    def test_poisson_from_docstring(self):
        """Test examples from poisson docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple Poisson spike train
            spikes = poisson(
                rate=10 * u.Hz,
                duration=1000 * u.ms,
                amplitude=1 * u.pA
            )
            self.assertEqual(spikes.shape[0], 10000)
            self.assertEqual(u.get_unit(spikes), u.pA)

            # High-frequency background activity
            spikes = poisson(
                rate=100 * u.Hz,
                duration=500 * u.ms,
                amplitude=0.5 * u.nA
            )
            self.assertEqual(spikes.shape[0], 5000)
            self.assertEqual(u.get_unit(spikes), u.nA)

            # Multiple independent spike trains
            spikes = poisson(
                rate=20 * u.Hz,
                duration=2000 * u.ms,
                amplitude=2 * u.pA,
                n=50  # 50 independent spike trains
            )
            self.assertEqual(spikes.shape, (20000, 50))

            # Windowed spiking activity
            spikes = poisson(
                rate=50 * u.Hz,
                duration=1000 * u.ms,
                amplitude=1 * u.nA,
                t_start=200 * u.ms,
                t_end=800 * u.ms
            )
            self.assertEqual(spikes.shape[0], 10000)
            # Check windowing
            self.assertTrue(np.all(u.get_magnitude(spikes[:2000]) == 0))
            self.assertTrue(np.all(u.get_magnitude(spikes[8000:]) == 0))

            # Low rate spontaneous activity
            spikes = poisson(
                rate=1 * u.Hz,
                duration=10000 * u.ms,
                amplitude=5 * u.pA,
                seed=123  # Reproducible spike pattern
            )
            self.assertEqual(spikes.shape[0], 100000)
            # Test reproducibility
            spikes2 = poisson(
                rate=1 * u.Hz,
                duration=10000 * u.ms,
                amplitude=5 * u.pA,
                seed=123
            )
            self.assertTrue(u.math.allclose(spikes, spikes2))

    def test_combined_stochastic(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test combining different stochastic processes
            duration = 400 * u.ms

            # Create a complex stochastic stimulus
            ou = ou_process(mean=0.5 * u.nA,
                            sigma=0.1 * u.nA,
                            tau=10. * u.ms,
                            duration=duration,
                            n=1)
            wiener = wiener_process(duration, sigma=0.2 * u.nA, n=1)

            combined = ou + wiener
            show(combined, duration, 'Combined OU + Wiener Process')
            self.assertEqual(combined.shape[0], 4000)
