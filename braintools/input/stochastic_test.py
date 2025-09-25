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

"""Tests for stochastic input functions."""

from unittest import TestCase

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braintools.input import wiener_process, ou_process, poisson_input

block = False

brainstate.environ.set(dt=0.1)


def show(current, duration, title=''):
    if plt is not None:
        ts = np.arange(0, duration, brainstate.environ.get_dt())
        if current.ndim == 1:
            plt.plot(ts, current)
        else:
            # For multi-dimensional inputs, plot each channel
            for i in range(current.shape[1]):
                plt.plot(ts, current[:, i], label=f'Channel {i+1}')
            plt.legend()
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestStochasticInputs(TestCase):
    def test_wiener_process(self):
        duration = 200
        current = wiener_process(duration, n=2, t_start=10., t_end=180.)
        show(current, duration, 'Wiener Process')
        self.assertEqual(current.shape, (2000, 2))

    def test_wiener_process_single_channel(self):
        duration = 300
        current = wiener_process(duration, n=1)
        show(current, duration, 'Wiener Process (Single Channel)')
        self.assertEqual(current.shape, (3000, 1))

    def test_wiener_process_with_seed(self):
        duration = 100
        # Test reproducibility with seed
        current1 = wiener_process(duration, n=3, seed=42)
        current2 = wiener_process(duration, n=3, seed=42)
        np.testing.assert_array_equal(current1, current2)
        self.assertEqual(current1.shape, (1000, 3))

    def test_ou_process(self):
        duration = 200
        current = ou_process(mean=1., sigma=0.1, tau=10., 
                           duration=duration, n=2, 
                           t_start=10., t_end=180.)
        show(current, duration, 'Ornstein-Uhlenbeck Process')
        self.assertEqual(current.shape, (2000, 2))

    def test_ou_process_different_params(self):
        duration = 400
        # Test with different OU parameters
        current = ou_process(mean=0.5, sigma=0.2, tau=20.,
                           duration=duration, n=3)
        show(current, duration, 'OU Process (Different Parameters)')
        self.assertEqual(current.shape, (4000, 3))
        
        # Check that mean is approximately correct over long time
        long_mean = np.mean(current[2000:])  # Use second half for better convergence
        self.assertAlmostEqual(long_mean, 0.5, delta=0.2)

    def test_ou_process_with_seed(self):
        duration = 150
        # Test reproducibility
        current1 = ou_process(mean=0., sigma=0.15, tau=15.,
                            duration=duration, n=2, seed=123)
        current2 = ou_process(mean=0., sigma=0.15, tau=15.,
                            duration=duration, n=2, seed=123)
        np.testing.assert_array_equal(current1, current2)

    def test_poisson_input(self):
        duration = 500 * u.ms
        current = poisson_input(rate=20 * u.Hz,
                              duration=duration,
                              n=3,
                              dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Poisson Input (20 Hz)')
        self.assertEqual(current.shape, (5000, 3))

    def test_poisson_input_high_rate(self):
        duration = 300 * u.ms
        current = poisson_input(rate=100 * u.Hz,
                              duration=duration,
                              n=2,
                              dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Poisson Input (High Rate: 100 Hz)')
        self.assertEqual(current.shape, (3000, 2))
        
        # Check that spike count is reasonable for the given rate
        spike_count = np.sum(current > 0.5)  # Count spikes
        expected_spikes = 100 * 0.3 * 2  # rate * duration_in_s * n_channels
        # Allow for statistical variation
        self.assertGreater(spike_count, expected_spikes * 0.5)
        self.assertLess(spike_count, expected_spikes * 2.0)

    def test_poisson_input_with_time_window(self):
        duration = 600 * u.ms
        current = poisson_input(rate=30 * u.Hz,
                              duration=duration,
                              n=1,
                              t_start=100 * u.ms,
                              t_end=500 * u.ms,
                              dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Poisson Input with Time Window')
        self.assertEqual(current.shape, (6000, 1))
        
        # Check that spikes only occur in the specified time window
        before_window = current[:1000]  # Before t_start
        after_window = current[5000:]   # After t_end
        self.assertTrue(np.all(before_window == 0))
        self.assertTrue(np.all(after_window == 0))

    def test_poisson_input_with_seed(self):
        duration = 200 * u.ms
        # Test reproducibility
        current1 = poisson_input(rate=50 * u.Hz,
                               duration=duration,
                               n=4,
                               seed=456,
                               dt=0.1 * u.ms)
        current2 = poisson_input(rate=50 * u.Hz,
                               duration=duration,
                               n=4,
                               seed=456,
                               dt=0.1 * u.ms)
        np.testing.assert_array_equal(current1, current2)

    def test_combined_stochastic(self):
        # Test combining different stochastic processes
        duration = 400
        dt = 0.1
        
        # Create a complex stochastic stimulus
        ou = ou_process(mean=0.5, sigma=0.1, tau=10.,
                       duration=duration, n=1)
        wiener = 0.2 * wiener_process(duration, n=1)
        
        combined = ou + wiener
        show(combined, duration, 'Combined OU + Wiener Process')
        self.assertEqual(combined.shape, (4000, 1))