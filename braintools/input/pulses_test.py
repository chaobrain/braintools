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

"""Tests for pulse input functions."""

from unittest import TestCase

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braintools.input import (spike_input, gaussian_pulse, exponential_decay, 
                             double_exponential, burst_input)

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


class TestPulseInputs(TestCase):
    def test_spike_input(self):
        current = spike_input(
            sp_times=[10, 20, 30, 200, 300],
            sp_lens=1.,  # can be a list to specify the spike length at each point
            sp_sizes=0.5,  # can be a list to specify the spike current size at each point
            duration=400.
        )
        show(current, 400, 'Spike Input Example')
        self.assertEqual(current.shape[0], 4000)

    def test_spike_input_variable_params(self):
        # Test with variable spike lengths and sizes
        current = spike_input(
            sp_times=[50, 150, 250],
            sp_lens=[1., 2., 3.],  # Different spike lengths
            sp_sizes=[0.5, 1.0, 0.3],  # Different spike amplitudes
            duration=350.
        )
        show(current, 350, 'Spike Input with Variable Parameters')
        self.assertEqual(current.shape[0], 3500)

    def test_gaussian_pulse(self):
        duration = 500 * u.ms
        current = gaussian_pulse(amplitude=1.0,
                                center=250 * u.ms,
                                sigma=30 * u.ms,
                                duration=duration,
                                dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Gaussian Pulse')
        self.assertEqual(current.shape[0], 5000)
        
    def test_gaussian_pulse_multiple(self):
        # Test multiple gaussian pulses
        duration = 1000 * u.ms
        current = gaussian_pulse(amplitude=0.8,
                                center=200 * u.ms,
                                sigma=20 * u.ms,
                                duration=duration,
                                dt=0.1 * u.ms)
        current += gaussian_pulse(amplitude=1.2,
                                 center=600 * u.ms,
                                 sigma=40 * u.ms,
                                 duration=duration,
                                 dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Multiple Gaussian Pulses')
        self.assertEqual(current.shape[0], 10000)

    def test_exponential_decay(self):
        duration = 500 * u.ms
        current = exponential_decay(amplitude=2.0,
                                   tau=50 * u.ms,
                                   t_start=50 * u.ms,
                                   t_end=450 * u.ms,
                                   duration=duration,
                                   dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Exponential Decay')
        self.assertEqual(current.shape[0], 5000)
        
    def test_exponential_decay_full_duration(self):
        duration = 300 * u.ms
        current = exponential_decay(amplitude=1.5,
                                   tau=30 * u.ms,
                                   duration=duration,
                                   dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Exponential Decay (Full Duration)')
        self.assertEqual(current.shape[0], 3000)

    def test_double_exponential(self):
        duration = 600 * u.ms
        current = double_exponential(amplitude=1.0,
                                    tau_rise=10 * u.ms,
                                    tau_decay=50 * u.ms,
                                    t_start=50 * u.ms,
                                    duration=duration,
                                    dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Double Exponential')
        self.assertEqual(current.shape[0], 6000)
        
    def test_double_exponential_different_taus(self):
        duration = 800 * u.ms
        current = double_exponential(amplitude=0.8,
                                    tau_rise=5 * u.ms,
                                    tau_decay=100 * u.ms,
                                    t_start=100 * u.ms,
                                    duration=duration,
                                    dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Double Exponential (Fast Rise, Slow Decay)')
        self.assertEqual(current.shape[0], 8000)

    def test_burst_input(self):
        duration = 1000 * u.ms
        current = burst_input(n_bursts=5,
                             burst_duration=50 * u.ms,
                             inter_burst_interval=150 * u.ms,
                             amplitude=1.0,
                             duration=duration,
                             dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Burst Input (5 bursts)')
        self.assertEqual(current.shape[0], 10000)
        
    def test_burst_input_with_offset(self):
        duration = 1500 * u.ms
        current = burst_input(n_bursts=3,
                             burst_duration=100 * u.ms,
                             inter_burst_interval=200 * u.ms,
                             amplitude=0.7,
                             t_start=200 * u.ms,
                             duration=duration,
                             dt=0.1 * u.ms)
        show(current, duration / u.ms, 'Burst Input with Time Offset')
        self.assertEqual(current.shape[0], 15000)

    def test_combined_pulses(self):
        # Test combining different pulse types
        duration = 1000 * u.ms
        dt = 0.1 * u.ms
        
        # Create a complex stimulus with multiple pulse types
        current = gaussian_pulse(amplitude=0.5, center=200 * u.ms, 
                                sigma=30 * u.ms, duration=duration, dt=dt)
        current += exponential_decay(amplitude=0.3, tau=50 * u.ms,
                                    t_start=400 * u.ms, duration=duration, dt=dt)
        current += double_exponential(amplitude=0.4, tau_rise=10 * u.ms,
                                     tau_decay=40 * u.ms, t_start=700 * u.ms,
                                     duration=duration, dt=dt)
        
        show(current, duration / u.ms, 'Combined Pulse Types')
        self.assertEqual(current.shape[0], 10000)