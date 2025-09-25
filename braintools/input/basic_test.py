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

"""Tests for basic input functions."""

from unittest import TestCase

import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from braintools.input import section_input, constant_input, step_input, ramp_input

block = False

brainstate.environ.set(dt=0.1)


def show(current, duration, title=''):
    if plt is not None:
        ts = np.arange(0, u.get_magnitude(duration), u.get_magnitude(brainstate.environ.get_dt()))
        plt.plot(ts, current)
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestBasicInputs(TestCase):
    def test_section_input(self):
        with brainstate.environ.context(dt=0.1):
            current1, duration = section_input(values=[0, 1., 0.],
                                               durations=[100, 300, 100],
                                               return_length=True,
                                               dt=0.1)
            show(current1, duration, 'values=[0, 1, 0], durations=[100, 300, 100]')
            self.assertEqual(current1.shape[0], 5000)

    def test_section_input_multidim(self):
        with brainstate.environ.context(dt=0.1):
            brainstate.random.seed(123)
            current = section_input(values=[0, jnp.ones(10), brainstate.random.random((3, 10))],
                                    durations=[100, 300, 100])
            self.assertTrue(current.shape == (5000, 3, 10))

    def test_section_input_different_dt(self):
        with brainstate.environ.context(dt=0.1):
            I1 = section_input(values=[0, 1, 2], durations=[10, 20, 30], dt=0.1)
            I2 = section_input(values=[0, 1, 2], durations=[10, 20, 30], dt=0.01)
            self.assertTrue(I1.shape[0] == 600)
            self.assertTrue(I2.shape[0] == 6000)

    def test_constant_input(self):
        with brainstate.environ.context(dt=0.1):
            current2, duration = constant_input([(0, 100), (1, 300), (0, 100)])
            show(current2, duration, '[(0, 100), (1, 300), (0, 100)]')
            self.assertEqual(current2.shape[0], 5000)

    def test_step_input(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 500 * u.ms
            # Test step function with multiple levels
            amplitudes = [0.0, 1.0, 0.5, 2.0]
            step_times = [0 * u.ms, 100 * u.ms, 250 * u.ms, 400 * u.ms]

            current = step_input(amplitudes, step_times, duration)
            show(current, u.maybe_decimal(duration / u.ms), 'Step Input: Multiple Levels')
            self.assertEqual(current.shape[0], 5000)

    def test_step_input_unsorted(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test that step function works with unsorted times
            duration = 300 * u.ms
            amplitudes = [1.0, 0.5, 2.0]
            step_times = [100 * u.ms, 0 * u.ms, 200 * u.ms]  # Unsorted

            current = step_input(amplitudes, step_times, duration)
            # Should automatically sort and produce correct output
            self.assertEqual(current.shape[0], 3000)

    def test_ramp_input(self):
        with brainstate.environ.context(dt=0.1):
            duration = 500
            current4 = ramp_input(0, 1, duration)

            show(current4, duration, r'$c_{start}$=0, $c_{end}$=%d, duration, '
                                     r'$t_{start}$=0, $t_{end}$=None' % duration)
            self.assertEqual(current4.shape[0], 5000)

    def test_ramp_input_with_times(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration, t_start, t_end = 500 * u.ms, 100 * u.ms, 400 * u.ms
            current5 = ramp_input(0, 1, duration, t_start, t_end)

            show(current5, duration)
            self.assertEqual(current5.shape[0], 5000)
