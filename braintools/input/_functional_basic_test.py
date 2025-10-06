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

"""Tests for basic input functions."""

from unittest import TestCase

import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from braintools.input import section, constant, step, ramp

block = False


def show(current, duration, title=''):
    if plt is not None:
        ts = np.arange(0, u.get_magnitude(duration), u.get_magnitude(brainstate.environ.get_dt()))
        plt.plot(ts, current)
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestBasicInputs(TestCase):
    def test_section(self):
        with brainstate.environ.context(dt=0.1):
            current1, duration = section(values=[0, 1., 0.],
                                         durations=[100, 300, 100],
                                         return_length=True)
            show(current1, duration, 'values=[0, 1, 0], durations=[100, 300, 100]')
            self.assertEqual(current1.shape[0], 5000)

    def test_section_multidim(self):
        with brainstate.environ.context(dt=0.1):
            brainstate.random.seed(123)
            current = section(values=[0, jnp.ones(10), brainstate.random.random((3, 10))],
                              durations=[100, 300, 100])
            self.assertTrue(current.shape == (5000, 3, 10))

    def test_section_different_dt(self):
        with brainstate.environ.context(dt=0.1):
            I1 = section(values=[0, 1, 2], durations=[10, 20, 30])
            self.assertTrue(I1.shape[0] == 600)
        with brainstate.environ.context(dt=0.01):
            I2 = section(values=[0, 1, 2], durations=[10, 20, 30])
            self.assertTrue(I2.shape[0] == 6000)

    def test_section_with_units(self):
        """Test section with units from docstring examples."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple step protocol
            current = section(
                values=[0, 10, 0] * u.pA,
                durations=[100, 200, 100] * u.ms
            )
            self.assertEqual(current.shape[0], 4000)

            # Multiple channel input
            values = [np.zeros(3), np.ones(3) * 5, np.zeros(3)] * u.nA
            current = section(
                values=values,
                durations=[50, 100, 50] * u.ms
            )
            self.assertEqual(current.shape, (2000, 3))

            # Get both current and duration
            current, duration = section(
                values=[0, 1, 2, 1, 0] * u.pA,
                durations=[20, 20, 40, 20, 20] * u.ms,
                return_length=True
            )
            self.assertEqual(current.shape[0], 1200)
            self.assertAlmostEqual(u.get_magnitude(duration), 120.0)

            # Complex protocol with different phases
            protocol_values = [0, 2, 5, 10, 5, 2, 0] * u.pA
            protocol_durations = [50, 30, 30, 100, 30, 30, 50] * u.ms
            current = section(protocol_values, protocol_durations)
            self.assertEqual(current.shape[0], 3200)

    def test_constant(self):
        with brainstate.environ.context(dt=0.1):
            current2, duration = constant([(0, 100), (1, 300), (0, 100)])
            show(current2, duration, '[(0, 100), (1, 300), (0, 100)]')
            self.assertEqual(current2.shape[0], 5000)

    def test_constant_with_units(self):
        """Test constant with units from docstring examples."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple two-phase protocol
            current, duration = constant([
                (0 * u.pA, 100 * u.ms),
                (10 * u.pA, 200 * u.ms)
            ])
            self.assertEqual(current.shape[0], 3000)
            self.assertAlmostEqual(u.get_magnitude(duration), 300.0)

            # Mixed scalar and array values
            with pytest.raises(u.UnitMismatchError):
                current, duration = constant([
                    (0, 50 * u.ms),
                    (np.array([1, 2, 3]) * u.nA, 100 * u.ms),
                    (0, 50 * u.ms)
                ])
                self.assertEqual(current.shape, (2000, 3))

            # Complex multi-phase stimulation
            phases = [
                (0 * u.pA, 20 * u.ms),  # baseline
                (5 * u.pA, 50 * u.ms),  # weak stimulus
                (10 * u.pA, 100 * u.ms),  # strong stimulus
                (2 * u.pA, 30 * u.ms),  # recovery
                (0 * u.pA, 50 * u.ms),  # rest
            ]
            current, total_time = constant(phases)
            self.assertEqual(current.shape[0], 2500)
            self.assertAlmostEqual(u.get_magnitude(total_time), 250.0)

            # Using arrays for spatial patterns
            spatial_pattern = np.array([[1, 0], [0, 1]]) * u.nA
            current, duration = constant([
                (np.zeros((2, 2)) * u.nA, 100 * u.ms),
                (spatial_pattern, 200 * u.ms),
                (np.zeros((2, 2)) * u.nA, 100 * u.ms)
            ])
            self.assertEqual(current.shape, (4000, 2, 2))

            # Ramp-like approximation with many steps
            steps = [(i * u.pA, 10 * u.ms) for i in range(11)]
            current, duration = constant(steps)
            self.assertEqual(current.shape[0], 1100)
            self.assertAlmostEqual(u.get_magnitude(duration), 110.0)

    def test_step(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration = 500 * u.ms
            # Test step function with multiple levels
            amplitudes = [0.0, 1.0, 0.5, 2.0]
            step_times = [0 * u.ms, 100 * u.ms, 250 * u.ms, 400 * u.ms]

            current = step(amplitudes, step_times, duration)
            show(current, duration / u.ms, 'Step Input: Multiple Levels')
            self.assertEqual(current.shape[0], 5000)

    def test_step_unsorted(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Test that step function works with unsorted times
            duration = 300 * u.ms
            amplitudes = [1.0, 0.5, 2.0]
            step_times = [100 * u.ms, 0 * u.ms, 200 * u.ms]  # Unsorted

            current = step(amplitudes, step_times, duration)
            # Should automatically sort and produce correct output
            self.assertEqual(current.shape[0], 3000)

    def test_step_examples(self):
        """Test step with examples from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple three-level step function
            current = step(
                amplitudes=[0, 10, 5] * u.pA,
                step_times=[0, 50, 150] * u.ms,
                duration=200 * u.ms
            )
            self.assertEqual(current.shape[0], 2000)

            # Staircase protocol
            amplitudes = [0, 2, 4, 6, 8, 10] * u.nA
            times = [0, 20, 40, 60, 80, 100] * u.ms
            current = step(amplitudes, times, 120 * u.ms)
            self.assertEqual(current.shape[0], 1200)

            # Multiple pulses with return to baseline
            current = step(
                amplitudes=[0, 5, 0, 10, 0] * u.pA,
                step_times=[0, 20, 40, 60, 80] * u.ms,
                duration=100 * u.ms
            )
            self.assertEqual(current.shape[0], 1000)

            # Unsorted times are automatically sorted
            current = step(
                amplitudes=[5, 0, 10] * u.pA,
                step_times=[50, 0, 100] * u.ms,  # Will be sorted to [0, 50, 100]
                duration=150 * u.ms
            )
            self.assertEqual(current.shape[0], 1500)

            # Protocol with negative values
            current = step(
                amplitudes=[-5, 0, 5, 0, -5] * u.pA,
                step_times=[0, 25, 50, 75, 100] * u.ms,
                duration=125 * u.ms
            )
            self.assertEqual(current.shape[0], 1250)

            # F-I curve protocol
            amplitudes = np.linspace(0, 50, 11) * u.pA
            times = np.linspace(0, 1000, 11) * u.ms
            current = step(amplitudes, times, 1100 * u.ms)
            self.assertEqual(current.shape[0], 11000)

    def test_ramp(self):
        with brainstate.environ.context(dt=0.1):
            duration = 500
            current4 = ramp(0, 1, duration)

            show(current4, duration, r'$c_{start}$=0, $c_{end}$=%d, duration, '
                                     r'$t_{start}$=0, $t_{end}$=None' % duration)
            self.assertEqual(current4.shape[0], 5000)

    def test_ramp_with_times(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            duration, t_start, t_end = 500 * u.ms, 100 * u.ms, 400 * u.ms
            current5 = ramp(0, 1, duration, t_start, t_end)

            show(current5, duration)
            self.assertEqual(current5.shape[0], 5000)

    def test_ramp_examples(self):
        """Test ramp with examples from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Simple linear ramp from 0 to 10 pA over 100 ms
            current = ramp(
                c_start=0 * u.pA,
                c_end=10 * u.pA,
                duration=100 * u.ms
            )
            self.assertEqual(current.shape[0], 1000)
            # Check that it ramps up linearly
            self.assertAlmostEqual(u.get_magnitude(current[0]), 0.0, places=5)
            self.assertTrue(u.get_magnitude(current[-1]) > 9.9)  # Close to 10

            # Decreasing ramp (10 to 0 pA)
            current = ramp(
                c_start=10 * u.pA,
                c_end=0 * u.pA,
                duration=100 * u.ms
            )
            self.assertEqual(current.shape[0], 1000)
            # Check that it ramps down
            self.assertTrue(u.get_magnitude(current[500]) < u.get_magnitude(current[400]))

            # Ramp with delay and early stop
            current = ramp(
                c_start=0 * u.nA,
                c_end=5 * u.nA,
                duration=200 * u.ms,
                t_start=50 * u.ms,  # Start ramping at 50 ms
                t_end=150 * u.ms  # Stop ramping at 150 ms
            )
            self.assertEqual(current.shape[0], 2000)
            # Check that ramp starts after t_start
            self.assertAlmostEqual(u.get_magnitude(current[0]), 0.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(current[499]), 0.0, places=5)
            # Check that there's a ramp between t_start and t_end
            self.assertTrue(u.get_magnitude(current[1000]) > 0)

            # Negative to positive ramp
            current = ramp(
                c_start=-5 * u.pA,
                c_end=5 * u.pA,
                duration=100 * u.ms
            )
            self.assertEqual(current.shape[0], 1000)
            # Check that it crosses zero
            self.assertTrue(u.get_magnitude(current[0]) < 0)
            self.assertTrue(u.get_magnitude(current[-1]) > 0)

            # Slow ramp for adaptation studies
            current = ramp(
                c_start=0 * u.pA,
                c_end=20 * u.pA,
                duration=1000 * u.ms,
                t_start=100 * u.ms,
                t_end=900 * u.ms
            )
            self.assertEqual(current.shape[0], 10000)

            # Ramp for I-V curve measurements
            current = ramp(
                c_start=-100 * u.pA,
                c_end=100 * u.pA,
                duration=500 * u.ms
            )
            self.assertEqual(current.shape[0], 5000)
            # Check range
            self.assertTrue(u.get_magnitude(current[0]) < -99)
            self.assertTrue(u.get_magnitude(current[-1]) > 99)

            # Sawtooth wave component
            current = ramp(
                c_start=0 * u.pA,
                c_end=10 * u.pA,
                duration=10 * u.ms,
                t_start=1 * u.ms,
                t_end=9 * u.ms
            )
            self.assertEqual(current.shape[0], 100)
