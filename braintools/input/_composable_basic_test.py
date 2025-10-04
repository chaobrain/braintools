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

"""Tests for composable basic input generators and their docstring examples."""

from unittest import TestCase

import brainstate
import brainunit as u
import numpy as np

from braintools.input import (
    Section, Constant, Step, Ramp,
    WienerProcess, Sinusoidal
)


class TestSection(TestCase):
    """Test Section class and its docstring examples."""

    def test_simple_three_phase_protocol(self):
        """Test simple three-phase protocol from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            section = Section(
                values=[0, 1, 0] * u.pA,
                durations=[100, 300, 100] * u.ms
            )
            array = section()
            self.assertEqual(array.shape[0], 5000)

            # Check values in each section
            self.assertAlmostEqual(u.get_magnitude(array[500]), 0, places=5)  # First section
            self.assertAlmostEqual(u.get_magnitude(array[2000]), 1, places=5)  # Second section
            self.assertAlmostEqual(u.get_magnitude(array[4500]), 0, places=5)  # Third section

    def test_multi_channel_input(self):
        """Test multi-channel input from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            values = [np.zeros(3), np.ones(3) * 5, np.zeros(3)] * u.nA
            section = Section(
                values=values,
                durations=[50, 100, 50] * u.ms
            )
            array = section()
            self.assertEqual(array.shape, (2000, 3))

            # Check multi-channel values
            np.testing.assert_allclose(u.get_magnitude(array[250]), [0, 0, 0], rtol=1e-5)
            np.testing.assert_allclose(u.get_magnitude(array[1000]), [5, 5, 5], rtol=1e-5)

    def test_combine_with_noise(self):
        """Test combining with noise from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            section = Section(
                values=[0, 1, 0],
                durations=[100, 300, 100] * u.ms
            )
            noisy_section = section + WienerProcess(500 * u.ms, sigma=0.1, seed=42)

            array = noisy_section()
            self.assertEqual(array.shape[0], 5000)

            # Should have noise added
            self.assertFalse(np.all(array[1000:1100] == u.get_magnitude(array[1000])))

    def test_modulation_with_sinusoid(self):
        """Test modulation with sinusoid from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            section = Section(
                values=[0, 1, 0],
                durations=[100, 300, 100] * u.ms
            )
            sine = Sinusoidal(0.2, 10 * u.Hz, 500 * u.ms)
            modulated = section * (1 + sine)

            array = modulated()
            self.assertEqual(array.shape[0], 5000)

    def test_smooth_protocol(self):
        """Test smooth protocol from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            protocol = Section(
                values=[0, 0.5, 1.0, 1.5, 1.0, 0.5, 0],
                durations=[50, 30, 100, 150, 100, 30, 50] * u.ms
            )
            smooth_protocol = protocol.smooth(tau=10 * u.ms)

            array = smooth_protocol()
            self.assertEqual(array.shape[0], 5100)

            # Should be smoothed (no sharp transitions)
            diff = np.diff(u.get_magnitude(array))
            self.assertTrue(np.max(np.abs(diff)) < 0.5)  # No large jumps

    def test_sequential_composition(self):
        """Test sequential composition from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            baseline = Section([0], [200] * u.ms)
            stim = Section([0.5, 1.0, 0.5], [50, 100, 50] * u.ms)
            recovery = Section([0], [200] * u.ms)
            full_protocol = baseline & stim & recovery

            array = full_protocol()
            self.assertEqual(array.shape[0], 6000)  # 200 + 200 + 200 = 600ms

    def test_values_durations_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        with self.assertRaises(ValueError):
            Section(
                values=[0, 1, 0],
                durations=[100, 200]  # Wrong length
            )


class TestConstant(TestCase):
    """Test Constant class and its docstring examples."""

    def test_simple_two_phase_protocol(self):
        """Test simple two-phase protocol from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            const = Constant([
                (0 * u.pA, 100 * u.ms),
                (10 * u.pA, 200 * u.ms)
            ])
            array = const()
            self.assertEqual(array.shape[0], 3000)

            # Check values
            self.assertAlmostEqual(u.get_magnitude(array[500]), 0, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1500]), 10, places=5)

    def test_multi_step_injection(self):
        """Test multi-step current injection from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            steps = Constant([
                (0 * u.nA, 50 * u.ms),
                (0.5 * u.nA, 50 * u.ms),
                (1.0 * u.nA, 50 * u.ms),
                (1.5 * u.nA, 50 * u.ms),
                (0 * u.nA, 50 * u.ms),
            ])
            array = steps()
            self.assertEqual(array.shape[0], 2500)

            # Check step values
            self.assertAlmostEqual(u.get_magnitude(array[250]), 0, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[750]), 0.5, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1250]), 1.0, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1750]), 1.5, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[2250]), 0, places=5)

    def test_smooth_transitions(self):
        """Test smooth transitions from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            const = Constant([
                (0, 100 * u.ms),
                (1, 100 * u.ms),
                (0.5, 100 * u.ms),
                (0, 100 * u.ms)
            ])
            smoothed = const.smooth(tau=20 * u.ms)

            array = smoothed()
            self.assertEqual(array.shape[0], 4000)

            # Should have smooth transitions
            diff = np.diff(u.get_magnitude(array))
            self.assertTrue(np.max(np.abs(diff)) < 0.2)

    def test_combine_with_oscillations(self):
        """Test combining with oscillations from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            baseline = Constant([(0.5, 500 * u.ms)])
            oscillation = Sinusoidal(0.2, 5 * u.Hz, 500 * u.ms)
            combined = baseline + oscillation

            array = combined()
            self.assertEqual(array.shape[0], 5000)

            # Mean should be around 0.5 (the baseline)
            self.assertAlmostEqual(np.mean(u.get_magnitude(array)), 0.5, delta=0.05)

    def test_paired_pulse_protocol(self):
        """Test paired-pulse protocol from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            protocol = Constant([
                (0 * u.pA, 100 * u.ms),  # baseline
                (5 * u.pA, 20 * u.ms),  # first pulse
                (0 * u.pA, 50 * u.ms),  # inter-pulse interval
                (5 * u.pA, 20 * u.ms),  # second pulse
                (0 * u.pA, 100 * u.ms),  # recovery
            ])

            array = protocol()
            self.assertEqual(array.shape[0], 2900)

            # Check pulse values
            self.assertAlmostEqual(u.get_magnitude(array[500]), 0, places=5)  # baseline
            self.assertAlmostEqual(u.get_magnitude(array[1100]), 5, places=5)  # first pulse
            self.assertAlmostEqual(u.get_magnitude(array[1400]), 0, places=5)  # interval
            self.assertAlmostEqual(u.get_magnitude(array[1800]), 5, places=5)  # second pulse
            self.assertAlmostEqual(u.get_magnitude(array[2500]), 0, places=5)  # recovery

    def test_transformations(self):
        """Test transformations from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            const = Constant([(1, 100 * u.ms), (2, 100 * u.ms)])

            # Scale amplitude
            scaled = const.scale(0.5)
            array = scaled()
            self.assertAlmostEqual(np.max(u.get_magnitude(array)), 1.0, places=5)

            # Clip to range
            clipped = const.clip(0, 1.5)
            array = clipped()
            self.assertTrue(np.all(u.get_magnitude(array) <= 1.5))

            # Repeat pattern
            repeated = const.repeat(3)
            array = repeated()
            self.assertEqual(array.shape[0], 6000)  # 200ms * 3


class TestStep(TestCase):
    """Test Step class and its docstring examples."""

    def test_simple_three_level_step(self):
        """Test simple three-level step function from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            steps = Step(
                amplitudes=[0, 10, 5] * u.pA,
                step_times=[0, 50, 150] * u.ms,
                duration=200 * u.ms
            )
            array = steps()
            self.assertEqual(array.shape[0], 2000)

            # Check step values
            self.assertAlmostEqual(u.get_magnitude(array[250]), 0, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1000]), 10, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1750]), 5, places=5)

    def test_staircase_protocol(self):
        """Test staircase protocol for I-V curve from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            amplitudes = np.arange(0, 101, 10) * u.pA
            times = np.arange(0, 1100, 100) * u.ms
            staircase = Step(amplitudes, times, 1200 * u.ms)

            array = staircase()
            self.assertEqual(array.shape[0], 12000)

            # Check some step values
            self.assertAlmostEqual(u.get_magnitude(array[500]), 0, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1500]), 10, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[5500]), 50, places=5)

    def test_multiple_pulses(self):
        """Test multiple pulses with return to baseline from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulses = Step(
                amplitudes=[0, 5, 0, 10, 0, 15, 0] * u.pA,
                step_times=[0, 20, 40, 60, 80, 100, 120] * u.ms,
                duration=150 * u.ms
            )

            array = pulses()
            self.assertEqual(array.shape[0], 1500)

            # Check pulse amplitudes
            self.assertAlmostEqual(u.get_magnitude(array[300]), 5, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[700]), 10, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1100]), 15, places=5)

    def test_combine_with_noise(self):
        """Test combining with noise from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            steps = Step([0, 1, 0.5], [0, 100, 200] * u.ms, 300 * u.ms)
            noise = WienerProcess(300 * u.ms, sigma=0.1, seed=123)
            noisy_steps = steps + noise

            array = noisy_steps()
            self.assertEqual(array.shape[0], 3000)

            # Should have noise added (variance should be higher)
            clean = steps()
            self.assertTrue(np.var(u.get_magnitude(array)) > np.var(u.get_magnitude(clean)))

    def test_smooth_steps(self):
        """Test smoothed steps for gradual transitions from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            sharp_steps = Step(
                [0, 1, 0.5, 1, 0],
                [0, 50, 100, 150, 200] * u.ms,
                250 * u.ms
            )
            smooth_steps = sharp_steps.smooth(tau=10 * u.ms)

            sharp_array = sharp_steps()
            smooth_array = smooth_steps()

            self.assertEqual(smooth_array.shape[0], 2500)

            # Smooth should have smaller max derivative
            sharp_diff = np.max(np.abs(np.diff(u.get_magnitude(sharp_array))))
            smooth_diff = np.max(np.abs(np.diff(u.get_magnitude(smooth_array))))
            self.assertTrue(smooth_diff < sharp_diff)

    def test_clipped_steps(self):
        """Test clipping to physiological range from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            steps = Step(
                [0, 1, 0.5, 1, 0],
                [0, 50, 100, 150, 200] * u.ms,
                250 * u.ms
            )
            clipped = steps.clip(0, 0.8)

            array = clipped()
            self.assertTrue(np.all(u.get_magnitude(array) <= 0.8))
            self.assertTrue(np.all(u.get_magnitude(array) >= 0))

    def test_unsorted_times(self):
        """Test unsorted times are automatically handled from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            steps = Step(
                amplitudes=[5, 0, 10] * u.pA,
                step_times=[50, 0, 100] * u.ms,
                duration=150 * u.ms
            )

            array = steps()
            self.assertEqual(array.shape[0], 1500)

            # Should be sorted: 0->0pA, 50->5pA, 100->10pA
            self.assertAlmostEqual(u.get_magnitude(array[250]), 0, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[750]), 5, places=5)
            self.assertAlmostEqual(u.get_magnitude(array[1250]), 10, places=5)

    def test_sequential_composition(self):
        """Test sequential composition from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            baseline = Step([0], [0 * u.ms], 100 * u.ms)
            test = Step([0, 1, 0], [0, 20, 80] * u.ms, 100 * u.ms)
            protocol = baseline & test & baseline

            array = protocol()
            self.assertEqual(array.shape[0], 3000)  # 100 + 100 + 100 = 300ms


class TestRamp(TestCase):
    """Test Ramp class and its docstring examples."""

    def test_simple_linear_ramp(self):
        """Test simple linear ramp from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(
                c_start=0 * u.pA,
                c_end=10 * u.pA,
                duration=100 * u.ms
            )
            array = ramp()
            self.assertEqual(array.shape[0], 1000)

            # Check linear progression
            self.assertAlmostEqual(u.get_magnitude(array[0]), 0, places=3)
            self.assertTrue(u.get_magnitude(array[-1]) > 9.9)  # Close to 10

            # Should be approximately linear
            mid_expected = 5.0
            mid_actual = u.get_magnitude(array[500])
            self.assertAlmostEqual(mid_actual, mid_expected, delta=0.1)

    def test_decreasing_ramp(self):
        """Test decreasing ramp from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            down_ramp = Ramp(
                c_start=10 * u.pA,
                c_end=0 * u.pA,
                duration=100 * u.ms
            )
            array = down_ramp()

            # Should decrease
            self.assertTrue(u.get_magnitude(array[200]) > u.get_magnitude(array[800]))
            self.assertTrue(u.get_magnitude(array[0]) > 9.9)
            self.assertAlmostEqual(u.get_magnitude(array[-1]), 0, places=3)

    def test_ramp_with_delay(self):
        """Test ramp with delay and early stop from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            delayed_ramp = Ramp(
                c_start=0 * u.nA,
                c_end=5 * u.nA,
                duration=200 * u.ms,
                t_start=50 * u.ms,
                t_end=150 * u.ms
            )

            array = delayed_ramp()
            self.assertEqual(array.shape[0], 2000)

            # Before t_start should be 0 (not c_start according to functional API)
            self.assertAlmostEqual(u.get_magnitude(array[250]), 0, places=5)

            # After t_end should remain at 0 (according to functional API behavior)
            self.assertAlmostEqual(u.get_magnitude(array[1750]), 0, places=5)

            # Middle of ramp should be intermediate
            mid_idx = 1000  # t=100ms
            self.assertTrue(0 < u.get_magnitude(array[mid_idx]) <= 5)

    def test_amplitude_modulation(self):
        """Test amplitude modulation from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            envelope = Ramp(0, 1, 500 * u.ms)
            carrier = Sinusoidal(1.0, 20 * u.Hz, 500 * u.ms)
            am_signal = envelope * carrier

            array = am_signal()
            self.assertEqual(array.shape[0], 5000)

            # Amplitude should increase over time
            early_max = np.max(np.abs(u.get_magnitude(array[:500])))
            late_max = np.max(np.abs(u.get_magnitude(array[-500:])))
            self.assertTrue(late_max > early_max)

    def test_sawtooth_by_repeating(self):
        """Test creating sawtooth wave by repeating from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            single_tooth = Ramp(0, 1, 50 * u.ms)
            sawtooth = single_tooth.repeat(10)

            array = sawtooth()
            self.assertEqual(array.shape[0], 5000)  # 50ms * 10

            # Check periodicity
            first_period = array[:500]
            second_period = array[500:1000]
            np.testing.assert_allclose(
                u.get_magnitude(first_period),
                u.get_magnitude(second_period),
                rtol=1e-5
            )

    def test_ramp_with_saturation(self):
        """Test ramp with saturation from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(-2, 2, 400 * u.ms)
            saturated = ramp.clip(-1, 1)

            array = saturated()
            self.assertEqual(array.shape[0], 4000)

            # Should be clipped
            self.assertTrue(np.all(u.get_magnitude(array) >= -1.01))
            self.assertTrue(np.all(u.get_magnitude(array) <= 1.01))

    def test_smooth_ramp(self):
        """Test smoothed ramp from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(-2, 2, 400 * u.ms)
            smooth_ramp = ramp.smooth(tau=5 * u.ms)

            array = smooth_ramp()
            self.assertEqual(array.shape[0], 4000)

    def test_iv_curve_protocol(self):
        """Test I-V curve measurement protocol from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            iv_ramp = Ramp(
                c_start=-100 * u.pA,
                c_end=100 * u.pA,
                duration=1000 * u.ms
            )
            wobble = Sinusoidal(5 * u.pA, 100 * u.Hz, 1000 * u.ms)
            iv_protocol = iv_ramp + wobble

            array = iv_protocol()
            self.assertEqual(array.shape[0], 10000)

            # Should have both ramp and oscillation
            # Mean should increase linearly
            early_mean = np.mean(u.get_magnitude(array[:1000]))
            late_mean = np.mean(u.get_magnitude(array[-1000:]))
            self.assertTrue(late_mean > early_mean)

    def test_sequential_ramps(self):
        """Test sequential ramps for plasticity protocols from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            up_ramp = Ramp(0, 1, 100 * u.ms)
            plateau = Constant([(1, 50 * u.ms)])
            down_ramp = Ramp(1, 0, 100 * u.ms)
            protocol = up_ramp & plateau & down_ramp

            array = protocol()
            self.assertEqual(array.shape[0], 2500)  # 100 + 50 + 100 = 250ms

            # Check shape: should go up, stay, then down
            self.assertTrue(u.get_magnitude(array[500]) < u.get_magnitude(array[900]))  # Rising
            self.assertAlmostEqual(u.get_magnitude(array[1200]), u.get_magnitude(array[1400]), places=3)  # Plateau
            self.assertTrue(u.get_magnitude(array[1700]) > u.get_magnitude(array[2300]))  # Falling


class TestIntegration(TestCase):
    """Test integration between different composable basic inputs."""

    def test_mixed_composition(self):
        """Test mixing different input types."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create a complex protocol
            baseline = Constant([(0, 100 * u.ms)])
            ramp_up = Ramp(0, 1, 100 * u.ms)
            steps = Step([1, 0.5, 1, 0], [0, 50, 100, 150] * u.ms, 200 * u.ms)
            sections = Section([0.5, 0], [50, 50] * u.ms)

            protocol = baseline & ramp_up & steps & sections

            array = protocol()
            self.assertEqual(array.shape[0], 5000)  # 100 + 100 + 200 + 100 = 500ms

    def test_all_transformations(self):
        """Test that all input types support standard transformations."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            inputs = [
                Section([0, 1, 0], [50, 100, 50] * u.ms),
                Constant([(0.5, 200 * u.ms)]),
                Step([0, 1, 0.5], [0, 50, 150] * u.ms, 200 * u.ms),
                Ramp(0, 1, 200 * u.ms)
            ]

            for inp in inputs:
                # All should support these operations
                scaled = inp.scale(0.5)
                clipped = inp.clip(0, 0.8)
                smoothed = inp.smooth(tau=10 * u.ms)
                shifted = inp.shift(20 * u.ms)
                repeated = inp.repeat(2)

                # All should generate arrays
                self.assertIsNotNone(scaled())
                self.assertIsNotNone(clipped())
                self.assertIsNotNone(smoothed())
                self.assertIsNotNone(shifted())
                self.assertIsNotNone(repeated())
