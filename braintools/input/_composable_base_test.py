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

"""
Tests for composable base classes and their docstring examples.
"""

from unittest import TestCase

import brainstate
import brainunit as u
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from braintools.input import (
    Composite, ConstantValue, Sequential,
    TimeShifted, Clipped, Smoothed, Repeated,
    Transformed
)
from braintools.input import Ramp, Step
from braintools.input import Sinusoidal
from braintools.input import WienerProcess


class TestInputBaseClass(TestCase):
    """Test the Input base class functionality and examples."""

    def test_basic_arithmetic_operations(self):
        """Test basic arithmetic operations from docstring examples."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Add two inputs
            ramp = Ramp(0, 1, 500 * u.ms)
            sine = Sinusoidal(0.5, 10 * u.Hz, 500 * u.ms)
            combined = ramp + sine
            self.assertIsInstance(combined, Composite)
            self.assertEqual(combined().shape[0], 5000)

            # Scale an input
            scaled_ramp = ramp * 2.0
            self.assertIsInstance(scaled_ramp, Composite)
            half_sine = sine.scale(0.5)
            self.assertIsInstance(half_sine, Composite)

            # Subtract a baseline
            centered = sine - 0.25
            self.assertIsInstance(centered, Composite)
            arr = centered()
            # Check that mean is approximately -0.25 (since sine has mean 0)
            self.assertAlmostEqual(u.get_magnitude(arr).mean(), -0.25, places=2)

    def test_complex_compositions(self):
        """Test complex composition examples from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Amplitude modulation
            carrier = Sinusoidal(1.0, 100 * u.Hz, 1000 * u.ms)
            envelope = Ramp(0, 1, 1000 * u.ms)
            am_signal = carrier * envelope
            self.assertIsInstance(am_signal, Composite)
            self.assertEqual(am_signal().shape[0], 10000)

            # Sequential stimulation protocol
            baseline = Step([0], [0 * u.ms], 200 * u.ms)
            stim = Step([1], [0 * u.ms], 500 * u.ms)
            recovery = Step([0], [0 * u.ms], 300 * u.ms)
            protocol = baseline & stim & recovery
            self.assertIsInstance(protocol, Sequential)
            # Total duration should be 1000ms = 10000 steps
            self.assertEqual(protocol().shape[0], 10000)

            # Overlay (maximum) for redundant stimulation
            stim1 = Step([0, 1, 0], [0, 100, 400] * u.ms, 500 * u.ms)
            stim2 = Step([0, 0.8, 0], [0, 200, 450] * u.ms, 500 * u.ms)
            combined_stim = stim1 | stim2
            self.assertIsInstance(combined_stim, Composite)
            self.assertEqual(combined_stim().shape[0], 5000)

    def test_transformations(self):
        """Test transformation examples from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            sine = Sinusoidal(0.5, 10 * u.Hz, 500 * u.ms)
            ramp = Ramp(0, 1, 500 * u.ms)

            # Time shifting
            delayed_sine = sine.shift(50 * u.ms)
            self.assertIsInstance(delayed_sine, TimeShifted)
            self.assertEqual(delayed_sine().shape[0], 5000)

            advanced_ramp = ramp.shift(-20 * u.ms)
            self.assertIsInstance(advanced_ramp, TimeShifted)

            # Clipping
            clipped = (ramp * 2).clip(0, 1.5)
            self.assertIsInstance(clipped, Clipped)
            arr = clipped()
            self.assertTrue(np.all(u.get_magnitude(arr) <= 1.5))
            self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

            # Smoothing
            smooth_steps = Step([0, 1, 0.5, 1, 0],
                                [0, 100, 200, 300, 400] * u.ms,
                                500 * u.ms).smooth(10 * u.ms)
            self.assertIsInstance(smooth_steps, Smoothed)

            # Repeating
            burst = Step([0, 1, 0], [0, 10, 20] * u.ms, 50 * u.ms)
            repeated_bursts = burst.repeat(10)
            self.assertIsInstance(repeated_bursts, Repeated)
            self.assertEqual(repeated_bursts().shape[0], 5000)  # 50ms * 10 = 500ms

            # Custom transformations
            rectified = sine.apply(lambda x: jnp.maximum(x, 0))
            self.assertIsInstance(rectified, Transformed)
            arr = rectified()
            self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

            squared = sine.apply(lambda x: x ** 2)
            self.assertIsInstance(squared, Transformed)

    def test_advanced_protocols(self):
        """Test advanced protocol examples from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Complex experimental protocol
            pre_baseline = Step([0], [0 * u.ms], 1000 * u.ms)
            conditioning = Sinusoidal(0.5, 5 * u.Hz, 2000 * u.ms)
            test_pulse = Step([2], [0 * u.ms], 100 * u.ms)
            post_baseline = Step([0], [0 * u.ms], 1000 * u.ms)

            protocol = (pre_baseline &
                        (conditioning + 0.5).clip(0, 1) &
                        test_pulse &
                        post_baseline)

            self.assertIsInstance(protocol, Sequential)
            # Total duration: 1000 + 2000 + 100 + 1000 = 4100ms = 41000 steps
            self.assertEqual(protocol().shape[0], 41000)

            # Noisy modulated signal
            signal = Sinusoidal(1.0, 20 * u.Hz, 1000 * u.ms)
            noise = WienerProcess(1000 * u.ms, sigma=0.1, seed=42)
            modulator = (Ramp(0.5, 1.5, 1000 * u.ms) +
                         Sinusoidal(0.2, 2 * u.Hz, 1000 * u.ms))
            noisy_modulated = (signal + noise) * modulator

            self.assertIsInstance(noisy_modulated, Composite)
            self.assertEqual(noisy_modulated().shape[0], 10000)

    def test_call_method(self):
        """Test __call__ method examples from docstring."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(0, 1, 100 * u.ms)

            # First call generates and caches
            arr1 = ramp()
            self.assertEqual(arr1.shape[0], 1000)

            # Second call uses cache (should be same object)
            arr2 = ramp()
            self.assertTrue(u.math.allclose(arr1, arr2))

            # Force regeneration
            arr3 = ramp(recompute=True)
            self.assertTrue(u.math.allclose(arr1, arr3))

    def test_operator_examples(self):
        """Test individual operator examples from docstrings."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Addition
            sine1 = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
            sine2 = Sinusoidal(0.5, 20 * u.Hz, 100 * u.ms)
            combined = sine1 + sine2
            self.assertIsInstance(combined, Composite)
            with_offset = sine1 + 0.5
            self.assertIsInstance(with_offset, Composite)

            # Subtraction
            ramp = Ramp(0, 2, 100 * u.ms)
            baseline = Step([0.5], [0 * u.ms], 100 * u.ms)
            corrected = ramp - baseline
            self.assertIsInstance(corrected, Composite)
            centered = ramp - 1.0
            self.assertIsInstance(centered, Composite)

            # Multiplication
            carrier = Sinusoidal(1.0, 100 * u.Hz, 500 * u.ms)
            envelope = Ramp(0, 1, 500 * u.ms)
            am_signal = carrier * envelope
            self.assertIsInstance(am_signal, Composite)
            doubled = carrier * 2.0
            self.assertIsInstance(doubled, Composite)

            # Division
            signal = Sinusoidal(2.0, 10 * u.Hz, 100 * u.ms)
            normalizer = Ramp(1, 2, 100 * u.ms)
            normalized = signal / normalizer
            self.assertIsInstance(normalized, Composite)
            halved = signal / 2.0
            self.assertIsInstance(halved, Composite)

            # Sequential composition (&)
            baseline = Step([0], [0 * u.ms], 100 * u.ms)
            stimulus = Step([1], [0 * u.ms], 200 * u.ms)
            recovery = Step([0], [0 * u.ms], 100 * u.ms)
            protocol = baseline & stimulus & recovery
            self.assertIsInstance(protocol, Sequential)
            self.assertEqual(protocol().shape[0], 4000)  # 400ms total

            # Overlay (|)
            stim1 = Step([0, 1, 0], [0, 100, 300] * u.ms, 400 * u.ms)
            stim2 = Step([0, 0.8, 0], [0, 150, 350] * u.ms, 400 * u.ms)
            combined_overlay = stim1 | stim2
            self.assertIsInstance(combined_overlay, Composite)

            # Negation
            sine = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
            inverted = -sine
            self.assertIsInstance(inverted, Transformed)

    def test_method_examples(self):
        """Test individual method examples from docstrings."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Scale
            ramp = Ramp(0, 1, 100 * u.ms)
            doubled = ramp.scale(2.0)
            self.assertIsInstance(doubled, Composite)
            reduced = ramp.scale(0.3)
            self.assertIsInstance(reduced, Composite)

            # Shift
            pulse = Step([1], [100 * u.ms], 200 * u.ms)
            delayed = pulse.shift(50 * u.ms)
            self.assertIsInstance(delayed, TimeShifted)
            advanced = pulse.shift(-30 * u.ms)
            self.assertIsInstance(advanced, TimeShifted)

            # Clip
            ramp = Ramp(-2, 2, 100 * u.ms)
            saturated = ramp.clip(0, 1)
            self.assertIsInstance(saturated, Clipped)
            capped = ramp.clip(max_val=1.5)
            self.assertIsInstance(capped, Clipped)
            rectified = ramp.clip(min_val=0)
            self.assertIsInstance(rectified, Clipped)

            # Smooth
            steps = Step([0, 1, 0.5, 1, 0],
                         [0, 50, 100, 150, 200] * u.ms,
                         250 * u.ms)
            smooth = steps.smooth(10 * u.ms)
            self.assertIsInstance(smooth, Smoothed)
            very_smooth = steps.smooth(50 * u.ms)
            self.assertIsInstance(very_smooth, Smoothed)

            # Repeat
            burst = Step([0, 1, 0], [0, 10, 20] * u.ms, 50 * u.ms)
            burst_train = burst.repeat(10)
            self.assertIsInstance(burst_train, Repeated)

            packet = Sinusoidal(1.0, 50 * u.Hz, 100 * u.ms)
            packets = packet.repeat(5)
            self.assertIsInstance(packets, Repeated)
            self.assertEqual(packets().shape[0], 5000)  # 500ms total

            # Apply
            sine = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)

            rectified = sine.apply(lambda x: jnp.maximum(x, 0))
            self.assertIsInstance(rectified, Transformed)

            squared = sine.apply(lambda x: x ** 2)
            self.assertIsInstance(squared, Transformed)

            sigmoid = sine.apply(lambda x: 1 / (1 + jnp.exp(-5 * x)))
            self.assertIsInstance(sigmoid, Transformed)

            # Noise addition example
            key = jrandom.PRNGKey(0)
            noisy = sine.apply(
                lambda x: x + 0.1 * jrandom.normal(key, x.shape)
            )
            self.assertIsInstance(noisy, Transformed)


class TestComposite(TestCase):
    """Test Composite class and its docstring examples."""

    def test_direct_construction(self):
        """Test direct construction examples."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(0, 1, 100 * u.ms)
            sine = Sinusoidal(0.5, 10 * u.Hz, 100 * u.ms)

            # Direct construction
            added = Composite(ramp, sine, '+')
            self.assertEqual(added().shape[0], 1000)

            # Via operators (more common)
            added_op = ramp + sine
            multiplied = ramp * sine
            maximum = ramp | sine  # Uses 'max' operator

            self.assertIsInstance(added_op, Composite)
            self.assertIsInstance(multiplied, Composite)
            self.assertIsInstance(maximum, Composite)

    def test_padding_behavior(self):
        """Test that shorter inputs are padded with zeros."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            short_input = Step([1], [0 * u.ms], 100 * u.ms)
            long_input = Step([0.5], [0 * u.ms], 200 * u.ms)

            combined = short_input + long_input
            # Should have duration of longer input
            self.assertEqual(combined().shape[0], 2000)

    def test_division_by_zero(self):
        """Test that division by zero returns numerator."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            numerator = Step([1], [0 * u.ms], 100 * u.ms)
            denominator = Step([0], [0 * u.ms], 100 * u.ms)

            result = numerator / denominator
            arr = result()
            # Should return numerator value when denominator is zero
            self.assertTrue(np.all(u.get_magnitude(arr) == 1.0))


class TestConstantValue(TestCase):
    """Test ConstantValue class and its docstring examples."""

    def test_implicit_creation(self):
        """Test implicit creation via operators."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            sine = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
            with_offset = sine + 0.5

            # Should create ConstantValue internally
            self.assertIsInstance(with_offset, Composite)
            self.assertIsInstance(with_offset.input2, ConstantValue)
            self.assertEqual(with_offset.input2.value, 0.5)

    def test_direct_construction(self):
        """Test direct construction."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            baseline = ConstantValue(0.1, 500 * u.ms)
            arr = baseline()
            self.assertEqual(arr.shape[0], 5000)
            self.assertTrue(np.all(u.get_magnitude(arr) == 0.1))


class TestSequential(TestCase):
    """Test Sequential class and its docstring examples."""

    def test_three_phase_protocol(self):
        """Test three-phase protocol example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            baseline = Step([0], [0 * u.ms], 500 * u.ms)
            stimulus = Ramp(0, 1, 1000 * u.ms)
            recovery = Step([0], [0 * u.ms], 500 * u.ms)

            # Chain using & operator
            protocol = baseline & stimulus & recovery
            self.assertIsInstance(protocol, Sequential)
            self.assertEqual(protocol().shape[0], 20000)  # 2000ms total

            # Direct construction
            two_phase = Sequential(baseline, stimulus)
            self.assertIsInstance(two_phase, Sequential)
            self.assertEqual(two_phase().shape[0], 15000)  # 1500ms


class TestTimeShifted(TestCase):
    """Test TimeShifted class and its docstring examples."""

    def test_delay_and_advance(self):
        """Test delay and advance examples."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulse = Step([1], [200 * u.ms], 500 * u.ms)

            # Delay by 100ms
            delayed = TimeShifted(pulse, 100 * u.ms)
            arr_delayed = delayed()
            self.assertEqual(arr_delayed.shape[0], 5000)

            # Advance by 50ms
            advanced = TimeShifted(pulse, -50 * u.ms)
            arr_advanced = advanced()
            self.assertEqual(arr_advanced.shape[0], 5000)

            # Via shift() method
            delayed_method = pulse.shift(100 * u.ms)
            self.assertIsInstance(delayed_method, TimeShifted)


class TestClipped(TestCase):
    """Test Clipped class and its docstring examples."""

    def test_clipping_modes(self):
        """Test different clipping modes."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(-2, 2, 200 * u.ms)

            # Clip to [0, 1] range
            saturated = Clipped(ramp, 0, 1)
            arr = saturated()
            self.assertTrue(np.all(u.get_magnitude(arr) >= 0))
            self.assertTrue(np.all(u.get_magnitude(arr) <= 1))

            # Only lower bound (rectification)
            rectified = Clipped(ramp, min_val=0)
            arr = rectified()
            self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

            # Only upper bound (saturation)
            capped = Clipped(ramp, max_val=1.5)
            arr = capped()
            self.assertTrue(np.all(u.get_magnitude(arr) <= 1.5))

            # Via clip() method
            saturated_method = ramp.clip(0, 1)
            self.assertIsInstance(saturated_method, Clipped)


class TestSmoothed(TestCase):
    """Test Smoothed class and its docstring examples."""

    def test_smoothing_levels(self):
        """Test different smoothing levels."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            steps = Step([0, 1, 0.5, 1, 0],
                         [0, 50, 100, 150, 200] * u.ms,
                         250 * u.ms)

            # Light smoothing (fast response)
            light = Smoothed(steps, 5 * u.ms)
            arr_light = light()
            self.assertEqual(arr_light.shape[0], 2500)

            # Heavy smoothing (slow response)
            heavy = Smoothed(steps, 25 * u.ms)
            arr_heavy = heavy()
            self.assertEqual(arr_heavy.shape[0], 2500)

            # Via smooth() method
            smooth = steps.smooth(10 * u.ms)
            self.assertIsInstance(smooth, Smoothed)

            # Heavy smoothing should have smaller variations
            light_var = np.var(u.get_magnitude(arr_light))
            heavy_var = np.var(u.get_magnitude(arr_heavy))
            self.assertLess(heavy_var, light_var)


class TestRepeated(TestCase):
    """Test Repeated class and its docstring examples."""

    def test_burst_train(self):
        """Test burst train example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Single burst
            burst = Step([0, 1, 0], [0, 10, 30] * u.ms, 50 * u.ms)

            # Burst train (10 bursts, 500ms total)
            train = Repeated(burst, 10)
            arr = train()
            self.assertEqual(arr.shape[0], 5000)  # 50ms * 10 = 500ms

            # Via repeat() method
            train_method = burst.repeat(10)
            self.assertIsInstance(train_method, Repeated)

    def test_oscillation_packets(self):
        """Test oscillation packet example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            packet = Sinusoidal(1.0, 100 * u.Hz, 100 * u.ms)
            packets = Repeated(packet, 5)
            arr = packets()
            self.assertEqual(arr.shape[0], 5000)  # 100ms * 5 = 500ms


class TestTransformed(TestCase):
    """Test Transformed class and its docstring examples."""

    def test_nonlinear_transformations(self):
        """Test various nonlinear transformation examples."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            sine = Sinusoidal(1.0, 10 * u.Hz, 200 * u.ms)

            # Half-wave rectification
            rectified = Transformed(sine, lambda x: jnp.maximum(x, 0))
            arr = rectified()
            self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

            # Squaring (frequency doubling)
            squared = Transformed(sine, lambda x: x ** 2)
            arr = squared()
            self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

            # Sigmoid nonlinearity
            sigmoid = Transformed(sine,
                                  lambda x: 1 / (1 + jnp.exp(-10 * x)))
            arr = sigmoid()
            self.assertTrue(np.all(u.get_magnitude(arr) >= 0))
            self.assertTrue(np.all(u.get_magnitude(arr) <= 1))

            # Via apply() method
            transformed = sine.apply(lambda x: jnp.abs(x))
            self.assertIsInstance(transformed, Transformed)


class TestPropertiesAndAttributes(TestCase):
    """Test properties and attributes of Input classes."""

    def test_dt_property(self):
        """Test dt property retrieval from global environment."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(0, 1, 100 * u.ms)
            self.assertEqual(ramp.dt, brainstate.environ.get_dt())

    def test_n_steps_property(self):
        """Test n_steps calculation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # 100ms duration with dt=0.1ms should give 1000 steps
            ramp = Ramp(0, 1, 100 * u.ms)
            self.assertEqual(ramp.n_steps, 1000)

            # 250ms duration with dt=0.1ms should give 2500 steps
            sine = Sinusoidal(1.0, 10 * u.Hz, 250 * u.ms)
            self.assertEqual(sine.n_steps, 2500)

    def test_shape_property(self):
        """Test shape property."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(0, 1, 100 * u.ms)
            self.assertEqual(ramp.shape, (1000,))

            # For multi-channel inputs (if supported)
            steps = Step([0, 1, 0], [0, 50, 100] * u.ms, 150 * u.ms)
            self.assertEqual(steps.shape, (1500,))


class TestEdgeCases(TestCase):
    """Test edge cases and error handling."""

    def test_concatenation_type_error(self):
        """Test that concatenation with non-Input raises TypeError."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(0, 1, 100 * u.ms)
            with self.assertRaises(TypeError):
                result = ramp & 0.5

    def test_overlay_type_error(self):
        """Test that overlay with non-Input raises TypeError."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(0, 1, 100 * u.ms)
            with self.assertRaises(TypeError):
                result = ramp | 0.5

    def test_unknown_operator(self):
        """Test that unknown operator raises ValueError."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Ramp(0, 1, 100 * u.ms)
            sine = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
            composite = Composite(ramp, sine, 'unknown')
            with self.assertRaises(ValueError):
                arr = composite()

    def test_zero_duration(self):
        """Test handling of zero or very small durations."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # This might create 0 or 1 step depending on implementation
            short = Step([1], [0 * u.ms], 0.01 * u.ms)
            arr = short()
            self.assertGreaterEqual(arr.shape[0], 0)
