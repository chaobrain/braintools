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
    CompositeInput, ConstantValue, SequentialInput,
    TimeShiftedInput, ClippedInput, SmoothedInput, RepeatedInput,
    TransformedInput
)
from braintools.input import RampInput, StepInput
from braintools.input import SinusoidalInput
from braintools.input import WienerProcess

# Set up environment
brainstate.environ.set(dt=0.1 * u.ms)


class TestInputBaseClass(TestCase):
    """Test the Input base class functionality and examples."""

    def test_basic_arithmetic_operations(self):
        """Test basic arithmetic operations from docstring examples."""
        # Add two inputs
        ramp = RampInput(0, 1, 500 * u.ms)
        sine = SinusoidalInput(0.5, 10 * u.Hz, 500 * u.ms)
        combined = ramp + sine
        self.assertIsInstance(combined, CompositeInput)
        self.assertEqual(combined().shape[0], 5000)

        # Scale an input
        scaled_ramp = ramp * 2.0
        self.assertIsInstance(scaled_ramp, CompositeInput)
        half_sine = sine.scale(0.5)
        self.assertIsInstance(half_sine, CompositeInput)

        # Subtract a baseline
        centered = sine - 0.25
        self.assertIsInstance(centered, CompositeInput)
        arr = centered()
        # Check that mean is approximately -0.25 (since sine has mean 0)
        self.assertAlmostEqual(u.get_magnitude(arr).mean(), -0.25, places=2)

    def test_complex_compositions(self):
        """Test complex composition examples from docstring."""
        # Amplitude modulation
        carrier = SinusoidalInput(1.0, 100 * u.Hz, 1000 * u.ms)
        envelope = RampInput(0, 1, 1000 * u.ms)
        am_signal = carrier * envelope
        self.assertIsInstance(am_signal, CompositeInput)
        self.assertEqual(am_signal().shape[0], 10000)

        # Sequential stimulation protocol
        baseline = StepInput([0], [0], 200 * u.ms)
        stim = StepInput([1], [0], 500 * u.ms)
        recovery = StepInput([0], [0], 300 * u.ms)
        protocol = baseline & stim & recovery
        self.assertIsInstance(protocol, SequentialInput)
        # Total duration should be 1000ms = 10000 steps
        self.assertEqual(protocol().shape[0], 10000)

        # Overlay (maximum) for redundant stimulation
        stim1 = StepInput([0, 1, 0], [0, 100, 400], 500 * u.ms)
        stim2 = StepInput([0, 0.8, 0], [0, 200, 450], 500 * u.ms)
        combined_stim = stim1 | stim2
        self.assertIsInstance(combined_stim, CompositeInput)
        self.assertEqual(combined_stim().shape[0], 5000)

    def test_transformations(self):
        """Test transformation examples from docstring."""
        sine = SinusoidalInput(0.5, 10 * u.Hz, 500 * u.ms)
        ramp = RampInput(0, 1, 500 * u.ms)

        # Time shifting
        delayed_sine = sine.shift(50 * u.ms)
        self.assertIsInstance(delayed_sine, TimeShiftedInput)
        self.assertEqual(delayed_sine().shape[0], 5000)

        advanced_ramp = ramp.shift(-20 * u.ms)
        self.assertIsInstance(advanced_ramp, TimeShiftedInput)

        # Clipping
        clipped = (ramp * 2).clip(0, 1.5)
        self.assertIsInstance(clipped, ClippedInput)
        arr = clipped()
        self.assertTrue(np.all(u.get_magnitude(arr) <= 1.5))
        self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

        # Smoothing
        smooth_steps = StepInput([0, 1, 0.5, 1, 0],
                                 [0, 100, 200, 300, 400],
                                 500 * u.ms).smooth(10 * u.ms)
        self.assertIsInstance(smooth_steps, SmoothedInput)

        # Repeating
        burst = StepInput([0, 1, 0], [0, 10, 20], 50 * u.ms)
        repeated_bursts = burst.repeat(10)
        self.assertIsInstance(repeated_bursts, RepeatedInput)
        self.assertEqual(repeated_bursts().shape[0], 5000)  # 50ms * 10 = 500ms

        # Custom transformations
        rectified = sine.apply(lambda x: jnp.maximum(x, 0))
        self.assertIsInstance(rectified, TransformedInput)
        arr = rectified()
        self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

        squared = sine.apply(lambda x: x ** 2)
        self.assertIsInstance(squared, TransformedInput)

    def test_advanced_protocols(self):
        """Test advanced protocol examples from docstring."""
        # Complex experimental protocol
        pre_baseline = StepInput([0], [0], 1000 * u.ms)
        conditioning = SinusoidalInput(0.5, 5 * u.Hz, 2000 * u.ms)
        test_pulse = StepInput([2], [0], 100 * u.ms)
        post_baseline = StepInput([0], [0], 1000 * u.ms)

        protocol = (pre_baseline &
                    (conditioning + 0.5).clip(0, 1) &
                    test_pulse &
                    post_baseline)

        self.assertIsInstance(protocol, SequentialInput)
        # Total duration: 1000 + 2000 + 100 + 1000 = 4100ms = 41000 steps
        self.assertEqual(protocol().shape[0], 41000)

        # Noisy modulated signal
        signal = SinusoidalInput(1.0, 20 * u.Hz, 1000 * u.ms)
        noise = WienerProcess(1000 * u.ms, sigma=0.1, seed=42)
        modulator = (RampInput(0.5, 1.5, 1000 * u.ms) +
                     SinusoidalInput(0.2, 2 * u.Hz, 1000 * u.ms))
        noisy_modulated = (signal + noise) * modulator

        self.assertIsInstance(noisy_modulated, CompositeInput)
        self.assertEqual(noisy_modulated().shape[0], 10000)

    def test_call_method(self):
        """Test __call__ method examples from docstring."""
        ramp = RampInput(0, 1, 100 * u.ms)

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
        # Addition
        sine1 = SinusoidalInput(1.0, 10 * u.Hz, 100 * u.ms)
        sine2 = SinusoidalInput(0.5, 20 * u.Hz, 100 * u.ms)
        combined = sine1 + sine2
        self.assertIsInstance(combined, CompositeInput)
        with_offset = sine1 + 0.5
        self.assertIsInstance(with_offset, CompositeInput)

        # Subtraction
        ramp = RampInput(0, 2, 100 * u.ms)
        baseline = StepInput([0.5], [0], 100 * u.ms)
        corrected = ramp - baseline
        self.assertIsInstance(corrected, CompositeInput)
        centered = ramp - 1.0
        self.assertIsInstance(centered, CompositeInput)

        # Multiplication
        carrier = SinusoidalInput(1.0, 100 * u.Hz, 500 * u.ms)
        envelope = RampInput(0, 1, 500 * u.ms)
        am_signal = carrier * envelope
        self.assertIsInstance(am_signal, CompositeInput)
        doubled = carrier * 2.0
        self.assertIsInstance(doubled, CompositeInput)

        # Division
        signal = SinusoidalInput(2.0, 10 * u.Hz, 100 * u.ms)
        normalizer = RampInput(1, 2, 100 * u.ms)
        normalized = signal / normalizer
        self.assertIsInstance(normalized, CompositeInput)
        halved = signal / 2.0
        self.assertIsInstance(halved, CompositeInput)

        # Sequential composition (&)
        baseline = StepInput([0], [0], 100 * u.ms)
        stimulus = StepInput([1], [0], 200 * u.ms)
        recovery = StepInput([0], [0], 100 * u.ms)
        protocol = baseline & stimulus & recovery
        self.assertIsInstance(protocol, SequentialInput)
        self.assertEqual(protocol().shape[0], 4000)  # 400ms total

        # Overlay (|)
        stim1 = StepInput([0, 1, 0], [0, 100, 300], 400 * u.ms)
        stim2 = StepInput([0, 0.8, 0], [0, 150, 350], 400 * u.ms)
        combined_overlay = stim1 | stim2
        self.assertIsInstance(combined_overlay, CompositeInput)

        # Negation
        sine = SinusoidalInput(1.0, 10 * u.Hz, 100 * u.ms)
        inverted = -sine
        self.assertIsInstance(inverted, TransformedInput)

    def test_method_examples(self):
        """Test individual method examples from docstrings."""
        # Scale
        ramp = RampInput(0, 1, 100 * u.ms)
        doubled = ramp.scale(2.0)
        self.assertIsInstance(doubled, CompositeInput)
        reduced = ramp.scale(0.3)
        self.assertIsInstance(reduced, CompositeInput)

        # Shift
        pulse = StepInput([1], [100 * u.ms], 200 * u.ms)
        delayed = pulse.shift(50 * u.ms)
        self.assertIsInstance(delayed, TimeShiftedInput)
        advanced = pulse.shift(-30 * u.ms)
        self.assertIsInstance(advanced, TimeShiftedInput)

        # Clip
        ramp = RampInput(-2, 2, 100 * u.ms)
        saturated = ramp.clip(0, 1)
        self.assertIsInstance(saturated, ClippedInput)
        capped = ramp.clip(max_val=1.5)
        self.assertIsInstance(capped, ClippedInput)
        rectified = ramp.clip(min_val=0)
        self.assertIsInstance(rectified, ClippedInput)

        # Smooth
        steps = StepInput([0, 1, 0.5, 1, 0],
                          [0, 50, 100, 150, 200],
                          250 * u.ms)
        smooth = steps.smooth(10 * u.ms)
        self.assertIsInstance(smooth, SmoothedInput)
        very_smooth = steps.smooth(50 * u.ms)
        self.assertIsInstance(very_smooth, SmoothedInput)

        # Repeat
        burst = StepInput([0, 1, 0], [0, 10, 20], 50 * u.ms)
        burst_train = burst.repeat(10)
        self.assertIsInstance(burst_train, RepeatedInput)

        packet = SinusoidalInput(1.0, 50 * u.Hz, 100 * u.ms)
        packets = packet.repeat(5)
        self.assertIsInstance(packets, RepeatedInput)
        self.assertEqual(packets().shape[0], 5000)  # 500ms total

        # Apply
        sine = SinusoidalInput(1.0, 10 * u.Hz, 100 * u.ms)

        rectified = sine.apply(lambda x: jnp.maximum(x, 0))
        self.assertIsInstance(rectified, TransformedInput)

        squared = sine.apply(lambda x: x ** 2)
        self.assertIsInstance(squared, TransformedInput)

        sigmoid = sine.apply(lambda x: 1 / (1 + jnp.exp(-5 * x)))
        self.assertIsInstance(sigmoid, TransformedInput)

        # Noise addition example
        key = jrandom.PRNGKey(0)
        noisy = sine.apply(
            lambda x: x + 0.1 * jrandom.normal(key, x.shape)
        )
        self.assertIsInstance(noisy, TransformedInput)


class TestCompositeInput(TestCase):
    """Test CompositeInput class and its docstring examples."""

    def test_direct_construction(self):
        """Test direct construction examples."""
        ramp = RampInput(0, 1, 100 * u.ms)
        sine = SinusoidalInput(0.5, 10 * u.Hz, 100 * u.ms)

        # Direct construction
        added = CompositeInput(ramp, sine, '+')
        self.assertEqual(added().shape[0], 1000)

        # Via operators (more common)
        added_op = ramp + sine
        multiplied = ramp * sine
        maximum = ramp | sine  # Uses 'max' operator

        self.assertIsInstance(added_op, CompositeInput)
        self.assertIsInstance(multiplied, CompositeInput)
        self.assertIsInstance(maximum, CompositeInput)

    def test_padding_behavior(self):
        """Test that shorter inputs are padded with zeros."""
        short_input = StepInput([1], [0], 100 * u.ms)
        long_input = StepInput([0.5], [0], 200 * u.ms)

        combined = short_input + long_input
        # Should have duration of longer input
        self.assertEqual(combined().shape[0], 2000)

    def test_division_by_zero(self):
        """Test that division by zero returns numerator."""
        numerator = StepInput([1], [0], 100 * u.ms)
        denominator = StepInput([0], [0], 100 * u.ms)

        result = numerator / denominator
        arr = result()
        # Should return numerator value when denominator is zero
        self.assertTrue(np.all(u.get_magnitude(arr) == 1.0))


class TestConstantValue(TestCase):
    """Test ConstantValue class and its docstring examples."""

    def test_implicit_creation(self):
        """Test implicit creation via operators."""
        sine = SinusoidalInput(1.0, 10 * u.Hz, 100 * u.ms)
        with_offset = sine + 0.5

        # Should create ConstantValue internally
        self.assertIsInstance(with_offset, CompositeInput)
        self.assertIsInstance(with_offset.input2, ConstantValue)
        self.assertEqual(with_offset.input2.value, 0.5)

    def test_direct_construction(self):
        """Test direct construction."""
        baseline = ConstantValue(0.1, 500 * u.ms)
        arr = baseline()
        self.assertEqual(arr.shape[0], 5000)
        self.assertTrue(np.all(u.get_magnitude(arr) == 0.1))


class TestSequentialInput(TestCase):
    """Test SequentialInput class and its docstring examples."""

    def test_three_phase_protocol(self):
        """Test three-phase protocol example."""
        baseline = StepInput([0], [0], 500 * u.ms)
        stimulus = RampInput(0, 1, 1000 * u.ms)
        recovery = StepInput([0], [0], 500 * u.ms)

        # Chain using & operator
        protocol = baseline & stimulus & recovery
        self.assertIsInstance(protocol, SequentialInput)
        self.assertEqual(protocol().shape[0], 20000)  # 2000ms total

        # Direct construction
        two_phase = SequentialInput(baseline, stimulus)
        self.assertIsInstance(two_phase, SequentialInput)
        self.assertEqual(two_phase().shape[0], 15000)  # 1500ms


class TestTimeShiftedInput(TestCase):
    """Test TimeShiftedInput class and its docstring examples."""

    def test_delay_and_advance(self):
        """Test delay and advance examples."""
        pulse = StepInput([1], [200 * u.ms], 500 * u.ms)

        # Delay by 100ms
        delayed = TimeShiftedInput(pulse, 100 * u.ms)
        arr_delayed = delayed()
        self.assertEqual(arr_delayed.shape[0], 5000)

        # Advance by 50ms
        advanced = TimeShiftedInput(pulse, -50 * u.ms)
        arr_advanced = advanced()
        self.assertEqual(arr_advanced.shape[0], 5000)

        # Via shift() method
        delayed_method = pulse.shift(100 * u.ms)
        self.assertIsInstance(delayed_method, TimeShiftedInput)


class TestClippedInput(TestCase):
    """Test ClippedInput class and its docstring examples."""

    def test_clipping_modes(self):
        """Test different clipping modes."""
        ramp = RampInput(-2, 2, 200 * u.ms)

        # Clip to [0, 1] range
        saturated = ClippedInput(ramp, 0, 1)
        arr = saturated()
        self.assertTrue(np.all(u.get_magnitude(arr) >= 0))
        self.assertTrue(np.all(u.get_magnitude(arr) <= 1))

        # Only lower bound (rectification)
        rectified = ClippedInput(ramp, min_val=0)
        arr = rectified()
        self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

        # Only upper bound (saturation)
        capped = ClippedInput(ramp, max_val=1.5)
        arr = capped()
        self.assertTrue(np.all(u.get_magnitude(arr) <= 1.5))

        # Via clip() method
        saturated_method = ramp.clip(0, 1)
        self.assertIsInstance(saturated_method, ClippedInput)


class TestSmoothedInput(TestCase):
    """Test SmoothedInput class and its docstring examples."""

    def test_smoothing_levels(self):
        """Test different smoothing levels."""
        steps = StepInput([0, 1, 0.5, 1, 0],
                          [0, 50, 100, 150, 200],
                          250 * u.ms)

        # Light smoothing (fast response)
        light = SmoothedInput(steps, 5 * u.ms)
        arr_light = light()
        self.assertEqual(arr_light.shape[0], 2500)

        # Heavy smoothing (slow response)
        heavy = SmoothedInput(steps, 25 * u.ms)
        arr_heavy = heavy()
        self.assertEqual(arr_heavy.shape[0], 2500)

        # Via smooth() method
        smooth = steps.smooth(10 * u.ms)
        self.assertIsInstance(smooth, SmoothedInput)

        # Heavy smoothing should have smaller variations
        light_var = np.var(u.get_magnitude(arr_light))
        heavy_var = np.var(u.get_magnitude(arr_heavy))
        self.assertLess(heavy_var, light_var)


class TestRepeatedInput(TestCase):
    """Test RepeatedInput class and its docstring examples."""

    def test_burst_train(self):
        """Test burst train example."""
        # Single burst
        burst = StepInput([0, 1, 0], [0, 10, 30], 50 * u.ms)

        # Burst train (10 bursts, 500ms total)
        train = RepeatedInput(burst, 10)
        arr = train()
        self.assertEqual(arr.shape[0], 5000)  # 50ms * 10 = 500ms

        # Via repeat() method
        train_method = burst.repeat(10)
        self.assertIsInstance(train_method, RepeatedInput)

    def test_oscillation_packets(self):
        """Test oscillation packet example."""
        packet = SinusoidalInput(1.0, 100 * u.Hz, 100 * u.ms)
        packets = RepeatedInput(packet, 5)
        arr = packets()
        self.assertEqual(arr.shape[0], 5000)  # 100ms * 5 = 500ms


class TestTransformedInput(TestCase):
    """Test TransformedInput class and its docstring examples."""

    def test_nonlinear_transformations(self):
        """Test various nonlinear transformation examples."""
        sine = SinusoidalInput(1.0, 10 * u.Hz, 200 * u.ms)

        # Half-wave rectification
        rectified = TransformedInput(sine, lambda x: jnp.maximum(x, 0))
        arr = rectified()
        self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

        # Squaring (frequency doubling)
        squared = TransformedInput(sine, lambda x: x ** 2)
        arr = squared()
        self.assertTrue(np.all(u.get_magnitude(arr) >= 0))

        # Sigmoid nonlinearity
        sigmoid = TransformedInput(sine,
                                   lambda x: 1 / (1 + jnp.exp(-10 * x)))
        arr = sigmoid()
        self.assertTrue(np.all(u.get_magnitude(arr) >= 0))
        self.assertTrue(np.all(u.get_magnitude(arr) <= 1))

        # Via apply() method
        transformed = sine.apply(lambda x: jnp.abs(x))
        self.assertIsInstance(transformed, TransformedInput)


class TestPropertiesAndAttributes(TestCase):
    """Test properties and attributes of Input classes."""

    def test_dt_property(self):
        """Test dt property retrieval from global environment."""
        ramp = RampInput(0, 1, 100 * u.ms)
        self.assertEqual(ramp.dt, brainstate.environ.get_dt())

    def test_n_steps_property(self):
        """Test n_steps calculation."""
        # 100ms duration with dt=0.1ms should give 1000 steps
        ramp = RampInput(0, 1, 100 * u.ms)
        self.assertEqual(ramp.n_steps, 1000)

        # 250ms duration with dt=0.1ms should give 2500 steps
        sine = SinusoidalInput(1.0, 10 * u.Hz, 250 * u.ms)
        self.assertEqual(sine.n_steps, 2500)

    def test_shape_property(self):
        """Test shape property."""
        ramp = RampInput(0, 1, 100 * u.ms)
        self.assertEqual(ramp.shape, (1000,))

        # For multi-channel inputs (if supported)
        steps = StepInput([0, 1, 0], [0, 50, 100], 150 * u.ms)
        self.assertEqual(steps.shape, (1500,))


class TestEdgeCases(TestCase):
    """Test edge cases and error handling."""

    def test_concatenation_type_error(self):
        """Test that concatenation with non-Input raises TypeError."""
        ramp = RampInput(0, 1, 100 * u.ms)
        with self.assertRaises(TypeError):
            result = ramp & 0.5

    def test_overlay_type_error(self):
        """Test that overlay with non-Input raises TypeError."""
        ramp = RampInput(0, 1, 100 * u.ms)
        with self.assertRaises(TypeError):
            result = ramp | 0.5

    def test_unknown_operator(self):
        """Test that unknown operator raises ValueError."""
        ramp = RampInput(0, 1, 100 * u.ms)
        sine = SinusoidalInput(1.0, 10 * u.Hz, 100 * u.ms)
        composite = CompositeInput(ramp, sine, 'unknown')
        with self.assertRaises(ValueError):
            arr = composite()

    def test_zero_duration(self):
        """Test handling of zero or very small durations."""
        # This might create 0 or 1 step depending on implementation
        short = StepInput([1], [0], 0.01 * u.ms)
        arr = short()
        self.assertGreaterEqual(arr.shape[0], 0)
