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

"""Tests for composable input API."""

from unittest import TestCase
import brainstate
import brainunit as u
import numpy as np
import matplotlib.pyplot as plt

from braintools.input import (
    # Composable classes
    RampInput, SinusoidalInput, SquareInput, 
    GaussianPulse, ExponentialDecay, BurstInput,
    WienerProcess, OUProcess, StepInput,
    # Base class for type checking
    Input
)

block = False
brainstate.environ.set(dt=0.1 * u.ms)


def show(current, title=''):
    """Helper function to visualize currents."""
    if plt is not None:
        if hasattr(current, '__call__'):
            # It's a composable Input object
            array = current()
            duration = u.get_magnitude(current.duration)
        else:
            # It's a regular array
            array = current
            duration = len(array) * 0.1  # Assume dt=0.1
            
        ts = np.arange(0, duration, 0.1)
        if array.ndim == 1:
            plt.plot(ts, array)
        else:
            for i in range(array.shape[1]):
                plt.plot(ts, array[:, i], label=f'Channel {i+1}')
            plt.legend()
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestComposableAPI(TestCase):
    """Test the composable input API."""
    
    def test_basic_composition(self):
        """Test basic addition of two inputs."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create two inputs
            ramp = RampInput(0, 1, 500 * u.ms)
            sine = SinusoidalInput(0.3, 10 * u.Hz, 500 * u.ms)
            
            # Combine them
            combined = ramp + sine
            
            # Check that result is an Input object
            self.assertIsInstance(combined, Input)
            
            # Generate the array
            array = combined()
            self.assertEqual(array.shape[0], 5000)
            
            show(combined, 'Ramp + Sinusoid')
    
    def test_multiplication(self):
        """Test multiplication for amplitude modulation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create envelope and carrier
            envelope = GaussianPulse(1.0, 250 * u.ms, 50 * u.ms, 500 * u.ms)
            carrier = SinusoidalInput(1.0, 20 * u.Hz, 500 * u.ms)
            
            # Amplitude modulation
            modulated = envelope * carrier
            
            array = modulated()
            self.assertEqual(array.shape[0], 5000)
            
            show(modulated, 'Amplitude Modulation (Gaussian * Sine)')
    
    def test_scaling(self):
        """Test scaling operation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            square = SquareInput(2.0, 5 * u.Hz, 400 * u.ms)
            
            # Scale to half amplitude
            scaled = square.scale(0.5)
            
            array = scaled()
            self.assertEqual(array.shape[0], 4000)
            # Check that max value is approximately 1.0 (half of 2.0)
            self.assertAlmostEqual(np.max(np.abs(array)), 1.0, places=5)
            
            show(scaled, 'Scaled Square Wave (0.5x)')
    
    def test_clipping(self):
        """Test clipping operation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create a ramp that goes from -2 to 2
            ramp = RampInput(-2, 2, 300 * u.ms)
            
            # Clip to [-1, 1]
            clipped = ramp.clip(-1, 1)
            
            array = clipped()
            self.assertEqual(array.shape[0], 3000)
            self.assertTrue(np.all(array >= -1.01))  # Small tolerance
            self.assertTrue(np.all(array <= 1.01))
            
            show(clipped, 'Clipped Ramp [-1, 1]')
    
    def test_time_shift(self):
        """Test time shifting operation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulse = GaussianPulse(1.0, 100 * u.ms, 20 * u.ms, 300 * u.ms)
            
            # Shift by 50ms
            shifted = pulse.shift(50 * u.ms)
            
            array_original = pulse()
            array_shifted = shifted()
            
            # Find peaks
            peak_original = np.argmax(array_original) * 0.1
            peak_shifted = np.argmax(array_shifted) * 0.1
            
            # Check that shift is approximately correct
            self.assertAlmostEqual(peak_shifted - peak_original, 50, delta=5)
            
            show(shifted, 'Time-Shifted Gaussian Pulse')
    
    def test_smoothing(self):
        """Test smoothing operation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create a square wave and smooth it
            square = SquareInput(1.0, 10 * u.Hz, 200 * u.ms)
            smoothed = square.smooth(tau=5 * u.ms)
            
            array_original = square()
            array_smoothed = smoothed()
            
            # Smoothed should have fewer sharp transitions
            diff_original = np.diff(array_original)
            diff_smoothed = np.diff(array_smoothed)
            
            self.assertTrue(np.max(np.abs(diff_smoothed)) < np.max(np.abs(diff_original)))
            
            show(smoothed, 'Smoothed Square Wave')
    
    def test_sequential_composition(self):
        """Test sequential composition with & operator."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create two bursts in sequence
            burst1 = BurstInput(3, 20 * u.ms, 50 * u.ms, 1.0, 200 * u.ms)
            burst2 = BurstInput(2, 30 * u.ms, 60 * u.ms, 0.5, 150 * u.ms)
            
            # Concatenate them
            sequential = burst1 & burst2
            
            array = sequential()
            # Total duration should be 200 + 150 = 350ms = 3500 steps
            self.assertEqual(array.shape[0], 3500)
            
            show(sequential, 'Sequential Bursts')
    
    def test_overlay_composition(self):
        """Test overlay composition with | operator."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create two step inputs
            step1 = StepInput([0, 0.5, 0], [0, 100, 200], 300 * u.ms)
            step2 = StepInput([0, 0.8, 0], [0, 150, 250], 300 * u.ms)
            
            # Overlay (take maximum)
            overlay = step1 | step2
            
            array = overlay()
            self.assertEqual(array.shape[0], 3000)
            
            # Check that maximum is taken
            # At t=120ms, step1=0.5, step2=0, max=0.5
            # At t=180ms, step1=0.5, step2=0.8, max=0.8
            self.assertAlmostEqual(array[1200], 0.5, places=5)
            self.assertAlmostEqual(array[1800], 0.8, places=5)
            
            show(overlay, 'Overlayed Steps (Maximum)')
    
    def test_complex_composition(self):
        """Test a complex composition of multiple operations."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create a complex input pattern
            # 1. Start with a ramp
            ramp = RampInput(0, 1, 600 * u.ms)
            
            # 2. Add sinusoidal oscillation
            sine = SinusoidalInput(0.2, 15 * u.Hz, 600 * u.ms)
            
            # 3. Create burst pattern
            bursts = BurstInput(4, 30 * u.ms, 100 * u.ms, 1.0, 600 * u.ms)
            
            # 4. Add noise
            noise = WienerProcess(600 * u.ms, sigma=0.05, seed=42)
            
            # Combine: (ramp + sine) * bursts + noise
            complex_input = ((ramp + sine) * bursts + noise).clip(0, 2)
            
            array = complex_input()
            self.assertEqual(array.shape[0], 6000)
            
            show(complex_input, 'Complex Composition: ((Ramp + Sine) * Bursts + Noise).clip(0, 2)')
    
    def test_stochastic_with_modulation(self):
        """Test stochastic process with deterministic modulation."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # OU process with time-varying mean
            ou = OUProcess(mean=0.5, sigma=0.1, tau=20 * u.ms, 
                          duration=500 * u.ms, n=1, seed=123)
            
            # Sinusoidal modulation of the mean
            sine_mean = SinusoidalInput(0.3, 3 * u.Hz, 500 * u.ms)
            
            # Add modulation
            modulated_ou = ou + sine_mean
            
            array = modulated_ou()
            self.assertEqual(array.shape, (5000, 1))
            
            show(modulated_ou, 'OU Process with Sinusoidal Mean')
    
    def test_repeated_pattern(self):
        """Test repeating a pattern."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create a short pattern
            pattern = ExponentialDecay(1.0, 20 * u.ms, 100 * u.ms)
            
            # Repeat it 5 times
            repeated = pattern.repeat(5)
            
            array = repeated()
            # 100ms * 5 = 500ms = 5000 steps
            self.assertEqual(array.shape[0], 5000)
            
            show(repeated, 'Repeated Exponential Decay (5x)')
    
    def test_custom_transformation(self):
        """Test applying custom function."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Create a sinusoid
            sine = SinusoidalInput(1.0, 5 * u.Hz, 300 * u.ms)
            
            # Apply custom transformation (rectification)
            rectified = sine.apply(lambda x: u.math.maximum(x, 0))
            
            array = rectified()
            self.assertEqual(array.shape[0], 3000)
            # Check that all values are non-negative
            self.assertTrue(np.all(array >= 0))
            
            show(rectified, 'Rectified Sinusoid')
    
    def test_backward_compatibility(self):
        """Test that functional API still works."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            from braintools.input import ramp_input, sinusoidal_input
            
            # Use functional API
            ramp = ramp_input(0, 1, 200 * u.ms)
            sine = sinusoidal_input(0.5, 10 * u.Hz, 200 * u.ms)
            
            # Simple array addition
            combined = ramp + sine
            
            self.assertEqual(combined.shape[0], 2000)