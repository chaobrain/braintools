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

"""Tests for composable waveform input classes."""

from unittest import TestCase

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

from braintools.input._composable_basic import Ramp, Constant, Step
from braintools.input._composable_pulses import GaussianPulse
from braintools.input._composable_waveforms import (
    Sinusoidal, Square, Triangular,
    Sawtooth, Chirp, NoisySinusoidal
)

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
            for i in range(current_value.shape[1]):
                plt.plot(ts, current_value[:, i], label=f'Channel {i + 1}')
            plt.legend()
        plt.title(title)
        plt.xlabel('Time [ms]')
        plt.ylabel('Current Value')
        plt.show(block=block)


class TestSinusoidal(TestCase):
    """Test Sinusoidal class."""

    def test_simple_sinusoidal(self):
        """Test simple sinusoidal input from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            sine = Sinusoidal(1.0, 10 * u.Hz, 1000 * u.ms)
            signal = sine()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Simple Sinusoidal')

    def test_amplitude_modulation(self):
        """Test amplitude-modulated signal from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            carrier = Sinusoidal(1.0, 100 * u.Hz, 500 * u.ms)
            envelope = Ramp(0, 1, 500 * u.ms)
            am_signal = carrier * envelope
            signal = am_signal()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'Amplitude Modulated Sinusoidal')

    def test_frequency_beats(self):
        """Test frequency beats from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            sine1 = Sinusoidal(1.0, 10 * u.Hz, 1000 * u.ms)
            sine2 = Sinusoidal(1.0, 11 * u.Hz, 1000 * u.ms)
            beats = sine1 + sine2  # 1 Hz beat frequency
            signal = beats()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Frequency Beats')

    def test_harmonics(self):
        """Test complex waveform with harmonics from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            fundamental = Sinusoidal(1.0, 5 * u.Hz, 2000 * u.ms)
            third = Sinusoidal(0.3, 15 * u.Hz, 2000 * u.ms)
            fifth = Sinusoidal(0.2, 25 * u.Hz, 2000 * u.ms)
            complex_wave = fundamental + third + fifth
            signal = complex_wave()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Complex Waveform with Harmonics')

    def test_windowed_sinusoid(self):
        """Test windowed sinusoid from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            resonance = Sinusoidal(
                amplitude=2.0,
                frequency=8 * u.Hz,  # Theta frequency
                duration=5000 * u.ms,
                t_start=1000 * u.ms,
                t_end=4000 * u.ms
            )
            signal = resonance()
            self.assertEqual(signal.shape[0], 50000)

            # Check windowing
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:10000] == 0))
            self.assertTrue(np.all(signal_mag[40000:] == 0))
            show(signal, 5000 * u.ms, 'Windowed Sinusoid')

    def test_positive_bias(self):
        """Test sinusoid with positive bias from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            positive_sine = Sinusoidal(
                amplitude=5.0,
                frequency=5 * u.Hz,
                duration=2000 * u.ms,
                bias=True  # Oscillates between 0 and 10
            )
            signal = positive_sine()
            self.assertEqual(signal.shape[0], 20000)

            # Check that all values are non-negative
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag >= -0.01))  # Small tolerance for numerical errors
            show(signal, 2000 * u.ms, 'Positive Bias Sinusoid')


class TestSquare(TestCase):
    """Test Square class."""

    def test_symmetric_square(self):
        """Test symmetric square wave from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            square = Square(1.0, 5 * u.Hz, 1000 * u.ms)
            signal = square()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Symmetric Square Wave')

    def test_pulse_train(self):
        """Test pulse train with duty cycle from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pulses = Square(
                amplitude=5.0,
                frequency=10 * u.Hz,
                duration=500 * u.ms,
                duty_cycle=0.2  # 20% high, 80% low
            )
            signal = pulses()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'Pulse Train (20% Duty Cycle)')

    def test_smoothed_square(self):
        """Test smoothed square wave from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            square = Square(2.0, 5 * u.Hz, 800 * u.ms)
            smoothed = square.smooth(tau=5 * u.ms)  # Low-pass filter
            signal = smoothed()
            self.assertEqual(signal.shape[0], 8000)
            show(signal, 800 * u.ms, 'Smoothed Square Wave')

    def test_clock_signal(self):
        """Test clock signal from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            clock = Square(
                amplitude=1.0,
                frequency=40 * u.Hz,
                duration=250 * u.ms,
                duty_cycle=0.5
            )
            signal = clock()
            self.assertEqual(signal.shape[0], 2500)
            show(signal, 250 * u.ms, 'Clock Signal')

    def test_square_with_offset(self):
        """Test square wave with DC offset from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            square = Square(3.0, 2 * u.Hz, 2000 * u.ms)
            offset = Constant([(2.0, 2000 * u.ms)])
            shifted_square = square + offset
            signal = shifted_square()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Square Wave with DC Offset')

    def test_gated_square(self):
        """Test gated square wave from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            square = Square(1.0, 50 * u.Hz, 1000 * u.ms)
            gate = Step([0, 1, 0], [0 * u.ms, 200 * u.ms, 800 * u.ms], 1000 * u.ms)
            gated = square * gate
            signal = gated()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Gated Square Wave')

    def test_positive_bias_square(self):
        """Test square wave with positive bias from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            positive_square = Square(
                amplitude=4.0,
                frequency=10 * u.Hz,
                duration=500 * u.ms,
                bias=True  # Alternates between 0 and 8
            )
            signal = positive_square()
            self.assertEqual(signal.shape[0], 5000)

            # Check values are non-negative
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag >= -0.01))
            show(signal, 500 * u.ms, 'Positive Bias Square Wave')

    def test_pwm_signal(self):
        """Test PWM-like signal from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            pwm = Square(
                amplitude=5.0,
                frequency=100 * u.Hz,
                duration=100 * u.ms,
                duty_cycle=0.1  # 10% duty cycle
            )
            signal = pwm()
            self.assertEqual(signal.shape[0], 1000)
            show(signal, 100 * u.ms, 'PWM Signal (10% Duty Cycle)')


class TestTriangular(TestCase):
    """Test Triangular class."""

    def test_simple_triangular(self):
        """Test simple triangular wave from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            tri = Triangular(2.0, 3 * u.Hz, 1000 * u.ms)
            signal = tri()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Simple Triangular Wave')

    def test_slow_ramp_iv_curve(self):
        """Test slow ramp for I-V curve from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp = Triangular(
                amplitude=100.0,
                frequency=0.5 * u.Hz,  # 2 second period
                duration=4000 * u.ms
            )
            signal = ramp()
            self.assertEqual(signal.shape[0], 40000)
            show(signal, 4000 * u.ms, 'Slow Ramp for I-V Curve')

    def test_clipped_triangular(self):
        """Test clipped triangular wave from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            tri = Triangular(5.0, 4 * u.Hz, 600 * u.ms)
            clipped = tri.clip(-3.0, 3.0)  # Trapezoidal shape
            signal = clipped()
            self.assertEqual(signal.shape[0], 6000)
            show(signal, 600 * u.ms, 'Clipped Triangular (Trapezoidal)')

    def test_triangular_with_envelope(self):
        """Test triangular wave with envelope from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            tri = Triangular(2.0, 10 * u.Hz, 1000 * u.ms)
            envelope = GaussianPulse(1.0, 500 * u.ms, 100 * u.ms, 1000 * u.ms)
            modulated = tri * envelope
            signal = modulated()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Triangular with Gaussian Envelope')

    def test_adaptation_test(self):
        """Test triangular for adaptation testing from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            adaptation_test = Triangular(
                amplitude=20.0,
                frequency=1 * u.Hz,
                duration=5000 * u.ms
            )
            signal = adaptation_test()
            self.assertEqual(signal.shape[0], 50000)
            show(signal, 5000 * u.ms, 'Adaptation Test Triangular')

    def test_positive_bias_triangular(self):
        """Test triangular with positive bias from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            positive_tri = Triangular(
                amplitude=3.0,
                frequency=5 * u.Hz,
                duration=800 * u.ms,
                bias=True  # Ramps between 0 and 6
            )
            signal = positive_tri()
            self.assertEqual(signal.shape[0], 8000)

            # Check non-negative values
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag >= -0.01))
            show(signal, 800 * u.ms, 'Positive Bias Triangular')

    def test_windowed_triangular(self):
        """Test windowed triangular from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            windowed_tri = Triangular(
                amplitude=4.0,
                frequency=2 * u.Hz,
                duration=3000 * u.ms,
                t_start=500 * u.ms,
                t_end=2500 * u.ms
            )
            signal = windowed_tri()
            self.assertEqual(signal.shape[0], 30000)

            # Check windowing
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:5000] == 0))
            self.assertTrue(np.all(signal_mag[25000:] == 0))
            show(signal, 3000 * u.ms, 'Windowed Triangular')


class TestSawtooth(TestCase):
    """Test Sawtooth class."""

    def test_simple_sawtooth(self):
        """Test simple sawtooth wave from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            saw = Sawtooth(1.0, 2 * u.Hz, 2000 * u.ms)
            signal = saw()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Simple Sawtooth Wave')

    def test_threshold_detection(self):
        """Test slow ramp for threshold detection from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            threshold_test = Sawtooth(
                amplitude=50.0,
                frequency=0.5 * u.Hz,  # 2 second ramp
                duration=4000 * u.ms
            )
            signal = threshold_test()
            self.assertEqual(signal.shape[0], 40000)
            show(signal, 4000 * u.ms, 'Threshold Detection Sawtooth')

    def test_sawtooth_with_offset(self):
        """Test sawtooth with DC offset from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            saw = Sawtooth(3.0, 3 * u.Hz, 1000 * u.ms)
            offset = Constant([(2.0, 1000 * u.ms)])
            shifted_saw = saw + offset
            signal = shifted_saw()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Sawtooth with DC Offset')

    def test_fast_reset(self):
        """Test fast reset sawtooth from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            reset_test = Sawtooth(
                amplitude=20.0,
                frequency=20 * u.Hz,
                duration=250 * u.ms
            )
            signal = reset_test()
            self.assertEqual(signal.shape[0], 2500)
            show(signal, 250 * u.ms, 'Fast Reset Sawtooth')

    def test_repeated_ramp(self):
        """Test repeated ramp protocol from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            ramp_protocol = Sawtooth(
                amplitude=100.0,
                frequency=1 * u.Hz,
                duration=10000 * u.ms
            )
            signal = ramp_protocol()
            self.assertEqual(signal.shape[0], 100000)
            # Show only first 2 seconds for clarity
            show(signal[:20000], 2000 * u.ms, 'Repeated Ramp Protocol (first 2s)')

    def test_positive_bias_sawtooth(self):
        """Test sawtooth with positive bias from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            positive_saw = Sawtooth(
                amplitude=5.0,
                frequency=4 * u.Hz,
                duration=500 * u.ms,
                bias=True  # Ramps from 0 to 10
            )
            signal = positive_saw()
            self.assertEqual(signal.shape[0], 5000)

            # Check non-negative values
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag >= -0.01))
            show(signal, 500 * u.ms, 'Positive Bias Sawtooth')

    def test_modulated_sawtooth(self):
        """Test modulated sawtooth from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            saw = Sawtooth(2.0, 5 * u.Hz, 1000 * u.ms)
            modulation = Sinusoidal(0.5, 1 * u.Hz, 1000 * u.ms, bias=True)
            modulated = saw * modulation
            signal = modulated()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Modulated Sawtooth')

    def test_windowed_sawtooth(self):
        """Test windowed sawtooth from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            windowed_saw = Sawtooth(
                amplitude=8.0,
                frequency=3 * u.Hz,
                duration=2000 * u.ms,
                t_start=400 * u.ms,
                t_end=1600 * u.ms
            )
            signal = windowed_saw()
            self.assertEqual(signal.shape[0], 20000)

            # Check windowing
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:4000] == 0))
            self.assertTrue(np.all(signal_mag[16000:] == 0))
            show(signal, 2000 * u.ms, 'Windowed Sawtooth')


class TestChirp(TestCase):
    """Test Chirp class."""

    def test_linear_chirp(self):
        """Test linear frequency sweep from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            chirp = Chirp(
                amplitude=1.0,
                f_start=1 * u.Hz,
                f_end=50 * u.Hz,
                duration=2000 * u.ms,
                method='linear'
            )
            signal = chirp()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Linear Chirp (1-50 Hz)')

    def test_logarithmic_chirp(self):
        """Test logarithmic sweep from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            log_chirp = Chirp(
                amplitude=2.0,
                f_start=0.1 * u.Hz,
                f_end=100 * u.Hz,
                duration=5000 * u.ms,
                method='logarithmic'
            )
            signal = log_chirp()
            self.assertEqual(signal.shape[0], 50000)
            show(signal[:20000], 2000 * u.ms, 'Logarithmic Chirp (first 2s)')

    def test_repeated_chirp(self):
        """Test repeated chirp from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            chirp = Chirp(1.0, 1 * u.Hz, 10 * u.Hz, 500 * u.ms)
            repeated = chirp.repeat(3)  # Repeat 3 times
            signal = repeated()
            self.assertEqual(signal.shape[0], 15000)  # 3 * 500ms
            show(signal, 1500 * u.ms, 'Repeated Chirp (3x)')

    def test_reverse_chirp(self):
        """Test reverse chirp from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            reverse_chirp = Chirp(
                amplitude=3.0,
                f_start=100 * u.Hz,
                f_end=1 * u.Hz,
                duration=2000 * u.ms
            )
            signal = reverse_chirp()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Reverse Chirp (100-1 Hz)')

    def test_resonance_test_chirp(self):
        """Test theta-gamma resonance test from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            resonance_test = Chirp(
                amplitude=1.0,
                f_start=4 * u.Hz,  # Theta start
                f_end=80 * u.Hz,  # Gamma end
                duration=10000 * u.ms,
                method='logarithmic'
            )
            signal = resonance_test()
            self.assertEqual(signal.shape[0], 100000)
            # Show only first 2 seconds for clarity
            show(signal[:20000], 2000 * u.ms, 'Theta-Gamma Resonance Test (first 2s)')

    def test_windowed_chirp(self):
        """Test windowed chirp from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            windowed_chirp = Chirp(
                amplitude=2.0,
                f_start=2 * u.Hz,
                f_end=40 * u.Hz,
                duration=3000 * u.ms,
                t_start=500 * u.ms,
                t_end=2500 * u.ms
            )
            signal = windowed_chirp()
            self.assertEqual(signal.shape[0], 30000)

            # Check windowing
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:5000] == 0))
            self.assertTrue(np.all(signal_mag[25000:] == 0))
            show(signal, 3000 * u.ms, 'Windowed Chirp')

    def test_chirp_with_envelope(self):
        """Test chirp with amplitude envelope from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            chirp = Chirp(1.0, 5 * u.Hz, 50 * u.Hz, 1000 * u.ms)
            envelope = Ramp(0.1, 1.0, 1000 * u.ms)
            ramped_chirp = chirp * envelope
            signal = ramped_chirp()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Chirp with Ramp Envelope')

    def test_broadband_chirp(self):
        """Test broadband chirp from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            low_chirp = Chirp(1.0, 0.5 * u.Hz, 5 * u.Hz, 2000 * u.ms)
            high_chirp = Chirp(0.5, 20 * u.Hz, 100 * u.Hz, 2000 * u.ms)
            broadband = low_chirp + high_chirp
            signal = broadband()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Broadband Chirp')

    def test_positive_bias_chirp(self):
        """Test chirp with positive bias from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            positive_chirp = Chirp(
                amplitude=5.0,
                f_start=10 * u.Hz,
                f_end=30 * u.Hz,
                duration=1000 * u.ms,
                bias=True  # Always positive
            )
            signal = positive_chirp()
            self.assertEqual(signal.shape[0], 10000)

            # Check non-negative values
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag >= -0.01))
            show(signal, 1000 * u.ms, 'Positive Bias Chirp')


class TestNoisySinusoidal(TestCase):
    """Test NoisySinusoidal class."""

    def test_small_noise(self):
        """Test sinusoid with small noise from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            noisy = NoisySinusoidal(
                amplitude=10.0,
                frequency=10 * u.Hz,
                noise_amplitude=1.0,  # 10% noise
                duration=1000 * u.ms
            )
            signal = noisy()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Sinusoid with Small Noise')

    def test_stochastic_resonance(self):
        """Test high noise for stochastic resonance from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            stochastic = NoisySinusoidal(
                amplitude=5.0,
                frequency=5 * u.Hz,
                noise_amplitude=10.0,  # Noise > signal
                duration=2000 * u.ms
            )
            signal = stochastic()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Stochastic Resonance (High Noise)')

    def test_filtered_noisy(self):
        """Test filtering noisy signal from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            noisy = NoisySinusoidal(1.0, 20 * u.Hz, 0.5, 500 * u.ms)
            filtered = noisy.smooth(tau=10 * u.ms)  # Low-pass filter
            signal = filtered()
            self.assertEqual(signal.shape[0], 5000)
            show(signal, 500 * u.ms, 'Filtered Noisy Sinusoid')

    def test_theta_with_noise(self):
        """Test theta rhythm with synaptic noise from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            theta_noisy = NoisySinusoidal(
                amplitude=2.0,
                frequency=8 * u.Hz,  # Theta frequency
                noise_amplitude=0.5,
                duration=5000 * u.ms
            )
            signal = theta_noisy()
            self.assertEqual(signal.shape[0], 50000)
            show(signal[:20000], 2000 * u.ms, 'Theta with Synaptic Noise (first 2s)')

    def test_cross_frequency_noisy(self):
        """Test combining multiple noisy oscillations from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            theta = NoisySinusoidal(1.0, 8 * u.Hz, 0.2, 1000 * u.ms, seed=42)
            gamma = NoisySinusoidal(0.5, 40 * u.Hz, 0.1, 1000 * u.ms, seed=43)
            cross_frequency = theta + gamma
            signal = cross_frequency()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Cross-Frequency Coupling (Noisy)')

    def test_windowed_noisy(self):
        """Test windowed noisy stimulation from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            windowed_noisy = NoisySinusoidal(
                amplitude=8.0,
                frequency=20 * u.Hz,
                noise_amplitude=2.0,
                duration=1000 * u.ms,
                t_start=200 * u.ms,
                t_end=800 * u.ms
            )
            signal = windowed_noisy()
            self.assertEqual(signal.shape[0], 10000)

            # Check windowing
            signal_mag = u.get_magnitude(signal)
            self.assertTrue(np.all(signal_mag[:2000] == 0))
            self.assertTrue(np.all(signal_mag[8000:] == 0))
            show(signal, 1000 * u.ms, 'Windowed Noisy Sinusoid')

    def test_reproducible_noisy(self):
        """Test reproducible noisy signal from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            reproducible1 = NoisySinusoidal(
                amplitude=15.0,
                frequency=40 * u.Hz,
                noise_amplitude=3.0,
                duration=500 * u.ms,
                seed=42  # Fixed random seed
            )
            signal1 = reproducible1()

            reproducible2 = NoisySinusoidal(
                amplitude=15.0,
                frequency=40 * u.Hz,
                noise_amplitude=3.0,
                duration=500 * u.ms,
                seed=42  # Same seed
            )
            signal2 = reproducible2()

            # Should be identical
            assert u.math.allclose(signal1, signal2)
            self.assertEqual(signal1.shape[0], 5000)
            show(signal1, 500 * u.ms, 'Reproducible Noisy Sinusoid')

    def test_weak_signal_detection(self):
        """Test weak signal in strong noise from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            weak_signal = NoisySinusoidal(
                amplitude=1.0,
                frequency=10 * u.Hz,
                noise_amplitude=5.0,  # 5x noise
                duration=10000 * u.ms
            )
            signal = weak_signal()
            self.assertEqual(signal.shape[0], 100000)
            # Show only first 1 second for clarity
            show(signal[:10000], 1000 * u.ms, 'Weak Signal in Strong Noise (first 1s)')

    def test_modulated_noisy(self):
        """Test modulating noisy sinusoid from docstring example."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            noisy = NoisySinusoidal(2.0, 30 * u.Hz, 0.5, 1000 * u.ms)
            envelope = GaussianPulse(1.0, 500 * u.ms, 100 * u.ms, 1000 * u.ms)
            burst = noisy * envelope
            signal = burst()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Noisy Sinusoid Burst')


class TestCombinedWaveforms(TestCase):
    """Test combinations of different waveforms."""

    def test_mixed_frequencies(self):
        """Test combining different frequency components."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            low_freq = Sinusoidal(2.0, 2 * u.Hz, 1000 * u.ms)
            mid_freq = Sinusoidal(1.0, 10 * u.Hz, 1000 * u.ms)
            high_freq = Sinusoidal(0.5, 50 * u.Hz, 1000 * u.ms)
            mixed = low_freq + mid_freq + high_freq
            signal = mixed()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Mixed Frequency Components')

    def test_square_plus_sine(self):
        """Test combining square and sine waves."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            square = Square(1.0, 5 * u.Hz, 800 * u.ms)
            sine = Sinusoidal(0.5, 15 * u.Hz, 800 * u.ms)
            combined = square + sine
            signal = combined()
            self.assertEqual(signal.shape[0], 8000)
            show(signal, 800 * u.ms, 'Square + Sine Wave')

    def test_triangular_times_chirp(self):
        """Test multiplying triangular and chirp."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            tri = Triangular(1.0, 2 * u.Hz, 1000 * u.ms)
            chirp = Chirp(1.0, 5 * u.Hz, 20 * u.Hz, 1000 * u.ms)
            modulated = tri * chirp
            signal = modulated()
            self.assertEqual(signal.shape[0], 10000)
            show(signal, 1000 * u.ms, 'Triangular-Modulated Chirp')

    def test_complex_protocol(self):
        """Test complex stimulation protocol."""
        with brainstate.environ.context(dt=0.1 * u.ms):
            # Base oscillation
            base = Sinusoidal(1.0, 10 * u.Hz, 2000 * u.ms)

            # Add ramp
            ramp = Ramp(0, 2.0, 2000 * u.ms)

            # Add noise
            noise = NoisySinusoidal(0.5, 40 * u.Hz, 0.2, 2000 * u.ms)

            # Combine all
            protocol = base + ramp + noise
            signal = protocol()
            self.assertEqual(signal.shape[0], 20000)
            show(signal, 2000 * u.ms, 'Complex Stimulation Protocol')
