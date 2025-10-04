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
Composable waveform input generators.

This module provides composable waveform generators that can be combined
with other input components using arithmetic operations to create complex
stimulation protocols.
"""

from typing import Optional

import brainstate
import brainunit as u

from . import _functional_waveforms as functional
from ._composable_base import Input
from ._deprecation import create_deprecated_class

ArrayLike = brainstate.typing.ArrayLike

__all__ = [
    'Sinusoidal',
    'Square',
    'Triangular',
    'Sawtooth',
    'Chirp',
    'NoisySinusoidal',
]


class Sinusoidal(Input):
    r"""Composable sinusoidal input generator.

    Creates a sinusoidal waveform with specified amplitude and frequency.
    This class is composable, allowing mathematical operations with other
    Input objects to create complex stimulation protocols.

    The sinusoidal waveform is defined as:

    .. math::
        I(t) = A \sin(2\pi f t)

    where A is amplitude and f is frequency.

    Parameters
    ----------
    amplitude : ArrayLike
        Peak amplitude of the sinusoidal wave.
    frequency : Quantity
        Frequency of oscillation. Must be in Hz units.
    duration : float or Quantity
        Total duration of the signal. Supports time units.
    t_start : float or Quantity, optional
        Time when sinusoid starts. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time when sinusoid ends. After this, output is 0.
        Default is duration.
    bias : bool, optional
        If True, adds DC offset equal to amplitude (non-negative output).
        If False, oscillates around 0 (default).

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the sinusoid
    frequency : Quantity
        Frequency of oscillation in Hz
    t_start : float or Quantity
        Start time of the sinusoid
    t_end : float or Quantity
        End time of the sinusoid
    bias : bool
        Whether DC bias is applied

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Create simple sinusoidal input:

    .. code-block:: python

        >>> sine = Sinusoidal(1.0, 10 * u.Hz, 1000 * u.ms)
        >>> signal = sine()

    Create amplitude-modulated signal:

    .. code-block:: python

        >>> from braintools.input import Ramp
        >>> carrier = Sinusoidal(1.0, 100 * u.Hz, 500 * u.ms)
        >>> envelope = Ramp(0, 1, 500 * u.ms)
        >>> am_signal = carrier * envelope

    Create frequency beats by combining sinusoids:

    .. code-block:: python

        >>> sine1 = Sinusoidal(1.0, 10 * u.Hz, 1000 * u.ms)
        >>> sine2 = Sinusoidal(1.0, 11 * u.Hz, 1000 * u.ms)
        >>> beats = sine1 + sine2  # 1 Hz beat frequency

    Create complex waveform with harmonics:

    .. code-block:: python

        >>> fundamental = Sinusoidal(1.0, 5 * u.Hz, 2000 * u.ms)
        >>> third = Sinusoidal(0.3, 15 * u.Hz, 2000 * u.ms)
        >>> fifth = Sinusoidal(0.2, 25 * u.Hz, 2000 * u.ms)
        >>> complex_wave = fundamental + third + fifth

    Test resonance with windowed sinusoid:

    .. code-block:: python

        >>> resonance = Sinusoidal(
        ...     amplitude=2.0,
        ...     frequency=8 * u.Hz,  # Theta frequency
        ...     duration=5000 * u.ms,
        ...     t_start=1000 * u.ms,
        ...     t_end=4000 * u.ms
        ... )

    Create phase-shifted sinusoids:

    .. code-block:: python

        >>> sine1 = Sinusoidal(1.0, 10 * u.Hz, 1000 * u.ms)
        >>> # Shift by 250ms (90 degrees for 10Hz)
        >>> sine2 = sine1.shift(25 * u.ms)
        >>> quadrature = sine1 + sine2

    Sinusoid with positive bias (rectified):

    .. code-block:: python

        >>> positive_sine = Sinusoidal(
        ...     amplitude=5.0,
        ...     frequency=5 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     bias=True  # Oscillates between 0 and 10
        ... )

    Notes
    -----
    - Phase starts at 0 (sine wave starts at 0)
    - Frequency should be less than Nyquist frequency (1/(2*dt))
    - With bias=True, output ranges from 0 to 2*amplitude
    - Without bias, output ranges from -amplitude to +amplitude
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    Square : Square wave generator
    Chirp : Frequency sweep generator
    NoisySinusoidal : Sinusoid with additive noise
    sinusoidal : Functional API for sinusoidal input
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: ArrayLike,
                 frequency: u.Quantity,
                 duration: ArrayLike,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 bias: bool = False):
        super().__init__(duration)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'

        self.amplitude = amplitude
        self.frequency = frequency
        self.t_start = t_start
        self.t_end = t_end
        self.bias = bias

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the sinusoidal input using the functional API."""
        return functional.sinusoidal(
            amplitude=self.amplitude,
            frequency=self.frequency,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end,
            bias=self.bias
        )


class Square(Input):
    r"""Composable square wave input generator.

    Creates a square wave that alternates between two levels at a specified
    frequency. This class is composable with other Input objects.

    The square wave alternates between +amplitude and -amplitude (or 0 and
    2*amplitude with bias).

    Parameters
    ----------
    amplitude : float
        Peak amplitude of the square wave.
    frequency : Quantity
        Frequency of oscillation. Must be in Hz units.
    duration : float or Quantity
        Total duration of the signal. Supports time units.
    duty_cycle : float, optional
        Fraction of period spent at high level (0 to 1).
        Default is 0.5 (symmetric square wave).
    bias : bool, optional
        If True, adds DC offset equal to amplitude (non-negative output).
        If False, alternates between +amplitude and -amplitude.
    t_start : float or Quantity, optional
        Time when square wave starts. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time when square wave ends. After this, output is 0.
        Default is duration.

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the square wave
    frequency : Quantity
        Frequency of oscillation in Hz
    duty_cycle : float
        Duty cycle of the square wave
    bias : bool
        Whether DC bias is applied
    t_start : float or Quantity
        Start time of the square wave
    t_end : float or Quantity
        End time of the square wave

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Create symmetric square wave:

    .. code-block:: python

        >>> square = Square(1.0, 5 * u.Hz, 1000 * u.ms)
        >>> signal = square()

    Create pulse train with 20% duty cycle:

    .. code-block:: python

        >>> pulses = Square(
        ...     amplitude=5.0,
        ...     frequency=10 * u.Hz,
        ...     duration=500 * u.ms,
        ...     duty_cycle=0.2  # 20% high, 80% low
        ... )

    Smooth square wave transitions:

    .. code-block:: python

        >>> square = Square(2.0, 5 * u.Hz, 800 * u.ms)
        >>> smoothed = square.smooth(tau=5 * u.ms)  # Low-pass filter

    Create clock signal for synchronization:

    .. code-block:: python

        >>> clock = Square(
        ...     amplitude=1.0,
        ...     frequency=40 * u.Hz,
        ...     duration=250 * u.ms,
        ...     duty_cycle=0.5
        ... )

    Combine with DC offset:

    .. code-block:: python

        >>> from braintools.input import Constant
        >>> square = Square(3.0, 2 * u.Hz, 2000 * u.ms)
        >>> offset = Constant([(2.0, 2000 * u.ms)])
        >>> shifted_square = square + offset

    Create gated stimulation:

    .. code-block:: python

        >>> from braintools.input import Step
        >>> square = Square(1.0, 50 * u.Hz, 1000 * u.ms)
        >>> gate = Step([0, 1, 0], [0 * u.ms, 200 * u.ms, 800 * u.ms], 1000 * u.ms)
        >>> gated = square * gate

    Square wave with positive bias:

    .. code-block:: python

        >>> positive_square = Square(
        ...     amplitude=4.0,
        ...     frequency=10 * u.Hz,
        ...     duration=500 * u.ms,
        ...     bias=True  # Alternates between 0 and 8
        ... )

    Create PWM-like signal:

    .. code-block:: python

        >>> pwm = Square(
        ...     amplitude=5.0,
        ...     frequency=100 * u.Hz,
        ...     duration=100 * u.ms,
        ...     duty_cycle=0.1  # 10% duty cycle
        ... )

    Notes
    -----
    - Without bias: alternates between +amplitude and -amplitude
    - With bias: alternates between 0 and 2*amplitude
    - Duty cycle controls fraction of time at high level
    - Transitions are instantaneous (limited by dt resolution)
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    Sinusoidal : Sinusoidal wave generator
    Triangular : Triangular wave generator
    square : Functional API for square wave input
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 duration: ArrayLike,
                 duty_cycle: float = 0.5,
                 bias: bool = False,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None):
        super().__init__(duration)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'

        self.amplitude = amplitude
        self.frequency = frequency
        self.duty_cycle = duty_cycle
        self.bias = bias
        self.t_start = t_start
        self.t_end = t_end

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the square wave using the functional API."""
        return functional.square(
            amplitude=self.amplitude,
            frequency=self.frequency,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end,
            duty_cycle=self.duty_cycle,
            bias=self.bias
        )


class Triangular(Input):
    r"""Composable triangular wave input generator.

    Creates a triangular (linear ramping) waveform that linearly increases
    and decreases between peak values. This class is composable with other
    Input objects.

    The triangular wave ramps linearly between -amplitude and +amplitude.

    Parameters
    ----------
    amplitude : float
        Peak amplitude of the triangular wave.
    frequency : Quantity
        Frequency of oscillation. Must be in Hz units.
    duration : float or Quantity
        Total duration of the signal. Supports time units.
    t_start : float or Quantity, optional
        Time when triangular wave starts. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time when triangular wave ends. After this, output is 0.
        Default is duration.
    bias : bool, optional
        If True, adds DC offset equal to amplitude (non-negative output).
        If False, oscillates between -amplitude and +amplitude.

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the triangular wave
    frequency : Quantity
        Frequency of oscillation in Hz
    t_start : float or Quantity
        Start time of the triangular wave
    t_end : float or Quantity
        End time of the triangular wave
    bias : bool
        Whether DC bias is applied

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Simple triangular wave:

    .. code-block:: python

        >>> tri = Triangular(2.0, 3 * u.Hz, 1000 * u.ms)
        >>> signal = tri()

    Slow ramp for I-V curve measurements:

    .. code-block:: python

        >>> ramp = Triangular(
        ...     amplitude=100.0,
        ...     frequency=0.5 * u.Hz,  # 2 second period
        ...     duration=4000 * u.ms
        ... )

    Clipped triangular wave:

    .. code-block:: python

        >>> tri = Triangular(5.0, 4 * u.Hz, 600 * u.ms)
        >>> clipped = tri.clip(-3.0, 3.0)  # Trapezoidal shape

    Triangular wave with envelope:

    .. code-block:: python

        >>> from braintools.input import GaussianPulse
        >>> tri = Triangular(2.0, 10 * u.Hz, 1000 * u.ms)
        >>> envelope = GaussianPulse(1.0, 500 * u.ms, 100 * u.ms, 1000 * u.ms)
        >>> modulated = tri * envelope

    Testing adaptation with slow ramps:

    .. code-block:: python

        >>> adaptation_test = Triangular(
        ...     amplitude=20.0,
        ...     frequency=1 * u.Hz,
        ...     duration=5000 * u.ms
        ... )

    Triangular wave with positive bias:

    .. code-block:: python

        >>> positive_tri = Triangular(
        ...     amplitude=3.0,
        ...     frequency=5 * u.Hz,
        ...     duration=800 * u.ms,
        ...     bias=True  # Ramps between 0 and 6
        ... )

    Create sawtooth approximation:

    .. code-block:: python

        >>> # Combine triangular with step for asymmetric ramp
        >>> tri = Triangular(1.0, 2 * u.Hz, 1000 * u.ms)
        >>> from braintools.input import Step
        >>> step = Step([0, 0.5], [0 * u.ms, 250 * u.ms], 1000 * u.ms)
        >>> asymmetric = tri + step

    Windowed triangular stimulation:

    .. code-block:: python

        >>> windowed_tri = Triangular(
        ...     amplitude=4.0,
        ...     frequency=2 * u.Hz,
        ...     duration=3000 * u.ms,
        ...     t_start=500 * u.ms,
        ...     t_end=2500 * u.ms
        ... )

    Notes
    -----
    - Ramps linearly between -amplitude and +amplitude
    - With bias=True, ramps between 0 and 2*amplitude
    - Peaks occur at 0, 0.5/frequency, 1/frequency, etc.
    - More suitable than sawtooth for symmetric ramping
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    Sawtooth : Sawtooth wave generator
    Sinusoidal : Sinusoidal wave generator
    triangular : Functional API for triangular wave
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 duration: ArrayLike,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 bias: bool = False):
        super().__init__(duration)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'

        self.amplitude = amplitude
        self.frequency = frequency
        self.t_start = t_start
        self.t_end = t_end
        self.bias = bias

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the triangular wave using the functional API."""
        return functional.triangular(
            amplitude=self.amplitude,
            frequency=self.frequency,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end,
            bias=self.bias
        )


class Sawtooth(Input):
    r"""Composable sawtooth wave input generator.

    Creates a sawtooth waveform that ramps up linearly and then drops sharply.
    This class is composable with other Input objects.

    The sawtooth ramps from -amplitude to +amplitude, then resets instantly.

    Parameters
    ----------
    amplitude : float
        Peak amplitude of the sawtooth wave.
    frequency : Quantity
        Frequency of oscillation. Must be in Hz units.
    duration : float or Quantity
        Total duration of the signal. Supports time units.
    t_start : float or Quantity, optional
        Time when sawtooth wave starts. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time when sawtooth wave ends. After this, output is 0.
        Default is duration.
    bias : bool, optional
        If True, adds DC offset equal to amplitude (non-negative output).
        If False, ramps from -amplitude to +amplitude.

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the sawtooth wave
    frequency : Quantity
        Frequency of oscillation in Hz
    t_start : float or Quantity
        Start time of the sawtooth wave
    t_end : float or Quantity
        End time of the sawtooth wave
    bias : bool
        Whether DC bias is applied

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Simple sawtooth wave:

    .. code-block:: python

        >>> saw = Sawtooth(1.0, 2 * u.Hz, 2000 * u.ms)
        >>> signal = saw()

    Slow ramp for threshold detection:

    .. code-block:: python

        >>> threshold_test = Sawtooth(
        ...     amplitude=50.0,
        ...     frequency=0.5 * u.Hz,  # 2 second ramp
        ...     duration=4000 * u.ms
        ... )

    Combine with DC offset:

    .. code-block:: python

        >>> from braintools.input import Constant
        >>> saw = Sawtooth(3.0, 3 * u.Hz, 1000 * u.ms)
        >>> offset = Constant([(2.0, 1000 * u.ms)])
        >>> shifted_saw = saw + offset

    Fast reset testing:

    .. code-block:: python

        >>> reset_test = Sawtooth(
        ...     amplitude=20.0,
        ...     frequency=20 * u.Hz,
        ...     duration=250 * u.ms
        ... )

    Repeated ramp protocol:

    .. code-block:: python

        >>> ramp_protocol = Sawtooth(
        ...     amplitude=100.0,
        ...     frequency=1 * u.Hz,
        ...     duration=10000 * u.ms
        ... )

    Sawtooth with positive bias:

    .. code-block:: python

        >>> positive_saw = Sawtooth(
        ...     amplitude=5.0,
        ...     frequency=4 * u.Hz,
        ...     duration=500 * u.ms,
        ...     bias=True  # Ramps from 0 to 10
        ... )

    Modulated sawtooth:

    .. code-block:: python

        >>> from braintools.input import Sinusoidal
        >>> saw = Sawtooth(2.0, 5 * u.Hz, 1000 * u.ms)
        >>> modulation = Sinusoidal(0.5, 1 * u.Hz, 1000 * u.ms, bias=True)
        >>> modulated = saw * modulation

    Windowed sawtooth stimulation:

    .. code-block:: python

        >>> windowed_saw = Sawtooth(
        ...     amplitude=8.0,
        ...     frequency=3 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     t_start=400 * u.ms,
        ...     t_end=1600 * u.ms
        ... )

    Create staircase by clipping sawtooth:

    .. code-block:: python

        >>> saw = Sawtooth(10.0, 1 * u.Hz, 3000 * u.ms)
        >>> # Clip to create discrete levels
        >>> staircase = saw.apply(lambda x: u.math.round(x / 2) * 2)

    Notes
    -----
    - Ramps linearly from -amplitude to +amplitude, then resets
    - With bias=True, ramps from 0 to 2*amplitude
    - The ramp is continuous, reset is instantaneous
    - Useful for finding thresholds and testing reset mechanisms
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    Triangular : Triangular wave generator
    Ramp : Single linear ramp
    sawtooth : Functional API for sawtooth wave
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 duration: ArrayLike,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 bias: bool = False):
        super().__init__(duration)
        assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'

        self.amplitude = amplitude
        self.frequency = frequency
        self.t_start = t_start
        self.t_end = t_end
        self.bias = bias

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the sawtooth wave using the functional API."""
        return functional.sawtooth(
            amplitude=self.amplitude,
            frequency=self.frequency,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end,
            bias=self.bias
        )


class Chirp(Input):
    r"""Composable chirp (frequency sweep) signal generator.

    Creates a sinusoidal signal with time-varying frequency that sweeps from
    a starting to ending frequency. This class is composable with other Input
    objects.

    For linear chirp:

    .. math::
        f(t) = f_0 + (f_1 - f_0) \frac{t}{T}

    For logarithmic chirp:

    .. math::
        f(t) = f_0 \left(\frac{f_1}{f_0}\right)^{t/T}

    Parameters
    ----------
    amplitude : float
        Peak amplitude of the chirp signal.
    f_start : Quantity
        Starting frequency. Must be in Hz units.
    f_end : Quantity
        Ending frequency. Must be in Hz units.
    duration : float or Quantity
        Total duration of the signal. Supports time units.
    t_start : float or Quantity, optional
        Time when chirp starts. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time when chirp ends. After this, output is 0.
        Default is duration.
    method : str, optional
        Sweep method: 'linear' or 'logarithmic'.
        Default is 'linear'.
    bias : bool, optional
        If True, adds DC offset equal to amplitude (non-negative output).
        If False, oscillates around 0.

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the chirp signal
    f_start : Quantity
        Starting frequency in Hz
    f_end : Quantity
        Ending frequency in Hz
    t_start : float or Quantity
        Start time of the chirp
    t_end : float or Quantity
        End time of the chirp
    method : str
        Sweep method ('linear' or 'logarithmic')
    bias : bool
        Whether DC bias is applied

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Linear frequency sweep:

    .. code-block:: python

        >>> chirp = Chirp(
        ...     amplitude=1.0,
        ...     f_start=1 * u.Hz,
        ...     f_end=50 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     method='linear'
        ... )
        >>> signal = chirp()

    Logarithmic sweep for spectral analysis:

    .. code-block:: python

        >>> log_chirp = Chirp(
        ...     amplitude=2.0,
        ...     f_start=0.1 * u.Hz,
        ...     f_end=100 * u.Hz,
        ...     duration=5000 * u.ms,
        ...     method='logarithmic'
        ... )

    Repeat chirp multiple times:

    .. code-block:: python

        >>> chirp = Chirp(1.0, 1 * u.Hz, 10 * u.Hz, 500 * u.ms)
        >>> repeated = chirp.repeat(3)  # Repeat 3 times

    Reverse chirp (high to low frequency):

    .. code-block:: python

        >>> reverse_chirp = Chirp(
        ...     amplitude=3.0,
        ...     f_start=100 * u.Hz,
        ...     f_end=1 * u.Hz,
        ...     duration=2000 * u.ms
        ... )

    Test resonance in theta-gamma range:

    .. code-block:: python

        >>> resonance_test = Chirp(
        ...     amplitude=1.0,
        ...     f_start=4 * u.Hz,   # Theta start
        ...     f_end=80 * u.Hz,    # Gamma end
        ...     duration=10000 * u.ms,
        ...     method='logarithmic'
        ... )

    Windowed chirp for specific testing:

    .. code-block:: python

        >>> windowed_chirp = Chirp(
        ...     amplitude=2.0,
        ...     f_start=2 * u.Hz,
        ...     f_end=40 * u.Hz,
        ...     duration=3000 * u.ms,
        ...     t_start=500 * u.ms,
        ...     t_end=2500 * u.ms
        ... )

    Chirp with amplitude envelope:

    .. code-block:: python

        >>> from braintools.input import Ramp
        >>> chirp = Chirp(1.0, 5 * u.Hz, 50 * u.Hz, 1000 * u.ms)
        >>> envelope = Ramp(0.1, 1.0, 1000 * u.ms)
        >>> ramped_chirp = chirp * envelope

    Multiple chirps with different ranges:

    .. code-block:: python

        >>> low_chirp = Chirp(1.0, 0.5 * u.Hz, 5 * u.Hz, 2000 * u.ms)
        >>> high_chirp = Chirp(0.5, 20 * u.Hz, 100 * u.Hz, 2000 * u.ms)
        >>> broadband = low_chirp + high_chirp

    Chirp with positive bias:

    .. code-block:: python

        >>> positive_chirp = Chirp(
        ...     amplitude=5.0,
        ...     f_start=10 * u.Hz,
        ...     f_end=30 * u.Hz,
        ...     duration=1000 * u.ms,
        ...     bias=True  # Always positive
        ... )

    Notes
    -----
    - Linear chirp: frequency changes linearly with time
    - Logarithmic chirp: frequency changes exponentially
    - Phase is continuous throughout the sweep
    - Useful for finding resonant frequencies
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    Sinusoidal : Fixed frequency sinusoid
    NoisySinusoidal : Sinusoid with noise
    chirp : Functional API for chirp signal
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 f_start: u.Quantity,
                 f_end: u.Quantity,
                 duration: ArrayLike,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 method: str = 'linear',
                 bias: bool = False):
        super().__init__(duration)
        assert f_start.unit.dim == u.Hz.dim, f'Start frequency must be in Hz. Got {f_start.unit}.'
        assert f_end.unit.dim == u.Hz.dim, f'End frequency must be in Hz. Got {f_end.unit}.'

        self.amplitude = amplitude
        self.f_start = f_start
        self.f_end = f_end
        self.t_start = t_start
        self.t_end = t_end
        self.method = method
        self.bias = bias

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the chirp signal using the functional API."""
        return functional.chirp(
            amplitude=self.amplitude,
            f_start=self.f_start,
            f_end=self.f_end,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end,
            method=self.method,
            bias=self.bias
        )


class NoisySinusoidal(Input):
    r"""Composable sinusoidal input with additive noise.

    Creates a sinusoidal waveform with added Gaussian white noise. This class
    is composable with other Input objects.

    The output is:

    .. math::
        I(t) = A \sin(2\pi f t) + \mathcal{N}(0, \sigma^2)

    where \u03c3 is the noise amplitude.

    Parameters
    ----------
    amplitude : float
        Peak amplitude of the sinusoidal component.
    frequency : Quantity
        Frequency of the sinusoidal oscillation. Must be in Hz units.
    noise_amplitude : float
        Standard deviation of additive Gaussian noise.
    duration : float or Quantity
        Total duration of the signal. Supports time units.
    t_start : float or Quantity, optional
        Time when noisy sinusoid starts. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time when noisy sinusoid ends. After this, output is 0.
        Default is duration.
    seed : int, optional
        Random seed for reproducible noise.
        Default is None (uses global random state).

    Attributes
    ----------
    amplitude : float
        Peak amplitude of the sinusoid
    frequency : Quantity
        Frequency of oscillation in Hz
    noise_amplitude : float
        Standard deviation of the noise
    t_start : float or Quantity
        Start time of the signal
    t_end : float or Quantity
        End time of the signal
    seed : int or None
        Random seed for reproducibility

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Sinusoid with small noise:

    .. code-block:: python

        >>> noisy = NoisySinusoidal(
        ...     amplitude=10.0,
        ...     frequency=10 * u.Hz,
        ...     noise_amplitude=1.0,  # 10% noise
        ...     duration=1000 * u.ms
        ... )
        >>> signal = noisy()

    High noise for stochastic resonance:

    .. code-block:: python

        >>> stochastic = NoisySinusoidal(
        ...     amplitude=5.0,
        ...     frequency=5 * u.Hz,
        ...     noise_amplitude=10.0,  # Noise > signal
        ...     duration=2000 * u.ms
        ... )

    Filter noisy signal:

    .. code-block:: python

        >>> noisy = NoisySinusoidal(1.0, 20 * u.Hz, 0.5, 500 * u.ms)
        >>> filtered = noisy.smooth(tau=10 * u.ms)  # Low-pass filter

    Theta rhythm with synaptic noise:

    .. code-block:: python

        >>> theta_noisy = NoisySinusoidal(
        ...     amplitude=2.0,
        ...     frequency=8 * u.Hz,  # Theta frequency
        ...     noise_amplitude=0.5,
        ...     duration=5000 * u.ms
        ... )

    Combine multiple noisy oscillations:

    .. code-block:: python

        >>> theta = NoisySinusoidal(1.0, 8 * u.Hz, 0.2, 1000 * u.ms, seed=42)
        >>> gamma = NoisySinusoidal(0.5, 40 * u.Hz, 0.1, 1000 * u.ms, seed=43)
        >>> cross_frequency = theta + gamma

    Windowed noisy stimulation:

    .. code-block:: python

        >>> windowed_noisy = NoisySinusoidal(
        ...     amplitude=8.0,
        ...     frequency=20 * u.Hz,
        ...     noise_amplitude=2.0,
        ...     duration=1000 * u.ms,
        ...     t_start=200 * u.ms,
        ...     t_end=800 * u.ms
        ... )

    Reproducible noisy signal:

    .. code-block:: python

        >>> reproducible = NoisySinusoidal(
        ...     amplitude=15.0,
        ...     frequency=40 * u.Hz,
        ...     noise_amplitude=3.0,
        ...     duration=500 * u.ms,
        ...     seed=42  # Fixed random seed
        ... )

    Test signal detection in noise:

    .. code-block:: python

        >>> # Weak signal in strong noise
        >>> weak_signal = NoisySinusoidal(
        ...     amplitude=1.0,
        ...     frequency=10 * u.Hz,
        ...     noise_amplitude=5.0,  # 5x noise
        ...     duration=10000 * u.ms
        ... )

    Modulate noisy sinusoid:

    .. code-block:: python

        >>> from braintools.input import GaussianPulse
        >>> noisy = NoisySinusoidal(2.0, 30 * u.Hz, 0.5, 1000 * u.ms)
        >>> envelope = GaussianPulse(1.0, 500 * u.ms, 100 * u.ms, 1000 * u.ms)
        >>> burst = noisy * envelope

    Notes
    -----
    - Noise is Gaussian white noise with zero mean
    - Signal-to-noise ratio = amplitude / noise_amplitude
    - Total variance = amplitude²/2 + noise_amplitude²
    - Useful for testing noise robustness
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    Sinusoidal : Clean sinusoidal wave
    WienerProcess : Pure noise process
    noisy_sinusoidal : Functional API for noisy sinusoid
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitude: float,
                 frequency: u.Quantity,
                 noise_amplitude: float,
                 duration: ArrayLike,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 seed: Optional[int] = None):
        super().__init__(duration)
        assert frequency.unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {frequency.unit}.'

        self.amplitude = amplitude
        self.frequency = frequency
        self.noise_amplitude = noise_amplitude
        self.t_start = t_start
        self.t_end = t_end
        self.seed = seed

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the noisy sinusoidal using the functional API."""
        return functional.noisy_sinusoidal(
            amplitude=self.amplitude,
            frequency=self.frequency,
            noise_amplitude=self.noise_amplitude,
            duration=self.duration,
            t_start=self.t_start,
            t_end=self.t_end,
            seed=self.seed
        )


SinusoidalInput = create_deprecated_class(Sinusoidal, 'SinusoidalInput', 'Sinusoidal')
SquareInput = create_deprecated_class(Square, 'SquareInput', 'Square')
TriangularInput = create_deprecated_class(Triangular, 'TriangularInput', 'Triangular')
SawtoothInput = create_deprecated_class(Sawtooth, 'SawtoothInput', 'Sawtooth')
ChirpInput = create_deprecated_class(Chirp, 'ChirpInput', 'Chirp')
NoisySinusoidalInput = create_deprecated_class(NoisySinusoidal, 'NoisySinusoidalInput', 'NoisySinusoidal')

__all__.extend(
    ['SinusoidalInput', 'SquareInput', 'TriangularInput', 'SawtoothInput', 'ChirpInput', 'NoisySinusoidalInput'])
