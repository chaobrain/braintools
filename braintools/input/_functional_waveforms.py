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
Waveform input generators (sinusoidal, square, triangular, etc.).
"""

from typing import Optional

import brainstate
import brainunit as u
import numpy as np

from braintools._misc import set_module_as
from ._deprecation import create_deprecated_function

__all__ = [
    'sinusoidal',
    'square',
    'triangular',
    'sawtooth',
    'chirp',
    'noisy_sinusoidal',
]


@set_module_as('braintools.input')
def sinusoidal(
    amplitude: brainstate.typing.ArrayLike,
    frequency: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    bias: bool = False
):
    """Generate sinusoidal current input.

    Creates a sinusoidal waveform with specified amplitude and frequency.
    Useful for testing frequency response, resonance properties, and 
    oscillatory behavior of neural models.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the sinusoidal wave. Supports current units.
    frequency : float or Quantity
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

    Returns
    -------
    current : ndarray or Quantity
        The sinusoidal current array with shape (n_timesteps,).

    Raises
    ------
    AssertionError
        If frequency is not in Hz units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple 10 Hz sinusoid

    .. code-block:: python

        >>> current = sinusoidal(
        ...     amplitude=5 * u.pA,
        ...     frequency=10 * u.Hz,
        ...     duration=1000 * u.ms
        ... )
    
    High-frequency stimulation

    .. code-block:: python

        >>> current = sinusoidal(
        ...     amplitude=2 * u.nA,
        ...     frequency=100 * u.Hz,
        ...     duration=500 * u.ms
        ... )
    
    Sinusoid with positive bias (always >= 0)

    .. code-block:: python

        >>> current = sinusoidal(
        ...     amplitude=10 * u.pA,
        ...     frequency=5 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     bias=True  # Oscillates between 0 and 20 pA
        ... )
    
    Windowed sinusoid

    .. code-block:: python

        >>> current = sinusoidal(
        ...     amplitude=8 * u.pA,
        ...     frequency=20 * u.Hz,
        ...     duration=1000 * u.ms,
        ...     t_start=200 * u.ms,
        ...     t_end=800 * u.ms
        ... )
    
    Testing resonance at theta frequency

    .. code-block:: python

        >>> current = sinusoidal(
        ...     amplitude=1 * u.nA,
        ...     frequency=8 * u.Hz,  # Theta band
        ...     duration=5000 * u.ms
        ... )
    
    Subthreshold membrane oscillations

    .. code-block:: python

        >>> current = sinusoidal(
        ...     amplitude=0.1 * u.nA,
        ...     frequency=40 * u.Hz,  # Gamma band
        ...     duration=200 * u.ms,
        ...     t_start=50 * u.ms,
        ...     t_end=150 * u.ms
        ... )
    
    Notes
    -----
    - The phase starts at 0 (sine wave starts at 0)
    - Frequency must be less than Nyquist frequency (1/(2*dt))
    - With bias=True, output ranges from 0 to 2*amplitude
    - Without bias, output ranges from -amplitude to +amplitude
    """
    # Get dt and time unit
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    # Convert to mantissa values
    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Check frequency unit
    freq_unit = u.get_unit(frequency)
    assert freq_unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {freq_unit}.'
    freq_value = u.Quantity(frequency).to(u.Hz).mantissa

    # Extract amplitude and unit
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)

    # Calculate indices
    n_steps = int(np.ceil(duration_value / dt_value))
    start_i = int(t_start_value / dt_value)
    end_i = int(t_end_value / dt_value)

    # Generate time array for active window
    times = np.arange(end_i - start_i) * dt_value

    # Generate sinusoidal wave
    # Convert frequency from Hz to match time unit
    if time_unit == u.ms:
        freq_factor = freq_value / 1000.0  # Hz to kHz for ms
    elif time_unit == u.second:
        freq_factor = freq_value
    else:
        # General conversion
        freq_factor = freq_value * u.Quantity(1 * time_unit).to(u.second).mantissa

    sin_values = amplitude_value * np.sin(2 * np.pi * freq_factor * times)

    if bias:
        sin_values += amplitude_value

    # Create full array with zeros outside window
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())
    currents[start_i:end_i] = sin_values

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def _square_wave(t, duty=0.5):
    """Helper function to generate square wave."""
    t = np.asarray(t)
    y = np.zeros(t.shape, dtype=brainstate.environ.dftype())

    # Calculate phase in [0, 2*pi]
    tmod = np.mod(t, 2 * np.pi)

    # Set to 1 for first part of period, -1 for second part
    mask_positive = tmod < (duty * 2 * np.pi)
    y[mask_positive] = 1
    y[~mask_positive] = -1

    return y


@set_module_as('braintools.input')
def square(
    amplitude: brainstate.typing.ArrayLike,
    frequency: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    duty_cycle: float = 0.5,
    bias: bool = False
):
    """Generate square wave current input.

    Creates a square wave that alternates between two levels at a specified
    frequency. Useful for testing step responses, synchronization, and
    driving rhythmic activity.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the square wave. Supports current units.
    frequency : float or Quantity
        Frequency of oscillation. Must be in Hz units.
    duration : float or Quantity
        Total duration of the signal. Supports time units.
    t_start : float or Quantity, optional
        Time when square wave starts. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time when square wave ends. After this, output is 0.
        Default is duration.
    duty_cycle : float, optional
        Fraction of period spent at high level (0 to 1).
        Default is 0.5 (symmetric square wave).
    bias : bool, optional
        If True, adds DC offset equal to amplitude (non-negative output).
        If False, alternates between +amplitude and -amplitude.

    Returns
    -------
    current : ndarray or Quantity
        The square wave current array with shape (n_timesteps,).

    Raises
    ------
    AssertionError
        If frequency is not in Hz units.
    ValueError
        If duty_cycle is not between 0 and 1.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Symmetric square wave at 2 Hz

    .. code-block:: python

        >>> current = square(
        ...     amplitude=10 * u.pA,
        ...     frequency=2 * u.Hz,
        ...     duration=2000 * u.ms
        ... )
    
    High-frequency pulse train

    .. code-block:: python

        >>> current = square(
        ...     amplitude=5 * u.nA,
        ...     frequency=50 * u.Hz,
        ...     duration=500 * u.ms,
        ...     duty_cycle=0.2  # 20% on, 80% off
        ... )
    
    Square wave with positive bias

    .. code-block:: python

        >>> current = square(
        ...     amplitude=8 * u.pA,
        ...     frequency=10 * u.Hz,
        ...     duration=1000 * u.ms,
        ...     bias=True  # Alternates between 0 and 16 pA
        ... )
    
    Windowed stimulation

    .. code-block:: python

        >>> current = square(
        ...     amplitude=3 * u.nA,
        ...     frequency=5 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     t_start=500 * u.ms,
        ...     t_end=1500 * u.ms
        ... )
    
    Asymmetric pulse train (10% duty cycle)

    .. code-block:: python

        >>> current = square(
        ...     amplitude=20 * u.pA,
        ...     frequency=1 * u.Hz,
        ...     duration=5000 * u.ms,
        ...     duty_cycle=0.1  # Short pulses
        ... )
    
    Clock signal for synchronization

    .. code-block:: python

        >>> current = square(
        ...     amplitude=1 * u.nA,
        ...     frequency=40 * u.Hz,
        ...     duration=250 * u.ms,
        ...     duty_cycle=0.5
        ... )
    
    Notes
    -----
    - Without bias: alternates between +amplitude and -amplitude
    - With bias: alternates between 0 and 2*amplitude
    - Duty cycle controls the fraction of time at high level
    - Transitions are instantaneous (limited by dt resolution)
    """
    # Validate duty cycle
    if not 0 <= duty_cycle <= 1:
        raise ValueError(f"duty_cycle must be between 0 and 1, got {duty_cycle}")

    # Get dt and time unit
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    # Convert to mantissa values
    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Check frequency unit
    freq_unit = u.get_unit(frequency)
    assert freq_unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {freq_unit}.'
    freq_value = u.Quantity(frequency).to(u.Hz).mantissa

    # Extract amplitude and unit
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)

    # Calculate indices
    n_steps = int(np.ceil(duration_value / dt_value))
    start_i = int(t_start_value / dt_value)
    end_i = int(t_end_value / dt_value)

    # Generate time array for active window
    times = np.arange(end_i - start_i) * dt_value

    # Convert frequency to match time unit
    if time_unit == u.ms:
        freq_factor = freq_value / 1000.0  # Hz to kHz for ms
    elif time_unit == u.second:
        freq_factor = freq_value
    else:
        freq_factor = freq_value * u.Quantity(1 * time_unit).to(u.second).mantissa

    # Generate square wave
    phase = 2 * np.pi * freq_factor * times
    square_values = amplitude_value * _square_wave(phase, duty=duty_cycle)

    if bias:
        square_values += amplitude_value

    # Create full array with zeros outside window
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())
    currents[start_i:end_i] = square_values

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def triangular(
    amplitude: brainstate.typing.ArrayLike,
    frequency: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    bias: bool = False
):
    """Generate triangular wave current input.

    Creates a triangular (linear ramping) waveform that linearly increases
    and decreases between peak values. Useful for testing linear responses
    and ramp sensitivity.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the triangular wave. Supports current units.
    frequency : float or Quantity
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

    Returns
    -------
    current : ndarray or Quantity
        The triangular wave current array with shape (n_timesteps,).

    Raises
    ------
    AssertionError
        If frequency is not in Hz units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple triangular wave at 5 Hz

    .. code-block:: python

        >>> current = triangular(
        ...     amplitude=10 * u.pA,
        ...     frequency=5 * u.Hz,
        ...     duration=1000 * u.ms
        ... )
    
    Slow triangular ramp for I-V curves

    .. code-block:: python

        >>> current = triangular(
        ...     amplitude=100 * u.pA,
        ...     frequency=0.5 * u.Hz,  # 2 second period
        ...     duration=4000 * u.ms
        ... )
    
    Triangular wave with positive bias

    .. code-block:: python

        >>> current = triangular(
        ...     amplitude=5 * u.nA,
        ...     frequency=10 * u.Hz,
        ...     duration=500 * u.ms,
        ...     bias=True  # Oscillates between 0 and 10 nA
        ... )
    
    Windowed triangular stimulation

    .. code-block:: python

        >>> current = triangular(
        ...     amplitude=8 * u.pA,
        ...     frequency=2 * u.Hz,
        ...     duration=3000 * u.ms,
        ...     t_start=500 * u.ms,
        ...     t_end=2500 * u.ms
        ... )
    
    High-frequency triangular wave

    .. code-block:: python

        >>> current = triangular(
        ...     amplitude=2 * u.nA,
        ...     frequency=50 * u.Hz,
        ...     duration=200 * u.ms
        ... )
    
    Testing adaptation with slow ramps

    .. code-block:: python

        >>> current = triangular(
        ...     amplitude=15 * u.pA,
        ...     frequency=1 * u.Hz,
        ...     duration=5000 * u.ms
        ... )
    
    Notes
    -----
    - The wave ramps linearly between -amplitude and +amplitude
    - With bias=True, ramps between 0 and 2*amplitude
    - Peaks occur at 0, 0.5/frequency, 1/frequency, etc.
    - More suitable than sawtooth for symmetric ramping
    """
    # Get dt and time unit
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    # Convert to mantissa values
    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Check frequency unit
    freq_unit = u.get_unit(frequency)
    assert freq_unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {freq_unit}.'
    freq_value = u.Quantity(frequency).to(u.Hz).mantissa

    # Extract amplitude and unit
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)

    # Calculate indices
    n_steps = int(np.ceil(duration_value / dt_value))
    start_i = int(t_start_value / dt_value)
    end_i = int(t_end_value / dt_value)

    # Generate time array for active window
    times = np.arange(end_i - start_i) * dt_value

    # Convert frequency to match time unit
    if time_unit == u.ms:
        freq_factor = freq_value / 1000.0
    elif time_unit == u.second:
        freq_factor = freq_value
    else:
        freq_factor = freq_value * u.Quantity(1 * time_unit).to(u.second).mantissa

    # Generate triangular wave using arcsin(sin(x))
    phase = 2 * np.pi * freq_factor * times
    triangular_values = (2.0 * amplitude_value / np.pi) * np.arcsin(np.sin(phase))

    if bias:
        triangular_values += amplitude_value

    # Create full array with zeros outside window
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())
    currents[start_i:end_i] = triangular_values

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def sawtooth(
    amplitude: brainstate.typing.ArrayLike,
    frequency: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    bias: bool = False
):
    """Generate sawtooth wave current input.

    Creates a sawtooth waveform that ramps up linearly and then drops sharply.
    Useful for testing reset dynamics, asymmetric responses, and ramp sensitivity.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the sawtooth wave. Supports current units.
    frequency : float or Quantity
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

    Returns
    -------
    current : ndarray or Quantity
        The sawtooth wave current array with shape (n_timesteps,).

    Raises
    ------
    AssertionError
        If frequency is not in Hz units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple sawtooth at 2 Hz

    .. code-block:: python

        >>> current = sawtooth(
        ...     amplitude=10 * u.pA,
        ...     frequency=2 * u.Hz,
        ...     duration=2000 * u.ms
        ... )
    
    Slow ramp for threshold detection

    .. code-block:: python

        >>> current = sawtooth(
        ...     amplitude=50 * u.pA,
        ...     frequency=0.5 * u.Hz,  # 2 second ramp
        ...     duration=4000 * u.ms
        ... )
    
    Sawtooth with positive bias

    .. code-block:: python

        >>> current = sawtooth(
        ...     amplitude=5 * u.nA,
        ...     frequency=10 * u.Hz,
        ...     duration=500 * u.ms,
        ...     bias=True  # Ramps from 0 to 10 nA
        ... )
    
    Windowed sawtooth stimulation

    .. code-block:: python

        >>> current = sawtooth(
        ...     amplitude=8 * u.pA,
        ...     frequency=5 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     t_start=400 * u.ms,
        ...     t_end=1600 * u.ms
        ... )
    
    Fast sawtooth for reset testing

    .. code-block:: python

        >>> current = sawtooth(
        ...     amplitude=20 * u.pA,
        ...     frequency=20 * u.Hz,
        ...     duration=250 * u.ms
        ... )
    
    Repeated ramp protocol

    .. code-block:: python

        >>> current = sawtooth(
        ...     amplitude=100 * u.pA,
        ...     frequency=1 * u.Hz,
        ...     duration=10000 * u.ms
        ... )
    
    Notes
    -----
    - Ramps linearly from -amplitude to +amplitude, then resets
    - With bias=True, ramps from 0 to 2*amplitude
    - The ramp is continuous, reset is instantaneous
    - Useful for finding thresholds and testing reset mechanisms
    """
    # Get dt and time unit
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    # Convert to mantissa values
    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Check frequency unit
    freq_unit = u.get_unit(frequency)
    assert freq_unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {freq_unit}.'
    freq_value = u.Quantity(frequency).to(u.Hz).mantissa

    # Extract amplitude and unit
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)

    # Calculate indices
    n_steps = int(np.ceil(duration_value / dt_value))
    start_i = int(t_start_value / dt_value)
    end_i = int(t_end_value / dt_value)

    # Generate time array for active window
    times = np.arange(end_i - start_i) * dt_value

    # Convert frequency to match time unit
    if time_unit == u.ms:
        freq_factor = freq_value / 1000.0
    elif time_unit == u.second:
        freq_factor = freq_value
    else:
        freq_factor = freq_value * u.Quantity(1 * time_unit).to(u.second).mantissa

    # Generate sawtooth wave
    phase = freq_factor * times
    sawtooth_values = 2 * amplitude_value * (phase - np.floor(phase) - 0.5)

    if bias:
        sawtooth_values += amplitude_value

    # Create full array with zeros outside window
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())
    currents[start_i:end_i] = sawtooth_values

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def chirp(
    amplitude: brainstate.typing.ArrayLike,
    f_start: brainstate.typing.ArrayLike,
    f_end: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    method: str = 'linear',
    bias: bool = False
):
    """Generate chirp (frequency sweep) current input.

    Creates a sinusoidal signal with time-varying frequency that sweeps from
    a starting to ending frequency. Useful for frequency response analysis,
    resonance detection, and spectral characterization.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the chirp signal. Supports current units.
    f_start : float or Quantity
        Starting frequency. Must be in Hz units.
    f_end : float or Quantity
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

    Returns
    -------
    current : ndarray or Quantity
        The chirp signal current array with shape (n_timesteps,).

    Raises
    ------
    AssertionError
        If frequencies are not in Hz units.
    ValueError
        If method is not 'linear' or 'logarithmic'.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Linear frequency sweep from 1 to 50 Hz

    .. code-block:: python

        >>> current = chirp(
        ...     amplitude=5 * u.pA,
        ...     f_start=1 * u.Hz,
        ...     f_end=50 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     method='linear'
        ... )
    
    Logarithmic sweep for spectral analysis

    .. code-block:: python

        >>> current = chirp(
        ...     amplitude=2 * u.nA,
        ...     f_start=0.1 * u.Hz,
        ...     f_end=100 * u.Hz,
        ...     duration=5000 * u.ms,
        ...     method='logarithmic'
        ... )
    
    Chirp with positive bias

    .. code-block:: python

        >>> current = chirp(
        ...     amplitude=10 * u.pA,
        ...     f_start=5 * u.Hz,
        ...     f_end=20 * u.Hz,
        ...     duration=1000 * u.ms,
        ...     bias=True  # Always positive
        ... )
    
    Windowed chirp for specific testing

    .. code-block:: python

        >>> current = chirp(
        ...     amplitude=8 * u.pA,
        ...     f_start=2 * u.Hz,
        ...     f_end=40 * u.Hz,
        ...     duration=3000 * u.ms,
        ...     t_start=500 * u.ms,
        ...     t_end=2500 * u.ms
        ... )
    
    Reverse chirp (high to low frequency)

    .. code-block:: python

        >>> current = chirp(
        ...     amplitude=3 * u.nA,
        ...     f_start=100 * u.Hz,
        ...     f_end=1 * u.Hz,
        ...     duration=2000 * u.ms
        ... )
    
    Testing resonance in theta-gamma range

    .. code-block:: python

        >>> current = chirp(
        ...     amplitude=1 * u.nA,
        ...     f_start=4 * u.Hz,   # Theta start
        ...     f_end=80 * u.Hz,    # Gamma end
        ...     duration=10000 * u.ms,
        ...     method='logarithmic'
        ... )
    
    Notes
    -----
    - Linear chirp: frequency changes linearly with time
    - Logarithmic chirp: frequency changes exponentially with time
    - Useful for finding resonant frequencies and transfer functions
    - Phase is continuous throughout the sweep
    """
    # Check method
    if method not in ['linear', 'logarithmic']:
        raise ValueError(f"method must be 'linear' or 'logarithmic', got {method}")

    # Get dt and time unit
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    # Convert to mantissa values
    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Check frequency units
    f_start_unit = u.get_unit(f_start)
    f_end_unit = u.get_unit(f_end)
    assert f_start_unit.dim == u.Hz.dim, f'Start frequency must be in Hz. Got {f_start_unit}.'
    assert f_end_unit.dim == u.Hz.dim, f'End frequency must be in Hz. Got {f_end_unit}.'

    f_start_value = u.Quantity(f_start).to(u.Hz).mantissa
    f_end_value = u.Quantity(f_end).to(u.Hz).mantissa

    # Extract amplitude and unit
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)

    # Calculate indices
    n_steps = int(np.ceil(duration_value / dt_value))
    start_i = int(t_start_value / dt_value)
    end_i = int(t_end_value / dt_value)

    # Generate time array for active window
    times = np.arange(end_i - start_i) * dt_value
    sweep_duration = t_end_value - t_start_value

    # Convert frequencies to match time unit
    if time_unit == u.ms:
        f0 = f_start_value / 1000.0  # Hz to kHz
        f1 = f_end_value / 1000.0
    elif time_unit == u.second:
        f0 = f_start_value
        f1 = f_end_value
    else:
        time_factor = u.Quantity(1 * time_unit).to(u.second).mantissa
        f0 = f_start_value * time_factor
        f1 = f_end_value * time_factor

    # Generate chirp based on method
    if method == 'linear':
        # Linear chirp: f(t) = f0 + (f1-f0)*t/T
        phase = 2 * np.pi * (f0 * times + 0.5 * (f1 - f0) * times ** 2 / sweep_duration)
    else:  # logarithmic
        # Logarithmic chirp
        if f0 <= 0 or f1 <= 0:
            raise ValueError("Logarithmic chirp requires positive frequencies")
        k = (f1 / f0) ** (1 / sweep_duration)
        phase = 2 * np.pi * f0 * (k ** times - 1) / np.log(k)

    chirp_values = amplitude_value * np.sin(phase)

    if bias:
        chirp_values += amplitude_value

    # Create full array with zeros outside window
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())
    currents[start_i:end_i] = chirp_values

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def noisy_sinusoidal(
    amplitude: brainstate.typing.ArrayLike,
    frequency: brainstate.typing.ArrayLike,
    noise_amplitude: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    seed: Optional[int] = None
):
    """Generate sinusoidal current input with additive noise.

    Creates a sinusoidal waveform with added Gaussian white noise. Useful for
    testing robustness to noise, stochastic resonance, and realistic synaptic
    input conditions.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the sinusoidal component. Supports current units.
    frequency : float or Quantity
        Frequency of the sinusoidal oscillation. Must be in Hz units.
    noise_amplitude : float or Quantity
        Standard deviation of additive Gaussian noise. Must have same units as amplitude.
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

    Returns
    -------
    current : ndarray or Quantity
        The noisy sinusoidal current array with shape (n_timesteps,).

    Raises
    ------
    AssertionError
        If frequency is not in Hz units.
    UnitMismatchError
        If amplitude and noise_amplitude have different units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Sinusoid with small noise

    .. code-block:: python

        >>> current = noisy_sinusoidal(
        ...     amplitude=10 * u.pA,
        ...     frequency=10 * u.Hz,
        ...     noise_amplitude=1 * u.pA,  # 10% noise
        ...     duration=1000 * u.ms
        ... )
    
    High noise for stochastic resonance

    .. code-block:: python

        >>> current = noisy_sinusoidal(
        ...     amplitude=5 * u.pA,
        ...     frequency=5 * u.Hz,
        ...     noise_amplitude=10 * u.pA,  # Noise > signal
        ...     duration=2000 * u.ms
        ... )
    
    Theta rhythm with synaptic noise

    .. code-block:: python

        >>> current = noisy_sinusoidal(
        ...     amplitude=2 * u.nA,
        ...     frequency=8 * u.Hz,  # Theta frequency
        ...     noise_amplitude=0.5 * u.nA,
        ...     duration=5000 * u.ms
        ... )
    
    Windowed noisy stimulation

    .. code-block:: python

        >>> current = noisy_sinusoidal(
        ...     amplitude=8 * u.pA,
        ...     frequency=20 * u.Hz,
        ...     noise_amplitude=2 * u.pA,
        ...     duration=1000 * u.ms,
        ...     t_start=200 * u.ms,
        ...     t_end=800 * u.ms
        ... )
    
    Reproducible noisy signal

    .. code-block:: python

        >>> current = noisy_sinusoidal(
        ...     amplitude=15 * u.pA,
        ...     frequency=40 * u.Hz,
        ...     noise_amplitude=3 * u.pA,
        ...     duration=500 * u.ms,
        ...     seed=42  # Fixed random seed
        ... )
    
    Subthreshold oscillation with realistic noise

    .. code-block:: python

        >>> current = noisy_sinusoidal(
        ...     amplitude=0.1 * u.nA,
        ...     frequency=60 * u.Hz,  # Gamma frequency
        ...     noise_amplitude=0.05 * u.nA,
        ...     duration=2000 * u.ms,
        ...     t_start=500 * u.ms
        ... )
    
    Notes
    -----
    - Noise is Gaussian white noise with zero mean
    - Signal-to-noise ratio = amplitude / noise_amplitude
    - Total variance = amplitude²/2 + noise_amplitude²
    - Useful for testing noise robustness and filtering properties
    """
    # Get dt and time unit
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    # Convert to mantissa values
    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Check frequency unit
    freq_unit = u.get_unit(frequency)
    assert freq_unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {freq_unit}.'
    freq_value = u.Quantity(frequency).to(u.Hz).mantissa

    # Extract amplitude and unit
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)
    noise_amplitude_value = u.Quantity(noise_amplitude).to(c_unit).mantissa

    # Setup random number generator
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Calculate indices
    n_steps = int(np.ceil(duration_value / dt_value))
    start_i = int(t_start_value / dt_value)
    end_i = int(t_end_value / dt_value)

    # Generate time array for active window
    times = np.arange(end_i - start_i) * dt_value

    # Convert frequency to match time unit
    if time_unit == u.ms:
        freq_factor = freq_value / 1000.0
    elif time_unit == u.second:
        freq_factor = freq_value
    else:
        freq_factor = freq_value * u.Quantity(1 * time_unit).to(u.second).mantissa

    # Generate sinusoidal component
    sin_component = amplitude_value * np.sin(2 * np.pi * freq_factor * times)

    # Add noise
    noise = noise_amplitude_value * rng.standard_normal(len(times))
    noisy_signal = sin_component + noise

    # Create full array with zeros outside window
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())
    currents[start_i:end_i] = noisy_signal

    return u.maybe_decimal(currents * c_unit)


sinusoidal_input = create_deprecated_function(sinusoidal, 'sinusoidal_input', 'sinusoidal')
square_input = create_deprecated_function(square, 'square_input', 'square')
triangular_input = create_deprecated_function(triangular, 'triangular_input', 'triangular')
sawtooth_input = create_deprecated_function(sawtooth, 'sawtooth_input', 'sawtooth')
chirp_input = create_deprecated_function(chirp, 'chirp_input', 'chirp')

__all__.extend(['sinusoidal', 'square', 'triangular', 'sawtooth', 'chirp'])
