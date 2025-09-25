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
Waveform input generators (sinusoidal, square, triangular, etc.).
"""

import brainstate
import brainunit as u
import numpy as np

__all__ = [
    'sinusoidal_input',
    'square_input',
    'triangular_input',
    'sawtooth_input',
    'chirp_input',
    'noisy_sinusoidal',
]


def sinusoidal_input(
    amplitude,
    frequency,
    duration,
    dt=None,
    t_start=None,
    t_end=None,
    bias=False
):
    """Sinusoidal input.

    Parameters
    ----------
    amplitude: float
      Amplitude of the sinusoid.
    frequency: Quantity
      Frequency of the sinus oscillation, in Hz
    duration: Quantity
      The input duration.
    t_start: Quantity
      The start time. Default is 0.
    t_end: Quantity
      The end time.
    dt: Quantity
      The numerical precision.
    bias: bool
      Whether the sinusoid oscillates around 0 (False), or
      has a positive DC bias, thus non-negative (True).
    """
    assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_start = 0. * u.ms if t_start is None else t_start
    if t_end is None:
        t_end = duration
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    start_i = int(t_start / dt)
    end_i = int(t_end / dt)
    sin_inputs = amplitude * u.math.sin(2 * u.math.pi * times * frequency)
    if bias:
        sin_inputs += amplitude
    currents = u.math.zeros(int(duration / dt),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(sin_inputs))
    currents = currents.at[start_i:end_i].set(sin_inputs)
    return currents


def _square(t, duty=0.5):
    t, w = np.asarray(t), np.asarray(duty)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in 'fFdD':
        ytype = t.dtype.char
    else:
        ytype = 'd'

    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # on the interval 0 to duty*2*pi function is 1
    tmod = np.mod(t, 2 * np.pi)
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    np.place(y, mask2, 1)

    # on the interval duty*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    np.place(y, mask3, -1)
    return y


def square_input(
    amplitude,
    frequency,
    duration,
    dt=None,
    bias=False,
    t_start=None,
    t_end=None
):
    """Oscillatory square input.

    Parameters
    ----------
    amplitude: float
      Amplitude of the square oscillation.
    frequency: Quantity
      Frequency of the square oscillation, in Hz.
    duration: Quantity
      The input duration.
    t_start: Quantity
      The start time.
    t_end: Quantity
      The end time.
    dt: Quantity
      The numerical precision.
    bias: bool
      Whether the sinusoid oscillates around 0 (False), or
      has a positive DC bias, thus non-negative (True).
    """
    if t_start is None:
        t_start = 0. * u.ms
    assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
    dt = brainstate.environ.get_dt() if dt is None else dt
    if t_end is None:
        t_end = duration
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    sin_inputs = amplitude * _square(2 * np.pi * times * frequency)
    if bias:
        sin_inputs += amplitude
    currents = u.math.zeros(int(duration / dt), dtype=brainstate.environ.dftype())
    start_i = int(t_start / dt)
    end_i = int(t_end / dt)
    currents = currents.at[start_i:end_i].set(sin_inputs)
    return currents


def triangular_input(
    amplitude,
    frequency,
    duration,
    dt=None,
    t_start=None,
    t_end=None,
    bias=False
):
    """Triangular wave input.

    Parameters
    ----------
    amplitude: float
        Amplitude of the triangular wave.
    frequency: Quantity
        Frequency of the triangular oscillation, in Hz.
    duration: Quantity
        The input duration.
    dt: Quantity
        The numerical precision.
    t_start: Quantity
        The start time.
    t_end: Quantity
        The end time.
    bias: bool
        Whether the wave oscillates around 0 (False), or
        has a positive DC bias, thus non-negative (True).
    
    Returns
    -------
    current : array
        The formatted triangular wave input.
    """
    if t_start is None:
        t_start = 0. * u.ms
    assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    
    # Generate triangular wave using arcsin
    period = 1.0 / frequency
    phase = 2 * u.math.pi * times * frequency
    triangular = (2.0 * amplitude / u.math.pi) * u.math.arcsin(u.math.sin(phase))
    
    if bias:
        triangular += amplitude
    
    currents = u.math.zeros(int(duration / dt),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(triangular))
    start_i = int(t_start / dt)
    end_i = int(t_end / dt)
    currents = currents.at[start_i:end_i].set(triangular)
    return currents


def sawtooth_input(
    amplitude,
    frequency,
    duration,
    dt=None,
    t_start=None,
    t_end=None,
    bias=False
):
    """Sawtooth wave input.

    Parameters
    ----------
    amplitude: float
        Amplitude of the sawtooth wave.
    frequency: Quantity
        Frequency of the sawtooth oscillation, in Hz.
    duration: Quantity
        The input duration.
    dt: Quantity
        The numerical precision.
    t_start: Quantity
        The start time.
    t_end: Quantity
        The end time.
    bias: bool
        Whether the wave oscillates around 0 (False), or
        has a positive DC bias, thus non-negative (True).
    
    Returns
    -------
    current : array
        The formatted sawtooth wave input.
    """
    if t_start is None:
        t_start = 0. * u.ms
    assert frequency.unit.dim == u.Hz.dim, f'The frequency must be in Hz. But got {frequency.unit}.'
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    
    # Generate sawtooth wave
    phase = times * frequency
    sawtooth = 2 * amplitude * (phase - u.math.floor(phase) - 0.5)
    
    if bias:
        sawtooth += amplitude
    
    currents = u.math.zeros(int(duration / dt),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(sawtooth))
    start_i = int(t_start / dt)
    end_i = int(t_end / dt)
    currents = currents.at[start_i:end_i].set(sawtooth)
    return currents


def chirp_input(
    amplitude,
    f_start,
    f_end,
    duration,
    dt=None,
    t_start=None,
    t_end=None,
    method='linear',
    bias=False
):
    """Chirp signal (frequency sweep) input.

    Parameters
    ----------
    amplitude: float
        Amplitude of the chirp signal.
    f_start: Quantity
        Starting frequency in Hz.
    f_end: Quantity
        Ending frequency in Hz.
    duration: float
        Total duration of the input.
    dt: float
        The numerical precision.
    t_start: float
        The start time.
    t_end: float
        The end time.
    method: str
        'linear' for linear frequency sweep, 'logarithmic' for logarithmic sweep.
    bias: bool
        Whether to add positive DC bias.
    
    Returns
    -------
    current : array
        The chirp signal input.
    """
    if t_start is None:
        t_start = 0. * u.ms
    assert f_start.unit.dim == u.Hz.dim, f'Start frequency must be in Hz. Got {f_start.unit}.'
    assert f_end.unit.dim == u.Hz.dim, f'End frequency must be in Hz. Got {f_end.unit}.'
    
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    
    f0 = f_start / u.Hz  # Convert to dimensionless Hz value
    f1 = f_end / u.Hz    # Convert to dimensionless Hz value
    T = (t_end - t_start) / u.ms
    times_ms = times / u.ms
    
    if method == 'linear':
        # Linear chirp: f(t) = f0 + (f1-f0)*t/T
        # phase needs to be dimensionless, frequencies in Hz, times in ms
        phase = 2 * u.math.pi * (f0 * times_ms / 1000 + 0.5 * (f1 - f0) * times_ms**2 / (T * 1000))
    elif method == 'logarithmic':
        # Logarithmic chirp
        k = (f1 / f0) ** (1 / T)
        phase = 2 * u.math.pi * f0 * (k**times_ms - 1) / (u.math.log(k) * 1000)
    else:
        raise ValueError(f"method must be 'linear' or 'logarithmic', got {method}")
    
    chirp = amplitude * u.math.sin(phase)
    
    if bias:
        chirp += amplitude
    
    currents = u.math.zeros(int(duration / dt),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(chirp))
    start_i = int(t_start / dt)
    end_i = int(t_end / dt)
    currents = currents.at[start_i:end_i].set(chirp)
    return currents


def noisy_sinusoidal(
    amplitude,
    frequency,
    noise_amplitude,
    duration,
    dt=None,
    t_start=None,
    t_end=None,
    seed=None
):
    """Sinusoidal input with additive noise.

    Parameters
    ----------
    amplitude: float
        Amplitude of the sinusoid.
    frequency: Quantity
        Frequency of the sinus oscillation, in Hz.
    noise_amplitude: float
        Amplitude of the additive Gaussian noise.
    duration: float
        The input duration.
    dt: float
        The numerical precision.
    t_start: float
        The start time.
    t_end: float
        The end time.
    seed: int
        Random seed for noise generation.
    
    Returns
    -------
    current : array
        The noisy sinusoidal input.
    """
    if t_start is None:
        t_start = 0. * u.ms
    assert frequency.unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {frequency.unit}.'
    
    if seed is None:
        rng = brainstate.random.DEFAULT
    else:
        rng = brainstate.random.RandomState(seed)
    
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    
    # Generate sinusoidal component
    sin_component = amplitude * u.math.sin(2 * u.math.pi * times * frequency)
    
    # Add noise
    noise = noise_amplitude * rng.standard_normal(len(times))
    noisy_signal = sin_component + noise
    
    currents = u.math.zeros(int(duration / dt),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(noisy_signal))
    start_i = int(t_start / dt)
    end_i = int(t_end / dt)
    currents = currents.at[start_i:end_i].set(noisy_signal)
    return currents