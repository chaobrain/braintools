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
Pulse and burst input generators.
"""

from typing import Optional, Union, Sequence

import brainstate
import brainunit as u
import numpy as np

from braintools._misc import set_module_as
from ._deprecation import create_deprecated_function

__all__ = [
    'spike',
    'gaussian_pulse',
    'exponential_decay',
    'double_exponential',
    'burst',
]


@set_module_as('braintools.input')
def spike(
    sp_times: Sequence,
    sp_lens: Union[float, Sequence],
    sp_sizes: Union[float, Sequence],
    duration: brainstate.typing.ArrayLike,
):
    """Format current input like a series of short-time spikes.

    Creates a series of rectangular current pulses (spikes) at specified times
    with specified durations and amplitudes. Useful for simulating synaptic
    inputs or direct current injection protocols.

    Parameters
    ----------
    sp_times : list or array-like
        The spike time-points. Supports time units (e.g., ms).
    sp_lens : float or list
        The length of each spike. If scalar, same duration for all spikes.
        If list, must match length of sp_times. Supports time units.
    sp_sizes : float or list
        The amplitude of each spike. If scalar, same amplitude for all.
        If list, must match length of sp_times. Supports current units.
    duration : float or Quantity
        The total duration of the signal.

    Returns
    -------
    current : ndarray or Quantity
        The formatted spike input current.

    Raises
    ------
    ValueError
        If sp_times is not an iterable.
    UnitMismatchError
        If spike sizes have different units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple spike train with uniform properties

    .. code-block:: python

        >>> current = spike(
        ...     sp_times=[10, 20, 30, 200, 300] * u.ms,
        ...     sp_lens=1 * u.ms,  # All spikes 1ms long
        ...     sp_sizes=0.5 * u.nA,  # All spikes 0.5nA amplitude
        ...     duration=400 * u.ms
        ... )
    
    Variable spike properties

    .. code-block:: python

        >>> current = spike(
        ...     sp_times=[10, 50, 100] * u.ms,
        ...     sp_lens=[1, 2, 0.5] * u.ms,  # Different durations
        ...     sp_sizes=[0.5, 1.0, 0.3] * u.nA,  # Different amplitudes
        ...     duration=150 * u.ms
        ... )
    
    High-frequency burst

    .. code-block:: python

        >>> import numpy as np
        >>> times = np.arange(0, 50, 2) * u.ms  # Every 2ms
        >>> current = spike(
        ...     sp_times=times,
        ...     sp_lens=0.5 * u.ms,
        ...     sp_sizes=1.0 * u.pA,
        ...     duration=100 * u.ms
        ... )
    
    Notes
    -----
    - All spike times, lengths, and sizes are converted to appropriate units
    - Overlapping spikes will overwrite each other (last one wins)
    - Spikes extending beyond duration are truncated
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)
    duration_value = u.Quantity(duration).to(time_unit).mantissa

    # Handle various input types including Quantity arrays
    if not hasattr(sp_times, '__iter__'):
        raise ValueError("sp_times must be an iterable")

    # Convert times to mantissa
    sp_times_values = []
    for t in sp_times:
        sp_times_values.append(u.Quantity(t).to(time_unit).mantissa)

    # Handle spike lengths
    if not isinstance(sp_lens, (tuple, list)) and u.math.size(sp_lens) == 1:
        sp_lens = [sp_lens] * len(sp_times)
    sp_lens_values = []
    for length in sp_lens:
        sp_lens_values.append(u.Quantity(length).to(time_unit).mantissa)

    # Handle spike sizes and extract units
    if not isinstance(sp_sizes, (tuple, list)) and u.math.size(sp_sizes) == 1:
        sp_sizes = [sp_sizes] * len(sp_times)

    c_unit = u.get_unit(sp_sizes[0])
    sp_sizes_values = []
    for size in sp_sizes:
        sp_sizes_values.append(u.Quantity(size).to(c_unit).mantissa)

    # Create current array
    n_steps = int(np.ceil(duration_value / dt_value))
    current = np.zeros(n_steps, dtype=brainstate.environ.dftype())

    # Add spikes
    for time, dur, size in zip(sp_times_values, sp_lens_values, sp_sizes_values):
        start_i = int(time / dt_value)
        end_i = int((time + dur) / dt_value)
        if start_i < n_steps:
            end_i = min(end_i, n_steps)
            current[start_i:end_i] = size

    return u.maybe_decimal(current * c_unit)


@set_module_as('braintools.input')
def gaussian_pulse(
    amplitude: brainstate.typing.ArrayLike,
    center: brainstate.typing.ArrayLike,
    sigma: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    n: int = 1
):
    """Generate Gaussian pulse input.

    Creates a Gaussian-shaped current pulse centered at a specific time with
    a specified width. Useful for smooth, physiologically realistic inputs.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the Gaussian pulse. Supports current units.
    center : float or Quantity
        Center time of the Gaussian pulse. Supports time units.
    sigma : float or Quantity
        Standard deviation (width) of the Gaussian pulse. Supports time units.
    duration : float or Quantity
        Total duration of the input.
    n : int, optional
        Number of parallel pulses to generate. Default is 1.
    
    Returns
    -------
    current : ndarray or Quantity
        The Gaussian pulse input. Shape is (n_timesteps,) if n=1,
        or (n_timesteps, n) if n>1.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Single Gaussian pulse

    .. code-block:: python

        >>> current = gaussian_pulse(
        ...     amplitude=10 * u.pA,
        ...     center=50 * u.ms,
        ...     sigma=10 * u.ms,
        ...     duration=100 * u.ms
        ... )
    
    Multiple identical pulses

    .. code-block:: python

        >>> currents = gaussian_pulse(
        ...     amplitude=5 * u.nA,
        ...     center=25 * u.ms,
        ...     sigma=5 * u.ms,
        ...     duration=50 * u.ms,
        ...     n=10  # Generate 10 identical pulses
        ... )
    
    Narrow pulse (approximating delta function)

    .. code-block:: python

        >>> current = gaussian_pulse(
        ...     amplitude=100 * u.pA,
        ...     center=10 * u.ms,
        ...     sigma=0.5 * u.ms,
        ...     duration=20 * u.ms
        ... )
    
    Wide pulse (slow activation)

    .. code-block:: python

        >>> current = gaussian_pulse(
        ...     amplitude=2 * u.nA,
        ...     center=100 * u.ms,
        ...     sigma=30 * u.ms,
        ...     duration=200 * u.ms
        ... )
    
    Notes
    -----
    - The pulse is effectively zero at distances > 3*sigma from center
    - Total charge delivered depends on both amplitude and sigma
    - For n>1, all pulses are identical
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Extract units and values
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)
    center_value = u.Quantity(center).to(time_unit).mantissa
    sigma_value = u.Quantity(sigma).to(time_unit).mantissa
    duration_value = u.Quantity(duration).to(time_unit).mantissa

    # Generate time array
    n_steps = int(np.ceil(duration_value / dt_value))
    times = np.arange(n_steps) * dt_value

    # Generate Gaussian pulse
    gaussian = amplitude_value * np.exp(-0.5 * ((times - center_value) / sigma_value) ** 2)

    if n > 1:
        gaussian = np.tile(gaussian[:, None], (1, n))

    return u.maybe_decimal(gaussian * c_unit)


@set_module_as('braintools.input')
def exponential_decay(
    amplitude: brainstate.typing.ArrayLike,
    tau: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None
):
    """Generate exponentially decaying input.

    Creates an input that decays exponentially from an initial amplitude.
    Useful for modeling synaptic currents or adaptation processes.

    Parameters
    ----------
    amplitude : float or Quantity
        Initial amplitude of the exponential decay. Supports current units.
    tau : float or Quantity
        Time constant of the exponential decay. Supports time units.
    duration : float or Quantity
        Total duration of the input signal.
    t_start : float or Quantity, optional
        Start time of the decay. Before this, current is 0.
        Default is 0.
    t_end : float or Quantity, optional
        End time of the decay. After this, current is 0.
        Default is duration.
    
    Returns
    -------
    current : ndarray or Quantity
        The exponentially decaying input.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple exponential decay

    .. code-block:: python

        >>> current = exponential_decay(
        ...     amplitude=10 * u.pA,
        ...     tau=20 * u.ms,
        ...     duration=100 * u.ms
        ... )
    
    Fast decay (mimicking AMPA receptor)

    .. code-block:: python

        >>> current = exponential_decay(
        ...     amplitude=1 * u.nA,
        ...     tau=2 * u.ms,
        ...     duration=20 * u.ms
        ... )
    
    Slow decay (mimicking NMDA receptor)

    .. code-block:: python

        >>> current = exponential_decay(
        ...     amplitude=0.5 * u.nA,
        ...     tau=100 * u.ms,
        ...     duration=500 * u.ms
        ... )
    
    Delayed decay

    .. code-block:: python

        >>> current = exponential_decay(
        ...     amplitude=5 * u.pA,
        ...     tau=10 * u.ms,
        ...     duration=100 * u.ms,
        ...     t_start=20 * u.ms,  # Start decay at 20ms
        ...     t_end=80 * u.ms      # End at 80ms
        ... )
    
    Notes
    -----
    - The decay follows: I(t) = amplitude * exp(-t/tau)
    - At t=tau, the current is amplitude/e (~37% of initial)
    - At t=3*tau, the current is ~5% of initial amplitude
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa
    duration_value = u.Quantity(duration).to(time_unit).mantissa

    # Extract amplitude and tau
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)
    tau_value = u.Quantity(tau).to(time_unit).mantissa

    # Generate time array for decay
    decay_duration = t_end_value - t_start_value
    n_decay_steps = int(np.ceil(decay_duration / dt_value))
    times = np.arange(n_decay_steps) * dt_value

    # Generate exponential decay
    exp_decay = amplitude_value * np.exp(-times / tau_value)

    # Create full current array
    n_steps = int(np.ceil(duration_value / dt_value))
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())

    # Insert decay at appropriate position
    start_i = int(t_start_value / dt_value)
    end_i = min(start_i + n_decay_steps, n_steps)
    actual_length = end_i - start_i
    currents[start_i:end_i] = exp_decay[:actual_length]

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def double_exponential(
    amplitude: brainstate.typing.ArrayLike,
    tau_rise: brainstate.typing.ArrayLike,
    tau_decay: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None
):
    """Generate double exponential input (alpha function).

    Creates an input with a rapid rise and slower decay, commonly used to
    model synaptic currents. The waveform follows:
    I(t) = A * N * (exp(-t/tau_decay) - exp(-t/tau_rise))
    where N is a normalization factor ensuring peak amplitude equals A.

    Parameters
    ----------
    amplitude : float or Quantity
        Peak amplitude of the double exponential. Supports current units.
    tau_rise : float or Quantity
        Rise time constant. Must be smaller than tau_decay.
        Supports time units.
    tau_decay : float or Quantity
        Decay time constant. Must be larger than tau_rise.
        Supports time units.
    duration : float or Quantity
        Total duration of the input signal.
    t_start : float or Quantity, optional
        Start time of the waveform. Default is 0.
    t_end : float or Quantity, optional
        End time of the waveform. Default is duration.
    
    Returns
    -------
    current : ndarray or Quantity
        The double exponential input.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    AMPA-like synaptic current

    .. code-block:: python

        >>> current = double_exponential(
        ...     amplitude=1 * u.nA,
        ...     tau_rise=0.5 * u.ms,
        ...     tau_decay=5 * u.ms,
        ...     duration=30 * u.ms
        ... )
    
    NMDA-like synaptic current

    .. code-block:: python

        >>> current = double_exponential(
        ...     amplitude=0.5 * u.nA,
        ...     tau_rise=2 * u.ms,
        ...     tau_decay=100 * u.ms,
        ...     duration=500 * u.ms
        ... )
    
    GABA-A like inhibitory current

    .. code-block:: python

        >>> current = double_exponential(
        ...     amplitude=-0.8 * u.nA,  # Negative for inhibition
        ...     tau_rise=0.5 * u.ms,
        ...     tau_decay=10 * u.ms,
        ...     duration=50 * u.ms
        ... )
    
    Delayed synaptic input

    .. code-block:: python

        >>> current = double_exponential(
        ...     amplitude=2 * u.pA,
        ...     tau_rise=1 * u.ms,
        ...     tau_decay=15 * u.ms,
        ...     duration=100 * u.ms,
        ...     t_start=20 * u.ms  # Delay of 20ms
        ... )
    
    Notes
    -----
    - Peak occurs at t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * ln(tau_decay/tau_rise)
    - The function is normalized so the peak value equals the specified amplitude
    - tau_decay must be greater than tau_rise for realistic waveforms
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa
    duration_value = u.Quantity(duration).to(time_unit).mantissa

    # Extract parameters
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)
    tau_rise_value = u.Quantity(tau_rise).to(time_unit).mantissa
    tau_decay_value = u.Quantity(tau_decay).to(time_unit).mantissa

    # Generate time array
    waveform_duration = t_end_value - t_start_value
    n_waveform_steps = int(np.ceil(waveform_duration / dt_value))
    times = np.arange(n_waveform_steps) * dt_value

    # Calculate normalization factor
    if tau_decay_value > tau_rise_value:
        t_peak = (tau_rise_value * tau_decay_value /
                  (tau_decay_value - tau_rise_value) *
                  np.log(tau_decay_value / tau_rise_value))
        norm = 1.0 / (np.exp(-t_peak / tau_decay_value) - np.exp(-t_peak / tau_rise_value))
    else:
        # If tau values are invalid, use simple normalization
        norm = 1.0

    # Generate double exponential
    double_exp = amplitude_value * norm * (
        np.exp(-times / tau_decay_value) -
        np.exp(-times / tau_rise_value)
    )
    double_exp = np.where(times >= 0, double_exp, 0)

    # Create full current array
    n_steps = int(np.ceil(duration_value / dt_value))
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())

    # Insert waveform at appropriate position
    start_i = int(t_start_value / dt_value)
    end_i = min(start_i + n_waveform_steps, n_steps)
    actual_length = end_i - start_i
    currents[start_i:end_i] = double_exp[:actual_length]

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def burst(
    burst_amp: brainstate.typing.ArrayLike,
    burst_freq: brainstate.typing.ArrayLike,
    burst_duration: brainstate.typing.ArrayLike,
    inter_burst_interval: brainstate.typing.ArrayLike,
    n_bursts: int,
    duration: brainstate.typing.ArrayLike,
):
    """Generate burst pattern input.

    Creates a pattern of oscillatory bursts separated by quiet periods.
    Each burst consists of sinusoidal oscillations at a specified frequency.
    Useful for studying burst-induced plasticity or rhythmic stimulation.

    Parameters
    ----------
    burst_amp : float or Quantity
        Amplitude of oscillation during burst. Supports current units.
    burst_freq : float or Quantity
        Frequency of oscillation within each burst. Must be in Hz units.
    burst_duration : float or Quantity
        Duration of each burst. Supports time units.
    inter_burst_interval : float or Quantity
        Time between end of one burst and start of next. Supports time units.
    n_bursts : int
        Number of bursts to generate.
    duration : float or Quantity
        Total duration of the input signal.
    
    Returns
    -------
    current : ndarray or Quantity
        The burst pattern input.

    Raises
    ------
    AssertionError
        If burst_freq is not in Hz units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Theta burst stimulation

    .. code-block:: python

        >>> current = burst(
        ...     burst_amp=10 * u.pA,
        ...     burst_freq=100 * u.Hz,  # 100Hz within burst
        ...     burst_duration=50 * u.ms,  # 50ms bursts
        ...     inter_burst_interval=150 * u.ms,  # 150ms between bursts
        ...     n_bursts=5,
        ...     duration=1000 * u.ms
        ... )
    
    Gamma burst pattern

    .. code-block:: python

        >>> current = burst(
        ...     burst_amp=5 * u.nA,
        ...     burst_freq=40 * u.Hz,  # Gamma frequency
        ...     burst_duration=100 * u.ms,
        ...     inter_burst_interval=100 * u.ms,
        ...     n_bursts=10,
        ...     duration=2000 * u.ms
        ... )
    
    High-frequency stimulation protocol

    .. code-block:: python

        >>> current = burst(
        ...     burst_amp=20 * u.pA,
        ...     burst_freq=200 * u.Hz,
        ...     burst_duration=20 * u.ms,
        ...     inter_burst_interval=80 * u.ms,
        ...     n_bursts=20,
        ...     duration=2000 * u.ms
        ... )
    
    Slow oscillatory bursts

    .. code-block:: python

        >>> current = burst(
        ...     burst_amp=1 * u.nA,
        ...     burst_freq=5 * u.Hz,  # Slow oscillation
        ...     burst_duration=500 * u.ms,
        ...     inter_burst_interval=500 * u.ms,
        ...     n_bursts=3,
        ...     duration=3000 * u.ms
        ... )
    
    Notes
    -----
    - Total period per burst = burst_duration + inter_burst_interval
    - Bursts that would extend beyond duration are truncated
    - The oscillation within each burst starts at phase 0
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Extract units and values
    burst_amp_value, c_unit = u.split_mantissa_unit(burst_amp)
    burst_duration_value = u.Quantity(burst_duration).to(time_unit).mantissa
    inter_burst_interval_value = u.Quantity(inter_burst_interval).to(time_unit).mantissa
    duration_value = u.Quantity(duration).to(time_unit).mantissa

    # Handle frequency (must be in Hz)
    freq_unit = u.get_unit(burst_freq)
    assert freq_unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {freq_unit}.'
    burst_freq_value = u.Quantity(burst_freq).to(u.Hz).mantissa

    # Create current array
    n_steps = int(np.ceil(duration_value / dt_value))
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())

    # Generate each burst
    for i in range(n_bursts):
        burst_start_value = i * (burst_duration_value + inter_burst_interval_value)
        if burst_start_value >= duration_value:
            break

        # Generate sinusoidal burst
        n_burst_steps = int(np.ceil(burst_duration_value / dt_value))
        times = np.arange(n_burst_steps) * dt_value

        # Convert frequency from Hz to angular frequency considering time unit
        # If time is in ms, need to convert: cycles/s * ms = cycles/s * s/1000 = cycles/1000
        if time_unit == u.ms:
            freq_factor = burst_freq_value / 1000.0  # Hz to kHz for ms time scale
        elif time_unit == u.second:
            freq_factor = burst_freq_value
        else:
            # General conversion
            freq_factor = burst_freq_value * u.Quantity(1 * time_unit).to(u.second).mantissa

        burst = burst_amp_value * np.sin(2 * np.pi * freq_factor * times)

        # Insert burst into current array
        start_i = int(burst_start_value / dt_value)
        end_i = min(start_i + n_burst_steps, n_steps)
        actual_length = end_i - start_i
        currents[start_i:end_i] = burst[:actual_length]

    return u.maybe_decimal(currents * c_unit)


spike_input = create_deprecated_function(spike, 'spike_input', 'spike')
burst_input = create_deprecated_function(burst, 'burst_input', 'burst')

__all__.extend(['spike', 'burst'])
