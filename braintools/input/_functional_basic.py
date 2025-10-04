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
Basic input current generators.
"""

from __future__ import annotations

import functools
from typing import Sequence, Optional

import brainstate
import brainunit as u
import numpy as np

from braintools._misc import set_module_as
from ._deprecation import create_deprecated_function

__all__ = [
    'section',
    'constant',
    'step',
    'ramp',
]


@set_module_as('braintools.input')
def section(
    values: Sequence,
    durations: Sequence,
    return_length: bool = False
):
    """Format an input current with different sections.

    Create a piecewise constant current input where each section has a 
    specified value and duration. This is useful for creating experimental
    protocols with different stimulation phases.

    Parameters
    ----------
    values : list or array-like
        The current values for each period. Can be scalars or arrays.
        If arrays, they must be broadcastable to a common shape.
        Supports units (e.g., pA, nA).
    durations : list or array-like  
        The duration for each period. Must have same length as values.
        Supports time units (e.g., ms, s).
    return_length : bool, optional
        If True, returns a tuple (current, total_duration).
        If False, returns only the current array.
        Default is False.

    Returns
    -------
    current : ndarray or Quantity
        The formatted current array with shape (n_timesteps,) or 
        (n_timesteps, ...) for array values.
    total_duration : Quantity, optional
        Total duration (only if return_length=True).

    Raises
    ------
    ValueError
        If lengths of values and durations don't match.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple step protocol

    .. code-block:: python

        >>> current = section(
        ...     values=[0, 10, 0] * u.pA,
        ...     durations=[100, 200, 100] * u.ms
        ... )
    
    Multiple channel input

    .. code-block:: python

        >>> import numpy as np
        >>> values = [np.zeros(3), np.ones(3) * 5, np.zeros(3)] * u.nA
        >>> current = section(
        ...     values=values,
        ...     durations=[50, 100, 50] * u.ms
        ... )
    
    Get both current and duration

    .. code-block:: python

        >>> current, duration = section(
        ...     values=[0, 1, 2, 1, 0] * u.pA,
        ...     durations=[20, 20, 40, 20, 20] * u.ms,
        ...     return_length=True
        ... )
        >>> print(f"Total duration: {duration}")
    
    Complex protocol with different phases

    .. code-block:: python

        >>> protocol_values = [0, 2, 5, 10, 5, 2, 0] * u.pA
        >>> protocol_durations = [50, 30, 30, 100, 30, 30, 50] * u.ms
        >>> current = section(protocol_values, protocol_durations)
    
    Notes
    -----
    The function handles unit conversions automatically. All values are
    converted to the same unit as the first value, and all durations
    are converted to the dt time unit.
    """
    if len(durations) != len(values):
        raise ValueError(f'"values" and "durations" must be the same length, while '
                         f'we got {len(values)} != {len(durations)}.')
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Extract units from values and durations
    value_unit = u.get_unit(values[0])
    value_mantissas = []
    for val in values:
        value_mantissas.append(u.Quantity(val).to(value_unit).mantissa)

    # Get shape
    i_shape = ()
    for val in value_mantissas:
        i_shape = u.math.broadcast_shapes(i_shape, np.shape(val))

    # Convert durations to mantissa values
    duration_mantissas = []
    all_duration_value = 0
    for duration in durations:
        dur_mantissa = u.Quantity(duration).to(time_unit).mantissa
        duration_mantissas.append(dur_mantissa)
        all_duration_value += dur_mantissa

    # format the current
    currents = []
    for c_size, duration in zip(value_mantissas, duration_mantissas):
        length = int(np.ceil(duration / dt_value))
        current = np.ones((length,) + i_shape, dtype=brainstate.environ.dftype())
        currents.append(current * c_size)
    currents = np.concatenate(currents, axis=0)

    # Apply unit if present
    currents = u.maybe_decimal(currents * value_unit)

    # returns
    if return_length:
        return currents, u.maybe_decimal(all_duration_value * time_unit)
    else:
        return currents


@set_module_as('braintools.input')
def constant(I_and_duration):
    """Format constant input currents with specified durations.

    Creates a sequence of constant current pulses, where each pulse has a
    specified amplitude and duration. This function is similar to section
    but uses a different input format.

    Parameters
    ----------
    I_and_duration : list of tuples
        List of (current_value, duration) pairs. Each tuple specifies:
        - current_value: The amplitude (scalar or array, with or without units)
        - duration: The time duration (with or without time units)

    Returns
    -------
    current : ndarray or Quantity
        The concatenated current array.
    total_duration : Quantity
        The total duration of all segments.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> import numpy as np
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple two-phase protocol

    .. code-block:: python

        >>> current, duration = constant([
        ...     (0 * u.pA, 100 * u.ms),
        ...     (10 * u.pA, 200 * u.ms)
        ... ])
    
    Mixed scalar and array values

    .. code-block:: python

        >>> current, duration = constant([
        ...     (0, 50 * u.ms),
        ...     (np.array([1, 2, 3]) * u.nA, 100 * u.ms),
        ...     (0, 50 * u.ms)
        ... ])
    
    Complex multi-phase stimulation

    .. code-block:: python

        >>> phases = [
        ...     (0 * u.pA, 20 * u.ms),      # baseline
        ...     (5 * u.pA, 50 * u.ms),      # weak stimulus
        ...     (10 * u.pA, 100 * u.ms),    # strong stimulus
        ...     (2 * u.pA, 30 * u.ms),      # recovery
        ...     (0 * u.pA, 50 * u.ms),      # rest
        ... ]
        >>> current, total_time = constant(phases)
        >>> print(f"Total stimulation time: {total_time}")
    
    Using arrays for spatial patterns

    .. code-block:: python

        >>> spatial_pattern = np.array([[1, 0], [0, 1]]) * u.nA
        >>> current, duration = constant([
        ...     (np.zeros((2, 2)) * u.nA, 100 * u.ms),
        ...     (spatial_pattern, 200 * u.ms),
        ...     (np.zeros((2, 2)) * u.nA, 100 * u.ms)
        ... ])
    
    Ramp-like approximation with many steps

    .. code-block:: python

        >>> steps = [(i * u.pA, 10 * u.ms) for i in range(11)]
        >>> current, duration = constant(steps)
    
    Notes
    -----
    - All current values are converted to the same unit as the first value
    - All durations are converted to the dt time unit
    - The function automatically broadcasts arrays to compatible shapes
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # get input current dimension, shape, and duration
    I_shape = ()
    c_unit = None
    I_mantissas = []
    duration_mantissas = []

    for I_val, duration in I_and_duration:
        # Extract mantissa and unit from current
        I_unit = u.get_unit(I_val)

        # Check unit consistency
        if c_unit is None:
            c_unit = I_unit

        I_mantissa = u.Quantity(I_val).to(c_unit).mantissa
        I_mantissas.append(I_mantissa)

        # Handle duration
        dur_mantissa = u.Quantity(duration).to(time_unit).mantissa
        duration_mantissas.append(dur_mantissa)

        # Update shape
        I_shape = u.math.broadcast_shapes(I_shape, np.shape(I_mantissa))

    # get the current
    currents = []
    for c_size, duration in zip(I_mantissas, duration_mantissas):
        length = int(np.ceil(duration / dt_value))
        current = np.ones((length,) + I_shape, dtype=brainstate.environ.dftype()) * c_size
        currents.append(current)
    currents = np.concatenate(currents, axis=0)

    # Apply unit if present
    currents = u.maybe_decimal(currents * c_unit)
    I_duration = u.maybe_decimal(np.sum(duration_mantissas) * time_unit)
    return currents, I_duration


@set_module_as('braintools.input')
def step(
    amplitudes,
    step_times,
    duration: brainstate.typing.ArrayLike = None,
):
    """Generate step function input with multiple levels.

    Creates a step function where the amplitude changes at specified time points.
    The amplitude remains constant between consecutive step times. This is useful
    for creating protocols with discrete amplitude changes.

    Parameters
    ----------
    amplitudes : list or array-like
        Amplitude values for each step. The i-th amplitude starts at step_times[i]
        and continues until step_times[i+1] (or the end). Supports units.
    step_times : list or array-like
        Time points where steps occur. Will be automatically sorted.
        Supports time units.
    duration : float or Quantity
        Total duration of the input signal.

    Returns
    -------
    current : ndarray or Quantity
        The step function input array with shape (n_timesteps,).

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple three-level step function

    .. code-block:: python

        >>> current = step(
        ...     amplitudes=[0, 10, 5] * u.pA,
        ...     step_times=[0, 50, 150] * u.ms,
        ...     duration=200 * u.ms
        ... )
    
    Staircase protocol

    .. code-block:: python

        >>> amplitudes = [0, 2, 4, 6, 8, 10] * u.nA
        >>> times = [0, 20, 40, 60, 80, 100] * u.ms
        >>> current = step(amplitudes, times, 120 * u.ms)
    
    Multiple pulses with return to baseline

    .. code-block:: python

        >>> current = step(
        ...     amplitudes=[0, 5, 0, 10, 0] * u.pA,
        ...     step_times=[0, 20, 40, 60, 80] * u.ms,
        ...     duration=100 * u.ms
        ... )
    
    Unsorted times are automatically sorted

    .. code-block:: python

        >>> current = step(
        ...     amplitudes=[5, 0, 10] * u.pA,
        ...     step_times=[50, 0, 100] * u.ms,  # Will be sorted to [0, 50, 100]
        ...     duration=150 * u.ms
        ... )
    
    Protocol with negative values

    .. code-block:: python

        >>> current = step(
        ...     amplitudes=[-5, 0, 5, 0, -5] * u.pA,
        ...     step_times=[0, 25, 50, 75, 100] * u.ms,
        ...     duration=125 * u.ms
        ... )
    
    F-I curve protocol

    .. code-block:: python

        >>> import numpy as np
        >>> amplitudes = np.linspace(0, 50, 11) * u.pA
        >>> times = np.linspace(0, 1000, 11) * u.ms
        >>> current = step(amplitudes, times, 1100 * u.ms)
    
    Notes
    -----
    - Step times and their corresponding amplitudes are automatically sorted by time
    - The last amplitude continues until the end of the duration
    - All amplitudes are converted to the same unit as the first amplitude
    - Times before 0 or after duration are ignored
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)
    duration = duration if duration is not None else functools.reduce(lambda a, b: a + b, step_times)
    duration_value = u.Quantity(duration).to(time_unit).mantissa

    # Extract mantissa and units from amplitudes
    amp_mantissas = []
    c_unit = None
    for amp in amplitudes:
        if c_unit is None: c_unit = u.get_unit(amp)
        amp_mantissas.append(u.Quantity(amp).to(c_unit).mantissa)

    n_steps = int(duration_value / dt_value)
    currents = np.zeros(n_steps, dtype=brainstate.environ.dftype())

    # Convert step times to mantissa values for sorting
    step_times_values = [
        u.Quantity(t).to(time_unit).mantissa for t in step_times
    ]

    # Sort step times and amplitudes together
    sorted_indices = np.argsort(step_times_values)
    sorted_times = [step_times_values[i] for i in sorted_indices]
    sorted_amps = [amp_mantissas[i] for i in sorted_indices]

    # Set amplitude for each interval
    for i, (time, amp) in enumerate(zip(sorted_times, sorted_amps)):
        start_i = int(time / dt_value)
        if i < len(sorted_times) - 1:
            end_i = int(sorted_times[i + 1] / dt_value)
        else:
            end_i = n_steps
        if start_i < n_steps:
            currents[start_i:min(end_i, n_steps)] = amp

    # Apply unit if present
    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def ramp(
    c_start: brainstate.typing.ArrayLike,
    c_end: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
):
    """Generate a linearly ramped input current.

    Creates a current that changes linearly from a starting value to an ending
    value over a specified time window. The ramp can be increasing or decreasing,
    and can be confined to a portion of the total duration.

    Parameters
    ----------
    c_start : float or Quantity
        The starting current amplitude. Supports current units.
    c_end : float or Quantity  
        The ending current amplitude. Must have same units as c_start.
    duration : float or Quantity
        The total duration of the signal.
    t_start : float or Quantity, optional
        Time point when the ramp begins. Before this, current is 0.
        Default is 0.
    t_end : float or Quantity, optional
        Time point when the ramp ends. After this, current remains at c_end.
        Default is duration.

    Returns
    -------
    current : ndarray or Quantity
        The ramped current array with shape (n_timesteps,).

    Raises
    ------
    UnitMismatchError
        If c_start and c_end have different units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Simple linear ramp from 0 to 10 pA over 100 ms

    .. code-block:: python

        >>> current = ramp(
        ...     c_start=0 * u.pA,
        ...     c_end=10 * u.pA,
        ...     duration=100 * u.ms
        ... )

    Decreasing ramp (10 to 0 pA)

    .. code-block:: python

        >>> current = ramp(
        ...     c_start=10 * u.pA,
        ...     c_end=0 * u.pA,
        ...     duration=100 * u.ms
        ... )

    Ramp with delay and early stop

    .. code-block:: python

        >>> current = ramp(
        ...     c_start=0 * u.nA,
        ...     c_end=5 * u.nA,
        ...     duration=200 * u.ms,
        ...     t_start=50 * u.ms,   # Start ramping at 50 ms
        ...     t_end=150 * u.ms      # Stop ramping at 150 ms
        ... )

    Negative to positive ramp

    .. code-block:: python

        >>> current = ramp(
        ...     c_start=-5 * u.pA,
        ...     c_end=5 * u.pA,
        ...     duration=100 * u.ms
        ... )

    Slow ramp for adaptation studies

    .. code-block:: python

        >>> current = ramp(
        ...     c_start=0 * u.pA,
        ...     c_end=20 * u.pA,
        ...     duration=1000 * u.ms,
        ...     t_start=100 * u.ms,
        ...     t_end=900 * u.ms
        ... )

    Ramp for I-V curve measurements

    .. code-block:: python

        >>> current = ramp(
        ...     c_start=-100 * u.pA,
        ...     c_end=100 * u.pA,
        ...     duration=500 * u.ms
        ... )

    Sawtooth wave component

    .. code-block:: python

        >>> current = ramp(
        ...     c_start=0 * u.pA,
        ...     c_end=10 * u.pA,
        ...     duration=10 * u.ms,
        ...     t_start=1 * u.ms,
        ...     t_end=9 * u.ms
        ... )


    Notes
    -----
    - The ramp is perfectly linear between t_start and t_end
    - Before t_start, the current is 0 (not c_start)
    - After t_end, the current remains at the value it reached
    - Unit consistency is enforced between c_start and c_end

    """
    dt = brainstate.environ.get_dt()
    dt, time_unit = u.split_mantissa_unit(dt)
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end
    duration = u.Quantity(duration).to(time_unit).mantissa
    t_start = u.Quantity(t_start).to(time_unit).mantissa
    t_end = u.Quantity(t_end).to(time_unit).mantissa

    c_start, c_unit = u.split_mantissa_unit(c_start)
    c_end = u.Quantity(c_end).to(c_unit).mantissa

    length = int(np.ceil(duration / dt))
    p1 = int(np.ceil(t_start / dt))
    p2 = int(np.ceil(t_end / dt))

    current = np.zeros(length, dtype=brainstate.environ.dftype())
    cc = np.linspace(c_start, c_end, p2 - p1)
    current[p1: p2] = cc
    return u.maybe_decimal(current * c_unit)


section_input = create_deprecated_function(section, 'section_input', 'section')
constant_input = create_deprecated_function(constant, 'constant_input', 'constant')
step_input = create_deprecated_function(step, 'step_input', 'step')
ramp_input = create_deprecated_function(ramp, 'ramp_input', 'ramp')

__all__.extend(['section', 'constant', 'step', 'ramp'])
