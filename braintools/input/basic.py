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
Basic input current generators.
"""

from typing import Sequence

import brainstate
import brainunit as u
import numpy as np

__all__ = [
    'section_input',
    'constant_input',
    'step_input',
    'ramp_input',
]


def section_input(
    values: Sequence,
    durations: Sequence,
    dt: brainstate.typing.ArrayLike = None,
    return_length: bool = False
):
    """Format an input current with different sections.

    For example:

    If you want to get an input where the size is 0 bwteen 0-100 ms,
    and the size is 1. between 100-200 ms.

    >>> section_input(values=[0, 1],
    >>>               durations=[100, 100])

    Parameters
    ----------
    values : list, np.ndarray
        The current values for each period duration.
    durations : list, np.ndarray
        The duration for each period.
    dt : float
        Default is None.
    return_length : bool
        Return the final duration length.

    Returns
    -------
    current_and_duration: tuple
        (The formatted current, total duration)
    """
    if len(durations) != len(values):
        raise ValueError(f'"values" and "durations" must be the same length, while '
                         f'we got {len(values)} != {len(durations)}.')
    dt = brainstate.environ.get_dt() if dt is None else dt

    # get input currents
    values = [u.math.array(val) for val in values]
    i_shape = ()
    for val in values:
        shape = u.math.shape(val)
        if len(shape) > len(i_shape):
            i_shape = shape

    # format the current
    all_duration = None
    currents = []
    for c_size, duration in zip(values, durations):
        current = u.math.ones(
            (int(np.ceil(u.maybe_decimal(duration / dt))),) + i_shape,
            dtype=brainstate.environ.dftype()
        )
        current = current * c_size
        currents.append(current)
        if all_duration is None:
            all_duration = duration
        else:
            all_duration += duration
    currents = u.math.concatenate(currents, axis=0)

    # returns
    if return_length:
        return currents, all_duration
    else:
        return currents


def constant_input(
    I_and_duration,
    dt=None
):
    """Format constant input in durations.

    For example:

    If you want to get an input where the size is 0 bwteen 0-100 ms,
    and the size is 1. between 100-200 ms.

    >>> import brainpy.math as bm
    >>> constant_input([(0, 100), (1, 100)])
    >>> constant_input([(bm.zeros(100), 100), (bm.random.rand(100), 100)])

    Parameters
    ----------
    I_and_duration : list
        This parameter receives the current size and the current
        duration pairs, like `[(Isize1, duration1), (Isize2, duration2)]`.
    dt : float
        Default is None.

    Returns
    -------
    current_and_duration : tuple
        (The formatted current, total duration)
    """
    dt = brainstate.environ.get_dt() if dt is None else dt

    # get input current dimension, shape, and duration
    I_duration = None
    I_shape = ()
    for I in I_and_duration:
        I_duration = I[1] if I_duration is None else I_duration + I[1]
        shape = u.math.shape(I[0])
        if len(shape) > len(I_shape):
            I_shape = shape

    # get the current
    currents = []
    for c_size, duration in I_and_duration:
        length = int(np.ceil(u.maybe_decimal(duration / dt)))
        current = u.math.ones((length,) + I_shape, dtype=brainstate.environ.dftype()) * c_size
        currents.append(current)
    return u.math.concatenate(currents, axis=0), I_duration


def step_input(
    amplitudes,
    step_times,
    duration,
    dt=None
):
    """Generate step function input with multiple levels.

    Parameters
    ----------
    amplitudes: list or array
        Amplitude values for each step.
    step_times: list or array
        Time points where steps occur.
    duration: float
        Total duration of the input.
    dt: float
        The numerical precision.
    
    Returns
    -------
    current : array
        The step function input.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    n_steps = int(u.maybe_decimal(duration / dt))
    
    currents = u.math.zeros(n_steps, dtype=brainstate.environ.dftype())
    
    # Convert step times to dimensionless values for sorting
    step_times_ms = [u.maybe_decimal(t / u.ms) if hasattr(t, 'unit') else t for t in step_times]
    
    # Sort step times and amplitudes together
    sorted_indices = np.argsort(step_times_ms)
    sorted_times = [step_times[i] for i in sorted_indices]
    sorted_amps = [amplitudes[i] for i in sorted_indices]
    
    # Set amplitude for each interval
    for i, (time, amp) in enumerate(zip(sorted_times, sorted_amps)):
        start_i = int(u.maybe_decimal(time / dt))
        if i < len(sorted_times) - 1:
            end_i = int(u.maybe_decimal(sorted_times[i + 1] / dt))
        else:
            end_i = n_steps
        
        if start_i < n_steps:
            currents = currents.at[start_i:min(end_i, n_steps)].set(amp)
    
    return u.maybe_decimal(currents)


def ramp_input(
    c_start,
    c_end,
    duration,
    t_start=None,
    t_end=None,
    dt=None
):
    """Get the gradually changed input current.

    Parameters
    ----------
    c_start : float
        The minimum (or maximum) current size.
    c_end : float
        The maximum (or minimum) current size.
    duration : int, float
        The total duration.
    t_start : float
        The ramped current start time-point. Default is 0.
    t_end : float
        The ramped current end time-point. Default is the None.
    dt : float, int, optional
        The numerical precision.

    Returns
    -------
    current : bm.ndarray
      The formatted current
    """
    u.fail_for_unit_mismatch(c_start, c_end)
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_start = 0. * u.get_unit(dt) if t_start is None else t_start
    t_end = duration if t_end is None else t_end
    current = u.math.zeros(int(np.ceil(u.maybe_decimal(duration / dt))),
                           dtype=brainstate.environ.dftype(),
                           unit=u.get_unit(c_start))
    p1 = int(np.ceil(u.maybe_decimal(t_start / dt)))
    p2 = int(np.ceil(u.maybe_decimal(t_end / dt)))
    cc = u.math.linspace(c_start, c_end, p2 - p1)
    current = current.at[p1: p2].set(cc)
    return u.maybe_decimal(current)