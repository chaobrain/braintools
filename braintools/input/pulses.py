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
Pulse and burst input generators.
"""

import brainstate
import brainunit as u
import numpy as np

__all__ = [
    'spike_input',
    'gaussian_pulse',
    'exponential_decay',
    'double_exponential',
    'burst_input',
]


def spike_input(
    sp_times,
    sp_lens,
    sp_sizes,
    duration,
    dt=None
):
    """Format current input like a series of short-time spikes.

    For example:

    If you want to generate a spike train at 10 ms, 20 ms, 30 ms, 200 ms, 300 ms,
    and each spike lasts 1 ms and the spike current is 0.5, then you can use the
    following funtions:

    >>> spike_input(sp_times=[10, 20, 30, 200, 300],
    >>>             sp_lens=1.,  # can be a list to specify the spike length at each point
    >>>             sp_sizes=0.5,  # can be a list to specify the current size at each point
    >>>             duration=400.)

    Parameters
    ----------
    sp_times : list, tuple
        The spike time-points. Must be an iterable object.
    sp_lens : int, float, list, tuple
        The length of each point-current, mimicking the spike durations.
    sp_sizes : int, float, list, tuple
        The current sizes.
    duration : int, float
        The total current duration.
    dt : float
        The default is None.

    Returns
    -------
    current : bm.ndarray
        The formatted input current.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    assert isinstance(sp_times, (list, tuple))
    if not isinstance(sp_lens, (tuple, list)):
        sp_lens = [sp_lens] * len(sp_times)
    if not isinstance(sp_sizes, (tuple, list)):
        sp_sizes = [sp_sizes] * len(sp_times)
    for size in sp_sizes[1:]:
        u.fail_for_unit_mismatch(sp_sizes[0], size)

    current = u.math.zeros(int(np.ceil(u.maybe_decimal(duration / dt))),
                           dtype=brainstate.environ.dftype(),
                           unit=u.get_unit(sp_sizes[0]))
    for time, dur, size in zip(sp_times, sp_lens, sp_sizes):
        pp = int(u.maybe_decimal(time / dt))
        p_len = int(u.maybe_decimal(dur / dt))
        current = current.at[pp: pp + p_len].set(size)
    return u.maybe_decimal(current)


def gaussian_pulse(
    amplitude,
    center,
    sigma,
    duration,
    dt=None,
    n=1
):
    """Gaussian pulse input.

    Parameters
    ----------
    amplitude: float
        Peak amplitude of the Gaussian pulse.
    center: float
        Center time of the Gaussian pulse.
    sigma: float
        Standard deviation (width) of the Gaussian pulse.
    duration: float
        Total duration of the input.
    dt: float
        The numerical precision.
    n: int
        Number of parallel pulses to generate.
    
    Returns
    -------
    current : array
        The formatted Gaussian pulse input.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    times = u.math.arange(0. * u.ms, duration, dt)
    
    # Generate Gaussian pulse
    gaussian = amplitude * u.math.exp(-0.5 * ((times - center) / sigma) ** 2)
    
    if n > 1:
        gaussian = u.math.tile(gaussian[:, None], (1, n))
    
    return u.maybe_decimal(gaussian)


def exponential_decay(
    amplitude,
    tau,
    duration,
    dt=None,
    t_start=None,
    t_end=None
):
    """Exponentially decaying input.

    Parameters
    ----------
    amplitude: float
        Initial amplitude of the exponential decay.
    tau: float
        Time constant of the exponential decay.
    duration: float
        Total duration of the input.
    dt: float
        The numerical precision.
    t_start: float
        The start time of the decay.
    t_end: float
        The end time of the decay.
    
    Returns
    -------
    current : array
        The formatted exponentially decaying input.
    """
    if t_start is None:
        t_start = 0. * u.ms
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    
    # Generate exponential decay
    exp_decay = amplitude * u.math.exp(-times / tau)
    
    currents = u.math.zeros(int(u.maybe_decimal(duration / dt)),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(exp_decay))
    start_i = int(u.maybe_decimal(t_start / dt))
    end_i = int(u.maybe_decimal(t_end / dt))
    currents = currents.at[start_i:end_i].set(exp_decay)
    return u.maybe_decimal(currents)


def double_exponential(
    amplitude,
    tau_rise,
    tau_decay,
    duration,
    dt=None,
    t_start=None,
    t_end=None
):
    """Double exponential input (alpha function).

    Creates an input with shape: A * (exp(-t/tau_decay) - exp(-t/tau_rise))

    Parameters
    ----------
    amplitude: float
        Peak amplitude of the double exponential.
    tau_rise: float
        Rise time constant.
    tau_decay: float
        Decay time constant.
    duration: float
        Total duration of the input.
    dt: float
        The numerical precision.
    t_start: float
        The start time.
    t_end: float
        The end time.
    
    Returns
    -------
    current : array
        The formatted double exponential input.
    """
    if t_start is None:
        t_start = 0. * u.ms
    dt = brainstate.environ.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    
    # Normalization factor to ensure peak amplitude equals specified amplitude
    t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * u.math.log(tau_decay / tau_rise)
    norm = 1.0 / (u.math.exp(-t_peak / tau_decay) - u.math.exp(-t_peak / tau_rise))
    
    # Generate double exponential
    double_exp = amplitude * norm * (u.math.exp(-times / tau_decay) - u.math.exp(-times / tau_rise))
    double_exp = u.math.where(times >= 0 * u.ms, double_exp, 0)
    
    currents = u.math.zeros(int(u.maybe_decimal(duration / dt)),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(double_exp))
    start_i = int(u.maybe_decimal(t_start / dt))
    end_i = int(u.maybe_decimal(t_end / dt))
    currents = currents.at[start_i:end_i].set(double_exp)
    return u.maybe_decimal(currents)


def burst_input(
    burst_amp,
    burst_freq,
    burst_duration,
    inter_burst_interval,
    n_bursts,
    duration,
    dt=None
):
    """Generate burst pattern input.

    Parameters
    ----------
    burst_amp: float
        Amplitude during burst.
    burst_freq: Quantity
        Frequency of oscillation within each burst, in Hz.
    burst_duration: float
        Duration of each burst.
    inter_burst_interval: float
        Time between bursts.
    n_bursts: int
        Number of bursts.
    duration: float
        Total duration of the input.
    dt: float
        The numerical precision.
    
    Returns
    -------
    current : array
        The burst pattern input.
    """
    assert burst_freq.unit.dim == u.Hz.dim, f'Frequency must be in Hz. Got {burst_freq.unit}.'
    dt = brainstate.environ.get_dt() if dt is None else dt
    
    currents = u.math.zeros(int(u.maybe_decimal(duration / dt)),
                            dtype=brainstate.environ.dftype())
    
    for i in range(n_bursts):
        burst_start = i * (burst_duration + inter_burst_interval)
        if burst_start >= duration:
            break
        
        # Generate sinusoidal burst
        times = u.math.arange(0. * u.ms, burst_duration, dt)
        burst = burst_amp * u.math.sin(2 * u.math.pi * u.maybe_decimal(times * burst_freq))
        
        start_i = int(u.maybe_decimal(burst_start / dt))
        end_i = min(start_i + len(burst), len(currents))
        actual_end_i = min(start_i + len(burst), end_i)
        currents = currents.at[start_i:actual_end_i].set(burst[:actual_end_i - start_i])
    
    return u.maybe_decimal(currents)