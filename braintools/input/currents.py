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


from typing import Sequence

import brainstate
import brainunit as u
import numpy as np

__all__ = [
    'section_input',
    'constant_input',
    'spike_input',
    'ramp_input',
    'wiener_process',
    'ou_process',
    'sinusoidal_input',
    'square_input',
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


def ramp_input(
    c_start,
    c_end,
    duration,
    t_start=0,
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
        The ramped current start time-point.
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
    t_end = duration if t_end is None else t_end
    current = u.math.zeros(int(np.ceil(u.maybe_decimal(duration / dt))),
                           dtype=brainstate.environ.dftype(),
                           unit=u.get_unit(c_start))
    p1 = int(np.ceil(u.maybe_decimal(t_start / dt)))
    p2 = int(np.ceil(u.maybe_decimal(t_end / dt)))
    cc = u.math.linspace(c_start, c_end, p2 - p1)
    current = current.at[p1: p2].set(cc)
    return u.maybe_decimal(current)


def wiener_process(
    duration,
    dt=None,
    n=1,
    t_start=0.,
    t_end=None,
    seed=None
):
    """Stimulus sampled from a Wiener process, i.e.
    drawn from standard normal distribution N(0, sqrt(dt)).

    Parameters
    ----------
    duration: float
      The input duration.
    dt: float
      The numerical precision.
    n: int
      The variable number.
    t_start: float
      The start time.
    t_end: float
      The end time.
    seed: int
      The noise seed.
    """
    if seed is None:
        rng = brainstate.random.DEFAULT
    else:
        rng = brainstate.random.RandomState(seed)

    dt = brainstate.environ.get_dt() if dt is None else dt
    t_end = duration if t_end is None else t_end
    i_start = int(u.maybe_decimal(t_start / dt))
    i_end = int(u.maybe_decimal(t_end / dt))
    noises = rng.standard_normal((i_end - i_start, n)) * u.math.sqrt(dt)
    currents = u.math.zeros((int(u.maybe_decimal(duration / dt)), n),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(noises))
    currents = currents.at[i_start: i_end].set(noises)
    return u.maybe_decimal(currents)


def ou_process(
    mean,
    sigma,
    tau,
    duration,
    dt=None,
    n=1,
    t_start=0.,
    t_end=None,
    seed=None
):
    r"""Ornstein–Uhlenbeck input.

    .. math::

       dX = (mu - X)/\tau * dt + \sigma*dW

    Parameters
    ----------
    mean: float
      Drift of the OU process.
    sigma: float
      Standard deviation of the Wiener process, i.e. strength of the noise.
    tau: float
      Timescale of the OU process, in ms.
    duration: float
      The input duration.
    dt: float
      The numerical precision.
    n: int
      The variable number.
    t_start: float
      The start time.
    t_end: float
      The end time.
    seed: optional, int
      The random seed.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    dt_sqrt = u.math.sqrt(dt)
    t_end = duration if t_end is None else t_end
    i_start = int(u.maybe_decimal(t_start / dt))
    i_end = int(u.maybe_decimal(t_end / dt))
    rng = brainstate.random.RandomState(seed) if seed is not None else brainstate.random.DEFAULT

    def _f(x, _):
        x = x + dt * ((mean - x) / tau) + sigma * dt_sqrt * rng.rand(n)
        return x, x

    _, noises = brainstate.compile.scan(_f,
                                        u.math.full(n, mean, dtype=brainstate.environ.dftype()),
                                        u.math.arange(i_end - i_start))
    currents = u.math.zeros((int(u.maybe_decimal(duration / dt)), n),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(noises))
    currents = currents.at[i_start: i_end].set(noises)
    return u.maybe_decimal(currents)


def sinusoidal_input(
    amplitude,
    frequency,
    duration,
    dt=None,
    t_start=0.,
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
      The start time.
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
    if t_end is None:
        t_end = duration
    times = u.math.arange(0. * u.ms, t_end - t_start, dt)
    start_i = int(u.maybe_decimal(t_start / dt))
    end_i = int(u.maybe_decimal(t_end / dt))
    sin_inputs = amplitude * u.math.sin(2 * u.math.pi * u.maybe_decimal(times * frequency))
    if bias:
        sin_inputs += amplitude
    currents = u.math.zeros(int(u.maybe_decimal(duration / dt)),
                            dtype=brainstate.environ.dftype(),
                            unit=u.get_unit(sin_inputs))
    currents = currents.at[start_i:end_i].set(sin_inputs)
    return u.maybe_decimal(currents)


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
    sin_inputs = amplitude * _square(2 * np.pi * u.maybe_decimal(times * frequency))
    if bias:
        sin_inputs += amplitude
    currents = u.math.zeros(int(u.maybe_decimal(duration / dt)), dtype=brainstate.environ.dftype())
    start_i = int(u.maybe_decimal(t_start / dt))
    end_i = int(u.maybe_decimal(t_end / dt))
    currents = currents.at[start_i:end_i].set(sin_inputs)
    return u.maybe_decimal(currents)
