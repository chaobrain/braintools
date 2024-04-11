# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
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


import braincore as bc
import jax.lax
import jax.numpy as jnp
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


def section_input(values, durations, dt=None, return_length=False):
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
  current_and_duration
  """
  if len(durations) != len(values):
    raise ValueError(f'"values" and "durations" must be the same length, while '
                     f'we got {len(values)} != {len(durations)}.')

  dt = bc.environ.get_dt() if dt is None else dt

  # get input current shape, and duration
  I_duration = sum(durations)
  I_shape = ()
  for val in values:
    shape = jnp.shape(val)
    if len(shape) > len(I_shape):
      I_shape = shape

  # get the current
  start = 0
  I_current = jnp.zeros((int(np.ceil(I_duration / dt)),) + I_shape, dtype=bc.environ.dftype())
  for c_size, duration in zip(values, durations):
    length = int(duration / dt)
    I_current = I_current.at[start: start + length].set(c_size)
    start += length

  if return_length:
    return I_current, I_duration
  else:
    return I_current


def constant_input(I_and_duration, dt=None):
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
  dt = bc.environ.get_dt() if dt is None else dt

  # get input current dimension, shape, and duration
  I_duration = 0.
  I_shape = ()
  for I in I_and_duration:
    I_duration += I[1]
    shape = jnp.shape(I[0])
    if len(shape) > len(I_shape):
      I_shape = shape

  # get the current
  start = 0
  I_current = jnp.zeros((int(np.ceil(I_duration / dt)),) + I_shape, dtype=bc.environ.dftype())
  for c_size, duration in I_and_duration:
    length = int(duration / dt)
    I_current = I_current.at[start: start + length].set(c_size)
    start += length
  return I_current, I_duration



def spike_input(sp_times, sp_lens, sp_sizes, duration, dt=None):
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
  dt = bc.environ.get_dt() if dt is None else dt
  assert isinstance(sp_times, (list, tuple))
  if isinstance(sp_lens, (float, int)):
    sp_lens = [sp_lens] * len(sp_times)
  if isinstance(sp_sizes, (float, int)):
    sp_sizes = [sp_sizes] * len(sp_times)

  current = jnp.zeros(int(np.ceil(duration / dt)), dtype=bc.environ.dftype())
  for time, dur, size in zip(sp_times, sp_lens, sp_sizes):
    pp = int(time / dt)
    p_len = int(dur / dt)
    current = current.at[pp: pp + p_len].set(size)
  return current


def ramp_input(c_start, c_end, duration, t_start=0, t_end=None, dt=None):
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
  dt = bc.environ.get_dt() if dt is None else dt
  t_end = duration if t_end is None else t_end

  current = jnp.zeros(int(np.ceil(duration / dt)), dtype=bc.environ.dftype())
  p1 = int(np.ceil(t_start / dt))
  p2 = int(np.ceil(t_end / dt))
  cc = jnp.array(jnp.linspace(c_start, c_end, p2 - p1))
  current = current.at[p1: p2].set(cc)
  return current



def wiener_process(duration, dt=None, n=1, t_start=0., t_end=None, seed=None):
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
  dt = bc.environ.get_dt() if dt is None else dt
  t_end = duration if t_end is None else t_end
  i_start = int(t_start / dt)
  i_end = int(t_end / dt)
  noises = bc.random.standard_normal((i_end - i_start, n)) * jnp.sqrt(dt)
  currents = jnp.zeros((int(duration / dt), n), dtype=bc.environ.dftype())
  currents = currents.at[i_start: i_end].set(noises)
  return currents


def ou_process(mean, sigma, tau, duration, dt=None, n=1, t_start=0., t_end=None, seed=None):
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
  dt = bc.environ.get_dt() if dt is None else dt
  dt_sqrt = jnp.sqrt(dt)
  t_end = duration if t_end is None else t_end
  i_start = int(t_start / dt)
  i_end = int(t_end / dt)

  def _f(x, key):
    x = x + dt * ((mean - x) / tau) + sigma * dt_sqrt * bc.random.rand(n, key=key)
    return x, x

  _, noises = jax.lax.scan(_f, jnp.full(n, mean, dtype=bc.environ.dftype()), bc.random.split_keys(i_end - i_start))
  currents = jnp.zeros((int(duration / dt), n), dtype=bc.environ.dftype())
  return currents.at[i_start: i_end].set(noises)


def sinusoidal_input(amplitude, frequency, duration, dt=None, t_start=0., t_end=None, bias=False):
  """Sinusoidal input.

  Parameters
  ----------
  amplitude: float
    Amplitude of the sinusoid.
  frequency: float
    Frequency of the sinus oscillation, in Hz
  duration: float
    The input duration.
  t_start: float
    The start time.
  t_end: float
    The end time.
  dt: float
    The numerical precision.
  bias: bool
    Whether the sinusoid oscillates around 0 (False), or
    has a positive DC bias, thus non-negative (True).
  """
  dt = bc.environ.get_dt() if dt is None else dt
  if t_end is None:
    t_end = duration
  times = jnp.arange(0, t_end - t_start, dt)
  start_i = int(t_start / dt)
  end_i = int(t_end / dt)
  sin_inputs = amplitude * jnp.sin(2 * jnp.pi * times * (frequency / 1000.0))
  if bias: sin_inputs += amplitude
  currents = jnp.zeros(int(duration / dt), dtype=bc.environ.dftype())
  return currents.at[start_i:end_i].set(sin_inputs)


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


def square_input(amplitude, frequency, duration, dt=None, bias=False, t_start=0., t_end=None):
  """Oscillatory square input.

  Parameters
  ----------
  amplitude: float
    Amplitude of the square oscillation.
  frequency: float
    Frequency of the square oscillation, in Hz.
  duration: float
    The input duration.
  t_start: float
    The start time.
  t_end: float
    The end time.
  dt: float
    The numerical precision.
  bias: bool
    Whether the sinusoid oscillates around 0 (False), or
    has a positive DC bias, thus non-negative (True).
  """
  dt = bc.environ.get_dt() if dt is None else dt
  if t_end is None: t_end = duration
  times = np.arange(0, t_end - t_start, dt)
  sin_inputs = amplitude * _square(2 * np.pi * times * (frequency / 1000.0))
  if bias: sin_inputs += amplitude
  currents = jnp.zeros(int(duration / dt), dtype=bc.environ.dftype())
  start_i = int(t_start / dt)
  end_i = int(t_end / dt)
  return currents.at[start_i:end_i].set(sin_inputs)
