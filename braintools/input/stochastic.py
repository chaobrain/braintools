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
Stochastic and random process input generators.
"""

import brainstate
import brainunit as u
import numpy as np

__all__ = [
    'wiener_process',
    'ou_process',
    'poisson_input',
]


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


def poisson_input(
    rate,
    duration,
    dt=None,
    amplitude=1.0,
    n=1,
    seed=None
):
    """Generate Poisson spike train input.

    Parameters
    ----------
    rate: float or Quantity
        Mean firing rate in Hz.
    duration: float
        Total duration of the input.
    dt: float
        The numerical precision.
    amplitude: float
        Amplitude of each spike.
    n: int
        Number of independent Poisson processes.
    seed: int
        Random seed.
    
    Returns
    -------
    current : array
        The Poisson spike train input.
    """
    if seed is None:
        rng = brainstate.random.DEFAULT
    else:
        rng = brainstate.random.RandomState(seed)
    
    dt = brainstate.environ.get_dt() if dt is None else dt
    n_steps = int(u.maybe_decimal(duration / dt))
    
    # Convert rate to probability per timestep
    if hasattr(rate, 'unit'):
        assert rate.unit.dim == u.Hz.dim, f'Rate must be in Hz. Got {rate.unit}.'
        prob = u.maybe_decimal(rate * dt)
    else:
        prob = rate * u.maybe_decimal(dt)
    
    # Generate Poisson spikes
    spikes = rng.random((n_steps, n)) < prob
    currents = spikes.astype(brainstate.environ.dftype()) * amplitude
    
    return u.maybe_decimal(currents)