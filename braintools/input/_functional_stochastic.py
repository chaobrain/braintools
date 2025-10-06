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
Stochastic and random process input generators.
"""

from typing import Optional

import brainstate
import brainunit as u
import numpy as np

from braintools._misc import set_module_as
from ._deprecation import create_deprecated_function

__all__ = [
    'wiener_process',
    'ou_process',
    'poisson',
]


@set_module_as('braintools.input')
def wiener_process(
    duration: brainstate.typing.ArrayLike,
    sigma: brainstate.typing.ArrayLike = 1.0,
    n: int = 1,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    seed: Optional[int] = None
):
    """Generate Wiener process (Brownian motion) input.

    Creates a stochastic input following a Wiener process, where increments
    are drawn from a normal distribution N(0, σ²dt). Useful for modeling
    synaptic noise or random fluctuations in neural systems.

    Parameters
    ----------
    duration : float or Quantity
        Total duration of the input signal. Supports time units.
    sigma : float or Quantity, optional
        Standard deviation of the noise. Supports current units.
        Default is 1.0.
    n : int, optional
        Number of independent Wiener processes to generate.
        Default is 1.
    t_start : float or Quantity, optional
        Start time of the process. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        End time of the process. After this, output is 0.
        Default is duration.
    seed : int, optional
        Random seed for reproducibility.
        Default is None (uses global random state).

    Returns
    -------
    current : ndarray or Quantity
        The Wiener process input. Shape is (n_timesteps,) if n=1,
        or (n_timesteps, n) if n>1.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple Wiener process

    .. code-block:: python

        >>> noise = wiener_process(
        ...     duration=100 * u.ms,
        ...     sigma=0.5 * u.pA
        ... )
    
    Multiple independent processes

    .. code-block:: python

        >>> noises = wiener_process(
        ...     duration=200 * u.ms,
        ...     sigma=1.0 * u.nA,
        ...     n=10  # 10 independent processes
        ... )
    
    Windowed noise

    .. code-block:: python

        >>> noise = wiener_process(
        ...     duration=500 * u.ms,
        ...     sigma=2.0 * u.pA,
        ...     t_start=100 * u.ms,
        ...     t_end=400 * u.ms
        ... )
    
    Reproducible noise

    .. code-block:: python

        >>> noise = wiener_process(
        ...     duration=100 * u.ms,
        ...     sigma=0.3 * u.nA,
        ...     seed=42  # Fixed seed for reproducibility
        ... )
    
    Notes
    -----
    - The variance scales with dt: Var(dW) = σ²dt
    - The process has zero mean: E[W(t)] = 0
    - Increments are independent and normally distributed
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Extract sigma
    sigma_value, c_unit = u.split_mantissa_unit(sigma)

    # Setup random number generator
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Calculate indices
    i_start = int(t_start_value / dt_value)
    i_end = int(t_end_value / dt_value)
    n_steps = int(np.ceil(duration_value / dt_value))

    # Generate noise
    dt_sqrt = np.sqrt(dt_value)
    shape = (i_end - i_start,) if n == 1 else (i_end - i_start, n)
    noise_values = rng.standard_normal(shape) * sigma_value * dt_sqrt

    # Create full array with zeros outside the window
    full_shape = (n_steps,) if n == 1 else (n_steps, n)
    currents = np.zeros(full_shape, dtype=brainstate.environ.dftype())
    currents[i_start:i_end] = noise_values

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def ou_process(
    mean: brainstate.typing.ArrayLike,
    sigma: brainstate.typing.ArrayLike,
    tau: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    n: int = 1,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    seed: Optional[int] = None
):
    r"""Generate Ornstein-Uhlenbeck process input.

    Creates a stochastic input following the Ornstein-Uhlenbeck process,
    which models a particle undergoing Brownian motion with friction.
    The process tends to revert to a mean value over time.

    The dynamics follow:
    
    .. math::
        dX = \frac{\mu - X}{\tau} dt + \sigma dW

    Parameters
    ----------
    mean : float or Quantity
        Mean value (drift) of the OU process. Supports current units.
    sigma : float or Quantity
        Standard deviation of the noise. Supports current units.
    tau : float or Quantity
        Time constant of the process. Larger values = slower fluctuations.
        Supports time units.
    duration : float or Quantity
        Total duration of the input signal. Supports time units.
    n : int, optional
        Number of independent OU processes to generate.
        Default is 1.
    t_start : float or Quantity, optional
        Start time of the process. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        End time of the process. After this, output is 0.
        Default is duration.
    seed : int, optional
        Random seed for reproducibility.
        Default is None (uses global random state).

    Returns
    -------
    current : ndarray or Quantity
        The OU process input. Shape is (n_timesteps,) if n=1,
        or (n_timesteps, n) if n>1.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple OU process

    .. code-block:: python

        >>> current = ou_process(
        ...     mean=0.5 * u.nA,
        ...     sigma=0.2 * u.nA,
        ...     tau=10 * u.ms,
        ...     duration=500 * u.ms
        ... )
    
    Fast fluctuations around zero

    .. code-block:: python

        >>> current = ou_process(
        ...     mean=0 * u.pA,
        ...     sigma=5 * u.pA,
        ...     tau=2 * u.ms,  # Fast time constant
        ...     duration=200 * u.ms
        ... )
    
    Slow fluctuations with drift

    .. code-block:: python

        >>> current = ou_process(
        ...     mean=1.0 * u.nA,
        ...     sigma=0.3 * u.nA,
        ...     tau=50 * u.ms,  # Slow time constant
        ...     duration=1000 * u.ms
        ... )
    
    Multiple independent processes

    .. code-block:: python

        >>> currents = ou_process(
        ...     mean=0 * u.pA,
        ...     sigma=2 * u.pA,
        ...     tau=5 * u.ms,
        ...     duration=300 * u.ms,
        ...     n=10  # 10 independent processes
        ... )
    
    Windowed OU process

    .. code-block:: python

        >>> current = ou_process(
        ...     mean=0.5 * u.nA,
        ...     sigma=0.1 * u.nA,
        ...     tau=20 * u.ms,
        ...     duration=500 * u.ms,
        ...     t_start=100 * u.ms,
        ...     t_end=400 * u.ms
        ... )
    
    Notes
    -----
    - The process has mean-reverting behavior controlled by tau
    - Variance at steady state: σ²/(2/τ)
    - Autocorrelation decays exponentially with time constant tau
    - Useful for modeling synaptic background activity
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Extract parameters
    mean_value, c_unit = u.split_mantissa_unit(mean)
    sigma_value = u.Quantity(sigma).to(c_unit).mantissa
    tau_value = u.Quantity(tau).to(time_unit).mantissa

    # Setup random number generator
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Calculate indices
    i_start = int(t_start_value / dt_value)
    i_end = int(t_end_value / dt_value)
    n_steps = int(np.ceil(duration_value / dt_value))

    # Generate OU process
    dt_sqrt = np.sqrt(dt_value)
    dt_over_tau = dt_value / tau_value

    # Initialize process at mean
    x = np.full(n, mean_value, dtype=brainstate.environ.dftype())
    process_values = []

    for _ in range(i_end - i_start):
        # OU dynamics: dx = (mean - x) * dt/tau + sigma * sqrt(dt) * dW
        noise = rng.standard_normal(n if n > 1 else ())
        x = x + (mean_value - x) * dt_over_tau + sigma_value * dt_sqrt * noise
        process_values.append(x.copy() if n > 1 else x)

    # Stack the process values
    if n > 1:
        noise_values = np.stack(process_values, axis=0)
    else:
        noise_values = np.squeeze(np.array(process_values))

    # Create full array with zeros outside the window
    full_shape = (n_steps,) if n == 1 else (n_steps, n)
    currents = np.zeros(full_shape, dtype=brainstate.environ.dftype())
    currents[i_start:i_end] = noise_values

    return u.maybe_decimal(currents * c_unit)


@set_module_as('braintools.input')
def poisson(
    rate: brainstate.typing.ArrayLike,
    duration: brainstate.typing.ArrayLike,
    amplitude: brainstate.typing.ArrayLike = 1.0,
    n: int = 1,
    t_start: Optional[brainstate.typing.ArrayLike] = None,
    t_end: Optional[brainstate.typing.ArrayLike] = None,
    seed: Optional[int] = None
):
    """Generate Poisson spike train input.

    Creates spike trains where spikes occur randomly according to a Poisson
    process with a specified rate. Useful for modeling random synaptic inputs
    or background activity.

    Parameters
    ----------
    rate : float or Quantity
        Mean firing rate. Must be in Hz units.
    duration : float or Quantity
        Total duration of the input signal. Supports time units.
    amplitude : float or Quantity, optional
        Amplitude of each spike. Supports current units.
        Default is 1.0.
    n : int, optional
        Number of independent Poisson processes to generate.
        Default is 1.
    t_start : float or Quantity, optional
        Start time of spiking. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        End time of spiking. After this, output is 0.
        Default is duration.
    seed : int, optional
        Random seed for reproducibility.
        Default is None (uses global random state).

    Returns
    -------
    current : ndarray or Quantity
        The Poisson spike train input. Shape is (n_timesteps,) if n=1,
        or (n_timesteps, n) if n>1.

    Raises
    ------
    AssertionError
        If rate is not in Hz units.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    
    Simple Poisson spike train

    .. code-block:: python

        >>> spikes = poisson(
        ...     rate=10 * u.Hz,
        ...     duration=1000 * u.ms,
        ...     amplitude=1 * u.pA
        ... )
    
    High-frequency background activity

    .. code-block:: python

        >>> spikes = poisson(
        ...     rate=100 * u.Hz,
        ...     duration=500 * u.ms,
        ...     amplitude=0.5 * u.nA
        ... )
    
    Multiple independent spike trains

    .. code-block:: python

        >>> spikes = poisson(
        ...     rate=20 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     amplitude=2 * u.pA,
        ...     n=50  # 50 independent spike trains
        ... )
    
    Windowed spiking activity

    .. code-block:: python

        >>> spikes = poisson(
        ...     rate=50 * u.Hz,
        ...     duration=1000 * u.ms,
        ...     amplitude=1 * u.nA,
        ...     t_start=200 * u.ms,
        ...     t_end=800 * u.ms
        ... )
    
    Low rate spontaneous activity

    .. code-block:: python

        >>> spikes = poisson(
        ...     rate=1 * u.Hz,
        ...     duration=10000 * u.ms,
        ...     amplitude=5 * u.pA,
        ...     seed=123  # Reproducible spike pattern
        ... )
    
    Notes
    -----
    - Spike probability per timestep = rate * dt
    - Mean number of spikes = rate * duration
    - Inter-spike intervals follow exponential distribution
    - Useful for modeling synaptic background noise
    """
    dt = brainstate.environ.get_dt()
    dt_value, time_unit = u.split_mantissa_unit(dt)

    # Handle time parameters
    t_start = 0. * time_unit if t_start is None else t_start
    t_end = duration if t_end is None else t_end

    duration_value = u.Quantity(duration).to(time_unit).mantissa
    t_start_value = u.Quantity(t_start).to(time_unit).mantissa
    t_end_value = u.Quantity(t_end).to(time_unit).mantissa

    # Handle rate (must be in Hz)
    rate_unit = u.get_unit(rate)
    assert rate_unit.dim == u.Hz.dim, f'Rate must be in Hz. Got {rate_unit}.'
    rate_value = u.Quantity(rate).to(u.Hz).mantissa

    # Extract amplitude
    amplitude_value, c_unit = u.split_mantissa_unit(amplitude)

    # Setup random number generator
    rng = np.random if seed is None else np.random.RandomState(seed)

    # Calculate indices
    i_start = int(t_start_value / dt_value)
    i_end = int(t_end_value / dt_value)
    n_steps = int(np.ceil(duration_value / dt_value))

    # Convert rate to probability per timestep
    # Need to account for time unit conversion
    if time_unit == u.ms:
        spike_prob = rate_value * dt_value / 1000.0  # Hz = 1/s, dt in ms
    elif time_unit == u.second:
        spike_prob = rate_value * dt_value
    else:
        # General conversion
        spike_prob = rate_value * dt_value * u.Quantity(1 * time_unit).to(u.second).mantissa

    # Generate Poisson spikes in the active window
    shape = (i_end - i_start,) if n == 1 else (i_end - i_start, n)
    spikes = rng.random(shape) < spike_prob
    spike_values = spikes.astype(brainstate.environ.dftype()) * amplitude_value

    # Create full array with zeros outside the window
    full_shape = (n_steps,) if n == 1 else (n_steps, n)
    currents = np.zeros(full_shape, dtype=brainstate.environ.dftype())
    currents[i_start:i_end] = spike_values

    return u.maybe_decimal(currents * c_unit)


poisson_input = create_deprecated_function(poisson, 'poisson_input', 'poisson')

__all__.append('poisson')
