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
Composable stochastic input generators.

This module provides composable stochastic input generators that can be combined
with other input components using arithmetic operations.
"""

from typing import Optional

import brainstate
import brainunit as u

from . import _functional_stochastic as functional
from ._composable_base import Input
from ._deprecation import create_deprecated_class

ArrayLike = brainstate.typing.ArrayLike

__all__ = [
    'WienerProcess',
    'OUProcess',
    'Poisson',
]


class WienerProcess(Input):
    r"""Generate Wiener process (Brownian motion) input.

    A Wiener process (also known as Brownian motion) is a continuous-time
    stochastic process with independent Gaussian increments. It's widely used
    to model synaptic noise and random fluctuations in neural systems.

    The process follows:

    .. math::
        dW(t) \sim \mathcal{N}(0, \sigma^2 dt)

    where increments are independent and normally distributed.

    Parameters
    ----------
    duration : float or Quantity
        Total duration of the input signal. Supports time units.
    n : int, optional
        Number of independent Wiener processes to generate.
        Default is 1.
    t_start : float or Quantity, optional
        Start time of the process. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        End time of the process. After this, output is 0.
        Default is duration.
    sigma : float, optional
        Standard deviation of the noise. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility.
        Default is None (uses global random state).

    Attributes
    ----------
    n : int
        Number of independent processes
    t_start : float or Quantity
        Start time of the process
    t_end : float or Quantity
        End time of the process
    sigma : float
        Standard deviation of the noise
    seed : int or None
        Random seed for reproducibility

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Create simple Wiener process:

    .. code-block:: python

        >>> noise = WienerProcess(duration=100 * u.ms, sigma=0.5)
        >>> signal = noise()

    Create multiple independent processes:

    .. code-block:: python

        >>> multi_noise = WienerProcess(
        ...     duration=200 * u.ms,
        ...     n=5,  # 5 independent processes
        ...     sigma=1.0
        ... )

    Create windowed noise (active only between t_start and t_end):

    .. code-block:: python

        >>> windowed = WienerProcess(
        ...     duration=500 * u.ms,
        ...     sigma=2.0,
        ...     t_start=100 * u.ms,
        ...     t_end=400 * u.ms
        ... )

    Combine with other inputs using arithmetic operations:

    .. code-block:: python

        >>> from braintools.input import Ramp, Step
        >>> # Noisy background with linear drift
        >>> drift = Ramp(0, 0.5, 500 * u.ms)
        >>> noise = WienerProcess(500 * u.ms, sigma=0.1)
        >>> drifting_noise = noise + drift


        >>> # Modulated noise
        >>> envelope = Step([0, 1.0], [0 * u.ms, 100 * u.ms], 500 * u.ms)
        >>> modulated = noise * envelope

    Create reproducible noise with seed:

    .. code-block:: python

        >>> fixed_noise = WienerProcess(
        ...     duration=100 * u.ms,
        ...     sigma=0.3,
        ...     seed=42  # Fixed seed for reproducibility
        ... )

    Notes
    -----
    - The variance of increments scales with dt: Var(dW) = σ²dt
    - The process has zero mean: E[W(t)] = 0
    - Increments are independent and normally distributed
    - The process is non-differentiable but continuous
    - Useful for modeling synaptic background noise
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    OUProcess : Ornstein-Uhlenbeck process with mean reversion
    Poisson : Poisson spike train generator
    wiener_process : Functional API for Wiener process
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 duration: ArrayLike,
                 n: int = 1,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 sigma: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(duration)
        self.n = n
        self.t_start = t_start
        self.t_end = t_end
        self.sigma = sigma
        self.seed = seed

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the Wiener process using the functional API.

        Returns
        -------
        ArrayLike
            Generated Wiener process array with shape (n_steps,) if n=1,
            or (n_steps, n) if n>1.
        """
        # Use the functional API to generate the Wiener process
        return functional.wiener_process(
            duration=self.duration,
            sigma=self.sigma,
            n=self.n,
            t_start=self.t_start,
            t_end=self.t_end,
            seed=self.seed
        )


class OUProcess(Input):
    r"""Generate Ornstein-Uhlenbeck process input.

    The Ornstein-Uhlenbeck (OU) process is a stochastic process that models
    a particle undergoing Brownian motion with friction. It exhibits
    mean-reverting behavior, making it useful for modeling fluctuations
    around a steady state in neural systems.

    The process follows the stochastic differential equation:

    .. math::
        dX_t = \frac{\mu - X_t}{\tau} dt + \sigma dW_t

    where:
    - μ is the mean (drift target)
    - τ is the time constant
    - σ is the noise amplitude
    - W_t is a Wiener process

    Parameters
    ----------
    mean : float
        Mean value (drift target) of the OU process.
    sigma : float
        Standard deviation of the noise.
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

    Attributes
    ----------
    mean : float
        Mean value (drift target)
    sigma : float
        Noise amplitude
    tau : float or Quantity
        Time constant
    n : int
        Number of independent processes
    t_start : float or Quantity
        Start time of the process
    t_end : float or Quantity
        End time of the process
    seed : int or None
        Random seed

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Simple OU process:

    .. code-block:: python

        >>> ou = OUProcess(
        ...     mean=0.5,
        ...     sigma=0.2,
        ...     tau=10 * u.ms,
        ...     duration=500 * u.ms
        ... )
        >>> signal = ou()

    Fast fluctuations around zero:

    .. code-block:: python

        >>> fast_ou = OUProcess(
        ...     mean=0.0,
        ...     sigma=0.5,
        ...     tau=2 * u.ms,  # Fast time constant
        ...     duration=200 * u.ms
        ... )

    Slow fluctuations with drift:

    .. code-block:: python

        >>> slow_ou = OUProcess(
        ...     mean=1.0,
        ...     sigma=0.3,
        ...     tau=50 * u.ms,  # Slow time constant
        ...     duration=1000 * u.ms
        ... )

    Multiple independent processes:

    .. code-block:: python

        >>> multi_ou = OUProcess(
        ...     mean=0.0,
        ...     sigma=0.2,
        ...     tau=5 * u.ms,
        ...     duration=300 * u.ms,
        ...     n=10  # 10 independent processes
        ... )

    Windowed OU process:

    .. code-block:: python

        >>> windowed_ou = OUProcess(
        ...     mean=0.5,
        ...     sigma=0.1,
        ...     tau=20 * u.ms,
        ...     duration=500 * u.ms,
        ...     t_start=100 * u.ms,
        ...     t_end=400 * u.ms
        ... )

    Combine with other inputs:

    .. code-block:: python

        >>> from braintools.input import Sinusoidal, Step
        >>> # OU process with time-varying mean
        >>> ou = OUProcess(mean=0.5, sigma=0.1, tau=20 * u.ms, duration=500 * u.ms)
        >>> sine_mean = Sinusoidal(0.3, 2 * u.Hz, 500 * u.ms)
        >>> modulated_ou = ou + sine_mean


        >>> # Gated OU process
        >>> gate = Step([0, 1.0], [0 * u.ms, 50 * u.ms], 500 * u.ms)
        >>> gated_ou = ou * gate

    Create reproducible OU process:

    .. code-block:: python

        >>> fixed_ou = OUProcess(
        ...     mean=0.0,
        ...     sigma=0.15,
        ...     tau=15 * u.ms,
        ...     duration=200 * u.ms,
        ...     seed=123  # Fixed seed
        ... )

    Notes
    -----
    - The process has mean-reverting behavior controlled by tau
    - Steady-state variance: σ²/(2/τ)
    - Autocorrelation decays exponentially with time constant tau
    - For tau → ∞, approaches a Wiener process
    - For tau → 0, approaches white noise
    - Useful for modeling synaptic background activity
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    WienerProcess : Wiener process without mean reversion
    Poisson : Poisson spike train generator
    ou_process : Functional API for OU process
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 mean: float,
                 sigma: float,
                 tau: ArrayLike,
                 duration: ArrayLike,
                 n: int = 1,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 seed: Optional[int] = None):
        super().__init__(duration)
        self.mean = mean
        self.sigma = sigma
        self.tau = tau
        self.n = n
        self.t_start = t_start
        self.t_end = t_end
        self.seed = seed

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the OU process using the functional API.

        Returns
        -------
        ArrayLike
            Generated OU process array with shape (n_steps,) if n=1,
            or (n_steps, n) if n>1.
        """
        # Use the functional API to generate the OU process
        return functional.ou_process(
            mean=self.mean,
            sigma=self.sigma,
            tau=self.tau,
            duration=self.duration,
            n=self.n,
            t_start=self.t_start,
            t_end=self.t_end,
            seed=self.seed
        )


class Poisson(Input):
    r"""Generate Poisson spike train input.

    Creates spike trains where spikes occur randomly according to a Poisson
    process with a specified rate. This is useful for modeling random
    synaptic inputs or background activity in neural systems.

    The probability of a spike in each time bin is:

    .. math::
        P(\text{spike}) = \lambda \cdot dt

    where λ is the rate and dt is the time step.

    Parameters
    ----------
    rate : Quantity
        Mean firing rate. Must be in Hz units.
    duration : float or Quantity
        Total duration of the input signal. Supports time units.
    n : int, optional
        Number of independent Poisson processes to generate.
        Default is 1.
    t_start : float or Quantity, optional
        Start time of spiking. Before this, output is 0.
        Default is 0.
    t_end : float or Quantity, optional
        End time of spiking. After this, output is 0.
        Default is duration.
    amplitude : float, optional
        Amplitude of each spike. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility.
        Default is None (uses global random state).

    Attributes
    ----------
    rate : Quantity
        Firing rate in Hz
    n : int
        Number of independent spike trains
    t_start : float or Quantity
        Start time of spiking
    t_end : float or Quantity
        End time of spiking
    amplitude : float
        Spike amplitude
    seed : int or None
        Random seed

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> brainstate.environ.set(dt=0.1 * u.ms)

    Simple Poisson spike train:

    .. code-block:: python

        >>> spikes = Poisson(
        ...     rate=10 * u.Hz,
        ...     duration=1000 * u.ms
        ... )
        >>> signal = spikes()

    High-frequency background activity:

    .. code-block:: python

        >>> background = Poisson(
        ...     rate=100 * u.Hz,
        ...     duration=500 * u.ms,
        ...     amplitude=0.5  # Smaller amplitude
        ... )

    Multiple independent spike trains:

    .. code-block:: python

        >>> multi_spikes = Poisson(
        ...     rate=20 * u.Hz,
        ...     duration=2000 * u.ms,
        ...     n=50,  # 50 independent spike trains
        ...     amplitude=2.0
        ... )

    Windowed spiking activity:

    .. code-block:: python

        >>> burst = Poisson(
        ...     rate=50 * u.Hz,
        ...     duration=1000 * u.ms,
        ...     t_start=200 * u.ms,
        ...     t_end=800 * u.ms,
        ...     amplitude=1.0
        ... )

    Low rate spontaneous activity:

    .. code-block:: python

        >>> spontaneous = Poisson(
        ...     rate=1 * u.Hz,
        ...     duration=10000 * u.ms,
        ...     amplitude=5.0
        ... )

    Combine with envelopes for rate modulation:

    .. code-block:: python

        >>> from braintools.input import GaussianPulse, Sinusoidal
        >>> # Poisson spikes with Gaussian envelope
        >>> poisson = Poisson(50 * u.Hz, 1000 * u.ms)
        >>> envelope = GaussianPulse(1.0, 500 * u.ms, 100 * u.ms, 1000 * u.ms)
        >>> modulated = poisson * envelope


        >>> # Rhythmic modulation of spike rate
        >>> rhythm = Sinusoidal(0.5, 5 * u.Hz, 1000 * u.ms)
        >>> rhythmic_spikes = poisson * (1 + rhythm)

    Create reproducible spike pattern:

    .. code-block:: python

        >>> fixed_spikes = Poisson(
        ...     rate=30 * u.Hz,
        ...     duration=500 * u.ms,
        ...     seed=456  # Fixed seed for reproducibility
        ... )

    Inhomogeneous Poisson process (time-varying rate):

    .. code-block:: python

        >>> from braintools.input import Ramp
        >>> # Base Poisson process
        >>> base_poisson = Poisson(10 * u.Hz, 1000 * u.ms)
        >>> # Increasing rate envelope
        >>> ramp = Ramp(0.1, 1.0, 1000 * u.ms)
        >>> increasing_rate = base_poisson * ramp

    Notes
    -----
    - Spike probability per timestep = rate * dt
    - Mean number of spikes = rate * duration
    - Inter-spike intervals follow exponential distribution
    - For small dt, probability of multiple spikes per bin is negligible
    - Can be combined with continuous inputs for rate modulation
    - Useful for modeling synaptic background noise
    - Can be combined with other Input classes using +, -, *, /

    See Also
    --------
    WienerProcess : Continuous noise process
    OUProcess : Ornstein-Uhlenbeck process
    poisson : Functional API for Poisson input
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 rate: u.Quantity,
                 duration: ArrayLike,
                 n: int = 1,
                 t_start: Optional[ArrayLike] = None,
                 t_end: Optional[ArrayLike] = None,
                 amplitude: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(duration)
        assert rate.unit.dim == u.Hz.dim, f'Rate must be in Hz. Got {rate.unit}.'
        self.rate = rate
        self.n = n
        self.t_start = t_start
        self.t_end = t_end
        self.amplitude = amplitude
        self.seed = seed

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the Poisson input using the functional API.

        Returns
        -------
        ArrayLike
            Generated Poisson spike train with shape (n_steps,) if n=1,
            or (n_steps, n) if n>1.
        """
        # Use the functional API to generate the Poisson input
        return functional.poisson(
            rate=self.rate,
            duration=self.duration,
            amplitude=self.amplitude,
            n=self.n,
            t_start=self.t_start,
            t_end=self.t_end,
            seed=self.seed
        )


PoissonInput = create_deprecated_class(Poisson, 'PoissonInput', 'Poisson')

__all__.append('PoissonInput')
