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
Composable stochastic input generators.
"""

from typing import Optional, Union
import brainstate
import brainunit as u
import numpy as np
from ._composable_base import Input

__all__ = [
    'WienerProcess',
    'OUProcess',
    'PoissonInput',
]


class WienerProcess(Input):
    """Generate Wiener process (Brownian motion) input.
    
    Examples
    --------
    >>> # Create noisy background with drift
    >>> noise = WienerProcess(500, sigma=0.1, n=2)
    >>> drift = RampInput(0, 0.5, 500)
    >>> drifting_noise = noise + drift
    """
    
    def __init__(self,
                 duration: Union[float, u.Quantity],
                 n: int = 1,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 sigma: float = 1.0,
                 seed: Optional[int] = None):
        """Initialize Wiener process.
        
        Parameters
        ----------
        duration : float or Quantity
            The total duration.
        n : int
            Number of independent processes.
        t_start : float or Quantity, optional
            The start time.
        t_end : float or Quantity, optional
            The end time.
        sigma : float
            Standard deviation of the noise.
        seed : int, optional
            Random seed.
        """
        super().__init__(duration)
        
        self.n = n
        self.t_start = t_start
        self.t_end = t_end
        self.sigma = sigma
        self.seed = seed
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the Wiener process array."""
        if self.seed is None:
            rng = brainstate.random.DEFAULT
        else:
            rng = brainstate.random.RandomState(self.seed)
        
        # Get dt unit
        dt_unit = u.get_unit(self.dt)
        dt_value = u.get_magnitude(self.dt)
        
        t_start = 0. * dt_unit if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        # Convert to same unit as dt
        t_start_value = u.Quantity(t_start).to(dt_unit).mantissa if hasattr(t_start, 'unit') else t_start
        t_end_value = u.Quantity(t_end).to(dt_unit).mantissa if hasattr(t_end, 'unit') else t_end
        
        currents = u.math.zeros((self.n_steps, self.n), dtype=brainstate.environ.dftype())
        
        start_i = int(t_start_value / dt_value)
        end_i = int(t_end_value / dt_value)
        
        # Generate Wiener increments
        dt_sqrt = u.math.sqrt(u.get_magnitude(self.dt))
        wiener = rng.standard_normal((end_i - start_i, self.n)) * self.sigma * dt_sqrt
        
        # Cumulative sum to get Wiener process
        wiener_cumsum = np.cumsum(wiener, axis=0)
        currents = currents.at[start_i:end_i].set(wiener_cumsum)
        
        return currents


class OUProcess(Input):
    """Generate Ornstein-Uhlenbeck process input.
    
    Examples
    --------
    >>> # Create OU process with time-varying mean
    >>> ou = OUProcess(mean=0.5, sigma=0.1, tau=20, duration=500, n=2)
    >>> sine_mean = SinusoidalInput(0.3, 2 * u.Hz, 500)
    >>> modulated_ou = ou + sine_mean
    """
    
    def __init__(self,
                 mean: float,
                 sigma: float,
                 tau: Union[float, u.Quantity],
                 duration: Union[float, u.Quantity],
                 n: int = 1,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 seed: Optional[int] = None):
        """Initialize OU process.
        
        Parameters
        ----------
        mean : float
            Mean value (drift target).
        sigma : float
            Noise amplitude.
        tau : float or Quantity
            Time constant.
        duration : float or Quantity
            Total duration.
        n : int
            Number of independent processes.
        t_start : float or Quantity, optional
            Start time.
        t_end : float or Quantity, optional
            End time.
        seed : int, optional
            Random seed.
        """
        super().__init__(duration)
        
        self.mean = mean
        self.sigma = sigma
        self.tau = tau
        self.n = n
        self.t_start = t_start
        self.t_end = t_end
        self.seed = seed
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the OU process array."""
        if self.seed is None:
            rng = brainstate.random.DEFAULT
        else:
            rng = brainstate.random.RandomState(self.seed)
        
        # Get dt unit
        dt_unit = u.get_unit(self.dt)
        dt_value = u.get_magnitude(self.dt)
        
        t_start = 0. * dt_unit if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        # Convert to same unit as dt
        t_start_value = u.Quantity(t_start).to(dt_unit).mantissa if hasattr(t_start, 'unit') else t_start
        t_end_value = u.Quantity(t_end).to(dt_unit).mantissa if hasattr(t_end, 'unit') else t_end
        
        currents = u.math.zeros((self.n_steps, self.n), dtype=brainstate.environ.dftype())
        
        start_i = int(t_start_value / dt_value)
        end_i = int(t_end_value / dt_value)
        
        # Generate OU process
        dt_over_tau = self.dt / self.tau
        noise_amp = self.sigma * u.math.sqrt(2 * dt_over_tau)
        
        ou_values = u.math.ones((end_i - start_i, self.n)) * self.mean
        
        for i in range(1, end_i - start_i):
            drift = dt_over_tau * (self.mean - ou_values[i-1])
            noise = noise_amp * rng.standard_normal(self.n)
            ou_values = ou_values.at[i].set(ou_values[i-1] + drift + noise)
        
        currents = currents.at[start_i:end_i].set(ou_values)
        
        # Squeeze if n=1 for consistency with functional API
        if self.n == 1:
            return u.math.squeeze(currents, axis=-1)
        return currents


class PoissonInput(Input):
    """Generate Poisson spike train input.
    
    Examples
    --------
    >>> # Create Poisson input with rate modulation
    >>> poisson = PoissonInput(50 * u.Hz, 1000 * u.ms, n=3)
    >>> envelope = GaussianPulse(1.0, 500 * u.ms, 100 * u.ms, 1000 * u.ms)
    >>> modulated = poisson * envelope
    """
    
    def __init__(self,
                 rate: u.Quantity,
                 duration: Union[float, u.Quantity],
                 n: int = 1,
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 seed: Optional[int] = None):
        """Initialize Poisson input.
        
        Parameters
        ----------
        rate : Quantity
            Firing rate in Hz.
        duration : float or Quantity
            Total duration.
        n : int
            Number of independent spike trains.
        t_start : float or Quantity, optional
            Start time.
        t_end : float or Quantity, optional
            End time.
        seed : int, optional
            Random seed.
        """
        super().__init__(duration)
        assert rate.unit.dim == u.Hz.dim, f'Rate must be in Hz. Got {rate.unit}.'
        
        self.rate = rate
        self.n = n
        self.t_start = t_start
        self.t_end = t_end
        self.seed = seed
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the Poisson input array."""
        if self.seed is None:
            rng = brainstate.random.DEFAULT
        else:
            rng = brainstate.random.RandomState(self.seed)
        
        # Get dt unit
        dt_unit = u.get_unit(self.dt)
        dt_value = u.get_magnitude(self.dt)
        
        t_start = 0. * dt_unit if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        # Convert to same unit as dt
        t_start_value = u.Quantity(t_start).to(dt_unit).mantissa if hasattr(t_start, 'unit') else t_start
        t_end_value = u.Quantity(t_end).to(dt_unit).mantissa if hasattr(t_end, 'unit') else t_end
        
        currents = u.math.zeros((self.n_steps, self.n), dtype=brainstate.environ.dftype())
        
        start_i = int(t_start_value / dt_value)
        end_i = int(t_end_value / dt_value)
        
        # Generate Poisson spikes
        spike_prob = self.rate * self.dt
        spikes = rng.random((end_i - start_i, self.n)) < spike_prob
        currents = currents.at[start_i:end_i].set(spikes.astype(brainstate.environ.dftype()))
        
        # Squeeze if n=1 for consistency with functional API
        if self.n == 1:
            return u.math.squeeze(currents, axis=-1)
        return currents