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
Composable pulse input generators.
"""

from typing import Sequence, Optional, Union
import brainstate
import brainunit as u
import numpy as np
from ._composable_base import Input

__all__ = [
    'SpikeInput',
    'GaussianPulse',
    'ExponentialDecay',
    'DoubleExponential',
    'BurstInput',
]


class SpikeInput(Input):
    """Generate spike input at given times.
    
    Examples
    --------
    >>> # Create spike train and add to background activity
    >>> spikes = SpikeInput([10, 50, 100, 150], sp_sizes=1.0, duration=200)
    >>> background = ConstantInput([(0.1, 200)])
    >>> combined = spikes + background
    """
    
    def __init__(self,
                 sp_times: Sequence[Union[float, u.Quantity]],
                 duration: Union[float, u.Quantity],
                 sp_lens: Union[float, Sequence[float]] = 1.,
                 sp_sizes: Union[float, Sequence[float]] = 1.,
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize spike input.
        
        Parameters
        ----------
        sp_times : list
            The spike time points.
        duration : float or Quantity
            The total duration.
        sp_lens : float or list
            The spike length(s).
        sp_sizes : float or list
            The spike amplitude(s).
        dt : float or Quantity, optional
            The numerical precision.
        """
        super().__init__(duration, dt)
        
        self.sp_times = sp_times
        self.sp_lens = sp_lens if isinstance(sp_lens, (list, tuple)) else [sp_lens] * len(sp_times)
        self.sp_sizes = sp_sizes if isinstance(sp_sizes, (list, tuple)) else [sp_sizes] * len(sp_times)
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the spike input array."""
        currents = u.math.zeros(self.n_steps, dtype=brainstate.environ.dftype())
        
        for time, size, length in zip(self.sp_times, self.sp_sizes, self.sp_lens):
            start = int(time / self.dt)
            end = int((time + length) / self.dt)
            currents = currents.at[start:end].set(size)
        
        return currents


class GaussianPulse(Input):
    """Generate a Gaussian pulse input.
    
    Examples
    --------
    >>> # Create multiple Gaussian pulses
    >>> pulse1 = GaussianPulse(1.0, 100 * u.ms, 20 * u.ms, 500 * u.ms)
    >>> pulse2 = GaussianPulse(0.8, 300 * u.ms, 30 * u.ms, 500 * u.ms)
    >>> double_pulse = pulse1 + pulse2
    """
    
    def __init__(self,
                 amplitude: float,
                 center: Union[float, u.Quantity],
                 sigma: Union[float, u.Quantity],
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize Gaussian pulse.
        
        Parameters
        ----------
        amplitude : float
            Amplitude of the pulse.
        center : float or Quantity
            Center time of the pulse.
        sigma : float or Quantity
            Standard deviation (width) of the pulse.
        duration : float or Quantity
            Total duration of the input.
        dt : float or Quantity, optional
            The numerical precision.
        """
        super().__init__(duration, dt)
        
        self.amplitude = amplitude
        self.center = center
        self.sigma = sigma
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the Gaussian pulse array."""
        times = u.math.arange(0. * u.ms, self.duration, self.dt)
        gaussian = self.amplitude * u.math.exp(-0.5 * ((times - self.center) / self.sigma) ** 2)
        return gaussian


class ExponentialDecay(Input):
    """Generate exponential decay input.
    
    Examples
    --------
    >>> # Create exponential decay with step onset
    >>> decay = ExponentialDecay(2.0, 30 * u.ms, 500 * u.ms, t_start=100 * u.ms)
    >>> step = StepInput([0, 1], [0, 100], 500 * u.ms)
    >>> gated_decay = decay * step  # Gate the decay
    """
    
    def __init__(self,
                 amplitude: float,
                 tau: Union[float, u.Quantity],
                 duration: Union[float, u.Quantity],
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize exponential decay.
        
        Parameters
        ----------
        amplitude : float
            Initial amplitude of the decay.
        tau : float or Quantity
            Decay time constant.
        duration : float or Quantity
            Total duration of the input.
        t_start : float or Quantity, optional
            Start time of the decay.
        t_end : float or Quantity, optional
            End time of the decay.
        dt : float or Quantity, optional
            The numerical precision.
        """
        super().__init__(duration, dt)
        
        self.amplitude = amplitude
        self.tau = tau
        self.t_start = t_start
        self.t_end = t_end
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the exponential decay array."""
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        times = u.math.arange(0. * u.ms, t_end - t_start, self.dt)
        exp_decay = self.amplitude * u.math.exp(-times / self.tau)
        
        currents = u.math.zeros(self.n_steps,
                                dtype=brainstate.environ.dftype(),
                                unit=u.get_unit(exp_decay))
        start_i = int(t_start / self.dt)
        end_i = int(t_end / self.dt)
        currents = currents.at[start_i:end_i].set(exp_decay)
        
        return currents


class DoubleExponential(Input):
    """Generate double exponential (alpha function) input.
    
    Examples
    --------
    >>> # Create synaptic-like current with noise
    >>> alpha = DoubleExponential(1.0, 5 * u.ms, 20 * u.ms, 200 * u.ms)
    >>> noise = WienerProcess(200 * u.ms, sigma=0.05)
    >>> synaptic = alpha + noise
    """
    
    def __init__(self,
                 amplitude: float,
                 tau_rise: Union[float, u.Quantity],
                 tau_decay: Union[float, u.Quantity],
                 duration: Union[float, u.Quantity],
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize double exponential.
        
        Parameters
        ----------
        amplitude : float
            Peak amplitude.
        tau_rise : float or Quantity
            Rise time constant.
        tau_decay : float or Quantity
            Decay time constant.
        duration : float or Quantity
            Total duration of the input.
        t_start : float or Quantity, optional
            Start time of the pulse.
        dt : float or Quantity, optional
            The numerical precision.
        """
        super().__init__(duration, dt)
        
        self.amplitude = amplitude
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.t_start = t_start
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the double exponential array."""
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        
        times = u.math.arange(0. * u.ms, self.duration, self.dt)
        times_shifted = times - t_start
        
        # Alpha function: amplitude * (exp(-t/tau_decay) - exp(-t/tau_rise))
        # Only for t >= 0
        alpha = u.math.where(
            times_shifted >= 0 * u.ms,
            self.amplitude * (u.math.exp(-times_shifted / self.tau_decay) - 
                            u.math.exp(-times_shifted / self.tau_rise)),
            0.0
        )
        
        # Normalize to peak at amplitude
        if self.tau_rise != self.tau_decay:
            t_peak = (self.tau_rise * self.tau_decay) / (self.tau_decay - self.tau_rise) * \
                    u.math.log(self.tau_decay / self.tau_rise)
            norm_factor = u.math.exp(-t_peak / self.tau_decay) - u.math.exp(-t_peak / self.tau_rise)
            alpha = alpha / norm_factor
        
        return alpha


class BurstInput(Input):
    """Generate burst pattern input.
    
    Examples
    --------
    >>> # Create burst pattern with ramped amplitude
    >>> burst = BurstInput(5, 30 * u.ms, 70 * u.ms, 1.0, 500 * u.ms)
    >>> ramp = RampInput(0.5, 1.5, 500 * u.ms)
    >>> modulated_burst = burst * ramp
    """
    
    def __init__(self,
                 n_bursts: int,
                 burst_duration: Union[float, u.Quantity],
                 inter_burst_interval: Union[float, u.Quantity],
                 amplitude: float,
                 duration: Union[float, u.Quantity],
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize burst input.
        
        Parameters
        ----------
        n_bursts : int
            Number of bursts.
        burst_duration : float or Quantity
            Duration of each burst.
        inter_burst_interval : float or Quantity
            Interval between burst starts.
        amplitude : float
            Amplitude during bursts.
        duration : float or Quantity
            Total duration of the input.
        t_start : float or Quantity, optional
            Start time of first burst.
        dt : float or Quantity, optional
            The numerical precision.
        """
        super().__init__(duration, dt)
        
        self.n_bursts = n_bursts
        self.burst_duration = burst_duration
        self.inter_burst_interval = inter_burst_interval
        self.amplitude = amplitude
        self.t_start = t_start
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the burst input array."""
        t_start = 0. * u.ms if self.t_start is None else self.t_start
        
        currents = u.math.zeros(self.n_steps, dtype=brainstate.environ.dftype())
        
        for i in range(self.n_bursts):
            burst_start = t_start + i * self.inter_burst_interval
            burst_end = burst_start + self.burst_duration
            
            start_i = int(burst_start / self.dt)
            end_i = int(burst_end / self.dt)
            
            if start_i >= self.n_steps:
                break
            if end_i > self.n_steps:
                end_i = self.n_steps
            
            currents = currents.at[start_i:end_i].set(self.amplitude)
        
        return currents