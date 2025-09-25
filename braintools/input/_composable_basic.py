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
Composable basic input current generators.
"""

from typing import Sequence, Optional, Union
import brainstate
import brainunit as u
import numpy as np
from ._composable_base import Input

__all__ = [
    'SectionInput',
    'ConstantInput', 
    'StepInput',
    'RampInput',
]


class SectionInput(Input):
    """Format an input current with different sections.
    
    Examples
    --------
    >>> # Create a section input and combine with others
    >>> section = SectionInput(values=[0, 1, 0], durations=[100, 300, 100])
    >>> sine = SinusoidalInput(0.5, 10 * u.Hz, 500)
    >>> combined = section + sine  # Add sinusoidal on top
    """
    
    def __init__(self, 
                 values: Sequence,
                 durations: Sequence,
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize section input.
        
        Parameters
        ----------
        values : list, np.ndarray
            The current values for each period duration.
        durations : list, np.ndarray
            The duration for each period.
        dt : float or Quantity, optional
            The numerical precision.
        """
        if len(durations) != len(values):
            raise ValueError(f'"values" and "durations" must be the same length, while '
                           f'we got {len(values)} != {len(durations)}.')
        
        # Calculate total duration
        total_duration = sum(durations)
        super().__init__(total_duration, dt)
        
        self.values = values
        self.durations = durations
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the section input array."""
        # Get input shape
        values = [u.math.array(val) for val in self.values]
        i_shape = ()
        for val in values:
            shape = u.math.shape(val)
            if len(shape) > len(i_shape):
                i_shape = shape
        
        # Format the current
        currents = []
        for c_size, duration in zip(values, self.durations):
            current = u.math.ones(
                (int(np.ceil(duration / self.dt)),) + i_shape,
                dtype=brainstate.environ.dftype()
            )
            current = current * c_size
            currents.append(current)
        
        return u.math.concatenate(currents, axis=0)


class ConstantInput(Input):
    """Format constant input in durations.
    
    Examples
    --------
    >>> # Create constant input and transform it
    >>> const = ConstantInput([(0, 100), (1, 300), (0, 100)])
    >>> smoothed = const.smooth(tau=10)  # Smooth transitions
    >>> scaled = const.scale(0.5)  # Scale to half amplitude
    """
    
    def __init__(self,
                 I_and_duration: Sequence[tuple],
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize constant input.
        
        Parameters
        ----------
        I_and_duration : list
            This parameter receives the current size and the current
            duration pairs, like `[(Isize1, duration1), (Isize2, duration2)]`.
        dt : float or Quantity, optional
            The numerical precision.
        """
        # Calculate total duration
        total_duration = sum(item[1] for item in I_and_duration)
        super().__init__(total_duration, dt)
        
        self.I_and_duration = I_and_duration
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the constant input array."""
        # Get input shape
        I_shape = ()
        for I in self.I_and_duration:
            shape = u.math.shape(I[0])
            if len(shape) > len(I_shape):
                I_shape = shape
        
        # Get the current
        currents = []
        for c_size, duration in self.I_and_duration:
            length = int(np.ceil(duration / self.dt))
            current = u.math.ones((length,) + I_shape, dtype=brainstate.environ.dftype()) * c_size
            currents.append(current)
        
        return u.math.concatenate(currents, axis=0)


class StepInput(Input):
    """Generate step function input with multiple levels.
    
    Examples
    --------
    >>> # Create step input and combine with noise
    >>> steps = StepInput([0, 1, 0.5], [0, 100, 200], 300)
    >>> noise = WienerProcess(300, sigma=0.1)
    >>> noisy_steps = steps + noise
    """
    
    def __init__(self,
                 amplitudes: Sequence[float],
                 step_times: Sequence[Union[float, u.Quantity]],
                 duration: Union[float, u.Quantity],
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize step input.
        
        Parameters
        ----------
        amplitudes : list or array
            Amplitude values for each step.
        step_times : list or array
            Time points where steps occur.
        duration : float or Quantity
            Total duration of the input.
        dt : float or Quantity, optional
            The numerical precision.
        """
        super().__init__(duration, dt)
        self.amplitudes = amplitudes
        self.step_times = step_times
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the step input array."""
        currents = u.math.zeros(self.n_steps, dtype=brainstate.environ.dftype())
        
        # Convert step times to dimensionless values for sorting
        step_times_ms = [t / u.ms if hasattr(t, 'unit') else t 
                         for t in self.step_times]
        
        # Sort step times and amplitudes together
        sorted_indices = np.argsort(step_times_ms)
        sorted_times = [self.step_times[i] for i in sorted_indices]
        sorted_amps = [self.amplitudes[i] for i in sorted_indices]
        
        # Set amplitude for each interval
        for i, (time, amp) in enumerate(zip(sorted_times, sorted_amps)):
            start_i = int(time / self.dt)
            if i < len(sorted_times) - 1:
                end_i = int(sorted_times[i + 1] / self.dt)
            else:
                end_i = self.n_steps
            
            if start_i < self.n_steps:
                currents = currents.at[start_i:min(end_i, self.n_steps)].set(amp)
        
        return currents


class RampInput(Input):
    """Get the gradually changed input current.
    
    Examples
    --------
    >>> # Create ramp and combine with oscillation
    >>> ramp = RampInput(0, 1, 500)
    >>> sine = SinusoidalInput(0.2, 5 * u.Hz, 500)
    >>> modulated = ramp * sine  # Amplitude modulation
    >>> 
    >>> # Create complex ramp with time window
    >>> ramp2 = RampInput(0, 2, 1000, t_start=200, t_end=800)
    >>> clipped = ramp2.clip(0, 1.5)  # Limit maximum value
    """
    
    def __init__(self,
                 c_start: float,
                 c_end: float,
                 duration: Union[float, u.Quantity],
                 t_start: Optional[Union[float, u.Quantity]] = None,
                 t_end: Optional[Union[float, u.Quantity]] = None,
                 dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize ramp input.
        
        Parameters
        ----------
        c_start : float
            The minimum (or maximum) current size.
        c_end : float
            The maximum (or minimum) current size.
        duration : float or Quantity
            The total duration.
        t_start : float or Quantity, optional
            The ramped current start time-point. Default is 0.
        t_end : float or Quantity, optional
            The ramped current end time-point. Default is duration.
        dt : float or Quantity, optional
            The numerical precision.
        """
        super().__init__(duration, dt)
        u.fail_for_unit_mismatch(c_start, c_end)
        
        self.c_start = c_start
        self.c_end = c_end
        self.t_start = t_start
        self.t_end = t_end
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the ramp input array."""
        t_start = 0. * u.get_unit(self.dt) if self.t_start is None else self.t_start
        t_end = self.duration if self.t_end is None else self.t_end
        
        current = u.math.zeros(self.n_steps,
                               dtype=brainstate.environ.dftype(),
                               unit=u.get_unit(self.c_start))
        
        p1 = int(np.ceil(t_start / self.dt))
        p2 = int(np.ceil(t_end / self.dt))
        cc = u.math.linspace(self.c_start, self.c_end, p2 - p1)
        current = current.at[p1: p2].set(cc)
        
        return current