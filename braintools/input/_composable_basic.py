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

import functools
from typing import Sequence, Optional, Union

import brainstate
import brainunit as u
import numpy as np

from ._composable_base import Input
from ._functional_basic import section_input, constant_input, step_input, ramp_input

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

    def __init__(
        self,
        values: Sequence,
        durations: Sequence
    ):
        """Initialize section input.
        
        Parameters
        ----------
        values : list, np.ndarray
            The current values for each period duration.
        durations : list, np.ndarray
            The duration for each period.
        """
        if len(durations) != len(values):
            raise ValueError(f'"values" and "durations" must be the same length, while '
                             f'we got {len(values)} != {len(durations)}.')

        # Calculate total duration
        total_duration = functools.reduce(u.math.add, durations)
        super().__init__(total_duration)

        self.values = values
        self.durations = durations

    def _generate(self) -> brainstate.typing.ArrayLike:
        """Generate the section input array."""
        # Use the functional API
        return section_input(self.values, self.durations, return_length=False)


class ConstantInput(Input):
    """Format constant input in durations.
    
    Examples
    --------
    >>> # Create constant input and transform it
    >>> const = ConstantInput([(0, 100), (1, 300), (0, 100)])
    >>> smoothed = const.smooth(tau=10)  # Smooth transitions
    >>> scaled = const.scale(0.5)  # Scale to half amplitude
    """

    def __init__(self, I_and_duration: Sequence[tuple]):
        """Initialize constant input.
        
        Parameters
        ----------
        I_and_duration : list
            This parameter receives the current size and the current
            duration pairs, like `[(Isize1, duration1), (Isize2, duration2)]`.
        """
        # Calculate total duration
        total_duration = functools.reduce(u.math.add, [item[1] for item in I_and_duration])
        super().__init__(total_duration)

        self.I_and_duration = I_and_duration

    def _generate(self) -> brainstate.typing.ArrayLike:
        """Generate the constant input array."""
        # Use the functional API
        return constant_input(self.I_and_duration)


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
                 duration: Union[float, u.Quantity]):
        """Initialize step input.
        
        Parameters
        ----------
        amplitudes : list or array
            Amplitude values for each step.
        step_times : list or array
            Time points where steps occur.
        duration : float or Quantity
            Total duration of the input.
        """
        super().__init__(duration)
        self.amplitudes = amplitudes
        self.step_times = step_times

    def _generate(self) -> brainstate.typing.ArrayLike:
        """Generate the step input array."""
        # Convert step_times to have proper units if they don't
        dt_unit = u.get_unit(self.dt)
        step_times_with_units = []
        for t in self.step_times:
            if hasattr(t, 'unit'):
                step_times_with_units.append(t)
            else:
                # Assume it's in the same unit as dt if no unit is provided
                step_times_with_units.append(t * dt_unit)
        
        # Use the functional API
        return step_input(
            self.amplitudes,
            step_times_with_units,
            self.duration
        )


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
                 t_end: Optional[Union[float, u.Quantity]] = None):
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
        """
        super().__init__(duration)
        u.fail_for_unit_mismatch(c_start, c_end)

        self.c_start = c_start
        self.c_end = c_end
        self.t_start = t_start
        self.t_end = t_end

    def _generate(self) -> brainstate.typing.ArrayLike:
        """Generate the ramp input array."""
        # Use the functional API
        return ramp_input(
            self.c_start,
            self.c_end,
            self.duration,
            self.t_start,
            self.t_end
        )
