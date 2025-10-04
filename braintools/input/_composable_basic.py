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
Composable basic input current generators.

This module provides composable versions of basic input current generators
that can be combined using operators and transformations.
"""

import functools
from typing import Sequence, Optional, Union

import brainstate
import brainunit as u

from ._composable_base import Input
from ._deprecation import create_deprecated_class
from ._functional_basic import section, constant, step, ramp

__all__ = [
    'Section',
    'Constant',
    'Step',
    'Ramp',
]


class Section(Input):
    """Generate input current with different sections.
    
    A section input consists of different constant values maintained for
    specified durations. This is useful for creating protocols with distinct
    phases, such as baseline, stimulation, and recovery periods.
    
    Parameters
    ----------
    values : Sequence
        The current values for each period. Can be scalars or arrays for
        multi-channel inputs. Units are preserved if provided.
    durations : Sequence
        The duration for each period. Should have the same length as values.
        Can be specified with or without units.
    
    Attributes
    ----------
    values : Sequence
        The stored current values for each section.
    durations : Sequence
        The stored durations for each section.
    duration : Quantity or float
        Total duration calculated as sum of all section durations.
    
    Raises
    ------
    ValueError
        If values and durations have different lengths.
    
    See Also
    --------
    Constant : Similar but with (value, duration) pairs.
    Step : Creates steps at specific time points.
    
    Notes
    -----
    The Section class uses the functional API internally to generate
    the actual current arrays. It provides a composable interface that
    allows combining with other inputs using operators.
    
    Examples
    --------
    Simple three-phase protocol:
    
    .. code-block:: python

        >>> section = Section(
        ...     values=[0, 1, 0] * u.pA,
        ...     durations=[100, 300, 100] * u.ms
        ... )
        >>> array = section()  # Generate the array
    
    Multi-channel input:
    
    .. code-block:: python

        >>> values = [np.zeros(3), np.ones(3) * 5, np.zeros(3)] * u.nA
        >>> section = Section(
        ...     values=values,
        ...     durations=[50, 100, 50] * u.ms
        ... )
    
    Combine with other inputs:
    
    .. code-block:: python

        >>> # Add noise to section input
        >>> from braintools.input import WienerProcess
        >>> noisy_section = section + WienerProcess(500 * u.ms, sigma=0.1)
    

        >>> # Modulate with sinusoid
        >>> from braintools.input import Sinusoidal
        >>> sine = Sinusoidal(0.2, 10 * u.Hz, 500 * u.ms)
        >>> modulated = section * (1 + sine)
    
    Complex protocol with smooth transitions:
    
    .. code-block:: python

        >>> # Create step protocol and smooth it
        >>> protocol = Section(
        ...     values=[0, 0.5, 1.0, 1.5, 1.0, 0.5, 0],
        ...     durations=[50, 30, 100, 150, 100, 30, 50]
        ... )
        >>> smooth_protocol = protocol.smooth(tau=10 * u.ms)
    
    Sequential composition:
    
    .. code-block:: python

        >>> baseline = Section([0], [200])
        >>> stim = Section([0.5, 1.0, 0.5], [50, 100, 50])
        >>> recovery = Section([0], [200])
        >>> full_protocol = baseline & stim & recovery
    """
    __module__ = 'braintools.input'

    def __init__(
        self,
        values: Sequence,
        durations: Sequence
    ):
        """Initialize section input.
        
        Parameters
        ----------
        values : Sequence
            The current values for each period.
        durations : Sequence
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

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the section input array."""
        # Use the functional API
        return section(self.values, self.durations, return_length=False)


class Constant(Input):
    """Generate constant input with specified durations.
    
    Creates a piecewise constant input where each piece has a specific
    value and duration. This is similar to Section but uses
    (value, duration) pairs for convenience.
    
    Parameters
    ----------
    I_and_duration : Sequence[tuple]
        List of (value, duration) pairs. Each tuple specifies the current
        value and how long it should be maintained. Values can include units.
    
    Attributes
    ----------
    I_and_duration : Sequence[tuple]
        The stored (value, duration) pairs.
    duration : Quantity or float
        Total duration calculated as sum of all durations.
    
    See Also
    --------
    Section : Similar but with separate values and durations lists.
    Step : Creates steps at specific time points.
    
    Notes
    -----
    Constant internally uses the functional constant API.
    The composable interface allows for easy combination with other
    inputs and transformations.
    
    Examples
    --------
    Simple two-phase protocol:
    
    .. code-block:: python

        >>> const = Constant([
        ...     (0 * u.pA, 100 * u.ms),
        ...     (10 * u.pA, 200 * u.ms)
        ... ])
        >>> array = const()
    
    Multi-step current injection:
    
    .. code-block:: python

        >>> # Incrementally increasing steps
        >>> steps = Constant([
        ...     (0 * u.nA, 50 * u.ms),
        ...     (0.5 * u.nA, 50 * u.ms),
        ...     (1.0 * u.nA, 50 * u.ms),
        ...     (1.5 * u.nA, 50 * u.ms),
        ...     (0 * u.nA, 50 * u.ms),
        ... ])
    
    Smooth transitions between levels:
    
    .. code-block:: python

        >>> # Create sharp steps and smooth them
        >>> const = Constant([
        ...     (0, 100),
        ...     (1, 100),
        ...     (0.5, 100),
        ...     (0, 100)
        ... ])
        >>> smoothed = const.smooth(tau=20 * u.ms)
    
    Combine with oscillations:
    
    .. code-block:: python

        >>> from braintools.input import Sinusoidal
        >>> baseline = Constant([(0.5, 500)])
        >>> oscillation = Sinusoidal(0.2, 5 * u.Hz, 500)
        >>> combined = baseline + oscillation
    
    Create complex protocols:
    
    .. code-block:: python

        >>> # Paired-pulse protocol
        >>> protocol = Constant([
        ...     (0 * u.pA, 100 * u.ms),    # baseline
        ...     (5 * u.pA, 20 * u.ms),     # first pulse
        ...     (0 * u.pA, 50 * u.ms),     # inter-pulse interval
        ...     (5 * u.pA, 20 * u.ms),     # second pulse
        ...     (0 * u.pA, 100 * u.ms),    # recovery
        ... ])
        >>> # Add noise for more realistic stimulation
        >>> from braintools.input import WienerProcess
        >>> noisy_protocol = protocol + WienerProcess(290 * u.ms, sigma=0.1)
    
    Use transformations:
    
    .. code-block:: python

        >>> # Scale amplitude
        >>> scaled = const.scale(0.5)
        >>>
        >>> # Clip to physiological range
        >>> clipped = const.clip(-80, 40)
        >>>
        >>> # Repeat pattern
        >>> repeated = const.repeat(3)
    """
    __module__ = 'braintools.input'

    def __init__(self, I_and_duration: Sequence[tuple]):
        """Initialize constant input.
        
        Parameters
        ----------
        I_and_duration : Sequence[tuple]
            List of (value, duration) pairs.
        """
        # Calculate total duration
        total_duration = functools.reduce(u.math.add, [item[1] for item in I_and_duration])
        super().__init__(total_duration)

        self.I_and_duration = I_and_duration

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the constant input array."""
        # Use the functional API - it returns (current, duration) tuple
        current, _ = constant(self.I_and_duration)
        return current


class Step(Input):
    """Generate step function input with multiple levels.
    
    Creates a step function where the input jumps to specified amplitude
    values at given time points. This is useful for protocols requiring
    instantaneous changes in stimulation level.
    
    Parameters
    ----------
    amplitudes : Sequence[float]
        Amplitude values for each step. The length should match step_times.
        Can include units for dimensional consistency.
    step_times : Sequence[Union[float, u.Quantity]]
        Time points where steps occur. Will be automatically sorted if not
        in ascending order.
    duration : Union[float, u.Quantity]
        Total duration of the input signal.
    
    Attributes
    ----------
    amplitudes : Sequence[float]
        The stored amplitude values.
    step_times : Sequence
        The stored step time points.
    duration : Quantity or float
        The total duration of the input.
    
    See Also
    --------
    Section : For specifying durations instead of time points.
    Constant : For piecewise constant inputs with durations.
    Ramp : For linearly changing inputs.
    
    Notes
    -----
    If step_times are not sorted, they will be automatically sorted along
    with their corresponding amplitudes. The functional step API is
    used internally for array generation.
    
    Examples
    --------
    Simple three-level step function:
    
    .. code-block:: python

        >>> steps = Step(
        ...     amplitudes=[0, 10, 5] * u.pA,
        ...     step_times=[0, 50, 150] * u.ms,
        ...     duration=200 * u.ms
        ... )
        >>> array = steps()
    
    Staircase protocol for I-V curve:
    
    .. code-block:: python

        >>> # Incrementally increasing current steps
        >>> amplitudes = np.arange(0, 101, 10) * u.pA
        >>> times = np.arange(0, 1100, 100) * u.ms
        >>> staircase = Step(amplitudes, times, 1200 * u.ms)
    
    Multiple pulses with return to baseline:
    
    .. code-block:: python

        >>> pulses = Step(
        ...     amplitudes=[0, 5, 0, 10, 0, 15, 0] * u.pA,
        ...     step_times=[0, 20, 40, 60, 80, 100, 120] * u.ms,
        ...     duration=150 * u.ms
        ... )
    
    Combine with noise for realistic stimulation:
    
    .. code-block:: python

        >>> from braintools.input import WienerProcess
        >>> steps = Step([0, 1, 0.5], [0, 100, 200] * u.ms, 300 * u.ms)
        >>> noise = WienerProcess(300 * u.ms, sigma=0.1)
        >>> noisy_steps = steps + noise
    
    Create complex protocols with transformations:
    
    .. code-block:: python

        >>> # Smoothed steps for gradual transitions
        >>> sharp_steps = Step(
        ...     [0, 1, 0.5, 1, 0],
        ...     [0, 50, 100, 150, 200] * u.ms,
        ...     250 * u.ms
        ... )
        >>> smooth_steps = sharp_steps.smooth(tau=10 * u.ms)
        >>>
        >>> # Clipped to physiological range
        >>> clipped = sharp_steps.clip(0, 0.8)
    
    Unsorted times are automatically handled:
    
    .. code-block:: python

        >>> # Times will be sorted to [0, 50, 100]
        >>> steps = Step(
        ...     amplitudes=[5, 0, 10] * u.pA,
        ...     step_times=[50, 0, 100] * u.ms,
        ...     duration=150 * u.ms
        ... )
    
    Sequential composition:
    
    .. code-block:: python

        >>> baseline = Step([0], [0 * u.ms], 100 * u.ms)
        >>> test = Step([0, 1, 0], [0, 20, 80] * u.ms, 100 * u.ms)
        >>> protocol = baseline & test & baseline
    """
    __module__ = 'braintools.input'

    def __init__(self,
                 amplitudes: Sequence[float],
                 step_times: Sequence[Union[float, u.Quantity]],
                 duration: Union[float, u.Quantity] = None):
        """Initialize step input.
        
        Parameters
        ----------
        amplitudes : Sequence[float]
            Amplitude values for each step.
        step_times : Sequence
            Time points where steps occur.
        duration : Union[float, u.Quantity]
            Total duration of the input.
        """
        duration = duration if duration is not None else functools.reduce(lambda x, y: x + y, step_times)
        super().__init__(duration)
        self.amplitudes = amplitudes
        self.step_times = step_times

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the step input array."""
        return step(
            self.amplitudes,
            self.step_times,
            self.duration
        )


class Ramp(Input):
    """Generate linearly changing (ramp) input current.
    
    Creates a linear ramp from a starting value to an ending value over
    a specified duration. Optionally, the ramp can be confined to a
    specific time window within the total duration.
    
    Parameters
    ----------
    c_start : float
        The starting current value. Can include units.
    c_end : float
        The ending current value. Must have same units as c_start.
    duration : Union[float, u.Quantity]
        The total duration of the input signal.
    t_start : Union[float, u.Quantity], optional
        Time point when the ramp starts. Before this, the output is c_start.
        Default is 0.
    t_end : Union[float, u.Quantity], optional
        Time point when the ramp ends. After this, the output is c_end.
        Default is duration.
    
    Attributes
    ----------
    c_start : float
        The starting current value.
    c_end : float  
        The ending current value.
    duration : Quantity or float
        The total duration.
    t_start : Quantity or float or None
        The ramp start time.
    t_end : Quantity or float or None
        The ramp end time.
    
    Raises
    ------
    UnitMismatchError
        If c_start and c_end have incompatible units.
    
    See Also
    --------
    Step : For instantaneous changes.
    Section : For piecewise constant inputs.
    
    Notes
    -----
    The ramp is linear between t_start and t_end. Before t_start, the
    output is c_start. After t_end, the output is c_end. This uses the
    functional ramp API internally.
    
    Examples
    --------
    Simple linear ramp:
    
    .. code-block:: python

        >>> ramp = Ramp(
        ...     c_start=0 * u.pA,
        ...     c_end=10 * u.pA,
        ...     duration=100 * u.ms
        ... )
        >>> array = ramp()
    
    Decreasing ramp (from high to low):
    
    .. code-block:: python

        >>> down_ramp = Ramp(
        ...     c_start=10 * u.pA,
        ...     c_end=0 * u.pA,
        ...     duration=100 * u.ms
        ... )
    
    Ramp with delay and early stop:
    
    .. code-block:: python

        >>> # Ramp starts at 50ms and ends at 150ms
        >>> delayed_ramp = Ramp(
        ...     c_start=0 * u.nA,
        ...     c_end=5 * u.nA,
        ...     duration=200 * u.ms,
        ...     t_start=50 * u.ms,
        ...     t_end=150 * u.ms
        ... )
    
    Combine with oscillations for amplitude modulation:
    
    .. code-block:: python

        >>> from braintools.input import Sinusoidal
        >>> envelope = Ramp(0, 1, 500 * u.ms)
        >>> carrier = Sinusoidal(1.0, 20 * u.Hz, 500 * u.ms)
        >>> am_signal = envelope * carrier
    
    Create sawtooth wave by repeating:
    
    .. code-block:: python

        >>> single_tooth = Ramp(0, 1, 50 * u.ms)
        >>> sawtooth = single_tooth.repeat(10)  # 500ms total
    
    Complex protocols with transformations:
    
    .. code-block:: python

        >>> # Ramp with saturation
        >>> ramp = Ramp(-2, 2, 400 * u.ms)
        >>> saturated = ramp.clip(-1, 1)
        >>>
        >>> # Smoothed ramp (reduces sharp corners)
        >>> smooth_ramp = ramp.smooth(tau=5 * u.ms)
    
    I-V curve measurement protocol:
    
    .. code-block:: python

        >>> # Slow voltage ramp for I-V curves
        >>> iv_ramp = Ramp(
        ...     c_start=-100 * u.pA,
        ...     c_end=100 * u.pA,
        ...     duration=1000 * u.ms
        ... )
        >>> # Add small oscillation to avoid hysteresis
        >>> from braintools.input import Sinusoidal
        >>> wobble = Sinusoidal(5 * u.pA, 100 * u.Hz, 1000 * u.ms)
        >>> iv_protocol = iv_ramp + wobble
    
    Sequential ramps for plasticity protocols:
    
    .. code-block:: python

        >>> up_ramp = Ramp(0, 1, 100 * u.ms)
        >>> plateau = Constant([(1, 50)])
        >>> down_ramp = Ramp(1, 0, 100 * u.ms)
        >>> protocol = up_ramp & plateau & down_ramp
    """
    __module__ = 'braintools.input'

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
            The starting current value.
        c_end : float
            The ending current value.
        duration : Union[float, u.Quantity]
            The total duration.
        t_start : Union[float, u.Quantity], optional
            The ramp start time. Default is 0.
        t_end : Union[float, u.Quantity], optional
            The ramp end time. Default is duration.
        """
        super().__init__(duration)
        u.fail_for_unit_mismatch(c_start, c_end)

        self.c_start = c_start
        self.c_end = c_end
        self.t_start = t_start
        self.t_end = t_end

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the ramp input array."""
        # Use the functional API
        return ramp(
            self.c_start,
            self.c_end,
            self.duration,
            self.t_start,
            self.t_end
        )


SectionInput = create_deprecated_class(Section, 'SectionInput', 'Section')
ConstantInput = create_deprecated_class(Constant, 'ConstantInput', 'Constant')
StepInput = create_deprecated_class(Step, 'StepInput', 'Step')
RampInput = create_deprecated_class(Ramp, 'RampInput', 'Ramp')

__all__.extend(['SectionInput', 'ConstantInput', 'StepInput', 'RampInput'])
