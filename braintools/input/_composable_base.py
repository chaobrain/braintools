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
Base classes for composable input current construction.

This module provides the fundamental building blocks for creating complex,
composable input currents for neural simulations. The key feature is the
ability to combine simple inputs using mathematical operators and
transformations to build sophisticated stimulation protocols.
"""
import functools
from typing import Optional, Union, Callable

import brainstate
import brainunit as u
import numpy as np

from ._deprecation import create_deprecated_class

__all__ = [
    'Input',
    'Composite',
    'ConstantValue',
    'Sequential',
    'TimeShifted',
    'Clipped',
    'Smoothed',
    'Repeated',
    'Transformed'
]


class Input:
    """Base class for composable input currents.
    
    This class provides a composable API for building complex input currents
    by combining simpler components using operators and transformations.
    
    Parameters
    ----------
    duration : float or Quantity
        The total duration of the input signal. If float, assumes the same
        unit as the global dt setting.
    
    Attributes
    ----------
    duration : float or Quantity
        The total duration of the input.
    dt : Quantity
        The time step, retrieved from global environment.
    n_steps : int
        Number of time steps in the generated array.
    shape : tuple
        Shape of the generated input array.
    
    Methods
    -------
    __call__(recompute=False)
        Generate and return the input current array.
    scale(factor)
        Scale the input by a constant factor.
    shift(time_shift)
        Shift the input in time (delay or advance).
    clip(min_val, max_val)
        Clip the input values to a specified range.
    smooth(tau)
        Apply exponential smoothing with time constant tau.
    repeat(n_times)
        Repeat the input pattern multiple times.
    apply(func)
        Apply a custom transformation function.
    
    Notes
    -----
    The Input class supports the following operators:
    - Addition (+): Add two inputs or add a constant
    - Subtraction (-): Subtract inputs or constants
    - Multiplication (*): Multiply inputs or scale by constant
    - Division (/): Divide inputs or divide by constant
    - Negation (-): Negate the input values
    - Sequential (&): Concatenate inputs in time
    - Overlay (|): Take maximum of two inputs at each point
    
    Examples
    --------
    Basic arithmetic operations:
    
    .. code-block:: python

        >>> import brainunit as u
        >>> import brainstate
        >>> from braintools.input import Ramp, Sinusoidal, Step
        >>> brainstate.environ.set(dt=0.1 * u.ms)
    

        >>> # Add two inputs
        >>> ramp = Ramp(0, 1, 500 * u.ms)
        >>> sine = Sinusoidal(0.5, 10 * u.Hz, 500 * u.ms)
        >>> combined = ramp + sine
    

        >>> # Scale an input
        >>> scaled_ramp = ramp * 2.0
        >>> half_sine = sine.scale(0.5)
    

        >>> # Subtract a baseline
        >>> centered = sine - 0.25
    
    Complex compositions:
    
    .. code-block:: python

        >>> # Amplitude modulation
        >>> carrier = Sinusoidal(1.0, 100 * u.Hz, 1000 * u.ms)
        >>> envelope = Ramp(0, 1, 1000 * u.ms)
        >>> am_signal = carrier * envelope
    

        >>> # Sequential stimulation protocol
        >>> baseline = Step([0], [0 * u.ms], 200 * u.ms)
        >>> stim = Step([1], [0 * u.ms], 500 * u.ms)
        >>> recovery = Step([0], [0 * u.ms], 300 * u.ms)
        >>> protocol = baseline & stim & recovery
    

        >>> # Overlay (maximum) for redundant stimulation
        >>> stim1 = Step([0, 1, 0], [0, 100, 400] * u.ms, 500 * u.ms)
        >>> stim2 = Step([0, 0.8, 0], [0, 200, 450] * u.ms, 500 * u.ms)
        >>> combined_stim = stim1 | stim2
    
    Transformations:
    
    .. code-block:: python

        >>> # Time shifting for delayed responses
        >>> delayed_sine = sine.shift(50 * u.ms)
        >>> advanced_ramp = ramp.shift(-20 * u.ms)
    

        >>> # Clipping for saturation effects
        >>> clipped = (ramp * 2).clip(0, 1.5)
    

        >>> # Smoothing for filtering
        >>> smooth_steps = Step([0, 1, 0.5, 1, 0],
        ...                     [0, 100, 200, 300, 400] * u.ms,
        ...                     500 * u.ms).smooth(10 * u.ms)
    

        >>> # Repeating patterns
        >>> burst = Step([0, 1, 0], [0, 10, 20] * u.ms, 50 * u.ms)
        >>> repeated_bursts = burst.repeat(10)
    

        >>> # Custom transformations
        >>> import jax.numpy as jnp
        >>> rectified = sine.apply(lambda x: jnp.maximum(x, 0))
        >>> squared = sine.apply(lambda x: x ** 2)
    
    Advanced protocols:
    
    .. code-block:: python

        >>> # Complex experimental protocol
        >>> pre_baseline = Step([0], [0 * u.ms], 1000 * u.ms)
        >>> conditioning = Sinusoidal(0.5, 5 * u.Hz, 2000 * u.ms)
        >>> test_pulse = Step([2], [0 * u.ms], 100 * u.ms)
        >>> post_baseline = Step([0], [0 * u.ms], 1000 * u.ms)
        >>>
        >>> protocol = (pre_baseline &
        ...            (conditioning + 0.5).clip(0, 1) &
        ...            test_pulse &
        ...            post_baseline)
    

        >>> # Noisy modulated signal
        >>> from braintools.input import WienerProcess
        >>> signal = Sinusoidal(1.0, 20 * u.Hz, 1000 * u.ms)
        >>> noise = WienerProcess(1000 * u.ms, sigma=0.1)
        >>> modulator = (Ramp(0.5, 1.5, 1000 * u.ms) +
        ...             Sinusoidal(0.2, 2 * u.Hz, 1000 * u.ms))
        >>> noisy_modulated = (signal + noise) * modulator
    """

    __module__ = 'braintools.input'

    def __init__(self, duration: Union[float, u.Quantity]):
        """Initialize the Input base class.
        
        Parameters
        ----------
        duration : float or Quantity
            The total duration of the input.
        """
        self.duration = duration
        self._cached_array = None

    @property
    def dt(self):
        """Get the time step from global environment.
        
        Returns
        -------
        dt : Quantity
            The simulation time step.
        """
        return brainstate.environ.get_dt()

    def __call__(self, recompute: bool = False) -> brainstate.typing.ArrayLike:
        """Generate and return the input current array.
        
        Parameters
        ----------
        recompute : bool, default=False
            If True, force recomputation even if cached.
            If False, use cached result if available.
            
        Returns
        -------
        current : array or Quantity
            The generated input current array with shape (n_steps,).
            
        Examples
        --------

        .. code-block:: python

            >>> ramp = Ramp(0, 1, 100 * u.ms)
            >>> # First call generates and caches
            >>> arr1 = ramp()
            >>> # Second call uses cache (faster)
            >>> arr2 = ramp()
            >>> # Force regeneration
            >>> arr3 = ramp(recompute=True)
        """
        if self._cached_array is None or recompute:
            self._cached_array = self.generate()
        return self._cached_array

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the input current array. 
        
        Must be implemented by subclasses.
        
        Returns
        -------
        array : ndarray or Quantity
            The generated input array.
            
        Raises
        ------
        NotImplementedError
            If called on base Input class.
        """
        raise NotImplementedError("Subclasses must implement _generate()")

    @property
    def shape(self):
        """Get the shape of the input array.
        
        Returns
        -------
        shape : tuple
            Shape of the generated array.
        """
        return self().shape

    @property
    def n_steps(self):
        """Get the number of time steps.
        
        Returns
        -------
        n_steps : int
            Number of time steps based on duration and dt.
        """
        return int(np.ceil(self.duration / self.dt))

    def __add__(self, other):
        """Add two inputs or add a constant.
        
        Parameters
        ----------
        other : Input or scalar
            Another input to add, or a constant value.
            
        Returns
        -------
        result : Composite
            The sum of the two inputs.
            
        Examples
        --------

        .. code-block:: python

            >>> sine1 = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
            >>> sine2 = Sinusoidal(0.5, 20 * u.Hz, 100 * u.ms)
            >>> # Add two inputs
            >>> combined = sine1 + sine2
            >>> # Add a DC offset
            >>> with_offset = sine1 + 0.5
        """
        if isinstance(other, Input):
            return Composite(self, other, operator='+')
        else:
            return Composite(self, ConstantValue(other, self.duration), operator='+')

    def __radd__(self, other):
        """Right addition (for scalar + Input)."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract two inputs or subtract a constant.
        
        Parameters
        ----------
        other : Input or scalar
            Another input to subtract, or a constant value.
            
        Returns
        -------
        result : Composite
            The difference of the two inputs.
            
        Examples
        --------

        .. code-block:: python

            >>> ramp = Ramp(0, 2, 100 * u.ms)
            >>> baseline = Step([0.5], [0], 100 * u.ms)
            >>> # Subtract baseline
            >>> corrected = ramp - baseline
            >>> # Remove DC offset
            >>> centered = ramp - 1.0
        """
        if isinstance(other, Input):
            return Composite(self, other, operator='-')
        else:
            return Composite(self, ConstantValue(other, self.duration), operator='-')

    def __rsub__(self, other):
        """Right subtraction (for scalar - Input)."""
        if isinstance(other, Input):
            return Composite(other, self, operator='-')
        else:
            return Composite(ConstantValue(other, self.duration), self, operator='-')

    def __mul__(self, other):
        """Multiply two inputs or multiply by a constant.
        
        Parameters
        ----------
        other : Input or scalar
            Another input for modulation, or a scaling factor.
            
        Returns
        -------
        result : Composite
            The product of the two inputs.
            
        Examples
        --------

        .. code-block:: python

            >>> carrier = Sinusoidal(1.0, 100 * u.Hz, 500 * u.ms)
            >>> envelope = Ramp(0, 1, 500 * u.ms)
            >>> # Amplitude modulation
            >>> am_signal = carrier * envelope
            >>> # Simple scaling
            >>> doubled = carrier * 2.0
        """
        if isinstance(other, Input):
            return Composite(self, other, operator='*')
        else:
            return Composite(self, ConstantValue(other, self.duration), operator='*')

    def __rmul__(self, other):
        """Right multiplication (for scalar * Input)."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide two inputs or divide by a constant.
        
        Parameters
        ----------
        other : Input or scalar
            Divisor input or constant.
            
        Returns
        -------
        result : Composite
            The quotient of the two inputs.
            
        Examples
        --------

        .. code-block:: python

            >>> signal = Sinusoidal(2.0, 10 * u.Hz, 100 * u.ms)
            >>> normalizer = Ramp(1, 2, 100 * u.ms)
            >>> # Normalize by varying factor
            >>> normalized = signal / normalizer
            >>> # Scale down
            >>> halved = signal / 2.0
        """
        if isinstance(other, Input):
            return Composite(self, other, operator='/')
        else:
            return Composite(self, ConstantValue(other, self.duration), operator='/')

    def __rtruediv__(self, other):
        """Right division (for scalar / Input)."""
        if isinstance(other, Input):
            return Composite(other, self, operator='/')
        else:
            return Composite(ConstantValue(other, self.duration), self, operator='/')

    def __and__(self, other):
        """Concatenate two inputs in time (sequential composition).
        
        Parameters
        ----------
        other : Input
            The input to append after this one.
            
        Returns
        -------
        result : Sequential
            The concatenated inputs.
            
        Examples
        --------

        .. code-block:: python

            >>> baseline = Step([0], [0], 100 * u.ms)
            >>> stimulus = Step([1], [0], 200 * u.ms)
            >>> recovery = Step([0], [0], 100 * u.ms)
            >>> # Create sequential protocol
            >>> protocol = baseline & stimulus & recovery
            >>> # Total duration is 400 ms
        """
        if not isinstance(other, Input):
            raise TypeError("Can only concatenate with another Input object")
        return Sequential(self, other)

    def __or__(self, other):
        """Overlay two inputs (take maximum at each point).
        
        Parameters
        ----------
        other : Input
            The input to overlay with this one.
            
        Returns
        -------
        result : Composite
            The maximum of the two inputs at each time point.
            
        Examples
        --------

        .. code-block:: python

            >>> stim1 = Step([0, 1, 0], [0, 100, 300], 400 * u.ms)
            >>> stim2 = Step([0, 0.8, 0], [0, 150, 350], 400 * u.ms)
            >>> # Take maximum at each point
            >>> combined = stim1 | stim2
            >>> # Results in 1.0 from 100-150ms, 0.8 from 150-300ms, etc.
        """
        if isinstance(other, Input):
            return Composite(self, other, operator='max')
        else:
            raise TypeError("Can only overlay with another Input object")

    def __neg__(self):
        """Negate the input.
        
        Returns
        -------
        result : Composite
            The negated input.
            
        Examples
        --------

        .. code-block:: python

            >>> sine = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
            >>> # Invert the signal
            >>> inverted = -sine
        """
        return self.apply(lambda x: -x)

    def scale(self, factor: float):
        """Scale the input by a factor.
        
        Parameters
        ----------
        factor : float
            The scaling factor to apply.
            
        Returns
        -------
        scaled : Composite
            The scaled input (equivalent to self * factor).
            
        Examples
        --------

        .. code-block:: python

            >>> ramp = Ramp(0, 1, 100 * u.ms)
            >>> # Double the amplitude
            >>> doubled = ramp.scale(2.0)
            >>> # Reduce to 30%
            >>> reduced = ramp.scale(0.3)
        """
        return self * factor

    def shift(self, time_shift: Union[float, u.Quantity]):
        """Shift the input in time.
        
        Parameters
        ----------
        time_shift : float or Quantity
            The amount to shift. Positive values delay the signal
            (shift right), negative values advance it (shift left).
            
        Returns
        -------
        shifted : TimeShifted
            The time-shifted input.
            
        Examples
        --------

        .. code-block:: python

            >>> pulse = Step([1], [100 * u.ms], 200 * u.ms)
            >>> # Delay by 50ms (pulse now at 150ms)
            >>> delayed = pulse.shift(50 * u.ms)
            >>> # Advance by 30ms (pulse now at 70ms)
            >>> advanced = pulse.shift(-30 * u.ms)
        """
        return TimeShifted(self, time_shift)

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """Clip the input values to a range.
        
        Parameters
        ----------
        min_val : float, optional
            Minimum value. If None, no lower bound.
        max_val : float, optional
            Maximum value. If None, no upper bound.
            
        Returns
        -------
        clipped : Clipped
            The clipped input.
            
        Examples
        --------

        .. code-block:: python

            >>> ramp = Ramp(-2, 2, 100 * u.ms)
            >>> # Clip to [0, 1]
            >>> saturated = ramp.clip(0, 1)
            >>> # Only upper bound
            >>> capped = ramp.clip(max_val=1.5)
            >>> # Only lower bound (rectification)
            >>> rectified = ramp.clip(min_val=0)
        """
        return Clipped(self, min_val, max_val)

    def smooth(self, tau: Union[float, u.Quantity]):
        """Apply exponential smoothing to the input.
        
        Parameters
        ----------
        tau : float or Quantity
            The smoothing time constant. Larger values give
            more smoothing.
            
        Returns
        -------
        smoothed : Smoothed
            The exponentially smoothed input.
            
        Notes
        -----
        Implements exponential smoothing using:
        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
        where alpha = dt / tau.
            
        Examples
        --------

        .. code-block:: python

            >>> steps = Step([0, 1, 0.5, 1, 0],
            ...                   [0, 50, 100, 150, 200],
            ...                   250 * u.ms)
            >>> # Smooth transitions with 10ms time constant
            >>> smooth = steps.smooth(10 * u.ms)
            >>> # Heavy smoothing with 50ms time constant
            >>> very_smooth = steps.smooth(50 * u.ms)
        """
        return Smoothed(self, tau)

    def repeat(self, n_times: int):
        """Repeat the input pattern n times.
        
        Parameters
        ----------
        n_times : int
            Number of times to repeat the pattern.
            
        Returns
        -------
        repeated : Repeated
            The repeated input with duration n_times * original_duration.
            
        Examples
        --------

        .. code-block:: python

            >>> # Create a burst pattern
            >>> burst = Step([0, 1, 0], [0, 10, 20], 50 * u.ms)
            >>> # Repeat 10 times for 500ms total
            >>> burst_train = burst.repeat(10)
            >>>
            >>> # Repeated sine wave packets
            >>> packet = Sinusoidal(1.0, 50 * u.Hz, 100 * u.ms)
            >>> packets = packet.repeat(5)  # 500ms total
        """
        return Repeated(self, n_times)

    def apply(self, func: Callable):
        """Apply a custom function to the input.
        
        Parameters
        ----------
        func : callable
            Function to apply to the generated array.
            Should accept and return an array-like.
            
        Returns
        -------
        transformed : Transformed
            The transformed input.
            
        Examples
        --------

        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> sine = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
            >>>
            >>> # Rectification
            >>> rectified = sine.apply(lambda x: jnp.maximum(x, 0))
            >>>
            >>> # Squaring
            >>> squared = sine.apply(lambda x: x ** 2)
            >>>
            >>> # Custom nonlinearity
            >>> sigmoid = sine.apply(lambda x: 1 / (1 + jnp.exp(-5 * x)))
            >>>
            >>> # Add noise
            >>> import jax.random as jrandom
            >>> key = jrandom.PRNGKey(0)
            >>> noisy = sine.apply(
            ...     lambda x: x + 0.1 * jrandom.normal(key, x.shape)
            ... )
        """
        return Transformed(self, func)


class Composite(Input):
    """Composite input created by combining two inputs with an operator.
    
    Parameters
    ----------
    input1 : Input
        First input operand.
    input2 : Input  
        Second input operand.
    operator : str
        The operator to apply: '+', '-', '*', '/', 'max', 'min'.
        
    Notes
    -----
    When inputs have different durations, the shorter one is padded
    with zeros to match the longer duration.
    
    Division by zero returns the numerator value (avoiding NaN).
    
    Examples
    --------

    .. code-block:: python

        >>> # Direct construction (usually use operators instead)
        >>> ramp = Ramp(0, 1, 100 * u.ms)
        >>> sine = Sinusoidal(0.5, 10 * u.Hz, 100 * u.ms)
        >>> added = Composite(ramp, sine, '+')
        >>>
        >>> # More commonly created via operators
        >>> added = ramp + sine
        >>> multiplied = ramp * sine
        >>> maximum = ramp | sine  # Uses 'max' operator
    """
    __module__ = 'braintools.input'

    def __init__(self, input1: Input, input2: Input, operator: str):
        """Initialize a composite input.
        
        Parameters
        ----------
        input1 : Input
            First input.
        input2 : Input
            Second input.
        operator : str
            The operator to apply ('+', '-', '*', '/', 'max', 'min').
        """
        # Use the maximum duration of the two inputs
        duration = u.math.maximum(input1.duration, input2.duration)
        super().__init__(duration)
        self.input1 = input1
        self.input2 = input2
        self.operator = operator

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the composite input."""
        arr1 = self.input1()
        arr2 = self.input2()

        # Ensure arrays have the same length (pad with zeros if needed)
        max_len = max(len(arr1), len(arr2))
        if len(arr1) < max_len:
            padding = u.math.zeros(max_len - len(arr1), dtype=arr1.dtype, unit=u.get_unit(arr1))
            arr1 = u.math.concatenate([arr1, padding])
        if len(arr2) < max_len:
            padding = u.math.zeros(max_len - len(arr2), dtype=arr2.dtype, unit=u.get_unit(arr2))
            arr2 = u.math.concatenate([arr2, padding])

        # Apply the operator
        if self.operator == '+':
            return arr1 + arr2
        elif self.operator == '-':
            return arr1 - arr2
        elif self.operator == '*':
            return arr1 * arr2
        elif self.operator == '/':
            # Avoid division by zero
            return u.math.where(arr2 != 0, arr1 / arr2, arr1)
        elif self.operator == 'max':
            return u.math.maximum(arr1, arr2)
        elif self.operator == 'min':
            return u.math.minimum(arr1, arr2)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class ConstantValue(Input):
    """A constant value input.
    
    Parameters
    ----------
    value : float
        The constant value.
    duration : float or Quantity
        Duration for which to generate the constant.
        
    Examples
    --------

    .. code-block:: python

        >>> # Usually created implicitly
        >>> sine = Sinusoidal(1.0, 10 * u.Hz, 100 * u.ms)
        >>> with_offset = sine + 0.5  # Creates ConstantValue(0.5, 100*u.ms)
        >>>
        >>> # Direct construction
        >>> baseline = ConstantValue(0.1, 500 * u.ms)
    """
    __module__ = 'braintools.input'

    def __init__(self, value: float, duration: Union[float, u.Quantity]):
        super().__init__(duration)
        self.value = value

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate constant array."""
        return u.math.ones(self.n_steps, dtype=brainstate.environ.dftype()) * self.value


class Sequential(Input):
    """Sequential composition of two inputs.
    
    Concatenates two inputs in time, with the second input
    following immediately after the first.
    
    Parameters
    ----------
    *inputs : Any
        All sequential inputs.
        
    Examples
    --------

    .. code-block:: python

        >>> # Three-phase protocol
        >>> baseline = Step([0], [0], 500 * u.ms)
        >>> stimulus = Ramp(0, 1, 1000 * u.ms)
        >>> recovery = Step([0], [0], 500 * u.ms)
        >>>
        >>> # Chain using & operator
        >>> protocol = baseline & stimulus & recovery
        >>> # Total duration is 2000 ms
        >>>
        >>> # Direct construction
        >>> two_phase = Sequential(baseline, stimulus)
    """
    __module__ = 'braintools.input'

    def __init__(self, *inputs):
        # Total duration is sum of both durations
        duration = functools.reduce(u.math.add, [inp.duration for inp in inputs])
        super().__init__(duration)
        self.inputs = inputs

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the sequential input."""
        arrs = [inp() for inp in self.inputs]
        return u.math.concatenate(arrs)


class TimeShifted(Input):
    """Time-shifted version of an input.
    
    Shifts an input forward (delay) or backward (advance) in time.
    The total duration remains the same, with zero-padding added
    as needed.
    
    Parameters
    ----------
    input_obj : Input
        The input to shift.
    time_shift : float or Quantity
        Amount to shift. Positive = delay (shift right),
        negative = advance (shift left).
        
    Examples
    --------

    .. code-block:: python

        >>> pulse = Step([1], [200 * u.ms], 500 * u.ms)
        >>>
        >>> # Delay by 100ms (pulse now at 300ms)
        >>> delayed = TimeShifted(pulse, 100 * u.ms)
        >>>
        >>> # Advance by 50ms (pulse now at 150ms)
        >>> advanced = TimeShifted(pulse, -50 * u.ms)
        >>>
        >>> # Usually created via shift() method
        >>> delayed = pulse.shift(100 * u.ms)
    """
    __module__ = 'braintools.input'

    def __init__(self, input_obj: Input, time_shift: Union[float, u.Quantity]):
        """Initialize time-shifted input.
        
        Parameters
        ----------
        input_obj : Input
            The input to shift.
        time_shift : float or Quantity
            Amount to shift (positive = delay, negative = advance).
        """
        super().__init__(input_obj.duration)
        self.input_obj = input_obj
        self.time_shift = time_shift

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the shifted input."""
        arr = self.input_obj()
        shift_steps = int(self.time_shift / self.dt)

        if shift_steps > 0:
            # Delay: pad with zeros at the beginning
            padding = u.math.zeros(shift_steps, dtype=arr.dtype, unit=u.get_unit(arr))
            return u.math.concatenate([padding, arr[:-shift_steps]])
        elif shift_steps < 0:
            # Advance: pad with zeros at the end
            shift_steps = -shift_steps
            padding = u.math.zeros(shift_steps, dtype=arr.dtype, unit=u.get_unit(arr))
            return u.math.concatenate([arr[shift_steps:], padding])
        else:
            return arr


class Clipped(Input):
    """Clipped version of an input.
    
    Clips input values to stay within specified bounds,
    useful for modeling saturation or rectification.
    
    Parameters
    ----------
    input_obj : Input
        The input to clip.
    min_val : float, optional
        Minimum value. If None, no lower bound.
    max_val : float, optional
        Maximum value. If None, no upper bound.
        
    Examples
    --------

    .. code-block:: python

        >>> ramp = Ramp(-2, 2, 200 * u.ms)
        >>>
        >>> # Clip to [0, 1] range
        >>> saturated = Clipped(ramp, 0, 1)
        >>>
        >>> # Only lower bound (rectification)
        >>> rectified = Clipped(ramp, min_val=0)
        >>>
        >>> # Only upper bound (saturation)
        >>> capped = Clipped(ramp, max_val=1.5)
        >>>
        >>> # Usually created via clip() method
        >>> saturated = ramp.clip(0, 1)
    """
    __module__ = 'braintools.input'

    def __init__(self, input_obj: Input, min_val: Optional[float] = None,
                 max_val: Optional[float] = None):
        """Initialize clipped input.
        
        Parameters
        ----------
        input_obj : Input
            The input to clip.
        min_val : float, optional
            Minimum value.
        max_val : float, optional
            Maximum value.
        """
        super().__init__(input_obj.duration)
        self.input_obj = input_obj
        self.min_val = min_val
        self.max_val = max_val

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the clipped input."""
        arr = self.input_obj()
        if self.min_val is not None:
            arr = u.math.maximum(arr, self.min_val)
        if self.max_val is not None:
            arr = u.math.minimum(arr, self.max_val)
        return arr


class Smoothed(Input):
    """Exponentially smoothed version of an input.
    
    Applies exponential smoothing (low-pass filtering) to an input,
    useful for removing sharp transitions or high-frequency noise.
    
    Parameters
    ----------
    input_obj : Input
        The input to smooth.
    tau : float or Quantity
        Smoothing time constant. Larger values give more smoothing.
        
    Notes
    -----
    Implements exponential smoothing:
    y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    where alpha = dt / tau.
    
    The cutoff frequency is approximately 1 / (2 * pi * tau).
        
    Examples
    --------

    .. code-block:: python

        >>> # Sharp steps
        >>> steps = Step([0, 1, 0.5, 1, 0],
        ...                   [0, 50, 100, 150, 200],
        ...                   250 * u.ms)
        >>>
        >>> # Light smoothing (fast response)
        >>> light = Smoothed(steps, 5 * u.ms)
        >>>
        >>> # Heavy smoothing (slow response)
        >>> heavy = Smoothed(steps, 25 * u.ms)
        >>>
        >>> # Usually created via smooth() method
        >>> smooth = steps.smooth(10 * u.ms)
    """
    __module__ = 'braintools.input'

    def __init__(self, input_obj: Input, tau: Union[float, u.Quantity]):
        """Initialize smoothed input.
        
        Parameters
        ----------
        input_obj : Input
            The input to smooth.
        tau : float or Quantity
            Smoothing time constant.
        """
        super().__init__(input_obj.duration)
        self.input_obj = input_obj
        self.tau = tau

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the smoothed input."""
        arr, arr_unit = u.split_mantissa_unit(self.input_obj())
        alpha = self.dt / self.tau

        smoothed = np.zeros_like(arr)
        smoothed[0] = arr[0]
        for i in range(1, len(arr)):
            smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i - 1]
        return u.maybe_decimal(smoothed * arr_unit)


class Repeated(Input):
    """Repeated version of an input pattern.
    
    Repeats an input pattern multiple times, useful for
    creating periodic stimulation protocols.
    
    Parameters
    ----------
    input_obj : Input
        The input pattern to repeat.
    n_times : int
        Number of times to repeat.
        
    Notes
    -----
    The total duration is n_times * original_duration.
    
    Examples
    --------

    .. code-block:: python

        >>> # Single burst
        >>> burst = Step([0, 1, 0], [0, 10, 30], 50 * u.ms)
        >>>
        >>> # Burst train (10 bursts, 500ms total)
        >>> train = Repeated(burst, 10)
        >>>
        >>> # Oscillation packets
        >>> packet = Sinusoidal(1.0, 100 * u.Hz, 100 * u.ms)
        >>> packets = Repeated(packet, 5)  # 500ms total
        >>>
        >>> # Usually created via repeat() method
        >>> train = burst.repeat(10)
    """
    __module__ = 'braintools.input'

    def __init__(self, input_obj: Input, n_times: int):
        """Initialize repeated input.
        
        Parameters
        ----------
        input_obj : Input
            The input pattern to repeat.
        n_times : int
            Number of times to repeat.
        """
        # Total duration is n_times * original duration
        super().__init__(input_obj.duration * n_times)
        self.input_obj = input_obj
        self.n_times = n_times

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the repeated input."""
        arr = self.input_obj()
        return u.math.tile(arr, self.n_times)


class Transformed(Input):
    """Custom transformation applied to an input.
    
    Applies an arbitrary function to transform an input,
    enabling custom nonlinearities and processing.
    
    Parameters
    ----------
    input_obj : Input
        The input to transform.
    func : callable
        Function to apply to the array. Should accept
        and return an array-like object.
        
    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> sine = Sinusoidal(1.0, 10 * u.Hz, 200 * u.ms)
        >>>
        >>> # Half-wave rectification
        >>> rectified = Transformed(sine, lambda x: jnp.maximum(x, 0))
        >>>
        >>> # Squaring (frequency doubling)
        >>> squared = Transformed(sine, lambda x: x ** 2)
        >>>
        >>> # Sigmoid nonlinearity
        >>> sigmoid = Transformed(sine,
        ...     lambda x: 1 / (1 + jnp.exp(-10 * x)))
        >>>
        >>> # Usually created via apply() method
        >>> transformed = sine.apply(lambda x: jnp.abs(x))
    """
    __module__ = 'braintools.input'

    def __init__(self, input_obj: Input, func: Callable):
        """Initialize transformed input.
        
        Parameters
        ----------
        input_obj : Input
            The input to transform.
        func : callable
            Function to apply to the array.
        """
        super().__init__(input_obj.duration)
        self.input_obj = input_obj
        self.func = func

    def generate(self) -> brainstate.typing.ArrayLike:
        """Generate the transformed input."""
        arr = self.input_obj()
        return self.func(arr)


CompositeInput = create_deprecated_class(Composite, 'CompositeInput', 'Composite')
SequentialInput = create_deprecated_class(Sequential, 'SequentialInput', 'Sequential')
TimeShiftedInput = create_deprecated_class(TimeShifted, 'TimeShiftedInput', 'TimeShifted')
ClippedInput = create_deprecated_class(Clipped, 'ClippedInput', 'Clipped')
SmoothedInput = create_deprecated_class(Smoothed, 'SmoothedInput', 'Smoothed')
RepeatedInput = create_deprecated_class(Repeated, 'RepeatedInput', 'Repeated')
TransformedInput = create_deprecated_class(Transformed, 'TransformedInput', 'Transformed')

__all__.extend(
    ['CompositeInput', 'SequentialInput', 'TimeShiftedInput', 'ClippedInput', 'SmoothedInput', 'RepeatedInput',
     'TransformedInput'])
