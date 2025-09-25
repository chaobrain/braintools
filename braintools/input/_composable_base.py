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
Base classes for composable input current construction.
"""

from typing import Optional, Union, Callable
import brainstate
import brainunit as u
import numpy as np

__all__ = [
    'Input',
    'CompositeInput',
]


class Input:
    """Base class for composable input currents.
    
    This class provides a composable API for building complex input currents
    by combining simpler components using operators like +, -, *, /, &, |.
    
    Examples
    --------
    >>> # Create and combine inputs
    >>> ramp = RampInput(0, 1, 500 * u.ms)
    >>> sine = SinusoidalInput(0.5, 10 * u.Hz, 500 * u.ms)
    >>> combined = ramp + sine  # Add two inputs
    >>> 
    >>> # Chain operations
    >>> complex_input = (ramp * 0.5) + sine.shift(100 * u.ms)
    >>> 
    >>> # Apply transformations
    >>> filtered = sine.low_pass(20 * u.Hz)
    >>> scaled = ramp.scale(2.0).clip(0, 1.5)
    """
    
    def __init__(self, duration: Union[float, u.Quantity], dt: Optional[Union[float, u.Quantity]] = None):
        """Initialize the Input base class.
        
        Parameters
        ----------
        duration : float or Quantity
            The total duration of the input.
        dt : float or Quantity, optional
            The numerical precision. If None, uses the global dt.
        """
        self.duration = duration
        self.dt = brainstate.environ.get_dt() if dt is None else dt
        self._cached_array = None
    
    def __call__(self, recompute: bool = False) -> brainstate.typing.ArrayLike:
        """Generate and return the input current array.
        
        Parameters
        ----------
        recompute : bool
            If True, force recomputation even if cached.
            
        Returns
        -------
        current : array
            The generated input current.
        """
        if self._cached_array is None or recompute:
            self._cached_array = self._generate()
        return self._cached_array
    
    def _generate(self) -> brainstate.typing.ArrayLike:
        """Generate the input current array. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _generate()")
    
    @property
    def shape(self):
        """Get the shape of the input array."""
        return self().shape
    
    @property
    def n_steps(self):
        """Get the number of time steps."""
        return int(np.ceil(self.duration / self.dt))
    
    def __add__(self, other):
        """Add two inputs or add a constant."""
        if isinstance(other, Input):
            return CompositeInput(self, other, operator='+')
        else:
            return CompositeInput(self, ConstantValue(other, self.duration, self.dt), operator='+')
    
    def __radd__(self, other):
        """Right addition."""
        return self.__add__(other)
    
    def __sub__(self, other):
        """Subtract two inputs or subtract a constant."""
        if isinstance(other, Input):
            return CompositeInput(self, other, operator='-')
        else:
            return CompositeInput(self, ConstantValue(other, self.duration, self.dt), operator='-')
    
    def __rsub__(self, other):
        """Right subtraction."""
        if isinstance(other, Input):
            return CompositeInput(other, self, operator='-')
        else:
            return CompositeInput(ConstantValue(other, self.duration, self.dt), self, operator='-')
    
    def __mul__(self, other):
        """Multiply two inputs or multiply by a constant."""
        if isinstance(other, Input):
            return CompositeInput(self, other, operator='*')
        else:
            return CompositeInput(self, ConstantValue(other, self.duration, self.dt), operator='*')
    
    def __rmul__(self, other):
        """Right multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """Divide two inputs or divide by a constant."""
        if isinstance(other, Input):
            return CompositeInput(self, other, operator='/')
        else:
            return CompositeInput(self, ConstantValue(other, self.duration, self.dt), operator='/')
    
    def __rtruediv__(self, other):
        """Right division."""
        if isinstance(other, Input):
            return CompositeInput(other, self, operator='/')
        else:
            return CompositeInput(ConstantValue(other, self.duration, self.dt), self, operator='/')
    
    def __and__(self, other):
        """Concatenate two inputs in time (sequential composition)."""
        if not isinstance(other, Input):
            raise TypeError("Can only concatenate with another Input object")
        return SequentialInput(self, other)
    
    def __or__(self, other):
        """Overlay two inputs (take maximum at each point)."""
        if isinstance(other, Input):
            return CompositeInput(self, other, operator='max')
        else:
            raise TypeError("Can only overlay with another Input object")
    
    def __neg__(self):
        """Negate the input."""
        return CompositeInput(ConstantValue(0, self.duration, self.dt), self, operator='-')
    
    def scale(self, factor: float):
        """Scale the input by a factor.
        
        Parameters
        ----------
        factor : float
            The scaling factor.
            
        Returns
        -------
        scaled : Input
            The scaled input.
        """
        return self * factor
    
    def shift(self, time_shift: Union[float, u.Quantity]):
        """Shift the input in time.
        
        Parameters
        ----------
        time_shift : float or Quantity
            The amount to shift (positive shifts right/delays).
            
        Returns
        -------
        shifted : Input
            The time-shifted input.
        """
        return TimeShiftedInput(self, time_shift)
    
    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """Clip the input values to a range.
        
        Parameters
        ----------
        min_val : float, optional
            Minimum value.
        max_val : float, optional
            Maximum value.
            
        Returns
        -------
        clipped : Input
            The clipped input.
        """
        return ClippedInput(self, min_val, max_val)
    
    def smooth(self, tau: Union[float, u.Quantity]):
        """Apply exponential smoothing to the input.
        
        Parameters
        ----------
        tau : float or Quantity
            The smoothing time constant.
            
        Returns
        -------
        smoothed : Input
            The smoothed input.
        """
        return SmoothedInput(self, tau)
    
    def repeat(self, n_times: int):
        """Repeat the input pattern n times.
        
        Parameters
        ----------
        n_times : int
            Number of times to repeat.
            
        Returns
        -------
        repeated : Input
            The repeated input.
        """
        return RepeatedInput(self, n_times)
    
    def apply(self, func: Callable):
        """Apply a custom function to the input.
        
        Parameters
        ----------
        func : callable
            Function to apply to the array.
            
        Returns
        -------
        transformed : Input
            The transformed input.
        """
        return TransformedInput(self, func)


class CompositeInput(Input):
    """Composite input created by combining two inputs with an operator."""
    
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
        duration = max(getattr(input1.duration, 'magnitude', input1.duration),
                      getattr(input2.duration, 'magnitude', input2.duration))
        if hasattr(input1.duration, 'unit'):
            duration = duration * input1.duration.unit
        elif hasattr(input2.duration, 'unit'):
            duration = duration * input2.duration.unit
            
        super().__init__(duration, input1.dt)
        self.input1 = input1
        self.input2 = input2
        self.operator = operator
    
    def _generate(self)-> brainstate.typing.ArrayLike:
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
    """A constant value input."""
    
    def __init__(self, value: float, duration: Union[float, u.Quantity], 
                 dt: Optional[Union[float, u.Quantity]] = None):
        super().__init__(duration, dt)
        self.value = value
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate constant array."""
        return u.math.ones(self.n_steps, dtype=brainstate.environ.dftype()) * self.value


class SequentialInput(Input):
    """Sequential composition of two inputs."""
    
    def __init__(self, input1: Input, input2: Input):
        """Initialize sequential input.
        
        Parameters
        ----------
        input1 : Input
            First input (comes first in time).
        input2 : Input
            Second input (comes after first).
        """
        # Total duration is sum of both durations
        duration1 = getattr(input1.duration, 'magnitude', input1.duration)
        duration2 = getattr(input2.duration, 'magnitude', input2.duration)
        
        if hasattr(input1.duration, 'unit'):
            total_duration = (duration1 + duration2) * input1.duration.unit
        elif hasattr(input2.duration, 'unit'):
            total_duration = (duration1 + duration2) * input2.duration.unit
        else:
            total_duration = duration1 + duration2
            
        super().__init__(total_duration, input1.dt)
        self.input1 = input1
        self.input2 = input2
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the sequential input."""
        arr1 = self.input1()
        arr2 = self.input2()
        return u.math.concatenate([arr1, arr2])


class TimeShiftedInput(Input):
    """Time-shifted version of an input."""
    
    def __init__(self, input_obj: Input, time_shift: Union[float, u.Quantity]):
        """Initialize time-shifted input.
        
        Parameters
        ----------
        input_obj : Input
            The input to shift.
        time_shift : float or Quantity
            Amount to shift (positive = delay, negative = advance).
        """
        super().__init__(input_obj.duration, input_obj.dt)
        self.input_obj = input_obj
        self.time_shift = time_shift
    
    def _generate(self)-> brainstate.typing.ArrayLike:
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


class ClippedInput(Input):
    """Clipped version of an input."""
    
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
        super().__init__(input_obj.duration, input_obj.dt)
        self.input_obj = input_obj
        self.min_val = min_val
        self.max_val = max_val
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the clipped input."""
        arr = self.input_obj()
        if self.min_val is not None:
            arr = u.math.maximum(arr, self.min_val)
        if self.max_val is not None:
            arr = u.math.minimum(arr, self.max_val)
        return arr


class SmoothedInput(Input):
    """Exponentially smoothed version of an input."""
    
    def __init__(self, input_obj: Input, tau: Union[float, u.Quantity]):
        """Initialize smoothed input.
        
        Parameters
        ----------
        input_obj : Input
            The input to smooth.
        tau : float or Quantity
            Smoothing time constant.
        """
        super().__init__(input_obj.duration, input_obj.dt)
        self.input_obj = input_obj
        self.tau = tau
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the smoothed input."""
        arr = self.input_obj()
        alpha = self.dt / self.tau
        
        smoothed = u.math.zeros_like(arr)
        smoothed = smoothed.at[0].set(arr[0])
        
        for i in range(1, len(arr)):
            smoothed = smoothed.at[i].set(alpha * arr[i] + (1 - alpha) * smoothed[i-1])
        
        return smoothed


class RepeatedInput(Input):
    """Repeated version of an input pattern."""
    
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
        orig_duration = getattr(input_obj.duration, 'magnitude', input_obj.duration)
        if hasattr(input_obj.duration, 'unit'):
            total_duration = (orig_duration * n_times) * input_obj.duration.unit
        else:
            total_duration = orig_duration * n_times
            
        super().__init__(total_duration, input_obj.dt)
        self.input_obj = input_obj
        self.n_times = n_times
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the repeated input."""
        arr = self.input_obj()
        return u.math.tile(arr, self.n_times)


class TransformedInput(Input):
    """Custom transformation applied to an input."""
    
    def __init__(self, input_obj: Input, func: Callable):
        """Initialize transformed input.
        
        Parameters
        ----------
        input_obj : Input
            The input to transform.
        func : callable
            Function to apply to the array.
        """
        super().__init__(input_obj.duration, input_obj.dt)
        self.input_obj = input_obj
        self.func = func
    
    def _generate(self)-> brainstate.typing.ArrayLike:
        """Generate the transformed input."""
        arr = self.input_obj()
        return self.func(arr)