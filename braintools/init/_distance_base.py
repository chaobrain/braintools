# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
Distance profile classes for connectivity generation.

This module provides distance-dependent connectivity profiles that define how
connection probability and weight strength vary with spatial distance.
All classes inherit from the DistanceProfile base class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Callable

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike

__all__ = [
    # Base class
    'DistanceProfile',

    # Composition classes
    'ComposedProfile',
    'ClipProfile',
    'ApplyProfile',
    'PipeProfile',
]


# =============================================================================
# Base Class
# =============================================================================


class DistanceProfile(ABC):
    """
    Base class for distance-dependent connectivity profiles.

    Distance profiles define how connection probability and weight strength vary with
    spatial distance between neurons. DistanceProfile supports composition through
    arithmetic operations and functional transformations, enabling the creation of
    complex distance-dependent patterns from simple ones.

    Supported Operations
    --------------------

    - Arithmetic: +, -, *, / (element-wise operations with other profiles, scalars, or quantities)
    - Composition: | (pipe operator for chaining transformations)
    - Transformations: .clip(), .apply()

    Examples
    --------

    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import DistanceProfile
        >>>
        >>> class LinearDecayProfile(DistanceProfile):
        ...     def __init__(self, max_dist):
        ...         self.max_dist = max_dist
        ...
        ...     def probability(self, distances):
        ...         return np.maximum(0, 1 - distances / self.max_dist)
        ...
        ...     def weight_scaling(self, distances):
        ...         return self.probability(distances)

    Composition Examples
    --------------------

    .. code-block:: python

        >>> from braintools.init import GaussianProfile, ExponentialProfile
        >>>
        >>> # Scale a Gaussian profile
        >>> profile = GaussianProfile(50.0 * u.um) * 0.5
        >>>
        >>> # Combine two profiles
        >>> combined = GaussianProfile(50.0 * u.um) + ExponentialProfile(100.0 * u.um) * 0.3
        >>>
        >>> # Clip profile values
        >>> clipped_profile = GaussianProfile(50.0 * u.um).clip(0.1, 0.9)
        >>>
        >>> # Apply custom function
        >>> transformed = GaussianProfile(50.0 * u.um).apply(lambda x: x ** 2)
        >>>
        >>> # Chain transformations with pipe operator
        >>> chained = GaussianProfile(50.0 * u.um) | (lambda x: x * 2) | (lambda x: np.minimum(x, 1.0))
    """
    __module__ = 'braintools.init'

    @abstractmethod
    def probability(self, distances: ArrayLike) -> np.ndarray:
        """
        Calculate connection probability based on distance.

        Parameters
        ----------
        distances : Quantity
            Array of distances between neuron pairs.

        Returns
        -------
        probability : ndarray
            Connection probabilities (values between 0 and 1).
        """
        pass

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        """
        Calculate weight scaling factor based on distance.

        Parameters
        ----------
        distances : Quantity
            Array of distances between neuron pairs.

        Returns
        -------
        scaling : ndarray
            Weight scaling factors (typically between 0 and 1).
        """
        return self.probability(distances)

    def __call__(self, distances: ArrayLike) -> np.ndarray:
        """
        Call the profile's weight_scaling method.

        Parameters
        ----------
        distances : Quantity
            Array of distances between neuron pairs.

        Returns
        -------
        scaling : ndarray
            Weight scaling factors.
        """
        return self.weight_scaling(distances)

    def __add__(self, other: Union['DistanceProfile', ArrayLike]) -> 'ComposedProfile':
        """Add two profiles or add a scalar/quantity."""
        return ComposedProfile(self, other, lambda x, y: x + y, '+')

    def __radd__(self, other: ArrayLike) -> 'ComposedProfile':
        """Right addition."""
        return ComposedProfile(other, self, lambda x, y: x + y, '+')

    def __sub__(self, other: Union['DistanceProfile', ArrayLike]) -> 'ComposedProfile':
        """Subtract two profiles or subtract a scalar/quantity."""
        return ComposedProfile(self, other, lambda x, y: x - y, '-')

    def __rsub__(self, other: ArrayLike) -> 'ComposedProfile':
        """Right subtraction."""
        return ComposedProfile(other, self, lambda x, y: x - y, '-')

    def __mul__(self, other: Union['DistanceProfile', ArrayLike]) -> 'ComposedProfile':
        """Multiply two profiles or multiply by a scalar."""
        return ComposedProfile(self, other, lambda x, y: x * y, '*')

    def __rmul__(self, other: ArrayLike) -> 'ComposedProfile':
        """Right multiplication."""
        return ComposedProfile(other, self, lambda x, y: x * y, '*')

    def __truediv__(self, other: Union['DistanceProfile', ArrayLike]) -> 'ComposedProfile':
        """Divide two profiles or divide by a scalar."""
        return ComposedProfile(self, other, lambda x, y: x / y, '/')

    def __rtruediv__(self, other: ArrayLike) -> 'ComposedProfile':
        """Right division."""
        return ComposedProfile(other, self, lambda x, y: x / y, '/')

    def __or__(self, other: Union['DistanceProfile', Callable]) -> 'PipeProfile':
        """Pipe operator for functional composition."""
        return PipeProfile(self, other)

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'ClipProfile':
        """Clip values to a specified range."""
        return ClipProfile(self, min_val, max_val)

    def apply(self, func: Callable) -> 'ApplyProfile':
        """Apply an arbitrary function to the output."""
        return ApplyProfile(self, func)


# =============================================================================
# Composition Classes
# =============================================================================

class ComposedProfile(DistanceProfile):
    """
    Binary operation composition of distance profiles.

    Allows composing two profiles using arithmetic operations.
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        left: Union[DistanceProfile, ArrayLike],
        right: Union[DistanceProfile, ArrayLike],
        op: Callable,
        op_symbol: str
    ):
        self.left = left
        self.right = right
        self.op = op
        self.op_symbol = op_symbol

    def _evaluate(
        self,
        obj: Union[DistanceProfile, ArrayLike],
        distances: ArrayLike,
    ) -> np.ndarray:
        """Helper to evaluate a profile or return a constant."""
        if isinstance(obj, DistanceProfile):
            return obj.weight_scaling(distances)
        elif isinstance(obj, ArrayLike):
            return obj
        elif hasattr(obj, '__array__'):
            return obj
        else:
            raise TypeError(f"Operand must be DistanceProfile, scalar, or Quantity. Got {type(obj)}")

    def probability(self, distances: ArrayLike) -> np.ndarray:
        left_val = self._evaluate(self.left, distances)
        right_val = self._evaluate(self.right, distances)
        return self.op(left_val, right_val)

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f"({self.left} {self.op_symbol} {self.right})"


class ClipProfile(DistanceProfile):
    """
    Clip a distance profile's output to a specified range.
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        base: DistanceProfile,
        min_val: Optional[float],
        max_val: Optional[float]
    ):
        self.base = base
        self.min_val = min_val
        self.max_val = max_val

    def probability(self, distances: ArrayLike) -> np.ndarray:
        values = self.base.probability(distances)
        if self.min_val is not None:
            values = np.maximum(values, self.min_val)
        if self.max_val is not None:
            values = np.minimum(values, self.max_val)
        return values

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        values = self.base.weight_scaling(distances)
        if self.min_val is not None:
            values = np.maximum(values, self.min_val)
        if self.max_val is not None:
            values = np.minimum(values, self.max_val)
        return values

    def __repr__(self):
        return f"{self.base}.clip({self.min_val}, {self.max_val})"


class ApplyProfile(DistanceProfile):
    """
    Apply an arbitrary function to a distance profile's output.
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        base: DistanceProfile,
        func: Callable
    ):
        self.base = base
        self.func = func

    def probability(self, distances: ArrayLike) -> np.ndarray:
        values = self.base.probability(distances)
        return self.func(values)

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        values = self.base.weight_scaling(distances)
        return self.func(values)

    def __repr__(self):
        return f"{self.base}.apply({self.func})"


class PipeProfile(DistanceProfile):
    """
    Pipe/compose distance profiles or functions.
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        base: DistanceProfile,
        func: Union[DistanceProfile, Callable]
    ):
        self.base = base
        self.func = func

    def probability(self, distances: ArrayLike) -> np.ndarray:
        values = self.base.probability(distances)
        if isinstance(self.func, DistanceProfile):
            # For chaining profiles, apply the second profile to the same distances
            # and combine with the first profile's output
            return self.func.probability(distances)
        elif callable(self.func):
            return self.func(values)
        else:
            raise TypeError(f"Right operand must be DistanceProfile or callable. Got {type(self.func)}")

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        values = self.base.weight_scaling(distances)
        if isinstance(self.func, DistanceProfile):
            return self.func.weight_scaling(distances)
        elif callable(self.func):
            return self.func(values)
        else:
            raise TypeError(f"Right operand must be DistanceProfile or callable. Got {type(self.func)}")

    def __repr__(self):
        return f"({self.base} | {self.func})"
