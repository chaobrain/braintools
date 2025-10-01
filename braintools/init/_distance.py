# Copyright 2025 BrainSim Ecosystem Limited. All Rights Reserved.
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
    'DistanceProfile',
    'GaussianProfile',
    'ExponentialProfile',
    'PowerLawProfile',
    'LinearProfile',
    'StepProfile',
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

    @abstractmethod
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
        pass

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
# Distance Profiles
# =============================================================================

class GaussianProfile(DistanceProfile):
    """
    Gaussian distance profile.

    Connection probability and weight scaling follow a Gaussian (bell curve) profile.

    Parameters
    ----------
    sigma : Quantity
        Standard deviation of the Gaussian profile.
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import GaussianProfile
        >>>
        >>> profile = GaussianProfile(sigma=50.0 * u.um, max_distance=200.0 * u.um)
        >>> distances = np.array([0, 25, 50, 100, 200]) * u.um
        >>> probs = profile.probability(distances)
    """

    def __init__(
        self,
        sigma: ArrayLike,
        max_distance: Optional[ArrayLike] = None
    ):
        self.sigma = sigma
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        sigma, unit = u.split_mantissa_unit(self.sigma)
        dist_vals = distances.to(unit).mantissa

        prob = np.exp(-0.5 * (dist_vals / sigma) ** 2)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob[dist_vals > max_val] = 0.0

        return prob

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'GaussianProfile(sigma={self.sigma}, max_distance={self.max_distance})'


class ExponentialProfile(DistanceProfile):
    """
    Exponential distance profile.

    Connection probability and weight scaling decay exponentially with distance.

    Parameters
    ----------
    decay_constant : Quantity
        Distance constant for exponential decay.
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import ExponentialProfile
        >>>
        >>> profile = ExponentialProfile(
        ...     decay_constant=100.0 * u.um,
        ...     max_distance=500.0 * u.um
        ... )
        >>> distances = np.array([0, 50, 100, 200, 500]) * u.um
        >>> probs = profile.probability(distances)
    """

    def __init__(
        self,
        decay_constant: ArrayLike,
        max_distance: Optional[ArrayLike] = None,
    ):
        self.decay_constant = decay_constant
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        decay, unit = u.split_mantissa_unit(self.decay_constant)
        dist_vals = distances.to(unit).mantissa

        prob = np.exp(-dist_vals / decay)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob[dist_vals > max_val] = 0.0

        return prob

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'ExponentialProfile(decay_constant={self.decay_constant}, max_distance={self.max_distance})'


class PowerLawProfile(DistanceProfile):
    """
    Power-law distance profile.

    Connection probability follows a power-law decay: p(d) = d^(-exponent).

    Parameters
    ----------
    exponent : float
        Power-law exponent (positive values cause decay with distance).
    min_distance : Quantity, optional
        Minimum distance to avoid division by zero (default: 1e-6).
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import PowerLawProfile
        >>>
        >>> profile = PowerLawProfile(
        ...     exponent=2.0,
        ...     min_distance=1.0 * u.um,
        ...     max_distance=1000.0 * u.um
        ... )
        >>> distances = np.array([1, 10, 100, 1000]) * u.um
        >>> probs = profile.probability(distances)
    """

    def __init__(
        self,
        exponent: float,
        min_distance: Optional[ArrayLike] = None,
        max_distance: Optional[ArrayLike] = None
    ):
        self.exponent = exponent
        self.min_distance = min_distance
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        dist_vals = distances.mantissa

        min_val = 1e-6 if self.min_distance is None else u.Quantity(self.min_distance).to(distances.unit).mantissa
        dist_vals = np.maximum(dist_vals, min_val)

        prob = dist_vals ** (-self.exponent)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(distances.unit).mantissa
            prob[distances.mantissa > max_val] = 0.0

        return prob

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'PowerLawProfile(exponent={self.exponent}, min_distance={self.min_distance}, max_distance={self.max_distance})'


class LinearProfile(DistanceProfile):
    """
    Linear distance profile.

    Connection probability decreases linearly from 1 at distance 0 to 0 at max_distance.

    Parameters
    ----------
    max_distance : Quantity
        Maximum connection distance.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import LinearProfile
        >>>
        >>> profile = LinearProfile(max_distance=200.0 * u.um)
        >>> distances = np.array([0, 50, 100, 150, 200]) * u.um
        >>> probs = profile.probability(distances)
    """

    def __init__(self, max_distance: ArrayLike):
        self.max_distance = max_distance

    def probability(self, distances: ArrayLike) -> np.ndarray:
        max_val, unit = u.split_mantissa_unit(self.max_distance)
        dist_vals = distances.to(unit).mantissa

        prob = np.maximum(0, 1 - dist_vals / max_val)
        return prob

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'LinearProfile(max_distance={self.max_distance})'


class StepProfile(DistanceProfile):
    """
    Step function distance profile.

    Connection probability has two distinct values: one inside the threshold distance
    and another outside.

    Parameters
    ----------
    threshold : Quantity
        Distance threshold.
    inside_prob : float, optional
        Probability for distances <= threshold (default: 1.0).
    outside_prob : float, optional
        Probability for distances > threshold (default: 0.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import StepProfile
        >>>
        >>> profile = StepProfile(
        ...     threshold=100.0 * u.um,
        ...     inside_prob=0.8,
        ...     outside_prob=0.1
        ... )
        >>> distances = np.array([50, 100, 150]) * u.um
        >>> probs = profile.probability(distances)
    """

    def __init__(self,
                 threshold: ArrayLike,
                 inside_prob: float = 1.0,
                 outside_prob: float = 0.0):
        self.threshold = threshold
        self.inside_prob = inside_prob
        self.outside_prob = outside_prob

    def probability(self, distances: ArrayLike) -> np.ndarray:
        threshold, unit = u.split_mantissa_unit(self.threshold)
        dist_vals = distances.to(unit).mantissa

        prob = np.where(dist_vals <= threshold, self.inside_prob, self.outside_prob)
        return prob

    def weight_scaling(self, distances: ArrayLike) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'StepProfile(threshold={self.threshold}, inside_prob={self.inside_prob}, outside_prob={self.outside_prob})'


# =============================================================================
# Composition Classes
# =============================================================================

class ComposedProfile(DistanceProfile):
    """
    Binary operation composition of distance profiles.

    Allows composing two profiles using arithmetic operations.
    """

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

    def _evaluate(self, obj: Union[DistanceProfile, ArrayLike], distances: ArrayLike) -> np.ndarray:
        """Helper to evaluate a profile or return a constant."""
        if isinstance(obj, DistanceProfile):
            return obj.weight_scaling(distances)
        elif isinstance(obj, (float, int)):
            return obj
        elif isinstance(obj, (u.Quantity, u.Unit)):
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
