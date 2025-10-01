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
Base classes and utilities for parameter initialization.

This module provides the foundational Initialization base class and helper
functions for all initialization strategies (weights, delays, distances).
"""

from abc import ABC, abstractmethod
from typing import Union, Optional

import brainunit as u
import jax
import numpy as np

__all__ = [
    'Initialization',
    'Initializer',
    'init_call',
    'Compose',
]


# =============================================================================
# Base Class
# =============================================================================

class Initialization(ABC):
    """
    Base class for all parameter initialization strategies.

    This abstract class defines the interface for initialization strategies used to generate
    connectivity parameters such as weights and delays. All initialization classes must
    implement the ``__call__`` method.

    Initialization objects support composition through arithmetic operations and functional
    composition, enabling the creation of complex initialization strategies from simple ones.

    Supported Operations:
        - Arithmetic: +, -, *, / (element-wise operations)
        - Composition: | (pipe operator for chaining transformations)
        - Transformations: .clip(), .add(), .multiply(), .apply()

    Examples
    --------
    Create a custom initialization class:

    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Initialization
        >>>
        >>> class CustomInit(Initialization):
        ...     def __init__(self, value):
        ...         self.value = value
        ...
        ...     def __call__(self, size, **kwargs):
        ...         return np.full(size, self.value)

    Compose initializations:

    .. code-block:: python

        >>> from braintools.init import Normal, Uniform
        >>>
        >>> weight_init = Normal(0.5 * u.nS, 0.1 * u.nS) * 2.0 + 0.1 * u.nS
        >>>
        >>> delay_init = Uniform(1.0 * u.ms, 3.0 * u.ms).clip(0.5 * u.ms, 5.0 * u.ms)
        >>>
        >>> combined = (Normal(1.0 * u.nS, 0.2 * u.nS) |
        ...             lambda x: x.clip(0, 2 * u.nS) |
        ...             lambda x: x * 0.5)
    """

    @abstractmethod
    def __call__(self, size, **kwargs):
        """
        Generate parameter values.

        Parameters
        ----------
        size : int or tuple
            Shape of the output array.
        **kwargs :
            Additional keyword arguments (e.g., rng, distances, neuron_indices).
            rng : numpy.random.Generator, optional
                Random number generator (default: np.random).

        Returns
        -------
        values : array_like
            Generated parameter values.
        """
        pass

    def __add__(self, other):
        """Add two initializations or add a scalar/quantity."""
        return AddInit(self, other)

    def __radd__(self, other):
        """Right addition."""
        return AddInit(other, self)

    def __sub__(self, other):
        """Subtract two initializations or subtract a scalar/quantity."""
        return SubInit(self, other)

    def __rsub__(self, other):
        """Right subtraction."""
        return SubInit(other, self)

    def __mul__(self, other):
        """Multiply two initializations or multiply by a scalar."""
        return MulInit(self, other)

    def __rmul__(self, other):
        """Right multiplication."""
        return MulInit(other, self)

    def __truediv__(self, other):
        """Divide two initializations or divide by a scalar."""
        return DivInit(self, other)

    def __rtruediv__(self, other):
        """Right division."""
        return DivInit(other, self)

    def __or__(self, other):
        """Pipe operator for functional composition."""
        return PipeInit(self, other)

    def clip(self, min_val=None, max_val=None):
        """Clip values to a specified range."""
        return ClipInit(self, min_val, max_val)

    def add(self, value):
        """Add a constant value."""
        return AddInit(self, value)

    def multiply(self, value):
        """Multiply by a constant value."""
        return MulInit(self, value)

    def apply(self, func):
        """Apply an arbitrary function to the output."""
        return ApplyInit(self, func)


# =============================================================================
# Type Aliases
# =============================================================================

Initializer = Union[Initialization, float, int, np.ndarray, jax.Array, u.Quantity]


# =============================================================================
# Helper Functions
# =============================================================================

def init_call(init: Optional[Initialization], n: int, **kwargs):
    """
    Helper function to call initialization functions.

    This utility function provides a unified interface for calling initialization strategies,
    whether they are Initialization objects, scalars, or arrays.

    Parameters
    ----------
    init : Initialization, float, int, array, or None
        The initialization strategy or value.
    n : int
        Number of connections or parameters to generate.
    **kwargs :
        Additional keyword arguments passed to the initialization.
        rng : numpy.random.Generator, optional
            Random number generator (default: np.random).

    Returns
    -------
    values : array_like or None
        Generated parameter values, or None if init is None.

    Raises
    ------
    ValueError
        If array size doesn't match the number of connections.
    TypeError
        If init is not a valid initialization type.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import init_call, Normal
        >>>
        >>> weights = init_call(Normal(0.5 * u.siemens, 0.1 * u.siemens), 100)
        >>>
        >>> # With custom rng
        >>> rng = np.random.default_rng(0)
        >>> weights = init_call(Normal(0.5 * u.siemens, 0.1 * u.siemens), 100, rng=rng)
        >>>
        >>> scalar_weights = init_call(0.5, 100)
    """
    if init is None:
        return None
    elif isinstance(init, Initialization):
        return init(n, **kwargs)
    elif isinstance(init, (float, int)):
        return init
    elif isinstance(init, (u.Quantity, np.ndarray, jax.Array)):
        if u.math.size(init) in [1, n]:
            return init
        else:
            raise ValueError('Quantity must be scalar or match number of connections')
    elif hasattr(init, '__array__'):
        return init
    else:
        raise TypeError(f"Initialization must be an Initialization class, scalar, or array. Got {type(init)}")


# =============================================================================
# Composition Classes (Internal)
# =============================================================================

class BinaryOpInit(Initialization):
    """Base class for binary operations on initializations."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def _get_value(self, obj, size, **kwargs):
        """Helper to extract value from Initialization or scalar."""
        if isinstance(obj, Initialization):
            return obj(size, **kwargs)
        elif isinstance(obj, (float, int)):
            return obj
        elif isinstance(obj, (u.Quantity, np.ndarray, jax.Array)):
            return obj
        else:
            raise TypeError(f"Operand must be Initialization, scalar, or array. Got {type(obj)}")


class AddInit(BinaryOpInit):
    """Addition of two initializations."""

    def __call__(self, size, **kwargs):
        left_val = self._get_value(self.left, size, **kwargs)
        right_val = self._get_value(self.right, size, **kwargs)
        return left_val + right_val

    def __repr__(self):
        return f"({self.left} + {self.right})"


class SubInit(BinaryOpInit):
    """Subtraction of two initializations."""

    def __call__(self, size, **kwargs):
        left_val = self._get_value(self.left, size, **kwargs)
        right_val = self._get_value(self.right, size, **kwargs)
        return left_val - right_val

    def __repr__(self):
        return f"({self.left} - {self.right})"


class MulInit(BinaryOpInit):
    """Multiplication of two initializations."""

    def __call__(self, size, **kwargs):
        left_val = self._get_value(self.left, size, **kwargs)
        right_val = self._get_value(self.right, size, **kwargs)
        return left_val * right_val

    def __repr__(self):
        return f"({self.left} * {self.right})"


class DivInit(BinaryOpInit):
    """Division of two initializations."""

    def __call__(self, size, **kwargs):
        left_val = self._get_value(self.left, size, **kwargs)
        right_val = self._get_value(self.right, size, **kwargs)
        return left_val / right_val

    def __repr__(self):
        return f"({self.left} / {self.right})"


class ClipInit(Initialization):
    """Clip values to a range."""

    def __init__(self, base, min_val, max_val):
        self.base = base
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, size, **kwargs):
        values = self.base(size, **kwargs)

        if self.min_val is not None:
            if isinstance(values, u.Quantity):
                min_val = u.Quantity(self.min_val).to(values.unit).mantissa * values.unit
                values = u.math.maximum(values, min_val)
            else:
                values = np.maximum(values, self.min_val)

        if self.max_val is not None:
            if isinstance(values, u.Quantity):
                max_val = u.Quantity(self.max_val).to(values.unit).mantissa * values.unit
                values = u.math.minimum(values, max_val)
            else:
                values = np.minimum(values, self.max_val)

        return values

    def __repr__(self):
        return f"{self.base}.clip({self.min_val}, {self.max_val})"


class ApplyInit(Initialization):
    """Apply arbitrary function to initialization output."""

    def __init__(self, base, func):
        self.base = base
        self.func = func

    def __call__(self, size, **kwargs):
        values = self.base(size, **kwargs)
        return self.func(values)

    def __repr__(self):
        return f"{self.base}.apply({self.func})"


class PipeInit(Initialization):
    """Pipe/compose two initializations or functions."""

    def __init__(self, base, func):
        self.base = base
        self.func = func

    def __call__(self, size, **kwargs):
        values = self.base(size, **kwargs)
        if isinstance(self.func, Initialization):
            return self.func(size, **kwargs)
        elif callable(self.func):
            return self.func(values)
        else:
            raise TypeError(f"Right operand of pipe must be callable or Initialization. Got {type(self.func)}")

    def __repr__(self):
        return f"({self.base} | {self.func})"


# =============================================================================
# Public Composition Functions
# =============================================================================

class Compose(Initialization):
    """
    Compose multiple initialization strategies.

    This class allows functional composition of multiple initializations,
    applying them in sequence.

    Parameters
    ----------
    *inits : Initialization or callable
        Sequence of initializations or functions to compose.
        Applied from left to right (first init is applied first).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Normal, Compose
        >>>
        >>> init = Compose(
        ...     Normal(1.0 * u.nS, 0.2 * u.nS),
        ...     lambda x: u.math.maximum(x, 0 * u.nS),
        ...     lambda x: x * 0.5
        ... )
        >>>
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """

    def __init__(self, *inits):
        if len(inits) == 0:
            raise ValueError("Compose requires at least one initialization")
        self.inits = inits

    def __call__(self, size, **kwargs):
        result = self.inits[0](size, **kwargs) if isinstance(self.inits[0], Initialization) else self.inits[0]
        for init in self.inits[1:]:
            if isinstance(init, Initialization):
                result = init(size if isinstance(result, (int, float)) else len(result), **kwargs)
            elif callable(init):
                result = init(result)
            else:
                raise TypeError(f"Each argument must be Initialization or callable. Got {type(init)}")
        return result

    def __repr__(self):
        return f"Compose({', '.join(map(str, self.inits))})"
