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

    Examples
    --------
    Create a custom initialization class:

    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import Initialization

        class CustomInit(Initialization):
            def __init__(self, value):
                self.value = value

            def __call__(self, rng, size, **kwargs):
                return np.full(size, self.value)
    """

    @abstractmethod
    def __call__(self, rng, size, **kwargs):
        """
        Generate parameter values.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.
        size : int or tuple
            Shape of the output array.
        **kwargs : dict
            Additional keyword arguments (e.g., distances, neuron_indices).

        Returns
        -------
        values : array_like
            Generated parameter values.
        """
        pass


# =============================================================================
# Type Aliases
# =============================================================================

Initializer = Union[Initialization, float, int, np.ndarray, jax.Array, u.Quantity]


# =============================================================================
# Helper Functions
# =============================================================================

def init_call(init: Optional[Initialization], rng: np.random.Generator, n: int, **kwargs):
    """
    Helper function to call initialization functions.

    This utility function provides a unified interface for calling initialization strategies,
    whether they are Initialization objects, scalars, or arrays.

    Parameters
    ----------
    init : Initialization, float, int, array, or None
        The initialization strategy or value.
    rng : numpy.random.Generator
        Random number generator.
    n : int
        Number of connections or parameters to generate.
    **kwargs :
        Additional keyword arguments passed to the initialization.

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

        import numpy as np
        import brainunit as u
        from braintools.conn import init_call, NormalWeight

        rng = np.random.default_rng(0)

        weights = init_call(NormalWeight(0.5 * u.siemens, 0.1 * u.siemens), rng, 100)

        scalar_weights = init_call(0.5, rng, 100)
    """
    if init is None:
        return None
    elif isinstance(init, Initialization):
        return init(rng, n, **kwargs)
    elif isinstance(init, (float, int)):
        return init
    elif isinstance(init, (u.Quantity, np.ndarray)):
        if u.math.size(init) in [1, n]:
            return init
        else:
            raise ValueError('Quantity must be scalar or match number of connections')
    elif hasattr(init, '__array__'):
        return init
    else:
        raise TypeError(f"Initialization must be an Initialization class, scalar, or array. Got {type(init)}")
