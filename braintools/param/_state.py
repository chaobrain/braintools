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
Array state containers with bijective transformations.

This module provides state container classes that wrap array values with
optional bijective transformations, enabling constrained optimization
by storing parameters in an unconstrained space internally.
"""

import brainstate
import brainunit as u

from ._transform import Transform, IdentityT

__all__ = [
    'ArrayHidden',
    'ArrayParam',
]


class ArrayHidden(brainstate.HiddenState, u.CustomArray):
    """
    Hidden state container with bijective transformation support.

    Wraps array values as hidden states (non-trainable) with optional bijective
    transformations. The internal value is stored in the unconstrained space,
    while the ``data`` property provides access to the constrained value.

    Parameters
    ----------
    value : array_like
        Initial value in the constrained space. Will be transformed to
        unconstrained space for internal storage.
    t : Transform, optional
        Bijective transformation to apply. Default is ``IdentityT()``.

    Attributes
    ----------
    value : array_like
        The internal unconstrained value (inherited from ``HiddenState``).
    t : Transform
        The bijective transformation.
    data : array_like
        The constrained value (``t.forward(value)``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from braintools.param import ArrayHidden, SoftplusT
    >>> # Create a hidden state constrained to be positive
    >>> hidden = ArrayHidden(jnp.array([1.0, 2.0]), t=SoftplusT(0.0))
    >>> hidden.data  # Returns positive values
    """
    __module__ = 'braintools.param'

    def __init__(
        self,
        value: brainstate.typing.ArrayLike,
        t: Transform = IdentityT()
    ):
        if not isinstance(value, brainstate.typing.ArrayLike):
            raise TypeError(f'value must be array-like, got {value}')
        value = t.inverse(value)
        super().__init__(value)
        self.t = t

    def __repr__(self) -> str:
        return f"ArrayHidden(data={self.data}, t={repr(self.t)})"

    @property
    def data(self):
        """
        Get the constrained value after applying the forward transformation.

        Returns
        -------
        array_like
            The value in the constrained space.
        """
        return self.t(self.value)

    @data.setter
    def data(self, v):
        """
        Set the constrained value by applying the inverse transformation.

        Parameters
        ----------
        v : array_like
            The new value in the constrained space.
        """
        self.value = self.t.inverse(v)


class ArrayParam(brainstate.ParamState, u.CustomArray):
    """
    Trainable parameter state container with bijective transformation support.

    Wraps array values as trainable parameters with optional bijective
    transformations. The internal value is stored in the unconstrained space,
    enabling gradient-based optimization, while the ``data`` property provides
    access to the constrained value.

    Parameters
    ----------
    value : array_like
        Initial value in the constrained space. Will be transformed to
        unconstrained space for internal storage.
    t : Transform, optional
        Bijective transformation to apply. Default is ``IdentityT()``.

    Attributes
    ----------
    value : array_like
        The internal unconstrained value (inherited from ``ParamState``).
    t : Transform
        The bijective transformation.
    data : array_like
        The constrained value (``t.forward(value)``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from braintools.param import ArrayParam, SigmoidT
    >>> # Create a trainable parameter constrained to [0, 1]
    >>> param = ArrayParam(jnp.array([0.5]), t=SigmoidT(0.0, 1.0))
    >>> param.data  # Returns value in [0, 1]

    Notes
    -----
    This class inherits from ``brainstate.ParamState``, making it automatically
    trackable for gradient computation and optimization.
    """
    __module__ = 'braintools.param'

    def __init__(
        self,
        value: brainstate.typing.ArrayLike,
        t: Transform = IdentityT()
    ):
        if not isinstance(value, brainstate.typing.ArrayLike):
            raise TypeError(f'value must be array-like, got {value}')
        value = t.inverse(value)
        super().__init__(value)
        self.t = t

    def __repr__(self) -> str:
        return f"ArrayParam(data={self.data}, t={repr(self.t)})"

    @property
    def data(self):
        """
        Get the constrained value after applying the forward transformation.

        Returns
        -------
        array_like
            The value in the constrained space.
        """
        return self.t(self.value)

    @data.setter
    def data(self, v):
        """
        Set the constrained value by applying the inverse transformation.

        Parameters
        ----------
        v : array_like
            The new value in the constrained space.
        """
        self.value = self.t.inverse(v)
