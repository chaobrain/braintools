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


from abc import ABC, abstractmethod

import jax.numpy as jnp
from brainstate.typing import ArrayLike
from jax import Array

__all__ = [
    'Transform',
    'Identity',
]


class Transform(ABC):
    r"""
    Abstract base class for bijective parameter transformations.

    This class provides the interface for implementing bijective (one-to-one and onto)
    transformations that map parameters between different domains. These transformations
    are essential in optimization and statistical inference where parameters need to be
    constrained to specific domains (e.g., positive values, bounded intervals).

    A bijective transformation :math:`f: \mathcal{X} \rightarrow \mathcal{Y}` must satisfy:

    1. **Injectivity** (one-to-one): :math:`f(x_1) = f(x_2) \Rightarrow x_1 = x_2`
    2. **Surjectivity** (onto): :math:`\forall y \in \mathcal{Y}, \exists x \in \mathcal{X} : f(x) = y`
    3. **Invertibility**: :math:`f^{-1}(f(x)) = x` and :math:`f(f^{-1}(y)) = y`

    Methods
    -------
    forward(x)
        Apply the forward transformation :math:`y = f(x)`
    inverse(y)
        Apply the inverse transformation :math:`x = f^{-1}(y)`
    log_abs_det_jacobian(x, y)
        Compute the log absolute determinant of the Jacobian

    Notes
    -----
    Subclasses must implement both `forward` and `inverse` methods to ensure
    the transformation is truly bijective. The implementation should guarantee
    numerical stability and handle edge cases appropriately.

    Examples
    --------
    >>> class SquareTransform(Transform):
    ...     def forward(self, x):
    ...         return x**2
    ...     def inverse(self, y):
    ...         return jnp.sqrt(y)
    """
    __module__ = 'braintools.param'

    def __repr__(self) -> str:
        """Return a string representation of the transform."""
        return f"{self.__class__.__name__}()"

    def __call__(self, x: ArrayLike) -> Array:
        r"""
        Apply the forward transformation to the input.

        Parameters
        ----------
        x : array_like
            Input array to transform.

        Returns
        -------
        Array
            Transformed output array.

        Notes
        -----
        This method provides a convenient callable interface that delegates
        to the forward method, allowing Transform objects to be used as functions.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> Array:
        r"""
        Apply the forward transformation.

        Transforms input from the unconstrained domain to the constrained domain.
        This method implements the mathematical function :math:`y = f(x)` where
        :math:`x` is in the unconstrained space and :math:`y` is in the target domain.

        Parameters
        ----------
        x : array_like
            Input array in the unconstrained domain.

        Returns
        -------
        Array
            Transformed output in the constrained domain.

        Notes
        -----
        Implementations must ensure numerical stability and handle boundary
        conditions appropriately.
        """

    @abstractmethod
    def inverse(self, y: ArrayLike) -> Array:
        r"""
        Apply the inverse transformation.

        Transforms input from the constrained domain back to the unconstrained domain.
        This method implements the mathematical function :math:`x = f^{-1}(y)` where
        :math:`y` is in the constrained space and :math:`x` is in the unconstrained domain.

        Parameters
        ----------
        y : array_like
            Input array in the constrained domain.

        Returns
        -------
        Array
            Transformed output in the unconstrained domain.

        Notes
        -----
        Implementations must ensure that inverse(forward(x)) = x for all valid x,
        and forward(inverse(y)) = y for all y in the target domain.
        """
        pass

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Compute the log absolute determinant of the Jacobian of the forward transformation.

        For a bijective transformation :math:`f: \mathcal{X} \rightarrow \mathcal{Y}`,
        this computes:

        .. math::
            \log \left| \det \frac{\partial f(x)}{\partial x} \right|

        This is essential for computing probability densities under change of variables
        and is widely used in normalizing flows and variational inference.

        Parameters
        ----------
        x : array_like
            Input in the unconstrained domain.
        y : array_like
            Output in the constrained domain (i.e., y = forward(x)).
            This parameter is provided for efficiency since it may already
            be computed.

        Returns
        -------
        Array
            Log absolute determinant of the Jacobian.

        Notes
        -----
        The default implementation raises NotImplementedError. Subclasses
        should override this method to provide an efficient implementation.

        For element-wise transformations, the log determinant is simply
        the sum of log absolute derivatives:

        .. math::
            \log \left| \det J \right| = \sum_i \log \left| \frac{\partial f(x_i)}{\partial x_i} \right|
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement log_abs_det_jacobian. "
            "Override this method in your subclass."
        )


class Identity(Transform):
    """Identity transformation (no-op)."""
    __module__ = 'braintools.param'

    def forward(self, x: ArrayLike) -> Array:
        return x

    def inverse(self, y: ArrayLike) -> Array:
        return y

    def log_abs_det_jacobian(self, x: ArrayLike, y: ArrayLike) -> Array:
        """Log determinant is 0 for identity (det(I) = 1)."""
        return jnp.zeros(jnp.shape(x)[:-1] if jnp.ndim(x) > 0 else ())
