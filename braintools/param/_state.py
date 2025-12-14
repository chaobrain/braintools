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


import brainstate
import brainunit as u

from ._base import Transform, Identity

__all__ = [
    'Param',
]


class Param(brainstate.ParamState, u.CustomArray):
    """
    Trainable parameter wrapper with bijective transform support.

    ``ArrayParam`` is a specialized parameter state that allows applying bijective
    transformations to parameter values. This is useful for constrained optimization
    where you want to optimize in an unconstrained space but use parameters in a
    constrained space (e.g., ensuring parameters remain positive or bounded).

    The transform operates in two directions:

    - **Forward transform**: Applied when accessing the parameter via the ``.data`` property.
      This transforms the internal unconstrained value to the constrained space.
    - **Inverse transform**: Applied when setting values or during initialization.
      This transforms constrained values to the internal unconstrained representation.

    This design ensures that the internal parameter storage (accessed via ``.value``)
    maintains values in the unconstrained space, making gradient-based optimization
    more stable and effective.

    Parameters
    ----------
    value : ArrayLike
        The initial parameter value in the constrained space. This will be transformed
        to the unconstrained space using ``transform.inverse()`` for internal storage.
    transform : Transform, optional
        A bijective transformation with both forward (``__call__``) and inverse
        (``inverse()``) methods. Defaults to ``Identity()`` which applies
        no transformation.

    Attributes
    ----------
    value : ArrayLike
        The internal unconstrained parameter value. This is what gets optimized during
        training and is stored in the unconstrained space.
    transform : Transform
        The bijective transformation object used to convert between constrained and
        unconstrained spaces.
    data : ArrayLike (property)
        The parameter value in the constrained space. When accessed, returns
        ``transform(value)``. When set, applies ``transform.inverse()`` and updates
        ``value``.

    Examples
    --------
    Using ArrayParam with identity transform (no transformation):

    >>> import braintools
    >>> import jax.numpy as jnp
    >>>
    >>> # Simple parameter without transformation
    >>> param = braintools.param.Param(jnp.array([1.0, 2.0, 3.0]))
    >>> print(param.data)  # Access constrained value
    [1. 2. 3.]
    >>> print(param.value)  # Access unconstrained value (same as data for identity)
    [1. 2. 3.]

    Using ArrayParam with custom transform for positive constraints:

    >>> class Exp(braintools.param.Transform):
    ...     def forward(self, value):
    ...         return jnp.exp(value)  # unconstrained -> constrained (positive)
    ...     def inverse(self, value):
    ...         return jnp.log(value)  # constrained (positive) -> unconstrained
    >>>
    >>> # Parameter that stays positive
    >>> positive_param = braintools.param.Param(jnp.array([1.0, 2.0]), transform=Exp())
    >>> print(positive_param.data)  # Positive values
    [1. 2.]
    >>> print(positive_param.value)  # Log space (unconstrained)
    [0. 0.6931472]
    >>>
    >>> # During optimization, gradients are computed w.r.t. the unconstrained value
    >>> # But when you access .data, you always get positive values

    Notes
    -----
    - The transform must be bijective (one-to-one and onto) to ensure unique mappings
      between constrained and unconstrained spaces.
    - Common use cases include: positive constraints (exp/log), bounded constraints
      (sigmoid/logit), or more complex domain constraints.
    - When using gradient-based optimization, gradients are computed with respect to
      the unconstrained ``.value``, which often leads to better optimization dynamics.

    """
    __module__ = 'braintools.param'

    def __init__(
        self,
        value: brainstate.typing.ArrayLike,
        transform: Transform = Identity()
    ):
        if not isinstance(value, brainstate.typing.ArrayLike):
            raise TypeError(f'value must be array-like, got {value}')
        value = transform.inverse(value)
        super().__init__(value)
        self.transform = transform

    def __repr__(self) -> str:
        return f"Param(data={self.data}, transform={repr(self.transform)})"

    @property
    def data(self):
        return self.transform(self.value)

    @data.setter
    def data(self, v):
        self.value = self.transform.inverse(v)
