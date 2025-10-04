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
Orthogonal weight initialization strategies.

This module provides orthogonal initialization methods:
- Orthogonal: Standard orthogonal matrix initialization
- DeltaOrthogonal: Orthogonal initialization for deep networks
- Identity: Identity matrix initialization
"""
import warnings

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from ._init_base import Initialization

__all__ = [
    'Orthogonal',
    'DeltaOrthogonal',
    'Identity',
]


class Orthogonal(Initialization):
    """
    Orthogonal matrix initialization.

    Generates a random orthogonal matrix using QR decomposition. For non-square
    matrices, generates an orthogonal matrix of the appropriate shape.

    This initialization helps preserve the norm of gradients during backpropagation
    and is particularly useful for recurrent networks.

    Reference: Saxe et al., "Exact solutions to the nonlinear dynamics of learning
    in deep linear neural networks", ICLR 2014.

    Parameters
    ----------
    scale : float, optional
        Multiplicative factor to apply to the orthogonal matrix (default: 1.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import Orthogonal
        >>>
        >>> init = Orthogonal(scale=1.0)
        >>> rng = np.random.default_rng(0)
        >>> weights = init((100, 100), rng=rng)
        >>>
        >>> # For recurrent networks, often use sqrt(2) scale
        >>> init_recurrent = Orthogonal(scale=np.sqrt(2))
        >>> weights = init_recurrent((50, 50), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: float = 1.0, unit: u.Unit = None):
        self.scale = scale
        if unit is not None:
            warnings.warn(
                'The `unit` parameter is deprecated and will be removed in future versions. '
                'Please handle units separately from initialization.', DeprecationWarning
            )
        else:
            unit = u.UNITLESS
        self.unit = unit

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        shape = (size,) if isinstance(size, int) else size

        if len(shape) < 2:
            raise ValueError("Orthogonal initialization requires at least 2D shape")

        # Flatten all dimensions except the last
        num_rows = np.prod(shape[:-1])
        num_cols = shape[-1]

        # Generate a random matrix
        flat_shape = (num_rows, num_cols)
        matrix = rng.normal(0, 1, flat_shape)

        # Compute QR decomposition
        if num_rows < num_cols:
            matrix = matrix.T
        q, r = jnp.linalg.qr(matrix)

        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = jnp.diag(r)
        q *= jnp.sign(d)

        if num_rows < num_cols:
            q = q.T

        q = q.reshape(shape)
        return u.maybe_decimal(self.scale * q * self.unit)

    def __repr__(self):
        return f'Orthogonal(scale={self.scale})'


class DeltaOrthogonal(Initialization):
    """
    Delta-orthogonal initialization for deep CNNs.

    Initializes convolution kernels to be delta functions in spatial dimensions
    combined with orthogonal initialization in the channel dimensions. This is
    particularly useful for very deep convolutional networks.

    Reference: Xiao et al., "Dynamical Isometry and a Mean Field Theory of CNNs:
    How to Train 10,000-Layer Vanilla Convolutional Neural Networks", ICML 2018.

    Parameters
    ----------
    scale : float, optional
        Multiplicative factor to apply to the orthogonal matrix (default: 1.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import DeltaOrthogonal
        >>>
        >>> # For a 3x3 convolutional kernel with 64 input and 128 output channels
        >>> init = DeltaOrthogonal(scale=1.0)
        >>> rng = np.random.default_rng(0)
        >>> weights = init((128, 64, 3, 3), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: float = 1.0, unit: u.Unit = None):
        self.scale = scale
        if unit is not None:
            warnings.warn(
                'The `unit` parameter is deprecated and will be removed in future versions. '
                'Please handle units separately from initialization.', DeprecationWarning
            )
        else:
            unit = u.UNITLESS
        self.unit = unit

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        shape = (size,) if isinstance(size, int) else size

        if len(shape) < 3:
            raise ValueError("DeltaOrthogonal requires at least 3D shape (out_channels, in_channels, ...)")

        out_channels = shape[0]
        in_channels = shape[1]
        receptive_field_shape = shape[2:]

        # Initialize with zeros
        weights = jnp.zeros(shape)

        # Compute center index for each spatial dimension
        center = tuple(s // 2 for s in receptive_field_shape)

        # Create orthogonal matrix for the channel dimensions
        ortho_matrix = Orthogonal(scale=self.scale)((out_channels, in_channels), rng=rng)

        # Place the orthogonal matrix at the center of the receptive field
        weights = weights.at[(slice(None), slice(None)) + center].set(ortho_matrix)

        return u.maybe_decimal(weights * self.unit)

    def __repr__(self):
        return f'DeltaOrthogonal(scale={self.scale})'


class Identity(Initialization):
    """
    Identity matrix initialization.

    Initializes weights to an identity matrix, optionally scaled by a scale factor.
    For non-square matrices, creates a matrix that is as close to identity as possible.

    Parameters
    ----------
    scale : float, optional
        Multiplicative factor to apply to the identity matrix (default: 1.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import Identity
        >>>
        >>> init = Identity(scale=1.0)
        >>> weights = init((100, 100))
        >>>
        >>> # For rectangular matrices
        >>> weights = init((100, 50))  # Will create identity-like matrix
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: float = 1.0, unit: u.Unit = u.UNITLESS):
        self.scale = scale
        self.unit = unit

    def __call__(self, size, **kwargs):
        shape = (size,) if isinstance(size, int) else size

        if len(shape) == 1:
            # For 1D, create a vector of ones
            return u.maybe_decimal(self.scale * jnp.ones(shape) * self.unit)
        elif len(shape) == 2:
            # For 2D, create identity or identity-like matrix
            rows, cols = shape
            matrix = jnp.eye(rows, cols)
            return u.maybe_decimal(self.scale * matrix * self.unit)
        else:
            # For higher dimensions, create identity in the last two dimensions
            matrix = jnp.zeros(shape)
            min_dim = min(shape[-2], shape[-1])
            for i in range(min_dim):
                matrix = matrix.at[Ellipsis, i, i].set(1.0)
            return u.maybe_decimal(self.scale * matrix * self.unit)

    def __repr__(self):
        return f'Identity(scale={self.scale}, unit={self.unit})'
