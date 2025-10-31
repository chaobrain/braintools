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
Variance scaling weight initialization strategies.

This module provides variance scaling initialization methods commonly used in deep learning:
- Kaiming/He initialization (KaimingUniform, KaimingNormal)
- Xavier/Glorot initialization (XavierUniform, XavierNormal)
- LeCun initialization (LecunUniform, LecunNormal)

These methods scale the initial weights based on the number of input/output units
to maintain stable gradients during training.
"""
from typing import Literal

import brainstate
import numpy as np
import brainunit as u
import jax.numpy as jnp
from brainstate.typing import ArrayLike

from ._init_base import Initialization

__all__ = [
    'VarianceScaling',
    'KaimingUniform',
    'KaimingNormal',
    'XavierUniform',
    'XavierNormal',
    'LecunUniform',
    'LecunNormal',
]


class VarianceScaling(Initialization):
    """
    Base class for variance scaling initializations.

    Variance scaling methods compute an appropriate scale factor based on the
    number of input and/or output units, then sample from a distribution with
    that scale.

    Parameters
    ----------
    scale : float
        Scaling factor (positive float).
    mode : {'fan_in', 'fan_out', 'fan_avg'}
        Mode for computing the scale factor:
        - 'fan_in': scale by number of input units
        - 'fan_out': scale by number of output units
        - 'fan_avg': scale by average of input and output units
    distribution : {'uniform', 'normal', 'truncated_normal'}
        Distribution to sample from.
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        scale: ArrayLike = 1.0,
        mode: Literal['fan_in', 'fan_out', 'fan_avg'] = 'fan_in',
        distribution: Literal['uniform', 'normal', 'truncated_normal'] = 'normal',
        unit: u.Unit = None,
    ):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.unit = u.UNITLESS if unit is None else unit

    def _compute_fans(self, shape):
        """Compute number of input and output units from shape."""
        if len(shape) < 1:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # For higher dimensional tensors (e.g., conv kernels)
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        return fan_in, fan_out

    def _get_fan(self, shape):
        """Get the fan value based on mode."""
        fan_in, fan_out = self._compute_fans(shape)
        if self.mode == 'fan_in':
            return fan_in
        elif self.mode == 'fan_out':
            return fan_out
        elif self.mode == 'fan_avg':
            return (fan_in + fan_out) / 2.0
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        shape = (size,) if isinstance(size, int) else size
        scale, unit = u.split_mantissa_unit(self.scale)
        fan = self._get_fan(shape)
        variance = scale / max(1.0, fan)

        if self.distribution == 'uniform':
            limit = jnp.sqrt(3.0 * variance)
            samples = rng.uniform(-limit, limit, shape)
        elif self.distribution == 'normal':
            stddev = jnp.sqrt(variance)
            samples = rng.normal(0.0, stddev, shape)
        elif self.distribution == 'truncated_normal':
            stddev = jnp.sqrt(variance)
            # Truncate at 2 standard deviations
            samples = rng.normal(0.0, stddev, shape)
            samples = jnp.clip(samples, -2 * stddev, 2 * stddev)
        else:
            raise ValueError(f"Invalid distribution: {self.distribution}")

        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'scale={self.scale}, '
                f'mode={self.mode}, '
                f'distribution={self.distribution})')


class KaimingUniform(VarianceScaling):
    """
    Kaiming/He uniform initialization.

    Samples from a uniform distribution with bounds computed to maintain
    variance across layers. Recommended for ReLU and leaky ReLU activations.

    Reference: He et al., "Delving Deep into Rectifiers: Surpassing Human-Level
    Performance on ImageNet Classification", ICCV 2015.

    Parameters
    ----------
    mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
        Mode for computing scale factor (default: 'fan_in').
    nonlinearity : {'relu', 'leaky_relu'}, optional
        Type of nonlinearity (default: 'relu').
        For leaky_relu, the scale is computed based on the negative slope.
    negative_slope : float, optional
        Negative slope for leaky_relu (default: 0.01).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import KaimingUniform
        >>>
        >>> init = KaimingUniform()
        >>> rng = np.random.default_rng(0)
        >>> weights = init((100, 50), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        scale: ArrayLike = None,
        mode: Literal['fan_in', 'fan_out', 'fan_avg'] = 'fan_in',
        nonlinearity: Literal['relu', 'leaky_relu'] = 'relu',
        negative_slope: float = 0.01,
        unit: u.Unit = None,
    ):
        # Compute scale based on nonlinearity
        if scale is None:
            if nonlinearity == 'relu':
                scale = jnp.sqrt(2.0)
            elif nonlinearity == 'leaky_relu':
                scale = jnp.sqrt(2.0 / (1 + negative_slope ** 2))
            else:
                raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        super().__init__(scale=scale, mode=mode, distribution='uniform', unit=unit)
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope

    def __repr__(self):
        return f'KaimingUniform(mode={self.mode}, nonlinearity={self.nonlinearity}, unit={self.unit})'


class KaimingNormal(VarianceScaling):
    """
    Kaiming/He normal initialization.

    Samples from a normal distribution with standard deviation computed to maintain
    variance across layers. Recommended for ReLU and leaky ReLU activations.

    Reference: He et al., "Delving Deep into Rectifiers: Surpassing Human-Level
    Performance on ImageNet Classification", ICCV 2015.

    Parameters
    ----------
    mode : {'fan_in', 'fan_out', 'fan_avg'}, optional
        Mode for computing scale factor (default: 'fan_in').
    nonlinearity : {'relu', 'leaky_relu'}, optional
        Type of nonlinearity (default: 'relu').
        For leaky_relu, the scale is computed based on the negative slope.
    negative_slope : float, optional
        Negative slope for leaky_relu (default: 0.01).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import KaimingNormal
        >>>
        >>> init = KaimingNormal()
        >>> rng = np.random.default_rng(0)
        >>> weights = init((100, 50), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        scale: ArrayLike = None,
        mode: Literal['fan_in', 'fan_out', 'fan_avg'] = 'fan_in',
        nonlinearity: Literal['relu', 'leaky_relu'] = 'relu',
        negative_slope: float = 0.01,
        unit: u.Unit = None,
    ):
        # Compute scale based on nonlinearity
        if scale is None:
            if nonlinearity == 'relu':
                scale = jnp.sqrt(2.0)
            elif nonlinearity == 'leaky_relu':
                scale = jnp.sqrt(2.0 / (1 + negative_slope ** 2))
            else:
                raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        super().__init__(scale=scale, mode=mode, distribution='normal', unit=unit)
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope

    def __repr__(self):
        return f'KaimingNormal(mode={self.mode}, nonlinearity={self.nonlinearity}, unit={self.unit})'


class XavierUniform(VarianceScaling):
    """
    Xavier/Glorot uniform initialization.

    Samples from a uniform distribution with bounds computed to maintain
    variance across layers. Recommended for tanh and sigmoid activations.

    Reference: Glorot & Bengio, "Understanding the difficulty of training deep
    feedforward neural networks", AISTATS 2010.

    Parameters
    ----------
    scale : float, optional
        Scaling factor (default: 1.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import XavierUniform
        >>>
        >>> init = XavierUniform()
        >>> rng = np.random.default_rng(0)
        >>> weights = init((100, 50), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: ArrayLike = 1.0, unit: u.Unit = None):
        super().__init__(scale=scale, mode='fan_avg', distribution='uniform', unit=unit)

    def __repr__(self):
        return f'XavierUniform(scale={self.scale}, unit={self.unit})'


class XavierNormal(VarianceScaling):
    """
    Xavier/Glorot normal initialization.

    Samples from a normal distribution with standard deviation computed to maintain
    variance across layers. Recommended for tanh and sigmoid activations.

    Reference: Glorot & Bengio, "Understanding the difficulty of training deep
    feedforward neural networks", AISTATS 2010.

    Parameters
    ----------
    scale : float, optional
        Scaling factor (default: 1.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import XavierNormal
        >>>
        >>> init = XavierNormal()
        >>> rng = np.random.default_rng(0)
        >>> weights = init((100, 50), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: ArrayLike = 1.0, unit: u.Unit = None):
        super().__init__(scale=scale, mode='fan_avg', distribution='normal', unit=unit)

    def __repr__(self):
        return f'XavierNormal(scale={self.scale}, unit={self.unit})'


class LecunUniform(VarianceScaling):
    """
    LeCun uniform initialization.

    Samples from a uniform distribution with bounds computed to maintain
    variance across layers. Similar to Xavier but uses fan_in only.
    Recommended for SELU activations.

    Reference: LeCun et al., "Efficient BackProp", 1998.

    Parameters
    ----------
    scale : float, optional
        Scaling factor (default: 1.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import LecunUniform
        >>>
        >>> init = LecunUniform()
        >>> rng = np.random.default_rng(0)
        >>> weights = init((100, 50), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: ArrayLike = 1.0, unit: u.Unit = None):
        super().__init__(scale=scale, mode='fan_in', distribution='uniform', unit=unit)

    def __repr__(self):
        return f'LecunUniform(scale={self.scale}, unit={self.unit})'


class LecunNormal(VarianceScaling):
    """
    LeCun normal initialization.

    Samples from a normal distribution with standard deviation computed to maintain
    variance across layers. Similar to Xavier but uses fan_in only.
    Recommended for SELU activations.

    Reference: LeCun et al., "Efficient BackProp", 1998.

    Parameters
    ----------
    scale : float, optional
        Scaling factor (default: 1.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.init import LecunNormal
        >>>
        >>> init = LecunNormal()
        >>> rng = np.random.default_rng(0)
        >>> weights = init((100, 50), rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: ArrayLike = 1.0, unit: u.Unit = None):
        super().__init__(scale=scale, mode='fan_in', distribution='normal', unit=unit)

    def __repr__(self):
        return f'LecunNormal(scale={self.scale}, unit={self.unit})'
