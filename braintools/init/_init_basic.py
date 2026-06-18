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
Basic weight initialization distributions.

This module provides fundamental weight initialization strategies including:
- Constant values
- Uniform distributions
- Normal/Gaussian distributions
- Log-normal distributions
- Gamma distributions
- Exponential distributions
- Truncated normal distributions
- Beta distributions
- Weibull distributions
"""


import warnings
from typing import Optional

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.typing import ArrayLike
from jax.scipy.special import ndtr, ndtri

from ._init_base import Initialization

__all__ = [
    'Constant',
    'ZeroInit',
    'Uniform',
    'Normal',
    'LogNormal',
    'Gamma',
    'Exponential',
    'TruncatedNormal',
    'Beta',
    'Weibull',
]


# =============================================================================
# Parameter validation helpers
# =============================================================================

def _check_non_negative(value: ArrayLike, name: str) -> None:
    """Raise if a (possibly unit-carrying) value is negative."""
    if np.any(np.asarray(u.get_mantissa(value)) < 0):
        raise ValueError(f'`{name}` must be non-negative, got {value!r}.')


def _check_positive(value: ArrayLike, name: str) -> None:
    """Raise if a (possibly unit-carrying) value is not strictly positive."""
    if np.any(np.asarray(u.get_mantissa(value)) <= 0):
        raise ValueError(f'`{name}` must be positive, got {value!r}.')


def _check_dimensionless(value: ArrayLike, name: str) -> None:
    """Raise if a shape-like parameter carries a physical unit."""
    if isinstance(value, u.Quantity) and u.get_unit(value) != u.UNITLESS:
        raise ValueError(f'`{name}` must be dimensionless, got {value!r}.')


def _check_low_high(low: ArrayLike, high: ArrayLike) -> None:
    """Raise if ``low`` is not strictly less than ``high`` (units allowed)."""
    lo, unit = u.split_mantissa_unit(low)
    hi = u.Quantity(high).to(unit).mantissa
    if np.any(np.asarray(lo) >= np.asarray(hi)):
        raise ValueError(
            f'`low` must be strictly less than `high`, got low={low!r}, high={high!r}.'
        )


def _resolve_deprecated_unit(unit: Optional[u.Unit], *values: ArrayLike) -> u.Unit:
    """Resolve the deprecated ``unit=`` argument into ``self.unit``.

    Returns :data:`brainunit.UNITLESS` when ``unit`` is None. Otherwise emits a
    ``DeprecationWarning`` and, if any supplied value already carries a physical
    unit, raises ``ValueError`` (combining the two would silently create a
    compound unit such as ``mS * nS``).
    """
    if unit is None:
        return u.UNITLESS
    warnings.warn(
        'The `unit` parameter is deprecated and will be removed in a future '
        'version. Specify units directly on the value arguments instead.',
        DeprecationWarning,
        stacklevel=3,
    )
    for v in values:
        if v is None:
            continue
        if isinstance(v, u.Quantity) and u.get_unit(v) != u.UNITLESS:
            raise ValueError(
                'The deprecated `unit` argument cannot be combined with a value '
                'that already carries a unit; this would produce a compound unit. '
                'Specify the unit directly on the value instead.'
            )
    return unit


class Constant(Initialization):
    """
    Constant value initialization.

    Returns the same value for all connections.

    Parameters
    ----------
    value : Quantity
        The constant value to use.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Constant
        >>>
        >>> init = Constant(0.5 * u.siemens)
        >>> weights = init(100)
    """
    __module__ = 'braintools.init'

    def __init__(self, value: ArrayLike, unit: u.Unit = None):
        if value is None:
            raise ValueError('`value` must not be None for Constant initialization.')
        self.value = value
        self.unit = _resolve_deprecated_unit(unit, value)

    def __call__(self, size, **kwargs):
        return u.maybe_decimal(u.math.full(size, self.value) * self.unit)

    def __repr__(self):
        return f'Constant(value={self.value})'


class ZeroInit(Constant):
    """
    Zero initialization.

    Special case of constant initialization that initializes all values to zero.
    Useful for initializing connections that start with no synaptic weight.

    Parameters
    ----------
    unit : Unit
        The unit of the zero values (default: unitless).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import ZeroInit
        >>>
        >>> # Create zero weights with siemens unit
        >>> init = ZeroInit(u.siemens)
        >>> weights = init(100)
        >>> assert np.all(weights == 0.0 * u.siemens)
        >>>
        >>> # Create unitless zero weights
        >>> init = ZeroInit()
        >>> weights = init((10, 20))
        >>> assert np.all(weights == 0.0)
    """
    __module__ = 'braintools.init'

    def __init__(self, unit: u.Unit = u.UNITLESS):
        # ZeroInit's `unit` is a first-class argument (not deprecated), so set the
        # attributes directly rather than routing through Constant's deprecated path.
        self.value = 0.0
        self.unit = u.UNITLESS if unit is None else unit

    def __repr__(self):
        return f'ZeroInit(unit={self.unit})'


class Uniform(Initialization):
    """
    Uniform distribution initialization.

    Generates values uniformly distributed between low and high.

    Parameters
    ----------
    low : Quantity
        Lower bound (inclusive).
    high : Quantity
        Upper bound (exclusive).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Uniform
        >>>
        >>> init = Uniform(0.1 * u.siemens, 1.0 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, low: ArrayLike, high: ArrayLike, unit: u.Unit = None):
        _check_low_high(low, high)
        self.low = low
        self.high = high
        self.unit = _resolve_deprecated_unit(unit, low, high)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        low, unit = u.split_mantissa_unit(self.low)
        high = u.Quantity(self.high).to(unit).mantissa
        samples = rng.uniform(low, high, size)
        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return f'Uniform(low={self.low}, high={self.high})'


class Normal(Initialization):
    """
    Normal (Gaussian) distribution initialization.

    Generates values from a normal distribution with specified mean and standard deviation.

    Parameters
    ----------
    mean : Quantity
        Mean of the distribution.
    std : Quantity
        Standard deviation of the distribution.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Normal
        >>>
        >>> init = Normal(0.5 * u.siemens, 0.1 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, mean: ArrayLike, std: ArrayLike, unit: u.Unit = None):
        _check_non_negative(std, 'std')
        self.mean = mean
        self.std = std
        self.unit = _resolve_deprecated_unit(unit, mean, std)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa
        samples = rng.normal(mean, std, size)
        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return f'Normal(mean={self.mean}, std={self.std})'


class LogNormal(Initialization):
    """
    Log-normal distribution initialization.

    Generates values from a log-normal distribution. The parameters are the desired
    mean and standard deviation in linear space (not log-space).

    Parameters
    ----------
    mean : Quantity
        Desired mean of the distribution (in linear space).
    std : Quantity
        Desired standard deviation of the distribution (in linear space).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import LogNormal
        >>>
        >>> init = LogNormal(0.5 * u.siemens, 0.2 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, mean: ArrayLike, std: ArrayLike, unit: u.Unit = None):
        # The linear-space parametrization requires a strictly positive mean
        # (log(mean) and division by mean**2 are otherwise undefined).
        _check_positive(mean, 'mean')
        _check_non_negative(std, 'std')
        self.mean = mean
        self.std = std
        self.unit = _resolve_deprecated_unit(unit, mean, std)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa

        mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + std ** 2))
        sigma = np.sqrt(np.log(1 + std ** 2 / mean ** 2))

        samples = rng.lognormal(mu, sigma, size)
        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return f'LogNormal(mean={self.mean}, std={self.std})'


class Gamma(Initialization):
    """
    Gamma distribution initialization.

    Generates values from a gamma distribution.

    Parameters
    ----------
    shape : float
        Shape parameter (k) of the gamma distribution.
    scale : Quantity
        Scale parameter (theta) of the gamma distribution.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Gamma
        >>>
        >>> init = Gamma(shape=2.0, scale=0.5 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, shape: float, scale: ArrayLike, unit: u.Unit = None):
        _check_dimensionless(shape, 'shape')
        _check_positive(shape, 'shape')
        self.shape = shape
        self.scale = scale
        self.unit = _resolve_deprecated_unit(unit, scale)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.gamma(self.shape, scale, size)
        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return f'Gamma(shape={self.shape}, scale={self.scale})'


class Exponential(Initialization):
    """
    Exponential distribution initialization.

    Generates values from an exponential distribution.

    Parameters
    ----------
    scale : Quantity
        Scale parameter (1/lambda) of the exponential distribution.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Exponential
        >>>
        >>> init = Exponential(0.5 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, scale: ArrayLike, unit: u.Unit = None):
        self.scale = scale
        self.unit = _resolve_deprecated_unit(unit, scale)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.exponential(scale, size)
        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return f'Exponential(scale={self.scale})'


class TruncatedNormal(Initialization):
    """
    Truncated normal distribution initialization.

    Generates values from a normal distribution truncated to specified bounds.

    Sampling uses the inverse-CDF (probability integral transform) method, so it
    is backend-agnostic: it works with the default ``brainstate.random`` backend
    as well as a NumPy :class:`numpy.random.Generator` passed via ``rng``.

    Parameters
    ----------
    mean : Quantity
        Mean of the underlying normal distribution.
    std : Quantity
        Standard deviation of the underlying normal distribution (must be >= 0).
    low : Quantity, optional
        Lower bound (default: -inf).
    high : Quantity, optional
        Upper bound (default: +inf).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import TruncatedNormal
        >>>
        >>> init = TruncatedNormal(
        ...     mean=0.5 * u.siemens,
        ...     std=0.2 * u.siemens,
        ...     low=0.0 * u.siemens,
        ...     high=1.0 * u.siemens
        ... )
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        mean: ArrayLike,
        std: ArrayLike,
        low: Optional[ArrayLike] = None,
        high: Optional[ArrayLike] = None,
        unit: u.Unit = None,
    ):
        _check_non_negative(std, 'std')
        if low is not None and high is not None:
            _check_low_high(low, high)
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
        self.unit = _resolve_deprecated_unit(unit, mean, std, low, high)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa

        lo = None if self.low is None else u.Quantity(self.low).to(unit).mantissa
        hi = None if self.high is None else u.Quantity(self.high).to(unit).mantissa

        if std == 0:
            # Degenerate distribution: all mass at the (clamped) mean.
            samples = jnp.full(size, mean, dtype=float)
        else:
            a = -jnp.inf if lo is None else (lo - mean) / std
            b = jnp.inf if hi is None else (hi - mean) / std
            # Inverse-CDF sampling: draw u ~ U(0, 1), map into the truncated CDF
            # interval [F(a), F(b)], then invert through the normal quantile.
            cdf_a = 0.0 if lo is None else ndtr(a)
            cdf_b = 1.0 if hi is None else ndtr(b)
            u_samples = rng.uniform(0.0, 1.0, size)
            p = cdf_a + u_samples * (cdf_b - cdf_a)
            # Keep the quantile finite (avoids +-inf at the open ends).
            p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
            samples = mean + std * ndtri(p)

        if lo is not None or hi is not None:
            samples = jnp.clip(
                samples,
                -jnp.inf if lo is None else lo,
                jnp.inf if hi is None else hi,
            )
        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return f'TruncatedNormal(mean={self.mean}, std={self.std}, low={self.low}, high={self.high})'


class Beta(Initialization):
    """
    Beta distribution initialization (rescaled to desired range).

    Generates values from a beta distribution and rescales them to [low, high].

    Parameters
    ----------
    alpha : float
        Alpha shape parameter (must be > 0).
    beta : float
        Beta shape parameter (must be > 0).
    low : Quantity
        Lower bound of the output range.
    high : Quantity
        Upper bound of the output range.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Beta
        >>>
        >>> init = Beta(alpha=2.0, beta=5.0, low=0.0 * u.siemens, high=1.0 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(
        self,
        alpha: float,
        beta: float,
        low: ArrayLike,
        high: ArrayLike,
        unit: u.Unit = None,
    ):
        _check_dimensionless(alpha, 'alpha')
        _check_positive(alpha, 'alpha')
        _check_dimensionless(beta, 'beta')
        _check_positive(beta, 'beta')
        _check_low_high(low, high)
        self.alpha = alpha
        self.beta = beta
        self.low = low
        self.high = high
        self.unit = _resolve_deprecated_unit(unit, low, high)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        samples = rng.beta(self.alpha, self.beta, size)
        low, unit = u.split_mantissa_unit(self.low)
        high = u.Quantity(self.high).to(unit).mantissa
        return u.maybe_decimal((low + (high - low) * samples) * unit * self.unit)

    def __repr__(self):
        return f'Beta(alpha={self.alpha}, beta={self.beta}, low={self.low}, high={self.high})'


class Weibull(Initialization):
    """
    Weibull distribution initialization.

    Generates values from a Weibull distribution.

    Parameters
    ----------
    shape : float
        Shape parameter (k) of the Weibull distribution (must be > 0).
    scale : Quantity
        Scale parameter (lambda) of the Weibull distribution.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Weibull
        >>>
        >>> init = Weibull(shape=1.5, scale=0.5 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(1000, rng=rng)
    """
    __module__ = 'braintools.init'

    def __init__(self, shape: float, scale: ArrayLike, unit: u.Unit = None):
        _check_dimensionless(shape, 'shape')
        _check_positive(shape, 'shape')
        self.shape = shape
        self.scale = scale
        self.unit = _resolve_deprecated_unit(unit, scale)

    def __call__(self, size, **kwargs):
        rng = kwargs.get('rng', brainstate.random)
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.weibull(self.shape, size) * scale
        return u.maybe_decimal(samples * unit * self.unit)

    def __repr__(self):
        return f'Weibull(shape={self.shape}, scale={self.scale})'
