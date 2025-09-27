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
Delay initialization classes for connectivity generation.

This module provides delay initialization strategies for synaptic connections.
All classes inherit from the DelayInit base class.
"""

from typing import Optional

import brainunit as u
import numpy as np

from ._init_base import Initialization

__all__ = [
    'DelayInit',
    'ConstantDelay',
    'UniformDelay',
    'NormalDelay',
    'GammaDelay',
]


# =============================================================================
# Base Class
# =============================================================================

class DelayInit(Initialization):
    """
    Base class for delay initialization strategies.

    All delay initialization classes should inherit from this base class.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import DelayInit

        class CustomDelay(DelayInit):
            def __init__(self, value):
                self.value = value

            def __call__(self, rng, size, **kwargs):
                return np.full(size, self.value)
    """
    pass


# =============================================================================
# Delay Distributions
# =============================================================================

class ConstantDelay(DelayInit):
    """
    Constant delay initialization.

    Returns the same delay value for all connections.

    Parameters
    ----------
    value : Quantity
        The constant delay value (with time units).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import ConstantDelay

        init = ConstantDelay(2.0 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 100)
    """

    def __init__(self, value: u.Quantity):
        self.value = value

    def __call__(self, rng, size, **kwargs):
        return u.math.full(size, self.value)

    def __repr__(self):
        return f'ConstantDelay(value={self.value})'


class UniformDelay(DelayInit):
    """
    Uniform delay distribution initialization.

    Generates delay values uniformly distributed between low and high.

    Parameters
    ----------
    low : Quantity
        Lower bound (inclusive).
    high : Quantity
        Upper bound (exclusive).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import UniformDelay

        init = UniformDelay(1.0 * u.ms, 5.0 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 1000)
    """

    def __init__(self, low: u.Quantity, high: u.Quantity):
        self.low = low
        self.high = high

    def __call__(self, rng, size, **kwargs):
        low, unit = u.split_mantissa_unit(self.low)
        high = u.Quantity(self.high).to(unit).mantissa
        samples = rng.uniform(low, high, size)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'UniformDelay(low={self.low}, high={self.high})'


class NormalDelay(DelayInit):
    """
    Normal delay distribution initialization.

    Generates delay values from a normal distribution, clipped to ensure non-negative values.

    Parameters
    ----------
    mean : Quantity
        Mean of the distribution.
    std : Quantity
        Standard deviation of the distribution.
    min_delay : Quantity, optional
        Minimum delay value (default: 0).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import NormalDelay

        init = NormalDelay(mean=2.0 * u.ms, std=0.5 * u.ms, min_delay=0.1 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 1000)
    """

    def __init__(self, mean: u.Quantity, std: u.Quantity,
                 min_delay: Optional[u.Quantity] = None):
        self.mean = mean
        self.std = std
        self.min_delay = min_delay if min_delay is not None else 0.0 * mean.unit

    def __call__(self, rng, size, **kwargs):
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa
        min_val = u.Quantity(self.min_delay).to(unit).mantissa

        samples = rng.normal(mean, std, size)
        samples = np.maximum(samples, min_val)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'NormalDelay(mean={self.mean}, std={self.std}, min_delay={self.min_delay})'


class GammaDelay(DelayInit):
    """
    Gamma delay distribution initialization.

    Generates delay values from a gamma distribution.

    Parameters
    ----------
    shape : float
        Shape parameter (k) of the gamma distribution.
    scale : Quantity
        Scale parameter (theta) of the gamma distribution.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import GammaDelay

        init = GammaDelay(shape=2.0, scale=1.0 * u.ms)
        rng = np.random.default_rng(0)
        delays = init(rng, 1000)
    """

    def __init__(self, shape: float, scale: u.Quantity):
        self.shape = shape
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.gamma(self.shape, scale, size)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'GammaDelay(shape={self.shape}, scale={self.scale})'
