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

from typing import Optional, Union

import brainunit as u
import numpy as np

from ._base import Initialization

__all__ = [
    'DelayInit',
    'ConstantDelay',
    'UniformDelay',
    'NormalDelay',
    'GammaDelay',
    'DistanceProportionalDelay',
    'DistanceModulatedDelay',
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


class DistanceProportionalDelay(DelayInit):
    """
    Distance-proportional delay initialization.

    Generates delays proportional to distance: delay = base_delay + distance / velocity.
    Models axonal conduction delays.

    Parameters
    ----------
    base_delay : Quantity
        Minimum delay at zero distance (synaptic delay).
    velocity : Quantity
        Conduction velocity (e.g., 0.5 m/s for unmyelinated, 5-10 m/s for myelinated).
    max_delay : Quantity, optional
        Maximum delay cap (default: no limit).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import DistanceProportionalDelay

        init = DistanceProportionalDelay(
            base_delay=0.5 * u.ms,
            velocity=1.0 * u.meter / u.second,
            max_delay=10.0 * u.ms
        )

        rng = np.random.default_rng(0)
        distances = np.array([0, 100, 500, 1000]) * u.um
        delays = init(rng, 4, distances=distances)
    """

    def __init__(
        self,
        base_delay: u.Quantity,
        velocity: u.Quantity,
        max_delay: Optional[u.Quantity] = None
    ):
        self.base_delay = base_delay
        self.velocity = velocity
        self.max_delay = max_delay

    def __call__(self, rng, size, distances: Optional[u.Quantity] = None, **kwargs):
        if distances is None:
            return u.math.full(size, self.base_delay)

        base_val, time_unit = u.split_mantissa_unit(self.base_delay)

        velocity_in_units = self.velocity.to(distances.unit / time_unit)
        conduction_delays = distances / velocity_in_units

        total_delays = self.base_delay + conduction_delays

        if self.max_delay is not None:
            total_delays = u.math.minimum(total_delays, self.max_delay)

        return total_delays

    def __repr__(self):
        return f'DistanceProportionalDelay(base_delay={self.base_delay}, velocity={self.velocity}, max_delay={self.max_delay})'


class DistanceModulatedDelay(DelayInit):
    """
    Delay distribution modulated by distance.

    Generates delays from a base distribution and then adds distance-dependent
    component or modulates based on distance.

    Parameters
    ----------
    base_dist : DelayInit
        Base delay distribution (e.g., NormalDelay for synaptic variability).
    distance_factor : Quantity
        Factor for distance contribution (e.g., 0.01 ms/um).
    mode : str
        How to combine base and distance: 'add' or 'multiply'.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import brainunit as u
        from braintools.conn import DistanceModulatedDelay, NormalDelay

        init = DistanceModulatedDelay(
            base_dist=NormalDelay(1.0 * u.ms, 0.2 * u.ms),
            distance_factor=0.005 * u.ms / u.um,
            mode='add'
        )

        rng = np.random.default_rng(0)
        distances = np.linspace(0, 500, 100) * u.um
        delays = init(rng, 100, distances=distances)
    """

    def __init__(
        self,
        base_dist: DelayInit,
        distance_factor: u.Quantity,
        mode: str = 'add'
    ):
        self.base_dist = base_dist
        self.distance_factor = distance_factor
        if mode not in ['add', 'multiply']:
            raise ValueError(f"mode must be 'add' or 'multiply', got {mode}")
        self.mode = mode

    def __call__(self, rng, size, distances: Optional[u.Quantity] = None, **kwargs):
        base_delays = self.base_dist(rng, size, **kwargs)

        if distances is None:
            return base_delays

        if self.mode == 'add':
            distance_contribution = distances * self.distance_factor
            return base_delays + distance_contribution
        else:
            factor_val, factor_unit = u.split_mantissa_unit(self.distance_factor)
            dist_val = distances.to(factor_unit).mantissa
            modulation = 1.0 + dist_val * factor_val
            return base_delays * modulation

    def __repr__(self):
        return f'DistanceModulatedDelay(base_dist={self.base_dist}, distance_factor={self.distance_factor}, mode={self.mode})'
