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
Weight initialization classes for connectivity generation.

This module provides weight initialization strategies for synaptic connections.
All classes inherit from the Initialization base class.
"""

from typing import Optional, Union

import brainunit as u
import numpy as np
from scipy.stats import truncnorm
from brainstate.typing import ArrayLike

from ._init import Initialization

__all__ = [
    'Initialization',
    'Constant',
    'Uniform',
    'Normal',
    'LogNormal',
    'Gamma',
    'Exponential',
    'ExponentialDecay',
    'TruncatedNormal',
    'Beta',
    'Weibull',
    'Mixture',
    'Conditional',
    'Scaled',
    'Clipped',
    'DistanceModulated',
    'DistanceProportional',
]


# =============================================================================
# Basic Weight Distributions
# =============================================================================

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
        >>> rng = np.random.default_rng(0)
        >>> weights = init(rng, 100)
    """

    def __init__(self, value: ArrayLike):
        self.value = value

    def __call__(self, rng, size, **kwargs):
        if isinstance(size, int):
            return u.math.full(size, self.value)
        return u.math.full(size, self.value)

    def __repr__(self):
        return f'Constant(value={self.value})'


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
        >>> weights = init(rng, 1000)
    """

    def __init__(self, low: ArrayLike, high: ArrayLike):
        self.low = low
        self.high = high

    def __call__(self, rng, size, **kwargs):
        low, unit = u.split_mantissa_unit(self.low)
        high = u.Quantity(self.high).to(unit).mantissa
        samples = rng.uniform(low, high, size)
        return u.maybe_decimal(samples * unit)

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
        >>> weights = init(rng, 1000)
    """

    def __init__(self, mean: ArrayLike, std: ArrayLike):
        self.mean = mean
        self.std = std

    def __call__(self, rng, size, **kwargs):
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa
        samples = rng.normal(mean, std, size)
        return u.maybe_decimal(samples * unit)

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
        >>> weights = init(rng, 1000)
    """

    def __init__(self, mean: ArrayLike, std: ArrayLike):
        self.mean = mean
        self.std = std

    def __call__(self, rng, size, **kwargs):
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa

        mu = np.log(mean ** 2 / np.sqrt(mean ** 2 + std ** 2))
        sigma = np.sqrt(np.log(1 + std ** 2 / mean ** 2))

        samples = rng.lognormal(mu, sigma, size)
        return u.maybe_decimal(samples * unit)

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
        >>> weights = init(rng, 1000)
    """

    def __init__(self, shape: float, scale: ArrayLike):
        self.shape = shape
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.gamma(self.shape, scale, size)
        return u.maybe_decimal(samples * unit)

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
        >>> weights = init(rng, 1000)
    """

    def __init__(self, scale: ArrayLike):
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.exponential(scale, size)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Exponential(scale={self.scale})'


class ExponentialDecay(Initialization):
    """
    Distance-dependent exponential decay weight distribution.

    Generates weights that decay exponentially with distance: w = max_weight * exp(-d / decay_constant).

    Parameters
    ----------
    max_weight : Quantity
        Maximum weight at zero distance.
    decay_constant : Quantity
        Distance constant for exponential decay.
    min_weight : Quantity, optional
        Minimum weight floor (default: 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import ExponentialDecay
        >>>
        >>> init = ExponentialDecay(
        ...     max_weight=1.0 * u.siemens,
        ...     decay_constant=100.0 * u.um,
        ...     min_weight=0.01 * u.siemens
        ... )
        >>> rng = np.random.default_rng(0)
        >>> distances = np.array([0, 50, 100, 200]) * u.um
        >>> weights = init(rng, 4, distances=distances)
    """

    def __init__(
        self,
        max_weight: ArrayLike,
        decay_constant: ArrayLike,
        min_weight: Optional[ArrayLike] = None
    ):
        self.max_weight = max_weight
        self.decay_constant = decay_constant
        self.min_weight = min_weight if min_weight is not None else 0.0 * max_weight.unit

    def __call__(self, rng, size, distances: Optional[ArrayLike] = None, **kwargs):
        if distances is None:
            return u.math.full(size, self.max_weight)

        max_val, weight_unit = u.split_mantissa_unit(self.max_weight)
        decay_val, dist_unit = u.split_mantissa_unit(self.decay_constant)
        min_val = u.Quantity(self.min_weight).to(weight_unit).mantissa

        dist_vals = distances.to(dist_unit).mantissa
        weights = max_val * np.exp(-dist_vals / decay_val)
        weights = np.maximum(weights, min_val)

        return u.maybe_decimal(weights * weight_unit)

    def __repr__(self):
        return f'ExponentialDecay(max_weight={self.max_weight}, decay_constant={self.decay_constant}, min_weight={self.min_weight})'


class TruncatedNormal(Initialization):
    """
    Truncated normal distribution initialization.

    Generates values from a normal distribution truncated to specified bounds.
    Requires scipy to be installed.

    Parameters
    ----------
    mean : Quantity
        Mean of the underlying normal distribution.
    std : Quantity
        Standard deviation of the underlying normal distribution.
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
        >>> weights = init(rng, 1000)
    """

    def __init__(
        self,
        mean: ArrayLike,
        std: ArrayLike,
        low: Optional[ArrayLike] = None,
        high: Optional[ArrayLike] = None
    ):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

    def __call__(self, rng, size, **kwargs):
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa

        a = -np.inf if self.low is None else (u.Quantity(self.low).to(unit).mantissa - mean) / std
        b = np.inf if self.high is None else (u.Quantity(self.high).to(unit).mantissa - mean) / std

        samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)
        return u.maybe_decimal(samples * unit)

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
        >>> weights = init(rng, 1000)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        low: ArrayLike,
        high: ArrayLike
    ):
        self.alpha = alpha
        self.beta = beta
        self.low = low
        self.high = high

    def __call__(self, rng, size, **kwargs):
        samples = rng.beta(self.alpha, self.beta, size)
        low, unit = u.split_mantissa_unit(self.low)
        high = u.Quantity(self.high).to(unit).mantissa
        return u.maybe_decimal((low + (high - low) * samples) * unit)

    def __repr__(self):
        return f'Beta(alpha={self.alpha}, beta={self.beta}, low={self.low}, high={self.high})'


class Weibull(Initialization):
    """
    Weibull distribution initialization.

    Generates values from a Weibull distribution.

    Parameters
    ----------
    shape : float
        Shape parameter (k) of the Weibull distribution.
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
        >>> weights = init(rng, 1000)
    """

    def __init__(self, shape: float, scale: ArrayLike):
        self.shape = shape
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.weibull(self.shape, size) * scale
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Weibull(shape={self.shape}, scale={self.scale})'


# =============================================================================
# Composite Weight Distributions
# =============================================================================

class Mixture(Initialization):
    """
    Mixture of multiple weight distributions.

    Randomly selects from multiple distributions for each connection according to specified weights.

    Parameters
    ----------
    distributions : list of Initialization
        List of initialization distributions to mix.
    weights : list of float, optional
        Probability weights for each distribution (must sum to 1).
        If None, uses equal weights.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Mixture, Normal, Uniform
        >>>
        >>> init = Mixture(
        ...     distributions=[
        ...         Normal(0.5 * u.siemens, 0.1 * u.siemens),
        ...         Uniform(0.8 * u.siemens, 1.2 * u.siemens)
        ...     ],
        ...     weights=[0.7, 0.3]
        ... )
        >>> rng = np.random.default_rng(0)
        >>> weights = init(rng, 1000)
    """

    def __init__(self, distributions: list, weights: Optional[list] = None):
        self.distributions = distributions
        self.weights = weights if weights is not None else [1.0 / len(distributions)] * len(distributions)

    def __call__(self, rng, size, **kwargs):
        choices = rng.choice(len(self.distributions), size=size, p=self.weights)

        if isinstance(size, int):
            samples = np.zeros(size)
            unit = None
        else:
            samples = np.zeros(size)
            unit = None

        for i, dist in enumerate(self.distributions):
            mask = (choices == i)
            if np.any(mask):
                dist_samples = dist(rng, np.sum(mask), **kwargs)
                if unit is None:
                    unit = dist_samples.unit
                samples[mask] = dist_samples.to(unit).mantissa

        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Mixture(distributions={self.distributions}, weights={self.weights})'


class Conditional(Initialization):
    """
    Conditional weight distribution based on neuron properties.

    Uses different distributions based on a condition function applied to neuron indices.

    Parameters
    ----------
    condition_fn : callable
        Function that takes neuron indices and returns boolean array.
    true_dist : Initialization
        Distribution to use when condition is True.
    false_dist : Initialization
        Distribution to use when condition is False.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Conditional, Constant, Normal
        >>>
        >>> def is_excitatory(indices):
        ...     return indices < 800
        >>>
        >>> init = Conditional(
        ...     condition_fn=is_excitatory,
        ...     true_dist=Normal(0.5 * u.siemens, 0.1 * u.siemens),
        ...     false_dist=Normal(-0.3 * u.siemens, 0.05 * u.siemens)
        ... )
        >>> rng = np.random.default_rng(0)
        >>> weights = init(rng, 1000, neuron_indices=np.arange(1000))
    """

    def __init__(
        self,
        condition_fn,
        true_dist: Initialization,
        false_dist: Initialization
    ):
        self.condition_fn = condition_fn
        self.true_dist = true_dist
        self.false_dist = false_dist

    def __call__(self, rng, size, neuron_indices: Optional[np.ndarray] = None, **kwargs):
        if neuron_indices is None:
            neuron_indices = np.arange(size if isinstance(size, int) else np.prod(size))

        conditions = self.condition_fn(neuron_indices)

        true_samples = self.true_dist(rng, np.sum(conditions), **kwargs)
        false_samples = self.false_dist(rng, np.sum(~conditions), **kwargs)

        if isinstance(size, int):
            samples = np.zeros(size)
        else:
            samples = np.zeros(size)

        unit = true_samples.unit
        samples[conditions] = true_samples.to(unit).mantissa
        samples[~conditions] = false_samples.to(unit).mantissa

        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Conditional(condition_fn={self.condition_fn}, true_dist={self.true_dist}, false_dist={self.false_dist})'


class Scaled(Initialization):
    """
    Scaled version of another distribution.

    Multiplies the output of another distribution by a constant factor.

    Parameters
    ----------
    base_dist : Initialization
        Base distribution to scale.
    scale_factor : float or Quantity
        Factor to multiply the base distribution by.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Scaled, Normal
        >>>
        >>> base = Normal(1.0 * u.siemens, 0.2 * u.siemens)
        >>> init = Scaled(base, scale_factor=0.5)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(rng, 1000)
    """

    def __init__(
        self,
        base_dist: Initialization,
        scale_factor: ArrayLike
    ):
        self.base_dist = base_dist
        self.scale_factor = scale_factor

    def __call__(self, rng, size, **kwargs):
        base_samples = self.base_dist(rng, size, **kwargs)
        return base_samples * self.scale_factor

    def __repr__(self):
        return f'Scaled(base_dist={self.base_dist}, scale_factor={self.scale_factor})'


class Clipped(Initialization):
    """
    Clipped version of another distribution.

    Clips the output of another distribution to specified minimum and maximum values.

    Parameters
    ----------
    base_dist : Initialization
        Base distribution to clip.
    min_val : Quantity, optional
        Minimum value (default: no lower bound).
    max_val : Quantity, optional
        Maximum value (default: no upper bound).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import Clipped, Normal
        >>>
        >>> base = Normal(0.5 * u.siemens, 0.3 * u.siemens)
        >>> init = Clipped(base, min_val=0.0 * u.siemens, max_val=1.0 * u.siemens)
        >>> rng = np.random.default_rng(0)
        >>> weights = init(rng, 1000)
    """

    def __init__(
        self,
        base_dist: Initialization,
        min_val: Optional[ArrayLike] = None,
        max_val: Optional[ArrayLike] = None
    ):
        self.base_dist = base_dist
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, rng, size, **kwargs):
        samples = self.base_dist(rng, size, **kwargs)

        if self.min_val is not None:
            min_val = u.Quantity(self.min_val).to(samples.unit).mantissa
            samples = u.math.maximum(samples, min_val * samples.unit)

        if self.max_val is not None:
            max_val = u.Quantity(self.max_val).to(samples.unit).mantissa
            samples = u.math.minimum(samples, max_val * samples.unit)

        return samples

    def __repr__(self):
        return f'Clipped(base_dist={self.base_dist}, min_val={self.min_val}, max_val={self.max_val})'


class DistanceModulated(Initialization):
    """
    Weight distribution modulated by distance.

    Generates weights from a base distribution and then modulates them based on
    distance using a specified function (e.g., exponential decay, gaussian).

    Parameters
    ----------
    base_dist : Initialization
        Base weight distribution.
    distance_profile : callable or str
        Distance modulation function. Can be:
        - 'exponential': exp(-d / sigma)
        - 'gaussian': exp(-d^2 / (2 * sigma^2))
        - 'linear': max(0, 1 - d / sigma)
        - callable: custom function f(distances, sigma)
    sigma : Quantity
        Characteristic distance scale.
    min_weight : Quantity, optional
        Minimum weight floor (default: 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import DistanceModulated, Normal
        >>>
        >>> init = DistanceModulated(
        ...     base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
        ...     distance_profile='exponential',
        ...     sigma=100.0 * u.um,
        ...     min_weight=0.01 * u.nS
        ... )
        >>>
        >>> rng = np.random.default_rng(0)
        >>> distances = np.linspace(0, 300, 100) * u.um
        >>> weights = init(rng, 100, distances=distances)
    """

    def __init__(
        self,
        base_dist: Initialization,
        distance_profile: Union[str, callable],
        sigma: ArrayLike,
        min_weight: Optional[ArrayLike] = None
    ):
        self.base_dist = base_dist
        self.sigma = sigma
        self.min_weight = min_weight

        if isinstance(distance_profile, str):
            if distance_profile == 'exponential':
                self.profile_func = lambda d, s: np.exp(-d / s)
            elif distance_profile == 'gaussian':
                self.profile_func = lambda d, s: np.exp(-d ** 2 / (2 * s ** 2))
            elif distance_profile == 'linear':
                self.profile_func = lambda d, s: np.maximum(0, 1 - d / s)
            else:
                raise ValueError(
                    f"Unknown distance profile: {distance_profile}. Use 'exponential', 'gaussian', 'linear', or a callable.")
        elif callable(distance_profile):
            self.profile_func = distance_profile
        else:
            raise TypeError("distance_profile must be a string or callable")

    def __call__(self, rng, size, distances: Optional[ArrayLike] = None, **kwargs):
        base_weights = self.base_dist(rng, size, **kwargs)

        if distances is None:
            return base_weights

        sigma_val, dist_unit = u.split_mantissa_unit(self.sigma)
        dist_vals = distances.to(dist_unit).mantissa

        modulation = self.profile_func(dist_vals, sigma_val)

        if isinstance(base_weights, u.Quantity):
            weight_vals, weight_unit = u.split_mantissa_unit(base_weights)
            modulated = weight_vals * modulation

            if self.min_weight is not None:
                min_val = u.Quantity(self.min_weight).to(weight_unit).mantissa
                modulated = np.maximum(modulated, min_val)

            return u.maybe_decimal(modulated * weight_unit)
        else:
            modulated = base_weights * modulation
            if self.min_weight is not None:
                modulated = np.maximum(modulated, self.min_weight)
            return modulated

    def __repr__(self):
        return f'DistanceModulated(base_dist={self.base_dist}, sigma={self.sigma}, min_weight={self.min_weight})'


class DistanceProportional(Initialization):
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

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import DistanceProportional
        >>>
        >>> init = DistanceProportional(
        ...     base_delay=0.5 * u.ms,
        ...     velocity=1.0 * u.meter / u.second,
        ...     max_delay=10.0 * u.ms
        ... )
        >>>
        >>> rng = np.random.default_rng(0)
        >>> distances = np.array([0, 100, 500, 1000]) * u.um
        >>> delays = init(rng, 4, distances=distances)
    """

    def __init__(
        self,
        base_delay: ArrayLike,
        velocity: ArrayLike,
        max_delay: Optional[ArrayLike] = None
    ):
        self.base_delay = base_delay
        self.velocity = velocity
        self.max_delay = max_delay

    def __call__(self, rng, size, distances: Optional[ArrayLike] = None, **kwargs):
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
        return f'DistanceProportional(base_delay={self.base_delay}, velocity={self.velocity}, max_delay={self.max_delay})'
