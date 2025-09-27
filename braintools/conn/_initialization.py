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
Parameter initialization classes for connectivity generation.

This module provides extensible parameter initialization classes for weights,
delays, and other connectivity parameters. These classes enable flexible
and composable parameter specification across all connectivity patterns.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Any, Dict

import jax
import numpy as np
import brainunit as u


__all__ = [
    # Base classes
    'Initialization', 'Initialization', 'Initialization',
    # Weight distributions
    'Constant', 'Uniform', 'Normal', 'LogNormal', 'Gamma', 'Exponential',
    'ExponentialDecay', 'TruncatedNormal', 'Beta', 'Weibull',
    # Delay distributions
    'ConstantDelay', 'UniformDelay', 'NormalDelay', 'GammaDelay',
    # Distance profiles
    'DistanceProfile', 'GaussianProfile', 'ExponentialProfile',
    'PowerLawProfile', 'LinearProfile', 'StepProfile',
    # Composite distributions
    'Mixture', 'Conditional', 'Scaled', 'Clipped'
]


# =============================================================================
# Base Classes
# =============================================================================

class Initialization(ABC):
    """Base class for all parameter initialization strategies."""

    @abstractmethod
    def __call__(self, rng, size, **kwargs):
        """Generate parameter values."""
        pass


Initializer = Union[Initialization, float, int, np.ndarray, jax.Array, u.Quantity]


# =============================================================================
# Weight Initialization Classes
# =============================================================================

class Constant(Initialization):
    """Constant weight initialization."""

    def __init__(self, value: u.Quantity):
        self.value = value

    def __call__(self, rng, size, **kwargs):
        if isinstance(size, int):
            return u.math.full(size, self.value)
        return u.math.full(size, self.value)

    def __repr__(self):
        return f'Constant(value={self.value})'


class Uniform(Initialization):
    """Uniform weight distribution."""

    def __init__(self, low: u.Quantity, high: u.Quantity):
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
    """Normal (Gaussian) weight distribution."""

    def __init__(self, mean: u.Quantity, std: u.Quantity):
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
    """Log-normal weight distribution."""

    def __init__(self, mean: u.Quantity, std: u.Quantity):
        self.mean = mean
        self.std = std

    def __call__(self, rng, size, **kwargs):
        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa

        # Convert to log-space parameters
        mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
        sigma = np.sqrt(np.log(1 + std**2 / mean**2))

        samples = rng.lognormal(mu, sigma, size)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'LogNormal(mean={self.mean}, std={self.std})'


class Gamma(Initialization):
    """Gamma weight distribution."""

    def __init__(self, shape: float, scale: u.Quantity):
        self.shape = shape
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.gamma(self.shape, scale, size)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Gamma(shape={self.shape}, scale={self.scale})'


class Exponential(Initialization):
    """Exponential weight distribution."""

    def __init__(self, scale: u.Quantity):
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.exponential(scale, size)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Exponential(scale={self.scale})'


class ExponentialDecay(Initialization):
    """Distance-dependent exponential decay weight distribution."""

    def __init__(self, max_weight: u.Quantity, decay_constant: u.Quantity,
                 min_weight: Optional[u.Quantity] = None):
        self.max_weight = max_weight
        self.decay_constant = decay_constant
        self.min_weight = min_weight if min_weight is not None else 0.0 * max_weight.unit

    def __call__(self, rng, size, distances: Optional[u.Quantity] = None, **kwargs):
        if distances is None:
            # Without distances, return max_weight
            return u.math.full(size, self.max_weight)

        # Apply exponential decay based on distance
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
    """Truncated normal weight distribution."""

    def __init__(self, mean: u.Quantity, std: u.Quantity,
                 low: Optional[u.Quantity] = None, high: Optional[u.Quantity] = None):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

    def __call__(self, rng, size, **kwargs):
        try:
            from scipy.stats import truncnorm
        except ImportError:
            raise ImportError("TruncatedNormal requires scipy. Install with: pip install scipy")

        mean, unit = u.split_mantissa_unit(self.mean)
        std = u.Quantity(self.std).to(unit).mantissa

        # Set bounds
        a = -np.inf if self.low is None else (u.Quantity(self.low).to(unit).mantissa - mean) / std
        b = np.inf if self.high is None else (u.Quantity(self.high).to(unit).mantissa - mean) / std

        samples = truncnorm.rvs(a, b, loc=mean, scale=std, size=size, random_state=rng)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'TruncatedNormal(mean={self.mean}, std={self.std}, low={self.low}, high={self.high})'


class Beta(Initialization):
    """Beta weight distribution (rescaled to desired range)."""

    def __init__(self, alpha: float, beta: float, low: u.Quantity, high: u.Quantity):
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
    """Weibull weight distribution."""

    def __init__(self, shape: float, scale: u.Quantity):
        self.shape = shape
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.weibull(self.shape, size) * scale
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'Weibull(shape={self.shape}, scale={self.scale})'


# =============================================================================
# Delay Initialization Classes
# =============================================================================

class ConstantDelay(Initialization):
    """Constant delay initialization."""

    def __init__(self, value: u.Quantity):
        self.value = value

    def __call__(self, rng, size, **kwargs):
        return u.math.full(size, self.value)

    def __repr__(self):
        return f'ConstantDelay(value={self.value})'


class UniformDelay(Initialization):
    """Uniform delay distribution."""

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


class NormalDelay(Initialization):
    """Normal delay distribution."""

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


class GammaDelay(Initialization):
    """Gamma delay distribution."""

    def __init__(self, shape: float, scale: u.Quantity):
        self.shape = shape
        self.scale = scale

    def __call__(self, rng, size, **kwargs):
        scale, unit = u.split_mantissa_unit(self.scale)
        samples = rng.gamma(self.shape, scale, size)
        return u.maybe_decimal(samples * unit)

    def __repr__(self):
        return f'GammaDelay(shape={self.shape}, scale={self.scale})'


# =============================================================================
# Distance Profiles
# =============================================================================

class DistanceProfile(ABC):
    """Base class for distance-dependent connectivity profiles."""

    @abstractmethod
    def probability(self, distances: u.Quantity) -> np.ndarray:
        """Calculate connection probability based on distance."""
        pass

    @abstractmethod
    def weight_scaling(self, distances: u.Quantity) -> np.ndarray:
        """Calculate weight scaling factor based on distance."""
        pass


class GaussianProfile(DistanceProfile):
    """Gaussian distance profile."""

    def __init__(self, sigma: u.Quantity, max_distance: Optional[u.Quantity] = None):
        self.sigma = sigma
        self.max_distance = max_distance

    def probability(self, distances: u.Quantity) -> np.ndarray:
        sigma, unit = u.split_mantissa_unit(self.sigma)
        dist_vals = distances.to(unit).mantissa

        prob = np.exp(-0.5 * (dist_vals / sigma) ** 2)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob[dist_vals > max_val] = 0.0

        return prob

    def weight_scaling(self, distances: u.Quantity) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'GaussianProfile(sigma={self.sigma}, max_distance={self.max_distance})'


class ExponentialProfile(DistanceProfile):
    """Exponential distance profile."""

    def __init__(
        self,
        decay_constant: u.Quantity,
        max_distance: Optional[u.Quantity] = None,
    ):
        self.decay_constant = decay_constant
        self.max_distance = max_distance

    def probability(self, distances: u.Quantity) -> np.ndarray:
        decay, unit = u.split_mantissa_unit(self.decay_constant)
        dist_vals = distances.to(unit).mantissa

        prob = np.exp(-dist_vals / decay)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(unit).mantissa
            prob[dist_vals > max_val] = 0.0

        return prob

    def weight_scaling(self, distances: u.Quantity) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'ExponentialProfile(decay_constant={self.decay_constant}, max_distance={self.max_distance})'


class PowerLawProfile(DistanceProfile):
    """Power-law distance profile."""

    def __init__(self, exponent: float, min_distance: Optional[u.Quantity] = None,
                 max_distance: Optional[u.Quantity] = None):
        self.exponent = exponent
        self.min_distance = min_distance
        self.max_distance = max_distance

    def probability(self, distances: u.Quantity) -> np.ndarray:
        dist_vals = distances.mantissa

        # Avoid division by zero
        min_val = 1e-6 if self.min_distance is None else u.Quantity(self.min_distance).to(distances.unit).mantissa
        dist_vals = np.maximum(dist_vals, min_val)

        prob = dist_vals ** (-self.exponent)

        if self.max_distance is not None:
            max_val = u.Quantity(self.max_distance).to(distances.unit).mantissa
            prob[distances.mantissa > max_val] = 0.0

        return prob

    def weight_scaling(self, distances: u.Quantity) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'PowerLawProfile(exponent={self.exponent}, min_distance={self.min_distance}, max_distance={self.max_distance})'


class LinearProfile(DistanceProfile):
    """Linear distance profile."""

    def __init__(self, max_distance: u.Quantity):
        self.max_distance = max_distance

    def probability(self, distances: u.Quantity) -> np.ndarray:
        max_val, unit = u.split_mantissa_unit(self.max_distance)
        dist_vals = distances.to(unit).mantissa

        prob = np.maximum(0, 1 - dist_vals / max_val)
        return prob

    def weight_scaling(self, distances: u.Quantity) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'LinearProfile(max_distance={self.max_distance})'


class StepProfile(DistanceProfile):
    """Step function distance profile."""

    def __init__(self, threshold: u.Quantity, inside_prob: float = 1.0, outside_prob: float = 0.0):
        self.threshold = threshold
        self.inside_prob = inside_prob
        self.outside_prob = outside_prob

    def probability(self, distances: u.Quantity) -> np.ndarray:
        threshold, unit = u.split_mantissa_unit(self.threshold)
        dist_vals = distances.to(unit).mantissa

        prob = np.where(dist_vals <= threshold, self.inside_prob, self.outside_prob)
        return prob

    def weight_scaling(self, distances: u.Quantity) -> np.ndarray:
        return self.probability(distances)

    def __repr__(self):
        return f'StepProfile(threshold={self.threshold}, inside_prob={self.inside_prob}, outside_prob={self.outside_prob})'


# =============================================================================
# Composite Distributions
# =============================================================================

class Mixture(Initialization):
    """Mixture of multiple weight distributions."""

    def __init__(self, distributions: list, weights: Optional[list] = None):
        self.distributions = distributions
        self.weights = weights if weights is not None else [1.0 / len(distributions)] * len(distributions)

    def __call__(self, rng, size, **kwargs):
        # Choose which distribution to sample from for each element
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
    """Conditional weight distribution based on neuron properties."""

    def __init__(self, condition_fn, true_dist: Initialization, false_dist: Initialization):
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
    """Scaled version of another distribution."""

    def __init__(self, base_dist: Initialization, scale_factor: Union[float, u.Quantity]):
        self.base_dist = base_dist
        self.scale_factor = scale_factor

    def __call__(self, rng, size, **kwargs):
        base_samples = self.base_dist(rng, size, **kwargs)
        return base_samples * self.scale_factor

    def __repr__(self):
        return f'Scaled(base_dist={self.base_dist}, scale_factor={self.scale_factor})'


class Clipped(Initialization):
    """Clipped version of another distribution."""

    def __init__(self, base_dist: Initialization, min_val: Optional[u.Quantity] = None,
                 max_val: Optional[u.Quantity] = None):
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
