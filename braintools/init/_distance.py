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
Distance profile classes for connectivity generation.

This module provides distance-dependent connectivity profiles that define how
connection probability and weight strength vary with spatial distance.
All classes inherit from the DistanceProfile base class.
"""

from abc import ABC, abstractmethod
from typing import Optional

import brainunit as u
import numpy as np

__all__ = [
    'DistanceProfile',
    'GaussianProfile',
    'ExponentialProfile',
    'PowerLawProfile',
    'LinearProfile',
    'StepProfile',
]


# =============================================================================
# Base Class
# =============================================================================


class DistanceProfile(ABC):
    """
    Base class for distance-dependent connectivity profiles.

    Distance profiles define how connection probability and weight strength vary with
    spatial distance between neurons. As a subclass of Initialization, DistanceProfile
    can be composed with other initialization strategies using arithmetic operations
    and functional composition.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import DistanceProfile
        >>>
        >>> class LinearDecayProfile(DistanceProfile):
        ...     def __init__(self, max_dist):
        ...         self.max_dist = max_dist
        ...
        ...     def probability(self, distances):
        ...         return np.maximum(0, 1 - distances / self.max_dist)
        ...
        ...     def weight_scaling(self, distances):
        ...         return self.probability(distances)
        ...
        ...     def __call__(self, rng, size, **kwargs):
        ...         # Use distance-based weights if distances are provided
        ...         if 'distances' in kwargs:
        ...             return self.weight_scaling(kwargs['distances'])
        ...         # Fallback to uniform random weights
        ...         return rng.random(size)

    Composition Examples
    --------------------
    .. code-block:: python

        >>> from braintools.init import GaussianProfile, Normal
        >>>
        >>> # Scale a Gaussian profile
        >>> profile = GaussianProfile(50.0 * u.um) * 0.5 * u.nS
        >>>
        >>> # Add noise to distance-based weights
        >>> noisy_profile = GaussianProfile(50.0 * u.um) + Normal(0, 0.1 * u.nS)
        >>>
        >>> # Clip profile values
        >>> clipped_profile = GaussianProfile(50.0 * u.um).clip(0.1, 0.9)
    """


    @abstractmethod
    def probability(self, distances: u.Quantity) -> np.ndarray:
        """
        Calculate connection probability based on distance.

        Parameters
        ----------
        distances : Quantity
            Array of distances between neuron pairs.

        Returns
        -------
        probability : ndarray
            Connection probabilities (values between 0 and 1).
        """
        pass

    @abstractmethod
    def weight_scaling(self, distances: u.Quantity) -> np.ndarray:
        """
        Calculate weight scaling factor based on distance.

        Parameters
        ----------
        distances : Quantity
            Array of distances between neuron pairs.

        Returns
        -------
        scaling : ndarray
            Weight scaling factors (typically between 0 and 1).
        """
        pass


# =============================================================================
# Distance Profiles
# =============================================================================

class GaussianProfile(DistanceProfile):
    """
    Gaussian distance profile.

    Connection probability and weight scaling follow a Gaussian (bell curve) profile.

    Parameters
    ----------
    sigma : Quantity
        Standard deviation of the Gaussian profile.
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import GaussianProfile
        >>>
        >>> profile = GaussianProfile(sigma=50.0 * u.um, max_distance=200.0 * u.um)
        >>> distances = np.array([0, 25, 50, 100, 200]) * u.um
        >>> probs = profile.probability(distances)
    """

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
    """
    Exponential distance profile.

    Connection probability and weight scaling decay exponentially with distance.

    Parameters
    ----------
    decay_constant : Quantity
        Distance constant for exponential decay.
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import ExponentialProfile
        >>>
        >>> profile = ExponentialProfile(
        ...     decay_constant=100.0 * u.um,
        ...     max_distance=500.0 * u.um
        ... )
        >>> distances = np.array([0, 50, 100, 200, 500]) * u.um
        >>> probs = profile.probability(distances)
    """

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
    """
    Power-law distance profile.

    Connection probability follows a power-law decay: p(d) = d^(-exponent).

    Parameters
    ----------
    exponent : float
        Power-law exponent (positive values cause decay with distance).
    min_distance : Quantity, optional
        Minimum distance to avoid division by zero (default: 1e-6).
    max_distance : Quantity, optional
        Maximum connection distance (connections beyond this are set to 0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import PowerLawProfile
        >>>
        >>> profile = PowerLawProfile(
        ...     exponent=2.0,
        ...     min_distance=1.0 * u.um,
        ...     max_distance=1000.0 * u.um
        ... )
        >>> distances = np.array([1, 10, 100, 1000]) * u.um
        >>> probs = profile.probability(distances)
    """

    def __init__(self, exponent: float, min_distance: Optional[u.Quantity] = None,
                 max_distance: Optional[u.Quantity] = None):
        self.exponent = exponent
        self.min_distance = min_distance
        self.max_distance = max_distance

    def probability(self, distances: u.Quantity) -> np.ndarray:
        dist_vals = distances.mantissa

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
    """
    Linear distance profile.

    Connection probability decreases linearly from 1 at distance 0 to 0 at max_distance.

    Parameters
    ----------
    max_distance : Quantity
        Maximum connection distance.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import LinearProfile
        >>>
        >>> profile = LinearProfile(max_distance=200.0 * u.um)
        >>> distances = np.array([0, 50, 100, 150, 200]) * u.um
        >>> probs = profile.probability(distances)
    """

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
    """
    Step function distance profile.

    Connection probability has two distinct values: one inside the threshold distance
    and another outside.

    Parameters
    ----------
    threshold : Quantity
        Distance threshold.
    inside_prob : float, optional
        Probability for distances <= threshold (default: 1.0).
    outside_prob : float, optional
        Probability for distances > threshold (default: 0.0).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.init import StepProfile
        >>>
        >>> profile = StepProfile(
        ...     threshold=100.0 * u.um,
        ...     inside_prob=0.8,
        ...     outside_prob=0.1
        ... )
        >>> distances = np.array([50, 100, 150]) * u.um
        >>> probs = profile.probability(distances)
    """

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
