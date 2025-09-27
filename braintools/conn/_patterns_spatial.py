# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
Spatial connectivity patterns for composable system.

Includes patterns that depend on spatial positions:
- Distance-dependent connectivity
- Gaussian connectivity
- Regular spatial patterns
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist
import brainunit as u

from ._composable_base import Connectivity, ConnectionResult


__all__ = [
    # Spatial patterns
    'DistanceDependent',
    'Gaussian',
    'Regular',
    'Exponential',
    'PowerLaw',
    'Ring',
    'Grid',
]


class DistanceDependent(Connectivity):
    """Distance-dependent connectivity with various decay functions and unit support.

    Parameters
    ----------
    sigma : float or Quantity
        Characteristic distance scale with proper units (e.g., µm, mm).
    decay : str
        Decay function ('gaussian', 'exponential', 'linear', 'power').
    max_prob : float
        Maximum connection probability at distance 0.
    exponent : float
        Exponent for power-law decay.
    weight_scale : float or Quantity, optional
        Scaling factor for weights with proper units (e.g., nS, pA).

    Examples
    --------
    Gaussian distance decay with units:

    .. code-block:: python

        >>> import brainunit as u
        >>> conn = DistanceDependent(sigma=100 * u.um, decay='gaussian',
        ...                         weight_scale=1.0 * u.nS)
        >>> # Positions with units
        >>> positions = np.random.uniform(0, 1000, (100, 2)) * u.um
        >>> result = conn(100, 100, positions, positions)
        >>> print(f"Weights: {result.weights}")

    Exponential decay with distance cutoff:

    .. code-block:: python

        >>> local_conn = DistanceDependent(sigma=50 * u.um, decay='exponential')
        >>> # Only connections within 200 µm
        >>> constrained = local_conn.filter_distance(max_dist=200 * u.um)

    AMPA-like conductance based on distance:

    .. code-block:: python

        >>> # Realistic AMPA conductance scaling
        >>> ampa_conn = DistanceDependent(
        ...     sigma=80 * u.um,
        ...     decay='exponential',
        ...     weight_scale=2.0 * u.nS,  # Peak conductance
        ...     max_prob=0.3
        ... )
    """

    __module__ = 'braintools.conn'

    def __init__(self,
                 sigma: Union[float, u.Quantity],
                 decay: str = 'gaussian',
                 max_prob: float = 1.0,
                 exponent: float = 2.0,
                 weight_scale: Union[float, u.Quantity] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)

        # Handle sigma with units
        if isinstance(sigma, u.Quantity):
            self.sigma = sigma
        else:
            # Default to micrometers if no units provided
            self.sigma = sigma * u.um

        # Handle weight scale with units
        if isinstance(weight_scale, u.Quantity):
            self.weight_scale = weight_scale
        else:
            self.weight_scale = weight_scale

        self.decay = decay
        self.max_prob = max_prob
        self.exponent = exponent

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[Union[np.ndarray, u.Quantity]] = None,
                  post_positions: Optional[Union[np.ndarray, u.Quantity]] = None) -> ConnectionResult:
        """Generate distance-dependent connectivity with unit support."""
        if pre_positions is None or post_positions is None:
            raise ValueError("Distance-dependent connectivity requires neuron positions")

        # Handle positions with units
        if isinstance(pre_positions, u.Quantity):
            pre_pos = pre_positions
            post_pos = post_positions
        else:
            # Default to micrometers if no units provided
            pre_pos = pre_positions * u.um
            post_pos = post_positions * u.um

        # Calculate pairwise distances with units
        pre_coords = u.get_magnitude(pre_pos)
        post_coords = u.get_magnitude(post_pos)
        distances_magnitude = cdist(pre_coords, post_coords)
        distances = distances_magnitude * pre_pos.unit

        # Convert sigma to same units as distances
        sigma_magnitude = u.get_magnitude(self.sigma.in_unit(distances.unit))

        # Calculate connection probabilities based on distance
        dist_ratio = distances_magnitude / sigma_magnitude

        if self.decay == 'gaussian':
            probs = self.max_prob * np.exp(-(dist_ratio ** 2) / 2)
        elif self.decay == 'exponential':
            probs = self.max_prob * np.exp(-dist_ratio)
        elif self.decay == 'linear':
            probs = np.maximum(0, self.max_prob * (1 - dist_ratio))
        elif self.decay == 'power':
            probs = self.max_prob / (1 + dist_ratio ** self.exponent)
        else:
            raise ValueError(f"Unknown decay function: {self.decay}")

        # Generate connections based on probabilities
        rng = np.random.RandomState(self.seed)
        random_vals = rng.random_sample(probs.shape)

        # Find connections
        pre_indices, post_indices = np.where(random_vals < probs)

        # Calculate weights with proper units
        weight_factors = probs[pre_indices, post_indices]
        if isinstance(self.weight_scale, u.Quantity):
            weights = weight_factors * self.weight_scale
        else:
            weights = weight_factors * self.weight_scale

        # Store positions for distance calculations
        positions = (pre_pos, post_pos)

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            weights=weights,
            positions=positions,
            metadata={'pattern': 'distance_dependent', 'sigma': self.sigma, 'decay': self.decay}
        )


class Gaussian(Connectivity):
    """Gaussian connectivity pattern.

    Parameters
    ----------
    sigma : float
        Standard deviation for Gaussian.
    """

    __module__ = 'braintools.conn'

    def __init__(self, sigma: float, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate Gaussian connectivity."""
        if isinstance(pre_size, int):
            pre_num = pre_size
        else:
            pre_num = int(np.prod(np.asarray(pre_size)))

        if isinstance(post_size, int):
            post_num = post_size
        else:
            post_num = int(np.prod(np.asarray(post_size)))

        rng = np.random.RandomState(self.seed)

        # Generate random weights from Gaussian distribution
        n_connections = int(0.1 * pre_num * post_num)  # 10% connectivity
        pre_indices = rng.choice(pre_num, n_connections)
        post_indices = rng.choice(post_num, n_connections)

        # Gaussian weights
        weights = rng.normal(0, self.sigma, n_connections)

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            weights=weights,
            metadata={'pattern': 'gaussian', 'sigma': self.sigma}
        )


class Regular(Connectivity):
    """Regular connectivity patterns.

    Parameters
    ----------
    pattern : str
        Type of regular pattern ('ring', 'grid', 'lattice').
    neighbors : int
        Number of neighbors for ring pattern.
    """

    __module__ = 'braintools.conn'

    def __init__(self, pattern: str = 'ring', neighbors: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.neighbors = neighbors

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate regular connectivity."""
        if isinstance(pre_size, int):
            pre_num = pre_size
        else:
            pre_num = int(np.prod(np.asarray(pre_size)))

        if isinstance(post_size, int):
            post_num = post_size
        else:
            post_num = int(np.prod(np.asarray(post_size)))

        pre_indices = []
        post_indices = []

        if self.pattern == 'ring':
            # Ring connectivity: each neuron connects to nearest neighbors
            for i in range(pre_num):
                for k in range(1, self.neighbors + 1):
                    # Forward connections
                    j = (i + k) % post_num
                    pre_indices.append(i)
                    post_indices.append(j)

                    # Backward connections
                    j = (i - k) % post_num
                    pre_indices.append(i)
                    post_indices.append(j)

        elif self.pattern == 'grid':
            # 2D grid connectivity (requires square layout)
            side = int(np.sqrt(min(pre_num, post_num)))
            for i in range(side):
                for j in range(side):
                    idx = i * side + j
                    if idx >= pre_num:
                        break

                    # Connect to neighbors in grid
                    neighbors = []
                    if i > 0: neighbors.append((i-1) * side + j)  # Up
                    if i < side-1: neighbors.append((i+1) * side + j)  # Down
                    if j > 0: neighbors.append(i * side + (j-1))  # Left
                    if j < side-1: neighbors.append(i * side + (j+1))  # Right

                    for neighbor in neighbors:
                        if neighbor < post_num:
                            pre_indices.append(idx)
                            post_indices.append(neighbor)

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            metadata={'pattern': self.pattern, 'neighbors': self.neighbors}
        )


# Convenience aliases that inherit from DistanceDependent
class Exponential(DistanceDependent):
    def __init__(self, sigma, **kwargs):
        super().__init__(sigma=sigma, decay='exponential', **kwargs)


class PowerLaw(DistanceDependent):
    def __init__(self, sigma, exponent=2.0, **kwargs):
        super().__init__(sigma=sigma, decay='power', exponent=exponent, **kwargs)


# Convenience aliases that inherit from Regular
class Ring(Regular):
    def __init__(self, neighbors=2, **kwargs):
        super().__init__(pattern='ring', neighbors=neighbors, **kwargs)


class Grid(Regular):
    def __init__(self, **kwargs):
        super().__init__(pattern='grid', **kwargs)