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

from typing import Optional, Tuple, Union

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike
from scipy.spatial.distance import cdist

from braintools.init._distance_base import DistanceProfile
from braintools.init._init_base import init_call, Initializer
from ._base import PointConnectivity, ConnectionResult

__all__ = [
    # Spatial patterns
    'DistanceDependent',
    'Gaussian',
    'Exponential',
    'Ring',
    'Grid',
    'RadialPatches',
]


class DistanceDependent(PointConnectivity):
    """Distance-dependent connectivity for spatially arranged point neurons.

    Parameters
    ----------
    distance_profile : DistanceProfile
        Distance profile class (e.g., GaussianProfile, ExponentialProfile).
    weight : Initialization, optional
        Weight initialization for connections.
        If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for connections.
        If None, no delays are generated.
    max_prob : float
        Maximum connection probability scaling factor.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.init import GaussianProfile, Exponential, Constant
        >>>
        >>> # Gaussian distance-dependent connectivity
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> conn = DistanceDependent(
        ...     distance_profile=GaussianProfile(
        ...         sigma=100 * u.um,
        ...         max_distance=300 * u.um
        ...     ),
        ...     weight=Exponential(3.0 * u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     max_prob=0.3
        ... )
        >>> result = conn(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """

    def __init__(
        self,
        distance_profile: Optional[Union[ArrayLike, DistanceProfile]] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        max_prob: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.distance_profile = distance_profile
        self.weight_init = weight
        self.delay_init = delay
        self.max_prob = max_prob

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate distance-dependent connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for spatial connectivity")

        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Calculate distance matrix
        pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
        post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa
        distances = u.maybe_decimal(cdist(pre_pos_val, post_pos_val) * pos_unit)

        # Calculate connection probabilities using distance profile
        probs = self.max_prob * self.distance_profile.probability(distances)

        # Vectorized connection generation
        random_vals = self.rng.random((pre_num, post_num))
        connection_mask = (probs > 0) & (random_vals < probs)

        pre_indices, post_indices = np.where(connection_mask)
        connection_distances = distances[connection_mask]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point',
            )

        n_connections = len(pre_indices)

        # Generate weights using initialization class
        # Pass distances for distance-dependent weight distributions
        weights = init_call(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            distances=connection_distances
        )

        # Generate delays using initialization class
        delays = init_call(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='point',
            metadata={
                'pattern': 'distance_dependent',
                'distance_profile': self.distance_profile,
                'weight_initialization': self.weight_init,
                'delay_initialization': self.delay_init,
                'max_prob': self.max_prob
            }
        )


class Gaussian(DistanceDependent):
    """Gaussian distance-dependent connectivity.

    Parameters
    ----------
    distance_profile : DistanceProfile
        Must be a GaussianProfile instance.
    **kwargs
        Additional arguments passed to DistanceDependent.
    """
    pass


class Exponential(DistanceDependent):
    """Exponential distance-dependent connectivity.

    Parameters
    ----------
    distance_profile : DistanceProfile
        Must be an ExponentialProfile instance.
    **kwargs
        Additional arguments passed to DistanceDependent.
    """
    pass


class Ring(PointConnectivity):
    """Ring connectivity pattern where each neuron connects to its neighbors.

    Parameters
    ----------
    neighbors : int
        Number of neighbors on each side to connect to.
    weight : Initialization, optional
        Weight initialization for connections.
    delay : Initialization, optional
        Delay initialization for connections.
    bidirectional : bool
        If True, connections are bidirectional.

    Examples
    --------
    .. code-block:: python

        >>> ring = Ring(neighbors=2, weight=1.0 * u.nS)
        >>> result = ring(pre_size=100, post_size=100)
    """

    def __init__(
        self,
        neighbors: int = 2,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        bidirectional: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.neighbors = neighbors
        self.weight_init = weight
        self.delay_init = delay
        self.bidirectional = bidirectional

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate ring connectivity."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if pre_size != post_size:
            raise ValueError("Ring networks require pre_size == post_size")

        pre_indices = []
        post_indices = []

        # Connect each neuron to its neighbors
        for i in range(n):
            for offset in range(1, self.neighbors + 1):
                # Forward connections
                target = (i + offset) % n
                pre_indices.append(i)
                post_indices.append(target)

                # Backward connections if bidirectional
                if self.bidirectional and offset > 0:
                    target = (i - offset) % n
                    pre_indices.append(i)
                    post_indices.append(target)

        n_connections = len(pre_indices)

        # Generate weights and delays
        weights = init_call(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )
        delays = init_call(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata={
                'pattern': 'ring',
                'neighbors': self.neighbors,
                'bidirectional': self.bidirectional
            }
        )


class Grid(PointConnectivity):
    """2D grid connectivity pattern where neurons connect to their grid neighbors.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the 2D grid (rows, cols).
    connectivity : str
        Type of neighborhood: 'von_neumann' (4 neighbors) or 'moore' (8 neighbors).
    weight : Initialization, optional
        Weight initialization for connections.
    delay : Initialization, optional
        Delay initialization for connections.
    periodic : bool
        If True, use periodic boundary conditions (wrap around edges).

    Examples
    --------
    .. code-block:: python

        >>> grid = Grid(
        ...     grid_shape=(10, 10),
        ...     connectivity='moore',
        ...     weight=1.0 * u.nS,
        ...     periodic=True
        ... )
        >>> result = grid(pre_size=100, post_size=100)
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        connectivity: str = 'von_neumann',
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        periodic: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grid_shape = grid_shape
        self.connectivity = connectivity
        self.weight_init = weight
        self.delay_init = delay
        self.periodic = periodic

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate grid connectivity."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if pre_size != post_size:
            raise ValueError("Grid networks require pre_size == post_size")

        rows, cols = self.grid_shape
        if rows * cols != n:
            raise ValueError(f"Grid shape {self.grid_shape} doesn't match population size {n}")

        pre_indices = []
        post_indices = []

        # Define neighbor offsets
        if self.connectivity == 'von_neumann':
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif self.connectivity == 'moore':
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            raise ValueError(f"Unknown connectivity type: {self.connectivity}")

        # Create connections
        for i in range(rows):
            for j in range(cols):
                source_idx = i * cols + j

                for di, dj in offsets:
                    ni, nj = i + di, j + dj

                    # Handle boundary conditions
                    if self.periodic:
                        ni = ni % rows
                        nj = nj % cols
                    else:
                        if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                            continue

                    target_idx = ni * cols + nj
                    pre_indices.append(source_idx)
                    post_indices.append(target_idx)

        n_connections = len(pre_indices)

        # Generate weights and delays
        weights = init_call(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )
        delays = init_call(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata={
                'pattern': 'grid',
                'grid_shape': self.grid_shape,
                'connectivity': self.connectivity,
                'periodic': self.periodic
            }
        )


class RadialPatches(PointConnectivity):
    """Radial patch connectivity where connections form radial patches around neurons.

    Parameters
    ----------
    patch_radius : float or Quantity
        Radius of each patch.
    n_patches : int
        Number of patches per neuron.
    prob : float
        Connection probability within each patch.
    weight : Initialization, optional
        Weight initialization.
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> patches = RadialPatches(
        ...     patch_radius=50 * u.um,
        ...     n_patches=3,
        ...     prob=0.5,
        ...     weight=1.0 * u.nS
        ... )
        >>> result = patches(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """

    def __init__(
        self,
        patch_radius: Union[float, u.Quantity],
        n_patches: int = 1,
        prob: float = 1.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_radius = patch_radius
        self.n_patches = n_patches
        self.prob = prob
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate radial patch connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for radial patch connectivity")

        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Calculate distances
        pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
        post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa
        # distances = cdist(pre_pos_val, post_pos_val)

        # Get radius value
        if isinstance(self.patch_radius, u.Quantity):
            radius_val = u.Quantity(self.patch_radius).to(pos_unit).mantissa
        else:
            radius_val = self.patch_radius

        # For each pre neuron, select random patch centers and connect within radius
        pre_indices = []
        post_indices = []

        for i in range(pre_num):
            # Select random patch centers from post population
            patch_centers = self.rng.choice(post_num, size=min(self.n_patches, post_num), replace=False)

            # For each patch, find neurons within radius
            for center in patch_centers:
                # Find candidates within radius of patch center
                center_pos = post_pos_val[center]
                dists_from_center = np.sqrt(np.sum((post_pos_val - center_pos) ** 2, axis=1))
                candidates = np.where(dists_from_center <= radius_val)[0]

                # Apply connection probability
                if len(candidates) > 0:
                    random_vals = self.rng.random(len(candidates))
                    selected = candidates[random_vals < self.prob]
                    pre_indices.extend([i] * len(selected))
                    post_indices.extend(selected)

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point'
            )

        # Remove duplicates
        connections = set(zip(pre_indices, post_indices))
        pre_indices, post_indices = zip(*connections)
        pre_indices = np.array(pre_indices, dtype=np.int64)
        post_indices = np.array(post_indices, dtype=np.int64)

        n_connections = len(pre_indices)

        weights = init_call(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )
        delays = init_call(
            self.delay_init,
            n_connections,
            rng=self.rng,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        return ConnectionResult(
            pre_indices,
            post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=pre_positions,
            post_positions=post_positions,
            metadata={
                'pattern': 'radial_patches',
                'patch_radius': self.patch_radius,
                'n_patches': self.n_patches,
            }
        )
