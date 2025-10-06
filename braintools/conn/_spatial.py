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

from typing import Optional, Union

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike
from scipy.spatial.distance import cdist

from braintools.init._distance_base import DistanceProfile
from braintools.init._distance_impl import GaussianProfile, ExponentialProfile
from braintools.init._init_base import param, Initializer
from ._base import PointConnectivity, ConnectionResult

__all__ = [
    'DistanceDependent',
    'Gaussian',
    'Exponential',
    'Ring',
    'Grid2d',
    'RadialPatches',
]


class DistanceDependent(PointConnectivity):
    """Distance-dependent connectivity for spatially arranged point neurons.

    This class creates synaptic connections between neurons based on their spatial distances,
    where the connection probability is determined by a distance-dependent profile. The
    connectivity pattern enables modeling of local and distance-modulated neural circuits
    commonly observed in biological neural networks.

    The connection generation process:

    1. Computes pairwise distances between pre- and post-synaptic neuron positions
    2. Calculates connection probabilities using the provided distance profile
    3. Stochastically generates connections based on these probabilities
    4. Assigns weights and delays to established connections using initializers

    Parameters
    ----------
    distance_profile : DistanceProfile or ArrayLike
        A distance profile object that defines how connection probability varies with
        distance. Common profiles include:

        - ``GaussianProfile``: Probability decreases as a Gaussian function of distance
        - ``ExponentialProfile``: Probability decays exponentially with distance
        - Custom profiles implementing the ``DistanceProfile`` interface

        Alternatively, can be an array-like object specifying probabilities directly.
    weight : Initializer, optional
        Initializer for synaptic weights. Can be:

        - A constant value (e.g., ``1.0 * u.nS``)
        - An initializer object (e.g., ``Normal(mean=1.0*u.nS, std=0.1*u.nS)``)
        - ``None`` to create unweighted connections

        Some initializers support distance-dependent weight generation.
    delay : Initializer, optional
        Initializer for synaptic delays. Can be:

        - A constant value (e.g., ``1.0 * u.ms``)
        - An initializer object (e.g., ``Uniform(0.5*u.ms, 2.0*u.ms)``)
        - ``None`` to create connections without explicit delays
    **kwargs
        Additional keyword arguments passed to the parent ``PointConnectivity`` class,
        such as ``seed`` for reproducible random number generation.

    Returns
    -------
    ConnectionResult
        A connection result object containing:

        - ``pre_indices``: Pre-synaptic neuron indices
        - ``post_indices``: Post-synaptic neuron indices
        - ``weights``: Connection weights (if weight initializer provided)
        - ``delays``: Connection delays (if delay initializer provided)
        - ``metadata``: Additional information about the connectivity pattern

    Notes
    -----
    - Requires neuron positions to be provided via ``pre_positions`` and ``post_positions``
      arguments when calling the connector, or a pre-computed ``distances`` array
    - Position arrays should have shape ``(n_neurons, n_dimensions)`` with units
    - The actual number of connections is stochastic and depends on the distance profile
      and random sampling
    - Empty connection results are returned if no connections are established

    See Also
    --------
    Gaussian : Specialized class for Gaussian distance-dependent connectivity
    Exponential : Specialized class for exponential distance-dependent connectivity
    braintools.init.GaussianProfile : Gaussian distance profile
    braintools.init.ExponentialProfile : Exponential distance profile

    Examples
    --------
    Basic usage with Gaussian distance profile:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import DistanceDependent
        >>> from braintools.init import GaussianProfile, Normal, Constant
        >>>
        >>> # Create neuron positions in 2D space
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>>
        >>> # Define Gaussian distance-dependent connectivity
        >>> conn = DistanceDependent(
        ...     distance_profile=GaussianProfile(
        ...         sigma=100 * u.um,
        ...         max_distance=300 * u.um
        ...     ),
        ...     weight=Normal(mean=1.5*u.nS, std=0.3*u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     seed=42
        ... )
        >>>
        >>> # Generate connections
        >>> result = conn(
        ...     pre_size=500,
        ...     post_size=500,
        ...     pre_positions=positions,
        ...     post_positions=positions
        ... )
        >>> print(f"Generated {len(result.pre_indices)} connections")

    Using exponential distance profile:

    .. code-block:: python

        >>> from braintools.init import ExponentialProfile
        >>>
        >>> conn = DistanceDependent(
        ...     distance_profile=ExponentialProfile(
        ...         scale=150 * u.um,
        ...         max_distance=500 * u.um
        ...     ),
        ...     weight=Constant(2.0 * u.nS)
        ... )

    Pre-computed distances:

    .. code-block:: python

        >>> # Use pre-computed distance matrix instead of positions
        >>> from scipy.spatial.distance import cdist
        >>> distances = cdist(positions.mantissa, positions.mantissa) * u.um
        >>> result = conn(
        ...     pre_size=500,
        ...     post_size=500,
        ...     distances=distances
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        distance_profile: Union[ArrayLike, DistanceProfile],
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.distance_profile = distance_profile
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate distance-dependent connections."""

        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Calculate distance matrix
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)
        distances = kwargs.get('distances', None)
        if distances is not None:
            # Use provided distances directly
            distances = distances
        elif pre_positions is None or post_positions is None:
            raise ValueError('Positions required for distance-dependent connectivity, '
                             'for example: pre_positions=positions, post_positions=positions, '
                             'or provide distances directly.')
        else:
            pre_pos_val, pos_unit = u.split_mantissa_unit(pre_positions)
            post_pos_val = u.Quantity(post_positions).to(pos_unit).mantissa
            distances = u.maybe_decimal(cdist(pre_pos_val, post_pos_val) * pos_unit)

        # Calculate connection probabilities using distance profile
        probs = self.distance_profile.probability(distances)

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
        weights = param(
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
        delays = param(
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
            }
        )


class Gaussian(DistanceDependent):
    """Gaussian distance-dependent connectivity for spatially organized neural networks.

    This specialized connectivity class implements connections where the probability of
    connection follows a Gaussian (normal) distribution as a function of distance between
    neurons. This pattern is commonly observed in cortical networks where neurons preferentially
    connect to nearby neighbors with decreasing probability at larger distances.

    The connection probability follows:

    .. math::

        P(d) = \\exp\\left(-\\frac{d^2}{2\\sigma^2}\\right) \\quad \\text{for } d \\leq d_{\\text{max}}

    where:

    - :math:`d` is the distance between neurons
    - :math:`\\sigma` is the spread parameter (standard deviation)
    - :math:`d_{\\text{max}}` is the maximum connection distance

    Parameters
    ----------
    distance_profile : GaussianProfile
        A Gaussian distance profile that defines the spatial connection probability.
        Must be an instance of ``braintools.init.GaussianProfile`` with:

        - ``sigma``: Standard deviation controlling the spread
        - ``max_distance``: Maximum distance for connections (optional)
        - ``normalize``: Whether to normalize to unit maximum (optional)
    weight : Initializer, optional
        Initializer for synaptic weights. Supports:

        - Constant values (e.g., ``1.5 * u.nS``)
        - Stochastic initializers (e.g., ``Normal``, ``Uniform``, ``LogNormal``)
        - Distance-dependent initializers
        - ``None`` for unweighted connections
    delay : Initializer, optional
        Initializer for synaptic transmission delays. Supports:

        - Constant values (e.g., ``1.0 * u.ms``)
        - Stochastic initializers
        - Distance-dependent delays
        - ``None`` for connections without explicit delays
    **kwargs
        Additional arguments passed to ``DistanceDependent``, including:

        - ``seed``: Random seed for reproducible connection generation

    Returns
    -------
    ConnectionResult
        Connection result containing indices, weights, delays, and metadata about
        the Gaussian connectivity pattern.

    Raises
    ------
    TypeError
        If ``distance_profile`` is not an instance of ``GaussianProfile``.
    ValueError
        If required positions are not provided when generating connections.

    Notes
    -----

    - The ``sigma`` parameter controls the spatial extent of connections: larger values
      create more diffuse connectivity, while smaller values restrict connections to
      immediate neighbors
    - Setting ``max_distance`` limits computational cost by preventing consideration of
      very distant neuron pairs with negligible connection probability
    - The Gaussian profile is appropriate for modeling local cortical connectivity and
      spatial receptive fields
    - Connection generation is stochastic; the same configuration will produce different
      results without fixing the random seed

    See Also
    --------
    Exponential : Exponential distance-dependent connectivity
    DistanceDependent : General distance-dependent connectivity base class
    braintools.init.GaussianProfile : Gaussian distance profile implementation

    Examples
    --------
    Simple Gaussian connectivity in a 2D population:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import Gaussian
        >>> from braintools.init import GaussianProfile, Constant
        >>>
        >>> # Create 2D positions
        >>> positions = np.random.uniform(0, 1000, (400, 2)) * u.um
        >>>
        >>> # Gaussian connectivity with sigma=100um
        >>> conn = Gaussian(
        ...     distance_profile=GaussianProfile(
        ...         sigma=100 * u.um,
        ...         max_distance=300 * u.um
        ...     ),
        ...     weight=Constant(1.0 * u.nS),
        ...     delay=Constant(1.0 * u.ms)
        ... )
        >>>
        >>> result = conn(
        ...     pre_size=400,
        ...     post_size=400,
        ...     pre_positions=positions,
        ...     post_positions=positions
        ... )

    Local connectivity with narrow Gaussian:

    .. code-block:: python

        >>> # Narrow Gaussian for highly local connections
        >>> local_conn = Gaussian(
        ...     distance_profile=GaussianProfile(
        ...         sigma=50 * u.um,  # Narrow spread
        ...         max_distance=150 * u.um
        ...     ),
        ...     weight=Constant(2.0 * u.nS)
        ... )

    Broad connectivity with distance-dependent weights:

    .. code-block:: python

        >>> from braintools.init import DistanceModulated as DDWeight, Normal
        >>>
        >>> broad_conn = Gaussian(
        ...     distance_profile=GaussianProfile(
        ...         sigma=200 * u.um,  # Broader spread
        ...         max_distance=600 * u.um
        ...     ),
        ...     weight=DDWeight(lambda d: (3.0 * u.nS) * np.exp(-d/(100*u.um)))
        ... )

    Modeling cortical column connectivity:

    .. code-block:: python

        >>> # Model local connectivity within a cortical column
        >>> # Positions in 3D (x, y for horizontal, z for layer depth)
        >>> column_positions = np.random.randn(500, 3) * np.array([50, 50, 200]) * u.um
        >>>
        >>> # Anisotropic connectivity using custom Gaussian profile
        >>> column_conn = Gaussian(
        ...     distance_profile=GaussianProfile(
        ...         sigma=75 * u.um,
        ...         max_distance=250 * u.um
        ...     ),
        ...     weight=Normal(mean=1.5*u.nS, std=0.3*u.nS),
        ...     seed=42
        ... )
        >>>
        >>> result = column_conn(
        ...     pre_size=500,
        ...     post_size=500,
        ...     pre_positions=column_positions,
        ...     post_positions=column_positions
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        distance_profile: GaussianProfile,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        if not isinstance(distance_profile, GaussianProfile):
            raise TypeError(
                "distance_profile must be an instance of GaussianProfile for Gaussian connectivity."
            )
        super().__init__(
            distance_profile=distance_profile,
            weight=weight,
            delay=delay,
            **kwargs
        )


class Exponential(DistanceDependent):
    """Exponential distance-dependent connectivity for spatially organized neural networks.

    This specialized connectivity class creates connections where the probability follows an
    exponential decay as a function of distance between neurons. This pattern is characteristic
    of many biological neural systems where connection probability drops off smoothly with
    distance, such as in hippocampal networks and certain cortical projections.

    The connection probability follows:

    .. math::

        P(d) = \\exp\\left(-\\frac{d}{\\lambda}\\right) \\quad \\text{for } d \\leq d_{\\text{max}}

    where:

    - :math:`d` is the distance between neurons
    - :math:`\\lambda` is the spatial scale (decay length constant)
    - :math:`d_{\\text{max}}` is the maximum connection distance

    Parameters
    ----------
    distance_profile : ExponentialProfile
        An exponential distance profile that defines the spatial connection probability decay.
        Must be an instance of ``braintools.init.ExponentialProfile`` with:

        - ``scale``: Characteristic decay length (λ) - larger values create longer-range connections
        - ``max_distance``: Maximum distance for connections (optional)
        - ``normalize``: Whether to normalize to unit maximum (optional)
    weight : Initializer, optional
        Initializer for synaptic weights. Supports:

        - Constant values (e.g., ``2.0 * u.nS``)
        - Stochastic initializers (e.g., ``Normal``, ``Uniform``, ``LogNormal``)
        - Distance-dependent weight distributions
        - ``None`` for unweighted connections
    delay : Initializer, optional
        Initializer for synaptic transmission delays. Supports:

        - Constant values (e.g., ``1.5 * u.ms``)
        - Stochastic delay distributions
        - Distance-proportional delays
        - ``None`` for connections without explicit delays
    **kwargs
        Additional arguments passed to ``DistanceDependent``, such as:

        - ``seed``: Random seed for reproducible connection patterns

    Returns
    -------
    ConnectionResult
        Connection result containing pre/post indices, weights, delays, and metadata
        about the exponential connectivity pattern.

    Raises
    ------
    TypeError
        If ``distance_profile`` is not an instance of ``ExponentialProfile``.
    ValueError
        If neuron positions are not provided when generating connections.

    Notes
    -----
    - The ``scale`` parameter (λ) determines how quickly connection probability decays:

      - At distance λ, probability is reduced to ~37% (1/e) of the maximum
      - Smaller scale values create more localized connectivity
      - Larger scale values allow longer-range connections

    - Exponential decay is generally slower than Gaussian decay at intermediate distances
      but faster at very short distances, making it suitable for modeling projections
      with moderate spatial extent
    - Setting ``max_distance`` improves computational efficiency by limiting the maximum
      connection range
    - Unlike Gaussian profiles, exponential profiles have no inflection point and decay
      monotonically
    - Connection generation is stochastic and requires a fixed seed for reproducibility

    See Also
    --------
    Gaussian : Gaussian distance-dependent connectivity
    DistanceDependent : General distance-dependent connectivity base class
    braintools.init.ExponentialProfile : Exponential distance profile implementation

    Examples
    --------
    Basic exponential connectivity:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import Exponential
        >>> from braintools.init import ExponentialProfile, Constant
        >>>
        >>> # Create random 2D positions
        >>> positions = np.random.uniform(0, 1000, (300, 2)) * u.um
        >>>
        >>> # Exponential connectivity with scale=150um
        >>> conn = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         scale=150 * u.um,
        ...         max_distance=500 * u.um
        ...     ),
        ...     weight=Constant(1.5 * u.nS),
        ...     delay=Constant(1.0 * u.ms)
        ... )
        >>>
        >>> result = conn(
        ...     pre_size=300,
        ...     post_size=300,
        ...     pre_positions=positions,
        ...     post_positions=positions
        ... )

    Short-range exponential connectivity:

    .. code-block:: python

        >>> # Rapid decay for highly local connections
        >>> local_conn = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         scale=75 * u.um,  # Short decay length
        ...         max_distance=225 * u.um
        ...     ),
        ...     weight=Constant(2.5 * u.nS)
        ... )

    Long-range connections with variable weights:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>>
        >>> # Longer-range connectivity with stochastic weights
        >>> long_range_conn = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         scale=300 * u.um,  # Long decay length
        ...         max_distance=900 * u.um
        ...     ),
        ...     weight=Normal(mean=1.0*u.nS, std=0.2*u.nS),
        ...     seed=42
        ... )

    Distance-dependent delays:

    .. code-block:: python

        >>> from braintools.init import DistanceModulated as DDDelay, LogNormal
        >>>
        >>> # Model conduction delays proportional to distance
        >>> conn_with_delays = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         scale=200 * u.um,
        ...         max_distance=600 * u.um
        ...     ),
        ...     weight=Constant(1.0 * u.nS),
        ...     delay=DDDelay(lambda d: (0.5 * u.ms) + d / (300 * u.um/u.ms))
        ... )

    Modeling hippocampal connectivity:

    .. code-block:: python

        >>> # Exponential connectivity pattern typical of hippocampal networks
        >>> positions_1d = np.linspace(0, 2000, 200).reshape(-1, 1) * u.um
        >>>
        >>> hippo_conn = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         scale=250 * u.um,  # Moderate spatial extent
        ...         max_distance=1000 * u.um
        ...     ),
        ...     weight=LogNormal(mean=1.5*u.nS, sigma=0.4),
        ...     delay=Normal(mean=2.0*u.ms, std=0.3*u.ms)
        ... )
        >>>
        >>> result = hippo_conn(
        ...     pre_size=200,
        ...     post_size=200,
        ...     pre_positions=positions_1d,
        ...     post_positions=positions_1d
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        distance_profile: ExponentialProfile,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        if not isinstance(distance_profile, ExponentialProfile):
            raise TypeError(
                "distance_profile must be an instance of ExponentialProfile for Exponential connectivity."
            )
        super().__init__(
            distance_profile=distance_profile,
            weight=weight,
            delay=delay,
            **kwargs
        )


class Ring(PointConnectivity):
    """Ring topology connectivity where neurons are arranged in a circular structure.

    This class creates connections in a one-dimensional ring (circular) topology where each
    neuron connects to a specified number of neighboring neurons on each side. This pattern
    is commonly used to model spatially organized networks with periodic boundary conditions,
    such as orientation columns in visual cortex or head direction cells.

    The connectivity pattern:

    - Each neuron at position i connects to neurons at positions (i ± k) mod N, for k in [1, neighbors]
    - Connections wrap around at the boundaries (periodic boundary conditions)
    - Optionally bidirectional, creating symmetric connectivity

    Parameters
    ----------
    neighbors : int, default=2
        Number of neighbors on each side to connect to. For example:

        - ``neighbors=1``: Each neuron connects to its immediate neighbors (2 connections per neuron if bidirectional)
        - ``neighbors=2``: Each neuron connects to 2 neighbors on each side (4 connections per neuron if bidirectional)
        - ``neighbors=N//2``: Maximally connected ring topology
    weight : Initializer, optional
        Initializer for synaptic weights. Supports:

        - Constant values (e.g., ``1.0 * u.nS``)
        - Stochastic initializers (e.g., ``Normal``, ``Uniform``)
        - ``None`` for unweighted connections
    delay : Initializer, optional
        Initializer for synaptic transmission delays. Supports:

        - Constant values (e.g., ``1.0 * u.ms``)
        - Stochastic delay distributions
        - ``None`` for connections without explicit delays
    bidirectional : bool, default=True
        If ``True``, connections are bidirectional (neuron i connects to i+k and i-k).
        If ``False``, connections are unidirectional (only forward, i to i+k).
    **kwargs
        Additional arguments passed to ``PointConnectivity``, such as ``seed`` for
        reproducible random number generation.

    Returns
    -------
    ConnectionResult
        Connection result containing pre/post indices, weights, delays, and metadata
        about the ring connectivity pattern.

    Raises
    ------
    ValueError
        If ``pre_size`` and ``post_size`` are not equal (ring topology requires same size).

    Notes
    -----
    - Ring connectivity enforces ``pre_size == post_size`` as each neuron must have the
      same number of potential neighbors
    - The total number of connections is:

      - Bidirectional: ``N × 2 × neighbors``
      - Unidirectional: ``N × neighbors``

      where N is the population size
    - This topology is useful for modeling networks with periodic spatial structure
    - Self-connections are excluded (neurons do not connect to themselves)
    - The ring wraps around, so neuron 0 connects to neuron N-1 when neighbors > 0

    See Also
    --------
    Grid2d : Two-dimensional grid connectivity
    DistanceDependent : Distance-based connectivity for spatially embedded networks

    Examples
    --------
    Basic ring connectivity:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import Ring
        >>> from braintools.init import Constant
        >>>
        >>> # Each neuron connects to 2 neighbors on each side
        >>> ring = Ring(neighbors=2, weight=Constant(1.0 * u.nS))
        >>> result = ring(pre_size=100, post_size=100)
        >>> print(f"Total connections: {len(result.pre_indices)}")  # 400 connections

    Unidirectional ring (feed-forward):

    .. code-block:: python

        >>> # Only forward connections
        >>> ff_ring = Ring(
        ...     neighbors=1,
        ...     weight=Constant(1.5 * u.nS),
        ...     bidirectional=False
        ... )
        >>> result = ff_ring(pre_size=100, post_size=100)
        >>> print(f"Total connections: {len(result.pre_indices)}")  # 100 connections

    Ring with stochastic weights:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>>
        >>> ring = Ring(
        ...     neighbors=3,
        ...     weight=Normal(mean=1.0*u.nS, std=0.2*u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     seed=42
        ... )

    Modeling orientation preference in visual cortex:

    .. code-block:: python

        >>> # Ring of neurons representing orientation preferences
        >>> n_orientations = 180  # One neuron per degree
        >>>
        >>> # Local connectivity in orientation space
        >>> orientation_ring = Ring(
        ...     neighbors=10,  # Connect to neurons with similar orientations
        ...     weight=Constant(2.0 * u.nS),
        ...     delay=Constant(0.5 * u.ms),
        ...     bidirectional=True
        ... )
        >>>
        >>> result = orientation_ring(
        ...     pre_size=n_orientations,
        ...     post_size=n_orientations
        ... )
    """
    __module__ = 'braintools.conn'

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
        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )
        delays = param(
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


class Grid2d(PointConnectivity):
    """Two-dimensional grid connectivity for spatially arranged neural populations.

    This class creates connections on a 2D regular lattice where each neuron connects to
    its immediate spatial neighbors. This pattern is commonly used to model cortical
    sheets, retinal ganglion cell arrays, or any neural population with regular 2D
    spatial organization.

    Two types of neighborhoods are supported:

    - **Von Neumann neighborhood** (4 neighbors): Connects to orthogonal neighbors (up, down, left, right)
    - **Moore neighborhood** (8 neighbors): Connects to all 8 surrounding neighbors (orthogonal + diagonal)

    The connectivity pattern respects or wraps around grid boundaries depending on the
    ``periodic`` parameter.

    Parameters
    ----------
    connectivity : str, default='von_neumann'
        Type of neighborhood connectivity:

        - ``'von_neumann'``: 4-neighbor connectivity (orthogonal only)

          .. code-block:: text

              · N ·
              W · E
              · S ·

        - ``'moore'``: 8-neighbor connectivity (orthogonal + diagonal)

          .. code-block:: text

              NW N NE
              W  ·  E
              SW S SE

    weight : Initializer, optional
        Initializer for synaptic weights. Supports:

        - Constant values (e.g., ``1.0 * u.nS``)
        - Stochastic initializers (e.g., ``Normal``, ``Uniform``)
        - ``None`` for unweighted connections
    delay : Initializer, optional
        Initializer for synaptic transmission delays. Supports:

        - Constant values (e.g., ``1.0 * u.ms``)
        - Stochastic delay distributions
        - ``None`` for connections without explicit delays
    periodic : bool, default=False
        Whether to use periodic boundary conditions:

        - ``True``: Grid wraps around at edges (torus topology)
        - ``False``: Neurons at edges have fewer neighbors
    **kwargs
        Additional arguments passed to ``PointConnectivity``, such as ``seed`` for
        reproducible random number generation.

    Returns
    -------
    ConnectionResult
        Connection result containing pre/post indices, weights, delays, and metadata
        about the grid connectivity pattern.

    Raises
    ------
    ValueError
        If ``pre_size`` and ``post_size`` are not equal, not 2D tuples, or if
        ``connectivity`` type is invalid.

    Notes
    -----
    - Grid connectivity requires ``pre_size == post_size`` as a 2D tuple ``(rows, cols)``
    - The number of connections per neuron depends on connectivity type and boundary conditions:

      - Von Neumann, non-periodic: 2-4 neighbors (fewer at edges)
      - Von Neumann, periodic: 4 neighbors for all neurons
      - Moore, non-periodic: 3-8 neighbors (fewer at edges/corners)
      - Moore, periodic: 8 neighbors for all neurons

    - Neurons are indexed in row-major order: index = row × n_cols + col
    - Self-connections are excluded
    - Total connections:

      - Von Neumann: ~4 × rows × cols (periodic) or fewer (non-periodic)
      - Moore: ~8 × rows × cols (periodic) or fewer (non-periodic)

    See Also
    --------
    Ring : One-dimensional ring connectivity
    DistanceDependent : Distance-based connectivity for arbitrary spatial arrangements

    Examples
    --------
    Basic 2D grid with Von Neumann connectivity:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import Grid2d
        >>> from braintools.init import Constant
        >>>
        >>> # 10×10 grid with 4-neighbor connectivity
        >>> grid = Grid2d(
        ...     connectivity='von_neumann',
        ...     weight=Constant(1.0 * u.nS),
        ...     periodic=False
        ... )
        >>> result = grid(pre_size=(10, 10), post_size=(10, 10))

    Moore neighborhood with periodic boundaries:

    .. code-block:: python

        >>> # 8-neighbor connectivity with wraparound
        >>> moore_grid = Grid2d(
        ...     connectivity='moore',
        ...     weight=Constant(1.5 * u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     periodic=True  # Torus topology
        ... )
        >>> result = moore_grid(pre_size=(20, 20), post_size=(20, 20))
        >>> print(f"Connections per neuron: {len(result.pre_indices) // 400}")  # Should be 8

    Stochastic weights on grid:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>>
        >>> grid = Grid2d(
        ...     connectivity='von_neumann',
        ...     weight=Normal(mean=2.0*u.nS, std=0.4*u.nS),
        ...     delay=Constant(0.5 * u.ms),
        ...     seed=42
        ... )

    Modeling cortical sheet connectivity:

    .. code-block:: python

        >>> # Model local connectivity in a cortical layer
        >>> cortex_grid = Grid2d(
        ...     connectivity='moore',  # Neurons connect to all immediate neighbors
        ...     weight=Normal(mean=1.0*u.nS, std=0.2*u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     periodic=False  # Cortex has edges, not periodic
        ... )
        >>>
        >>> # 50×50 grid representing a cortical microcircuit
        >>> result = cortex_grid(pre_size=(50, 50), post_size=(50, 50))
        >>>
        >>> # Check edge vs. interior connectivity
        >>> # Interior neurons have 8 connections, edge neurons have fewer

    Retinal ganglion cell lattice with periodic boundaries:

    .. code-block:: python

        >>> # Model retinotopic connectivity
        >>> retina_grid = Grid2d(
        ...     connectivity='von_neumann',
        ...     weight=Constant(0.5 * u.nS),
        ...     periodic=True  # Periodic for theoretical modeling
        ... )
        >>> result = retina_grid(pre_size=(30, 40), post_size=(30, 40))
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,

        connectivity: str = 'von_neumann',
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        periodic: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.connectivity = connectivity
        self.weight_init = weight
        self.delay_init = delay
        self.periodic = periodic

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate grid connectivity."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        if pre_size != post_size:
            raise ValueError("Grid2d networks require pre_size == post_size")

        if not isinstance(pre_size, (tuple, list)):
            raise ValueError("Grid2d networks require pre_size == post_size")
        if len(pre_size) != 2:
            raise ValueError("Grid2d networks require pre_size and post_size to be 2D shapes")

        rows, cols = pre_size

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
        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None)
        )
        delays = param(
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
                'grid_shape': pre_size,
                'connectivity': self.connectivity,
                'periodic': self.periodic
            }
        )


class RadialPatches(PointConnectivity):
    """Radial patch connectivity where neurons connect within multiple localized spatial patches.

    This class creates connections by forming multiple circular patches of connectivity for each
    presynaptic neuron. For each presynaptic neuron, the algorithm randomly selects patch centers
    from the postsynaptic population, then connects to all neurons within a specified radius of
    each center with a given probability. This pattern is useful for modeling clustered or
    patchy connectivity observed in cortical networks, such as long-range horizontal connections
    in visual cortex or patchy connectivity in association areas.

    The connection generation process:

    1. For each presynaptic neuron, randomly select ``n_patches`` centers from the postsynaptic population
    2. For each patch center, identify all postsynaptic neurons within ``patch_radius``
    3. Connect to each candidate neuron with probability ``prob``
    4. Remove duplicate connections (a neuron may appear in multiple patches)

    Parameters
    ----------
    patch_radius : float or Quantity
        Radius of each circular patch. Neurons within this distance from a patch center
        are candidates for connection. Should have spatial units (e.g., ``50 * u.um``)
        if positions have units, or be a scalar if positions are unitless.
    n_patches : int, default=1
        Number of random patches to create for each presynaptic neuron. For example:

        - ``n_patches=1``: Single patch per neuron
        - ``n_patches=3``: Three distinct patches per neuron
        - Larger values create more distributed connectivity patterns

        If ``n_patches`` exceeds the postsynaptic population size, it is automatically
        limited to the population size.
    prob : float, default=1.0
        Connection probability within each patch (range: 0.0 to 1.0). For example:

        - ``prob=1.0``: Connect to all neurons within patch radius (deterministic)
        - ``prob=0.5``: Connect to each neuron with 50% probability (stochastic)
        - ``prob=0.1``: Sparse connectivity within patches

        This enables control over connection density within patches.
    weight : Initializer, optional
        Initializer for synaptic weights. Supports:

        - Constant values (e.g., ``1.0 * u.nS``)
        - Stochastic initializers (e.g., ``Normal``, ``Uniform``, ``LogNormal``)
        - Distance-dependent weight distributions
        - ``None`` for unweighted connections
    delay : Initializer, optional
        Initializer for synaptic transmission delays. Supports:

        - Constant values (e.g., ``1.0 * u.ms``)
        - Stochastic delay distributions
        - Distance-proportional delays
        - ``None`` for connections without explicit delays
    **kwargs
        Additional arguments passed to ``PointConnectivity``, such as ``seed`` for
        reproducible random number generation.

    Returns
    -------
    ConnectionResult
        Connection result containing pre/post indices, weights, delays, and metadata
        about the radial patch connectivity pattern.

    Raises
    ------
    ValueError
        If ``pre_positions`` or ``post_positions`` are not provided when generating connections.

    Notes
    -----
    - This connectivity pattern requires neuron positions via ``pre_positions`` and
      ``post_positions`` arguments when calling the connector
    - Position arrays should have shape ``(n_neurons, n_dimensions)`` with consistent units
    - Patch centers are selected randomly without replacement for each presynaptic neuron
    - Duplicate connections (from overlapping patches) are automatically removed
    - The actual number of connections is stochastic and depends on:

      - Spatial distribution of neurons
      - Number and radius of patches
      - Connection probability ``prob``
      - Random patch center selection

    - Empty connection results are returned if no connections are established
    - This pattern models the patchy horizontal connectivity observed in cortical circuits

    See Also
    --------
    DistanceDependent : General distance-based connectivity
    Gaussian : Gaussian distance-dependent connectivity
    Exponential : Exponential distance-dependent connectivity

    Examples
    --------
    Basic radial patch connectivity:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import RadialPatches
        >>> from braintools.init import Constant
        >>>
        >>> # Create random 2D positions
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>>
        >>> # Three patches per neuron with 50um radius
        >>> patches = RadialPatches(
        ...     patch_radius=50 * u.um,
        ...     n_patches=3,
        ...     prob=0.5,
        ...     weight=Constant(1.0 * u.nS),
        ...     seed=42
        ... )
        >>>
        >>> result = patches(
        ...     pre_size=500,
        ...     post_size=500,
        ...     pre_positions=positions,
        ...     post_positions=positions
        ... )

    Dense patches with deterministic connectivity:

    .. code-block:: python

        >>> # Connect to all neurons within patch radius
        >>> dense_patches = RadialPatches(
        ...     patch_radius=75 * u.um,
        ...     n_patches=2,
        ...     prob=1.0,  # Deterministic connectivity
        ...     weight=Constant(1.5 * u.nS),
        ...     delay=Constant(1.0 * u.ms)
        ... )

    Sparse, distributed patches:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>>
        >>> # Many small, sparse patches
        >>> sparse_patches = RadialPatches(
        ...     patch_radius=30 * u.um,
        ...     n_patches=5,  # More patches
        ...     prob=0.2,  # Sparse within each patch
        ...     weight=Normal(mean=2.0*u.nS, std=0.4*u.nS),
        ...     seed=42
        ... )

    Modeling cortical horizontal connections:

    .. code-block:: python

        >>> # Model patchy long-range connections in visual cortex
        >>> # Neurons at similar locations
        >>> local_positions = np.random.randn(300, 2) * 100 * u.um
        >>>
        >>> # Create patchy connectivity pattern
        >>> horizontal_conn = RadialPatches(
        ...     patch_radius=80 * u.um,  # Size of patches
        ...     n_patches=4,  # Multiple patches per neuron
        ...     prob=0.6,  # Moderate connection probability
        ...     weight=Normal(mean=1.0*u.nS, std=0.2*u.nS),
        ...     delay=Constant(2.0 * u.ms),  # Longer delay for horizontal connections
        ...     seed=42
        ... )
        >>>
        >>> result = horizontal_conn(
        ...     pre_size=300,
        ...     post_size=300,
        ...     pre_positions=local_positions,
        ...     post_positions=local_positions
        ... )

    Projections between different populations:

    .. code-block:: python

        >>> # Different source and target populations
        >>> source_positions = np.random.uniform(0, 500, (200, 2)) * u.um
        >>> target_positions = np.random.uniform(500, 1000, (300, 2)) * u.um
        >>>
        >>> # Cross-population patchy connectivity
        >>> projection = RadialPatches(
        ...     patch_radius=60 * u.um,
        ...     n_patches=3,
        ...     prob=0.4,
        ...     weight=Constant(0.8 * u.nS)
        ... )
        >>>
        >>> result = projection(
        ...     pre_size=200,
        ...     post_size=300,
        ...     pre_positions=source_positions,
        ...     post_positions=target_positions
        ... )

    3D spatial connectivity:

    .. code-block:: python

        >>> # Patches in 3D space (e.g., cortical columns)
        >>> positions_3d = np.random.randn(400, 3) * np.array([100, 100, 200]) * u.um
        >>>
        >>> patches_3d = RadialPatches(
        ...     patch_radius=100 * u.um,  # Spherical patches in 3D
        ...     n_patches=2,
        ...     prob=0.7,
        ...     weight=Constant(1.2 * u.nS)
        ... )
        >>>
        >>> result = patches_3d(
        ...     pre_size=400,
        ...     post_size=400,
        ...     pre_positions=positions_3d,
        ...     post_positions=positions_3d
        ... )
    """
    __module__ = 'braintools.conn'

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
        radius_val = u.Quantity(self.patch_radius).to(pos_unit).mantissa

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

        weights = param(
            self.weight_init,
            n_connections,
            rng=self.rng,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )
        delays = param(
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
