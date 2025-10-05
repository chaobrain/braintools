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
Point neuron connectivity patterns.

This module provides connectivity patterns specifically designed for point
neuron models (single-compartment integrate-and-fire neurons). These patterns
focus on synaptic connections between individual neurons with realistic
biological constraints and dynamics.

Key Features:
- Synaptic weight and delay modeling
- Spatial connectivity patterns
- Topological network structures
- Dale's principle enforcement
- Degree constraints and pruning
"""

from typing import Optional, Tuple, Union

import brainunit as u
import numpy as np
from scipy.spatial.distance import cdist

from braintools.init._init_base import param, Initializer
from ._base import PointConnectivity, ConnectionResult

__all__ = [
    'Random',
    'FixedProb',
    'ClusteredRandom',
]


class Random(PointConnectivity):
    """Random connectivity with fixed connection probability.

    This is the fundamental random connectivity pattern for point neurons,
    where each potential connection is made with a fixed probability.

    Parameters
    ----------
    prob : float
        Connection probability between 0 and 1.
    allow_self_connections : bool
        Whether to allow neurons to connect to themselves.
    weight : Initializer, optional
        Weight initialization. Can be:

        - Initialization class (e.g., Normal, LogNormal, Constant)
        - Scalar value (float/int, will use nS units)
        - Quantity scalar or array
        - Array-like values

        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization. Can be:

        - Initialization class (e.g., ConstantDelay, UniformDelay)
        - Scalar value (float/int, will use ms units)
        - Quantity scalar or array
        - Array-like values

        If None, no delays are generated.
    seed : int, optional
        Random seed for reproducible results.

    Examples
    --------
    Basic random connectivity:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import Random
        >>> from braintools.init import Constant
        >>>
        >>> # With weights and delays
        >>> conn = Random(
        ...     prob=0.1,
        ...     weight=Constant(2.0 * u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     seed=42
        ... )
        >>> result = conn(pre_size=1000, post_size=1000)
        >>>
        >>> # Topology only (no weights or delays)
        >>> topology_only = Random(prob=0.1, seed=42)
        >>> result = topology_only(pre_size=1000, post_size=1000)
        >>>
        >>> # Using scalar values (automatic units)
        >>> simple_conn = Random(prob=0.1, weight=2.5, delay=1.0, seed=42)
        >>> result = simple_conn(pre_size=1000, post_size=1000)

    Random with realistic synaptic weights:

    .. code-block:: python

        >>> from braintools.init import LogNormal, Normal
        >>>
        >>> # AMPA-like excitatory synapses
        >>> ampa_conn = Random(
        ...     prob=0.05,
        ...     weight=LogNormal(mean=1.0 * u.nS, std=0.5 * u.nS),
        ...     delay=Normal(mean=1.5 * u.ms, std=0.3 * u.ms)
        ... )

    Inhibitory connections with Dale's principle:

    .. code-block:: python

        >>> from braintools.init import Normal, Constant
        >>>
        >>> # GABA-like inhibitory synapses
        >>> gaba_conn = Random(
        ...     prob=0.08,
        ...     weight=Normal(mean=-0.8 * u.nS, std=0.2 * u.nS),
        ...     delay=Constant(0.8 * u.ms)
        ... )
    """

    __module__ = 'braintools.conn'

    def __init__(
        self,
        prob: float,
        allow_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prob = prob
        self.allow_self_connections = allow_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate random point neuron connections."""
        if isinstance(pre_size, (tuple, list)):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, (tuple, list)):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Generate all potential connections
        pre_indices = []
        post_indices = []
        for i in range(pre_num):
            for j in range(post_num):
                if not self.allow_self_connections and i == j:
                    continue
                if self.rng.random() < self.prob:
                    pre_indices.append(i)
                    post_indices.append(j)

        n_connections = len(pre_indices)
        if n_connections == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='point'
            )

        # Generate weights using the initialization class
        weights = param(
            self.weight_init,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
        )

        # Generate delays using the initialization class
        delays = param(
            self.delay_init,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            rng=self.rng
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
                'pattern': 'random',
                'probability': self.prob,
                'allow_self_connections': self.allow_self_connections,
                'weight_initialization': self.weight_init,
                'delay_initialization': self.delay_init,
            }
        )


# Convenience aliases for common patterns
class FixedProb(Random):
    """Alias for Random connectivity with fixed probability (:class:`Random`)."""

    __module__ = 'braintools.conn'


class ClusteredRandom(PointConnectivity):
    """Random connectivity with spatial clustering and distance-dependent connection enhancement.

    This class creates stochastic connections where the connection probability is enhanced
    for spatially neighboring neurons. The connectivity combines a baseline random probability
    with increased connectivity within a specified spatial radius, modeling the observation
    that nearby neurons in cortical circuits often exhibit higher connection probabilities
    than distant neurons. This pattern is useful for creating spatially-aware random networks
    that capture local clustering while maintaining global randomness.

    The connection probability is computed as:

    .. math::

        P(d) = \\begin{cases}
        \\min(p_{\\text{base}} \\times f_{\\text{cluster}}, 1) & \\text{if } d \\leq r_{\\text{cluster}} \\\\
        p_{\\text{base}} & \\text{if } d > r_{\\text{cluster}}
        \\end{cases}

    where:

    - :math:`d` is the distance between neurons
    - :math:`p_{\\text{base}}` is the baseline connection probability
    - :math:`r_{\\text{cluster}}` is the cluster radius
    - :math:`f_{\\text{cluster}}` is the cluster enhancement factor

    Parameters
    ----------
    prob : float
        Baseline connection probability for all neuron pairs (range: 0.0 to 1.0).
        This probability applies to neurons outside the cluster radius and serves
        as the base rate that is enhanced within clusters. For example:

        - ``prob=0.05``: 5% baseline connectivity
        - ``prob=0.1``: 10% baseline connectivity
        - ``prob=0.01``: Sparse baseline connectivity
    cluster_radius : float or Quantity
        Spatial radius within which connection probability is enhanced. Neuron pairs
        with distance ≤ ``cluster_radius`` have their connection probability multiplied
        by ``cluster_factor``. Should have spatial units (e.g., ``100 * u.um``) if
        positions have units, or be a scalar if positions are unitless.
    cluster_factor : float, default=2.0
        Multiplicative factor applied to the baseline probability within the cluster
        radius. For example:

        - ``cluster_factor=2.0``: Double the probability within clusters
        - ``cluster_factor=5.0``: Five times higher probability for nearby neurons
        - ``cluster_factor=10.0``: Strong local clustering

        The final probability is clipped to the range [0, 1], so
        ``prob × cluster_factor`` can exceed 1.0.
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
        about the clustered random connectivity pattern.

    Raises
    ------
    ValueError
        If ``pre_positions`` or ``post_positions`` are not provided when generating connections.

    Notes
    -----
    - This connectivity pattern requires neuron positions via ``pre_positions`` and
      ``post_positions`` arguments when calling the connector
    - Position arrays should have shape ``(n_neurons, n_dimensions)`` with consistent units
    - The algorithm computes all pairwise distances, which has O(N²) memory complexity
    - Connection probabilities within cluster radius are: ``min(prob × cluster_factor, 1.0)``
    - The expected number of connections depends on spatial neuron distribution:

      - For uniformly distributed neurons, denser regions will have more local connections
      - The global connection density is higher than pure random connectivity

    - Empty connection results are returned if no connections are established
    - This pattern captures the distance-dependent connectivity observed in local cortical circuits
      while maintaining stochastic variability

    See Also
    --------
    FixedProb : Simple random connectivity without spatial clustering
    DistanceDependent : General distance-based connectivity
    Gaussian : Smooth Gaussian distance-dependent connectivity
    Exponential : Exponential distance-dependent connectivity

    Examples
    --------
    Basic clustered random connectivity:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import ClusteredRandom
        >>> from braintools.init import Constant
        >>>
        >>> # Create random 2D positions
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>>
        >>> # 5% baseline, enhanced 5× within 100um
        >>> clustered = ClusteredRandom(
        ...     prob=0.05,
        ...     cluster_radius=100 * u.um,
        ...     cluster_factor=5.0,
        ...     weight=Constant(1.0 * u.nS),
        ...     seed=42
        ... )
        >>>
        >>> result = clustered(
        ...     pre_size=500,
        ...     post_size=500,
        ...     pre_positions=positions,
        ...     post_positions=positions
        ... )

    Strong local clustering:

    .. code-block:: python

        >>> # Very strong enhancement for nearby neurons
        >>> strong_cluster = ClusteredRandom(
        ...     prob=0.02,  # Sparse baseline
        ...     cluster_radius=75 * u.um,
        ...     cluster_factor=20.0,  # 20× enhancement within clusters
        ...     weight=Constant(1.5 * u.nS),
        ...     delay=Constant(1.0 * u.ms)
        ... )

    Moderate clustering with stochastic weights:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>>
        >>> moderate_cluster = ClusteredRandom(
        ...     prob=0.1,
        ...     cluster_radius=150 * u.um,
        ...     cluster_factor=3.0,
        ...     weight=Normal(mean=2.0*u.nS, std=0.4*u.nS),
        ...     delay=Constant(1.0 * u.ms),
        ...     seed=42
        ... )

    Modeling local cortical connectivity:

    .. code-block:: python

        >>> # Model cortical microcircuit with local clustering
        >>> cortical_positions = np.random.randn(400, 2) * 150 * u.um
        >>>
        >>> cortical_conn = ClusteredRandom(
        ...     prob=0.08,  # 8% baseline connectivity
        ...     cluster_radius=100 * u.um,  # Local neighborhood
        ...     cluster_factor=4.0,  # 4× higher within 100um
        ...     weight=Normal(mean=1.0*u.nS, std=0.2*u.nS),
        ...     delay=Constant(1.5 * u.ms),
        ...     seed=42
        ... )
        >>>
        >>> result = cortical_conn(
        ...     pre_size=400,
        ...     post_size=400,
        ...     pre_positions=cortical_positions,
        ...     post_positions=cortical_positions
        ... )

    Different pre and post populations:

    .. code-block:: python

        >>> # Clustered connectivity between different populations
        >>> pre_pos = np.random.uniform(0, 500, (200, 2)) * u.um
        >>> post_pos = np.random.uniform(0, 500, (300, 2)) * u.um
        >>>
        >>> inter_cluster = ClusteredRandom(
        ...     prob=0.03,
        ...     cluster_radius=80 * u.um,
        ...     cluster_factor=6.0,
        ...     weight=Constant(0.8 * u.nS)
        ... )
        >>>
        >>> result = inter_cluster(
        ...     pre_size=200,
        ...     post_size=300,
        ...     pre_positions=pre_pos,
        ...     post_positions=post_pos
        ... )

    3D spatial clustering:

    .. code-block:: python

        >>> # Clustering in 3D space (e.g., cortical volume)
        >>> positions_3d = np.random.randn(300, 3) * np.array([100, 100, 200]) * u.um
        >>>
        >>> cluster_3d = ClusteredRandom(
        ...     prob=0.04,
        ...     cluster_radius=120 * u.um,  # Spherical clustering
        ...     cluster_factor=8.0,
        ...     weight=Constant(1.2 * u.nS)
        ... )
        >>>
        >>> result = cluster_3d(
        ...     pre_size=300,
        ...     post_size=300,
        ...     pre_positions=positions_3d,
        ...     post_positions=positions_3d
        ... )

    Comparing with pure random connectivity:

    .. code-block:: python

        >>> from braintools.conn import FixedProb
        >>>
        >>> # Pure random (no clustering)
        >>> random_conn = FixedProb(prob=0.1)
        >>> random_result = random_conn(pre_size=500, post_size=500)
        >>>
        >>> # Clustered random (same baseline probability)
        >>> clustered_conn = ClusteredRandom(
        ...     prob=0.1,
        ...     cluster_radius=100 * u.um,
        ...     cluster_factor=3.0
        ... )
        >>> clustered_result = clustered_conn(
        ...     pre_size=500,
        ...     post_size=500,
        ...     pre_positions=positions,
        ...     post_positions=positions
        ... )
        >>> # clustered_result will have more connections due to spatial enhancement
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        prob: float,
        cluster_radius: Union[float, u.Quantity],
        cluster_factor: float = 2.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prob = prob
        self.cluster_radius = cluster_radius
        self.cluster_factor = cluster_factor
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate clustered random connectivity."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']
        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for clustered random connectivity")

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
        distances = cdist(pre_pos_val, post_pos_val)

        # Get radius value
        radius_val = u.Quantity(self.cluster_radius).to(pos_unit).mantissa

        # Calculate connection probabilities
        probs = np.full((pre_num, post_num), self.prob)
        within_cluster = distances <= radius_val
        probs[within_cluster] *= self.cluster_factor
        probs = np.clip(probs, 0, 1)

        # Vectorized connection generation
        random_vals = self.rng.random((pre_num, post_num))
        connection_mask = random_vals < probs

        pre_indices, post_indices = np.where(connection_mask)

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
                'pattern': 'clustered_random',
                'prob': self.prob,
                'cluster_radius': self.cluster_radius
            }
        )
