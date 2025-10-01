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

from typing import Optional, Tuple, Union, Callable, Dict, Any

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike
from scipy.spatial.distance import cdist

from ._base import PointNeuronConnectivity, ConnectionResult
from braintools.init._init_base import init_call, Initializer
from braintools.init._distance import DistanceProfile

__all__ = [
    # Basic connectivity patterns
    'Random',
    'AllToAll',
    'OneToOne',
    'FixedProbability',

    # Spatial patterns
    'DistanceDependent',
    'Gaussian',
    'Exponential',
    'Ring',
    'Grid',
    'RadialPatches',

    # Topological patterns
    'SmallWorld',
    'ScaleFree',
    'Regular',
    'Modular',
    'ClusteredRandom',

    # Biological patterns
    'ExcitatoryInhibitory',
    'SynapticPlasticity',
    'ActivityDependent',

    # Custom patterns
    'Custom',
]


class Random(PointNeuronConnectivity):
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
        >>> from braintools.init import Constant, Constant
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
        weights = init_call(
            self.weight_init,
            self.rng,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions
        )

        # Generate delays using the initialization class
        delays = init_call(
            self.delay_init,
            self.rng,
            n_connections,
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
                'pattern': 'random',
                'probability': self.prob,
                'allow_self_connections': self.allow_self_connections,
                'weight_initialization': self.weight_init,
                'delay_initialization': self.delay_init,
            }
        )


class AllToAll(PointNeuronConnectivity):
    """Fully connected network where every neuron connects to every other neuron.

    Parameters
    ----------
    include_self_connections : bool
        Whether neurons connect to themselves.
    weight : Initialization, optional
        Weight initialization for all connections.
        If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for all connections.
        If None, no delays are generated.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.init import Constant
        >>> all_to_all = AllToAll(
        ...     weight=Constant(0.5 * u.nS),
        ...     delay=Constant(1.0 * u.ms)
        ... )
        >>> result = all_to_all(pre_size=50, post_size=50)
    """

    def __init__(
        self,
        include_self_connections: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.include_self_connections = include_self_connections
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate all-to-all connections."""
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

        # Vectorized generation using meshgrid
        pre_grid, post_grid = np.meshgrid(np.arange(pre_num), np.arange(post_num), indexing='ij')
        pre_indices = pre_grid.flatten()
        post_indices = post_grid.flatten()

        # Remove self-connections if needed
        if not self.include_self_connections and pre_num == post_num:
            mask = pre_indices != post_indices
            pre_indices = pre_indices[mask]
            post_indices = post_indices[mask]

        n_connections = len(pre_indices)

        # Generate weights and delays using initialization classes
        weights = init_call(
            self.weight_init,
            self.rng,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init,
            self.rng,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            model_type='point',
            metadata={
                'pattern': 'all_to_all',
                'include_self_connections': self.include_self_connections,
                'weight_initialization': self.weight_init,
                'delay_initialization': self.delay_init
            }
        )


class OneToOne(PointNeuronConnectivity):
    """One-to-one connectivity where neuron i connects to neuron i.

    Parameters
    ----------
    weight : Initialization, optional
        Weight initialization for each connection.
        If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for each connection.
        If None, no delays are generated.
    circular : bool
        If True and sizes differ, use circular indexing.

    Examples
    --------
    .. code-block:: python

        >>> one_to_one = OneToOne(weight=1.5 * u.nS)
        >>> result = one_to_one(pre_size=100, post_size=100)
    """

    def __init__(
        self,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        circular: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.weight_init = weight
        self.delay_init = delay
        self.circular = circular

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate one-to-one connections."""
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

        if self.circular:
            n_connections = max(pre_num, post_num)
            pre_indices = np.arange(n_connections) % pre_num
            post_indices = np.arange(n_connections) % post_num
        else:
            n_connections = min(pre_num, post_num)
            pre_indices = np.arange(n_connections)
            post_indices = np.arange(n_connections)

        # Generate weights and delays using initialization classes
        weights = init_call(
            self.weight_init,
            self.rng,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init,
            self.rng,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            model_type='point',
            metadata={'pattern': 'one_to_one', 'circular': self.circular}
        )


class DistanceDependent(PointNeuronConnectivity):
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
        >>> from braintools.init import GaussianProfile, ExponentialDecay, Constant
        >>>
        >>> # Gaussian distance-dependent connectivity
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> conn = DistanceDependent(
        ...     distance_profile=GaussianProfile(
        ...         sigma=100 * u.um,
        ...         max_distance=300 * u.um
        ...     ),
        ...     weight=ExponentialDecay(
        ...         max_weight=3.0 * u.nS,
        ...         decay_constant=80 * u.um
        ...     ),
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
        pre_positions = kwargs.get('pre_positions')
        post_positions = kwargs.get('post_positions')

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
            self.rng,
            n_connections,
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
            self.rng,
            n_connections,
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


class SmallWorld(PointNeuronConnectivity):
    """Watts-Strogatz small-world network topology.

    Parameters
    ----------
    k : int
        Number of nearest neighbors each node connects to.
    p : float
        Probability of rewiring each edge.
    weight : Initialization, optional
        Weight initialization for each connection.
        If None, no weights are generated.
    delay : Initialization, optional
        Delay initialization for each connection.
        If None, no delays are generated.

    Examples
    --------
    .. code-block:: python

        >>> sw = SmallWorld(k=6, p=0.3, weight=0.8 * u.nS)
        >>> result = sw(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        k: int = 6,
        p: float = 0.3,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.p = p
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate small-world network connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if pre_size != post_size:
            raise ValueError("Small-world networks require pre_size == post_size")

        # Vectorized generation of regular ring lattice
        k_half = self.k // 2
        sources = np.repeat(np.arange(n), k_half * 2)

        # Create offsets for forward and backward connections
        offsets = np.tile(np.concatenate([np.arange(1, k_half + 1), -np.arange(1, k_half + 1)]), n)
        targets = (sources + offsets) % n

        # Vectorized rewiring
        rewire_mask = self.rng.random(len(sources)) < self.p
        n_rewire = np.sum(rewire_mask)

        if n_rewire > 0:
            # Generate random targets for rewiring
            new_targets = self.rng.randint(0, n, size=n_rewire)

            # Avoid self-connections in rewired edges
            self_conn_mask = new_targets == sources[rewire_mask]
            while np.any(self_conn_mask):
                new_targets[self_conn_mask] = self.rng.randint(0, n, size=np.sum(self_conn_mask))
                self_conn_mask = new_targets == sources[rewire_mask]

            targets[rewire_mask] = new_targets

        pre_indices = sources
        post_indices = targets
        n_connections = len(pre_indices)

        # Generate weights and delays using initialization classes
        weights = init_call(
            self.weight_init,
            self.rng,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init,
            self.rng,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={'pattern': 'small_world', 'k': self.k, 'p': self.p}
        )


class ExcitatoryInhibitory(PointNeuronConnectivity):
    """Standard excitatory-inhibitory network following Dale's principle.

    Parameters
    ----------
    exc_ratio : float
        Fraction of neurons that are excitatory.
    exc_prob : float
        Connection probability from excitatory neurons.
    inh_prob : float
        Connection probability from inhibitory neurons.
    exc_weight : Initialization, optional
        Weight initialization for excitatory connections.
        If None, no excitatory weights are generated.
    inh_weight : Initialization, optional
        Weight initialization for inhibitory connections.
        If None, no inhibitory weights are generated.
    delay : Initialization, optional
        Delay initialization for all connections.
        If None, no delays are generated.

    Examples
    --------
    .. code-block:: python

        >>> ei_net = ExcitatoryInhibitory(
        ...     exc_ratio=0.8,
        ...     exc_prob=0.1,
        ...     inh_prob=0.2,
        ...     exc_weight=1.0 * u.nS,
        ...     inh_weight=-0.8 * u.nS
        ... )
        >>> result = ei_net(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        exc_ratio: float = 0.8,
        exc_prob: float = 0.1,
        inh_prob: float = 0.2,
        exc_weight: Optional[Initializer] = None,
        inh_weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.exc_ratio = exc_ratio
        self.exc_prob = exc_prob
        self.inh_prob = inh_prob
        self.exc_weight_init = exc_weight
        self.inh_weight_init = inh_weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate excitatory-inhibitory network."""
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

        # Determine which neurons are excitatory vs inhibitory
        n_exc = int(pre_num * self.exc_ratio)

        # Vectorized generation for excitatory connections
        exc_random = self.rng.random((n_exc, post_num))
        exc_mask = exc_random < self.exc_prob
        exc_pre, exc_post = np.where(exc_mask)

        # Vectorized generation for inhibitory connections
        n_inh = pre_num - n_exc
        inh_random = self.rng.random((n_inh, post_num))
        inh_mask = inh_random < self.inh_prob
        inh_pre, inh_post = np.where(inh_mask)
        inh_pre = inh_pre + n_exc  # Offset to correct neuron indices

        # Combine excitatory and inhibitory connections
        pre_indices = np.concatenate([exc_pre, inh_pre])
        post_indices = np.concatenate([exc_post, inh_post])
        is_excitatory = np.concatenate([np.ones(len(exc_pre), dtype=bool), np.zeros(len(inh_pre), dtype=bool)])

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                model_type='point',
                pre_positions=kwargs.get('pre_positions'),
                post_positions=kwargs.get('post_positions'),
            )

        n_connections = len(pre_indices)
        n_exc_conn = len(exc_pre)
        n_inh_conn = len(inh_pre)

        # Generate weights separately for excitatory and inhibitory
        exc_weights = init_call(
            self.exc_weight_init,
            self.rng,
            n_exc_conn,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        ) if n_exc_conn > 0 else None

        inh_weights = init_call(
            self.inh_weight_init,
            self.rng,
            n_inh_conn,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        ) if n_inh_conn > 0 else None

        # Combine weights in correct order
        weights = None
        if exc_weights is not None or inh_weights is not None:
            # Handle scalar and array weights
            if exc_weights is not None:
                if u.math.isscalar(exc_weights):
                    exc_weights_array = np.full(
                        n_exc_conn,
                        u.get_mantissa(exc_weights) if isinstance(exc_weights, u.Quantity) else exc_weights
                    )
                    exc_unit = u.get_unit(exc_weights) if isinstance(exc_weights, u.Quantity) else None
                else:
                    exc_weights_array = (
                        u.get_mantissa(exc_weights)
                        if isinstance(exc_weights, u.Quantity) else
                        np.asarray(exc_weights)
                    )
                    exc_unit = u.get_unit(exc_weights) if isinstance(exc_weights, u.Quantity) else None
            else:
                exc_weights_array = np.zeros(n_exc_conn)
                exc_unit = None

            if inh_weights is not None:
                if u.math.isscalar(inh_weights):
                    inh_weights_array = np.full(
                        n_inh_conn,
                        u.get_mantissa(inh_weights) if isinstance(inh_weights, u.Quantity) else inh_weights
                    )
                    inh_unit = u.get_unit(inh_weights) if isinstance(inh_weights, u.Quantity) else None
                else:
                    inh_weights_array = (
                        u.get_mantissa(inh_weights)
                        if isinstance(inh_weights, u.Quantity) else np.asarray(inh_weights)
                    )
                    inh_unit = u.get_unit(inh_weights) if isinstance(inh_weights, u.Quantity) else None
            else:
                inh_weights_array = np.zeros(n_inh_conn)
                inh_unit = None

            # Concatenate weights
            weights_array = np.concatenate([exc_weights_array, inh_weights_array])
            common_unit = exc_unit or inh_unit

            if common_unit is not None:
                weights = u.maybe_decimal(weights_array * common_unit)
            else:
                weights = weights_array

        # Generate delays
        delays = init_call(
            self.delay_init,
            self.rng,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=weights,
            delays=delays,
            pre_size=pre_size,
            post_size=post_size,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={
                'pattern': 'excitatory_inhibitory',
                'exc_ratio': self.exc_ratio,
                'n_excitatory': n_exc,
                'n_inhibitory': n_inh,
            }
        )


class Custom(PointNeuronConnectivity):
    """Custom connectivity pattern using user-defined function.

    Parameters
    ----------
    connection_func : callable
        Function that generates connections. Should accept (pre_size, post_size, rng, **kwargs)
        and return (pre_indices, post_indices, weights, delays).

    Examples
    --------
    .. code-block:: python

        >>> def my_pattern(pre_size, post_size, rng, **kwargs):
        ...     # Custom logic here
        ...     pre_indices = [...]
        ...     post_indices = [...]
        ...     weights = [...]
        ...     delays = [...]
        ...     return pre_indices, post_indices, weights, delays
        >>>
        >>> custom_conn = Custom(my_pattern)
        >>> result = custom_conn(pre_size=100, post_size=100)
    """

    def __init__(self, connection_func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.connection_func = connection_func

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate custom connections."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        pre_indices, post_indices, weights, delays = self.connection_func(
            rng=self.rng, **kwargs
        )

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=u.math.array(weights) if weights is not None else None,
            delays=delays,
            pre_size=pre_size,
            post_size=post_size,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={'pattern': 'custom'}
        )


# Convenience aliases for common patterns
class FixedProbability(Random):
    """Alias for Random connectivity with fixed probability."""
    pass


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


class Ring(PointNeuronConnectivity):
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
            self.rng,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init,
            self.rng,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={
                'pattern': 'ring',
                'neighbors': self.neighbors,
                'bidirectional': self.bidirectional
            }
        )


class Grid(PointNeuronConnectivity):
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
            self.rng,
            n_connections,
            param_type='weight',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init,
            self.rng,
            n_connections,
            param_type='delay',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={
                'pattern': 'grid',
                'grid_shape': self.grid_shape,
                'connectivity': self.connectivity,
                'periodic': self.periodic
            }
        )


class RadialPatches(PointNeuronConnectivity):
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
        pre_positions = kwargs.get('pre_positions')
        post_positions = kwargs.get('post_positions')

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
        distances = cdist(pre_pos_val, post_pos_val)

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
            self.weight_init, self.rng, n_connections,
            param_type='weight', pre_size=pre_size, post_size=post_size,
            pre_positions=pre_positions, post_positions=post_positions
        )
        delays = init_call(
            self.delay_init, self.rng, n_connections,
            param_type='delay', pre_size=pre_size, post_size=post_size,
            pre_positions=pre_positions, post_positions=post_positions
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
            metadata={'pattern': 'radial_patches', 'patch_radius': self.patch_radius, 'n_patches': self.n_patches}
        )


class ScaleFree(PointNeuronConnectivity):
    """Barabási-Albert scale-free network with preferential attachment.

    Parameters
    ----------
    m : int
        Number of edges to attach from a new node to existing nodes.
    weight : Initialization, optional
        Weight initialization.
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> sf = ScaleFree(m=3, weight=1.0 * u.nS)
        >>> result = sf(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        m: int = 3,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.m = m
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate scale-free network using preferential attachment."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if pre_size != post_size:
            raise ValueError("Scale-free networks require pre_size == post_size")

        # Start with a small complete graph
        m0 = max(self.m, 2)
        pre_indices = []
        post_indices = []

        # Initial complete graph
        for i in range(m0):
            for j in range(i + 1, m0):
                pre_indices.extend([i, j])
                post_indices.extend([j, i])

        # Track degree for preferential attachment
        degree = np.zeros(n, dtype=np.int64)
        degree[:m0] = 2 * (m0 - 1)

        # Add remaining nodes with preferential attachment
        for new_node in range(m0, n):
            # Probability proportional to degree
            prob = degree[:new_node] / np.sum(degree[:new_node])

            # Select m targets without replacement
            targets = self.rng.choice(new_node, size=min(self.m, new_node), replace=False, p=prob)

            # Add bidirectional connections
            for target in targets:
                pre_indices.extend([new_node, target])
                post_indices.extend([target, new_node])
                degree[new_node] += 1
                degree[target] += 1

        pre_indices = np.array(pre_indices, dtype=np.int64)
        post_indices = np.array(post_indices, dtype=np.int64)
        n_connections = len(pre_indices)

        weights = init_call(
            self.weight_init, self.rng, n_connections,
            param_type='weight', pre_size=pre_size, post_size=post_size,
            pre_positions=kwargs.get('pre_positions'), post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init, self.rng, n_connections,
            param_type='delay', pre_size=pre_size, post_size=post_size,
            pre_positions=kwargs.get('pre_positions'), post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={'pattern': 'scale_free', 'm': self.m}
        )


class Regular(PointNeuronConnectivity):
    """Regular network where all neurons have the same degree.

    Parameters
    ----------
    degree : int
        Number of connections per neuron.
    weight : Initialization, optional
        Weight initialization.
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> reg = Regular(degree=10, weight=1.0 * u.nS)
        >>> result = reg(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        degree: int,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.degree = degree
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate regular network."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if pre_size != post_size:
            raise ValueError("Regular networks require pre_size == post_size")

        # Each neuron connects to 'degree' random targets
        pre_indices = np.repeat(np.arange(n), self.degree)
        post_indices = np.zeros(n * self.degree, dtype=np.int64)

        for i in range(n):
            # Select random targets excluding self
            targets = self.rng.choice(n - 1, size=self.degree, replace=False)
            targets = np.where(targets >= i, targets + 1, targets)  # Adjust for excluded self
            post_indices[i * self.degree:(i + 1) * self.degree] = targets

        n_connections = len(pre_indices)

        weights = init_call(
            self.weight_init, self.rng, n_connections,
            param_type='weight', pre_size=pre_size, post_size=post_size,
            pre_positions=kwargs.get('pre_positions'), post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init, self.rng, n_connections,
            param_type='delay', pre_size=pre_size, post_size=post_size,
            pre_positions=kwargs.get('pre_positions'), post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={'pattern': 'regular', 'degree': self.degree}
        )


class Modular(PointNeuronConnectivity):
    """Modular network with intra-module and inter-module connectivity.

    Parameters
    ----------
    n_modules : int
        Number of modules.
    intra_prob : float
        Connection probability within modules.
    inter_prob : float
        Connection probability between modules.
    weight : Initialization, optional
        Weight initialization.
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> mod = Modular(n_modules=5, intra_prob=0.3, inter_prob=0.01)
        >>> result = mod(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        n_modules: int,
        intra_prob: float = 0.3,
        inter_prob: float = 0.01,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_modules = n_modules
        self.intra_prob = intra_prob
        self.inter_prob = inter_prob
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate modular network."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if pre_size != post_size:
            raise ValueError("Modular networks require pre_size == post_size")

        # Assign neurons to modules
        module_size = n // self.n_modules
        modules = np.repeat(np.arange(self.n_modules), module_size)
        if len(modules) < n:
            modules = np.concatenate([modules, np.full(n - len(modules), self.n_modules - 1)])

        # Generate connections with module-dependent probabilities
        pre_indices = []
        post_indices = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # Determine probability based on modules
                prob = self.intra_prob if modules[i] == modules[j] else self.inter_prob

                if self.rng.random() < prob:
                    pre_indices.append(i)
                    post_indices.append(j)

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions'),
                post_positions=kwargs.get('post_positions'),
                model_type='point'
            )

        pre_indices = np.array(pre_indices, dtype=np.int64)
        post_indices = np.array(post_indices, dtype=np.int64)
        n_connections = len(pre_indices)

        weights = init_call(
            self.weight_init, self.rng, n_connections,
            param_type='weight', pre_size=pre_size, post_size=post_size,
            pre_positions=kwargs.get('pre_positions'), post_positions=kwargs.get('post_positions')
        )
        delays = init_call(
            self.delay_init, self.rng, n_connections,
            param_type='delay', pre_size=pre_size, post_size=post_size,
            pre_positions=kwargs.get('pre_positions'), post_positions=kwargs.get('post_positions')
        )

        return ConnectionResult(
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions'),
            post_positions=kwargs.get('post_positions'),
            metadata={'pattern': 'modular',
                      'n_modules': self.n_modules,
                      'intra_prob': self.intra_prob,
                      'inter_prob': self.inter_prob}
        )


class ClusteredRandom(PointNeuronConnectivity):
    """Random connectivity with spatial clustering.

    Parameters
    ----------
    prob : float
        Base connection probability.
    cluster_radius : float or Quantity
        Radius for enhanced clustering.
    cluster_factor : float
        Multiplication factor for probability within cluster radius.
    weight : Initialization, optional
        Weight initialization.
    delay : Initialization, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> clustered = ClusteredRandom(
        ...     prob=0.05,
        ...     cluster_radius=100 * u.um,
        ...     cluster_factor=5.0
        ... )
        >>> result = clustered(
        ...     pre_size=500, post_size=500,
        ...     pre_positions=positions, post_positions=positions
        ... )
    """

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
        pre_positions = kwargs.get('pre_positions')
        post_positions = kwargs.get('post_positions')

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
        if isinstance(self.cluster_radius, u.Quantity):
            radius_val = u.Quantity(self.cluster_radius).to(pos_unit).mantissa
        else:
            radius_val = self.cluster_radius

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

        weights = init_call(
            self.weight_init, self.rng, n_connections,
            param_type='weight', pre_size=pre_size, post_size=post_size,
            pre_positions=pre_positions, post_positions=post_positions
        )
        delays = init_call(
            self.delay_init, self.rng, n_connections,
            param_type='delay', pre_size=pre_size, post_size=post_size,
            pre_positions=pre_positions, post_positions=post_positions
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
            metadata={'pattern': 'clustered_random', 'prob': self.prob, 'cluster_radius': self.cluster_radius}
        )


class SynapticPlasticity(PointNeuronConnectivity):
    """Connectivity with initial weights modulated by plasticity rule metadata.

    This class stores plasticity parameters in metadata for use during simulation.

    Parameters
    ----------
    base_connectivity : PointNeuronConnectivity
        Base connectivity pattern.
    plasticity_type : str
        Type of plasticity ('stdp', 'bcm', 'oja', etc.).
    plasticity_params : dict
        Parameters for plasticity rule.

    Examples
    --------
    .. code-block:: python

        >>> base = Random(prob=0.1, weight=1.0 * u.nS)
        >>> plastic = SynapticPlasticity(
        ...     base_connectivity=base,
        ...     plasticity_type='stdp',
        ...     plasticity_params={'tau_pre': 20*u.ms, 'tau_post': 20*u.ms, 'A_plus': 0.01, 'A_minus': 0.01}
        ... )
        >>> result = plastic(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        base_connectivity: PointNeuronConnectivity,
        plasticity_type: str,
        plasticity_params: Dict[str, Any],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_connectivity = base_connectivity
        self.plasticity_type = plasticity_type
        self.plasticity_params = plasticity_params

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate connections with plasticity metadata."""
        result = self.base_connectivity.generate(**kwargs)

        # Add plasticity information to metadata
        result.metadata['plasticity_type'] = self.plasticity_type
        result.metadata['plasticity_params'] = self.plasticity_params

        return result


class ActivityDependent(PointNeuronConnectivity):
    """Activity-dependent connectivity pruning/strengthening.

    This class generates initial connectivity and stores activity-dependent
    parameters in metadata for use during simulation.

    Parameters
    ----------
    base_connectivity : PointNeuronConnectivity
        Base connectivity pattern.
    pruning_threshold : float
        Activity threshold below which connections are pruned.
    strengthening_factor : float
        Factor by which active connections are strengthened.

    Examples
    --------
    .. code-block:: python

        >>> base = Random(prob=0.2, weight=1.0 * u.nS)
        >>> activity_dep = ActivityDependent(
        ...     base_connectivity=base,
        ...     pruning_threshold=0.1,
        ...     strengthening_factor=1.5
        ... )
        >>> result = activity_dep(pre_size=1000, post_size=1000)
    """

    def __init__(
        self,
        base_connectivity: PointNeuronConnectivity,
        pruning_threshold: float = 0.1,
        strengthening_factor: float = 1.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_connectivity = base_connectivity
        self.pruning_threshold = pruning_threshold
        self.strengthening_factor = strengthening_factor

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate connections with activity-dependent metadata."""
        result = self.base_connectivity.generate(**kwargs)

        # Add activity-dependent parameters to metadata
        result.metadata['activity_dependent'] = True
        result.metadata['pruning_threshold'] = self.pruning_threshold
        result.metadata['strengthening_factor'] = self.strengthening_factor

        return result
