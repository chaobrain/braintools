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

from typing import Optional, Union

import brainunit as u
import numpy as np
from scipy.spatial.distance import cdist

from braintools.init._init_base import init_call, Initializer
from ._base import PointConnectivity, ConnectionResult

__all__ = [
    'SmallWorld',
    'ScaleFree',
    'Regular',
    'RandomModular',
    'ModularPattern',
    'ClusteredRandom',
]


class SmallWorld(PointConnectivity):
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
                'pattern': 'small_world',
                'k': self.k,
                'p': self.p
            }
        )


class ScaleFree(PointConnectivity):
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
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata={
                'pattern': 'scale_free',
                'm': self.m,
            }
        )


class Regular(PointConnectivity):
    """
    Regular network where all neurons have the same degree.

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
            pre_indices=pre_indices,
            post_indices=post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata={
                'pattern': 'regular',
                'degree': self.degree
            }
        )


class RandomModular(PointConnectivity):
    """Modular network with intra-module and inter-module random connectivity.

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

        >>> mod = RandomModular(n_modules=5, intra_prob=0.3, inter_prob=0.01)
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
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None),
                model_type='point'
            )

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
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata={
                'pattern': 'modular',
                'n_modules': self.n_modules,
                'intra_prob': self.intra_prob,
                'inter_prob': self.inter_prob
            }
        )


class ModularPattern(PointConnectivity):
    """Modular network using a Connectivity instance for intra-module patterns.

    This class creates a modular network structure where:
    - Intra-module connectivity is defined by a Connectivity instance or list of instances
    - Inter-module connectivity is defined by probability and optional weight/delay

    Parameters
    ----------
    n_modules : int
        Number of modules in the network.
    intra_conn : PointConnectivity or list of PointConnectivity
        Connectivity instance(s) for generating connections within each module.
        If a single instance, the same connectivity pattern is used for all modules.
        If a list, must have length equal to n_modules, with one connectivity per module.
    inter_prob : float
        Connection probability between different modules.
    inter_weight : Initializer, optional
        Weight initialization for inter-module connections.
        If None, uses the same initialization as intra_conn.
    inter_delay : Initializer, optional
        Delay initialization for inter-module connections.
        If None, uses the same initialization as intra_conn.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.conn import Random, ModularPattern
        >>> import brainunit as u
        >>> # Create intra-module connectivity with 30% connection probability
        >>> intra = Random(prob=0.3, weight=1.0 * u.nS)
        >>> # Create modular network with 5 modules and 1% inter-module probability
        >>> mod = ModularPattern(
        ...     n_modules=5,
        ...     intra_conn=intra,
        ...     inter_prob=0.01,
        ...     inter_weight=0.1 * u.nS  # Weaker inter-module connections
        ... )
        >>> result = mod(pre_size=1000, post_size=1000)

        >>> # Different connectivity per module
        >>> intra_list = [
        ...     Random(prob=0.3, weight=1.0 * u.nS),
        ...     Random(prob=0.5, weight=1.5 * u.nS),
        ...     SmallWorld(k=6, p=0.3, weight=2.0 * u.nS)
        ... ]
        >>> mod = ModularPattern(n_modules=3, intra_conn=intra_list, inter_prob=0.01)
        >>> result = mod(pre_size=900, post_size=900)
    """

    def __init__(
        self,
        n_modules: int,
        intra_conn: Union[PointConnectivity, list[PointConnectivity]],
        inter_prob: float = 0.01,
        inter_weight: Optional[Initializer] = None,
        inter_delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_modules = n_modules

        # Handle both single connectivity and list of connectivities
        if isinstance(intra_conn, list):
            if len(intra_conn) != n_modules:
                raise ValueError(f"Length of intra_conn list ({len(intra_conn)}) must equal n_modules ({n_modules})")
            self.intra_conn = intra_conn
        else:
            self.intra_conn = [intra_conn] * n_modules

        self.inter_prob = inter_prob
        self.inter_weight_init = inter_weight
        self.inter_delay_init = inter_delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate modular network with custom intra-module connectivity."""
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

        all_pre_indices = []
        all_post_indices = []
        all_weights = []
        all_delays = []

        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)

        # Generate intra-module connections using the provided connectivity
        for mod_id in range(self.n_modules):
            module_neurons = np.where(modules == mod_id)[0]
            mod_n = len(module_neurons)

            if mod_n == 0:
                continue

            # Get the connectivity instance for this module
            conn = self.intra_conn[mod_id]

            # Set the RNG for the intra_conn to match this instance
            conn.rng = self.rng

            # Generate connections within this module
            intra_result = conn(
                pre_size=mod_n,
                post_size=mod_n,
                pre_positions=pre_positions[module_neurons] if pre_positions is not None else None,
                post_positions=post_positions[module_neurons] if post_positions is not None else None
            )

            # Map local indices to global indices
            if len(intra_result.pre_indices) > 0:
                global_pre = module_neurons[intra_result.pre_indices]
                global_post = module_neurons[intra_result.post_indices]
                all_pre_indices.append(global_pre)
                all_post_indices.append(global_post)

                if intra_result.weights is not None:
                    all_weights.append(intra_result.weights)
                if intra_result.delays is not None:
                    all_delays.append(intra_result.delays)

        # Generate inter-module connections
        inter_pre = []
        inter_post = []

        for i in range(n):
            for j in range(n):
                if i == j or modules[i] == modules[j]:
                    continue

                if self.rng.random() < self.inter_prob:
                    inter_pre.append(i)
                    inter_post.append(j)

        if len(inter_pre) > 0:
            inter_pre = np.array(inter_pre, dtype=np.int64)
            inter_post = np.array(inter_post, dtype=np.int64)
            n_inter = len(inter_pre)

            all_pre_indices.append(inter_pre)
            all_post_indices.append(inter_post)

            # Generate inter-module weights and delays
            inter_weights = init_call(
                self.inter_weight_init,
                n_inter,
                rng=self.rng,
                param_type='weight',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None)
            )
            inter_delays = init_call(
                self.inter_delay_init,
                n_inter,
                rng=self.rng,
                param_type='delay',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None)
            )

            if inter_weights is not None:
                all_weights.append(inter_weights)
            if inter_delays is not None:
                all_delays.append(inter_delays)

        # Combine all connections
        if len(all_pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None),
                model_type='point'
            )

        final_pre = np.concatenate(all_pre_indices)
        final_post = np.concatenate(all_post_indices)
        final_weights = np.concatenate(all_weights) if len(all_weights) > 0 else None
        final_delays = np.concatenate(all_delays) if len(all_delays) > 0 else None

        return ConnectionResult(
            final_pre,
            final_post,
            pre_size=pre_size,
            post_size=post_size,
            weights=final_weights,
            delays=final_delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata={
                'pattern': 'modular_pattern',
                'n_modules': self.n_modules,
                'inter_prob': self.inter_prob,
                'intra_conn': [type(conn).__name__ for conn in self.intra_conn]
            }
        )


class ClusteredRandom(PointConnectivity):
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
            metadata={'pattern': 'clustered_random', 'prob': self.prob, 'cluster_radius': self.cluster_radius}
        )
