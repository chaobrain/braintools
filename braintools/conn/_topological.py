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

import numbers
from typing import Optional, Dict, Sequence, List, Tuple

import brainunit as u
import numpy as np

from braintools.init._init_base import param, Initializer
from ._base import PointConnectivity, ConnectionResult
from ._sampling import bernoulli_pairs, randint, sample_rows_without_replacement

__all__ = [
    'SmallWorld',
    'ScaleFree',
    'Regular',
    'ModularRandom',
    'ModularGeneral',
    'HierarchicalRandom',
    'CorePeripheryRandom',
]


def _combine_block_params(blocks):
    """Combine per-block weights/delays into a single per-edge array.

    Each entry in ``blocks`` is a ``(n_edges, value)`` tuple where ``value`` is a
    scalar (e.g. ``1.0 * u.nS``), a per-edge array, or ``None``. Returns ``None`` if
    every block lacks the parameter; otherwise each value is broadcast to its block's
    edge count (missing blocks filled with 0) and concatenated, unit-aware.
    """
    if not blocks or all(value is None for _, value in blocks):
        return None

    # Determine the unit from the first block that provides one so all blocks share it.
    unit = None
    for _, value in blocks:
        if value is not None:
            unit = u.get_unit(value)
            break

    mantissas = []
    for n_edges, value in blocks:
        if value is None:
            mantissas.append(np.zeros(n_edges))
            continue
        mantissa = u.Quantity(value).to(unit).mantissa if unit is not None else u.get_mantissa(value)
        mantissa = np.asarray(mantissa)
        if mantissa.ndim == 0:
            mantissa = np.full(n_edges, mantissa)
        mantissas.append(mantissa)

    combined = np.concatenate(mantissas)
    if unit is None:
        return combined
    # ``maybe_decimal`` returns a plain array for a dimensionless unit.
    return u.maybe_decimal(combined * unit)


def _check_prob(value, name: str) -> float:
    """Validate a connection probability in ``[0, 1]``."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f'{name} must be in [0, 1], got {value}.')
    return value


def _block_bernoulli(
    rng: np.random.RandomState,
    blocks: Sequence[Tuple[int, int, int, int, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample independent Bernoulli connections block by block.

    Each block is ``(pre_start, pre_stop, post_start, post_stop, p)`` in a
    square (recurrent) index space; pairs ``(i, i)`` are never drawn. Every
    block draws its edge count from a binomial and then that many distinct
    pairs (:func:`~braintools.conn._sampling.bernoulli_pairs`), so nothing of
    size ``N_pre * N_post`` is built. The result is sorted by ``(pre, post)``.
    """
    pre_parts: List[np.ndarray] = []
    post_parts: List[np.ndarray] = []
    for pre_start, pre_stop, post_start, post_stop, p in blocks:
        n_pre, n_post = pre_stop - pre_start, post_stop - post_start
        if n_pre <= 0 or n_post <= 0 or p <= 0:
            continue
        diagonal = pre_start == post_start and pre_stop == post_stop
        pre, post = bernoulli_pairs(rng, n_pre, n_post, p, allow_self_connections=not diagonal)
        pre_parts.append(pre + pre_start)
        post_parts.append(post + post_start)
    return _sorted_pairs(pre_parts, post_parts)


def _sorted_pairs(pre_parts, post_parts) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate edge lists and sort them by ``(pre, post)``."""
    if not pre_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pre = np.concatenate(pre_parts).astype(np.int64, copy=False)
    post = np.concatenate(post_parts).astype(np.int64, copy=False)
    order = np.lexsort((post, pre))
    return pre[order], post[order]


def _rewire_targets(rng, n: int, sources: np.ndarray, targets: np.ndarray, rewire: np.ndarray) -> np.ndarray:
    """Give the rewired edges new targets, vectorised over all edges.

    For every source, the rewired edges receive distinct targets drawn
    uniformly from the nodes that are neither the source itself nor one of its
    kept (not rewired) targets. Candidates are drawn for all pending edges at
    once; a candidate equal to the source, to a kept target, or to a target
    already taken by another rewired edge of the same source is redrawn.
    """
    targets = np.array(targets, dtype=np.int64)
    idx = np.nonzero(rewire)[0]
    if idx.size == 0:
        return targets
    src = sources[idx].astype(np.int64)
    kept = ~np.asarray(rewire, dtype=bool)
    kept_codes = np.sort(sources[kept].astype(np.int64) * n + targets[kept])
    new_t = np.empty(idx.size, dtype=np.int64)
    pending = np.arange(idx.size)
    while pending.size:
        new_t[pending] = randint(rng, n, pending.size)
        codes = src * n + new_t
        is_pending = np.zeros(idx.size, dtype=bool)
        is_pending[pending] = True
        # Among equal codes, an already accepted edge keeps its target.
        order = np.lexsort((is_pending, codes))
        dup = np.zeros(idx.size, dtype=bool)
        dup[order[1:]] = codes[order[1:]] == codes[order[:-1]]
        bad = is_pending & ((new_t == src) | dup | np.isin(codes, kept_codes))
        pending = np.nonzero(bad)[0]
    targets[idx] = new_t
    return targets


def _preferential_attachment(rng, n: int, m: int, m0: int) -> np.ndarray:
    """Barabási–Albert target selection with the repeated-nodes list.

    Returns an ``(n - m0, m)`` array: row ``v - m0`` holds the ``m`` distinct
    existing nodes that node ``v`` attaches to. Each target is drawn uniformly
    from the repeated-nodes list (probability proportional to the attachment
    weight) and repeats are redrawn, which is successive sampling without
    replacement proportional to the weights. The loop over new nodes does
    ``O(m)`` work each, so the total cost is ``O(n * m)``.
    """
    targets = np.empty((max(n - m0, 0), m), dtype=np.int64)
    repeated = np.repeat(np.arange(m0), 2 * (m0 - 1)).tolist()
    block = 4096
    uniforms = rng.random(block).tolist()
    pos = 0
    for row, node in enumerate(range(m0, n)):
        length = len(repeated)
        chosen = []
        seen = set()
        while len(chosen) < m:
            if pos == block:
                uniforms = rng.random(block).tolist()
                pos = 0
            candidate = repeated[int(uniforms[pos] * length)]
            pos += 1
            if candidate not in seen:
                seen.add(candidate)
                chosen.append(candidate)
        targets[row] = chosen
        repeated.extend(chosen)
        repeated.extend([node] * m)
    return targets


def _group_sizes(n: int, n_groups: int, ratios, label: str) -> List[int]:
    """Split ``n`` neurons into ``n_groups`` consecutive groups.

    ``ratios`` (length ``n_groups - 1``) gives fixed sizes (integers) or
    proportions (floats) of the first groups, the last group taking the rest;
    ``None`` splits evenly (the first ``n % n_groups`` groups get one extra).
    """
    if ratios is None:
        base_size, remainder = divmod(n, n_groups)
        return [base_size + (1 if i < remainder else 0) for i in range(n_groups)]
    sizes = []
    remaining = n
    for ratio in ratios:
        # Integral values (incl. numpy ints, but not bool) are fixed sizes;
        # everything else is treated as a proportion of the total size.
        if isinstance(ratio, numbers.Integral) and not isinstance(ratio, bool):
            size = int(ratio)
        else:
            size = round(ratio * n)
        if size < 0:
            raise ValueError(f"{label} size cannot be negative: {size}")
        if size > remaining:
            raise ValueError(
                f"{label} size {size} exceeds remaining neurons {remaining}. "
                f"Check your {label.lower()}_ratios configuration."
            )
        sizes.append(size)
        remaining -= size
    sizes.append(remaining)
    return sizes


class SmallWorld(PointConnectivity):
    """Watts-Strogatz small-world network topology.

    This class implements the Watts-Strogatz model for generating small-world networks,
    which exhibit both high clustering coefficient (like regular lattices) and short
    average path length (like random graphs). The algorithm starts with a regular ring
    lattice where each node connects to its k nearest neighbors, then randomly rewires
    each edge with probability p.

    The small-world property is characteristic of many real-world networks, including
    neural networks, social networks, and power grids, making this a biologically
    plausible connectivity pattern for neural simulations.

    Parameters
    ----------
    k : int, default=6
        Number of nearest neighbors each node connects to in the initial ring lattice.
        Must be a positive even number strictly less than the network size. Higher
        values create more local connections and increase the clustering coefficient.
    p : float, default=0.3
        Rewiring probability for each edge. Valid range is [0, 1].
        - p=0: Regular ring lattice with high clustering, long path length
        - p=1: Random graph with low clustering, short path length
        - 0<p<1: Small-world network with high clustering and short path length
    weight : Initializer, optional
        Weight initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no delays are generated.
    **kwargs
        Additional keyword arguments passed to the parent PointConnectivity class,
        such as 'seed' for random number generation.

    Notes
    -----
    - This connectivity pattern requires pre_size == post_size (recurrent connectivity)
    - Self-connections are avoided during rewiring; duplicate edges are also rejected
    - The resulting network maintains exactly n*k directed edges (rewiring preserves the
      edge count), where n is the network size. Edges are directed (a→b does not imply b→a).
    - Based on the Watts-Strogatz model [1]_.
    - Generation runs on the host with NumPy and cannot be traced by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Rewiring is vectorised. The edges to rewire are chosen with the
       same draw as in 0.6, but their new targets are drawn in bulk:
       the rewired edges of a source get distinct targets, uniform over
       the nodes that are neither the source nor one of its kept targets
       (self-connections and duplicates are redrawn). For ``p > 0`` the
       edges for a given ``seed`` therefore differ from 0.6. ``p = 0``
       gives the same ring lattice as before.

    References
    ----------
    .. [1] Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world'
           networks. Nature, 393(6684), 440-442.

    Examples
    --------
    Create a small-world network with default parameters:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import SmallWorld
        >>> sw = SmallWorld(k=6, p=0.3, weight=0.8 * u.nS, seed=0)
        >>> result = sw(pre_size=1000, post_size=1000)
        >>> result.n_connections
        6000

    Create a small-world network with higher rewiring probability:

    .. code-block:: python

        >>> sw = SmallWorld(k=10, p=0.5, weight=1.0 * u.nS, delay=2.0 * u.ms)
        >>> result = sw(pre_size=500, post_size=500)

    Use with a custom weight initializer:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>> sw = SmallWorld(k=8, p=0.2, weight=Normal(mean=1.0, std=0.1))
        >>> result = sw(pre_size=1000, post_size=1000)
    """
    __module__ = 'braintools.conn'

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

        if int(np.prod(pre_size)) != int(np.prod(post_size)):
            raise ValueError("Small-world networks require pre_size == post_size")

        # Validate k: must be a positive even number strictly less than n so that the
        # ring lattice has exactly n*k edges without self-loops or duplicates.
        if self.k % 2 != 0:
            raise ValueError(f"k must be even for small-world networks, got {self.k}")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.k >= n:
            raise ValueError(f"k ({self.k}) must be less than the network size ({n})")

        # Vectorized generation of regular ring lattice
        k_half = self.k // 2
        sources = np.repeat(np.arange(n), k_half * 2)

        # Create offsets for forward and backward connections
        offsets = np.tile(np.concatenate([np.arange(1, k_half + 1), -np.arange(1, k_half + 1)]), n)
        targets = (sources + offsets) % n

        # Rewiring: each edge is rewired with probability p; the rewired edges of a
        # source get distinct new targets that are neither the source nor one of
        # its kept targets (no self-loops or duplicates, exactly n*k edges).
        rewire_mask = self.rng.random(len(sources)) < self.p
        targets = _rewire_targets(self.rng, n, sources, targets, rewire_mask)

        pre_indices = sources
        post_indices = targets
        n_connections = len(pre_indices)

        # Generate weights and delays using initialization classes
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
                'pattern': 'small_world',
                'k': self.k,
                'p': self.p
            }
        )


class ScaleFree(PointConnectivity):
    """Barabási-Albert scale-free network with preferential attachment.

    This class implements the Barabási-Albert model for generating scale-free networks,
    which exhibit a power-law degree distribution where P(k) ~ k^(-γ). The algorithm uses
    preferential attachment: new nodes preferentially connect to existing nodes with higher
    degree, following the principle of "rich get richer".

    Scale-free networks are ubiquitous in real-world systems including the Internet, social
    networks, protein interaction networks, and neural connectivity patterns in the brain.
    These networks are characterized by the presence of highly connected "hub" nodes and
    are remarkably robust to random failures but vulnerable to targeted attacks on hubs.

    Parameters
    ----------
    m : int, default=3
        Number of edges to attach from each new node to existing nodes. This parameter
        controls the minimum degree of nodes and affects the network density. Must be
        at least 1 and strictly less than the network size.
        Higher values create denser networks with more connections.
    weight : Initializer, optional
        Weight initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no delays are generated.
    **kwargs
        Additional keyword arguments passed to the parent PointConnectivity class,
        such as 'seed' for random number generation.

    Notes
    -----
    - This connectivity pattern requires pre_size == post_size (recurrent connectivity)
    - The algorithm starts with a complete graph of max(m, 2) nodes
    - Connections are bidirectional (undirected network)
    - The resulting degree distribution follows approximately P(k) ~ k^(-3)
    - The mean total degree is close to 2 * m for large networks
    - The algorithm complexity is O(n*m) where n is the network size
    - Based on the Barabási-Albert model [1]_; see [2]_ for a review.
    - Generation runs on the host with NumPy and cannot be traced by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Attachment targets are drawn from a list of edge endpoints, in
       which every node appears once per incident edge, instead of by
       normalising the full degree vector at each step. The attachment
       rule is unchanged, the cost is O(n * m), and the edges for a
       given ``seed`` differ from 0.6.

    References
    ----------
    .. [1] Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks.
           Science, 286(5439), 509-512.
    .. [2] Albert, R., & Barabási, A. L. (2002). Statistical mechanics of complex networks.
           Reviews of Modern Physics, 74(1), 47.

    Examples
    --------
    Create a scale-free network with default parameters:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import ScaleFree
        >>> sf = ScaleFree(m=3, weight=1.0 * u.nS, seed=0)
        >>> result = sf(pre_size=1000, post_size=1000)
        >>> result.n_connections
        5988

    Create a denser scale-free network with more attachments:

    .. code-block:: python

        >>> sf = ScaleFree(m=5, weight=0.5 * u.nS, delay=1.5 * u.ms)
        >>> result = sf(pre_size=500, post_size=500)

    Use with a custom weight initializer:

    .. code-block:: python

        >>> from braintools.init import Uniform
        >>> sf = ScaleFree(m=4, weight=Uniform(low=0.5, high=2.0))
        >>> result = sf(pre_size=1000, post_size=1000)
    """
    __module__ = 'braintools.conn'

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

        if int(np.prod(pre_size)) != int(np.prod(post_size)):
            raise ValueError("Scale-free networks require pre_size == post_size")

        # Validate m: the initial complete graph spans max(m, 2) nodes, so m must be
        # at least 1 and strictly less than n to keep indices in range.
        if self.m < 1:
            raise ValueError(f"m must be at least 1, got {self.m}")
        if self.m >= n:
            raise ValueError(f"m ({self.m}) must be less than the network size ({n})")

        # Start with a small complete graph; each pair is stored in both directions.
        m0 = max(self.m, 2)
        init_i, init_j = np.triu_indices(m0, 1)
        init_pre = np.stack([init_i, init_j], axis=1).reshape(-1)
        init_post = np.stack([init_j, init_i], axis=1).reshape(-1)

        # Preferential attachment with the repeated-nodes list: every node appears
        # once per unit of attachment weight (the initial nodes 2 * (m0 - 1) times,
        # later nodes once per edge), so a uniform draw from the list picks a node
        # with probability proportional to its weight.
        targets = _preferential_attachment(self.rng, n, self.m, m0)
        new_nodes = np.repeat(np.arange(m0, n, dtype=np.int64), self.m)
        flat_targets = targets.reshape(-1)
        pre_indices = np.concatenate([init_pre, np.stack([new_nodes, flat_targets], axis=1).reshape(-1)])
        post_indices = np.concatenate([init_post, np.stack([flat_targets, new_nodes], axis=1).reshape(-1)])
        pre_indices = pre_indices.astype(np.int64)
        post_indices = post_indices.astype(np.int64)
        n_connections = len(pre_indices)

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
    """Regular network where all neurons have the same degree.

    This class creates a regular random network where every node has exactly the same
    number of outgoing connections (out-degree). The targets for each node are chosen
    randomly without replacement, excluding self-connections. This topology is useful
    for creating homogeneous networks where all neurons have equal influence.

    Regular networks provide a baseline for comparing other network topologies and are
    particularly useful in studies of network dynamics where uniform connectivity is
    desired. Unlike regular lattices (e.g., ring or grid), connections are random rather
    than following a spatial pattern.

    Parameters
    ----------
    degree : int
        Number of outgoing connections per neuron. Must be less than the network size
        (since self-connections are excluded). All neurons will have exactly this many
        outgoing connections, creating a perfectly regular out-degree distribution.
    weight : Initializer, optional
        Weight initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no delays are generated.
    **kwargs
        Additional keyword arguments passed to the parent PointConnectivity class,
        such as 'seed' for random number generation.

    Notes
    -----
    - This connectivity pattern requires pre_size == post_size (recurrent connectivity)
    - All nodes have the same out-degree (number of outgoing connections)
    - In-degree (incoming connections) may vary across nodes
    - Self-connections are automatically excluded
    - The total number of connections is exactly n * degree, where n is the network size
    - Targets are selected randomly without replacement for each source neuron
    - Generation runs on the host with NumPy and cannot be traced by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       The targets of all sources are sampled in one vectorised call, so
       the edges for a given ``seed`` differ from 0.6 (same
       distribution). Edges are sorted by pre, then post index.

    Examples
    --------
    Create a regular network where each neuron connects to 10 others:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import Regular
        >>> import numpy as np
        >>> reg = Regular(degree=10, weight=1.0 * u.nS, seed=0)
        >>> result = reg(pre_size=1000, post_size=1000)
        >>> result.n_connections
        10000
        >>> bool(np.all(np.bincount(result.pre_indices) == 10))
        True

    Create a regular network with delays:

    .. code-block:: python

        >>> reg = Regular(degree=20, weight=0.8 * u.nS, delay=2.0 * u.ms)
        >>> result = reg(pre_size=500, post_size=500)

    Use with a custom weight initializer:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>> reg = Regular(degree=15, weight=Normal(mean=1.0, std=0.2))
        >>> result = reg(pre_size=1000, post_size=1000)
    """
    __module__ = 'braintools.conn'

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

        if int(np.prod(pre_size)) != int(np.prod(post_size)):
            raise ValueError("Regular networks require pre_size == post_size")

        # Each neuron connects to 'degree' distinct random targets, self excluded,
        # sampled for all rows at once (each row is sorted).
        degree = int(self.degree)
        post_indices = sample_rows_without_replacement(
            self.rng, n, n, degree, exclude=np.arange(n)
        ).reshape(-1)
        pre_indices = np.repeat(np.arange(n, dtype=np.int64), degree)

        n_connections = len(pre_indices)

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


class ModularRandom(PointConnectivity):
    """Modular network with intra-module and inter-module random connectivity.

    This class creates a modular network structure where neurons are divided into
    distinct modules (communities) with different connection probabilities within
    and between modules. Intra-module connections (within the same module) typically
    have higher probability than inter-module connections (between different modules),
    creating a community structure.

    Modular organization is a fundamental feature of brain networks, observed across
    multiple scales from cortical columns to large-scale brain areas. This topology
    enables specialized local processing within modules while maintaining global
    integration through sparse inter-module connections. It balances functional
    segregation with integration, supporting both specialized and distributed
    information processing.

    Parameters
    ----------
    n_modules : int
        Number of modules (communities) to divide the network into. Must not exceed
        the network size. Neurons are distributed approximately evenly across modules
        (the first ``remainder`` modules each receive one extra neuron).
    intra_prob : float, default=0.3
        Connection probability for neuron pairs within the same module. Valid range
        is [0, 1]. Higher values create denser intra-module connectivity and stronger
        community structure.
    inter_prob : float, default=0.01
        Connection probability for neuron pairs in different modules. Valid range
        is [0, 1]. Typically much smaller than intra_prob to create distinct modules.
        The ratio intra_prob/inter_prob determines the strength of modularity.
    weight : Initializer, optional
        Weight initialization for all connections (both intra and inter-module).
        Can be a scalar value, array, or an Initializer instance for more complex
        initialization patterns. If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization for all connections (both intra and inter-module).
        Can be a scalar value, array, or an Initializer instance for more complex
        initialization patterns. If None, no delays are generated.
    **kwargs
        Additional keyword arguments passed to the parent PointConnectivity class,
        such as 'seed' for random number generation.

    Notes
    -----
    - This connectivity pattern requires pre_size == post_size (recurrent connectivity)
    - Neurons are assigned to modules in sequential order using a balanced split
      (every module gets at least one neuron; n_modules must not exceed the network size)
    - Self-connections are automatically excluded
    - The same weight and delay initialization is used for all connections
    - For module-specific connectivity patterns, use ModularGeneral instead
    - Expected number of connections: n² * ((intra_prob + (n_modules-1)*inter_prob) / n_modules)
    - Modularity strength can be quantified by the Q-statistic (Newman, 2006)
    - See [1]_ and [2]_ for modular structure in networks and brains.
    - Generation runs on the host with NumPy and cannot be traced by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Each block of neuron pairs is sampled by drawing a binomial edge
       count and then that many distinct pairs, without a dense
       ``pre_size x post_size`` matrix, so time and memory grow with the
       number of edges. The edge set for a given ``seed`` differs from 0.6
       (same distribution); edges are sorted by pre, then post index.
       Probabilities outside ``[0, 1]`` now raise ``ValueError`` (0.6
       treated them as 0 or 1).

    References
    ----------
    .. [1] Girvan, M., & Newman, M. E. (2002). Community structure in social and
           biological networks. PNAS, 99(12), 7821-7826.
    .. [2] Sporns, O., & Betzel, R. F. (2016). Modular brain networks. Annual Review
           of Psychology, 67, 613-640.

    See Also
    --------
    ModularGeneral : Modular network with custom connectivity patterns per module

    Examples
    --------
    Create a modular network with 5 modules:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import ModularRandom
        >>> mod = ModularRandom(n_modules=5, intra_prob=0.3, inter_prob=0.01, seed=0)
        >>> result = mod(pre_size=1000, post_size=1000)
        >>> result.n_connections
        67881

    Create a strongly modular network with sparse inter-module connections:

    .. code-block:: python

        >>> mod = ModularRandom(
        ...     n_modules=10,
        ...     intra_prob=0.4,
        ...     inter_prob=0.005,
        ...     weight=1.0 * u.nS,
        ...     delay=2.0 * u.ms
        ... )
        >>> result = mod(pre_size=500, post_size=500)

    Create a weakly modular network:

    .. code-block:: python

        >>> mod = ModularRandom(n_modules=3, intra_prob=0.2, inter_prob=0.1)
        >>> result = mod(pre_size=1000, post_size=1000)

    Use with custom initializers:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>> mod = ModularRandom(
        ...     n_modules=8,
        ...     intra_prob=0.25,
        ...     inter_prob=0.02,
        ...     weight=Normal(mean=1.0, std=0.1)
        ... )
        >>> result = mod(pre_size=800, post_size=800)
    """
    __module__ = 'braintools.conn'

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

        if int(np.prod(pre_size)) != int(np.prod(post_size)):
            raise ValueError("Modular networks require pre_size == post_size")

        if self.n_modules > n:
            raise ValueError(
                f"n_modules ({self.n_modules}) must not exceed the network size ({n})"
            )

        intra_prob = _check_prob(self.intra_prob, 'intra_prob')
        inter_prob = _check_prob(self.inter_prob, 'inter_prob')

        # Assign neurons to modules using a balanced distribution so every module
        # receives at least one neuron (the first ``remainder`` modules get one extra).
        mod_sizes = _group_sizes(n, self.n_modules, None, 'Module')
        bounds = np.concatenate([[0], np.cumsum(mod_sizes)]).astype(int)

        # Intra-module blocks (one per module, self-connections excluded).
        pre_parts, post_parts = [], []
        intra_pre, intra_post = _block_bernoulli(
            self.rng, [(bounds[i], bounds[i + 1], bounds[i], bounds[i + 1], intra_prob)
                       for i in range(self.n_modules)]
        )
        pre_parts.append(intra_pre)
        post_parts.append(intra_post)

        # Inter-module pairs: Bernoulli(inter_prob) over the whole matrix, keeping
        # only the pairs in different modules (which stay independent draws).
        if inter_prob > 0 and self.n_modules > 1:
            modules = np.repeat(np.arange(self.n_modules), mod_sizes)
            pre, post = bernoulli_pairs(self.rng, n, n, inter_prob)
            between = modules[pre] != modules[post]
            pre_parts.append(pre[between])
            post_parts.append(post[between])

        pre_indices, post_indices = _sorted_pairs(pre_parts, post_parts)

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


class ModularGeneral(PointConnectivity):
    """Modular network using Connectivity instances for both intra and inter-module patterns.

    This class creates a modular network structure where:

    - Intra-module connectivity is defined by a sequence of Connectivity instances
    - Inter-module connectivity is defined by either a single PointConnectivity instance
      or a dict mapping (pre_mod, post_mod) tuples to PointConnectivity instances

    Parameters
    ----------
    intra_conn : Sequence of PointConnectivity
        Sequence of connectivity instances for generating connections within each module.
        The length of the sequence determines the number of modules.
    inter_conn : PointConnectivity, optional
        Default connectivity instance for inter-module connections.
        Applied to all module pairs not specified in inter_conn_pair.
        If None and a pair is not in inter_conn_pair, that pair is skipped.
    inter_conn_pair : dict, optional
        Dict mapping (pre_module_id, post_module_id) tuples to PointConnectivity instances
        for specific inter-module connections. Overrides inter_conn for specified pairs.
    module_ratios : Sequence of int or float, optional
        Ratios or sizes for the first n_modules-1 modules. Length must be n_modules-1.
        Each element can be:

        - int: Fixed size for that module
        - float: Proportion of total size (e.g., 0.3 means 30% of total)

        The last module gets the remaining neurons.
        If None, modules are evenly divided.

    Examples
    --------

    .. code-block:: python

        >>> from braintools.conn import Random, ModularGeneral, SmallWorld
        >>> import brainunit as u

        >>> # Different connectivity per module with uniform inter-module connectivity
        >>> intra_list = [
        ...     Random(prob=0.3, weight=1.0 * u.nS),
        ...     Random(prob=0.5, weight=1.5 * u.nS),
        ...     SmallWorld(k=6, p=0.3, weight=2.0 * u.nS)
        ... ]
        >>> inter = Random(prob=0.01, weight=0.1 * u.nS)
        >>> mod = ModularGeneral(intra_conn=intra_list, inter_conn=inter)
        >>> result = mod(pre_size=900, post_size=900)

        >>> # Different inter-module connectivity for specific module pairs
        >>> default_inter = Random(prob=0.01, weight=0.1 * u.nS)
        >>> specific_inter = {
        ...     (0, 1): Random(prob=0.05, weight=0.2 * u.nS),
        ...     (1, 2): Random(prob=0.03, weight=0.15 * u.nS),
        ... }
        >>> mod = ModularGeneral(
        ...     intra_conn=intra_list,
        ...     inter_conn=default_inter,
        ...     inter_conn_pair=specific_inter
        ... )
        >>> result = mod(pre_size=900, post_size=900)

        >>> # Custom module sizes with fixed sizes
        >>> mod = ModularGeneral(
        ...     intra_conn=intra_list,
        ...     inter_conn=inter,
        ...     module_ratios=[200, 300]  # Last module gets remaining 400
        ... )
        >>> result = mod(pre_size=900, post_size=900)

        >>> # Custom module sizes with ratios
        >>> mod = ModularGeneral(
        ...     intra_conn=intra_list,
        ...     inter_conn=inter,
        ...     module_ratios=[0.2, 0.3]  # 20%, 30%, and remaining 50%
        ... )
        >>> result = mod(pre_size=900, post_size=900)

        >>> # Mixed fixed and ratio sizes
        >>> mod = ModularGeneral(
        ...     intra_conn=intra_list,
        ...     inter_conn=inter,
        ...     module_ratios=[100, 0.5]  # 100 neurons, 50% of total, and remainder
        ... )
        >>> result = mod(pre_size=900, post_size=900)
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        intra_conn: Sequence[PointConnectivity],
        inter_conn: Optional[PointConnectivity] = None,
        inter_conn_pair: Optional[Dict[tuple[int, int], PointConnectivity]] = None,
        module_ratios: Optional[Sequence[int | float]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not isinstance(intra_conn, (list, tuple)):
            raise TypeError("intra_conn must be a list/tuple of PointConnectivity instances")

        self.intra_conn = intra_conn
        self.n_modules = len(intra_conn)
        self.inter_conn = inter_conn
        self.module_ratios = module_ratios
        if inter_conn_pair is None:
            inter_conn_pair = dict()
        if not isinstance(inter_conn_pair, dict):
            raise TypeError("inter_conn_pair must be a dict mapping (pre_mod, post_mod) to PointConnectivity")
        self.inter_conn_pair = inter_conn_pair

        if module_ratios is not None:
            if len(module_ratios) != self.n_modules - 1:
                raise ValueError(
                    f"Length of module_ratios ({len(module_ratios)}) must be n_modules-1 ({self.n_modules - 1})"
                )

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate modular network with custom intra-module connectivity."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if int(np.prod(pre_size)) != int(np.prod(post_size)):
            raise ValueError("Modular networks require pre_size == post_size")

        # Determine module sizes and boundaries
        if self.module_ratios is not None:
            mod_sizes = []
            remaining = n

            # Process first n_modules-1 sizes
            for ratio in self.module_ratios:
                # Integral values (incl. numpy ints, but not bool) are fixed sizes;
                # everything else is treated as a proportion of the total size.
                if isinstance(ratio, numbers.Integral) and not isinstance(ratio, bool):
                    # Fixed size
                    size = int(ratio)
                else:
                    # Proportional size (round to nearest neuron count)
                    size = round(ratio * n)

                if size < 0:
                    raise ValueError(f"Module size cannot be negative: {size}")
                if size > remaining:
                    raise ValueError(
                        f"Module size {size} exceeds remaining neurons {remaining}. "
                        f"Check your module_ratios configuration."
                    )

                mod_sizes.append(size)
                remaining -= size

            # Last module gets all remaining neurons
            if remaining < 0:
                raise ValueError(
                    f"Module sizes exceed total size: sum={n - remaining}, total={n}"
                )
            mod_sizes.append(remaining)
        else:
            # Evenly divide neurons across modules
            base_size = n // self.n_modules
            remainder = n % self.n_modules
            mod_sizes = [base_size + (1 if i < remainder else 0) for i in range(self.n_modules)]

        # Calculate module boundaries for fast indexing
        mod_boundaries = [0]
        for size in mod_sizes:
            mod_boundaries.append(mod_boundaries[-1] + size)

        all_pre_indices = []
        all_post_indices = []
        # Per-block (n_edges, weights-or-None) / (n_edges, delays-or-None). Weights and
        # delays from sub-connectivities may be scalars (e.g. ``1.0 * u.nS``), per-edge
        # arrays, or None; we broadcast/combine them once all blocks are collected.
        block_weights = []
        block_delays = []

        pre_positions = kwargs.get('pre_positions', None)
        post_positions = kwargs.get('post_positions', None)
        have_positions = pre_positions is not None and post_positions is not None

        def _collect(result, pre_offset, post_offset):
            n_edges = len(result.pre_indices)
            if n_edges == 0:
                return
            all_pre_indices.append(result.pre_indices + pre_offset)
            all_post_indices.append(result.post_indices + post_offset)
            block_weights.append((n_edges, result.weights))
            block_delays.append((n_edges, result.delays))

        # Generate intra-module connections using the provided connectivity
        for mod_id in range(self.n_modules):
            mod_start = mod_boundaries[mod_id]
            mod_end = mod_boundaries[mod_id + 1]
            mod_n = mod_end - mod_start

            if mod_n == 0:
                continue

            # Get the connectivity instance for this module
            conn = self.intra_conn[mod_id]

            # Generate connections within this module
            if have_positions:
                mod_pre_pos = pre_positions[mod_start:mod_end]
                mod_post_pos = post_positions[mod_start:mod_end]
            else:
                mod_pre_pos = None
                mod_post_pos = None

            # ``recompute=True`` so each block draws independently even when the same
            # connectivity instance (or equal-size modules) reuses the size-aware cache.
            # We do not mutate ``conn.rng`` (it belongs to the user-supplied object).
            intra_result = conn(
                pre_size=mod_n,
                post_size=mod_n,
                pre_positions=mod_pre_pos,
                post_positions=mod_post_pos,
                recompute=True,
            )

            _collect(intra_result, mod_start, mod_start)

        # Generate inter-module connections
        for pre_mod in range(self.n_modules):
            for post_mod in range(self.n_modules):
                if pre_mod == post_mod:
                    continue

                # Get the appropriate connectivity instance
                # First check inter_conn_pair, then fall back to default inter_conn
                if (pre_mod, post_mod) in self.inter_conn_pair:
                    conn = self.inter_conn_pair[(pre_mod, post_mod)]
                    if conn is None:
                        # Skip this pair if no connectivity is specified
                        continue
                else:
                    conn = self.inter_conn
                    if conn is None:
                        continue

                pre_start = mod_boundaries[pre_mod]
                pre_end = mod_boundaries[pre_mod + 1]
                post_start = mod_boundaries[post_mod]
                post_end = mod_boundaries[post_mod + 1]

                pre_n = pre_end - pre_start
                post_n = post_end - post_start

                if pre_n == 0 or post_n == 0:
                    continue

                # Generate connections between these two modules
                if have_positions:
                    pre_mod_pos = pre_positions[pre_start:pre_end]
                    post_mod_pos = post_positions[post_start:post_end]
                else:
                    pre_mod_pos = None
                    post_mod_pos = None

                # ``recompute=True`` so every module pair draws an independent block;
                # a single shared ``inter_conn`` instance would otherwise return the
                # same cached result for all equal-size pairs.
                inter_result = conn(
                    pre_size=pre_n,
                    post_size=post_n,
                    pre_positions=pre_mod_pos,
                    post_positions=post_mod_pos,
                    recompute=True,
                )

                _collect(inter_result, pre_start, post_start)

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
        # Broadcast/concatenate per-block weights and delays. Policy: if no block
        # supplies weights (resp. delays), the combined result is None; otherwise
        # blocks that lack them are filled with 0 so the per-edge array stays aligned.
        final_weights = _combine_block_params(block_weights)
        final_delays = _combine_block_params(block_delays)

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
                'pattern': 'modular_general',
                'n_modules': self.n_modules,
                'module_sizes': mod_sizes,
                'module_ratios': self.module_ratios,
                'inter_conn': type(self.inter_conn).__name__ if self.inter_conn is not None else None,
                'inter_conn_pair': {k: type(v).__name__ for k, v in self.inter_conn_pair.items()},
                'intra_conn': [type(conn).__name__ for conn in self.intra_conn]
            }
        )


class HierarchicalRandom(PointConnectivity):
    """Hierarchical network with multiple levels and asymmetric feedforward/feedback connections.

    This class creates a hierarchical network structure organized into distinct levels
    (layers), where connections flow primarily in a feedforward direction from lower to
    higher levels, with optional feedback connections. Within each level, recurrent
    connections can be specified. This architecture mirrors the hierarchical organization
    observed in cortical processing streams and deep neural networks.

    Hierarchical organization is fundamental to brain architecture, enabling progressive
    processing of information from sensory inputs through increasingly abstract
    representations. Each level performs specialized computations while maintaining
    bidirectional communication with adjacent levels for top-down modulation and
    bottom-up information flow.

    Parameters
    ----------
    n_levels : int
        Number of hierarchical levels in the network. Must be at least 2.
        Levels are numbered from 0 (lowest/input) to n_levels-1 (highest/output).
    feedforward_prob : float, default=0.3
        Connection probability from level i to level i+1 (feedforward connections).
        These connections propagate information up the hierarchy. Valid range is [0, 1].
    feedback_prob : float, default=0.1
        Connection probability from level i+1 to level i (feedback connections).
        These connections enable top-down modulation. Valid range is [0, 1].
        Typically lower than feedforward_prob to reflect the asymmetry in cortical
        connectivity.
    recurrent_prob : float, default=0.2
        Connection probability within each level (recurrent/lateral connections).
        Valid range is [0, 1]. These connections enable local processing and
        competition within each level.
    skip_prob : float, default=0.0
        Connection probability for skip connections that bypass adjacent levels
        (e.g., from level i to level i+2). Valid range is [0, 1]. Skip connections
        can enable faster information flow and gradient propagation.
    level_ratios : Sequence of int or float, optional
        Ratios or sizes for the first n_levels-1 levels. Length must be n_levels-1.
        Each element can be:

        - int: Fixed size for that level
        - float: Proportion of total size (e.g., 0.3 means 30% of total)

        The last level gets the remaining neurons.
        If None, levels are evenly divided.
    weight : Initializer, optional
        Weight initialization for all connections. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization for all connections. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no delays are generated.
    **kwargs
        Additional keyword arguments passed to the parent PointConnectivity class,
        such as 'seed' for random number generation.

    Notes
    -----

    - This connectivity pattern requires pre_size == post_size (recurrent connectivity)
    - Neurons are assigned to levels in sequential order
    - Self-connections are automatically excluded
    - Feedforward connections: level i → level i+1
    - Feedback connections: level i+1 → level i
    - Skip connections: level i → level i+k where k > 1
    - The same weight and delay initialization is used for all connection types
    - Expected connectivity follows a hierarchical pattern with stronger feedforward flow
    - Cortical hierarchies are reviewed in [1]_, [2]_ and [3]_.
    - Generation runs on the host with NumPy and cannot be traced by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Each block of neuron pairs is sampled by drawing a binomial edge
       count and then that many distinct pairs, without a dense
       ``pre_size x post_size`` matrix, so time and memory grow with the
       number of edges. The edge set for a given ``seed`` differs from 0.6
       (same distribution); edges are sorted by pre, then post index.
       Probabilities outside ``[0, 1]`` now raise ``ValueError`` (0.6
       treated them as 0 or 1).

    References
    ----------
    .. [1] Felleman, D. J., & Van Essen, D. C. (1991). Distributed hierarchical
           processing in the primate cerebral cortex. Cerebral Cortex, 1(1), 1-47.
    .. [2] Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding.
           Neuron, 76(4), 695-711.
    .. [3] Markov, N. T., et al. (2013). Cortical high-density counterstream architectures.
           Science, 342(6158), 1238406.

    Examples
    --------
    Create a 3-level hierarchical network:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import HierarchicalRandom
        >>> hier = HierarchicalRandom(
        ...     n_levels=3,
        ...     feedforward_prob=0.3,
        ...     feedback_prob=0.1,
        ...     recurrent_prob=0.2,
        ...     weight=1.0 * u.nS,
        ...     seed=0,
        ... )
        >>> result = hier(pre_size=900, post_size=900)
        >>> result.n_connections
        125653

    Create a deep hierarchy with skip connections:

    .. code-block:: python

        >>> hier = HierarchicalRandom(
        ...     n_levels=5,
        ...     feedforward_prob=0.4,
        ...     feedback_prob=0.15,
        ...     recurrent_prob=0.25,
        ...     skip_prob=0.05,
        ...     weight=0.8 * u.nS,
        ...     delay=2.0 * u.ms
        ... )
        >>> result = hier(pre_size=1000, post_size=1000)

    Create a hierarchy with custom level sizes:

    .. code-block:: python

        >>> # Bottom-heavy hierarchy (sensory-like)
        >>> hier = HierarchicalRandom(
        ...     n_levels=4,
        ...     feedforward_prob=0.3,
        ...     feedback_prob=0.1,
        ...     level_ratios=[0.4, 0.3, 0.2]  # 40%, 30%, 20%, 10%
        ... )
        >>> result = hier(pre_size=1000, post_size=1000)

    Use with custom initializers:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>> hier = HierarchicalRandom(
        ...     n_levels=3,
        ...     feedforward_prob=0.35,
        ...     feedback_prob=0.12,
        ...     weight=Normal(mean=1.0, std=0.2)
        ... )
        >>> result = hier(pre_size=900, post_size=900)
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        n_levels: int,
        feedforward_prob: float = 0.3,
        feedback_prob: float = 0.1,
        recurrent_prob: float = 0.2,
        skip_prob: float = 0.0,
        level_ratios: Optional[Sequence[int | float]] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if n_levels < 2:
            raise ValueError("n_levels must be at least 2")

        self.n_levels = n_levels
        self.feedforward_prob = feedforward_prob
        self.feedback_prob = feedback_prob
        self.recurrent_prob = recurrent_prob
        self.skip_prob = skip_prob
        self.level_ratios = level_ratios
        self.weight_init = weight
        self.delay_init = delay

        if level_ratios is not None:
            if len(level_ratios) != n_levels - 1:
                raise ValueError(
                    f"Length of level_ratios ({len(level_ratios)}) must be n_levels-1 ({n_levels - 1})"
                )

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate hierarchical network."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if int(np.prod(pre_size)) != int(np.prod(post_size)):
            raise ValueError("Hierarchical networks require pre_size == post_size")

        recurrent_prob = _check_prob(self.recurrent_prob, 'recurrent_prob')
        feedforward_prob = _check_prob(self.feedforward_prob, 'feedforward_prob')
        feedback_prob = _check_prob(self.feedback_prob, 'feedback_prob')
        skip_prob = _check_prob(self.skip_prob, 'skip_prob')

        # Level sizes and boundaries
        level_sizes = _group_sizes(n, self.n_levels, self.level_ratios, 'Level')
        bounds = np.concatenate([[0], np.cumsum(level_sizes)]).astype(int)

        # One Bernoulli block per (pre level, post level) pair.
        blocks = []
        for i_level in range(self.n_levels):
            for j_level in range(self.n_levels):
                if i_level == j_level:
                    prob = recurrent_prob  # recurrent connections within a level
                elif j_level == i_level + 1:
                    prob = feedforward_prob  # feedforward (i -> i+1)
                elif j_level == i_level - 1:
                    prob = feedback_prob  # feedback (i+1 -> i)
                elif j_level > i_level + 1:
                    prob = skip_prob  # forward skip connections
                else:
                    continue  # backward skip connections are not included
                blocks.append((bounds[i_level], bounds[i_level + 1], bounds[j_level], bounds[j_level + 1], prob))
        pre_indices, post_indices = _block_bernoulli(self.rng, blocks)

        metadata = {
            'pattern': 'hierarchical',
            'n_levels': self.n_levels,
            'level_sizes': level_sizes,
            'level_ratios': self.level_ratios,
            'feedforward_prob': self.feedforward_prob,
            'feedback_prob': self.feedback_prob,
            'recurrent_prob': self.recurrent_prob,
            'skip_prob': self.skip_prob
        }

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None),
                model_type='point',
                metadata=metadata
            )

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
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata=metadata
        )


class CorePeripheryRandom(PointConnectivity):
    """Core-periphery network with densely connected core and sparse periphery.

    This class creates a network with a core-periphery structure where a subset of
    nodes (the core) are densely interconnected, while the remaining nodes (the periphery)
    are sparsely connected to each other but maintain connections to the core. This
    architecture is common in brain networks, social networks, and infrastructure systems.

    Core-periphery organization enables efficient information integration through the
    densely connected core while maintaining specialized processing in the periphery.
    The core acts as a hub for global communication and coordination, while peripheral
    nodes maintain local specialization.

    Parameters
    ----------
    core_size : int or float
        Size of the core. If int, specifies the exact number of core neurons.
        If float in (0, 1), specifies the proportion of neurons in the core.
        Core neurons are the first core_size neurons in the population.
    core_core_prob : float, default=0.5
        Connection probability within the core (core→core). Higher values create a more
        densely connected core, which is characteristic of core-periphery networks.
    core_periphery_prob : float, default=0.2
        Connection probability from core to periphery (core→periphery).
        This parameter controls how strongly core neurons project to periphery.
    periphery_core_prob : float, default=0.2
        Connection probability from periphery to core (periphery→core).
        This parameter controls how periphery neurons feed back to core.
    periphery_periphery_prob : float, default=0.05
        Connection probability within the periphery (periphery→periphery).
        Typically much lower than core_core_prob, creating sparse peripheral connectivity.
    weight : Initializer, optional
        Weight initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization for each connection. Can be a scalar value, array,
        or an Initializer instance for more complex initialization patterns.
        If None, no delays are generated.
    **kwargs
        Additional keyword arguments passed to the parent PointConnectivity class,
        such as 'seed' for random number generation.

    Notes
    -----
    - This connectivity pattern requires pre_size == post_size (recurrent connectivity)
    - Core neurons are selected from the first core_size neurons (indices 0 to core_size-1)
    - Self-connections are automatically excluded
    - Core-periphery structure can be quantified using correlation measures
    - Typical parameter regime: core_core_prob >> core_periphery_prob ≈ periphery_core_prob > periphery_periphery_prob
    - Unlike the previous bidirectional parameter, now core→periphery and periphery→core
      probabilities are controlled independently for more flexibility
    - Core-periphery models follow [1]_; the rich club of the connectome is described in [2]_.
    - Generation runs on the host with NumPy and cannot be traced by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Each block of neuron pairs is sampled by drawing a binomial edge
       count and then that many distinct pairs, without a dense
       ``pre_size x post_size`` matrix, so time and memory grow with the
       number of edges. The edge set for a given ``seed`` differs from 0.6
       (same distribution); edges are sorted by pre, then post index.
       Probabilities outside ``[0, 1]`` now raise ``ValueError`` (0.6
       treated them as 0 or 1).

    References
    ----------
    .. [1] Borgatti, S. P., & Everett, M. G. (2000). Models of core/periphery structures.
           Social Networks, 21(4), 375-395.
    .. [2] van den Heuvel, M. P., & Sporns, O. (2011). Rich-club organization of the human
           connectome. Journal of Neuroscience, 31(44), 15775-15786.

    Examples
    --------
    Create a core-periphery network with 20% core:

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import CorePeripheryRandom
        >>> cp = CorePeripheryRandom(
        ...     core_size=0.2,
        ...     core_core_prob=0.5,
        ...     periphery_periphery_prob=0.05,
        ...     seed=0,
        ... )
        >>> result = cp(pre_size=1000, post_size=1000)
        >>> result.n_connections
        115661

    Create a network with fixed core size and asymmetric core-periphery connections:

    .. code-block:: python

        >>> cp = CorePeripheryRandom(
        ...     core_size=100,
        ...     core_core_prob=0.6,
        ...     core_periphery_prob=0.25,      # Strong core→periphery
        ...     periphery_core_prob=0.15,      # Weaker periphery→core
        ...     periphery_periphery_prob=0.03,
        ...     weight=1.0 * u.nS,
        ...     delay=2.0 * u.ms
        ... )
        >>> result = cp(pre_size=1000, post_size=1000)

    Create a network with strong feedback from periphery to core:

    .. code-block:: python

        >>> cp = CorePeripheryRandom(
        ...     core_size=0.15,
        ...     core_core_prob=0.7,
        ...     core_periphery_prob=0.2,
        ...     periphery_core_prob=0.3,       # Strong feedback
        ...     periphery_periphery_prob=0.02
        ... )
        >>> result = cp(pre_size=800, post_size=800)

    Use with custom weight initializer:

    .. code-block:: python

        >>> from braintools.init import Normal
        >>> cp = CorePeripheryRandom(
        ...     core_size=200,
        ...     core_core_prob=0.5,
        ...     weight=Normal(mean=1.0, std=0.2)
        ... )
        >>> result = cp(pre_size=1000, post_size=1000)
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        core_size: int | float,
        core_core_prob: float = 0.5,
        core_periphery_prob: float = 0.2,
        periphery_core_prob: float = 0.2,
        periphery_periphery_prob: float = 0.05,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.core_size = core_size
        self.core_core_prob = core_core_prob
        self.core_periphery_prob = core_periphery_prob
        self.periphery_core_prob = periphery_core_prob
        self.periphery_periphery_prob = periphery_periphery_prob
        self.weight_init = weight
        self.delay_init = delay

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate core-periphery network."""
        pre_size = kwargs['pre_size']
        post_size = kwargs['post_size']

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if int(np.prod(pre_size)) != int(np.prod(post_size)):
            raise ValueError("Core-periphery networks require pre_size == post_size")

        # Determine core size
        if isinstance(self.core_size, float):
            if not 0 < self.core_size < 1:
                raise ValueError("core_size as float must be in (0, 1)")
            n_core = int(self.core_size * n)
        else:
            n_core = self.core_size
            if n_core >= n:
                raise ValueError(f"core_size ({n_core}) must be less than network size ({n})")

        # Four Bernoulli blocks: core -> core, core -> periphery,
        # periphery -> core and periphery -> periphery.
        pre_indices, post_indices = _block_bernoulli(self.rng, [
            (0, n_core, 0, n_core, _check_prob(self.core_core_prob, 'core_core_prob')),
            (0, n_core, n_core, n, _check_prob(self.core_periphery_prob, 'core_periphery_prob')),
            (n_core, n, 0, n_core, _check_prob(self.periphery_core_prob, 'periphery_core_prob')),
            (n_core, n, n_core, n, _check_prob(self.periphery_periphery_prob, 'periphery_periphery_prob')),
        ])

        metadata = {
            'pattern': 'core_periphery',
            'core_size': n_core,
            'core_core_prob': self.core_core_prob,
            'core_periphery_prob': self.core_periphery_prob,
            'periphery_core_prob': self.periphery_core_prob,
            'periphery_periphery_prob': self.periphery_periphery_prob,
        }
        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=kwargs.get('pre_positions', None),
                post_positions=kwargs.get('post_positions', None),
                model_type='point',
                metadata=metadata,
            )

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
            pre_indices, post_indices,
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            model_type='point',
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            metadata=metadata,
        )
