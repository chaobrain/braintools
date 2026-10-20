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

from typing import Iterator, Optional, Tuple, Union

import brainunit as u
import numpy as np
from brainstate.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from braintools.init._distance_base import DistanceProfile
from braintools.init._distance_impl import (
    BimodalProfile,
    DoGProfile,
    ExponentialProfile,
    GaussianProfile,
    LinearProfile,
    LogisticProfile,
    MexicanHatProfile,
    PowerLawProfile,
    SigmoidProfile,
    StepProfile,
)
from braintools.init._init_base import param, Initializer
from ._base import PointConnectivity, ConnectionResult
from ._sampling import MAX_CHUNK_PAIRS, row_chunks, sample_rows_without_replacement

__all__ = [
    'DistanceDependent',
    'Gaussian',
    'Exponential',
    'Ring',
    'Grid2d',
    'RadialPatches',
]

# Built-in profiles whose ``probability`` is exactly zero beyond ``max_distance``
# (when it is set). Matched by exact type, so a subclass that overrides
# ``probability`` never has pairs silently dropped by the cutoff search.
_MAX_DISTANCE_PROFILES = (
    GaussianProfile,
    ExponentialProfile,
    PowerLawProfile,
    LinearProfile,
    SigmoidProfile,
    DoGProfile,
    LogisticProfile,
    BimodalProfile,
    MexicanHatProfile,
)

# Relative slack added to a cutoff radius for the KD-tree search, so that pairs
# whose distance rounds differently in the tree and in the profile are still
# handed to the profile, which applies its own ``d > max_distance`` test.
_CUTOFF_SLACK = 1e-9


def _size_num(size) -> int:
    """Number of elements of an int or shape-tuple population size."""
    if isinstance(size, (tuple, list)):
        return int(np.prod(size))
    return int(size)


def _position_values(pre_positions, post_positions) -> Tuple[np.ndarray, np.ndarray, object]:
    """Return float64 ``(N, d)`` mantissas of both position arrays, in the pre unit."""
    pre_val, unit = u.split_mantissa_unit(pre_positions)
    post_val = u.Quantity(post_positions).to(unit).mantissa
    pre_val = np.asarray(pre_val, dtype=np.float64)
    post_val = np.asarray(post_val, dtype=np.float64)
    if pre_val.ndim == 1:
        pre_val = pre_val[:, None]
    if post_val.ndim == 1:
        post_val = post_val[:, None]
    return pre_val, post_val, unit


def _profile_cutoff(profile):
    """Distance beyond which ``profile.probability`` is exactly zero, or None."""
    if type(profile) is StepProfile:
        return profile.threshold if profile.outside_prob == 0 else None
    if type(profile) in _MAX_DISTANCE_PROFILES:
        return getattr(profile, 'max_distance', None)
    return None


def _chunks_by_counts(counts: np.ndarray, max_pairs: Optional[int] = None) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` row ranges whose summed ``counts`` stay within ``max_pairs``.

    ``max_pairs`` defaults to the module's ``MAX_CHUNK_PAIRS``. Every chunk
    holds at least one row, so a single row heavier than ``max_pairs`` forms a
    chunk of its own.
    """
    max_pairs = MAX_CHUNK_PAIRS if max_pairs is None else max_pairs
    n = len(counts)
    csum = np.cumsum(np.asarray(counts, dtype=np.int64))
    start = 0
    while start < n:
        base = int(csum[start - 1]) if start else 0
        stop = int(np.searchsorted(csum, base + int(max_pairs), side='right'))
        stop = min(max(stop, start + 1), n)
        yield start, stop
        start = stop


def _pairs_within(
    pre_val: np.ndarray,
    post_val: np.ndarray,
    radius: float,
    max_pairs: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield all ``(pre, post, distance)`` pairs with ``distance <= radius``, in pre-row chunks.

    The candidate count of every pre row is obtained first (O(N) memory), and the
    rows are grouped so that each chunk holds at most ``max_pairs`` pairs (or one
    row). Chunks cover increasing ``pre`` ranges, but the pairs inside a chunk
    come in KD-tree traversal order (deterministic for given inputs); callers
    sort what they keep with :func:`_sort_pairs`, which is far cheaper than
    sorting every candidate.
    """
    n_post = len(post_val)
    if len(pre_val) == 0 or n_post == 0:
        return
    post_tree = cKDTree(post_val)
    counts = np.asarray(post_tree.query_ball_point(pre_val, radius, return_length=True))
    for start, stop in _chunks_by_counts(counts, max_pairs):
        if int(np.sum(counts[start:stop])) == 0:
            continue
        found = cKDTree(pre_val[start:stop]).sparse_distance_matrix(post_tree, radius, output_type='ndarray')
        yield found['i'] + start, found['j'], found['v']


def _sort_pairs(pre: np.ndarray, post: np.ndarray, n_post: int, *extra):
    """Sort ``pre``, ``post`` and parallel arrays by ``(pre, post)``."""
    order = np.argsort(pre * n_post + post)
    return (pre[order], post[order]) + tuple(e[order] for e in extra)


def _sorted_unique(values: np.ndarray) -> np.ndarray:
    """Sorted distinct values of an integer array.

    Same result as ``np.unique``, but always sort-based: NumPy >= 2.3 routes
    ``np.unique`` through a hash table that is an order of magnitude slower on
    the millions of encoded pair ids handled here.
    """
    values = np.sort(np.asarray(values).ravel())
    if values.size == 0:
        return values
    keep: np.ndarray = np.empty(values.size, dtype=bool)
    keep[0] = True
    np.not_equal(values[1:], values[:-1], out=keep[1:])
    return values[keep]


def _concat_pairs(pres, posts, *extra):
    """Concatenate per-chunk index lists (int64) and parallel float lists (float64)."""
    out = [np.concatenate(pres).astype(np.int64, copy=False) if pres else np.zeros(0, np.int64),
           np.concatenate(posts).astype(np.int64, copy=False) if posts else np.zeros(0, np.int64)]
    for values in extra:
        out.append(np.concatenate(values).astype(np.float64, copy=False) if values else np.zeros(0))
    return tuple(out)


def _drop_self_pairs(pre: np.ndarray, post: np.ndarray, *extra):
    """Remove pairs with ``pre == post`` from ``pre``, ``post`` and any parallel arrays."""
    keep = pre != post
    return (pre[keep], post[keep]) + tuple(e[keep] for e in extra)


def _empty_point_result(pre_size, post_size, pre_positions, post_positions) -> ConnectionResult:
    return ConnectionResult(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        pre_size=pre_size,
        post_size=post_size,
        pre_positions=pre_positions,
        post_positions=post_positions,
        model_type='point',
    )


def _same_population(pre_positions, post_positions, pre_num, post_num) -> bool:
    """Heuristically decide whether pre and post are the same population.

    Treated as the same population (so that ``pre_idx == post_idx`` is an autapse)
    when the same position array object is passed for both, or when the populations
    have equal size and identical positions.
    """
    if pre_positions is post_positions and pre_positions is not None:
        return True
    if pre_num != post_num:
        return False
    if pre_positions is None or post_positions is None:
        return False
    pre_val = u.get_mantissa(pre_positions)
    post_val = u.get_mantissa(post_positions)
    return np.array_equal(pre_val, post_val)


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

    .. versionchanged:: 0.7.0
       Generation is chunked over pre neurons and never holds an
       ``N_pre x N_post`` array. When the profile has a hard cutoff (the
       ``max_distance`` of the built-in profiles, or the ``threshold`` of a
       ``StepProfile`` with ``outside_prob=0``) and positions are given, only
       the pairs within the cutoff are found (KD-tree search) and drawn, which
       changes the edges a given ``seed`` produces in that case. Without a
       cutoff, the edges for a given ``seed`` are unchanged.

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
    allow_self_connections : bool, default=False
        Whether to keep self-connections (autapses) when ``pre`` and ``post`` are
        the same population (the same position array object, or equal-size
        populations with identical positions).
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
    - Distances are computed with the Euclidean metric over **all** position dimensions
      (e.g. a 3D position array uses x, y and z), not just the first two
    - The actual number of connections is stochastic and depends on the distance profile
      and random sampling
    - Empty connection results are returned if no connections are established
    - When ``pre`` and ``post`` are the same population, self-connections (autapses) are
      excluded by default; pass ``allow_self_connections=True`` to keep them
    - Connection probabilities are expected in ``[0, 1]`` and are clipped to this range
    - Edges are returned sorted by ``(pre, post)``
    - Without a cutoff, distances and probabilities are evaluated in blocks of
      pre neurons holding at most ``2**22`` pairs, so memory is bounded but the
      time is ``O(N_pre * N_post)``. With a cutoff, a ``scipy.spatial.cKDTree``
      search visits only the pairs within it, so time and memory scale with
      the number of candidate pairs (e.g. 1e5 neurons on a 2-D sheet with
      about 1e7 edges build in seconds)
    - A pre-computed ``distances`` matrix is processed in the same row blocks
    - This is host-side NumPy/SciPy build-time code; it is not traceable by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``

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
        >>> positions = np.random.RandomState(0).uniform(0, 1000, (500, 2)) * u.um
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
        Generated 12864 connections

    Using exponential distance profile:

    .. code-block:: python

        >>> from braintools.init import ExponentialProfile
        >>>
        >>> conn = DistanceDependent(
        ...     distance_profile=ExponentialProfile(
        ...         decay_constant=150 * u.um,
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
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # If the profile exposes a positive scale parameter, validate it so that a
        # non-positive sigma/decay_constant surfaces a clear error instead of NaN/empty.
        for attr in ('sigma', 'decay_constant'):
            val = getattr(distance_profile, attr, None)
            if val is not None and u.get_mantissa(val) <= 0:
                raise ValueError(f"distance_profile.{attr} must be > 0, got {val}.")
        self.distance_profile = distance_profile
        self.weight_init = weight
        self.delay_init = delay
        self.allow_self_connections = allow_self_connections

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate distance-dependent connections."""

        pre_num = _size_num(pre_size)
        post_num = _size_num(post_size)

        distances = kwargs.get('distances', None)
        if distances is None:
            if pre_positions is None or post_positions is None:
                raise ValueError('Positions required for distance-dependent connectivity, '
                                 'for example: pre_positions=positions, post_positions=positions, '
                                 'or provide distances directly.')
            if len(pre_positions) != pre_num:
                raise ValueError(
                    f"Number of pre_positions ({len(pre_positions)}) must equal "
                    f"pre_size ({pre_num})."
                )
            if len(post_positions) != post_num:
                raise ValueError(
                    f"Number of post_positions ({len(post_positions)}) must equal "
                    f"post_size ({post_num})."
                )

        drop_self = not self.allow_self_connections and _same_population(
            pre_positions, post_positions, pre_num, post_num
        )
        cutoff = _profile_cutoff(self.distance_profile) if distances is None else None
        if cutoff is None:
            pre_indices, post_indices, dist_val, dist_unit = self._sample_dense(
                pre_num, post_num, pre_positions, post_positions, distances
            )
        else:
            pre_indices, post_indices, dist_val, dist_unit = self._sample_cutoff(
                pre_positions, post_positions, cutoff
            )
        if drop_self:
            pre_indices, post_indices, dist_val = _drop_self_pairs(pre_indices, post_indices, dist_val)

        if len(pre_indices) == 0:
            return _empty_point_result(pre_size, post_size, pre_positions, post_positions)

        n_connections = len(pre_indices)
        connection_distances = u.maybe_decimal(dist_val * dist_unit)

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
            pre_indices,
            post_indices,
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

    def _connect(self, distances, n_draws: int, shape) -> np.ndarray:
        """Draw the connection mask for a block of distances (one uniform per pair)."""
        probs = self.distance_profile.probability(distances)
        # Probabilities are expected in [0, 1]; clip to guard against profiles or
        # custom arrays that exceed this range (e.g. distance 0 -> prob slightly >1).
        probs = np.clip(probs, 0.0, 1.0)
        random_vals = self.rng.random(n_draws).reshape(shape)
        return np.asarray((probs > 0) & (random_vals < probs), dtype=bool)

    def _sample_dense(self, pre_num, post_num, pre_positions, post_positions, distances):
        """Evaluate all pairs, in pre-row chunks of at most ``MAX_CHUNK_PAIRS`` pairs.

        One uniform is drawn per pair in row-major order, which reproduces the
        random stream of the former single-matrix implementation exactly.
        """
        if distances is not None:
            dist_all, dist_unit = u.split_mantissa_unit(distances)
            pos_val = None
        else:
            pre_val, post_val, dist_unit = _position_values(pre_positions, post_positions)
            pos_val = (pre_val, post_val)
        pres, posts, dists = [], [], []
        for start, stop in row_chunks(pre_num, post_num, MAX_CHUNK_PAIRS):
            if pos_val is None:
                block_val = np.asarray(dist_all[start:stop], dtype=np.float64)
            else:
                block_val = cdist(pos_val[0][start:stop], pos_val[1])
            mask = self._connect(u.maybe_decimal(block_val * dist_unit), block_val.size, block_val.shape)
            rows, cols = np.nonzero(mask)
            pres.append(rows + start)
            posts.append(cols)
            dists.append(block_val[rows, cols])
        return _concat_pairs(pres, posts, dists) + (dist_unit,)

    def _sample_cutoff(self, pre_positions, post_positions, cutoff):
        """Evaluate only the pairs within the profile's cutoff, found with a KD-tree."""
        pre_val, post_val, dist_unit = _position_values(pre_positions, post_positions)
        radius = float(u.Quantity(cutoff).to(dist_unit).mantissa)
        radius = radius * (1.0 + _CUTOFF_SLACK)
        pres, posts, dists = [], [], []
        for pre, post, dist in _pairs_within(pre_val, post_val, radius):
            mask = self._connect(u.maybe_decimal(dist * dist_unit), dist.size, dist.shape)
            # Chunks cover increasing pre ranges, so sorting each chunk's kept
            # pairs makes the concatenation sorted by (pre, post).
            pre, post, dist = _sort_pairs(pre[mask], post[mask], len(post_val), dist[mask])
            pres.append(pre)
            posts.append(post)
            dists.append(dist)
        return _concat_pairs(pres, posts, dists) + (dist_unit,)


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

    .. versionchanged:: 0.7.0
       With ``max_distance`` set, only the pairs within it are found (KD-tree
       search, chunked over pre neurons) and drawn, so large sheets build in
       time and memory proportional to the candidate pairs; this changes the
       edges a given ``seed`` produces. Without ``max_distance`` the edges are
       unchanged. See :class:`DistanceDependent`.

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

        >>> from braintools.init import DistanceModulated, Normal, GaussianProfile
        >>>
        >>> broad_conn = Gaussian(
        ...     distance_profile=GaussianProfile(
        ...         sigma=200 * u.um,  # Broader spread
        ...         max_distance=600 * u.um
        ...     ),
        ...     weight=DistanceModulated(
        ...         base_dist=Normal(mean=3.0*u.nS, std=0.5*u.nS),
        ...         distance_profile=GaussianProfile(sigma=100*u.um),
        ...     )
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

    .. versionchanged:: 0.7.0
       With ``max_distance`` set, only the pairs within it are found (KD-tree
       search, chunked over pre neurons) and drawn, so large sheets build in
       time and memory proportional to the candidate pairs; this changes the
       edges a given ``seed`` produces. Without ``max_distance`` the edges are
       unchanged. See :class:`DistanceDependent`.

    Parameters
    ----------
    distance_profile : ExponentialProfile
        An exponential distance profile that defines the spatial connection probability decay.
        Must be an instance of ``braintools.init.ExponentialProfile`` with:

        - ``decay_constant``: Characteristic decay length (λ) - larger values create longer-range connections
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
    - The ``decay_constant`` parameter (λ) determines how quickly connection probability decays:

      - At distance λ, probability is reduced to ~37% (1/e) of the maximum
      - Smaller decay_constant values create more localized connectivity
      - Larger decay_constant values allow longer-range connections

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
        >>> # Exponential connectivity with decay_constant=150um
        >>> conn = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         decay_constant=150 * u.um,
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
        ...         decay_constant=75 * u.um,  # Short decay length
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
        ...         decay_constant=300 * u.um,  # Long decay length
        ...         max_distance=900 * u.um
        ...     ),
        ...     weight=Normal(mean=1.0*u.nS, std=0.2*u.nS),
        ...     seed=42
        ... )

    Distance-dependent delays:

    .. code-block:: python

        >>> from braintools.init import DistanceModulated, Constant, Normal, GaussianProfile
        >>>
        >>> # Modulate delays by distance using a Gaussian profile
        >>> conn_with_delays = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         decay_constant=200 * u.um,
        ...         max_distance=600 * u.um
        ...     ),
        ...     weight=Constant(1.0 * u.nS),
        ...     delay=DistanceModulated(
        ...         base_dist=Normal(mean=1.0*u.ms, std=0.2*u.ms),
        ...         distance_profile=GaussianProfile(sigma=200*u.um),
        ...     )
        ... )

    Modeling hippocampal connectivity:

    .. code-block:: python

        >>> from braintools.init import LogNormal
        >>>
        >>> # Exponential connectivity pattern typical of hippocampal networks
        >>> positions_1d = np.linspace(0, 2000, 200).reshape(-1, 1) * u.um
        >>>
        >>> hippo_conn = Exponential(
        ...     distance_profile=ExponentialProfile(
        ...         decay_constant=250 * u.um,  # Moderate spatial extent
        ...         max_distance=1000 * u.um
        ...     ),
        ...     weight=LogNormal(mean=1.5*u.nS, std=0.4*u.nS),
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
    - ``neighbors`` must satisfy ``1 <= neighbors <= (N - 1) // 2``, otherwise a
      ``ValueError`` is raised
    - Edges are built with vectorised index arithmetic (``O(N * neighbors)``)
      and returned sorted by ``(pre, post)``; the output is deterministic and
      identical to earlier versions
    - This is host-side NumPy build-time code; it is not traceable by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``

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
        >>> print(f"Total connections: {len(result.pre_indices)}")
        Total connections: 400

    Unidirectional ring (feed-forward):

    .. code-block:: python

        >>> # Only forward connections
        >>> ff_ring = Ring(
        ...     neighbors=1,
        ...     weight=Constant(1.5 * u.nS),
        ...     bidirectional=False
        ... )
        >>> result = ff_ring(pre_size=100, post_size=100)
        >>> print(f"Total connections: {len(result.pre_indices)}")
        Total connections: 100

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

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate ring connectivity."""

        if isinstance(pre_size, tuple):
            n = int(np.prod(pre_size))
        else:
            n = pre_size

        if pre_size != post_size:
            raise ValueError("Ring networks require pre_size == post_size")

        if not (1 <= self.neighbors <= (n - 1) // 2):
            raise ValueError(
                f"neighbors must satisfy 1 <= neighbors <= (n-1)//2 = {(n - 1) // 2} "
                f"for a ring of {n} neurons, got {self.neighbors}."
            )

        # Index arithmetic: row i holds (i + offset) % n for offset in +-[1, neighbors].
        # Since 2 * neighbors < n, these targets are distinct and never equal i, so
        # sorting each row gives the edges in (pre, post) order without dedup.
        n = int(n)
        offsets: np.ndarray = np.arange(1, int(self.neighbors) + 1, dtype=np.int64)
        if self.bidirectional:
            offsets = np.concatenate([offsets, -offsets])
        targets = (np.arange(n, dtype=np.int64)[:, None] + offsets[None, :]) % n
        targets.sort(axis=1)
        pre_indices: np.ndarray = np.repeat(np.arange(n, dtype=np.int64), offsets.size)
        post_indices = targets.ravel()

        n_connections = len(pre_indices)

        # Generate weights and delays
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

    - On periodic grids with a side of at most 2, offsets that wrap onto the
      same cell give one edge, not duplicates
    - Edges are built with vectorised index arithmetic (``O(rows * cols)``) and
      returned sorted by ``(pre, post)``; the output is deterministic and
      identical to earlier versions
    - This is host-side NumPy build-time code; it is not traceable by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``

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
        >>> print(f"Connections per neuron: {len(result.pre_indices) // 400}")
        Connections per neuron: 8

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

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate grid connectivity."""
        if not isinstance(pre_size, (tuple, list)):
            raise ValueError("Grid2d networks require pre_size and post_size to be a (rows, cols) tuple")
        if pre_size != post_size:
            raise ValueError("Grid2d networks require pre_size == post_size")
        if len(pre_size) != 2:
            raise ValueError("Grid2d networks require pre_size and post_size to be 2D shapes")

        rows, cols = pre_size

        # Define neighbor offsets
        if self.connectivity == 'von_neumann':
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif self.connectivity == 'moore':
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            raise ValueError(f"Unknown connectivity type: {self.connectivity}")

        # Index arithmetic over the (n_cells, n_offsets) target table. Invalid
        # targets (off-grid when not periodic, or the cell itself) become -1; after
        # sorting each row, repeated targets (periodic wrap on grids with a side
        # <= 2) are adjacent and dropped, so rows come out in (pre, post) order.
        rows, cols = int(rows), int(cols)
        offsets = np.asarray(offsets, dtype=np.int64)
        source: np.ndarray = np.arange(rows * cols, dtype=np.int64)
        row_of, col_of = np.divmod(source, cols) if cols > 0 else (source, source)
        target_row = row_of[:, None] + offsets[None, :, 0]
        target_col = col_of[:, None] + offsets[None, :, 1]
        if self.periodic:
            target_row %= rows
            target_col %= cols
            target = target_row * cols + target_col
        else:
            on_grid = (target_row >= 0) & (target_row < rows) & (target_col >= 0) & (target_col < cols)
            target = np.where(on_grid, target_row * cols + target_col, -1)
        target[target == source[:, None]] = -1
        target.sort(axis=1)
        keep = target >= 0
        keep[:, 1:] &= target[:, 1:] != target[:, :-1]
        pre_indices = np.broadcast_to(source[:, None], target.shape)[keep]
        post_indices = target[keep]

        n_connections = len(pre_indices)

        # Generate weights and delays
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

    A post neuron that lies in several patches of the same pre neuron gets one
    independent ``prob`` draw per patch, so it is connected with probability
    ``1 - (1 - prob) ** m`` when it lies in ``m`` patches.

    .. versionchanged:: 0.7.0
       Vectorised: patch centres are drawn for all pre neurons at once as an
       ``(N_pre, n_patches)`` table, the patch members come from a chunked
       KD-tree search, and duplicates are removed through encoded pair ids.
       The connection distribution is unchanged, but the edges a given
       ``seed`` produces differ from earlier versions, and the edges are now
       returned sorted by ``(pre, post)``.

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
    allow_self_connections : bool, default=False
        Whether to keep self-connections (autapses) when ``pre`` and ``post`` are
        the same population (the same position array object, or equal-size
        populations with identical positions).
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
    - When ``pre`` and ``post`` are the same population, self-connections (autapses) are
      excluded by default; pass ``allow_self_connections=True`` to keep them
    - The actual number of connections is stochastic and depends on:

      - Spatial distribution of neurons
      - Number and radius of patches
      - Connection probability ``prob``
      - Random patch center selection

    - Empty connection results are returned if no connections are established
    - This pattern models the patchy horizontal connectivity observed in cortical circuits
    - Pre neurons are processed in chunks holding at most ``2**22`` candidate
      (pre, patch, member) triples, so memory is bounded by the chunk size plus
      the output; nothing of size ``N_pre x N_post`` is built
    - Edges are returned sorted by ``(pre, post)``
    - This is host-side NumPy/SciPy build-time code; it is not traceable by
      ``jax.jit``, ``jax.vmap`` or ``jax.grad``

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
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if u.get_mantissa(patch_radius) <= 0:
            raise ValueError(f"patch_radius must be > 0, got {patch_radius}.")
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be in [0, 1], got {prob}.")
        self.patch_radius = patch_radius
        self.n_patches = n_patches
        self.prob = prob
        self.weight_init = weight
        self.delay_init = delay
        self.allow_self_connections = allow_self_connections

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate radial patch connections."""

        if pre_positions is None or post_positions is None:
            raise ValueError("Positions required for radial patch connectivity")

        pre_num = _size_num(pre_size)
        post_num = _size_num(post_size)

        _, post_val, pos_unit = _position_values(pre_positions, post_positions)
        radius_val = float(u.Quantity(self.patch_radius).to(pos_unit).mantissa)
        drop_self = not self.allow_self_connections and _same_population(
            pre_positions, post_positions, pre_num, post_num
        )
        pair_ids = self._sample_pair_ids(pre_num, post_num, post_val, radius_val, drop_self)

        if pair_ids.size == 0:
            return _empty_point_result(pre_size, post_size, pre_positions, post_positions)

        pre_indices, post_indices = np.divmod(pair_ids, post_num)
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

    def _sample_pair_ids(self, pre_num, post_num, post_val, radius, drop_self) -> np.ndarray:
        """Return the sorted, distinct encoded edges ``pre * post_num + post``.

        Patch centres are an ``(pre_num, n_centres)`` table of distinct post
        indices per row. Rows are processed in chunks whose candidate triples
        (pre, patch, post-within-radius) number at most ``MAX_CHUNK_PAIRS``; the
        neighbourhoods of the chunk's distinct centres come from one KD-tree
        search, and each triple is kept with probability ``prob``.
        """
        n_centres = min(int(self.n_patches), post_num)
        if n_centres <= 0 or pre_num == 0 or self.prob <= 0:
            return np.zeros(0, dtype=np.int64)
        centres = sample_rows_without_replacement(self.rng, pre_num, post_num, n_centres)
        post_tree = cKDTree(post_val)
        degree = np.asarray(post_tree.query_ball_point(post_val, radius, return_length=True))
        row_load = degree[centres].sum(axis=1)

        chunks = []
        for start, stop in _chunks_by_counts(row_load):
            slots = centres[start:stop].ravel()
            uniq = _sorted_unique(slots)
            inverse = np.searchsorted(uniq, slots)
            found = cKDTree(post_val[uniq]).sparse_distance_matrix(post_tree, radius, output_type='ndarray')
            order = np.lexsort((found['j'], found['i']))
            neighbours = found['j'][order]
            n_nbr = np.bincount(found['i'], minlength=uniq.size)
            nbr_start = np.cumsum(n_nbr) - n_nbr
            # Expand every (row, patch) slot into the neighbour list of its centre.
            # Every centre lies in its own patch (distance 0), so total >= 1.
            counts = n_nbr[inverse]
            total = int(counts.sum())
            slot_offset = np.cumsum(counts) - counts
            flat = np.repeat(nbr_start[inverse] - slot_offset, counts) + np.arange(total)
            post = neighbours[flat]
            pre: np.ndarray = np.repeat(np.repeat(np.arange(start, stop, dtype=np.int64), n_centres), counts)
            if self.prob < 1.0:
                keep = self.rng.random(total) < self.prob
                pre, post = pre[keep], post[keep]
            if drop_self:
                pre, post = _drop_self_pairs(pre, post)
            chunks.append(_sorted_unique(pre * post_num + post))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
