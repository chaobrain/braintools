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
Multi-compartment neuron connectivity patterns.

This module provides connectivity patterns specifically designed for multi-compartment
neuron models with detailed morphology. These patterns can target specific compartments
(soma, dendrites, axon) and implement realistic synaptic placement and dynamics.

Key Features:
- Compartment-specific connectivity (soma, dendrites, axon targeting)
- Dendritic tree connectivity with branch-specific patterns
- Axonal projection patterns and targeting rules
- Morphology-aware spatial connectivity
- Realistic synaptic placement based on neuron morphology

Memory and time: no generator builds an ``N_pre x N_post`` array. Uniform
connection probabilities draw a binomial edge count and then that many distinct
pairs, so the cost scales with the number of edges. Pair-specific probabilities
(distance profiles, topographic maps, arborization fields) are evaluated in
pre-neuron chunks of at most ``MAX_CHUNK_PAIRS`` pairs; drawing their uniforms
chunk by chunk consumes the random stream in the same order as a single
full-matrix draw, so those classes give the same output as before.
"""

import warnings
from typing import Optional, Tuple, Union, Dict, List, Callable

import brainunit as u
import numpy as np
from scipy.spatial.distance import cdist

from braintools._deprecation import register_deprecation, warn_deprecated
from braintools.init import DistanceProfile, ExponentialProfile, GaussianProfile, LinearProfile
from braintools.init._init_base import param, Initializer
from ._base import MultiCompartmentConnectivity, ConnectionResult
from ._sampling import (
    MAX_CHUNK_PAIRS,
    bernoulli_pairs,
    randint,
    row_chunks,
    sample_rows_without_replacement,
)

__all__ = [
    'SOMA',
    'BASAL_DENDRITE',
    'APICAL_DENDRITE',
    'AXON',

    # Basic compartment patterns
    'CompartmentSpecific',
    'AllToAllCompartments',

    # Anatomical targeting patterns
    'SomaToDendrite',
    'AxonToSoma',
    'DendriteToSoma',
    'AxonToDendrite',
    'DendriteToDendrite',

    # Morphology-aware patterns
    'ProximalTargeting',
    'DistalTargeting',
    'BranchSpecific',
    'MorphologyDistance',

    # Dendritic patterns
    'DendriticTree',
    'BasalDendriteTargeting',
    'ApicalDendriteTargeting',
    'DendriticIntegration',

    # Axonal patterns
    'AxonalProjection',
    'AxonalBranching',
    'AxonalArborization',
    'TopographicProjection',

    # Synaptic patterns
    'SynapticPlacement',
    'SynapticClustering',

    # Custom patterns
    'CustomCompartment',
]

# Compartment type constants
SOMA = 0
BASAL_DENDRITE = 1
APICAL_DENDRITE = 2
AXON = 3

COMPARTMENT_NAMES = {
    SOMA: 'soma',
    BASAL_DENDRITE: 'basal_dendrite',
    APICAL_DENDRITE: 'apical_dendrite',
    AXON: 'axon'
}


def _validate_prob(prob, name='connection_prob'):
    """Validate that a probability is numeric and in [0, 1]."""
    if not isinstance(prob, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric, got {type(prob)}")
    if prob < 0 or prob > 1:
        raise ValueError(f"{name} must be in [0, 1], got {prob}")


def _validate_non_negative_int(value, name):
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value)}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _as_2d_positions(positions):
    """Reshape 1-D position arrays to (N, 1) so cdist works.

    cdist requires 2-D inputs; a flat ``(N,)`` array of scalar coordinates is
    reshaped to ``(N, 1)``.
    """
    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(-1, 1)
    return positions


def _check_position_counts(owner, pre_values, post_values, pre_num, post_num):
    """Raise ``ValueError`` unless there is exactly one position per neuron.

    ``pre_values``/``post_values`` are the 2-D position arrays and
    ``pre_num``/``post_num`` the flattened population sizes.
    """
    if len(pre_values) != pre_num:
        raise ValueError(
            f'{owner} got {len(pre_values)} pre_positions for pre_size {pre_num}; '
            f'one position per neuron is required.'
        )
    if len(post_values) != post_num:
        raise ValueError(
            f'{owner} got {len(post_values)} post_positions for post_size {post_num}; '
            f'one position per neuron is required.'
        )


def _length_and_positions(length, pre_positions, post_positions):
    """Express a length scale and both position arrays as plain numbers in one unit.

    A Quantity ``length`` sets the unit and the positions are converted to it. A
    plain-number ``length`` is taken to be in the unit of ``pre_positions`` (the
    convention of the kernel connectors), so ``150.0`` with micrometre positions
    means 150 um. Returns ``(length_value, pre_values, post_values)`` with the
    positions reshaped to 2-D for ``cdist``.
    """
    if isinstance(length, u.Quantity):
        length_val, unit = u.split_mantissa_unit(length)
        pre_val = u.Quantity(pre_positions).to(unit).mantissa
        post_val = u.Quantity(post_positions).to(unit).mantissa
    else:
        length_val = length
        pre_val, unit = u.split_mantissa_unit(pre_positions)
        post_val = u.Quantity(post_positions).to(unit).mantissa
    return (
        float(length_val),
        _as_2d_positions(pre_val),
        _as_2d_positions(post_val),
    )


def _chunks(n_pre, n_post):
    """Pre-row chunks of at most ``MAX_CHUNK_PAIRS`` pairs (read at call time)."""
    return row_chunks(n_pre, n_post, MAX_CHUNK_PAIRS)


def _uniform_pairs(rng, pre_num, post_num, prob, exclude_self):
    """Independent Bernoulli(``prob``) pairs without a dense mask.

    Draws the edge count from a binomial and then that many distinct pairs,
    sorted by ``(pre, post)``. With ``exclude_self`` (square blocks only) the
    pairs ``(i, i)`` are never drawn.
    """
    return bernoulli_pairs(rng, pre_num, post_num, prob, allow_self_connections=not exclude_self)


def _chunked_mask_pairs(rng, pre_num, post_num, chunk_probs):
    """Draw ``uniform < probability`` for every pair, one pre-row chunk at a time.

    ``chunk_probs(start, stop)`` returns the ``(stop - start, post_num)``
    probabilities (or a boolean ``(probs, eligible)`` tuple) of that chunk. The
    uniforms are drawn chunk by chunk in row order, which consumes ``rng``
    exactly like one ``rng.random((pre_num, post_num))`` call.

    Returns
    -------
    pre, post : np.ndarray
        int64 indices sorted by ``(pre, post)``.
    """
    pre_parts, post_parts = [], []
    for start, stop in _chunks(pre_num, post_num):
        probs = chunk_probs(start, stop)
        eligible = None
        if isinstance(probs, tuple):
            probs, eligible = probs
        mask = rng.random((stop - start, post_num)) < probs
        if eligible is not None:
            mask &= eligible
        i, j = np.nonzero(mask)
        pre_parts.append(i.astype(np.int64) + start)
        post_parts.append(j.astype(np.int64))
    if not pre_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(pre_parts), np.concatenate(post_parts)


def _unique_rows(values):
    """``np.unique(values, axis=0)`` with a flat inverse index."""
    uniq, inverse = np.unique(values, axis=0, return_inverse=True)
    return uniq, np.asarray(inverse).reshape(-1)


def _evaluate_pair_function(func, pre_values, post_unique, post_inverse):
    """Evaluate a scalar ``func(pre_row, post_row)`` on all pairs of a chunk.

    The function is called once per pair of *distinct* positions (rows that
    compare equal share one call); the results are scattered back to the
    ``(len(pre_values), len(post_inverse))`` pair grid.
    """
    pre_unique, pre_inverse = _unique_rows(pre_values)
    grid = np.empty((len(pre_unique), len(post_unique)), dtype=float)
    for a, pre_row in enumerate(pre_unique):
        for b, post_row in enumerate(post_unique):
            grid[a, b] = func(pre_row, post_row)
    return grid[pre_inverse][:, post_inverse]


def _uniform_branches(rng, n_branches, post_num, exclude_self):
    """Uniform distinct targets for every axon, ``n_branches[i]`` for axon ``i``.

    Counts are capped at the number of allowed targets. Axons with the same
    count are sampled together (a loop over the few distinct counts), so the
    work is ``O(total branches)``. Returns ``(pre, post)`` sorted by pre, post.
    """
    available = max(post_num - (1 if exclude_self else 0), 0)
    n_branches = np.minimum(n_branches, available)
    pre_parts, post_parts = [], []
    for k in np.unique(n_branches[n_branches > 0]):
        rows = np.nonzero(n_branches == k)[0].astype(np.int64)
        targets = sample_rows_without_replacement(
            rng, rows.size, post_num, int(k), exclude=rows if exclude_self else None
        )
        pre_parts.append(np.repeat(rows, int(k)))
        post_parts.append(targets.reshape(-1))
    if not pre_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return _sorted_by_pre_post(np.concatenate(pre_parts), np.concatenate(post_parts))


def _weighted_branches(rng, n_branches, post_num, pre_pos, post_pos, spread, exclude_self):
    """Distance-weighted distinct targets for every axon, in pre-row chunks.

    Axon ``i`` picks ``n_branches[i]`` distinct targets by successive sampling
    with weights ``w = exp(-d**2 / (2 * spread**2))`` (targets whose weight
    underflows to zero, and the axon's own neuron when ``exclude_self``, are not
    eligible; the count is capped at the number of eligible targets). Successive
    sampling is done with exponential keys: the targets with the smallest
    ``E / w`` (``E ~ Exp(1)``, compared in log space) are the ones drawn first
    (Efraimidis and Spirakis, 2006). ``pre_pos`` and ``post_pos`` must hold one
    row per neuron (``len(n_branches)`` and ``post_num`` rows; checked by the
    caller). Returns ``(pre, post)`` sorted by pre, post.
    """
    pre_num = len(n_branches)
    pre_parts, post_parts = [], []
    for start, stop in _chunks(pre_num, post_num):
        counts = n_branches[start:stop]
        if not counts.any():
            continue
        distances = cdist(pre_pos[start:stop], post_pos)
        log_w = -distances ** 2 / (2 * spread ** 2)
        eligible = np.exp(log_w) > 0
        if exclude_self:
            rows = np.arange(start, min(stop, post_num))
            eligible[rows - start, rows] = False
        counts = np.minimum(counts, np.count_nonzero(eligible, axis=1))
        k_max = int(counts.max())
        if k_max == 0:
            continue
        with np.errstate(divide='ignore'):
            log_keys = np.log(rng.standard_exponential(distances.shape)) - log_w
        log_keys[~eligible] = np.inf
        if k_max < post_num:
            cand = np.argpartition(log_keys, k_max - 1, axis=1)[:, :k_max]
        else:
            cand = np.broadcast_to(np.arange(post_num), log_keys.shape)
        cand = np.take_along_axis(cand, np.argsort(np.take_along_axis(log_keys, cand, axis=1), axis=1), axis=1)
        take = np.arange(k_max)[None, :] < counts[:, None]
        rows = np.broadcast_to(np.arange(start, stop, dtype=np.int64)[:, None], cand.shape)
        pre_parts.append(rows[take])
        post_parts.append(cand[take].astype(np.int64))
    if not pre_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return _sorted_by_pre_post(np.concatenate(pre_parts), np.concatenate(post_parts))


def _sorted_by_pre_post(*arrays):
    """Sort parallel edge arrays by ``(pre, post)`` (the first two arrays)."""
    order = np.lexsort((arrays[1], arrays[0]))
    return tuple(a[order] for a in arrays)


class CompartmentSpecific(MultiCompartmentConnectivity):
    """General compartment-specific connectivity pattern.

    This is the base class for targeting specific compartments in multi-compartment
    neurons. It provides flexible mapping from source compartments to target
    compartments with customizable connection rules.

    Parameters
    ----------
    compartment_mapping : dict
        Mapping from source compartment types to target compartment types.
        Keys and values can be compartment indices (int) or names (str).
    connection_prob : float or dict
        Connection probability. Can be global or per-compartment-pair.
    weight : Initializer, optional
        Weight initializer (e.g. ``braintools.init.Normal``).
    delay : Initializer, optional
        Delay initializer (e.g. ``braintools.init.Uniform``).
    morphology_info : dict, optional
        Information about neuron morphology structure.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself (autapse). When ``False`` and the
        pre/post populations have the same size, connections with
        ``pre_idx == post_idx`` are dropped regardless of compartment. Default ``False``.

    Notes
    -----
    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Connected pairs are sampled by drawing a binomial edge count and
       then that many distinct pairs, without a dense
       ``pre_size x post_size`` matrix, so time and memory grow with the
       number of edges. The edges for a given ``seed`` differ from 0.6
       (same distribution). This also applies to the subclasses that
       delegate to it (``SomaToDendrite``, ``AxonToSoma``, ``BranchSpecific``,
       ``AllToAllCompartments`` and the other targeting shortcuts).

    Examples
    --------
    Axon-to-soma connections:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import (
        ...     CompartmentSpecific, AXON, SOMA, BASAL_DENDRITE, APICAL_DENDRITE
        ... )
        >>> from braintools.init import Normal
        >>> axon_soma = CompartmentSpecific(
        ...     compartment_mapping={AXON: SOMA},
        ...     connection_prob=0.1,
        ...     weight=Normal(mean=2.0 * u.nS, std=0.5 * u.nS),
        ...     seed=0,
        ... )
        >>> result = axon_soma(pre_size=100, post_size=100)
        >>> result.n_connections
        993
        >>> bool(np.all(result.post_compartments == SOMA))
        True

    Complex multi-compartment targeting:

    .. code-block:: python

        >>> # Dendrites to soma, axon to dendrites
        >>> multi_target = CompartmentSpecific(
        ...     compartment_mapping={
        ...         BASAL_DENDRITE: SOMA,
        ...         APICAL_DENDRITE: SOMA,
        ...         AXON: [BASAL_DENDRITE, APICAL_DENDRITE]
        ...     },
        ...     connection_prob={
        ...         (BASAL_DENDRITE, SOMA): 0.3,
        ...         (APICAL_DENDRITE, SOMA): 0.2,
        ...         (AXON, BASAL_DENDRITE): 0.05,
        ...         (AXON, APICAL_DENDRITE): 0.08
        ...     }
        ... )

    Named compartment mapping:

    .. code-block:: python

        >>> named_mapping = CompartmentSpecific(
        ...     compartment_mapping={
        ...         'axon': ['basal_dendrite', 'apical_dendrite'],
        ...         'basal_dendrite': 'soma'
        ...     },
        ...     connection_prob=0.1
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        compartment_mapping: Dict[Union[int, str], Union[int, str, List[Union[int, str]]]],
        connection_prob: Union[float, Dict] = 0.1,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        morphology_info: Optional[Dict] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.compartment_mapping = self._normalize_compartment_mapping(compartment_mapping)
        self._validate_connection_prob(connection_prob)
        self.connection_prob = connection_prob
        self.weight_init = weight
        self.delay_init = delay
        self.morphology_info = morphology_info or {}
        self.allow_self_connections = allow_self_connections

    @staticmethod
    def _validate_compartment_index(idx: int, context: str = "compartment") -> None:
        """Validate that a compartment index is in valid range."""
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(f"{context} index must be an integer, got {type(idx)}")
        if idx < 0 or idx > 3:
            raise ValueError(
                f"{context} index must be in range [0, 3] (SOMA=0, BASAL_DENDRITE=1, APICAL_DENDRITE=2, AXON=3), got {idx}")

    def _validate_connection_prob(self, prob: Union[float, Dict]) -> None:
        """Validate connection probability values."""
        if isinstance(prob, dict):
            for key, value in prob.items():
                if not isinstance(value, (int, float, np.number)):
                    raise TypeError(f"Connection probability must be numeric, got {type(value)} for key {key}")
                if value < 0 or value > 1:
                    raise ValueError(f"Connection probability must be in [0, 1], got {value} for key {key}")
        else:
            if not isinstance(prob, (int, float, np.number)):
                raise TypeError(f"Connection probability must be numeric, got {type(prob)}")
            if prob < 0 or prob > 1:
                raise ValueError(f"Connection probability must be in [0, 1], got {prob}")

    def _normalize_compartment_mapping(self, mapping):
        """Convert compartment names to indices if needed and validate."""
        name_to_idx = {v: k for k, v in COMPARTMENT_NAMES.items()}
        normalized = {}

        for source, targets in mapping.items():
            # Convert source
            if isinstance(source, str):
                if source not in name_to_idx:
                    raise ValueError(f"Unknown compartment name: '{source}'. Valid names: {list(name_to_idx.keys())}")
                source_idx = name_to_idx[source]
            else:
                source_idx = source
                self._validate_compartment_index(source_idx, f"Source compartment")

            # Convert targets
            if isinstance(targets, (list, tuple)):
                target_indices = []
                for target in targets:
                    if isinstance(target, str):
                        if target not in name_to_idx:
                            raise ValueError(
                                f"Unknown compartment name: '{target}'. Valid names: {list(name_to_idx.keys())}")
                        target_idx = name_to_idx[target]
                    else:
                        target_idx = target
                        self._validate_compartment_index(target_idx, f"Target compartment")
                    target_indices.append(target_idx)
                normalized[source_idx] = target_indices
            else:
                if isinstance(targets, str):
                    if targets not in name_to_idx:
                        raise ValueError(
                            f"Unknown compartment name: '{targets}'. Valid names: {list(name_to_idx.keys())}")
                    target_idx = name_to_idx[targets]
                else:
                    target_idx = targets
                    self._validate_compartment_index(target_idx, f"Target compartment")
                normalized[source_idx] = target_idx

        return normalized

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate compartment-specific connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        all_pre_indices = []
        all_post_indices = []
        all_pre_compartments = []
        all_post_compartments = []

        # Pre-filter compartment pairs with non-zero probabilities
        active_pairs = []
        for source_comp, target_comps in self.compartment_mapping.items():
            if not isinstance(target_comps, (list, tuple)):
                target_comps = [target_comps]

            for target_comp in target_comps:
                # Get connection probability for this compartment pair
                if isinstance(self.connection_prob, dict):
                    if (source_comp, target_comp) not in self.connection_prob:
                        warnings.warn(
                            f"No connection probability provided for mapped pair "
                            f"({source_comp}, {target_comp}); this pair is dropped "
                            f"(probability defaults to 0.0).",
                            UserWarning,
                        )
                    prob = self.connection_prob.get((source_comp, target_comp), 0.0)
                else:
                    prob = self.connection_prob

                if prob > 0:
                    active_pairs.append((source_comp, target_comp, prob))

        # Self-connections (autapses) are never drawn when not allowed and the
        # populations match.
        exclude_self = not self.allow_self_connections and pre_num == post_num

        # Generate connections only for active pairs (binomial edge count, then
        # distinct pairs: no pre x post mask is built).
        for source_comp, target_comp, prob in active_pairs:
            pre_idx, post_idx = _uniform_pairs(self.rng, pre_num, post_num, prob, exclude_self)

            n_conns = len(pre_idx)
            if n_conns > 0:
                all_pre_indices.append(pre_idx)
                all_post_indices.append(post_idx)
                all_pre_compartments.append(np.full(n_conns, source_comp, dtype=np.int64))
                all_post_compartments.append(np.full(n_conns, target_comp, dtype=np.int64))

        if len(all_pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        # Concatenate all connections
        pre_indices = np.concatenate(all_pre_indices)
        post_indices = np.concatenate(all_post_indices)
        pre_compartments = np.concatenate(all_pre_compartments)
        post_compartments = np.concatenate(all_post_compartments)

        # Drop self-connections (autapses) when not allowed and populations match.
        if not self.allow_self_connections and pre_num == post_num:
            keep = pre_indices != post_indices
            pre_indices = pre_indices[keep]
            post_indices = post_indices[keep]
            pre_compartments = pre_compartments[keep]
            post_compartments = post_compartments[keep]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)

        # Generate weights and delays using init_call
        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments,
            metadata={
                'pattern': 'compartment_specific',
                'compartment_mapping': self.compartment_mapping,
                'connection_prob': self.connection_prob
            }
        )


class SomaToDendrite(CompartmentSpecific):
    """Specialized connectivity from soma to dendritic compartments.

    This creates connections specifically from soma compartments to dendritic
    compartments, commonly used for modeling antidromic activation or
    backpropagating action potentials.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.

    Parameters
    ----------
    target_dendrites : list, optional
        List of dendrite compartment types to target.
    prob_per_dendrite : float or dict
        Connection probability to each dendrite type.

    Examples
    --------
    Basic soma-to-dendrite connections:

    .. code-block:: python

        >>> from braintools.conn import SomaToDendrite, BASAL_DENDRITE, APICAL_DENDRITE
        >>> soma_dend = SomaToDendrite(
        ...     target_dendrites=[BASAL_DENDRITE, APICAL_DENDRITE],
        ...     prob_per_dendrite=0.8,
        ...     seed=0,
        ... )
        >>> result = soma_dend(pre_size=50, post_size=50)
        >>> result.n_connections
        3914

    Different probabilities for different dendrite types:

    .. code-block:: python

        >>> soma_dend_specific = SomaToDendrite(
        ...     target_dendrites=[BASAL_DENDRITE, APICAL_DENDRITE],
        ...     prob_per_dendrite={
        ...         BASAL_DENDRITE: 0.9,
        ...         APICAL_DENDRITE: 0.7
        ...     }
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        target_dendrites: List[int] = None,
        prob_per_dendrite: Union[float, Dict[int, float]] = 0.8,
        **kwargs
    ):
        if target_dendrites is None:
            target_dendrites = [BASAL_DENDRITE, APICAL_DENDRITE]

        # Create compartment mapping
        compartment_mapping = {SOMA: target_dendrites}

        # Create connection probabilities
        if isinstance(prob_per_dendrite, dict):
            connection_prob = {(SOMA, dendrite): prob for dendrite, prob in prob_per_dendrite.items()}
        else:
            connection_prob = {(SOMA, dendrite): prob_per_dendrite for dendrite in target_dendrites}

        super().__init__(
            compartment_mapping=compartment_mapping,
            connection_prob=connection_prob,
            **kwargs
        )


class AxonToSoma(CompartmentSpecific):
    """Specialized connectivity from axon to soma compartments.

    This creates connections specifically from axon compartments to soma
    compartments, representing direct synaptic input to the cell body.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import AxonToSoma
        >>> from braintools.init import LogNormal
        >>> axon_soma = AxonToSoma(
        ...     connection_prob=0.05,
        ...     weight=LogNormal(mean=1.0 * u.nS, std=0.3 * u.nS),
        ...     seed=0,
        ... )
        >>> result = axon_soma(pre_size=50, post_size=50)
        >>> result.n_connections
        127
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartment_mapping = {AXON: SOMA}
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class DendriteToSoma(CompartmentSpecific):
    """Specialized connectivity from dendrites to soma.

    This creates connections from dendritic compartments to soma compartments,
    modeling dendritic integration and signal propagation to the cell body.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.

    Parameters
    ----------
    source_dendrites : list, optional
        List of dendrite compartment types to use as sources.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.conn import DendriteToSoma, BASAL_DENDRITE, APICAL_DENDRITE
        >>> dend_soma = DendriteToSoma(
        ...     source_dendrites=[BASAL_DENDRITE, APICAL_DENDRITE],
        ...     connection_prob=0.6,
        ...     seed=0,
        ... )
        >>> result = dend_soma(pre_size=50, post_size=50)
        >>> result.n_connections
        2934
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        source_dendrites: List[int] = None,
        **kwargs
    ):
        if source_dendrites is None:
            source_dendrites = [BASAL_DENDRITE, APICAL_DENDRITE]

        compartment_mapping = {dendrite: SOMA for dendrite in source_dendrites}
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class AxonToDendrite(CompartmentSpecific):
    """Specialized connectivity from axon to dendritic compartments.

    This represents the most common type of synaptic connection where
    axons form synapses on dendritic branches.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import AxonToDendrite, BASAL_DENDRITE, APICAL_DENDRITE
        >>> from braintools.init import LogNormal
        >>> axon_dend = AxonToDendrite(
        ...     target_dendrites=[BASAL_DENDRITE, APICAL_DENDRITE],
        ...     connection_prob=0.1,
        ...     weight=LogNormal(mean=2.0 * u.nS, std=0.5 * u.nS),
        ...     seed=0,
        ... )
        >>> result = axon_dend(pre_size=50, post_size=50)
        >>> result.n_connections
        505
    """
    __module__ = 'braintools.conn'

    def __init__(self, target_dendrites: List[int] = None, **kwargs):
        if target_dendrites is None:
            target_dendrites = [BASAL_DENDRITE, APICAL_DENDRITE]

        compartment_mapping = {AXON: target_dendrites}
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class DendriteToDendrite(CompartmentSpecific):
    """Dendrite-to-dendrite connectivity patterns.

    This models lateral connections between dendritic branches,
    important for dendritic computation and integration.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.conn import DendriteToDendrite
        >>> from braintools.init import Normal
        >>> dend_dend = DendriteToDendrite(
        ...     connection_prob=0.2,
        ...     weight=Normal(mean=0.5 * u.nS, std=0.1 * u.nS),
        ...     seed=0,
        ... )
        >>> result = dend_dend(pre_size=50, post_size=50)
        >>> result.n_connections
        1993
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartment_mapping = {
            BASAL_DENDRITE: [BASAL_DENDRITE, APICAL_DENDRITE],
            APICAL_DENDRITE: [BASAL_DENDRITE, APICAL_DENDRITE]
        }
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


# Legacy ``MorphologyDistance(decay_function=<str>)`` values and the
# ``braintools.init`` profile each one maps onto (built from ``sigma``).
_LEGACY_DECAY_PROFILES = {
    'gaussian': GaussianProfile,
    'exponential': ExponentialProfile,
    'linear': LinearProfile,
}

_DECAY_STRING_DEPRECATION = register_deprecation(
    'braintools.conn.MorphologyDistance(decay_function=str)',
    'pass a braintools.init.*Profile instance',
    since='0.4.0',
    removed_in='1.0.0',
    kind='argument',
)


class MorphologyDistance(MultiCompartmentConnectivity):
    """Distance-dependent connectivity based on neuron positions.

    Connection probabilities decay with the Euclidean distance between
    presynaptic and postsynaptic neuron positions (the soma positions passed
    as ``pre_positions``/``post_positions``); the chosen target compartments
    come from ``compartment_mapping``. Positions are required.

    For every (source, target) compartment pair, each neuron pair connects
    with the probability given by the distance profile at their distance.

    Parameters
    ----------
    sigma : float or Quantity, optional
        Characteristic distance scale of the built-in Gaussian profile (and of
        the deprecated string decay functions). A plain number is interpreted
        in the unit of ``pre_positions``. Required unless ``decay_function`` is
        a :class:`braintools.init.DistanceProfile`, in which case it must be
        omitted because the profile carries its own length scale.
    decay_function : DistanceProfile, str or None, optional
        Distance-to-probability profile. ``None`` (default) is a Gaussian
        ``exp(-d**2 / (2 * sigma**2))``. A
        :class:`braintools.init.DistanceProfile` instance (for example
        ``braintools.init.ExponentialProfile(50 * u.um)``) is evaluated on the
        distances expressed in the unit of ``pre_positions``.

        .. deprecated:: 0.4.0
           The strings ``'gaussian'``, ``'exponential'`` and ``'linear'`` are
           deprecated and will be removed in 1.0.0. They map to
           ``GaussianProfile(sigma)``, ``ExponentialProfile(sigma)`` and
           ``LinearProfile(sigma)``.
    compartment_mapping : dict
        Mapping of which compartments can connect to which.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself. Default ``False``.

    Raises
    ------
    ValueError
        If ``sigma`` is missing while ``decay_function`` is not a
        ``DistanceProfile``, or given while it is. At generation time, if
        ``decay_function`` is not recognised or ``sigma`` is not positive.

    Warns
    -----
    BraintoolsDeprecationWarning
        If ``decay_function`` is one of the legacy strings.

    See Also
    --------
    braintools.init.GaussianProfile, braintools.init.ExponentialProfile,
    braintools.init.LinearProfile

    Notes
    -----
    Pair probabilities are evaluated in chunks of presynaptic rows of
    bounded size, so peak memory does not grow with
    ``pre_size * post_size``. The random draws are the same as in version
    0.6, so a given ``seed`` gives the same edges.
    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import MorphologyDistance, AXON, BASAL_DENDRITE
        >>> from braintools.init import ExponentialProfile
        >>> gaussian = MorphologyDistance(sigma=50 * u.um)
        >>> exponential = MorphologyDistance(
        ...     decay_function=ExponentialProfile(80 * u.um),
        ...     compartment_mapping={AXON: [BASAL_DENDRITE]},
        ...     seed=0,
        ... )
        >>> positions = np.random.RandomState(0).uniform(0, 200, (20, 2)) * u.um
        >>> result = exponential(
        ...     pre_size=20, post_size=20,
        ...     pre_positions=positions, post_positions=positions,
        ... )
        >>> bool(np.all(result.post_compartments == BASAL_DENDRITE))
        True
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma: Optional[Union[float, u.Quantity]] = None,
        decay_function: Optional[Union[DistanceProfile, str]] = None,
        compartment_mapping: Dict = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(decay_function, DistanceProfile):
            if sigma is not None:
                raise ValueError(
                    '`sigma` must be omitted when `decay_function` is a DistanceProfile; '
                    'the profile defines its own length scale.'
                )
        else:
            if sigma is None:
                raise ValueError(
                    '`sigma` is required unless `decay_function` is a '
                    'braintools.init.DistanceProfile instance.'
                )
            if isinstance(decay_function, str) and decay_function in _LEGACY_DECAY_PROFILES:
                warn_deprecated(_DECAY_STRING_DEPRECATION)
        self.sigma = sigma
        self.decay_function = decay_function
        self.compartment_mapping = compartment_mapping or {AXON: [BASAL_DENDRITE, APICAL_DENDRITE]}
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
        """Generate morphology-based distance-dependent connections."""
        if pre_positions is None or post_positions is None:
            raise ValueError(
                "MorphologyDistance requires both pre_positions and post_positions "
                "(distance-dependent connectivity cannot be computed without them)."
            )

        # Validate and normalize sizes FIRST
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        profile = self.decay_function
        if isinstance(profile, DistanceProfile):
            # The profile converts distances itself, so keep the position unit.
            pre_pos_val, distance_unit = u.split_mantissa_unit(pre_positions)
            post_pos_val = u.Quantity(post_positions).to(distance_unit).mantissa
            pre_pos_val = _as_2d_positions(pre_pos_val)
            post_pos_val = _as_2d_positions(post_pos_val)
        else:
            # sigma and positions as plain numbers in a common unit
            sigma_val, pre_pos_val, post_pos_val = _length_and_positions(
                self.sigma, pre_positions, post_positions
            )
            distance_unit = None
            if profile is None:
                profile = GaussianProfile(sigma_val)
            elif isinstance(profile, str) and profile in _LEGACY_DECAY_PROFILES:
                profile = _LEGACY_DECAY_PROFILES[profile](sigma_val)
            else:
                raise ValueError(f"Unknown decay function: {self.decay_function}")

        # Check position sizes and adjust before distance calculation
        if len(pre_pos_val) != pre_num:
            warnings.warn(
                f'Pre positions length {len(pre_pos_val)} does not match pre_size {pre_num}. Using min value.',
                UserWarning,
            )
            pre_num = min(len(pre_pos_val), pre_num)

        if len(post_pos_val) != post_num:
            warnings.warn(
                f'Post positions length {len(post_pos_val)} does not match post_size {post_num}. Using min value.',
                UserWarning,
            )
            post_num = min(len(post_pos_val), post_num)

        # Trim positions to actual sizes being used
        pre_pos_val = _as_2d_positions(pre_pos_val)[:pre_num]
        post_pos_val = _as_2d_positions(post_pos_val)[:post_num]

        def chunk_probs(start, stop):
            # Connection probability from the distance profile, one pre chunk at a time.
            distances = cdist(pre_pos_val[start:stop], post_pos_val)
            if distance_unit is not None and not distance_unit.is_unitless:
                distances = distances * distance_unit
            return np.asarray(profile.probability(distances))

        all_pre_indices = []
        all_post_indices = []
        all_pre_compartments = []
        all_post_compartments = []

        # Generate connections for each compartment mapping
        for source_comp, target_comps in self.compartment_mapping.items():
            if not isinstance(target_comps, (list, tuple)):
                target_comps = [target_comps]

            for target_comp in target_comps:
                pre_idx, post_idx = _chunked_mask_pairs(self.rng, pre_num, post_num, chunk_probs)

                n_conns = len(pre_idx)
                if n_conns > 0:
                    all_pre_indices.append(pre_idx)
                    all_post_indices.append(post_idx)
                    all_pre_compartments.append(np.full(n_conns, source_comp, dtype=np.int64))
                    all_post_compartments.append(np.full(n_conns, target_comp, dtype=np.int64))

        if len(all_pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        pre_indices = np.concatenate(all_pre_indices)
        post_indices = np.concatenate(all_post_indices)
        pre_compartments = np.concatenate(all_pre_compartments)
        post_compartments = np.concatenate(all_post_compartments)

        # Drop self-connections (autapses) when not allowed and populations match.
        if not self.allow_self_connections and pre_num == post_num:
            keep = pre_indices != post_indices
            pre_indices = pre_indices[keep]
            post_indices = post_indices[keep]
            pre_compartments = pre_compartments[keep]
            post_compartments = post_compartments[keep]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)

        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments,
            metadata={
                'pattern': 'morphology_distance',
                'sigma': self.sigma,
                # None is the Gaussian default; report it by name as before 0.4.
                'decay_function': 'gaussian' if self.decay_function is None else self.decay_function
            }
        )


class DendriticTree(MultiCompartmentConnectivity):
    """Dendritic tree connectivity with separate basal/apical probabilities.

    Generates axon-to-dendrite connections using two independent connection
    probabilities: ``branch_targeting['proximal']`` is applied to basal
    dendrites and ``branch_targeting['distal']`` to apical dendrites. The
    ``'proximal'``/``'distal'`` keys are treated purely as the basal/apical
    probabilities respectively (no actual proximal/distal position model).

    Parameters
    ----------
    branch_targeting : dict
        Per-target connection probabilities. ``'proximal'`` -> basal dendrite
        probability (default 0.5), ``'distal'`` -> apical dendrite probability
        (default 0.3). Both must be in [0, 1].
    tree_structure : dict, optional
        Reserved descriptive metadata about the dendritic tree. It is stored on
        the result metadata but does not affect connection generation.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself. Default ``False``.

    Notes
    -----
    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Connected pairs are sampled by drawing a binomial edge count and
       then that many distinct pairs, without a dense
       ``pre_size x post_size`` matrix, so time and memory grow with the
       number of edges. The edges for a given ``seed`` differ from 0.6
       (same distribution).

    Examples
    --------
    .. code-block:: python

        >>> from braintools.conn import DendriticTree
        >>> dend_tree = DendriticTree(
        ...     branch_targeting={'proximal': 0.8, 'distal': 0.2},
        ...     seed=0,
        ... )
        >>> result = dend_tree(pre_size=50, post_size=50)
        >>> result.n_connections
        2450
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        branch_targeting: Dict,
        tree_structure: Optional[Dict] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tree_structure = tree_structure
        self.branch_targeting = branch_targeting
        _validate_prob(branch_targeting.get('proximal', 0.5), "branch_targeting['proximal']")
        _validate_prob(branch_targeting.get('distal', 0.3), "branch_targeting['distal']")
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
        """Generate dendritic tree connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Determine connection probabilities based on branch targeting
        basal_prob = self.branch_targeting.get('proximal', 0.5)
        apical_prob = self.branch_targeting.get('distal', 0.3)

        all_pre_indices = []
        all_post_indices = []
        all_pre_compartments = []
        all_post_compartments = []

        # Basal, then apical dendrite connections (binomial edge count, then
        # distinct pairs; autapses are never drawn when excluded).
        exclude_self = not self.allow_self_connections and pre_num == post_num
        for prob, target in ((basal_prob, BASAL_DENDRITE), (apical_prob, APICAL_DENDRITE)):
            pre_idx, post_idx = _uniform_pairs(self.rng, pre_num, post_num, prob, exclude_self)
            if len(pre_idx) > 0:
                all_pre_indices.append(pre_idx)
                all_post_indices.append(post_idx)
                all_pre_compartments.append(np.full(len(pre_idx), AXON, dtype=np.int64))
                all_post_compartments.append(np.full(len(pre_idx), target, dtype=np.int64))

        if len(all_pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        pre_indices = np.concatenate(all_pre_indices)
        post_indices = np.concatenate(all_post_indices)
        pre_compartments = np.concatenate(all_pre_compartments)
        post_compartments = np.concatenate(all_post_compartments)

        # Drop self-connections (autapses) when not allowed and populations match.
        if not self.allow_self_connections and pre_num == post_num:
            keep = pre_indices != post_indices
            pre_indices = pre_indices[keep]
            post_indices = post_indices[keep]
            pre_compartments = pre_compartments[keep]
            post_compartments = post_compartments[keep]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)

        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments,
            metadata={
                'pattern': 'dendritic_tree',
                'tree_structure': self.tree_structure,
                'branch_targeting': self.branch_targeting
            }
        )


class AxonalProjection(MultiCompartmentConnectivity):
    """Axonal projection onto dendritic compartments.

    Every presynaptic axon contacts every postsynaptic neuron independently with
    a pair-specific probability, and each resulting synapse is placed on one
    dendritic compartment drawn uniformly from ``target_compartments``.

    The pair probability is built from up to two spatial terms (positions are
    needed for either):

    - ``topographic_map(pre_xy, post_xy)``, if given, replaces the uniform
      ``connection_prob``;
    - ``arborization_sigma``, if given, multiplies the probability by a Gaussian
      fall-off ``exp(-d**2 / (2 * arborization_sigma**2))`` of the distance ``d``
      between the two neurons, confining the axon's arbor to a local field.

    Without positions every pair uses ``connection_prob``.

    Parameters
    ----------
    topographic_map : callable, optional
        Function ``(pre_xy, post_xy) -> probability``. It receives one row of each
        position array as plain numbers in the unit of ``pre_positions`` and must
        return a value in ``[0, 1]``.
    connection_prob : float
        Probability for every pair when no ``topographic_map`` is given.
    arborization_sigma : float or Quantity, optional
        Width of the Gaussian arborization field. A plain number is interpreted in
        the unit of ``pre_positions``. ``None`` (default) applies no fall-off.
    target_compartments : list of int, optional
        Postsynaptic dendritic compartments to target. Each connection is randomly
        assigned one of these. Default ``[BASAL_DENDRITE, APICAL_DENDRITE]``.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself. Default ``False``.

    Notes
    -----
    With a spatial term (``topographic_map`` or ``arborization_sigma``) the
    pair probabilities are evaluated in chunks of presynaptic rows of
    bounded size, and the random draws are the same as in version 0.6.
    ``topographic_map`` stays a scalar function of one pre and one post
    position; within each chunk it is called once per pair of *distinct*
    positions and the values are shared by neurons at equal positions, so
    repeated positions cost no extra calls. It is still a Python call per
    distinct pair, so its cost grows with ``pre_size * post_size`` when all
    positions differ.

    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Without ``topographic_map`` and ``arborization_sigma`` the pairs are
       sampled with a binomial edge count and distinct pairs, so the
       edges for a given ``seed`` differ from 0.6 (same distribution).
       ``topographic_map`` is called once per pair of distinct positions
       instead of once per neuron pair.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import AxonalProjection
        >>> def retinotopic_map(source_pos, target_pos):
        ...     return 0.5 * np.exp(-np.sum((source_pos - target_pos) ** 2) / (2 * 50.0 ** 2))
        >>>
        >>> positions = np.random.RandomState(0).uniform(0, 300, (40, 2)) * u.um
        >>> topographic = AxonalProjection(topographic_map=retinotopic_map, seed=0)
        >>> result = topographic(pre_size=40, post_size=40,
        ...                      pre_positions=positions, post_positions=positions)
        >>> result.n_connections
        88
        >>> local_arbor = AxonalProjection(connection_prob=0.3, arborization_sigma=80 * u.um, seed=0)
        >>> result = local_arbor(pre_size=40, post_size=40,
        ...                      pre_positions=positions, post_positions=positions)
        >>> result.n_connections
        114
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        topographic_map: Optional[Callable] = None,
        connection_prob: float = 0.05,
        arborization_sigma: Optional[Union[float, u.Quantity]] = None,
        target_compartments: Optional[List[int]] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        _validate_prob(connection_prob)
        if topographic_map is not None and not callable(topographic_map):
            raise TypeError(f"topographic_map must be callable, got {type(topographic_map)}")
        self.topographic_map = topographic_map
        self.connection_prob = connection_prob
        self.arborization_sigma = arborization_sigma
        self.target_compartments = target_compartments or [BASAL_DENDRITE, APICAL_DENDRITE]
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
        """Generate axonal projection connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        has_positions = pre_positions is not None and post_positions is not None
        if not has_positions and (self.topographic_map is not None or self.arborization_sigma is not None):
            raise ValueError(
                "AxonalProjection with a topographic_map or arborization_sigma requires "
                "both pre_positions and post_positions."
            )

        exclude_self = not self.allow_self_connections and pre_num == post_num
        spatial = self.topographic_map is not None or self.arborization_sigma is not None

        if spatial:
            # Pair-specific probabilities, evaluated one pre chunk at a time.
            sigma_val, pre_pos_vals, post_pos_vals = _length_and_positions(
                1.0 if self.arborization_sigma is None else self.arborization_sigma,
                pre_positions, post_positions,
            )
            _check_position_counts('AxonalProjection', pre_pos_vals, post_pos_vals, pre_num, post_num)
            if self.topographic_map is not None:
                post_unique, post_inverse = _unique_rows(post_pos_vals)

            def chunk_probs(start, stop):
                probs = np.full((stop - start, post_num), self.connection_prob, dtype=float)
                if self.topographic_map is not None:
                    probs[:] = _evaluate_pair_function(
                        self.topographic_map, pre_pos_vals[start:stop], post_unique, post_inverse
                    )
                if self.arborization_sigma is not None:
                    distances = cdist(pre_pos_vals[start:stop], post_pos_vals)
                    probs = probs * np.exp(-distances ** 2 / (2 * sigma_val ** 2))
                return probs

            pre_indices, post_indices = _chunked_mask_pairs(self.rng, pre_num, post_num, chunk_probs)

            # Drop self-connections (autapses) when not allowed and populations match.
            if exclude_self:
                keep = pre_indices != post_indices
                pre_indices = pre_indices[keep]
                post_indices = post_indices[keep]
        else:
            # Uniform probability: binomial edge count, then distinct pairs.
            pre_indices, post_indices = _uniform_pairs(
                self.rng, pre_num, post_num, self.connection_prob, exclude_self
            )

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)
        pre_compartments = np.full(n_connections, AXON, dtype=np.int64)
        # Randomly assign each connection to one of the target dendritic compartments.
        post_compartments = self.rng.choice(
            self.target_compartments,
            size=n_connections
        ).astype(np.int64)

        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments,
            metadata={
                'pattern': 'axonal_projection',
                'topographic': self.topographic_map is not None,
                'connection_prob': self.connection_prob,
                'arborization_sigma': self.arborization_sigma,
            }
        )


# Additional specialized patterns
class ProximalTargeting(CompartmentSpecific):
    """Connectivity targeting proximal dendritic compartments.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        # Proximal dendrites are closer to soma
        compartment_mapping = {AXON: BASAL_DENDRITE}  # Basal dendrites are typically more proximal
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class DistalTargeting(CompartmentSpecific):
    """Connectivity targeting distal dendritic compartments.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        # Distal dendrites are farther from soma
        compartment_mapping = {AXON: APICAL_DENDRITE}  # Apical dendrites extend more distally
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class BranchSpecific(MultiCompartmentConnectivity):
    """Branch-specific dendritic targeting.

    Targets specific branches within the dendritic tree based on branch indices
    or branch-specific rules.

    Parameters
    ----------
    branch_indices : list of int
        Specific branch indices to target. These map to compartment types.
        Default targets basal and apical dendrites.
    connection_prob : float
        Connection probability per branch.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.

    Notes
    -----
    Branch indices are mapped to dendritic compartment types. By default:
    - Index 0, 1: BASAL_DENDRITE
    - Index 2+: APICAL_DENDRITE
    This can be extended with full morphological information.

    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Connected pairs are sampled by drawing a binomial edge count and
       then that many distinct pairs, without a dense
       ``pre_size x post_size`` matrix, so time and memory grow with the
       number of edges. The edges for a given ``seed`` differ from 0.6
       (same distribution).
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        branch_indices: List[int] = None,
        connection_prob: float = 0.3,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        _validate_prob(connection_prob)
        self.branch_indices = branch_indices or [0, 1, 2]
        self.connection_prob = connection_prob
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
        """Generate branch-specific connections."""
        # Map branch indices to compartment types
        # Simple mapping: lower indices -> basal, higher -> apical
        target_compartments = []
        for idx in self.branch_indices:
            if idx < 2:  # Lower branch indices map to basal dendrites
                if BASAL_DENDRITE not in target_compartments:
                    target_compartments.append(BASAL_DENDRITE)
            else:  # Higher branch indices map to apical dendrites
                if APICAL_DENDRITE not in target_compartments:
                    target_compartments.append(APICAL_DENDRITE)

        if not target_compartments:
            target_compartments = [BASAL_DENDRITE, APICAL_DENDRITE]

        return CompartmentSpecific(
            compartment_mapping={AXON: target_compartments},
            connection_prob=self.connection_prob,
            weight=self.weight_init,
            delay=self.delay_init,
            allow_self_connections=self.allow_self_connections,
            seed=self.seed
        ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)


class BasalDendriteTargeting(CompartmentSpecific):
    """Specific targeting of basal dendrites.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartment_mapping = {AXON: BASAL_DENDRITE}
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class ApicalDendriteTargeting(CompartmentSpecific):
    """Specific targeting of apical dendrites.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific` (binomial edge
       count, then distinct pairs), so the edges for a given ``seed`` differ
       from 0.6.
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartment_mapping = {AXON: APICAL_DENDRITE}
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class DendriticIntegration(MultiCompartmentConnectivity):
    """Dendritic integration connectivity patterns.

    Models connectivity that supports dendritic integration, with clustered
    synapses on the same dendritic branches for nonlinear integration.

    Parameters
    ----------
    cluster_size : int
        Number of synapses per cluster.
    n_clusters : int
        Number of clusters per postsynaptic neuron.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself. Default ``False``.

    Notes
    -----
    Each of the ``n_clusters`` clusters of a postsynaptic neuron draws
    ``min(cluster_size, pre_size)`` distinct presynaptic neurons and lands on
    the basal or the apical dendrite with equal probability. Synapses repeated
    across clusters of the same neuron are merged.

    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       All clusters are sampled in one vectorised call instead of a Python
       loop over postsynaptic neurons and clusters. The edges for a given
       ``seed`` differ from 0.6 (same distribution).

    Examples
    --------
    .. code-block:: python

        >>> from braintools.conn import DendriticIntegration
        >>> clusters = DendriticIntegration(cluster_size=4, n_clusters=3, seed=0)
        >>> result = clusters(pre_size=100, post_size=20)
        >>> result.n_connections
        236
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        cluster_size: int = 5,
        n_clusters: int = 10,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        _validate_non_negative_int(cluster_size, 'cluster_size')
        _validate_non_negative_int(n_clusters, 'n_clusters')
        self.cluster_size = cluster_size
        self.n_clusters = n_clusters
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
        """Generate clustered synaptic connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # One cluster per (post neuron, cluster) row, all rows at once: each
        # cluster draws ``k`` distinct presynaptic neurons and one dendrite
        # (basal or apical, equally likely).
        k = min(self.cluster_size, pre_num)
        n_rows = post_num * self.n_clusters
        if n_rows == 0 or k == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )
        pre_selected = sample_rows_without_replacement(self.rng, n_rows, pre_num, k)
        dendrite_type = np.where(randint(self.rng, 2, n_rows) == 0, BASAL_DENDRITE, APICAL_DENDRITE)

        pre_indices = pre_selected.reshape(-1)
        post_indices = np.repeat(np.arange(post_num, dtype=np.int64), self.n_clusters * k)
        pre_compartments = np.full(pre_indices.size, AXON, dtype=np.int64)
        post_compartments = np.repeat(dendrite_type.astype(np.int64), k)

        # Deduplicate (pre, post, pre_comp, post_comp) quads. The same neuron can be
        # sampled into multiple clusters of the same post neuron (replace=False only
        # holds within a cluster), which would otherwise produce duplicate synapses
        # that inflate weights after conversion.
        quads = np.stack([pre_indices, post_indices, pre_compartments, post_compartments])
        _, unique_cols = np.unique(quads, axis=1, return_index=True)
        unique_cols = np.sort(unique_cols)
        pre_indices = pre_indices[unique_cols]
        post_indices = post_indices[unique_cols]
        pre_compartments = pre_compartments[unique_cols]
        post_compartments = post_compartments[unique_cols]

        # Drop self-connections (autapses) when not allowed and populations match.
        if not self.allow_self_connections and pre_num == post_num:
            keep = pre_indices != post_indices
            pre_indices = pre_indices[keep]
            post_indices = post_indices[keep]
            pre_compartments = pre_compartments[keep]
            post_compartments = post_compartments[keep]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)

        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments,
            metadata={
                'pattern': 'dendritic_integration',
                'cluster_size': self.cluster_size,
                'n_clusters': self.n_clusters
            }
        )


class AxonalBranching(MultiCompartmentConnectivity):
    """Axonal branching patterns.

    Models realistic axonal branching where each presynaptic axon makes
    multiple synaptic contacts following branching structure.

    Parameters
    ----------
    branches_per_axon : int
        Average number of branches per axon (Poisson-distributed per axon).
    branch_spread : float or Quantity, optional
        Spatial spread of the branches. When given, positions are required and
        each axon picks its targets with probability proportional to
        ``exp(-d**2 / (2 * branch_spread**2))`` of the distance ``d``, so branches
        stay near the source neuron. A plain number is interpreted in the unit of
        ``pre_positions``. ``None`` (default) picks targets uniformly.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself. Default ``False``.

    Notes
    -----
    The number of branches of each axon is Poisson distributed and capped
    at the number of eligible targets; the targets of one axon are distinct.
    With ``branch_spread`` the targets are drawn by successive sampling
    without replacement, implemented with exponential keys [1]_ evaluated in
    chunks of presynaptic rows of bounded size.

    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       The branch counts and targets are drawn in bulk instead of in a
       Python loop over axons, and no ``pre_size x post_size`` distance
       matrix is built. The edges for a given ``seed`` differ from 0.6
       (same distribution).

    References
    ----------
    .. [1] Efraimidis, P. S., & Spirakis, P. G. (2006). Weighted random
           sampling with a reservoir. Information Processing Letters,
           97(5), 181-185.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import AxonalBranching
        >>> positions = np.random.RandomState(0).uniform(0, 500, (60, 2)) * u.um
        >>> branching = AxonalBranching(branches_per_axon=4, branch_spread=50 * u.um, seed=0)
        >>> result = branching(pre_size=60, post_size=60,
        ...                    pre_positions=positions, post_positions=positions)
        >>> result.n_connections
        237
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        branches_per_axon: int = 5,
        branch_spread: Optional[Union[float, u.Quantity]] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        _validate_non_negative_int(branches_per_axon, 'branches_per_axon')
        self.branches_per_axon = branches_per_axon
        self.branch_spread = branch_spread
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
        """Generate axonal branching connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        if self.branch_spread is not None and (pre_positions is None or post_positions is None):
            raise ValueError(
                "AxonalBranching with branch_spread requires both pre_positions "
                "and post_positions."
            )

        # A neuron is not a candidate target of its own axon when autapses are
        # excluded; choosing it and dropping it afterwards would shrink the branch
        # count below ``branches_per_axon`` (badly so with ``branch_spread``,
        # where the source itself, at distance 0, is the most likely target).
        exclude_self = not self.allow_self_connections and pre_num == post_num

        # Poisson branch count of every axon, drawn at once.
        n_branches = self.rng.poisson(self.branches_per_axon, size=pre_num).astype(np.int64)

        if self.branch_spread is None:
            pre_indices, post_indices = _uniform_branches(self.rng, n_branches, post_num, exclude_self)
        else:
            spread_val, pre_pos_val, post_pos_val = _length_and_positions(
                self.branch_spread, pre_positions, post_positions
            )
            _check_position_counts('AxonalBranching', pre_pos_val, post_pos_val, pre_num, post_num)
            pre_indices, post_indices = _weighted_branches(
                self.rng, n_branches, post_num, pre_pos_val, post_pos_val, spread_val, exclude_self
            )

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        # Drop self-connections (autapses) when not allowed and populations match.
        if not self.allow_self_connections and pre_num == post_num:
            keep = pre_indices != post_indices
            pre_indices = pre_indices[keep]
            post_indices = post_indices[keep]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)

        pre_compartments = np.full(n_connections, AXON, dtype=np.int64)
        post_compartments = self.rng.choice(
            [BASAL_DENDRITE, APICAL_DENDRITE],
            size=n_connections
        ).astype(np.int64)

        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments,
            metadata={
                'pattern': 'axonal_branching',
                'branches_per_axon': self.branches_per_axon,
                'branch_spread': self.branch_spread,
            }
        )


class AxonalArborization(MultiCompartmentConnectivity):
    """Axonal arborization patterns.

    Models spatial axonal arborization with local clustering of synaptic
    contacts around branch points.

    Parameters
    ----------
    arborization_radius : float or Quantity
        Radius of axonal arborization field. A plain number is interpreted in
        the unit of ``pre_positions`` (so the default ``150.0`` means 150 um
        for micrometre positions).
    density : float
        Synaptic density within arborization field. Must be in [0, 1].
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself. Default ``False``.

    Notes
    -----
    A pair within ``arborization_radius`` connects with probability
    ``density``. Pair probabilities are evaluated in chunks of presynaptic rows of
    bounded size, so peak memory does not grow with
    ``pre_size * post_size``. The random draws are the same as in version
    0.6, so a given ``seed`` gives the same edges.
    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import AxonalArborization
        >>> positions = np.random.RandomState(0).uniform(0, 500, (60, 2)) * u.um
        >>> arbor = AxonalArborization(arborization_radius=100 * u.um, density=0.3, seed=0)
        >>> result = arbor(pre_size=60, post_size=60,
        ...                pre_positions=positions, post_positions=positions)
        >>> result.n_connections
        113
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        arborization_radius: Union[float, u.Quantity] = 150.0,
        density: float = 0.3,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        _validate_prob(density, 'density')
        self.arborization_radius = arborization_radius
        self.density = density
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
        """Generate axonal arborization connections."""
        if pre_positions is None or post_positions is None:
            raise ValueError(
                "AxonalArborization requires both pre_positions and post_positions "
                "(spatial arborization cannot be computed without them)."
            )

        pre_num = int(np.prod(pre_size)) if isinstance(pre_size, tuple) else pre_size
        post_num = int(np.prod(post_size)) if isinstance(post_size, tuple) else post_size

        # Radius and positions as plain numbers in a common unit
        radius_val, pre_pos_val, post_pos_val = _length_and_positions(
            self.arborization_radius, pre_positions, post_positions
        )
        _check_position_counts('AxonalArborization', pre_pos_val, post_pos_val, pre_num, post_num)

        def chunk_probs(start, stop):
            # Density inside the arborization radius, one pre chunk at a time.
            within_radius = cdist(pre_pos_val[start:stop], post_pos_val) <= radius_val
            return self.density, within_radius

        pre_indices, post_indices = _chunked_mask_pairs(self.rng, pre_num, post_num, chunk_probs)

        # Drop self-connections (autapses) when not allowed and populations match.
        if not self.allow_self_connections and pre_num == post_num:
            keep = pre_indices != post_indices
            pre_indices = pre_indices[keep]
            post_indices = post_indices[keep]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)
        pre_compartments = np.full(n_connections, AXON, dtype=np.int64)
        post_compartments = self.rng.choice(
            [BASAL_DENDRITE, APICAL_DENDRITE],
            size=n_connections
        ).astype(np.int64)

        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments,
            metadata={
                'pattern': 'axonal_arborization',
                'arborization_radius': self.arborization_radius,
                'density': self.density
            }
        )


class TopographicProjection(MultiCompartmentConnectivity):
    """Topographic projection patterns.

    Shorthand for ``AxonalProjection(topographic_map=...)``; positions are required.

    Parameters
    ----------
    topographic_map : callable
        Function (pre_pos, post_pos) -> probability.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.

    Notes
    -----
    ``topographic_map`` is a scalar function of one pre and one post
    position. It is called once per pair of distinct positions (within each
    chunk of presynaptic rows), so neurons at equal positions share one
    call; see :class:`AxonalProjection`.

    With ``topographic_map=None`` there is no spatial term and every pair
    connects with the default ``AxonalProjection`` probability (0.05).

    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       With ``topographic_map=None`` the pairs are sampled by the uniform
       path of :class:`AxonalProjection` (binomial edge count, then distinct
       pairs), so the edges for a given ``seed`` differ from 0.6. With a
       map, the random draws are the same as in 0.6.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.conn import TopographicProjection
        >>> grid = np.stack(np.meshgrid(np.arange(6), np.arange(6)), -1).reshape(-1, 2)
        >>> positions = np.repeat(grid, 2, axis=0) * 40.0 * u.um  # two neurons per site
        >>> topo = TopographicProjection(
        ...     topographic_map=lambda a, b: 0.8 * float(np.hypot(*(a - b)) < 50.0), seed=0
        ... )
        >>> result = topo(pre_size=72, post_size=72,
        ...               pre_positions=positions, post_positions=positions)
        >>> result.n_connections
        443
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        topographic_map: Callable,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.topographic_map = topographic_map
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
        """Generate topographic projection connections."""
        return AxonalProjection(
            topographic_map=self.topographic_map,
            weight=self.weight_init,
            delay=self.delay_init,
            seed=self.seed
        ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)


class SynapticPlacement(MultiCompartmentConnectivity):
    """Synapse placement by compartment preference.

    Separates *whether* two neurons connect from *where* the synapse lands. Every
    presynaptic axon contacts every postsynaptic neuron independently with
    probability ``connection_prob`` (one synapse per connected pair), and each
    synapse is then placed on a postsynaptic compartment drawn with probability
    proportional to ``compartment_preferences``.

    Preferences express placement rules directly, for example
    ``{BASAL_DENDRITE: 1.0}`` for proximal-only input,
    ``{APICAL_DENDRITE: 1.0}`` for distal-only input, or
    ``{SOMA: 0.2, BASAL_DENDRITE: 0.5, APICAL_DENDRITE: 0.3}`` for a mixture.

    Parameters
    ----------
    connection_prob : float
        Probability that a presynaptic neuron makes a synapse onto a given
        postsynaptic neuron. Default 0.2.
    compartment_preferences : dict, optional
        Mapping from target compartment (``SOMA``, ``BASAL_DENDRITE``,
        ``APICAL_DENDRITE``, ``AXON``) to a non-negative relative preference.
        Preferences are normalized to sum to one. Default
        ``{BASAL_DENDRITE: 0.6, APICAL_DENDRITE: 0.4}``.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    allow_self_connections : bool, optional
        Whether a neuron may connect to itself. Default ``False``.

    Notes
    -----
    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Connected pairs are sampled by drawing a binomial edge count and
       then that many distinct pairs, without a dense
       ``pre_size x post_size`` matrix, so time and memory grow with the
       number of edges. The edges for a given ``seed`` differ from 0.6
       (same distribution).

    Examples
    --------
    .. code-block:: python

        >>> from braintools.conn import SynapticPlacement, APICAL_DENDRITE, BASAL_DENDRITE
        >>> distal_input = SynapticPlacement(
        ...     connection_prob=0.1,
        ...     compartment_preferences={APICAL_DENDRITE: 0.8, BASAL_DENDRITE: 0.2},
        ...     seed=0,
        ... )
        >>> result = distal_input(pre_size=100, post_size=100)
        >>> result.n_connections
        993
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        connection_prob: float = 0.2,
        compartment_preferences: Optional[Dict[int, float]] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        allow_self_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        _validate_prob(connection_prob)
        self.connection_prob = connection_prob
        self.compartment_preferences = compartment_preferences or {
            BASAL_DENDRITE: 0.6,
            APICAL_DENDRITE: 0.4
        }
        for comp, pref in self.compartment_preferences.items():
            CompartmentSpecific._validate_compartment_index(comp, "compartment_preferences")
            if not isinstance(pref, (int, float, np.number)):
                raise TypeError(f"compartment preference must be numeric, got {type(pref)} for {comp}")
            if pref < 0:
                raise ValueError(f"compartment preference must be non-negative, got {pref} for {comp}")
        if sum(self.compartment_preferences.values()) <= 0:
            raise ValueError("compartment_preferences must contain at least one positive preference")
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
        """Generate synapse placements."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Binomial edge count, then distinct pairs (autapses never drawn when excluded).
        exclude_self = not self.allow_self_connections and pre_num == post_num
        pre_indices, post_indices = _uniform_pairs(
            self.rng, pre_num, post_num, self.connection_prob, exclude_self
        )

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                model_type='multi_compartment',
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
            )

        n_connections = len(pre_indices)
        compartments = np.array(list(self.compartment_preferences.keys()), dtype=np.int64)
        prefs = np.array(list(self.compartment_preferences.values()), dtype=float)
        pre_compartments = np.full(n_connections, AXON, dtype=np.int64)
        post_compartments = self.rng.choice(compartments, size=n_connections, p=prefs / prefs.sum())

        weights = param(self.weight_init, n_connections, rng=self.rng)
        delays = param(self.delay_init, n_connections, rng=self.rng)

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            delays=delays,
            model_type='multi_compartment',
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            pre_compartments=pre_compartments,
            post_compartments=post_compartments.astype(np.int64),
            metadata={
                'pattern': 'synaptic_placement',
                'connection_prob': self.connection_prob,
                'compartment_preferences': dict(self.compartment_preferences),
            }
        )


class SynapticClustering(MultiCompartmentConnectivity):
    """Synaptic clustering patterns.

    Creates clusters of synapses on dendritic branches, important for
    nonlinear dendritic integration.

    Parameters
    ----------
    cluster_size : int
        Number of synapses per cluster.
    n_clusters_per_neuron : int
        Number of clusters per postsynaptic neuron.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.

    Notes
    -----
    Shorthand for :class:`DendriticIntegration`.

    Generation runs on the host with NumPy and cannot be traced by
    ``jax.jit``, ``jax.vmap`` or ``jax.grad``.

    .. versionchanged:: 0.7.0
       Vectorised with :class:`DendriticIntegration`; the edges for a
       given ``seed`` differ from 0.6 (same distribution).

    Examples
    --------
    .. code-block:: python

        >>> from braintools.conn import SynapticClustering
        >>> clustering = SynapticClustering(cluster_size=3, n_clusters_per_neuron=2, seed=0)
        >>> result = clustering(pre_size=50, post_size=10)
        >>> result.n_connections
        58
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        cluster_size: int = 5,
        n_clusters_per_neuron: int = 10,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        _validate_non_negative_int(cluster_size, 'cluster_size')
        _validate_non_negative_int(n_clusters_per_neuron, 'n_clusters_per_neuron')
        self.cluster_size = cluster_size
        self.n_clusters_per_neuron = n_clusters_per_neuron
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
        """Generate clustered synaptic connections."""
        return DendriticIntegration(
            cluster_size=self.cluster_size,
            n_clusters=self.n_clusters_per_neuron,
            weight=self.weight_init,
            delay=self.delay_init,
            seed=self.seed
        ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)


class AllToAllCompartments(CompartmentSpecific):
    """All-to-all compartment connectivity.

    .. versionchanged:: 0.7.0
       Pairs are sampled as in :class:`CompartmentSpecific`. With probability 1
       every pair is still connected, but random weight or delay initializers
       draw different values for a given ``seed`` than in 0.6.
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartments = [SOMA, BASAL_DENDRITE, APICAL_DENDRITE, AXON]
        compartment_mapping = {}
        for source in compartments:
            compartment_mapping[source] = compartments

        super().__init__(
            compartment_mapping=compartment_mapping,
            connection_prob=1.0,
            **kwargs
        )


class CustomCompartment(MultiCompartmentConnectivity):
    """Custom compartment connectivity using user-defined function."""
    __module__ = 'braintools.conn'

    def __init__(self, connection_func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.connection_func = connection_func

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate custom compartment connections."""
        # Validate required kwargs
        if 'pre_size' not in kwargs:
            raise ValueError("CustomCompartment.generate() requires 'pre_size' parameter")
        if 'post_size' not in kwargs:
            raise ValueError("CustomCompartment.generate() requires 'post_size' parameter")

        result_data = self.connection_func(**kwargs)

        if not isinstance(result_data, (tuple, list)):
            raise ValueError(
                f"Custom function must return a tuple/list of 4-6 values, "
                f"got {type(result_data)}"
            )
        if not (4 <= len(result_data) <= 6):
            raise ValueError(f"Custom function must return 4-6 values, got {len(result_data)}")

        if len(result_data) == 4:
            pre_indices, post_indices, pre_compartments, post_compartments = result_data
            weights = None
            delays = None
        elif len(result_data) == 5:
            pre_indices, post_indices, pre_compartments, post_compartments, weights = result_data
            delays = None
        else:  # len(result_data) == 6
            pre_indices, post_indices, pre_compartments, post_compartments, weights, delays = result_data

        if delays is not None and not isinstance(delays, u.Quantity):
            delays = np.asarray(delays) * u.ms

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=u.math.array(weights) if weights is not None else None,
            delays=delays,
            model_type='multi_compartment',
            pre_size=kwargs['pre_size'],
            post_size=kwargs['post_size'],
            pre_positions=kwargs.get('pre_positions', None),
            post_positions=kwargs.get('post_positions', None),
            pre_compartments=np.array(pre_compartments, dtype=np.int64),
            post_compartments=np.array(post_compartments, dtype=np.int64),
            metadata={'pattern': 'custom_compartment'}
        )
