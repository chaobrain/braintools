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
"""

import warnings
from typing import Optional, Tuple, Union, Dict, List, Callable

import brainunit as u
import numpy as np
from scipy.spatial.distance import cdist

from braintools.init._init_base import param, Initializer
from ._base import MultiCompartmentConnectivity, ConnectionResult

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


def _normalize_positions(positions, unit=None):
    """Normalize position arrays to numpy arrays with consistent units.

    Parameters
    ----------
    positions : array-like or Quantity
        Position data to normalize.
    unit : Unit, optional
        Target unit for conversion. If None, maintains original unit.

    Returns
    -------
    tuple
        (normalized_array, unit) where normalized_array is a numpy array
        and unit is the associated brainunit Unit or None.
    """
    if positions is None:
        return None, None
    if unit is not None:
        positions = u.Quantity(positions).to(unit).mantissa
    return positions, unit


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
    weight_distribution : str or callable
        Weight distribution for connections.
    weight_params : dict
        Parameters for weight distribution.
    morphology_info : dict, optional
        Information about neuron morphology structure.

    Examples
    --------
    Axon-to-soma connections:

    .. code-block:: python

        >>> import brainunit as u
        >>> axon_soma = CompartmentSpecific(
        ...     compartment_mapping={AXON: SOMA},
        ...     connection_prob=0.1,
        ...     weight_distribution='normal',
        ...     weight_params={'mean': 2.0 * u.nS, 'std': 0.5 * u.nS}
        ... )
        >>> result = axon_soma(pre_size=100, post_size=100)

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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.compartment_mapping = self._normalize_compartment_mapping(compartment_mapping)
        self._validate_connection_prob(connection_prob)
        self.connection_prob = connection_prob
        self.weight_init = weight
        self.delay_init = delay
        self.morphology_info = morphology_info or {}

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
                    prob = self.connection_prob.get((source_comp, target_comp), 0.0)
                else:
                    prob = self.connection_prob

                if prob > 0:
                    active_pairs.append((source_comp, target_comp, prob))

        # Generate connections only for active pairs
        for source_comp, target_comp, prob in active_pairs:
            # Vectorized connection generation
            random_matrix = self.rng.random((pre_num, post_num))
            connection_mask = random_matrix < prob
            pre_idx, post_idx = np.where(connection_mask)

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

        >>> soma_dend = SomaToDendrite(
        ...     target_dendrites=[BASAL_DENDRITE, APICAL_DENDRITE],
        ...     prob_per_dendrite=0.8
        ... )

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

    Examples
    --------
    .. code-block:: python

        >>> axon_soma = AxonToSoma(
        ...     connection_prob=0.05,
        ...     weight_distribution='lognormal',
        ...     weight_params={'mean': 1.0, 'sigma': 0.3}
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartment_mapping = {AXON: SOMA}
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class DendriteToSoma(CompartmentSpecific):
    """Specialized connectivity from dendrites to soma.

    This creates connections from dendritic compartments to soma compartments,
    modeling dendritic integration and signal propagation to the cell body.

    Parameters
    ----------
    source_dendrites : list, optional
        List of dendrite compartment types to use as sources.

    Examples
    --------
    .. code-block:: python

        >>> dend_soma = DendriteToSoma(
        ...     source_dendrites=[BASAL_DENDRITE, APICAL_DENDRITE],
        ...     connection_prob=0.6
        ... )
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

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> axon_dend = AxonToDendrite(
        ...     target_dendrites=[BASAL_DENDRITE, APICAL_DENDRITE],
        ...     connection_prob=0.1,
        ...     weight_distribution='lognormal',
        ...     weight_params={'mean': 2.0 * u.nS, 'sigma': 0.5}
        ... )
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

    Examples
    --------
    .. code-block:: python

        >>> dend_dend = DendriteToDendrite(
        ...     connection_prob=0.2,
        ...     weight_distribution='normal',
        ...     weight_params={'mean': 0.5, 'std': 0.1}
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartment_mapping = {
            BASAL_DENDRITE: [BASAL_DENDRITE, APICAL_DENDRITE],
            APICAL_DENDRITE: [BASAL_DENDRITE, APICAL_DENDRITE]
        }
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class MorphologyDistance(MultiCompartmentConnectivity):
    """Distance-dependent connectivity based on detailed morphology.

    This pattern uses the actual morphological structure of neurons
    to determine connection probabilities and strengths based on
    distances between specific compartments.

    Parameters
    ----------
    sigma : float or Quantity
        Characteristic distance scale for connectivity.
    decay_function : str
        Distance decay function ('gaussian', 'exponential', 'linear').
    compartment_mapping : dict
        Mapping of which compartments can connect to which.
    morphology_positions : dict, optional
        Detailed positions of compartments for each neuron.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> morph_dist = MorphologyDistance(
        ...     sigma=50 * u.um,
        ...     decay_function='gaussian',
        ...     compartment_mapping={AXON: [BASAL_DENDRITE, APICAL_DENDRITE]}
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        sigma: Union[float, u.Quantity],
        decay_function: str = 'gaussian',
        compartment_mapping: Dict = None,
        morphology_positions: Optional[Dict] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.decay_function = decay_function
        self.compartment_mapping = compartment_mapping or {AXON: [BASAL_DENDRITE, APICAL_DENDRITE]}
        self.morphology_positions = morphology_positions
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
        """Generate morphology-based distance-dependent connections."""
        if pre_positions is None or post_positions is None:
            return CompartmentSpecific(
                compartment_mapping=self.compartment_mapping,
                connection_prob=0.1,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed
            ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)

        # Validate and normalize sizes FIRST
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Normalize positions using helper
        if isinstance(self.sigma, u.Quantity):
            sigma_val, sigma_unit = u.split_mantissa_unit(self.sigma)
            pre_pos_val, _ = _normalize_positions(pre_positions, sigma_unit)
            post_pos_val, _ = _normalize_positions(post_positions, sigma_unit)
        else:
            sigma_val = self.sigma
            pre_pos_val, _ = _normalize_positions(pre_positions)
            post_pos_val, _ = _normalize_positions(post_positions)

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
        pre_pos_val = pre_pos_val[:pre_num]
        post_pos_val = post_pos_val[:post_num]

        # Calculate distances
        distances = cdist(pre_pos_val, post_pos_val)

        # Apply decay function
        if self.decay_function == 'gaussian':
            probs = np.exp(-distances ** 2 / (2 * sigma_val ** 2))
        elif self.decay_function == 'exponential':
            probs = np.exp(-distances / sigma_val)
        elif self.decay_function == 'linear':
            probs = np.maximum(0, 1 - distances / sigma_val)
        else:
            raise ValueError(f"Unknown decay function: {self.decay_function}")

        all_pre_indices = []
        all_post_indices = []
        all_pre_compartments = []
        all_post_compartments = []

        # Generate connections for each compartment mapping
        for source_comp, target_comps in self.compartment_mapping.items():
            if not isinstance(target_comps, (list, tuple)):
                target_comps = [target_comps]

            for target_comp in target_comps:
                random_matrix = self.rng.random((pre_num, post_num))
                connection_mask = random_matrix < probs
                pre_idx, post_idx = np.where(connection_mask)

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
                'decay_function': self.decay_function
            }
        )


class DendriticTree(MultiCompartmentConnectivity):
    """Dendritic tree connectivity patterns with branch-specific targeting.

    This models realistic dendritic connectivity considering the branching
    structure of dendritic trees and branch-specific connection rules.

    Parameters
    ----------
    tree_structure : dict
        Description of dendritic tree structure.
    branch_targeting : dict
        Rules for targeting specific branches (e.g., {'proximal': 0.8, 'distal': 0.2}).
    distance_dependence : bool
        Whether to include distance dependence within the tree.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> tree_structure = {
        ...     'basal': {'n_branches': 5, 'branch_length': 200 * u.um},
        ...     'apical': {'n_branches': 1, 'branch_length': 600 * u.um}
        ... }
        >>> dend_tree = DendriticTree(
        ...     tree_structure=tree_structure,
        ...     branch_targeting={'proximal': 0.8, 'distal': 0.2}
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        tree_structure: Dict,
        branch_targeting: Dict,
        distance_dependence: bool = True,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tree_structure = tree_structure
        self.branch_targeting = branch_targeting
        self.distance_dependence = distance_dependence
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

        # Basal dendrite connections
        random_matrix = self.rng.random((pre_num, post_num))
        connection_mask = random_matrix < basal_prob
        pre_idx, post_idx = np.where(connection_mask)
        if len(pre_idx) > 0:
            all_pre_indices.append(pre_idx)
            all_post_indices.append(post_idx)
            all_pre_compartments.append(np.full(len(pre_idx), AXON, dtype=np.int64))
            all_post_compartments.append(np.full(len(pre_idx), BASAL_DENDRITE, dtype=np.int64))

        # Apical dendrite connections
        random_matrix = self.rng.random((pre_num, post_num))
        connection_mask = random_matrix < apical_prob
        pre_idx, post_idx = np.where(connection_mask)
        if len(pre_idx) > 0:
            all_pre_indices.append(pre_idx)
            all_post_indices.append(post_idx)
            all_pre_compartments.append(np.full(len(pre_idx), AXON, dtype=np.int64))
            all_post_compartments.append(np.full(len(pre_idx), APICAL_DENDRITE, dtype=np.int64))

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
    """Axonal projection patterns with topographic organization.

    This models long-range axonal projections with topographic mapping
    and realistic axonal arborization patterns.

    Parameters
    ----------
    projection_type : str
        Type of projection ('local', 'long_range', 'topographic').
    topographic_map : callable, optional
        Function defining topographic mapping (pre_pos, post_pos) -> probability.
    arborization_pattern : str
        Pattern of axonal arborization ('diffuse', 'clustered', 'columnar').
    connection_prob : float
        Base connection probability.
    spatial_scale : float
        Spatial scale for clustered arborization pattern (default 10000.0).
        Controls how distance affects clustering: smaller values = tighter clustering.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.

    Examples
    --------
    .. code-block:: python

        >>> def retinotopic_map(source_pos, target_pos):
        ...     # Custom topographic mapping
        ...     return np.exp(-np.linalg.norm(source_pos - target_pos)**2 / 1000)
        >>>
        >>> axon_proj = AxonalProjection(
        ...     projection_type='topographic',
        ...     topographic_map=retinotopic_map,
        ...     arborization_pattern='clustered'
        ... )
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        projection_type: str = 'local',
        topographic_map: Optional[Callable] = None,
        arborization_pattern: str = 'diffuse',
        connection_prob: float = 0.05,
        spatial_scale: float = 10000.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.projection_type = projection_type
        self.topographic_map = topographic_map
        self.arborization_pattern = arborization_pattern
        self.connection_prob = connection_prob
        self.spatial_scale = spatial_scale
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
        """Generate axonal projection connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        if pre_positions is not None and post_positions is not None:

            # Normalize positions
            pre_pos_vals, pos_unit = u.split_mantissa_unit(pre_positions)
            post_pos_vals = u.Quantity(post_positions).to(pos_unit).mantissa

            # Calculate connection probabilities
            if self.topographic_map is not None:
                # Use custom topographic mapping
                probs = np.zeros((pre_num, post_num))
                for i in range(pre_num):
                    for j in range(post_num):
                        probs[i, j] = self.topographic_map(pre_pos_vals[i], post_pos_vals[j])
            else:
                # Use uniform probability
                probs = np.full((pre_num, post_num), self.connection_prob)

            # Modify based on arborization pattern
            if self.arborization_pattern == 'clustered':
                distances = cdist(pre_pos_vals, post_pos_vals)
                spatial_factor = np.exp(-distances ** 2 / self.spatial_scale)
                probs = probs * spatial_factor
        else:
            # Use uniform probability
            probs = np.full((pre_num, post_num), self.connection_prob)

        # Generate connections
        random_matrix = self.rng.random((pre_num, post_num))
        connection_mask = random_matrix < probs
        pre_indices, post_indices = np.where(connection_mask)

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
        post_compartments = np.full(n_connections, BASAL_DENDRITE, dtype=np.int64)

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
                'projection_type': self.projection_type,
                'arborization_pattern': self.arborization_pattern
            }
        )


# Additional specialized patterns
class ProximalTargeting(CompartmentSpecific):
    """Connectivity targeting proximal dendritic compartments."""
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        # Proximal dendrites are closer to soma
        compartment_mapping = {AXON: BASAL_DENDRITE}  # Basal dendrites are typically more proximal
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class DistalTargeting(CompartmentSpecific):
    """Connectivity targeting distal dendritic compartments."""
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
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        branch_indices: List[int] = None,
        connection_prob: float = 0.3,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.branch_indices = branch_indices or [0, 1, 2]
        self.connection_prob = connection_prob
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
            seed=self.seed
        ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)


class BasalDendriteTargeting(CompartmentSpecific):
    """Specific targeting of basal dendrites."""
    __module__ = 'braintools.conn'

    def __init__(self, **kwargs):
        compartment_mapping = {AXON: BASAL_DENDRITE}
        super().__init__(compartment_mapping=compartment_mapping, **kwargs)


class ApicalDendriteTargeting(CompartmentSpecific):
    """Specific targeting of apical dendrites."""
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
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        cluster_size: int = 5,
        n_clusters: int = 10,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cluster_size = cluster_size
        self.n_clusters = n_clusters
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

        # Create clusters for each postsynaptic neuron
        for post_idx in range(post_num):
            for cluster_idx in range(self.n_clusters):
                # Choose random presynaptic neurons for this cluster
                pre_selected = self.rng.choice(pre_num, size=min(self.cluster_size, pre_num), replace=False)

                # Randomly assign to basal or apical dendrite
                dendrite_type = self.rng.choice([BASAL_DENDRITE, APICAL_DENDRITE])

                all_pre_indices.append(pre_selected)
                all_post_indices.append(np.full(len(pre_selected), post_idx, dtype=np.int64))
                all_pre_compartments.append(np.full(len(pre_selected), AXON, dtype=np.int64))
                all_post_compartments.append(np.full(len(pre_selected), dendrite_type, dtype=np.int64))

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
        Average number of branches per axon.
    branch_spread : float
        Spatial spread of branches (relevant if positions provided).
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        branches_per_axon: int = 5,
        branch_spread: float = 100.0,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.branches_per_axon = branches_per_axon
        self.branch_spread = branch_spread
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
        """Generate axonal branching connections."""
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

        # Each presynaptic neuron branches to multiple postsynaptic targets
        for pre_idx in range(pre_num):
            n_branches = self.rng.poisson(self.branches_per_axon)
            n_branches = min(n_branches, post_num)

            if n_branches > 0:
                post_targets = self.rng.choice(post_num, size=n_branches, replace=False)
                all_pre_indices.append(np.full(n_branches, pre_idx, dtype=np.int64))
                all_post_indices.append(post_targets)

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
                'branches_per_axon': self.branches_per_axon
            }
        )


class AxonalArborization(MultiCompartmentConnectivity):
    """Axonal arborization patterns.

    Models spatial axonal arborization with local clustering of synaptic
    contacts around branch points.

    Parameters
    ----------
    arborization_radius : float or Quantity
        Radius of axonal arborization field.
    density : float
        Synaptic density within arborization field.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        arborization_radius: Union[float, u.Quantity] = 150.0,
        density: float = 0.3,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.arborization_radius = arborization_radius
        self.density = density
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
        """Generate axonal arborization connections."""
        if pre_positions is None or post_positions is None:
            return CompartmentSpecific(
                compartment_mapping={AXON: [BASAL_DENDRITE, APICAL_DENDRITE]},
                connection_prob=self.density,
                weight=self.weight_init,
                delay=self.delay_init,
                seed=self.seed
            ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)

        # Normalize positions using helper
        if isinstance(self.arborization_radius, u.Quantity):
            radius_val, radius_unit = u.split_mantissa_unit(self.arborization_radius)
            pre_pos_val, _ = _normalize_positions(pre_positions, radius_unit)
            post_pos_val, _ = _normalize_positions(post_positions, radius_unit)
        else:
            radius_val = self.arborization_radius
            pre_pos_val, _ = _normalize_positions(pre_positions)
            post_pos_val, _ = _normalize_positions(post_positions)

        # Calculate distances
        distances = cdist(pre_pos_val, post_pos_val)

        # Within arborization radius
        within_radius = distances <= radius_val

        # Apply density probability
        random_matrix = self.rng.random(distances.shape)
        connection_mask = within_radius & (random_matrix < self.density)

        pre_indices, post_indices = np.where(connection_mask)

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

    Alias for AxonalProjection with topographic mapping enabled.

    Parameters
    ----------
    topographic_map : callable
        Function (pre_pos, post_pos) -> probability.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
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
            projection_type='topographic',
            topographic_map=self.topographic_map,
            weight=self.weight_init,
            delay=self.delay_init,
            seed=self.seed
        ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)


class SynapticPlacement(MultiCompartmentConnectivity):
    """Synaptic placement rules.

    Controls where synapses are placed on compartments based on rules
    like distance from soma, compartment type preferences, etc.

    Parameters
    ----------
    placement_rule : str
        Placement rule ('uniform', 'proximal', 'distal', 'distance_weighted').
    compartment_preferences : dict
        Preference weights for different compartments.
    weight : Initializer, optional
        Weight initialization.
    delay : Initializer, optional
        Delay initialization.
    """
    __module__ = 'braintools.conn'

    def __init__(
        self,
        placement_rule: str = 'uniform',
        compartment_preferences: Optional[Dict[int, float]] = None,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.placement_rule = placement_rule
        self.compartment_preferences = compartment_preferences or {
            BASAL_DENDRITE: 0.6,
            APICAL_DENDRITE: 0.4
        }
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
        """Generate synapse placements."""
        compartments = list(self.compartment_preferences.keys())
        probs = list(self.compartment_preferences.values())
        probs = np.array(probs) / np.sum(probs)

        compartment_mapping = {AXON: compartments}

        connection_prob = {}
        for comp, prob in zip(compartments, probs):
            connection_prob[(AXON, comp)] = prob * 0.2

        return CompartmentSpecific(
            compartment_mapping=compartment_mapping,
            connection_prob=connection_prob,
            weight=self.weight_init,
            delay=self.delay_init,
            seed=self.seed
        ).generate(pre_size, post_size, pre_positions, post_positions, **kwargs)


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
    """All-to-all compartment connectivity."""
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

        if len(result_data) == 4:
            pre_indices, post_indices, pre_compartments, post_compartments = result_data
            weights = None
            delays = None
        elif len(result_data) == 5:
            pre_indices, post_indices, pre_compartments, post_compartments, weights = result_data
            delays = None
        elif len(result_data) == 6:
            pre_indices, post_indices, pre_compartments, post_compartments, weights, delays = result_data
        else:
            raise ValueError(f"Custom function must return 4-6 values, got {len(result_data)}")

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
