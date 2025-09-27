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
Unified interface and convenience functions for braintools.conn.

This module provides high-level interface functions that automatically
select the appropriate connectivity type based on the model type and
provide convenient access to all connectivity patterns across the
different neuron model types.
"""

from typing import Optional, Union, Dict, Any, Literal

import brainunit as u
import numpy as np

from . import _compartment
from . import _point
from . import _population
from ._base import ConnectionResult

__all__ = [
    'connect',
    'random_connectivity',
    'distance_connectivity',
    'structured_connectivity',
    'get_available_patterns',
    'validate_connectivity',
    'convert_connectivity',
]

ModelType = Literal['point', 'population_rate', 'multi_compartment']


def connect(
    pattern: str,
    model_type: ModelType,
    pre_size: Union[int, tuple],
    post_size: Union[int, tuple],
    pre_positions: Optional[np.ndarray] = None,
    post_positions: Optional[np.ndarray] = None,
    **pattern_kwargs
) -> ConnectionResult:
    """Universal connectivity function that dispatches to appropriate model type.

    This is the main entry point for creating connectivity patterns. It automatically
    selects the correct implementation based on the model type and pattern name.

    Parameters
    ----------
    pattern : str
        Name of the connectivity pattern (e.g., 'Random', 'DistanceDependent').
    model_type : str
        Type of neuron model ('point', 'population_rate', 'multi_compartment').
    pre_size : int or tuple
        Size of presynaptic population/network.
    post_size : int or tuple
        Size of postsynaptic population/network.
    pre_positions : np.ndarray, optional
        Positions of presynaptic elements for spatial patterns.
    post_positions : np.ndarray, optional
        Positions of postsynaptic elements for spatial patterns.
    **pattern_kwargs
        Additional arguments specific to the connectivity pattern.

    Returns
    -------
    result : ConnectionResult
        Generated connectivity with appropriate model type.

    Examples
    --------
    Point neuron random connectivity:

    .. code-block:: python

        >>> result = connect(
        ...     pattern='Random',
        ...     model_type='point',
        ...     pre_size=1000,
        ...     post_size=1000,
        ...     prob=0.1
        ... )

    Population rate E-I network:

    .. code-block:: python

        >>> result = connect(
        ...     pattern='ExcitatoryInhibitory',
        ...     model_type='population_rate',
        ...     pre_size=2,
        ...     post_size=2,
        ...     exc_self_coupling=0.5,
        ...     inh_to_exc_coupling=-1.2
        ... )

    Multi-compartment axon-to-dendrite connectivity:

    .. code-block:: python

        >>> result = connect(
        ...     pattern='AxonToDendrite',
        ...     model_type='multi_compartment',
        ...     pre_size=100,
        ...     post_size=100,
        ...     connection_prob=0.1
        ... )

    Spatial connectivity with positions:

    .. code-block:: python

        >>> import brainunit as u
        >>> positions = np.random.uniform(0, 1000, (500, 2)) * u.um
        >>> result = connect(
        ...     pattern='DistanceDependent',
        ...     model_type='point',
        ...     pre_size=500,
        ...     post_size=500,
        ...     pre_positions=positions,
        ...     post_positions=positions,
        ...     sigma=100 * u.um,
        ...     decay='gaussian'
        ... )
    """
    # Get the appropriate module based on model type
    if model_type == 'point':
        module = _point
    elif model_type == 'population_rate':
        module = _population
    elif model_type == 'multi_compartment':
        module = _compartment
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be one of 'point', 'population_rate', 'multi_compartment'")

    # Get the pattern class
    if not hasattr(module, pattern):
        available = get_available_patterns(model_type)
        raise ValueError(f"Pattern '{pattern}' not available for model type '{model_type}'. "
                         f"Available patterns: {available}")

    pattern_class = getattr(module, pattern)

    # Create and execute the connectivity pattern
    conn = pattern_class(**pattern_kwargs)
    result = conn.generate(
        pre_size=pre_size,
        post_size=post_size,
        pre_positions=pre_positions,
        post_positions=post_positions
    )

    return result


def random_connectivity(
    model_type: ModelType,
    pre_size: Union[int, tuple],
    post_size: Union[int, tuple],
    prob: float = 0.1,
    **kwargs
) -> ConnectionResult:
    """Convenience function for random connectivity across all model types.

    Parameters
    ----------
    model_type : str
        Type of neuron model.
    pre_size : int or tuple
        Size of presynaptic population.
    post_size : int or tuple
        Size of postsynaptic population.
    prob : float
        Connection probability.
    **kwargs
        Additional model-specific parameters.

    Examples
    --------
    .. code-block:: python

        >>> # Point neuron random connectivity
        >>> result = random_connectivity('point', 1000, 1000, prob=0.1)
        >>>
        >>> # Population rate random connectivity
        >>> result = random_connectivity('population_rate', 5, 5, prob=0.3)
        >>>
        >>> # Multi-compartment random connectivity
        >>> result = random_connectivity('multi_compartment', 100, 100, prob=0.05)
    """
    if model_type == 'point':
        return connect('Random', model_type, pre_size, post_size, prob=prob, **kwargs)
    elif model_type == 'population_rate':
        return connect('RandomPopulations', model_type, pre_size, post_size, prob=prob, **kwargs)
    elif model_type == 'multi_compartment':
        return connect('RandomCompartment', model_type, pre_size, post_size, connection_prob=prob, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def distance_connectivity(
    model_type: ModelType,
    pre_size: Union[int, tuple],
    post_size: Union[int, tuple],
    pre_positions: np.ndarray,
    post_positions: np.ndarray,
    sigma: Union[float, u.Quantity],
    **kwargs
) -> ConnectionResult:
    """Convenience function for distance-dependent connectivity.

    Parameters
    ----------
    model_type : str
        Type of neuron model.
    pre_size : int or tuple
        Size of presynaptic population.
    post_size : int or tuple
        Size of postsynaptic population.
    pre_positions : np.ndarray
        Positions of presynaptic elements.
    post_positions : np.ndarray
        Positions of postsynaptic elements.
    sigma : float or Quantity
        Characteristic distance scale.
    **kwargs
        Additional model-specific parameters.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> positions = np.random.uniform(0, 1000, (100, 2)) * u.um
        >>> result = distance_connectivity(
        ...     'point', 100, 100, positions, positions,
        ...     sigma=100 * u.um, decay='gaussian'
        ... )
    """
    if model_type == 'point':
        return connect('DistanceDependent', model_type, pre_size, post_size,
                       pre_positions=pre_positions, post_positions=post_positions,
                       sigma=sigma, **kwargs)
    elif model_type == 'population_rate':
        return connect('PopulationDistance', model_type, pre_size, post_size,
                       pre_positions=pre_positions, post_positions=post_positions,
                       sigma=sigma, **kwargs)
    elif model_type == 'multi_compartment':
        return connect('MorphologyDistance', model_type, pre_size, post_size,
                       pre_positions=pre_positions, post_positions=post_positions,
                       sigma=sigma, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def structured_connectivity(
    model_type: ModelType,
    structure_type: str,
    pre_size: Union[int, tuple],
    post_size: Union[int, tuple],
    **kwargs
) -> ConnectionResult:
    """Convenience function for structured connectivity patterns.

    Parameters
    ----------
    model_type : str
        Type of neuron model.
    structure_type : str
        Type of structure ('ei', 'hierarchical', 'small_world', etc.).
    pre_size : int or tuple
        Size of presynaptic population.
    post_size : int or tuple
        Size of postsynaptic population.
    **kwargs
        Structure-specific parameters.

    Examples
    --------
    .. code-block:: python

        >>> # Excitatory-inhibitory network
        >>> result = structured_connectivity(
        ...     'point', 'ei', 1000, 1000,
        ...     exc_ratio=0.8, exc_prob=0.1, inh_prob=0.2
        ... )
        >>>
        >>> # Hierarchical populations
        >>> result = structured_connectivity(
        ...     'population_rate', 'hierarchical', 7, 7,
        ...     hierarchy_levels=[4, 2, 1],
        ...     feedforward_strength=0.6
        ... )
    """
    structure_patterns = {
        'point': {
            'ei': 'ExcitatoryInhibitory',
            'small_world': 'SmallWorld',
            'scale_free': 'ScaleFree',
            'modular': 'Modular'
        },
        'population_rate': {
            'ei': 'ExcitatoryInhibitory',
            'hierarchical': 'HierarchicalPopulations',
            'wilson_cowan': 'WilsonCowanNetwork',
            'mean_field': 'MeanField'
        },
        'multi_compartment': {
            'dendritic_tree': 'DendriticTree',
            'axonal_projection': 'AxonalProjection',
            'morphology_distance': 'MorphologyDistance'
        }
    }

    if model_type not in structure_patterns:
        raise ValueError(f"Unknown model type: {model_type}")

    if structure_type not in structure_patterns[model_type]:
        available = list(structure_patterns[model_type].keys())
        raise ValueError(f"Unknown structure type '{structure_type}' for model '{model_type}'. "
                         f"Available: {available}")

    pattern = structure_patterns[model_type][structure_type]
    return connect(pattern, model_type, pre_size, post_size, **kwargs)


def get_available_patterns(model_type: Optional[ModelType] = None) -> Dict[str, list]:
    """Get list of available connectivity patterns for each model type.

    Parameters
    ----------
    model_type : str, optional
        Specific model type to query. If None, returns all patterns.

    Returns
    -------
    patterns : dict or list
        Dictionary mapping model types to available patterns, or list if model_type specified.

    Examples
    --------
    .. code-block:: python

        >>> # Get all patterns
        >>> all_patterns = get_available_patterns()
        >>> print(all_patterns.keys())
        >>>
        >>> # Get patterns for specific model type
        >>> point_patterns = get_available_patterns('point')
        >>> print(point_patterns)
    """
    patterns = {
        'point': [
            'Random', 'AllToAll', 'OneToOne', 'FixedProbability',
            'DistanceDependent', 'Gaussian', 'Exponential', 'Ring', 'Grid',
            'RadialPatches', 'SmallWorld', 'ScaleFree', 'Regular', 'Modular',
            'ClusteredRandom', 'ExcitatoryInhibitory', 'DalesPrinciple',
            'SynapticPlasticity', 'ActivityDependent', 'Custom'
        ],
        'population_rate': [
            'PopulationCoupling', 'MeanField', 'AllToAllPopulations', 'RandomPopulations',
            'ExcitatoryInhibitory', 'FeedforwardInhibition', 'RecurrentAmplification',
            'CompetitiveNetwork', 'HierarchicalPopulations', 'FeedforwardHierarchy',
            'RecurrentHierarchy', 'LayeredNetwork', 'PopulationDistance',
            'RateDependent', 'WilsonCowanNetwork', 'FiringRateNetworks',
            'CustomPopulation'
        ],
        'multi_compartment': [
            'CompartmentSpecific', 'RandomCompartment', 'AllToAllCompartments',
            'SomaToDendrite', 'AxonToSoma', 'DendriteToSoma', 'AxonToDendrite',
            'DendriteToDendrite', 'ProximalTargeting', 'DistalTargeting',
            'BranchSpecific', 'MorphologyDistance', 'DendriticTree',
            'BasalDendriteTargeting', 'ApicalDendriteTargeting', 'DendriticIntegration',
            'AxonalProjection', 'AxonalBranching', 'AxonalArborization',
            'TopographicProjection', 'SynapticPlacement', 'SynapticClustering',
            'ActivityDependentSynapses', 'CustomCompartment'
        ]
    }

    if model_type is not None:
        if model_type not in patterns:
            raise ValueError(f"Unknown model type: {model_type}")
        return patterns[model_type]

    return patterns


def validate_connectivity(
    result: ConnectionResult,
    expected_model_type: Optional[ModelType] = None,
    check_consistency: bool = True
) -> bool:
    """Validate a connectivity result for consistency and correctness.

    Parameters
    ----------
    result : ConnectionResult
        Connectivity result to validate.
    expected_model_type : str, optional
        Expected model type to check against.
    check_consistency : bool
        Whether to perform detailed consistency checks.

    Returns
    -------
    is_valid : bool
        Whether the connectivity result is valid.

    Examples
    --------
    .. code-block:: python

        >>> result = random_connectivity('point', 100, 100, prob=0.1)
        >>> is_valid = validate_connectivity(result, expected_model_type='point')
        >>> print(f"Connectivity is valid: {is_valid}")
    """
    try:
        # Check basic structure
        if not hasattr(result, 'pre_indices') or not hasattr(result, 'post_indices'):
            return False

        if not hasattr(result, 'model_type'):
            return False

        # Check model type if specified
        if expected_model_type is not None and result.model_type != expected_model_type:
            return False

        # Check consistency if requested
        if check_consistency:
            # Check array lengths
            if len(result.pre_indices) != len(result.post_indices):
                return False

            if result.weights is not None and len(result.weights) != len(result.pre_indices):
                return False

            if result.delays is not None and len(result.delays) != len(result.pre_indices):
                return False

            # Check model-specific fields
            if result.model_type == 'multi_compartment':
                if hasattr(result, 'pre_compartments') and result.pre_compartments is not None:
                    if len(result.pre_compartments) != len(result.pre_indices):
                        return False

                if hasattr(result, 'post_compartments') and result.post_compartments is not None:
                    if len(result.post_compartments) != len(result.pre_indices):
                        return False

        return True

    except Exception:
        return False


def convert_connectivity(
    result: ConnectionResult,
    target_model_type: ModelType,
    conversion_params: Optional[Dict[str, Any]] = None
) -> ConnectionResult:
    """Convert connectivity between different model types.

    This function attempts to convert connectivity results from one model type
    to another, with appropriate transformations and approximations.

    Parameters
    ----------
    result : ConnectionResult
        Original connectivity result to convert.
    target_model_type : str
        Target model type for conversion.
    conversion_params : dict, optional
        Parameters controlling the conversion process.

    Returns
    -------
    converted_result : ConnectionResult
        Converted connectivity result.

    Examples
    --------
    .. code-block:: python

        >>> # Convert point neuron connectivity to population rate
        >>> point_result = random_connectivity('point', 1000, 1000, prob=0.1)
        >>> pop_result = convert_connectivity(point_result, 'population_rate',
        ...                                  conversion_params={'n_populations': 10})

    Note
    ----
    Not all conversions are meaningful or possible. This function provides
    best-effort conversions with appropriate warnings.
    """
    if result.model_type == target_model_type:
        return result

    conversion_params = conversion_params or {}

    if result.model_type == 'point' and target_model_type == 'population_rate':
        # Convert point neuron connectivity to population connectivity
        # This requires aggregating individual neurons into populations
        n_populations = conversion_params.get('n_populations', 10)

        # Simple approach: group neurons into populations and average connectivity
        pre_size = result.pre_size if hasattr(result, 'pre_size') and result.pre_size is not None else np.max(result.pre_indices) + 1
        post_size = result.post_size if hasattr(result, 'post_size') and result.post_size is not None else np.max(result.post_indices) + 1

        pre_pop_size = pre_size // n_populations
        post_pop_size = post_size // n_populations

        # Create population coupling matrix
        coupling_matrix = np.zeros((n_populations, n_populations))

        for i in range(n_populations):
            for j in range(n_populations):
                # Find connections between populations i and j
                pre_start, pre_end = i * pre_pop_size, (i + 1) * pre_pop_size
                post_start, post_end = j * post_pop_size, (j + 1) * post_pop_size

                # Count connections in this block
                mask = ((result.pre_indices >= pre_start) & (result.pre_indices < pre_end) &
                        (result.post_indices >= post_start) & (result.post_indices < post_end))

                if np.any(mask):
                    if result.weights is not None:
                        coupling_matrix[i, j] = np.mean(result.weights[mask])
                    else:
                        coupling_matrix[i, j] = np.sum(mask) / (pre_pop_size * post_pop_size)

        # Create new population connectivity
        pre_indices, post_indices = np.where(coupling_matrix != 0)
        weights = coupling_matrix[pre_indices, post_indices]

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            weights=weights,
            pre_size=n_populations,
            post_size=n_populations,
            model_type='population_rate',
            metadata={
                'converted_from': 'point',
                'original_pre_size': pre_size,
                'original_post_size': post_size,
                'n_populations': n_populations
            }
        )

    elif result.model_type == 'population_rate' and target_model_type == 'point':
        # Convert population connectivity to point neuron connectivity
        # This requires expanding populations into individual neurons
        neurons_per_pop = conversion_params.get('neurons_per_population', 100)

        expanded_pre_indices = []
        expanded_post_indices = []
        expanded_weights = []

        n_connections_total = len(result.pre_indices)
        rng = np.random.RandomState(42)

        for conn_idx in range(n_connections_total):
            pre_pop = result.pre_indices[conn_idx]
            post_pop = result.post_indices[conn_idx]
            weight = result.weights[conn_idx] if result.weights is not None else 1.0

            # Create random connections within the population pair
            n_connections = int(neurons_per_pop * neurons_per_pop * 0.1)  # 10% connectivity

            for _ in range(n_connections):
                pre_neuron = pre_pop * neurons_per_pop + rng.randint(neurons_per_pop)
                post_neuron = post_pop * neurons_per_pop + rng.randint(neurons_per_pop)

                expanded_pre_indices.append(pre_neuron)
                expanded_post_indices.append(post_neuron)
                expanded_weights.append(weight / neurons_per_pop)  # Scale weight

        # Calculate sizes
        if hasattr(result, 'pre_size') and result.pre_size is not None:
            n_pre_pops = result.pre_size
        else:
            n_pre_pops = np.max(result.pre_indices) + 1

        if hasattr(result, 'post_size') and result.post_size is not None:
            n_post_pops = result.post_size
        else:
            n_post_pops = np.max(result.post_indices) + 1

        return ConnectionResult(
            np.array(expanded_pre_indices, dtype=np.int64),
            np.array(expanded_post_indices, dtype=np.int64),
            weights=np.array(expanded_weights),
            pre_size=n_pre_pops * neurons_per_pop,
            post_size=n_post_pops * neurons_per_pop,
            model_type='point',
            metadata={
                'converted_from': 'population_rate',
                'neurons_per_population': neurons_per_pop
            }
        )

    else:
        raise ValueError(f"Conversion from '{result.model_type}' to '{target_model_type}' not implemented")


# Convenience aliases for backward compatibility
def create_connectivity(*args, **kwargs):
    """Alias for connect() function."""
    return connect(*args, **kwargs)


def make_random(*args, **kwargs):
    """Alias for random_connectivity() function."""
    return random_connectivity(*args, **kwargs)


def make_distance(*args, **kwargs):
    """Alias for distance_connectivity() function."""
    return distance_connectivity(*args, **kwargs)
