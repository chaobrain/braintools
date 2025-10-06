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
Base classes and interfaces for the modular connectivity system.

This module provides the foundational architecture for decoupled connectivity
generation across different neuron model types:
- Point neurons: Single-compartment integrate-and-fire models
- Population rate models: Mean-field population dynamics
- Multi-compartment models: Detailed morphological neuron models
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any, Sequence

import brainevent
import brainunit as u
import numpy as np

__all__ = [
    'ConnectionResult',
    'Connectivity',
    'PointConnectivity',
    'MultiCompartmentConnectivity',
    'ScaledConnectivity',
    'CompositeConnectivity',
]


class ConnectionResult:
    """Universal container for connectivity results across all neuron model types.

    This class provides a unified interface for representing connections while
    maintaining type-specific information and capabilities.

    Parameters
    ----------
    pre_indices : np.ndarray
        Presynaptic element indices (neurons, populations, or compartments).
    post_indices : np.ndarray
        Postsynaptic element indices (neurons, populations, or compartments).
    pre_size : int or tuple of int, optional
        Size of the presynaptic population (number of elements or shape).
    post_size : int or tuple of int, optional
        Size of the postsynaptic population (number of elements or shape).
    weights : np.ndarray or Quantity, optional
        Connection weights with appropriate units for the model type.
    delays : np.ndarray or Quantity, optional
        Connection delays with time units (e.g., ms).
    model_type : str
        Type of neuron model ('point', 'population_rate', 'multi_compartment').
    pre_positions: np.ndarray, optional
        Positions of presynaptic elements (for distance calculations).
    post_positions: np.ndarray, optional
        Positions of postsynaptic elements (for distance calculations).
    pre_compartments: np.ndarray, optional
        Presynaptic compartment indices (for multi-compartment models).
    post_compartments: np.ndarray, optional
        Postsynaptic compartment indices (for multi-compartment models).
    metadata : dict, optional
        Model-specific metadata and parameters.
    """

    __module__ = 'braintools.conn'

    def __init__(
        self,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        pre_size: Optional[int | Sequence[int]],
        post_size: Optional[int | Sequence[int]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        pre_compartments: Optional[np.ndarray] = None,
        post_compartments: Optional[np.ndarray] = None,
        weights: Optional[Union[np.ndarray, u.Quantity]] = None,
        delays: Optional[Union[np.ndarray, u.Quantity]] = None,
        model_type: str = 'point',
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Core connectivity data
        self.pre_indices = np.asarray(pre_indices, dtype=np.int64)
        self.post_indices = np.asarray(post_indices, dtype=np.int64)
        self.model_type = model_type
        self.pre_positions = pre_positions
        self.post_positions = post_positions
        self.metadata = metadata or {}
        self.pre_size = pre_size
        self.post_size = post_size

        # Handle weights with units
        self.weights = weights

        # Handle delays with units
        self.delays = delays

        # Store type-specific fields
        self.pre_compartments = pre_compartments
        self.post_compartments = post_compartments

        # Validate consistency
        self._validate()

    def _validate(self):
        """Validate consistency of connection data."""
        n_connections = len(self.pre_indices)
        if len(self.post_indices) != n_connections:
            raise ValueError(
                f"pre_indices and post_indices must have same length, "
                f"got {len(self.pre_indices)} vs {len(self.post_indices)}"
            )
        if self.weights is not None:
            if not u.math.size(self.weights) in [n_connections, 1]:
                raise ValueError(
                    f"weights must have same length as indices, "
                    f"got {u.math.size(self.weights)} vs {n_connections}"
                )
        if self.delays is not None:
            if not u.math.size(self.delays) in [n_connections, 1]:
                raise ValueError(
                    f"delays must have same length as indices, "
                    f"got {len(self.delays)} vs {n_connections}"
                )

    @property
    def n_connections(self) -> int:
        """Number of connections."""
        return len(self.pre_indices)

    @property
    def positions(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Backward compatibility property for positions."""
        if self.pre_positions is not None and self.post_positions is not None:
            return (self.pre_positions, self.post_positions)
        return None

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the connectivity (pre_size, post_size)."""
        if self.pre_size is None:
            pre_size = (max(self.pre_indices) + 1 if self.pre_indices.size > 0 else 0)
        elif isinstance(self.pre_size, int):
            pre_size = self.pre_size
        else:
            pre_size = int(np.prod(self.pre_size))

        if self.post_size is None:
            post_size = (max(self.post_indices) + 1 if self.post_indices.size > 0 else 0)
        elif isinstance(self.post_size, int):
            post_size = self.post_size
        else:
            post_size = int(np.prod(self.post_size))
        return (pre_size, post_size)

    def weight2dense(self):
        """Convert to dense connectivity matrix."""
        weights = 1.0 if self.weights is None else self.weights
        weights, unit = u.split_mantissa_unit(weights)
        matrix = np.zeros(self.shape)
        matrix[self.pre_indices, self.post_indices] = weights

        return u.maybe_decimal(matrix * unit)

    def weight2csr(self):
        """Convert to sparse connectivity matrix in CSR format."""
        indices, indptr = compute_csr_indices_indptr(self.pre_indices, self.post_indices, self.shape)
        weights = 1.0 if self.weights is None else self.weights
        csr = brainevent.CSR((weights, indices, indptr), shape=self.shape)
        return csr

    def delay2matrix(self):
        """Convert delays to dense connectivity matrix."""
        delays = 0.0 if self.delays is None else self.delays
        delays, unit = u.split_mantissa_unit(delays)
        matrix = np.zeros(self.shape)
        matrix[self.pre_indices, self.post_indices] = delays
        return u.maybe_decimal(matrix * unit)

    def delay2csr(self):
        """Convert delays to sparse connectivity matrix in CSR format."""
        indices, indptr = compute_csr_indices_indptr(self.pre_indices, self.post_indices, self.shape)
        delays = 0.0 if self.delays is None else self.delays
        csr = brainevent.CSR((delays, indices, indptr), shape=self.shape)
        return csr

    def get_distances(self) -> Optional[u.Quantity]:
        """Calculate distances between connected elements."""
        if self.pre_positions is None or self.post_positions is None:
            return None

        pre_positions = self.pre_positions
        post_positions = self.post_positions
        if len(self.pre_indices) == 0:
            return u.maybe_decimal(u.Quantity([], unit=u.get_unit(pre_positions)))

        pre_coords = pre_positions[self.pre_indices]
        post_coords = post_positions[self.post_indices]

        diff = pre_coords - post_coords
        if isinstance(diff, u.Quantity):
            distances = u.math.sqrt(u.math.sum(diff ** 2, axis=1))
        else:
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
        return distances


class Connectivity(ABC):
    """Abstract base class for all connectivity patterns.

    This provides the common interface and shared functionality across
    all neuron model types while allowing for type-specific implementations.
    """

    __module__ = 'braintools.conn'

    def __init__(
        self,
        pre_size: Optional[Union[int, Tuple[int, ...]]] = None,
        post_size: Optional[Union[int, Tuple[int, ...]]] = None,
        seed: Optional[int] = None
    ):
        self.pre_size = pre_size
        self.post_size = post_size
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self._cached_result = None

    def __call__(
        self,
        pre_size: Optional[Union[int, Tuple[int, ...]]] = None,
        post_size: Optional[Union[int, Tuple[int, ...]]] = None,
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        recompute: bool = False,
        **kwargs
    ) -> ConnectionResult:
        """Generate connectivity pattern."""
        if self._cached_result is None or recompute:
            effective_pre_size = pre_size if pre_size is not None else self.pre_size
            effective_post_size = post_size if post_size is not None else self.post_size

            self._cached_result = self._generate(
                pre_size=effective_pre_size,
                post_size=effective_post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                **kwargs
            )
        return self._cached_result

    def _generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate the connectivity pattern. Subclasses must implement this."""
        return self.generate(
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            **kwargs
        )

    @abstractmethod
    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate the connectivity pattern. Subclasses must implement this."""
        pass

    # Arithmetic operations for composability
    def __add__(self, other):
        """Union of connectivity patterns."""
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, 'union')
        raise TypeError("Can only add Connectivity objects")

    def __mul__(self, other):
        """Intersection or scaling of connectivity patterns."""
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, 'intersection')
        elif isinstance(other, (int, float)):
            return ScaledConnectivity(self, weight_factor=other)
        raise TypeError("Can multiply by Connectivity object or scalar")

    def __sub__(self, other):
        """Difference of connectivity patterns."""
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, 'difference')
        raise TypeError("Can only subtract Connectivity objects")

    def weight_scale(self, factor: float):
        """Scale connection weights by a factor."""
        return ScaledConnectivity(self, weight_factor=factor)

    def delay_scale(self, factor: float):
        """Scale connection delays by a factor."""
        return ScaledConnectivity(self, delay_factor=factor)


class PointConnectivity(Connectivity):
    """Base class for point neuron connectivity patterns.

    Point neurons are single-compartment models where each connection
    represents a synapse between two neurons.
    """

    __module__ = 'braintools.conn'

    def _generate(self, **kwargs) -> ConnectionResult:
        """Generate point neuron connectivity."""
        result = self.generate(**kwargs)
        result.model_type = 'point'
        return result

    @abstractmethod
    def generate(self, **kwargs) -> ConnectionResult:
        """Generate point neuron specific connections."""
        pass


class MultiCompartmentConnectivity(Connectivity):
    """Base class for multi-compartment neuron connectivity patterns.

    Multi-compartment models have detailed morphology with multiple compartments
    (soma, dendrites, axon). Connections can target specific compartments.
    """

    __module__ = 'braintools.conn'

    def _generate(self, **kwargs) -> ConnectionResult:
        """Generate multi-compartment connectivity."""
        result = self.generate(**kwargs)
        result.model_type = 'multi_compartment'
        return result

    @abstractmethod
    def generate(self, **kwargs) -> ConnectionResult:
        """Generate multi-compartment specific connections."""
        pass


# Composite connectivity for combining patterns
class CompositeConnectivity(Connectivity):
    """Composite connectivity created by combining patterns."""

    __module__ = 'braintools.conn'

    def __init__(
        self,
        conn1: Connectivity,
        conn2: Connectivity,
        operator: str
    ):
        assert conn1.pre_size == conn2.pre_size, f"Pre sizes must match, got {conn1.pre_size} vs {conn2.pre_size}"
        assert conn1.post_size == conn2.post_size, f"Post sizes must match, got {conn1.post_size} vs {conn2.post_size}"
        assert operator in ['union', 'intersection', 'difference'], f"Invalid operator, got {operator}"
        super().__init__(pre_size=conn1.pre_size, post_size=conn1.post_size)
        self.conn1 = conn1
        self.conn2 = conn2
        self.operator = operator

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate composite connectivity."""
        result1 = self.conn1(**kwargs)
        result2 = self.conn2(**kwargs)

        if self.operator == 'union':
            return self._union(result1, result2)
        elif self.operator == 'intersection':
            return self._intersection(result1, result2)
        elif self.operator == 'difference':
            return self._difference(result1, result2)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def _union(
        self,
        result1: ConnectionResult,
        result2: ConnectionResult
    ) -> ConnectionResult:
        """Combine connections from both patterns using optimized array operations."""
        assert result1.pre_size == result2.pre_size, (
            f"Pre sizes must match for union, got {result1.pre_size} vs {result2.pre_size}"
        )
        assert result1.post_size == result2.post_size, (
            f"Post sizes must match for union, got {result1.post_size} vs {result2.post_size}"
        )

        # Concatenate all connections
        all_pre = np.concatenate([result1.pre_indices, result2.pre_indices])
        all_post = np.concatenate([result1.post_indices, result2.post_indices])

        # Handle compartment information for multi-compartment models
        is_multicompartment = result1.model_type == 'multi_compartment'
        all_pre_comp = None
        all_post_comp = None
        if is_multicompartment:
            pre_comp1 = (
                result1.pre_compartments
                if result1.pre_compartments is not None else
                np.zeros(len(result1.pre_indices), dtype=np.int64)
            )
            pre_comp2 = (
                result2.pre_compartments
                if result2.pre_compartments is not None else
                np.zeros(len(result2.pre_indices), dtype=np.int64)
            )
            post_comp1 = (
                result1.post_compartments
                if result1.post_compartments is not None
                else np.zeros(len(result1.post_indices), dtype=np.int64)
            )
            post_comp2 = (
                result2.post_compartments
                if result2.post_compartments is not None else
                np.zeros(len(result2.post_indices), dtype=np.int64)
            )
            all_pre_comp = np.concatenate([pre_comp1, pre_comp2])
            all_post_comp = np.concatenate([post_comp1, post_comp2])

        if len(all_pre) == 0:
            result_kwargs = {
                'pre_size': result1.pre_size,
                'post_size': result1.post_size,
                'pre_positions': result1.pre_positions,
                'post_positions': result1.post_positions,
                'model_type': result1.model_type,
                'metadata': merge_dict(result1.metadata, result2.metadata),
            }
            if is_multicompartment:
                result_kwargs['pre_compartments'] = np.array([], dtype=np.int64)
                result_kwargs['post_compartments'] = np.array([], dtype=np.int64)
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                **result_kwargs
            )

        # Create connection encoding for deduplication
        # For multi-compartment: encode (pre_idx, post_idx, pre_comp, post_comp)
        # For others: encode (pre_idx, post_idx)
        if isinstance(result1.pre_size, tuple):
            max_post = int(np.prod(result1.post_size))
        elif isinstance(result1.post_size, tuple):
            max_post = int(np.prod(result1.post_size))
        else:
            max_post = max(np.max(all_post) + 1, result1.post_size, result2.post_size)

        if is_multicompartment:
            # Encode with compartment information
            max_comp = 10  # Assume max 10 compartment types
            conn_codes = (all_pre * max_post * max_comp * max_comp +
                          all_post * max_comp * max_comp +
                          all_pre_comp * max_comp +
                          all_post_comp)
        else:
            conn_codes = all_pre * max_post + all_post

        # Find unique connections and their first occurrence indices
        unique_codes, unique_indices = np.unique(conn_codes, return_index=True)

        # Decode back to (pre, post) pairs and compartments
        if is_multicompartment:
            max_comp = 10
            pre_indices = unique_codes // (max_post * max_comp * max_comp)
            remainder = unique_codes % (max_post * max_comp * max_comp)
            post_indices = remainder // (max_comp * max_comp)
            remainder2 = remainder % (max_comp * max_comp)
            pre_compartments = remainder2 // max_comp
            post_compartments = remainder2 % max_comp
        else:
            pre_indices = unique_codes // max_post
            post_indices = unique_codes % max_post

        # Handle weights using array operations
        weights = None
        if result1.weights is not None or result2.weights is not None:
            # Concatenate weights, handling None cases
            weights1_vals, weights1_unit = (
                u.split_mantissa_unit(result1.weights) if result1.weights is not None else (None, None)
            )
            weights2_vals, weights2_unit = (
                u.split_mantissa_unit(result2.weights) if result2.weights is not None else (None, None)
            )

            # Determine common unit
            common_unit = weights1_unit or weights2_unit

            # Prepare weight arrays
            if weights1_vals is None:
                weights1_array = np.zeros(len(result1.pre_indices))
            elif u.math.isscalar(weights1_vals):
                weights1_array = np.full(len(result1.pre_indices), weights1_vals)
            else:
                weights1_array = np.asarray(weights1_vals)

            if weights2_vals is None:
                weights2_array = np.zeros(len(result2.pre_indices))
            elif u.math.isscalar(weights2_vals):
                weights2_array = np.full(len(result2.pre_indices), weights2_vals)
            else:
                weights2_array = np.asarray(weights2_vals)

            # Convert weights2 to common unit if needed
            if weights2_unit is not None and common_unit is not None and weights2_unit != common_unit:
                weights2_array = u.Quantity(weights2_array, unit=weights2_unit).to(common_unit).mantissa

            # Concatenate all weights (result1 has priority due to order)
            all_weights = np.concatenate([weights1_array, weights2_array])

            # Select weights for unique connections (first occurrence wins, result1 comes first)
            final_weights = all_weights[unique_indices]

            # Apply unit if needed
            if common_unit is not None:
                weights = u.maybe_decimal(final_weights * common_unit)
            else:
                weights = final_weights

        # Handle delays using similar array operations
        delays = None
        if result1.delays is not None or result2.delays is not None:
            # Concatenate delays, handling None cases
            delays1_vals, delays1_unit = (
                u.split_mantissa_unit(result1.delays) if result1.delays is not None else (None, None)
            )
            delays2_vals, delays2_unit = (
                u.split_mantissa_unit(result2.delays) if result2.delays is not None else (None, None)
            )

            # Determine common unit
            common_unit = delays1_unit or delays2_unit

            # Prepare delay arrays
            if delays1_vals is None:
                delays1_array = np.zeros(len(result1.pre_indices))
            elif u.math.isscalar(delays1_vals):
                delays1_array = np.full(len(result1.pre_indices), delays1_vals)
            else:
                delays1_array = np.asarray(delays1_vals)

            if delays2_vals is None:
                delays2_array = np.zeros(len(result2.pre_indices))
            elif u.math.isscalar(delays2_vals):
                delays2_array = np.full(len(result2.pre_indices), delays2_vals)
            else:
                delays2_array = np.asarray(delays2_vals)

            # Convert delays2 to common unit if needed
            if delays2_unit is not None and common_unit is not None and delays2_unit != common_unit:
                delays2_array = u.Quantity(delays2_array, delays2_unit).to(common_unit).mantissa

            # Concatenate all delays (result1 has priority due to order)
            all_delays = np.concatenate([delays1_array, delays2_array])

            # Select delays for unique connections (first occurrence wins, result1 comes first)
            final_delays = all_delays[unique_indices]

            # Apply unit if needed
            if common_unit is not None:
                delays = u.maybe_decimal(final_delays * common_unit)
            else:
                delays = final_delays

        result_kwargs = {
            'weights': weights,
            'delays': delays,
            'pre_size': result1.pre_size,
            'post_size': result1.post_size,
            'pre_positions': result1.pre_positions or result2.pre_positions,
            'post_positions': result1.post_positions or result2.post_positions,
            'model_type': result1.model_type,
            'metadata': merge_dict(
                result1.metadata,
                result2.metadata,
                {
                    'operation': 'union',
                    'n_conn1': len(result1.pre_indices),
                    'n_conn2': len(result2.pre_indices)
                }
            ),
        }

        if is_multicompartment:
            result_kwargs['pre_compartments'] = pre_compartments.astype(np.int64)
            result_kwargs['post_compartments'] = post_compartments.astype(np.int64)

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            **result_kwargs
        )

    def _intersection(self, result1: ConnectionResult, result2: ConnectionResult) -> ConnectionResult:
        """Keep only connections present in both patterns using optimized array operations."""
        # Handle compartment information
        is_multicompartment = result1.model_type == 'multi_compartment'

        # Create connection encoding for both results
        if isinstance(result1.post_size, tuple):
            max_post = int(np.prod(result1.post_size))
        else:
            max_post = max(
                np.max(result1.post_indices) + 1 if result1.post_indices.size > 0 else 0,
                np.max(result2.post_indices) + 1 if result2.post_indices.size > 0 else 0,
                result1.post_size if result1.post_size is not None else 1,
                result2.post_size if result2.post_size is not None else 1
            )

        if is_multicompartment:
            max_comp = 10
            pre_comp1 = (
                result1.pre_compartments
                if result1.pre_compartments is not None else
                np.zeros(len(result1.pre_indices), dtype=np.int64)
            )
            pre_comp2 = (
                result2.pre_compartments
                if result2.pre_compartments is not None else
                np.zeros(len(result2.pre_indices), dtype=np.int64)
            )
            post_comp1 = (
                result1.post_compartments
                if result1.post_compartments is not None
                else np.zeros(len(result1.post_indices), dtype=np.int64)
            )
            post_comp2 = (
                result2.post_compartments
                if result2.post_compartments is not None else
                np.zeros(len(result2.post_indices), dtype=np.int64)
            )

            conn_codes1 = (result1.pre_indices * max_post * max_comp * max_comp +
                           result1.post_indices * max_comp * max_comp +
                           pre_comp1 * max_comp +
                           post_comp1)
            conn_codes2 = (result2.pre_indices * max_post * max_comp * max_comp +
                           result2.post_indices * max_comp * max_comp +
                           pre_comp2 * max_comp +
                           post_comp2)
        else:
            conn_codes1 = result1.pre_indices * max_post + result1.post_indices
            conn_codes2 = result2.pre_indices * max_post + result2.post_indices

        # Find intersection using array operations
        common_codes = np.intersect1d(conn_codes1, conn_codes2, assume_unique=False)

        if common_codes.size == 0:
            result_kwargs = {
                'pre_size': result1.pre_size,
                'post_size': result1.post_size,
                'model_type': result1.model_type
            }
            if is_multicompartment:
                result_kwargs['pre_compartments'] = np.array([], dtype=np.int64)
                result_kwargs['post_compartments'] = np.array([], dtype=np.int64)
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                **result_kwargs
            )

        # Decode back to (pre, post) pairs and compartments
        if is_multicompartment:
            max_comp = 10
            pre_indices = common_codes // (max_post * max_comp * max_comp)
            remainder = common_codes % (max_post * max_comp * max_comp)
            post_indices = remainder // (max_comp * max_comp)
            remainder2 = remainder % (max_comp * max_comp)
            pre_compartments = remainder2 // max_comp
            post_compartments = remainder2 % max_comp
        else:
            pre_indices = common_codes // max_post
            post_indices = common_codes % max_post

        # Create masks for efficient indexing
        mask1 = np.isin(conn_codes1, common_codes)
        mask2 = np.isin(conn_codes2, common_codes)

        # Sort indices to align values from both results
        sort_idx1 = np.argsort(conn_codes1[mask1])
        sort_idx2 = np.argsort(conn_codes2[mask2])

        # Handle weights using array operations (multiply weights from both patterns)
        weights = None
        if result1.weights is not None or result2.weights is not None:
            weights1_vals, weights1_unit = (
                u.split_mantissa_unit(result1.weights) if result1.weights is not None else (None, None)
            )
            weights2_vals, weights2_unit = (
                u.split_mantissa_unit(result2.weights) if result2.weights is not None else (None, None)
            )

            # Prepare weight arrays
            if weights1_vals is None:
                weights1_array = np.ones(len(result1.pre_indices))
            elif u.math.isscalar(weights1_vals):
                weights1_array = np.full(len(result1.pre_indices), weights1_vals)
            else:
                weights1_array = np.asarray(weights1_vals)

            if weights2_vals is None:
                weights2_array = np.ones(len(result2.pre_indices))
            elif u.math.isscalar(weights2_vals):
                weights2_array = np.full(len(result2.pre_indices), weights2_vals)
            else:
                weights2_array = np.asarray(weights2_vals)

            # Extract and multiply weights for common connections
            weights1_common = weights1_array[mask1][sort_idx1]
            weights2_common = weights2_array[mask2][sort_idx2]
            final_weights = weights1_common * weights2_common

            # Handle units
            combined_unit = None
            if weights1_unit is not None and weights2_unit is not None:
                combined_unit = weights1_unit * weights2_unit
            elif weights1_unit is not None:
                combined_unit = weights1_unit
            elif weights2_unit is not None:
                combined_unit = weights2_unit

            if combined_unit is not None:
                weights = u.maybe_decimal(final_weights * combined_unit)
            else:
                weights = final_weights

        # Handle delays using array operations (average delays from both patterns)
        delays = None
        if result1.delays is not None or result2.delays is not None:
            delays1_vals, delays1_unit = (
                u.split_mantissa_unit(result1.delays) if result1.delays is not None else (None, None)
            )
            delays2_vals, delays2_unit = (
                u.split_mantissa_unit(result2.delays) if result2.delays is not None else (None, None)
            )

            common_unit = delays1_unit or delays2_unit or u.ms

            # Prepare delay arrays
            if delays1_vals is None:
                delays1_array = np.zeros(len(result1.pre_indices))
            elif u.math.isscalar(delays1_vals):
                delays1_array = np.full(len(result1.pre_indices), delays1_vals)
            else:
                delays1_array = np.asarray(delays1_vals)

            if delays2_vals is None:
                delays2_array = np.zeros(len(result2.pre_indices))
            elif u.math.isscalar(delays2_vals):
                delays2_array = np.full(len(result2.pre_indices), delays2_vals)
            else:
                delays2_array = np.asarray(delays2_vals)

            # Extract and average delays for common connections
            delays1_common = delays1_array[mask1][sort_idx1]
            delays2_common = delays2_array[mask2][sort_idx2]
            final_delays = (delays1_common + delays2_common) / 2.0

            delays = u.maybe_decimal(final_delays * common_unit)

        result_kwargs = {
            'pre_size': result1.pre_size,
            'post_size': result1.post_size,
            'weights': weights,
            'delays': delays,
            'model_type': result1.model_type,
            'pre_positions': result1.pre_positions,
            'post_positions': result1.post_positions,
            'metadata': {
                **result1.metadata,
                **result2.metadata,
                'operation': 'intersection'
            },
        }

        if is_multicompartment:
            result_kwargs['pre_compartments'] = pre_compartments.astype(np.int64)
            result_kwargs['post_compartments'] = post_compartments.astype(np.int64)

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            **result_kwargs
        )

    def _difference(self, result1: ConnectionResult, result2: ConnectionResult) -> ConnectionResult:
        """Keep connections from result1 that are not in result2 using optimized array operations."""
        # Handle compartment information
        is_multicompartment = result1.model_type == 'multi_compartment'

        # Create connection encoding for both results
        if isinstance(result1.post_size, tuple):
            max_post = int(np.prod(result1.post_size))
        else:
            max_post = max(
                np.max(result1.post_indices) + 1 if result1.post_indices.size > 0 else 0,
                np.max(result2.post_indices) + 1 if result2.post_indices.size > 0 else 0,
                result1.post_size if result1.post_size is not None else 1,
                result2.post_size if result2.post_size is not None else 1
            )

        if is_multicompartment:
            max_comp = 10
            pre_comp1 = (
                result1.pre_compartments
                if result1.pre_compartments is not None else
                np.zeros(len(result1.pre_indices), dtype=np.int64)
            )
            pre_comp2 = (
                result2.pre_compartments
                if result2.pre_compartments is not None else
                np.zeros(len(result2.pre_indices), dtype=np.int64)
            )
            post_comp1 = (
                result1.post_compartments
                if result1.post_compartments is not None
                else np.zeros(len(result1.post_indices), dtype=np.int64)
            )
            post_comp2 = (
                result2.post_compartments
                if result2.post_compartments is not None else
                np.zeros(len(result2.post_indices), dtype=np.int64)
            )

            conn_codes1 = (result1.pre_indices * max_post * max_comp * max_comp +
                           result1.post_indices * max_comp * max_comp +
                           pre_comp1 * max_comp +
                           post_comp1)
            conn_codes2 = (result2.pre_indices * max_post * max_comp * max_comp +
                           result2.post_indices * max_comp * max_comp +
                           pre_comp2 * max_comp +
                           post_comp2)
        else:
            conn_codes1 = result1.pre_indices * max_post + result1.post_indices
            conn_codes2 = result2.pre_indices * max_post + result2.post_indices

        # Find difference using array operations
        mask = ~np.isin(conn_codes1, conn_codes2)
        diff_codes = conn_codes1[mask]

        if diff_codes.size == 0:
            result_kwargs = {
                'pre_size': result1.pre_size,
                'post_size': result1.post_size,
                'model_type': result1.model_type
            }
            if is_multicompartment:
                result_kwargs['pre_compartments'] = np.array([], dtype=np.int64)
                result_kwargs['post_compartments'] = np.array([], dtype=np.int64)
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                **result_kwargs
            )

        # Decode back to (pre, post) pairs and compartments
        if is_multicompartment:
            max_comp = 10
            pre_indices = diff_codes // (max_post * max_comp * max_comp)
            remainder = diff_codes % (max_post * max_comp * max_comp)
            post_indices = remainder // (max_comp * max_comp)
            remainder2 = remainder % (max_comp * max_comp)
            pre_compartments = remainder2 // max_comp
            post_compartments = remainder2 % max_comp
        else:
            pre_indices = diff_codes // max_post
            post_indices = diff_codes % max_post

        # Handle weights using array operations
        weights = None
        if result1.weights is not None:
            weights_vals, weights_unit = u.split_mantissa_unit(result1.weights)

            if weights_vals is None:
                weights_array = np.ones(len(result1.pre_indices))
            elif u.math.isscalar(weights_vals):
                weights_array = np.full(len(result1.pre_indices), weights_vals)
            else:
                weights_array = np.asarray(weights_vals)

            # Extract weights for remaining connections
            final_weights = weights_array[mask]

            if weights_unit is not None:
                weights = u.maybe_decimal(final_weights * weights_unit)
            else:
                weights = final_weights

        # Handle delays using array operations
        delays = None
        if result1.delays is not None:
            delays_vals, delays_unit = u.split_mantissa_unit(result1.delays)

            if delays_vals is None:
                delays_array = np.zeros(len(result1.pre_indices))
            elif u.math.isscalar(delays_vals):
                delays_array = np.full(len(result1.pre_indices), delays_vals)
            else:
                delays_array = np.asarray(delays_vals)

            # Extract delays for remaining connections
            final_delays = delays_array[mask]

            if delays_unit is not None:
                delays = u.maybe_decimal(final_delays * delays_unit)
            else:
                delays = u.maybe_decimal(final_delays * u.ms)

        result_kwargs = {
            'pre_size': result1.pre_size,
            'post_size': result1.post_size,
            'weights': weights,
            'delays': delays,
            'model_type': result1.model_type,
            'pre_positions': result1.pre_positions,
            'post_positions': result1.post_positions,
            'metadata': {
                **result1.metadata,
                'operation': 'difference'
            },
        }

        if is_multicompartment:
            # Extract compartment info for remaining connections
            result_kwargs['pre_compartments'] = pre_compartments.astype(np.int64)
            result_kwargs['post_compartments'] = post_compartments.astype(np.int64)

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            **result_kwargs
        )


class ScaledConnectivity(Connectivity):
    """Connectivity with scaled weights or delays."""

    __module__ = 'braintools.conn'

    def __init__(
        self,
        base_connectivity: Connectivity,
        weight_factor: float = None,
        delay_factor: float = None
    ):
        super().__init__(
            pre_size=base_connectivity.pre_size,
            post_size=base_connectivity.post_size
        )
        self.base_connectivity = base_connectivity
        self.weight_factor = weight_factor
        self.delay_factor = delay_factor

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate scaled connectivity."""
        result = self.base_connectivity(**kwargs)

        # Scale weights
        if result.weights is not None and self.weight_factor is not None:
            result.weights = result.weights * self.weight_factor

        # Scale delays
        if result.delays is not None and self.delay_factor is not None:
            result.delays = result.delays * self.delay_factor

        return result


def compute_csr_indices_indptr(pre_indices, post_indices, shape):
    """
    Compute CSR indices and indptr from pre_indices and post_indices.

    Parameters
    ----------
    pre_indices : np.ndarray
        Row indices of nonzero elements.
    post_indices : np.ndarray
        Column indices of nonzero elements.
    shape : tuple
        (n_rows, n_cols) of the matrix.

    Returns
    -------
    indices : np.ndarray
        Column indices for CSR.
    indptr : np.ndarray
        Row pointer for CSR.
    """
    n_rows = shape[0]
    # Sort by pre_indices for CSR format
    order = np.argsort(pre_indices)
    sorted_pre = pre_indices[order]
    sorted_post = post_indices[order]

    # indices: sorted_post
    indices = sorted_post

    # indptr: count nonzeros per row
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.add.at(indptr, sorted_pre + 1, 1)
    np.cumsum(indptr, out=indptr)

    return indices, indptr


def merge_dict(*dicts):
    res = dict()
    for d in dicts:
        res.update(d)
    return res
