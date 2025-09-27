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
Composable connectivity base classes for building synaptic networks.

This module provides the foundation for composable connectivity patterns,
allowing users to combine, transform, and constrain connectivity rules
using intuitive arithmetic operations and method chaining.
"""

from typing import Optional, Tuple, Union, Callable, Sequence

import brainunit as u
import numpy as np

__all__ = [
    'Connectivity',
    'CompositeConnectivity',
    'ConstrainedConnectivity',
    'TransformedConnectivity',
    'ScaledConnectivity',
    'FilteredConnectivity',
    'LayeredConnectivity',
    'ConnectionResult'
]


class ConnectionResult:
    """Container for connectivity results with metadata and unit support.

    Parameters
    ----------
    pre_indices : np.ndarray
        Presynaptic neuron indices.
    post_indices : np.ndarray
        Postsynaptic neuron indices.
    weights : np.ndarray or Quantity, optional
        Synaptic weights for each connection with proper units (e.g., nS, pA).
    delays : np.ndarray or Quantity, optional
        Synaptic delays for each connection with proper units (e.g., ms).
    positions : tuple of (pre_positions, post_positions), optional
        Neuron positions with proper units (e.g., µm).
    metadata : dict, optional
        Additional metadata about the connectivity.

    Examples
    --------
    Basic usage:

    .. code-block:: python

        >>> import brainunit as u
        >>> result = ConnectionResult(
        ...     pre_indices=[0, 1, 2],
        ...     post_indices=[1, 2, 0],
        ...     weights=[0.5, 1.2, 0.8] * u.nS,  # Conductance weights
        ...     delays=[1.5, 2.0, 1.8] * u.ms    # Synaptic delays
        ... )
        >>> print(f"Weights: {result.weights}")
        >>> print(f"Delays: {result.delays}")

    Distance-based weights:

    .. code-block:: python

        >>> # Weights proportional to 1/distance
        >>> distances = [50, 100, 75] * u.um
        >>> weights = (1.0 * u.nS) / (distances / (10 * u.um))
        >>> result = ConnectionResult([0, 1, 2], [1, 2, 0], weights=weights)
    """

    def __init__(
        self,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        weights: Optional[Union[np.ndarray, u.Quantity]] = None,
        delays: Optional[Union[np.ndarray, u.Quantity]] = None,
        positions: Optional[Tuple[u.Quantity, u.Quantity]] = None,
        metadata: Optional[dict] = None
    ):
        self.pre_indices = np.asarray(pre_indices, dtype=np.int64)
        self.post_indices = np.asarray(post_indices, dtype=np.int64)

        # Handle weights with units
        if weights is not None:
            if isinstance(weights, u.Quantity):
                self.weights = weights
            else:
                self.weights = u.Quantity(np.asarray(weights))
        else:
            self.weights = None

        # Handle delays with units
        if delays is not None:
            if isinstance(delays, u.Quantity):
                self.delays = delays
            else:
                # Default to milliseconds if no units provided
                self.delays = np.asarray(delays) * u.ms
        else:
            self.delays = None

        # Handle positions with units
        self.positions = positions
        self.metadata = metadata or {}

        # Validate shapes
        n_connections = len(self.pre_indices)
        if len(self.post_indices) != n_connections:
            raise ValueError("pre_indices and post_indices must have same length")
        if self.weights is not None and len(self.weights) != n_connections:
            raise ValueError("weights must have same length as indices")
        if self.delays is not None and len(self.delays) != n_connections:
            raise ValueError("delays must have same length as indices")

    @property
    def n_connections(self) -> int:
        """Number of connections."""
        return len(self.pre_indices)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the connectivity matrix (max_pre+1, max_post+1)."""
        max_pre = int(np.max(self.pre_indices)) if len(self.pre_indices) > 0 else 0
        max_post = int(np.max(self.post_indices)) if len(self.post_indices) > 0 else 0
        return max_pre + 1, max_post + 1

    def to_matrix(self, pre_size: Optional[int] = None, post_size: Optional[int] = None) -> Union[
        np.ndarray, u.Quantity]:
        """Convert to dense connectivity matrix.

        Parameters
        ----------
        pre_size : int, optional
            Size of presynaptic population. If None, inferred from indices.
        post_size : int, optional
            Size of postsynaptic population. If None, inferred from indices.

        Returns
        -------
        matrix : np.ndarray or Quantity
            Dense connectivity matrix with weights (preserving units) or 1s for connections.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> result = ConnectionResult([0, 1], [1, 0], weights=[0.5, 1.2] * u.nS)
            >>> matrix = result.to_matrix(2, 2)
            >>> print(f"Matrix with units: {matrix}")
        """
        if pre_size is None:
            pre_size = int(np.max(self.pre_indices)) + 1 if len(self.pre_indices) > 0 else 0
        if post_size is None:
            post_size = int(np.max(self.post_indices)) + 1 if len(self.post_indices) > 0 else 0

        if len(self.pre_indices) > 0:
            if self.weights is not None:
                # Preserve units if weights have them
                if isinstance(self.weights, u.Quantity):
                    matrix = u.math.zeros((pre_size, post_size)) * self.weights.unit
                    matrix = matrix.at[self.pre_indices, self.post_indices].set(self.weights)
                else:
                    matrix = np.zeros((pre_size, post_size))
                    matrix[self.pre_indices, self.post_indices] = self.weights
            else:
                matrix = np.zeros((pre_size, post_size))
                matrix[self.pre_indices, self.post_indices] = 1.0
        else:
            matrix = np.zeros((pre_size, post_size))

        return matrix

    def get_distances(self) -> Optional[u.Quantity]:
        """Calculate distances between connected neurons.

        Returns
        -------
        distances : Quantity or None
            Distances with proper units (e.g., µm) if positions are available.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> pre_pos = [[0, 0], [10, 0]] * u.um
            >>> post_pos = [[5, 0], [15, 0]] * u.um
            >>> result = ConnectionResult([0, 1], [0, 1],
            ...                          positions=(pre_pos, post_pos))
            >>> distances = result.get_distances()
            >>> print(f"Connection distances: {distances}")
        """
        if self.positions is None:
            return None

        pre_positions, post_positions = self.positions
        if len(self.pre_indices) == 0:
            return u.Quantity([]) * pre_positions.unit

        # Get positions for connected neurons
        pre_coords = pre_positions[self.pre_indices]
        post_coords = post_positions[self.post_indices]

        # Calculate Euclidean distances
        diff = pre_coords - post_coords
        distances = u.math.sqrt(u.math.sum(diff ** 2, axis=1))

        return distances


class Connectivity:
    """Base class for composable connectivity patterns.

    This is the foundation class for all connectivity patterns in the composable
    system. It provides arithmetic operations for combining connectivities and
    methods for applying transformations and constraints.

    The composable connectivity system supports:
    - Arithmetic operations (+, -, *, /, |, &)
    - Transformations (scaling, filtering, constraining)
    - Spatial and temporal parameters
    - Probabilistic and deterministic patterns
    - Hierarchical and modular structures

    Examples
    --------
    Basic arithmetic operations:

    .. code-block:: python

        >>> from braintools.conn import Random, DistanceDependent
        >>> # Combine different connectivity types
        >>> random_conn = Random(prob=0.1)
        >>> distance_conn = DistanceDependent(sigma=50.0)
        >>> combined = random_conn + distance_conn  # Union
        >>> modulated = random_conn * distance_conn  # Intersection

    Complex compositions:

    .. code-block:: python

        >>> # Multi-layer cortical connectivity
        >>> local = DistanceDependent(sigma=100.0, prob=0.3)
        >>> long_range = Random(prob=0.01).filter_distance(min_dist=500.0)
        >>> feedback = Random(prob=0.05).scale_weights(0.8)
        >>> cortical = local + long_range + feedback

    Constraint application:

    .. code-block:: python

        >>> # Apply Dale's principle and degree constraints
        >>> excitatory = Random(prob=0.15).constrain_excitatory()
        >>> constrained = excitatory.limit_degrees(max_in=50, max_out=100)
    """

    __module__ = 'braintools.conn'

    def __init__(self,
                 pre_size: Union[int, Tuple[int, ...]] = None,
                 post_size: Union[int, Tuple[int, ...]] = None,
                 seed: Optional[int] = None):
        """Initialize connectivity base class.

        Parameters
        ----------
        pre_size : int or tuple, optional
            Size of presynaptic population.
        post_size : int or tuple, optional
            Size of postsynaptic population.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.pre_size = pre_size
        self.post_size = post_size
        self.seed = seed
        self.rng = np.random if seed is None else np.random.RandomState(seed)
        self._cached_result = None

    def __call__(self,
                 pre_size: Optional[Union[int, Tuple[int, ...]]] = None,
                 post_size: Optional[Union[int, Tuple[int, ...]]] = None,
                 pre_positions: Optional[np.ndarray] = None,
                 post_positions: Optional[np.ndarray] = None,
                 recompute: bool = False) -> ConnectionResult:
        """Generate connectivity pattern.

        Parameters
        ----------
        pre_size : int or tuple, optional
            Size of presynaptic population. Overrides instance setting.
        post_size : int or tuple, optional
            Size of postsynaptic population. Overrides instance setting.
        pre_positions : np.ndarray, optional
            Positions of presynaptic neurons for spatial connectivity.
        post_positions : np.ndarray, optional
            Positions of postsynaptic neurons for spatial connectivity.
        recompute : bool
            Force recomputation even if cached result exists.

        Returns
        -------
        result : ConnectionResult
            Generated connectivity with indices, weights, and metadata.
        """
        if self._cached_result is None or recompute:
            # Use provided sizes or fall back to instance defaults
            effective_pre_size = pre_size if pre_size is not None else self.pre_size
            effective_post_size = post_size if post_size is not None else self.post_size

            self._cached_result = self._generate(
                pre_size=effective_pre_size,
                post_size=effective_post_size,
                pre_positions=pre_positions,
                post_positions=post_positions
            )
        return self._cached_result

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate the connectivity pattern. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement _generate method")

    # Arithmetic operations for combining connectivities

    def __add__(self, other):
        """Union of two connectivity patterns (logical OR).

        Parameters
        ----------
        other : Connectivity
            Another connectivity pattern to combine.

        Returns
        -------
        result : CompositeConnectivity
            The union of the two connectivity patterns.
        """
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, operator='union')
        else:
            raise TypeError("Can only add Connectivity objects")

    def __mul__(self, other):
        """Intersection/modulation of connectivity patterns.

        For two Connectivity objects: intersection (logical AND).
        For scalar: weight scaling.

        Parameters
        ----------
        other : Connectivity or float
            Another connectivity pattern or scaling factor.

        Returns
        -------
        result : CompositeConnectivity or ScaledConnectivity
            The intersection or scaled connectivity.
        """
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, operator='intersection')
        elif isinstance(other, (int, float)):
            return ScaledConnectivity(self, weight_factor=other)
        else:
            raise TypeError("Can multiply by Connectivity object or scalar")

    def __rmul__(self, other):
        """Right multiplication for scalar * Connectivity."""
        if isinstance(other, (int, float)):
            return ScaledConnectivity(self, weight_factor=other)
        else:
            raise TypeError("Can only multiply by scalar")

    def __sub__(self, other):
        """Difference of connectivity patterns (A - B = A AND NOT B).

        Parameters
        ----------
        other : Connectivity
            Connectivity pattern to subtract.

        Returns
        -------
        result : CompositeConnectivity
            The difference of the two patterns.
        """
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, operator='difference')
        else:
            raise TypeError("Can only subtract Connectivity objects")

    def __or__(self, other):
        """Overlay connectivity patterns (same as union but explicit).

        Parameters
        ----------
        other : Connectivity
            Another connectivity pattern.

        Returns
        -------
        result : CompositeConnectivity
            The overlay of the two patterns.
        """
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, operator='overlay')
        else:
            raise TypeError("Can only overlay Connectivity objects")

    def __and__(self, other):
        """Intersection of connectivity patterns (same as multiplication).

        Parameters
        ----------
        other : Connectivity
            Another connectivity pattern.

        Returns
        -------
        result : CompositeConnectivity
            The intersection of the two patterns.
        """
        if isinstance(other, Connectivity):
            return CompositeConnectivity(self, other, operator='intersection')
        else:
            raise TypeError("Can only intersect Connectivity objects")

    # Transformation and constraint methods

    def scale_weights(self, factor: float):
        """Scale all connection weights by a factor.

        Parameters
        ----------
        factor : float
            Scaling factor for weights.

        Returns
        -------
        scaled : ScaledConnectivity
            Connectivity with scaled weights.
        """
        return ScaledConnectivity(self, weight_factor=factor)

    def scale_delays(self, factor: float):
        """Scale all connection delays by a factor.

        Parameters
        ----------
        factor : float
            Scaling factor for delays.

        Returns
        -------
        scaled : ScaledConnectivity
            Connectivity with scaled delays.
        """
        return ScaledConnectivity(self, delay_factor=factor)

    def filter_distance(self,
                        min_dist: Optional[float] = None,
                        max_dist: Optional[float] = None):
        """Filter connections by distance constraints.

        Parameters
        ----------
        min_dist : float, optional
            Minimum connection distance.
        max_dist : float, optional
            Maximum connection distance.

        Returns
        -------
        filtered : FilteredConnectivity
            Distance-filtered connectivity.
        """
        return FilteredConnectivity(self,
                                    filter_type='distance',
                                    min_dist=min_dist,
                                    max_dist=max_dist)

    def filter_weights(self,
                       min_weight: Optional[float] = None,
                       max_weight: Optional[float] = None):
        """Filter connections by weight constraints.

        Parameters
        ----------
        min_weight : float, optional
            Minimum connection weight.
        max_weight : float, optional
            Maximum connection weight.

        Returns
        -------
        filtered : FilteredConnectivity
            Weight-filtered connectivity.
        """
        return FilteredConnectivity(self,
                                    filter_type='weight',
                                    min_weight=min_weight,
                                    max_weight=max_weight)

    def limit_degrees(self,
                      max_in: Optional[int] = None,
                      max_out: Optional[int] = None,
                      method: str = 'random'):
        """Limit in-degree and out-degree of connections.

        Parameters
        ----------
        max_in : int, optional
            Maximum in-degree per neuron.
        max_out : int, optional
            Maximum out-degree per neuron.
        method : str
            Method for selecting connections to keep ('random', 'strongest', 'weakest').

        Returns
        -------
        constrained : ConstrainedConnectivity
            Degree-constrained connectivity.
        """
        return ConstrainedConnectivity(self,
                                       constraint_type='degree',
                                       max_in_degree=max_in,
                                       max_out_degree=max_out,
                                       selection_method=method)

    def constrain_excitatory(self, ratio: float = 0.8):
        """Apply Dale's principle for excitatory neurons.

        Parameters
        ----------
        ratio : float
            Fraction of neurons that are excitatory.

        Returns
        -------
        constrained : ConstrainedConnectivity
            Dale's principle constrained connectivity.
        """
        return ConstrainedConnectivity(self,
                                       constraint_type='dale_excitatory',
                                       excitatory_ratio=ratio)

    def constrain_inhibitory(self, ratio: float = 0.2):
        """Apply Dale's principle for inhibitory neurons.

        Parameters
        ----------
        ratio : float
            Fraction of neurons that are inhibitory.

        Returns
        -------
        constrained : ConstrainedConnectivity
            Dale's principle constrained connectivity.
        """
        return ConstrainedConnectivity(self,
                                       constraint_type='dale_inhibitory',
                                       inhibitory_ratio=ratio)

    def add_plasticity(self, rule: str = 'stdp', **params):
        """Add synaptic plasticity rules.

        Parameters
        ----------
        rule : str
            Type of plasticity rule ('stdp', 'homeostatic', 'bcm').
        **params
            Parameters for the plasticity rule.

        Returns
        -------
        plastic : TransformedConnectivity
            Connectivity with plasticity rules.
        """
        return TransformedConnectivity(self,
                                       transform_type='plasticity',
                                       rule=rule,
                                       **params)

    def add_noise(self, noise_level: float, noise_type: str = 'gaussian'):
        """Add noise to connection weights or probabilities.

        Parameters
        ----------
        noise_level : float
            Standard deviation of noise.
        noise_type : str
            Type of noise ('gaussian', 'uniform').

        Returns
        -------
        noisy : TransformedConnectivity
            Connectivity with added noise.
        """
        return TransformedConnectivity(self,
                                       transform_type='noise',
                                       noise_level=noise_level,
                                       noise_type=noise_type)

    def apply_transform(self, func: Callable):
        """Apply custom transformation function.

        Parameters
        ----------
        func : callable
            Function that takes ConnectionResult and returns modified ConnectionResult.

        Returns
        -------
        transformed : TransformedConnectivity
            Custom transformed connectivity.
        """
        return TransformedConnectivity(self,
                                       transform_type='custom',
                                       transform_func=func)


class CompositeConnectivity(Connectivity):
    """Composite connectivity created by combining two connectivity patterns.

    Parameters
    ----------
    conn1 : Connectivity
        First connectivity pattern.
    conn2 : Connectivity
        Second connectivity pattern.
    operator : str
        Operation to apply ('union', 'intersection', 'difference', 'overlay').
    """

    __module__ = 'braintools.conn'

    def __init__(self, conn1: Connectivity, conn2: Connectivity, operator: str):
        # Take size from first connectivity if available
        super().__init__(pre_size=conn1.pre_size, post_size=conn1.post_size)
        self.conn1 = conn1
        self.conn2 = conn2
        self.operator = operator

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate composite connectivity."""
        # Generate both connectivity patterns
        result1 = self.conn1(pre_size, post_size, pre_positions, post_positions)
        result2 = self.conn2(pre_size, post_size, pre_positions, post_positions)

        # Apply the specified operation
        if self.operator == 'union':
            return self._union(result1, result2)
        elif self.operator == 'intersection':
            return self._intersection(result1, result2)
        elif self.operator == 'difference':
            return self._difference(result1, result2)
        elif self.operator == 'overlay':
            return self._overlay(result1, result2)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def _union(self, result1: ConnectionResult, result2: ConnectionResult) -> ConnectionResult:
        """Combine connections from both patterns."""
        # Create sets of connections for deduplication
        conn1_set = set(zip(result1.pre_indices, result1.post_indices))
        conn2_set = set(zip(result2.pre_indices, result2.post_indices))

        # Union of connections
        all_connections = conn1_set | conn2_set

        if not all_connections:
            return ConnectionResult(np.array([], dtype=np.int64),
                                    np.array([], dtype=np.int64))

        pre_indices, post_indices = zip(*all_connections)

        # Handle weights: prefer weights from result1, then result2, then default to 1
        weights = np.ones(len(all_connections))
        if result1.weights is not None or result2.weights is not None:
            weight_dict = {}

            # Add weights from result2 first (will be overridden by result1)
            if result2.weights is not None:
                for (p, q), w in zip(zip(result2.pre_indices, result2.post_indices), result2.weights):
                    weight_dict[(p, q)] = w

            # Add weights from result1 (overrides result2)
            if result1.weights is not None:
                for (p, q), w in zip(zip(result1.pre_indices, result1.post_indices), result1.weights):
                    weight_dict[(p, q)] = w

            weight_values = [weight_dict.get(conn, 1.0) for conn in all_connections]

            # Handle units properly
            if any(isinstance(w, u.Quantity) for w in weight_values):
                # Find a unit from existing weights
                unit = None
                for w in weight_values:
                    if isinstance(w, u.Quantity):
                        unit = w.unit
                        break

                # Convert all weights to that unit
                final_weights = []
                for w in weight_values:
                    if isinstance(w, u.Quantity):
                        final_weights.append(w)
                    else:
                        final_weights.append(w * unit)
                weights = u.math.stack(final_weights)
            else:
                weights = np.array(weight_values)

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=weights,
            metadata={'operation': 'union', 'n_conn1': len(result1.pre_indices), 'n_conn2': len(result2.pre_indices)}
        )

    def _intersection(self, result1: ConnectionResult, result2: ConnectionResult) -> ConnectionResult:
        """Keep only connections present in both patterns."""
        conn1_set = set(zip(result1.pre_indices, result1.post_indices))
        conn2_set = set(zip(result2.pre_indices, result2.post_indices))

        # Intersection of connections
        common_connections = conn1_set & conn2_set

        if not common_connections:
            return ConnectionResult(np.array([], dtype=np.int64),
                                    np.array([], dtype=np.int64))

        pre_indices, post_indices = zip(*common_connections)

        # Multiply weights from both patterns
        weights = np.ones(len(common_connections))
        if result1.weights is not None or result2.weights is not None:
            weight_dict1 = {}
            weight_dict2 = {}

            if result1.weights is not None:
                for (p, q), w in zip(zip(result1.pre_indices, result1.post_indices), result1.weights):
                    weight_dict1[(p, q)] = w

            if result2.weights is not None:
                for (p, q), w in zip(zip(result2.pre_indices, result2.post_indices), result2.weights):
                    weight_dict2[(p, q)] = w

            weights = np.array([
                weight_dict1.get(conn, 1.0) * weight_dict2.get(conn, 1.0)
                for conn in common_connections
            ])

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=weights,
            metadata={'operation': 'intersection'}
        )

    def _difference(self, result1: ConnectionResult, result2: ConnectionResult) -> ConnectionResult:
        """Keep connections from result1 that are not in result2."""
        conn1_set = set(zip(result1.pre_indices, result1.post_indices))
        conn2_set = set(zip(result2.pre_indices, result2.post_indices))

        # Difference: result1 - result2
        diff_connections = conn1_set - conn2_set

        if not diff_connections:
            return ConnectionResult(np.array([], dtype=np.int64),
                                    np.array([], dtype=np.int64))

        pre_indices, post_indices = zip(*diff_connections)

        # Keep weights from result1 for remaining connections
        weights = None
        if result1.weights is not None:
            weight_dict = dict(zip(zip(result1.pre_indices, result1.post_indices), result1.weights))
            weights = np.array([weight_dict[conn] for conn in diff_connections])

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=weights,
            metadata={'operation': 'difference'}
        )

    def _overlay(self, result1: ConnectionResult, result2: ConnectionResult) -> ConnectionResult:
        """Overlay patterns with weight addition for overlapping connections."""
        # Similar to union but add weights instead of preferring one
        conn1_set = set(zip(result1.pre_indices, result1.post_indices))
        conn2_set = set(zip(result2.pre_indices, result2.post_indices))

        all_connections = conn1_set | conn2_set

        if not all_connections:
            return ConnectionResult(np.array([], dtype=np.int64),
                                    np.array([], dtype=np.int64))

        pre_indices, post_indices = zip(*all_connections)

        # Add weights for overlapping connections
        weight_dict1 = {}
        weight_dict2 = {}

        if result1.weights is not None:
            for (p, q), w in zip(zip(result1.pre_indices, result1.post_indices), result1.weights):
                weight_dict1[(p, q)] = w
        else:
            for (p, q) in zip(result1.pre_indices, result1.post_indices):
                weight_dict1[(p, q)] = 1.0

        if result2.weights is not None:
            for (p, q), w in zip(zip(result2.pre_indices, result2.post_indices), result2.weights):
                weight_dict2[(p, q)] = w
        else:
            for (p, q) in zip(result2.pre_indices, result2.post_indices):
                weight_dict2[(p, q)] = 1.0

        weights = np.array([
            weight_dict1.get(conn, 0.0) + weight_dict2.get(conn, 0.0)
            for conn in all_connections
        ])

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=weights,
            metadata={'operation': 'overlay'}
        )


class ScaledConnectivity(Connectivity):
    """Connectivity with scaled weights or delays."""

    __module__ = 'braintools.conn'

    def __init__(self, base_connectivity: Connectivity,
                 weight_factor: float = 1.0, delay_factor: float = 1.0):
        super().__init__(pre_size=base_connectivity.pre_size,
                         post_size=base_connectivity.post_size)
        self.base_connectivity = base_connectivity
        self.weight_factor = weight_factor
        self.delay_factor = delay_factor

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate scaled connectivity."""
        result = self.base_connectivity(pre_size, post_size, pre_positions, post_positions)

        # Scale weights
        scaled_weights = None
        if result.weights is not None:
            scaled_weights = result.weights * self.weight_factor
        elif self.weight_factor != 1.0:
            scaled_weights = np.ones(len(result.pre_indices)) * self.weight_factor

        # Scale delays
        scaled_delays = None
        if result.delays is not None:
            scaled_delays = result.delays * self.delay_factor
        elif self.delay_factor != 1.0:
            scaled_delays = np.ones(len(result.pre_indices)) * self.delay_factor

        return ConnectionResult(
            result.pre_indices,
            result.post_indices,
            weights=scaled_weights,
            delays=scaled_delays,
            metadata={**result.metadata, 'weight_factor': self.weight_factor, 'delay_factor': self.delay_factor}
        )


class FilteredConnectivity(Connectivity):
    """Connectivity with distance or weight filtering."""

    __module__ = 'braintools.conn'

    def __init__(self, base_connectivity: Connectivity, filter_type: str, **filter_params):
        super().__init__(pre_size=base_connectivity.pre_size,
                         post_size=base_connectivity.post_size)
        self.base_connectivity = base_connectivity
        self.filter_type = filter_type
        self.filter_params = filter_params

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate filtered connectivity."""
        result = self.base_connectivity(pre_size, post_size, pre_positions, post_positions)

        if self.filter_type == 'distance':
            return self._filter_by_distance(result, pre_positions, post_positions)
        elif self.filter_type == 'weight':
            return self._filter_by_weight(result)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

    def _filter_by_distance(self, result: ConnectionResult,
                            pre_positions: Optional[np.ndarray],
                            post_positions: Optional[np.ndarray]) -> ConnectionResult:
        """Filter connections by distance."""
        if pre_positions is None or post_positions is None:
            # Can't filter by distance without positions
            return result

        min_dist = self.filter_params.get('min_dist')
        max_dist = self.filter_params.get('max_dist')

        # Set defaults if not provided
        if min_dist is None:
            min_dist = 0.0
        if max_dist is None:
            max_dist = np.inf

        # Calculate distances
        pre_pos = pre_positions[result.pre_indices]
        post_pos = post_positions[result.post_indices]
        distances = np.linalg.norm(pre_pos - post_pos, axis=1)

        # Apply distance filter
        mask = (distances >= min_dist) & (distances <= max_dist)

        return ConnectionResult(
            result.pre_indices[mask],
            result.post_indices[mask],
            weights=result.weights[mask] if result.weights is not None else None,
            delays=result.delays[mask] if result.delays is not None else None,
            metadata={**result.metadata, 'distance_filter': {'min': min_dist, 'max': max_dist}}
        )

    def _filter_by_weight(self, result: ConnectionResult) -> ConnectionResult:
        """Filter connections by weight."""
        if result.weights is None:
            # Can't filter by weight without weights
            return result

        min_weight = self.filter_params.get('min_weight')
        max_weight = self.filter_params.get('max_weight')

        # Set defaults if not provided
        if min_weight is None:
            min_weight = -np.inf
        if max_weight is None:
            max_weight = np.inf

        mask = (result.weights >= min_weight) & (result.weights <= max_weight)

        return ConnectionResult(
            result.pre_indices[mask],
            result.post_indices[mask],
            weights=result.weights[mask],
            delays=result.delays[mask] if result.delays is not None else None,
            metadata={**result.metadata, 'weight_filter': {'min': min_weight, 'max': max_weight}}
        )


class ConstrainedConnectivity(Connectivity):
    """Connectivity with degree or biological constraints."""

    __module__ = 'braintools.conn'

    def __init__(self, base_connectivity: Connectivity, constraint_type: str, **constraint_params):
        super().__init__(pre_size=base_connectivity.pre_size,
                         post_size=base_connectivity.post_size)
        self.base_connectivity = base_connectivity
        self.constraint_type = constraint_type
        self.constraint_params = constraint_params

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate constrained connectivity."""
        result = self.base_connectivity(pre_size, post_size, pre_positions, post_positions)

        if self.constraint_type == 'degree':
            return self._apply_degree_constraints(result)
        elif self.constraint_type.startswith('dale_'):
            return self._apply_dale_principle(result, pre_size, post_size)
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")

    def _apply_degree_constraints(self, result: ConnectionResult) -> ConnectionResult:
        """Apply in-degree and out-degree constraints."""
        max_in = self.constraint_params.get('max_in_degree')
        max_out = self.constraint_params.get('max_out_degree')
        method = self.constraint_params.get('selection_method', 'random')

        mask = np.ones(len(result.pre_indices), dtype=bool)

        # Apply out-degree constraint
        if max_out is not None:
            for pre_idx in np.unique(result.pre_indices):
                out_mask = result.pre_indices == pre_idx
                out_indices = np.where(out_mask)[0]
                if len(out_indices) > max_out:
                    # Select which connections to keep
                    if method == 'random':
                        rng = np.random.RandomState(self.base_connectivity.seed)
                        keep_indices = rng.choice(out_indices, max_out, replace=False)
                    elif method == 'strongest' and result.weights is not None:
                        keep_indices = out_indices[np.argsort(result.weights[out_indices])[-max_out:]]
                    elif method == 'weakest' and result.weights is not None:
                        keep_indices = out_indices[np.argsort(result.weights[out_indices])[:max_out]]
                    else:
                        keep_indices = out_indices[:max_out]

                    # Remove other connections
                    remove_indices = np.setdiff1d(out_indices, keep_indices)
                    mask[remove_indices] = False

        # Apply in-degree constraint
        if max_in is not None:
            for post_idx in np.unique(result.post_indices):
                in_mask = mask & (result.post_indices == post_idx)
                in_indices = np.where(in_mask)[0]
                if len(in_indices) > max_in:
                    # Select which connections to keep
                    if method == 'random':
                        rng = np.random.RandomState(self.base_connectivity.seed)
                        keep_indices = rng.choice(in_indices, max_in, replace=False)
                    elif method == 'strongest' and result.weights is not None:
                        keep_indices = in_indices[np.argsort(result.weights[in_indices])[-max_in:]]
                    elif method == 'weakest' and result.weights is not None:
                        keep_indices = in_indices[np.argsort(result.weights[in_indices])[:max_in]]
                    else:
                        keep_indices = in_indices[:max_in]

                    # Remove other connections
                    remove_indices = np.setdiff1d(in_indices, keep_indices)
                    mask[remove_indices] = False

        return ConnectionResult(
            result.pre_indices[mask],
            result.post_indices[mask],
            weights=result.weights[mask] if result.weights is not None else None,
            delays=result.delays[mask] if result.delays is not None else None,
            metadata={**result.metadata, 'degree_constraints': self.constraint_params}
        )

    def _apply_dale_principle(self, result: ConnectionResult, pre_size, post_size) -> ConnectionResult:
        """Apply Dale's principle (excitatory/inhibitory constraints)."""
        if self.constraint_type == 'dale_excitatory':
            exc_ratio = self.constraint_params.get('excitatory_ratio', 0.8)
            n_exc = int(pre_size * exc_ratio) if isinstance(pre_size, int) else int(np.prod(pre_size) * exc_ratio)

            # Keep only connections from excitatory neurons (indices < n_exc)
            mask = result.pre_indices < n_exc

            # Ensure weights are positive for excitatory connections
            weights = result.weights
            if weights is not None:
                weights = weights[mask]
                weights = np.abs(weights)  # Force positive

        elif self.constraint_type == 'dale_inhibitory':
            inh_ratio = self.constraint_params.get('inhibitory_ratio', 0.2)
            n_exc = int(pre_size * (1 - inh_ratio)) if isinstance(pre_size, int) else int(
                np.prod(pre_size) * (1 - inh_ratio))

            # Keep only connections from inhibitory neurons (indices >= n_exc)
            mask = result.pre_indices >= n_exc

            # Ensure weights are negative for inhibitory connections
            weights = result.weights
            if weights is not None:
                weights = weights[mask]
                weights = -np.abs(weights)  # Force negative

        return ConnectionResult(
            result.pre_indices[mask],
            result.post_indices[mask],
            weights=weights,
            delays=result.delays[mask] if result.delays is not None else None,
            metadata={**result.metadata, 'dale_principle': self.constraint_type}
        )


class TransformedConnectivity(Connectivity):
    """Connectivity with applied transformations (plasticity, noise, etc.)."""

    __module__ = 'braintools.conn'

    def __init__(self, base_connectivity: Connectivity, transform_type: str, **transform_params):
        super().__init__(pre_size=base_connectivity.pre_size,
                         post_size=base_connectivity.post_size)
        self.base_connectivity = base_connectivity
        self.transform_type = transform_type
        self.transform_params = transform_params

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate transformed connectivity."""
        result = self.base_connectivity(pre_size, post_size, pre_positions, post_positions)

        if self.transform_type == 'noise':
            return self._add_noise(result)
        elif self.transform_type == 'plasticity':
            return self._add_plasticity(result)
        elif self.transform_type == 'custom':
            return self._apply_custom_transform(result)
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type}")

    def _add_noise(self, result: ConnectionResult) -> ConnectionResult:
        """Add noise to weights."""
        noise_level = self.transform_params.get('noise_level', 0.1)
        noise_type = self.transform_params.get('noise_type', 'gaussian')

        if result.weights is None:
            weights = np.ones(len(result.pre_indices))
        else:
            weights = result.weights.copy()

        rng = np.random.RandomState(self.base_connectivity.seed)

        if noise_type == 'gaussian':
            noise = rng.normal(0, noise_level, len(weights))
        elif noise_type == 'uniform':
            noise = rng.uniform(-noise_level, noise_level, len(weights))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        weights += noise

        return ConnectionResult(
            result.pre_indices,
            result.post_indices,
            weights=weights,
            delays=result.delays,
            metadata={**result.metadata, 'noise': {'level': noise_level, 'type': noise_type}}
        )

    def _add_plasticity(self, result: ConnectionResult) -> ConnectionResult:
        """Add plasticity rules (placeholder for now)."""
        # This would implement STDP, homeostatic plasticity, etc.
        # For now, just add metadata
        rule = self.transform_params.get('rule', 'stdp')
        return ConnectionResult(
            result.pre_indices,
            result.post_indices,
            weights=result.weights,
            delays=result.delays,
            metadata={**result.metadata, 'plasticity': {'rule': rule, 'params': self.transform_params}}
        )

    def _apply_custom_transform(self, result: ConnectionResult) -> ConnectionResult:
        """Apply custom transformation function."""
        transform_func = self.transform_params.get('transform_func')
        if transform_func is None:
            raise ValueError("Custom transform requires 'transform_func' parameter")

        return transform_func(result)


class LayeredConnectivity(Connectivity):
    """Connectivity for multi-layer networks."""

    __module__ = 'braintools.conn'

    def __init__(self, layers: Sequence[Connectivity], layer_sizes: Sequence[int]):
        """Initialize layered connectivity.

        Parameters
        ----------
        layers : sequence of Connectivity
            Connectivity patterns for each layer.
        layer_sizes : sequence of int
            Size of each layer.
        """
        total_size = sum(layer_sizes)
        super().__init__(pre_size=total_size, post_size=total_size)
        self.layers = layers
        self.layer_sizes = layer_sizes

    def _generate(self,
                  pre_size: Union[int, Tuple[int, ...]],
                  post_size: Union[int, Tuple[int, ...]],
                  pre_positions: Optional[np.ndarray] = None,
                  post_positions: Optional[np.ndarray] = None) -> ConnectionResult:
        """Generate layered connectivity."""
        all_pre_indices = []
        all_post_indices = []
        all_weights = []

        offset = 0
        for layer_conn, layer_size in zip(self.layers, self.layer_sizes):
            # Generate connectivity for this layer
            layer_result = layer_conn(layer_size, layer_size)

            # Offset indices for global indexing
            layer_pre = layer_result.pre_indices + offset
            layer_post = layer_result.post_indices + offset

            all_pre_indices.append(layer_pre)
            all_post_indices.append(layer_post)

            if layer_result.weights is not None:
                all_weights.append(layer_result.weights)
            elif len(all_weights) > 0:  # If any previous layer had weights, add default weights
                all_weights.append(np.ones(len(layer_pre)))

            offset += layer_size

        # Combine all layers
        combined_pre = np.concatenate(all_pre_indices) if all_pre_indices else np.array([], dtype=np.int64)
        combined_post = np.concatenate(all_post_indices) if all_post_indices else np.array([], dtype=np.int64)
        combined_weights = np.concatenate(all_weights) if all_weights else None

        return ConnectionResult(
            combined_pre,
            combined_post,
            weights=combined_weights,
            metadata={'layered': True, 'layer_sizes': self.layer_sizes}
        )
