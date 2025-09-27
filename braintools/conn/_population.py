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
Population rate model connectivity patterns.

This module provides connectivity patterns specifically designed for population
rate models, where connections represent coupling between population activities
rather than individual synapses. These patterns focus on population-level
dynamics, mean-field interactions, and hierarchical organization.

Key Features:
- Population-to-population coupling matrices
- Mean-field connectivity approximations
- Hierarchical population structures
- Rate-dependent connectivity rules
- Excitatory-inhibitory population dynamics
"""

from typing import Optional, Tuple, Union, Dict, List, Callable

import brainunit as u
import numpy as np
from scipy.spatial.distance import cdist

from ._base import PopulationRateConnectivity, ConnectionResult
from ._initialization import Initialization, Initializer
from ._common import init_call

__all__ = [
    # Basic population patterns
    'PopulationCoupling',
    'MeanField',
    'AllToAllPopulations',
    'RandomPopulations',

    # Population-specific patterns
    'ExcitatoryInhibitory',
    'FeedforwardInhibition',
    'RecurrentAmplification',
    'CompetitiveNetwork',

    # Hierarchical patterns
    'HierarchicalPopulations',
    'FeedforwardHierarchy',
    'RecurrentHierarchy',
    'LayeredNetwork',

    # Specialized patterns
    'PopulationDistance',
    'RateDependent',
    'WilsonCowanNetwork',
    'FiringRateNetworks',

    # Custom patterns
    'CustomPopulation',
]


class PopulationCoupling(PopulationRateConnectivity):
    """Direct coupling matrix between population rate models.

    This implements explicit coupling between populations where each connection
    represents the coupling strength from one population's firing rate to another's
    input current or rate equation.

    Parameters
    ----------
    coupling_matrix : np.ndarray or dict
        Coupling strengths between populations. Can be:
        - 2D array: coupling_matrix[i,j] = strength from pop i to pop j
        - Dict: {(source_pop, target_pop): strength}
    population_sizes : list, optional
        Sizes of each population for normalization.
    coupling_type : str
        Type of coupling ('additive', 'multiplicative', 'divisive').
    time_constants : list or Quantity, optional
        Time constants for each population.

    Examples
    --------
    Basic excitatory-inhibitory coupling:

    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>>
        >>> # 2 populations: E(0) and I(1)
        >>> coupling = np.array([
        ...     [0.5, 0.8],   # E -> [E, I]
        ...     [-1.2, -0.3]  # I -> [E, I]
        ... ])
        >>> conn = PopulationCoupling(coupling)
        >>> result = conn(pre_size=2, post_size=2)

    Dictionary-based coupling with units:

    .. code-block:: python

        >>> coupling_dict = {
        ...     ('exc', 'exc'): 0.5 * u.dimensionless,
        ...     ('exc', 'inh'): 0.8 * u.dimensionless,
        ...     ('inh', 'exc'): -1.2 * u.dimensionless,
        ...     ('inh', 'inh'): -0.3 * u.dimensionless
        ... }
        >>> conn = PopulationCoupling(coupling_dict)

    Realistic cortical network:

    .. code-block:: python

        >>> # 4 populations: PYR, PV, SST, VIP
        >>> cortical_coupling = np.array([
        ...     [0.3, 0.4, 0.2, 0.1],   # PYR
        ...     [-0.8, -0.2, 0.0, 0.3], # PV
        ...     [-0.4, -0.3, -0.1, 0.2], # SST
        ...     [0.0, -0.6, -0.4, 0.0]   # VIP
        ... ])
        >>> conn = PopulationCoupling(cortical_coupling)
    """

    def __init__(
        self,
        coupling_matrix: Union[np.ndarray, Dict],
        population_sizes: Optional[List[int]] = None,
        coupling_type: str = 'additive',
        time_constants: Optional[Union[List[float], u.Quantity]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.coupling_matrix = coupling_matrix
        self.population_sizes = population_sizes
        self.coupling_type = coupling_type
        self.time_constants = time_constants

    def generate(self,
                 pre_size: Union[int, Tuple[int, ...]],
                 post_size: Union[int, Tuple[int, ...]],
                 **kwargs) -> ConnectionResult:
        """Generate population coupling connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Convert coupling matrix to array format
        if isinstance(self.coupling_matrix, dict):
            max_pop = max(max(key) if isinstance(key, tuple) and isinstance(key[0], int) else 0
                          for key in self.coupling_matrix.keys()) + 1
            coupling_array = np.zeros((max_pop, max_pop))

            for key, value in self.coupling_matrix.items():
                if isinstance(key, tuple) and len(key) == 2:
                    if isinstance(key[0], int) and isinstance(key[1], int):
                        coupling_array[key[0], key[1]] = value
        else:
            coupling_array = np.asarray(self.coupling_matrix)

        # Vectorized connection generation
        rows, cols = coupling_array.shape
        max_rows = min(pre_num, rows)
        max_cols = min(post_num, cols)
        sub_matrix = coupling_array[:max_rows, :max_cols]

        # Find non-zero connections
        pre_indices, post_indices = np.where(sub_matrix != 0)
        weights = sub_matrix[pre_indices, post_indices]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                model_type='population_rate'
            )

        # Handle time constants as delays
        delays = None
        if self.time_constants is not None:
            if isinstance(self.time_constants, u.Quantity):
                # Use time constants as delays
                tau_values = self.time_constants.magnitude
                tau_unit = self.time_constants.unit
            else:
                tau_values = self.time_constants
                tau_unit = u.ms

            # Assign time constant of target population as delay
            delay_values = [tau_values[j] if j < len(tau_values) else 1.0
                            for j in post_indices]
            delays = np.array(delay_values) * tau_unit

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            weights=weights,
            delays=delays,
            pre_size=pre_size,
            post_size=post_size,
            model_type='population_rate',
            metadata={
                'pattern': 'population_coupling',
                'coupling_type': self.coupling_type,
                'coupling_matrix': coupling_array.tolist(),
                'population_sizes': self.population_sizes
            }
        )


class MeanField(PopulationRateConnectivity):
    """Mean-field connectivity for population rate models.

    This implements mean-field approximations where each population receives
    input proportional to the weighted average activity of source populations.
    Common in neural field theory and large-scale brain models.

    Parameters
    ----------
    field_strength : float or Quantity
        Overall strength of mean-field coupling.
    normalization : str
        How to normalize coupling ('none', 'source', 'target', 'sqrt').
    connectivity_fraction : float
        Fraction of populations that are connected (for sparse mean-field).
    distance_dependence : callable, optional
        Function for distance-dependent mean-field coupling.

    Examples
    --------
    Basic mean-field coupling:

    .. code-block:: python

        >>> import brainunit as u
        >>> mf = MeanField(
        ...     field_strength=0.1 * u.dimensionless,
        ...     normalization='sqrt'
        ... )
        >>> result = mf(pre_size=10, post_size=5)

    Sparse mean-field:

    .. code-block:: python

        >>> sparse_mf = MeanField(
        ...     field_strength=0.05,
        ...     connectivity_fraction=0.3,
        ...     normalization='source'
        ... )

    Distance-dependent mean-field:

    .. code-block:: python

        >>> def distance_decay(distances, sigma=100):
        ...     return np.exp(-distances**2 / (2 * sigma**2))
        >>>
        >>> spatial_mf = MeanField(
        ...     field_strength=0.2,
        ...     distance_dependence=distance_decay
        ... )
    """

    def __init__(
        self,
        field_strength: Union[float, u.Quantity] = 0.1,
        normalization: str = 'sqrt',
        connectivity_fraction: float = 1.0,
        distance_dependence: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.field_strength = field_strength
        self.normalization = normalization
        self.connectivity_fraction = connectivity_fraction
        self.distance_dependence = distance_dependence

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate mean-field connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Generate connections based on connectivity fraction
        if self.connectivity_fraction < 1.0:
            # Sparse connectivity - vectorized
            n_total_connections = pre_num * post_num
            n_actual_connections = int(n_total_connections * self.connectivity_fraction)

            random_matrix = self.rng.random((pre_num, post_num))
            threshold = np.sort(random_matrix.flatten())[n_actual_connections - 1]
            connection_mask = random_matrix <= threshold
            pre_indices, post_indices = np.where(connection_mask)
        else:
            # Dense connectivity - vectorized
            pre_indices, post_indices = np.meshgrid(np.arange(pre_num), np.arange(post_num), indexing='ij')
            pre_indices = pre_indices.flatten()
            post_indices = post_indices.flatten()

        # Calculate weights based on normalization
        weight_base = self.field_strength
        if self.normalization == 'source':
            weight_base = weight_base / pre_num
        elif self.normalization == 'target':
            weight_base = weight_base / post_num
        elif self.normalization == 'sqrt':
            weight_base = weight_base / np.sqrt(pre_num * post_num)
        elif self.normalization == 'both':
            weight_base = weight_base / (pre_num * post_num)

        weights = np.full(len(pre_indices), weight_base)

        # Apply distance dependence if provided
        if self.distance_dependence is not None and pre_positions is not None and post_positions is not None:
            distances = cdist(pre_positions[pre_indices].reshape(-1, pre_positions.shape[1]),
                            post_positions[post_indices].reshape(-1, post_positions.shape[1]))
            distance_factors = self.distance_dependence(distances.diagonal())
            weights = weights * distance_factors

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='population_rate'
            )

        return ConnectionResult(
            pre_indices.astype(np.int64),
            post_indices.astype(np.int64),
            weights=weights,
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='population_rate',
            metadata={
                'pattern': 'mean_field',
                'field_strength': self.field_strength,
                'normalization': self.normalization,
                'connectivity_fraction': self.connectivity_fraction
            }
        )


class ExcitatoryInhibitory(PopulationRateConnectivity):
    """Standard excitatory-inhibitory population network.

    Implements the canonical E-I dynamics where excitatory populations
    self-excite and drive inhibitory populations, which provide feedback
    inhibition to excitatory populations.

    Parameters
    ----------
    exc_self_coupling : float
        Excitatory self-coupling strength.
    exc_to_inh_coupling : float
        Excitatory to inhibitory coupling strength.
    inh_to_exc_coupling : float
        Inhibitory to excitatory coupling strength (typically negative).
    inh_self_coupling : float
        Inhibitory self-coupling strength (typically negative).
    exc_time_constant : float or Quantity
        Time constant for excitatory population.
    inh_time_constant : float or Quantity
        Time constant for inhibitory population.

    Examples
    --------
    Basic E-I network:

    .. code-block:: python

        >>> ei = ExcitatoryInhibitory(
        ...     exc_self_coupling=0.5,
        ...     exc_to_inh_coupling=0.8,
        ...     inh_to_exc_coupling=-1.2,
        ...     inh_self_coupling=-0.3
        ... )
        >>> result = ei(pre_size=2, post_size=2)  # 2 populations: E and I

    E-I with different time constants:

    .. code-block:: python

        >>> import brainunit as u
        >>> ei_tau = ExcitatoryInhibitory(
        ...     exc_self_coupling=0.4,
        ...     exc_to_inh_coupling=0.6,
        ...     inh_to_exc_coupling=-1.0,
        ...     inh_self_coupling=-0.2,
        ...     exc_time_constant=20 * u.ms,
        ...     inh_time_constant=10 * u.ms
        ... )
    """

    def __init__(
        self,
        exc_self_coupling: float = 0.5,
        exc_to_inh_coupling: float = 0.8,
        inh_to_exc_coupling: float = -1.2,
        inh_self_coupling: float = -0.3,
        exc_time_constant: Union[float, u.Quantity] = 20.0,
        inh_time_constant: Union[float, u.Quantity] = 10.0,
        **kwargs
    ):
        # Create the standard E-I coupling matrix
        coupling_matrix = np.array([
            [exc_self_coupling, exc_to_inh_coupling],
            [inh_to_exc_coupling, inh_self_coupling]
        ])

        # Handle time constants
        if isinstance(exc_time_constant, u.Quantity):
            time_constants = [exc_time_constant.magnitude, inh_time_constant.magnitude] * exc_time_constant.unit
        else:
            time_constants = [exc_time_constant, inh_time_constant] * u.ms

        super().__init__(**kwargs)
        self.coupling_matrix = coupling_matrix
        self.time_constants = time_constants

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate E-I network connections."""
        # Use PopulationCoupling logic
        coupling_conn = PopulationCoupling(
            self.coupling_matrix,
            time_constants=self.time_constants
        )
        result = coupling_conn.generate(**kwargs)
        result.metadata['pattern'] = 'excitatory_inhibitory'
        return result


class HierarchicalPopulations(PopulationRateConnectivity):
    """Hierarchical population connectivity with multiple levels.

    Creates connectivity between populations organized in a hierarchy,
    implementing feedforward, feedback, and lateral connections between
    different hierarchy levels.

    Parameters
    ----------
    hierarchy_levels : list
        Number of populations at each hierarchy level.
    feedforward_strength : list or float
        Coupling strengths for feedforward connections.
    feedback_strength : list or float
        Coupling strengths for feedback connections.
    lateral_strength : float
        Strength of lateral connections within levels.
    skip_connections : bool
        Whether to include skip connections across levels.

    Examples
    --------
    Visual processing hierarchy:

    .. code-block:: python

        >>> # 3 levels: V1 (4 pops), V2 (2 pops), IT (1 pop)
        >>> visual_hierarchy = HierarchicalPopulations(
        ...     hierarchy_levels=[4, 2, 1],
        ...     feedforward_strength=[0.6, 0.8],  # V1->V2, V2->IT
        ...     feedback_strength=[0.1, 0.2],     # IT->V2, V2->V1
        ...     lateral_strength=0.05
        ... )
        >>> result = visual_hierarchy(pre_size=7, post_size=7)

    Cortical hierarchy with skip connections:

    .. code-block:: python

        >>> cortical = HierarchicalPopulations(
        ...     hierarchy_levels=[8, 4, 2, 1],
        ...     feedforward_strength=0.5,  # Uniform strength
        ...     feedback_strength=0.15,
        ...     lateral_strength=0.08,
        ...     skip_connections=True
        ... )
    """

    def __init__(
        self,
        hierarchy_levels: List[int],
        feedforward_strength: Union[List[float], float] = 0.5,
        feedback_strength: Union[List[float], float] = 0.1,
        lateral_strength: float = 0.05,
        skip_connections: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hierarchy_levels = hierarchy_levels
        self.feedforward_strength = feedforward_strength
        self.feedback_strength = feedback_strength
        self.lateral_strength = lateral_strength
        self.skip_connections = skip_connections

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate hierarchical population connections."""
        total_pops = sum(self.hierarchy_levels)
        n_levels = len(self.hierarchy_levels)

        # Create level boundaries
        level_boundaries = [0]
        for level_size in self.hierarchy_levels:
            level_boundaries.append(level_boundaries[-1] + level_size)

        pre_indices = []
        post_indices = []
        weights = []

        # Normalize strength parameters
        if isinstance(self.feedforward_strength, (list, tuple)):
            ff_strengths = list(self.feedforward_strength)
        else:
            ff_strengths = [self.feedforward_strength] * (n_levels - 1)

        if isinstance(self.feedback_strength, (list, tuple)):
            fb_strengths = list(self.feedback_strength)
        else:
            fb_strengths = [self.feedback_strength] * (n_levels - 1)

        # Feedforward connections
        for level_idx in range(n_levels - 1):
            strength = ff_strengths[level_idx] if level_idx < len(ff_strengths) else 0.5

            # Connect each population in current level to each in next level
            for pre_pop in range(level_boundaries[level_idx], level_boundaries[level_idx + 1]):
                for post_pop in range(level_boundaries[level_idx + 1], level_boundaries[level_idx + 2]):
                    pre_indices.append(pre_pop)
                    post_indices.append(post_pop)
                    weights.append(strength)

        # Feedback connections
        for level_idx in range(n_levels - 1, 0, -1):
            strength = fb_strengths[level_idx - 1] if level_idx - 1 < len(fb_strengths) else 0.1

            # Connect each population in higher level to each in lower level
            for pre_pop in range(level_boundaries[level_idx], level_boundaries[level_idx + 1]):
                for post_pop in range(level_boundaries[level_idx - 1], level_boundaries[level_idx]):
                    pre_indices.append(pre_pop)
                    post_indices.append(post_pop)
                    weights.append(strength)

        # Lateral connections within levels
        if self.lateral_strength > 0:
            for level_idx in range(n_levels):
                level_size = self.hierarchy_levels[level_idx]
                if level_size > 1:
                    start_idx = level_boundaries[level_idx]
                    for i in range(level_size):
                        for j in range(level_size):
                            if i != j:  # No self-connections
                                pre_indices.append(start_idx + i)
                                post_indices.append(start_idx + j)
                                weights.append(self.lateral_strength)

        # Skip connections (optional)
        if self.skip_connections:
            for level_idx in range(n_levels - 2):
                skip_strength = ff_strengths[level_idx] * 0.3  # Weaker skip connections

                # Connect level i to level i+2
                for pre_pop in range(level_boundaries[level_idx], level_boundaries[level_idx + 1]):
                    for post_pop in range(level_boundaries[level_idx + 2], level_boundaries[level_idx + 3]):
                        pre_indices.append(pre_pop)
                        post_indices.append(post_pop)
                        weights.append(skip_strength)

        if not pre_indices:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=kwargs.get('pre_size', total_pops),
                post_size=kwargs.get('post_size', total_pops),
                model_type='population_rate'
            )

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=np.array(weights),
            pre_size=kwargs.get('pre_size', total_pops),
            post_size=kwargs.get('post_size', total_pops),
            model_type='population_rate',
            metadata={
                'pattern': 'hierarchical_populations',
                'hierarchy_levels': self.hierarchy_levels,
                'feedforward_strength': ff_strengths,
                'feedback_strength': fb_strengths,
                'lateral_strength': self.lateral_strength,
                'skip_connections': self.skip_connections
            }
        )


class WilsonCowanNetwork(PopulationRateConnectivity):
    """Wilson-Cowan population rate network.

    Implements the classic Wilson-Cowan model for excitatory-inhibitory
    population dynamics with specific parameter constraints and connectivity.

    Parameters
    ----------
    w_ee : float
        Excitatory-to-excitatory coupling strength.
    w_ei : float
        Excitatory-to-inhibitory coupling strength.
    w_ie : float
        Inhibitory-to-excitatory coupling strength.
    w_ii : float
        Inhibitory-to-inhibitory coupling strength.
    tau_e : float or Quantity
        Excitatory population time constant.
    tau_i : float or Quantity
        Inhibitory population time constant.

    Examples
    --------
    .. code-block:: python

        >>> wc = WilsonCowanNetwork(
        ...     w_ee=1.25, w_ei=1.0,
        ...     w_ie=-1.0, w_ii=-0.75,
        ...     tau_e=10 * u.ms, tau_i=20 * u.ms
        ... )
        >>> result = wc(pre_size=2, post_size=2)
    """

    def __init__(
        self,
        w_ee: float = 1.25,
        w_ei: float = 1.0,
        w_ie: float = -1.0,
        w_ii: float = -0.75,
        tau_e: Union[float, u.Quantity] = 10.0,
        tau_i: Union[float, u.Quantity] = 20.0,
        **kwargs
    ):
        coupling_matrix = np.array([[w_ee, w_ei], [w_ie, w_ii]])
        time_constants = [tau_e, tau_i]

        super().__init__(**kwargs)
        self.coupling_matrix = coupling_matrix
        self.time_constants = time_constants

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate Wilson-Cowan network connections."""
        coupling_conn = PopulationCoupling(
            self.coupling_matrix,
            time_constants=self.time_constants
        )
        result = coupling_conn.generate(**kwargs)
        result.metadata['pattern'] = 'wilson_cowan'
        return result


# Additional patterns for completeness
class AllToAllPopulations(PopulationRateConnectivity):
    """All-to-all population connectivity.

    Parameters
    ----------
    weight : Initializer, optional
        Weight initialization for all connections.
    """

    def __init__(self, weight: Optional[Initializer] = None, **kwargs):
        super().__init__(**kwargs)
        self.weight_init = weight

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate all-to-all population connectivity."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Vectorized all-to-all
        pre_indices, post_indices = np.meshgrid(np.arange(pre_num), np.arange(post_num), indexing='ij')
        pre_indices = pre_indices.flatten()
        post_indices = post_indices.flatten()
        n_connections = len(pre_indices)

        weights = init_call(self.weight_init, self.rng, n_connections)

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='population_rate',
            metadata={'pattern': 'all_to_all_populations'}
        )


class RandomPopulations(PopulationRateConnectivity):
    """Random population connectivity.

    Parameters
    ----------
    prob : float
        Connection probability between populations.
    weight : Initializer, optional
        Weight initialization.
    """

    def __init__(self, prob: float = 0.5, weight: Optional[Initializer] = None, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.weight_init = weight

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate random population connectivity."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        if isinstance(post_size, tuple):
            post_num = int(np.prod(post_size))
        else:
            post_num = post_size

        # Vectorized random connectivity
        random_matrix = self.rng.random((pre_num, post_num))
        connection_mask = random_matrix < self.prob
        pre_indices, post_indices = np.where(connection_mask)
        n_connections = len(pre_indices)

        if n_connections == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='population_rate'
            )

        weights = init_call(self.weight_init, self.rng, n_connections)

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='population_rate',
            metadata={'pattern': 'random_populations', 'prob': self.prob}
        )


class FeedforwardInhibition(PopulationRateConnectivity):
    """Feedforward inhibition pattern.

    Models feedforward inhibition where excitatory populations drive
    both downstream excitatory and inhibitory populations, with the
    inhibitory population providing rapid feedforward inhibition.

    Parameters
    ----------
    exc_to_exc : float
        Excitatory to excitatory coupling.
    exc_to_inh : float
        Excitatory to inhibitory coupling.
    inh_to_exc : float
        Inhibitory to excitatory coupling.
    """

    def __init__(
        self,
        exc_to_exc: float = 0.8,
        exc_to_inh: float = 1.2,
        inh_to_exc: float = -1.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.exc_to_exc = exc_to_exc
        self.exc_to_inh = exc_to_inh
        self.inh_to_exc = inh_to_exc

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        **kwargs
    ) -> ConnectionResult:
        """Generate feedforward inhibition connections."""
        coupling_matrix = np.array([
            [self.exc_to_exc, self.exc_to_inh],
            [0, 0],
        ])

        return PopulationCoupling(
            coupling_matrix,
            seed=self.seed
        ).generate(pre_size, post_size, **kwargs)


class RecurrentAmplification(PopulationRateConnectivity):
    """Recurrent amplification pattern.

    Models recurrent excitation that amplifies inputs through positive feedback.

    Parameters
    ----------
    self_coupling : float
        Strength of recurrent self-coupling.
    cross_coupling : float
        Strength of cross-population coupling.
    """

    def __init__(self, self_coupling: float = 1.5, cross_coupling: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.self_coupling = self_coupling
        self.cross_coupling = cross_coupling

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        **kwargs
    ) -> ConnectionResult:
        """Generate recurrent amplification connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        coupling_matrix = np.full((pre_num, pre_num), self.cross_coupling)
        np.fill_diagonal(coupling_matrix, self.self_coupling)

        return PopulationCoupling(
            coupling_matrix,
            seed=self.seed
        ).generate(pre_size, post_size, **kwargs)


class CompetitiveNetwork(PopulationRateConnectivity):
    """Competitive network pattern.

    Winner-take-all dynamics through lateral inhibition.

    Parameters
    ----------
    self_excitation : float
        Self-excitation strength.
    lateral_inhibition : float
        Lateral inhibition strength.
    """

    def __init__(self, self_excitation: float = 1.0, lateral_inhibition: float = -0.8, **kwargs):
        super().__init__(**kwargs)
        self.self_excitation = self_excitation
        self.lateral_inhibition = lateral_inhibition

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        **kwargs
    ) -> ConnectionResult:
        """Generate competitive network connections."""
        if isinstance(pre_size, tuple):
            pre_num = int(np.prod(pre_size))
        else:
            pre_num = pre_size

        coupling_matrix = np.full((pre_num, pre_num), self.lateral_inhibition)
        np.fill_diagonal(coupling_matrix, self.self_excitation)

        return PopulationCoupling(
            coupling_matrix,
            seed=self.seed
        ).generate(pre_size, post_size, **kwargs)


class FeedforwardHierarchy(PopulationRateConnectivity):
    """Feedforward hierarchy pattern.

    Strictly feedforward hierarchical connectivity without feedback.

    Parameters
    ----------
    hierarchy_levels : list
        Number of populations at each level.
    feedforward_strength : float
        Strength of feedforward connections.
    """

    def __init__(self, hierarchy_levels: List[int], feedforward_strength: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.hierarchy_levels = hierarchy_levels
        self.feedforward_strength = feedforward_strength

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate feedforward hierarchy connections."""
        return HierarchicalPopulations(
            hierarchy_levels=self.hierarchy_levels,
            feedforward_strength=self.feedforward_strength,
            feedback_strength=0.0,
            lateral_strength=0.0,
            seed=self.seed
        ).generate(**kwargs)


class RecurrentHierarchy(PopulationRateConnectivity):
    """Recurrent hierarchy pattern.

    Hierarchical connectivity with both feedforward and feedback connections.

    Parameters
    ----------
    hierarchy_levels : list
        Number of populations at each level.
    feedforward_strength : float
        Strength of feedforward connections.
    feedback_strength : float
        Strength of feedback connections.
    """

    def __init__(
        self,
        hierarchy_levels: List[int],
        feedforward_strength: float = 0.5,
        feedback_strength: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hierarchy_levels = hierarchy_levels
        self.feedforward_strength = feedforward_strength
        self.feedback_strength = feedback_strength

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate recurrent hierarchy connections."""
        return HierarchicalPopulations(
            hierarchy_levels=self.hierarchy_levels,
            feedforward_strength=self.feedforward_strength,
            feedback_strength=self.feedback_strength,
            lateral_strength=0.05,
            seed=self.seed
        ).generate(**kwargs)


class LayeredNetwork(PopulationRateConnectivity):
    """Layered network pattern.

    Alias for HierarchicalPopulations with uniform layer sizes.

    Parameters
    ----------
    n_layers : int
        Number of layers.
    populations_per_layer : int
        Number of populations in each layer.
    feedforward_strength : float
        Feedforward connection strength.
    """

    def __init__(
        self,
        n_layers: int,
        populations_per_layer: int,
        feedforward_strength: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hierarchy_levels = [populations_per_layer] * n_layers
        self.feedforward_strength = feedforward_strength

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate layered network connections."""
        return HierarchicalPopulations(
            hierarchy_levels=self.hierarchy_levels,
            feedforward_strength=self.feedforward_strength,
            feedback_strength=0.0,
            lateral_strength=0.1,
            seed=self.seed
        ).generate(**kwargs)


class PopulationDistance(PopulationRateConnectivity):
    """Distance-dependent population connectivity.

    Parameters
    ----------
    sigma : float or Quantity
        Distance scale.
    decay_function : str
        Decay function ('gaussian', 'exponential').
    weight : Initializer, optional
        Weight initialization.
    """

    def __init__(
        self,
        sigma: Union[float, u.Quantity],
        decay_function: str = 'gaussian',
        weight: Optional[Initializer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.decay_function = decay_function
        self.weight_init = weight

    def generate(
        self,
        pre_size: Union[int, Tuple[int, ...]],
        post_size: Union[int, Tuple[int, ...]],
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate distance-dependent connections."""
        if pre_positions is None or post_positions is None:
            raise ValueError("PopulationDistance requires pre_positions and post_positions")

        if isinstance(self.sigma, u.Quantity):
            sigma_val, sigma_unit = u.split_mantissa_unit(self.sigma)
            pre_pos_val = u.Quantity(pre_positions).to(sigma_unit).mantissa
            post_pos_val = u.Quantity(post_positions).to(sigma_unit).mantissa
        else:
            sigma_val = self.sigma
            pre_pos_val = pre_positions
            post_pos_val = post_positions

        distances = cdist(pre_pos_val, post_pos_val)

        if self.decay_function == 'gaussian':
            probs = np.exp(-distances ** 2 / (2 * sigma_val ** 2))
        elif self.decay_function == 'exponential':
            probs = np.exp(-distances / sigma_val)
        else:
            raise ValueError(f"Unknown decay function: {self.decay_function}")

        connection_mask = probs > 0.01
        pre_indices, post_indices = np.where(connection_mask)
        connection_probs = probs[connection_mask]

        if len(pre_indices) == 0:
            return ConnectionResult(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                pre_size=pre_size,
                post_size=post_size,
                pre_positions=pre_positions,
                post_positions=post_positions,
                model_type='population_rate'
            )

        n_connections = len(pre_indices)
        weights = init_call(self.weight_init, self.rng, n_connections)
        if weights is not None:
            weights = weights * connection_probs
        else:
            weights = connection_probs

        return ConnectionResult(
            pre_indices,
            post_indices,
            weights=weights,
            pre_size=pre_size,
            post_size=post_size,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='population_rate',
            metadata={'pattern': 'population_distance', 'sigma': self.sigma}
        )


class RateDependent(PopulationRateConnectivity):
    """Rate-dependent connectivity.

    Wrapper that adds rate-dependent metadata.

    Parameters
    ----------
    base_pattern : PopulationRateConnectivity
        Base connectivity pattern.
    rate_function : callable
        Function (rate) -> coupling_strength.
    """

    def __init__(
        self,
        base_pattern: PopulationRateConnectivity,
        rate_function: Callable,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_pattern = base_pattern
        self.rate_function = rate_function

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate rate-dependent connections."""
        result = self.base_pattern.generate(**kwargs)
        result.metadata['rate_dependent'] = True
        result.metadata['rate_function'] = self.rate_function
        return result


class FiringRateNetworks(PopulationRateConnectivity):
    """Firing rate networks pattern.

    Alias for MeanField with standard parameters.

    Parameters
    ----------
    field_strength : float
        Mean-field coupling strength.
    """

    def __init__(self, field_strength: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.field_strength = field_strength

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate firing rate network connections."""
        return MeanField(
            field_strength=self.field_strength,
            normalization='sqrt',
            seed=self.seed
        ).generate(**kwargs)


class CustomPopulation(PopulationRateConnectivity):
    """Custom population connectivity."""

    def __init__(self, connection_func: Callable, **kwargs):
        super().__init__(**kwargs)
        self.connection_func = connection_func

    def generate(self, **kwargs) -> ConnectionResult:
        """Generate custom population connections."""
        pre_indices, post_indices, weights = self.connection_func(**kwargs)

        return ConnectionResult(
            np.array(pre_indices, dtype=np.int64),
            np.array(post_indices, dtype=np.int64),
            weights=np.array(weights),
            model_type='population_rate',
            metadata={'pattern': 'custom_population'}
        )
