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
Modular Connectivity System for Neural Network Generation.

This module provides a comprehensive, modular system for building connectivity
patterns across different types of neural models. The system is designed with
complete decoupling between model types to ensure clean, specialized implementations.

**Supported Model Types:**
- **Point Neurons**: Single-compartment integrate-and-fire models
- **Population Rate Models**: Mean-field population dynamics
- **Multi-Compartment Models**: Detailed morphological neuron models

**Key Features:**
- **Model-Specific Modules**: Dedicated implementations for each neuron type
- **Unified Interface**: Common API across all model types via connect() function
- **Biological Realism**: Realistic parameters and constraints for each model type
- **Spatial Awareness**: Position-dependent connectivity with proper units
- **Composable Patterns**: Combine and transform connectivity patterns
- **Extensible Design**: Easy to add custom patterns for any model type

**Quick Start:**

.. code-block:: python

    import brainunit as u
    from braintools.conn import connect

    # Point neuron random connectivity
    result = connect(
        pattern='Random',
        model_type='point',
        pre_size=1000,
        post_size=1000,
        prob=0.1
    )

    # Population rate E-I dynamics
    result = connect(
        pattern='ExcitatoryInhibitory',
        model_type='population_rate',
        pre_size=2,
        post_size=2,
        exc_self_coupling=0.5,
        inh_to_exc_coupling=-1.2
    )

    # Multi-compartment axon-to-dendrite connectivity
    result = connect(
        pattern='AxonToDendrite',
        model_type='multi_compartment',
        pre_size=100,
        post_size=100,
        connection_prob=0.1
    )

**Model-Specific Usage:**

Point Neuron Connectivity:

.. code-block:: python

    import numpy as np
    import brainunit as u
    from braintools.conn import point

    # Realistic synaptic connectivity with proper units
    ampa_conn = point.Random(
        prob=0.05,
        weight='lognormal',
        weight_params={'mean': 1.0 * u.nS, 'sigma': 0.5},
        delay='normal',
        delay_params={'mean': 1.5 * u.ms, 'std': 0.3 * u.ms}
    )

    # Spatial connectivity
    positions = np.random.uniform(0, 1000, (500, 2)) * u.um
    spatial_conn = point.DistanceDependent(
        sigma=100 * u.um,
        decay='gaussian',
        max_prob=0.3
    )
    result = spatial_conn(500, 500, positions, positions)

    # E-I network with Dale's principle
    ei_network = point.ExcitatoryInhibitory(
        exc_ratio=0.8,
        exc_prob=0.1,
        inh_prob=0.2,
        exc_weight=1.0 * u.nS,
        inh_weight=-0.8 * u.nS
    )

Population Rate Model Connectivity:

.. code-block:: python

    from braintools.conn import population

    # Explicit coupling matrix
    coupling_matrix = np.array([
        [0.5, 0.8],   # Excitatory -> [E, I]
        [-1.2, -0.3]  # Inhibitory -> [E, I]
    ])
    pop_coupling = population.PopulationCoupling(coupling_matrix)

    # Hierarchical visual processing
    visual_hierarchy = population.HierarchicalPopulations(
        hierarchy_levels=[4, 2, 1],  # V1, V2, IT
        feedforward_strength=[0.6, 0.8],
        feedback_strength=[0.1, 0.2],
        lateral_strength=0.05
    )

    # Wilson-Cowan dynamics
    wc_network = population.WilsonCowanNetwork(
        w_ee=1.25, w_ei=1.0,
        w_ie=-1.0, w_ii=-0.75,
        tau_e=10 * u.ms, tau_i=20 * u.ms
    )

Multi-Compartment Model Connectivity:

.. code-block:: python

    from braintools.conn import compartment

    # Axon-to-dendrite synapses
    axon_dend = compartment.AxonToDendrite(
        connection_prob=0.1,
        weight_distribution='lognormal',
        weight_params={'mean': 2.0 * u.nS, 'sigma': 0.5}
    )

    # Specific compartment targeting
    soma_targeting = compartment.CompartmentSpecific(
        compartment_mapping={
            compartment.AXON: compartment.SOMA,
            compartment.BASAL_DENDRITE: compartment.SOMA
        },
        connection_prob=0.15
    )

    # Morphology-aware connectivity
    morph_conn = compartment.MorphologyDistance(
        sigma=50 * u.um,
        decay_function='gaussian',
        compartment_mapping={
            compartment.AXON: [
                compartment.BASAL_DENDRITE,
                compartment.APICAL_DENDRITE
            ]
        }
    )

Constraint Application:
----------------------

.. code-block:: python

    # Apply Dale's principle
    excitatory = Random(prob=0.15).constrain_excitatory()
    inhibitory = Random(prob=0.05).constrain_inhibitory()

    # Degree constraints
    constrained = Random(prob=0.2).limit_degrees(max_in=50, max_out=100)

    # Distance filtering
    local_only = Random(prob=0.3).filter_distance(max_dist=150.0)

    # Weight filtering and noise
    clean_strong = (Random(prob=0.1)
                   .add_noise(0.1)
                   .filter_weights(min_weight=0.5))

Modular and Hierarchical Networks:
---------------------------------

.. code-block:: python

    # Three-module network
    modules = [100, 80, 120]  # Module sizes
    within = Random(prob=0.3)  # Strong within-module
    between = Random(prob=0.05)  # Weak between-module

    modular_net = Modular(modules, within, between)

    # Hierarchical network with multiple scales
    micro = DistanceDependent(sigma=25.0)   # Microscale
    meso = DistanceDependent(sigma=100.0)   # Mesoscale
    macro = Random(prob=0.01)               # Macroscale

    hierarchical = micro + meso.scale_weights(0.5) + macro.scale_weights(0.1)

Custom Patterns:
---------------

.. code-block:: python

    from braintools.conn import Custom

    def my_connectivity(pre_size, post_size, pre_pos, post_pos):
        # Custom connectivity logic here
        pre_indices = [...]
        post_indices = [...]
        weights = [...]
        return ConnectionResult(pre_indices, post_indices, weights=weights)

    custom_conn = Custom(my_connectivity)

Available Patterns:
------------------

**Basic Patterns:**
- **Random**: Random connectivity with fixed probability
- **AllToAll**: Fully connected networks
- **OneToOne**: Direct one-to-one mappings
- **Custom**: User-defined connectivity patterns

**Spatial Patterns:**
- **DistanceDependent**: Spatial connectivity with various decay functions
- **Gaussian**: Gaussian-weighted connectivity
- **Regular**: Regular patterns (ring, grid, lattice)
- **CompartmentDistanceDependent**: Distance-dependent connectivity between compartments
- **DendriticTree**: Dendritic tree connectivity patterns
- **AxonalProjection**: Axonal projection patterns

**Topological Patterns:**
- **SmallWorld**: Watts-Strogatz small-world networks
- **ScaleFree**: Barabási-Albert scale-free networks
- **Modular**: Multi-module networks
- **Hierarchical**: Multi-scale hierarchical networks

**Multi-Compartment Patterns:**
- **CompartmentSpecific**: Target specific compartments
- **SomaToDendrite**: Soma-to-dendrite connections
- **AxonToSoma**: Axon-to-soma connections
- **DendriteToSoma**: Dendrite-to-soma connections
- **MultiCompartmentRandom**: Random with compartment-specific probabilities

**Population Rate Patterns:**
- **PopulationCoupling**: Direct population-to-population coupling
- **MeanField**: Mean-field connectivity approximations
- **HierarchicalPopulations**: Multi-level population hierarchies
- **ExcitatoryInhibitory**: Standard E-I network patterns
- **FeedforwardInhibition**: Feedforward inhibition patterns
- **RecurrentAmplification**: Recurrent amplification networks

Operations and Transformations:
------------------------------
- **Arithmetic**: +, -, *, / for combining patterns
- **Scaling**: .scale_weights(), .scale_delays()
- **Filtering**: .filter_distance(), .filter_weights()
- **Constraints**: .limit_degrees(), .constrain_excitatory(), .constrain_inhibitory()
- **Transformations**: .add_noise(), .add_plasticity(), .apply_transform()
"""

from ._conn_base import *
from ._conn_base import __all__ as base_all
from ._conn_compartment import *
from ._conn_compartment import __all__ as comp_all
from ._conn_kernel import *
from ._conn_kernel import __all__ as kernel_all
from ._conn_point import *
from ._conn_point import __all__ as point_all
from ._conn_population import *
from ._conn_population import __all__ as pop_all
from ._init_base import *
from ._init_base import __all__ as init_all
from ._init_delay import *
from ._init_delay import __all__ as delay_all
from ._init_distance import *
from ._init_distance import __all__ as distance_all
from ._init_weight import *
from ._init_weight import __all__ as weight_all

__all__ = base_all + comp_all + init_all + weight_all + delay_all + distance_all + kernel_all + point_all + pop_all
del init_all, weight_all, delay_all, distance_all, base_all, comp_all, kernel_all, point_all, pop_all
