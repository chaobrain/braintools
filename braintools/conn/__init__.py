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
Composable Connectivity Module for Synaptic Network Generation.

This module provides a powerful composable system for building complex synaptic
connectivity patterns between neuron populations. The system is inspired by the
composable input generation system and allows users to:

- Combine connectivity patterns using arithmetic operations (+, -, *, /)
- Apply transformations and constraints (scaling, filtering, degree limits)
- Build hierarchical and modular network structures
- Use probabilistic and deterministic connectivity rules
- Integrate spatial and temporal connectivity parameters

Key Features:
- **Composable Operations**: Combine patterns with +, -, *, / operators
- **Spatial Connectivity**: Distance-dependent and position-aware patterns
- **Biological Constraints**: Dale's principle, degree limits, plasticity rules
- **Modular Architecture**: Build complex networks from simple components
- **Extensible Design**: Easy to add custom connectivity patterns
- **Efficient Implementation**: Sparse representation and caching

Basic Usage:
-----------

.. code-block:: python

    import numpy as np
    from braintools.conn import Random, DistanceDependent

    # Simple random connectivity
    random_conn = Random(prob=0.1, seed=42)
    result = random_conn(pre_size=100, post_size=100)

    # Distance-dependent connectivity
    positions = np.random.uniform(0, 1000, (100, 2))
    distance_conn = DistanceDependent(sigma=100.0, decay='gaussian')
    result = distance_conn(100, 100, positions, positions)

    # Combine patterns
    local = Random(prob=0.3) * DistanceDependent(sigma=50.0)
    long_range = Random(prob=0.01).filter_distance(min_dist=200.0)
    complex_conn = local + long_range

Advanced Usage:
--------------

.. code-block:: python

    from braintools.conn import SmallWorld, Modular, AllToAll

    # Small-world network
    sw = SmallWorld(k=6, p=0.3)

    # Modular network
    within_module = Random(prob=0.4)
    between_module = Random(prob=0.05)
    modular = Modular([50, 50, 50], within_module, between_module)

    # Hierarchical cortical network
    local_circuits = DistanceDependent(sigma=100.0, max_prob=0.3)
    long_range = Random(prob=0.02).filter_distance(min_dist=500.0)
    feedback = Random(prob=0.08).constrain_inhibitory(ratio=0.2)

    cortical_network = (
        local_circuits +                           # Local connections
        long_range.scale_weights(0.5) +           # Weak long-range
        feedback.scale_weights(-0.8)              # Inhibitory feedback
    ).limit_degrees(max_in=100, max_out=50)       # Biological constraints

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
- **Random**: Random connectivity with fixed probability
- **DistanceDependent**: Spatial connectivity with various decay functions
- **SmallWorld**: Watts-Strogatz small-world networks
- **Regular**: Regular patterns (ring, grid, lattice)
- **AllToAll**: Fully connected networks
- **OneToOne**: Direct one-to-one mappings
- **Gaussian**: Gaussian-weighted connectivity
- **ScaleFree**: Barabási-Albert scale-free networks
- **Modular**: Multi-module networks
- **Hierarchical**: Multi-scale hierarchical networks
- **Custom**: User-defined connectivity patterns

Operations and Transformations:
------------------------------
- **Arithmetic**: +, -, *, / for combining patterns
- **Scaling**: .scale_weights(), .scale_delays()
- **Filtering**: .filter_distance(), .filter_weights()
- **Constraints**: .limit_degrees(), .constrain_excitatory(), .constrain_inhibitory()
- **Transformations**: .add_noise(), .add_plasticity(), .apply_transform()
"""

from ._composable_base import *
from ._composable_base import __all__ as base_all
from ._patterns_basic import *
from ._patterns_basic import __all__ as patterns_basic_all
from ._patterns_kernel import *
from ._patterns_kernel import __all__ as patterns_kernel_all
from ._patterns_spatial import *
from ._patterns_spatial import __all__ as patterns_spatial_all
from ._patterns_topological import *
from ._patterns_topological import __all__ as patterns_topological_all

# Combine all exports (composable + legacy)
__all__ = (
    base_all +
    patterns_basic_all +
    patterns_spatial_all +
    patterns_topological_all +
    patterns_kernel_all
)

# Clean up namespace
del (
    base_all,
    patterns_basic_all,
    patterns_spatial_all,
    patterns_topological_all,
    patterns_kernel_all,
)
