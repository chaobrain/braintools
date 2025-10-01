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
- **Multi-Compartment Models**: Detailed morphological neuron models

**Key Features:**

- **Direct Class Access**: All connectivity patterns available as classes
- **Biological Realism**: Realistic parameters and constraints for each model type
- **Spatial Awareness**: Position-dependent connectivity with proper units
- **Composable Patterns**: Combine and transform connectivity patterns
- **Extensible Design**: Easy to add custom patterns for any model type

**Quick Start:**

.. code-block:: python

    import brainunit as u
    from braintools.conn import Random, ExcitatoryInhibitory, AxonToDendrite

    # Point neuron random connectivity
    random_conn = Random(prob=0.1)
    result = random_conn(pre_size=1000, post_size=1000)

    # E-I network dynamics
    ei_conn = ExcitatoryInhibitory(
        exc_ratio=0.8,
        exc_prob=0.1,
        inh_prob=0.2,
        exc_weight=1.0 * u.nS,
        inh_weight=-0.8 * u.nS
    )
    result = ei_conn(pre_size=1000, post_size=1000)

    # Multi-compartment axon-to-dendrite connectivity
    axon_dend = AxonToDendrite(
        connection_prob=0.1,
        weight_distribution='lognormal',
        weight_params={'mean': 2.0 * u.nS, 'sigma': 0.5}
    )
    result = axon_dend(pre_size=100, post_size=100)

**Point Neuron Connectivity:**

.. code-block:: python

    import numpy as np
    import brainunit as u
    from braintools.conn import Random, DistanceDependent, ExcitatoryInhibitory

    # Realistic synaptic connectivity with proper units
    from braintools.init import LogNormal, Normal
    ampa_conn = Random(
        prob=0.05,
        weight=LogNormal(mean=1.0 * u.nS, sigma=0.5),
        delay=Normal(mean=1.5 * u.ms, std=0.3 * u.ms)
    )

    # Spatial connectivity
    positions = np.random.uniform(0, 1000, (500, 2)) * u.um
    spatial_conn = DistanceDependent(
        sigma=100 * u.um,
        decay='gaussian',
        max_prob=0.3
    )
    result = spatial_conn(500, 500, positions, positions)

    # E-I network with Dale's principle
    ei_network = ExcitatoryInhibitory(
        exc_ratio=0.8,
        exc_prob=0.1,
        inh_prob=0.2,
        exc_weight=1.0 * u.nS,
        inh_weight=-0.8 * u.nS
    )

**Multi-Compartment Model Connectivity:**

.. code-block:: python

    from braintools.conn import (
        AxonToDendrite, CompartmentSpecific, MorphologyDistance,
        AXON, SOMA, BASAL_DENDRITE, APICAL_DENDRITE
    )

    # Axon-to-dendrite synapses
    axon_dend = AxonToDendrite(
        connection_prob=0.1,
        weight_distribution='lognormal',
        weight_params={'mean': 2.0 * u.nS, 'sigma': 0.5}
    )

    # Specific compartment targeting
    soma_targeting = CompartmentSpecific(
        compartment_mapping={
            AXON: SOMA,
            BASAL_DENDRITE: SOMA
        },
        connection_prob=0.15
    )

    # Morphology-aware connectivity
    morph_conn = MorphologyDistance(
        sigma=50 * u.um,
        decay_function='gaussian',
        compartment_mapping={
            AXON: [BASAL_DENDRITE, APICAL_DENDRITE]
        }
    )


**Custom Patterns:**

.. code-block:: python

    from braintools.conn import Custom, ConnectionResult

    def my_connectivity(pre_size, post_size, pre_pos=None, post_pos=None):
        # Custom connectivity logic here
        pre_indices = [...]
        post_indices = [...]
        weights = [...]
        return ConnectionResult(pre_indices, post_indices, weights=weights)

    custom_conn = Custom(my_connectivity)
    result = custom_conn(pre_size=100, post_size=100)
"""

from ._base import *
from ._base import __all__ as base_all
from ._compartment import *
from ._compartment import __all__ as comp_all
from ._kernel import *
from ._kernel import __all__ as kernel_all
from ._point import *
from ._point import __all__ as point_all

__all__ = base_all + comp_all + kernel_all + point_all
del base_all, comp_all, kernel_all, point_all
