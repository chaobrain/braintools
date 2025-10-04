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

"""

from ._base import *
from ._base import __all__ as base_all
from ._biological import *
from ._biological import __all__ as biological_all
from ._compartment import *
from ._compartment import __all__ as comp_all
from ._kernel import *
from ._kernel import __all__ as kernel_all
from ._random import *
from ._random import __all__ as point_all
from ._regular import *
from ._regular import __all__ as regular_all
from ._spatial import *
from ._spatial import __all__ as spatial_all
from ._topological import *
from ._topological import __all__ as topological_all

__all__ = base_all + comp_all + kernel_all + point_all + spatial_all + topological_all + biological_all + regular_all
del base_all, comp_all, kernel_all, point_all, spatial_all, topological_all, biological_all
del regular_all
