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
Connectivity module for building synaptic connections between neuron populations.

This module provides various connectivity patterns commonly used in neural network simulations:

- Random connectivity patterns (_random.py)
- Regular/structured patterns (_regular.py)
- Distance-based patterns (_distance.py)
- Complex network topologies (_complex.py)
- Hierarchical and modular networks (_hierarchical.py)
- Convolutional and kernel-based patterns (_kernel.py)
- Statistical graph models (_statistical.py)
- IO functions for various formats (_io.py)

Key Features:
- Support for multiple connectivity paradigms
- Efficient sparse representation using index arrays
- Compatible with various neuromorphic formats (SONATA, NWB, HDF5, etc.)
- Statistical and data-driven connectivity models
- Convolutional connectivity for vision models
- Hierarchical patterns for cortical models
"""

# Import from submodules
from ._complex import *
from ._complex import __all__ as _complex_all
from ._distance import *
from ._distance import __all__ as _distance_all
from ._hierarchical import *
from ._hierarchical import __all__ as _hierarchical_all
from ._io import *
from ._io import __all__ as _io_all
from ._kernel import *
from ._kernel import __all__ as _kernel_all
from ._random import *
from ._random import __all__ as _random_all
from ._regular import *
from ._regular import __all__ as _regular_all
from ._statistical import *
from ._statistical import __all__ as _statistical_all

# Combine all exports
__all__ = (_random_all + _regular_all + _distance_all + _complex_all +
           _hierarchical_all + _kernel_all + _statistical_all + _io_all)

# Clean up namespace
del (_random_all, _regular_all, _distance_all, _complex_all,
     _hierarchical_all, _kernel_all, _statistical_all, _io_all)
