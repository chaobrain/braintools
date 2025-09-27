# Copyright 2025 BrainSim Ecosystem Limited. All Rights Reserved.
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

from typing import Optional, Tuple, Union, Callable, Dict, Any
import numpy as np
import brainunit as u
from scipy.spatial.distance import cdist

from ._base import PointNeuronConnectivity, ConnectionResult
from ._initialization import Initialization


def init_call(init: Optional[Initialization], rng: np.random.Generator, n: int, **kwargs):
    """Helper to call initialization functions."""
    if init is None:
        return None
    elif isinstance(init, Initialization):
        return init(rng, n, **kwargs)
    elif isinstance(init, (float, int)):
        return init
    elif isinstance(init, (u.Quantity, np.ndarray)):
        if u.math.size(init) in [1, n]:
            return init
        else:
            raise ValueError('Quantity must be scalar or match number of connections')
    elif hasattr(init, '__array__'):
        return init
    else:
        raise TypeError(f"Initialization must be an Initialization class, scalar, or array. Got {type(init)}")
