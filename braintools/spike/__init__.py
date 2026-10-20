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

"""Spike data: the :class:`SpikeTrains` container, encoders and bitwise operations.

- :class:`SpikeTrains` stores the spike times of a population in a padded,
  fixed-shape layout that works under ``jax.jit`` and ``jax.vmap``, and
  converts to and from dense rasters, flat event lists and ragged lists.
- The encoders (:class:`LatencyEncoder`, :class:`PoissonEncoder`, ...) turn
  numbers or time series into spike rasters.
- The bitwise functions (:func:`bitwise_or`, :func:`bitwise_and`, ...) combine
  spike tensors element-wise and stay differentiable for surrogate-gradient
  spikes.
"""

from ._bitwise import (
    bitwise_or,
    bitwise_and,
    bitwise_iand,
    bitwise_not,
    bitwise_xor,
    bitwise_ixor,
    bitwise,
)
from ._encoder import (
    LatencyEncoder,
    RateEncoder,
    PoissonEncoder,
    PopulationEncoder,
    BernoulliEncoder,
    DeltaEncoder,
    StepCurrentEncoder,
    SpikeCountEncoder,
    TemporalEncoder,
    RankOrderEncoder,
)
from ._spike_trains import SpikeTrains

__all__ = [
    # container
    'SpikeTrains',

    # encoders
    'LatencyEncoder',
    'RateEncoder',
    'PoissonEncoder',
    'PopulationEncoder',
    'BernoulliEncoder',
    'DeltaEncoder',
    'StepCurrentEncoder',
    'SpikeCountEncoder',
    'TemporalEncoder',
    'RankOrderEncoder',

    # bitwise operations
    'bitwise_or',
    'bitwise_and',
    'bitwise_iand',
    'bitwise_not',
    'bitwise_xor',
    'bitwise_ixor',
    'bitwise',
]
