# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


from . import analysis
from . import cogtask
from . import conn
from . import file
from . import fit
from . import init
from . import input
from . import metric
from . import optim
from . import quad
from . import spike
from . import surrogate
from . import trainer
from . import tree
from . import visualize
from ._deprecation import deprecated_class as _deprecated_class
from ._version import __version__, __version_info__

# Encoders moved to ``braintools.spike`` in 0.4.0. The top-level names are
# deprecated aliases that warn on instantiation (never on import/access).
LatencyEncoder = _deprecated_class(
    spike.LatencyEncoder, 'braintools.LatencyEncoder', 'braintools.spike.LatencyEncoder')
RateEncoder = _deprecated_class(
    spike.RateEncoder, 'braintools.RateEncoder', 'braintools.spike.RateEncoder')
PoissonEncoder = _deprecated_class(
    spike.PoissonEncoder, 'braintools.PoissonEncoder', 'braintools.spike.PoissonEncoder')
PopulationEncoder = _deprecated_class(
    spike.PopulationEncoder, 'braintools.PopulationEncoder', 'braintools.spike.PopulationEncoder')
BernoulliEncoder = _deprecated_class(
    spike.BernoulliEncoder, 'braintools.BernoulliEncoder', 'braintools.spike.BernoulliEncoder')
DeltaEncoder = _deprecated_class(
    spike.DeltaEncoder, 'braintools.DeltaEncoder', 'braintools.spike.DeltaEncoder')
StepCurrentEncoder = _deprecated_class(
    spike.StepCurrentEncoder, 'braintools.StepCurrentEncoder', 'braintools.spike.StepCurrentEncoder')
SpikeCountEncoder = _deprecated_class(
    spike.SpikeCountEncoder, 'braintools.SpikeCountEncoder', 'braintools.spike.SpikeCountEncoder')
TemporalEncoder = _deprecated_class(
    spike.TemporalEncoder, 'braintools.TemporalEncoder', 'braintools.spike.TemporalEncoder')
RankOrderEncoder = _deprecated_class(
    spike.RankOrderEncoder, 'braintools.RankOrderEncoder', 'braintools.spike.RankOrderEncoder')

# The bitwise operations moved to ``braintools.spike`` (without the ``spike_``
# prefix). The old names stay as plain, non-warning aliases: brainstate's
# ``nn/_elementwise.py`` binds them at import time.
spike_bitwise_or = spike.bitwise_or
spike_bitwise_and = spike.bitwise_and
spike_bitwise_iand = spike.bitwise_iand
spike_bitwise_not = spike.bitwise_not
spike_bitwise_xor = spike.bitwise_xor
spike_bitwise_ixor = spike.bitwise_ixor
spike_bitwise = spike.bitwise

__all__ = [
    '__version__',
    '__version_info__',
    'conn',
    'input',
    'init',
    'file',
    'fit',
    'metric',
    'visualize',
    'optim',
    'trainer',
    'tree',
    'quad',
    'surrogate',
    'cogtask',
    'analysis',
    'spike',

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

    'spike_bitwise_or',
    'spike_bitwise_and',
    'spike_bitwise_iand',
    'spike_bitwise_not',
    'spike_bitwise_xor',
    'spike_bitwise_ixor',
    'spike_bitwise',
]
