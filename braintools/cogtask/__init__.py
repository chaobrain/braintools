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
Composable Cognitive Task Framework
====================================

A modular, composable framework for constructing cognitive tasks for
neural network training and neuroscience simulations.

Quick Start
-----------

Using pre-built tasks:

>>> from braintools.cogtask import PerceptualDecisionMaking
>>> task = PerceptualDecisionMaking(t_stimulus=2000)
>>> X, Y = task.batch_sample(32)
>>> # train_step(X, Y)

Building custom tasks from phases:

>>> from braintools.cogtask import (
...     Task, Context, concat,
...     Fixation, Stimulus, Delay, Response,
...     Feature, circular, one_hot
... )
>>> import brainunit as u
>>>
>>> # Define features
>>> fix = Feature(1, 'fixation')
>>> stim = Feature(8, 'stimulus')
>>> choice = Feature(2, 'choice')
>>>
>>> # Build task
>>> task = Task(
...     phases=concat([
...         Fixation(100 * u.ms, inputs={'fixation': 1.0}),
...         Stimulus(500 * u.ms, inputs={'stimulus': circular('direction')}),
...         Delay(500 * u.ms, inputs={'fixation': 1.0}),
...         Response(100 * u.ms, outputs={'label': 'ground_truth'})
...     ]),
...     input_features=fix + stim,
...     output_features=fix + choice,
...     trial_init=lambda ctx: ctx.update(
...         ground_truth=ctx.rng.choice(2),
...         direction=ctx.rng.uniform(0, 2*3.14159)
...     )
... )

API Summary
-----------

Core:
    - Task, TaskConfig: Main task class and configuration
    - Context: Inter-phase state container
    - Phase, Sequence, Repeat, Parallel: Phase composition
    - If, Switch, While: Conditional phases
    - concat: Helper for sequential composition

Phases:
    - Fixation, Delay, Stimulus, Response, Cue: Basic phases
    - Sample, Test, Recall, Match, Comparison, Blank: Memory phases
    - DeclarativePhase: Base class for creating custom phases

Features:
    - Feature, FeatureSet, CircleFeature: Input/output encoding

Encoders:
    - circular, one_hot, von_mises, scalar, gaussian, identity, ctx_value

Labels:
    - label, match_label, comparison_label: Output label helpers

Pre-built Tasks:
    - Decision Making: PerceptualDecisionMaking, ContextDecisionMaking, etc.
    - Working Memory: DelayMatchSample, GoNoGo, etc.
    - Reasoning: HierarchicalReasoning, ProbabilisticReasoning
    - Motor: AntiReach, Reaching1D, EvidenceAccumulation
"""

from ._version import (
    __version_info__,
    __version__,
)
from .conditional import (
    If,
    Switch,
    While,
)
# Core
from .context import Context
# Encoders
from .encoder import (
    one_hot,
    circular,
    von_mises,
    scalar,
    gaussian,
    identity,
    ctx_value,
    cos_sin,
)
# Features
from .feature import (
    Feature,
    FeatureSet,
    CircleFeature,
    is_feature,
    as_feature,
)
# Labels
from .label import (
    label,
    match_label,
    comparison_label,
)
# Phases - import declarative phases from phase module
from .phase import (
    Fixation,
    Delay,
    Stimulus,
    Response,
    Cue,
    Sample,
    Test,
    Recall,
    Match,
    Comparison,
    Blank,
    DeclarativePhase,
    VariableDuration,
)
from .phase import (
    Phase,
    Sequence,
    Repeat,
    Parallel,
    concat,
    execute_phase,
    execute_phase_packed,
    phase_tree_is_variable,
)
from .task import Task
# Pre-built tasks
from .tasks import (
    # Decision Making
    PerceptualDecisionMaking,
    PerceptualDecisionMakingDelayResponse,
    ContextDecisionMaking,
    SingleContextDecisionMaking,
    PulseDecisionMaking,
    # Working Memory
    DelayMatchSample,
    DualDelayMatchSample,
    DelayComparison,
    DelayMatchCategory,
    DelayPairedAssociation,
    GoNoGo,
    IntervalDiscrimination,
    PostDecisionWager,
    ReadySetGo,
    DelayDirectionReproduction,
    ImmediateDirectionReproduction,
    DelayDirectionClassification,
    ImmediateDirectionClassification,
    # Reasoning
    HierarchicalReasoning,
    ProbabilisticReasoning,
    # Motor
    AntiReach,
    Reaching1D,
    EvidenceAccumulation,
)
# Utilities
from .utils import (
    TruncExp,
    UniformDuration,
    Transform,
    TransformIT,
    initialize,
    initialize2,
    interval_of,
    period_to_arr,
    firing_rate,
)

__all__ = [
    '__version_info__',
    '__version__',

    # Core
    'Context',
    'Phase',
    'Sequence',
    'Repeat',
    'Parallel',
    'concat',
    'execute_phase',
    'execute_phase_packed',
    'phase_tree_is_variable',
    'If',
    'Switch',
    'While',
    'Task',
    # Features
    'Feature',
    'FeatureSet',
    'CircleFeature',
    'is_feature',
    'as_feature',
    # Basic Phases
    'Fixation',
    'Delay',
    'Stimulus',
    'Response',
    'Cue',
    'DeclarativePhase',
    'VariableDuration',
    # Memory Phases
    'Sample',
    'Test',
    'Recall',
    'Match',
    'Comparison',
    'Blank',
    # Labels
    'label',
    'match_label',
    'comparison_label',
    # Encoders
    'one_hot',
    'circular',
    'von_mises',
    'scalar',
    'gaussian',
    'identity',
    'ctx_value',
    'cos_sin',
    # Decision Making Tasks
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'ContextDecisionMaking',
    'SingleContextDecisionMaking',
    'PulseDecisionMaking',
    # Working Memory Tasks
    'DelayMatchSample',
    'DualDelayMatchSample',
    'DelayComparison',
    'DelayMatchCategory',
    'DelayPairedAssociation',
    'GoNoGo',
    'IntervalDiscrimination',
    'PostDecisionWager',
    'ReadySetGo',
    'DelayDirectionReproduction',
    'ImmediateDirectionReproduction',
    'DelayDirectionClassification',
    'ImmediateDirectionClassification',
    # Reasoning Tasks
    'HierarchicalReasoning',
    'ProbabilisticReasoning',
    # Motor Tasks
    'AntiReach',
    'Reaching1D',
    'EvidenceAccumulation',
    # Utilities
    'TruncExp',
    'UniformDuration',
    'Transform',
    'TransformIT',
    'initialize',
    'initialize2',
    'interval_of',
    'period_to_arr',
    'firing_rate',
]
