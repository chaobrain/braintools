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

"""Decision making cognitive task classes."""

from typing import Sequence, Tuple

import brainunit as u
import jax.numpy as jnp

from ..context import Context
from ..encoder import one_hot, circular
from ..feature import Feature
from ..phase import concat, Phase, Repeat, Fixation, Stimulus, Delay, Response, Cue
from ..task import Task
from .._typing import Duration, Data

# Default noise sigma for stimulus channels (units of sqrt(ms))
_DEFAULT_NOISE_SIGMA = 1.0 * u.ms ** 0.5

__all__ = [
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'ContextDecisionMaking',
    'SingleContextDecisionMaking',
    'PulseDecisionMaking',
]


class PerceptualDecisionMaking(Task):
    """
    Perceptual Decision Making (PDM) task.

    Two-alternative forced choice task where the agent must determine
    the direction of noisy motion evidence.

    Structure: Fixation >> Stimulus >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation period duration (default: 100ms).
    t_stimulus : Duration
        Stimulus presentation duration (default: 2000ms).
    t_response : Duration
        Response window duration (default: 100ms).
    num_choices : int
        Number of choice alternatives (default: 2).
    coherences : sequence
        Motion coherence levels (0-100) (default: (0, 6.4, 12.8, 25.6, 51.2)).
    noise_sigma : Data
        Stimulus noise standard deviation (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = PerceptualDecisionMaking(t_stimulus=1500*u.ms, num_choices=4)
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 100.0 * u.ms,
        t_stimulus: Duration = 2000.0 * u.ms,
        t_response: Duration = 100.0 * u.ms,
        num_choices: int = 2,
        coherences: Sequence[float] = (0, 6.4, 12.8, 25.6, 51.2),
        noise_sigma: Data = _DEFAULT_NOISE_SIGMA,
        pop_per_choice: int = 4,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_response = t_response
        self.num_choices = num_choices
        self.coherences = jnp.asarray(coherences, dtype=jnp.float32)
        self.noise_sigma = noise_sigma
        self.pop_per_choice = pop_per_choice
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        stim_feat = Feature(self.num_choices * self.pop_per_choice, 'stimulus')
        input_features = fix_feat + stim_feat
        choice_feat = Feature(self.num_choices, 'choice')
        output_features = fix_feat + choice_feat
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            Stimulus(
                duration=self.t_stimulus,
                name='stimulus',
                inputs={
                    'fixation': 1.0,
                    'stimulus': circular('stimulus_direction', 'coherence', base_value=0.5, max_coherence=100.0)
                },
                outputs={'label': 0},
                noise={'stimulus': self.noise_sigma}
            ),
            Response(
                duration=self.t_response,
                name='response',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['ground_truth'] = ctx.rng.choice(self.num_choices)
        ctx['coherence'] = ctx.rng.choice(self.coherences)
        directions = jnp.linspace(0, 2 * jnp.pi, self.num_choices, endpoint=False)
        ctx['stimulus_direction'] = directions[ctx['ground_truth']]


class PerceptualDecisionMakingDelayResponse(Task):
    """
    PDM task with delay before response.

    Same as PDM but with an added delay period between stimulus and response.

    Structure: Fixation >> Stimulus >> Delay >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation period duration (default: 100ms).
    t_stimulus : Duration
        Stimulus presentation duration (default: 2000ms).
    t_delay : Duration
        Delay period duration (default: 500ms).
    t_response : Duration
        Response window duration (default: 100ms).
    num_choices : int
        Number of choice alternatives (default: 2).
    coherences : sequence
        Motion coherence levels (default: (0, 6.4, 12.8, 25.6, 51.2)).
    noise_sigma : Data
        Stimulus noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = PerceptualDecisionMakingDelayResponse()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 100.0 * u.ms,
        t_stimulus: Duration = 2000.0 * u.ms,
        t_delay: Duration = 500.0 * u.ms,
        t_response: Duration = 100.0 * u.ms,
        num_choices: int = 2,
        coherences: Sequence[float] = (0, 6.4, 12.8, 25.6, 51.2),
        noise_sigma: Data = _DEFAULT_NOISE_SIGMA,
        pop_per_choice: int = 4,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_delay = t_delay
        self.t_response = t_response
        self.num_choices = num_choices
        self.coherences = jnp.asarray(coherences, dtype=jnp.float32)
        self.noise_sigma = noise_sigma
        self.pop_per_choice = pop_per_choice
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        stim_feat = Feature(self.num_choices * self.pop_per_choice, 'stimulus')
        input_features = fix_feat + stim_feat
        choice_feat = Feature(self.num_choices, 'choice')
        output_features = fix_feat + choice_feat
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            Stimulus(
                duration=self.t_stimulus,
                name='stimulus',
                inputs={
                    'fixation': 1.0,
                    'stimulus': circular('stimulus_direction', 'coherence', base_value=0.5, max_coherence=100.0)
                },
                outputs={'label': 0},
                noise={'stimulus': self.noise_sigma}
            ),
            Delay(
                duration=self.t_delay,
                name='delay',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            Response(
                duration=self.t_response,
                name='response',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['ground_truth'] = ctx.rng.choice(self.num_choices)
        ctx['coherence'] = ctx.rng.choice(self.coherences)
        directions = jnp.linspace(0, 2 * jnp.pi, self.num_choices, endpoint=False)
        ctx['stimulus_direction'] = directions[ctx['ground_truth']]


class ContextDecisionMaking(Task):
    """
    Context-Dependent Decision Making task.

    Agent receives a context cue indicating which stimulus dimension to attend.
    Two stimulus modalities are presented; context determines which is relevant.

    Structure: Fixation >> Context >> Stimulus >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 300ms).
    t_context : Duration
        Context cue duration (default: 350ms).
    t_stimulus : Duration
        Stimulus duration (default: 750ms).
    t_response : Duration
        Response duration (default: 100ms).
    num_contexts : int
        Number of context types (default: 2).
    num_choices : int
        Number of choices per context (default: 2).
    coherences : sequence
        Coherence levels (default: (6.4, 12.8, 25.6, 51.2)).
    noise_sigma : Data
        Stimulus noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = ContextDecisionMaking()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_context: Duration = 350.0 * u.ms,
        t_stimulus: Duration = 750.0 * u.ms,
        t_response: Duration = 100.0 * u.ms,
        num_contexts: int = 2,
        num_choices: int = 2,
        coherences: Sequence[float] = (6.4, 12.8, 25.6, 51.2),
        noise_sigma: Data = _DEFAULT_NOISE_SIGMA,
        pop_per_choice: int = 4,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_context = t_context
        self.t_stimulus = t_stimulus
        self.t_response = t_response
        self.num_contexts = num_contexts
        self.num_choices = num_choices
        self.coherences = jnp.asarray(coherences, dtype=jnp.float32)
        self.noise_sigma = noise_sigma
        self.pop_per_choice = pop_per_choice
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        context_feat = Feature(self.num_contexts, 'context')
        mod1_feat = Feature(self.num_choices * self.pop_per_choice, 'modality1')
        mod2_feat = Feature(self.num_choices * self.pop_per_choice, 'modality2')
        input_features = fix_feat + context_feat + mod1_feat + mod2_feat
        choice_feat = Feature(self.num_choices, 'choice')
        output_features = fix_feat + choice_feat
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            Cue(
                duration=self.t_context,
                name='context',
                inputs={
                    'fixation': 1.0,
                    'context': one_hot('context')
                },
                outputs={'label': 0}
            ),
            # Parallel modalities
            Stimulus(
                duration=self.t_stimulus,
                name='modality1',
                inputs={
                    'fixation': 1.0,
                    'modality1': circular('mod1_direction', 'mod1_coherence', base_value=0.5, max_coherence=100.0)
                },
                outputs={'label': 0},
                noise={'modality1': self.noise_sigma}
            ) | Stimulus(
                duration=self.t_stimulus,
                name='modality2',
                inputs={
                    'modality2': circular('mod2_direction', 'mod2_coherence', base_value=0.5, max_coherence=100.0)
                },
                outputs={'label': 0},
                noise={'modality2': self.noise_sigma}
            ),
            Response(
                duration=self.t_response,
                name='response',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['context'] = ctx.rng.choice(self.num_contexts)
        ctx['mod1_gt'] = ctx.rng.choice(self.num_choices)
        ctx['mod2_gt'] = ctx.rng.choice(self.num_choices)
        ctx['ground_truth'] = jnp.where(ctx['context'] == 0, ctx['mod1_gt'], ctx['mod2_gt'])
        ctx['mod1_coherence'] = ctx.rng.choice(self.coherences)
        ctx['mod2_coherence'] = ctx.rng.choice(self.coherences)
        directions = jnp.linspace(0, 2 * jnp.pi, self.num_choices, endpoint=False)
        ctx['mod1_direction'] = directions[ctx['mod1_gt']]
        ctx['mod2_direction'] = directions[ctx['mod2_gt']]


class SingleContextDecisionMaking(Task):
    """
    Single-Context Decision Making task.

    Fixed context version of ContextDecisionMaking - agent always attends
    to the same modality.

    Structure: Fixation >> Stimulus >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 300ms).
    t_stimulus : Duration
        Stimulus duration (default: 750ms).
    t_response : Duration
        Response duration (default: 100ms).
    context : int
        Which context/modality to use (0 or 1) (default: 0).
    num_choices : int
        Number of choices (default: 2).
    coherences : sequence
        Coherence levels (default: (6.4, 12.8, 25.6, 51.2)).
    noise_sigma : Data
        Stimulus noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = SingleContextDecisionMaking(context=0)
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_stimulus: Duration = 750.0 * u.ms,
        t_response: Duration = 100.0 * u.ms,
        context: int = 0,
        num_choices: int = 2,
        coherences: Sequence[float] = (6.4, 12.8, 25.6, 51.2),
        noise_sigma: Data = _DEFAULT_NOISE_SIGMA,
        pop_per_choice: int = 4,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_response = t_response
        self.context = context
        self.num_choices = num_choices
        self.coherences = jnp.asarray(coherences, dtype=jnp.float32)
        self.noise_sigma = noise_sigma
        self.pop_per_choice = pop_per_choice
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        mod1_feat = Feature(self.num_choices * self.pop_per_choice, 'modality1')
        mod2_feat = Feature(self.num_choices * self.pop_per_choice, 'modality2')
        input_features = fix_feat + mod1_feat + mod2_feat
        choice_feat = Feature(self.num_choices, 'choice')
        output_features = fix_feat + choice_feat
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            # Parallel modalities
            Stimulus(
                duration=self.t_stimulus,
                name='modality1',
                inputs={
                    'fixation': 1.0,
                    'modality1': circular('mod1_direction', 'mod1_coherence', base_value=0.5, max_coherence=100.0)
                },
                outputs={'label': 0},
                noise={'modality1': self.noise_sigma}
            ) | Stimulus(
                duration=self.t_stimulus,
                name='modality2',
                inputs={
                    'modality2': circular('mod2_direction', 'mod2_coherence', base_value=0.5, max_coherence=100.0)
                },
                outputs={'label': 0},
                noise={'modality2': self.noise_sigma}
            ),
            Response(
                duration=self.t_response,
                name='response',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['context'] = self.context
        ctx['mod1_gt'] = ctx.rng.choice(self.num_choices)
        ctx['mod2_gt'] = ctx.rng.choice(self.num_choices)
        # self.context is a Python int (constructor arg), so a Python branch is correct here.
        ctx['ground_truth'] = ctx['mod1_gt'] if self.context == 0 else ctx['mod2_gt']
        ctx['mod1_coherence'] = ctx.rng.choice(self.coherences)
        ctx['mod2_coherence'] = ctx.rng.choice(self.coherences)
        directions = jnp.linspace(0, 2 * jnp.pi, self.num_choices, endpoint=False)
        ctx['mod1_direction'] = directions[ctx['mod1_gt']]
        ctx['mod2_direction'] = directions[ctx['mod2_gt']]


class PulseDecisionMaking(Task):
    """
    Pulse-Based Decision Making task.

    Agent accumulates evidence from discrete pulses. Each pulse
    provides a small amount of evidence for one direction.

    Structure: Fixation >> (Cue >> Delay) * N >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_cue : Duration
        Duration of each cue pulse (default: 100ms).
    t_delay : Duration
        Delay between pulses (default: 240ms).
    num_pulses : int
        Number of evidence pulses (default: 7).
    t_response : Duration
        Response duration (default: 150ms).
    num_choices : int
        Number of choices (default: 2).
    pulse_values : sequence
        Possible evidence values (positive = choice 1, negative = choice 0)
        (default: (-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08)).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = PulseDecisionMaking()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_cue: Duration = 100.0 * u.ms,
        t_delay: Duration = 240.0 * u.ms,
        num_pulses: int = 7,
        t_response: Duration = 150.0 * u.ms,
        num_choices: int = 2,
        pulse_values: Sequence[float] = (-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08),
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_cue = t_cue
        self.t_delay = t_delay
        self.num_pulses = num_pulses
        self.t_response = t_response
        self.num_choices = num_choices
        self.pulse_values = jnp.asarray(pulse_values, dtype=jnp.float32)
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        cue_feat = Feature(self.num_choices, 'cue')
        input_features = fix_feat + cue_feat
        choice_feat = Feature(self.num_choices, 'choice')
        output_features = fix_feat + choice_feat
        return input_features, output_features

    def _pulse_encoder(self, ctx: Context, feature) -> jnp.ndarray:
        """Encode pulse value as differential two-channel cue. JIT/vmap safe."""
        # `repeat_index` is written by Repeat; falls back to pulse_index for compatibility.
        idx = ctx.get('repeat_index', ctx.get('pulse_index', 0))
        pulses = ctx['pulses']  # shape (num_pulses,)
        pulse_val = pulses[idx]
        # positive evidence → channel 0, negative → channel 1
        pos_channel = jnp.where(pulse_val > 0, jnp.abs(pulse_val) * 10.0, 0.0)
        neg_channel = jnp.where(pulse_val <= 0, jnp.abs(pulse_val) * 10.0, 0.0)
        out = jnp.zeros(feature.num)
        out = out.at[0].set(pos_channel)
        out = out.at[1].set(neg_channel)
        return out

    def define_phases(self) -> Phase:
        pulse_block = concat([
            Cue(
                duration=self.t_cue,
                name='pulse',
                inputs={
                    'fixation': 1.0,
                    'cue': self._pulse_encoder
                },
                outputs={'label': 0},
            ),
            Delay(
                duration=self.t_delay,
                name='pulse_delay',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
        ])

        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            Repeat(pulse_block, self.num_pulses),
            Response(
                duration=self.t_response,
                name='response',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        idxs = ctx.rng.choice(len(self.pulse_values), size=self.num_pulses)
        pulses = self.pulse_values[idxs]
        ctx['pulses'] = pulses
        total_evidence = jnp.sum(pulses)
        ctx['ground_truth'] = jnp.where(total_evidence > 0, 1, 0)
        ctx['total_evidence'] = total_evidence
