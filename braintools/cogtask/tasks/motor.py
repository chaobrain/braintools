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

"""Motor and sensorimotor cognitive task classes."""

from typing import Sequence, Tuple

import brainunit as u
import jax.numpy as jnp

from ..context import Context
from ..encoder import one_hot, circular
from ..feature import Feature
from ..phase import concat, Phase, Fixation, Stimulus, Delay, Response, Cue
from ..task import Task
from .._typing import Duration, Data

_DEFAULT_NOISE_SIGMA = 1.0 * u.ms ** 0.5

__all__ = [
    'AntiReach',
    'Reaching1D',
    'EvidenceAccumulation',
]


class AntiReach(Task):
    """
    Anti-Reach (Anti-Saccade) task.

    Agent must reach toward (pro) or away from (anti) a stimulus.
    Pro trials: respond toward stimulus location.
    Anti trials: respond opposite to stimulus location.

    Structure: Fixation >> Stimulus >> Delay >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_stimulus : Duration
        Stimulus duration (default: 500ms).
    t_delay : Duration
        Delay duration (default: 500ms).
    t_response : Duration
        Response duration (default: 500ms).
    num_locations : int
        Number of possible stimulus/response locations (default: 8).
    anti_prob : float
        Probability of anti trial (default: 0.5).
    noise_sigma : Data
        Stimulus noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = AntiReach()
    >>> task = AntiReach(num_locations=4, anti_prob=0.7)
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_stimulus: Duration = 500.0 * u.ms,
        t_delay: Duration = 500.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        num_locations: int = 8,
        anti_prob: float = 0.5,
        noise_sigma: Data = _DEFAULT_NOISE_SIGMA,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_delay = t_delay
        self.t_response = t_response
        self.num_locations = num_locations
        self.anti_prob = anti_prob
        self.noise_sigma = noise_sigma
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        stim_feat = Feature(self.num_locations, 'stimulus')
        rule_feat = Feature(2, 'rule')  # pro/anti
        input_features = fix_feat + stim_feat + rule_feat
        resp_feat = Feature(self.num_locations, 'response')
        output_features = fix_feat + resp_feat
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            # Fixation with rule cue in parallel
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ) | Cue(
                duration=self.t_fixation,
                name='rule_cue',
                inputs={'rule': one_hot('rule')},
                outputs={'label': 0}
            ),
            Stimulus(
                duration=self.t_stimulus,
                name='stimulus',
                inputs={
                    'fixation': 1.0,
                    'stimulus': one_hot('stimulus_class')
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
        # Sample stimulus location
        ctx['stim_loc'] = ctx.rng.choice(self.num_locations)
        ctx['stimulus_class'] = ctx['stim_loc']

        # Determine pro or anti
        ctx['is_anti'] = ctx.rng.random() < self.anti_prob
        ctx['rule'] = jnp.where(ctx['is_anti'], 1, 0)  # 0=pro, 1=anti

        anti_target = (ctx['stim_loc'] + self.num_locations // 2) % self.num_locations
        ctx['ground_truth'] = jnp.where(ctx['is_anti'], anti_target, ctx['stim_loc'])


class Reaching1D(Task):
    """
    1D Reaching task.

    Agent must reach to a target location after a delay.

    Structure: Fixation >> Target >> Delay >> Reach

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_stimulus : Duration
        Target presentation duration (default: 500ms).
    t_delay : Duration
        Delay duration (default: 500ms).
    t_response : Duration
        Movement duration (default: 500ms).
    num_targets : int
        Number of possible target locations (default: 8).
    noise_sigma : Data
        Target location noise (default: 0.5 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = Reaching1D()
    >>> task = Reaching1D(num_targets=4, t_delay=1000*u.ms)
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_stimulus: Duration = 500.0 * u.ms,
        t_delay: Duration = 500.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        num_targets: int = 8,
        noise_sigma: Data = 0.5 * u.ms ** 0.5,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_delay = t_delay
        self.t_response = t_response
        self.num_targets = num_targets
        self.noise_sigma = noise_sigma
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        target_feat = Feature(self.num_targets, 'target')
        input_features = fix_feat + target_feat
        reach_feat = Feature(self.num_targets, 'reach')
        output_features = fix_feat + reach_feat
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
                    'target': one_hot('stimulus_class')
                },
                outputs={'label': 0},
                noise={'target': self.noise_sigma}
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
        ctx['target_loc'] = ctx.rng.choice(self.num_targets)
        ctx['stimulus_class'] = ctx['target_loc']
        ctx['ground_truth'] = ctx['target_loc']


class EvidenceAccumulation(Task):
    """
    Evidence Accumulation task.

    Similar to PDM but designed for spiking networks. Agent accumulates
    noisy evidence over time and makes a decision.

    Structure: Fixation >> Evidence >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_evidence : Duration
        Evidence accumulation duration (default: 2000ms).
    t_response : Duration
        Response duration (default: 500ms).
    num_choices : int
        Number of choices (default: 2).
    coherences : sequence
        Evidence coherence levels (default: (0, 6.4, 12.8, 25.6, 51.2)).
    noise_sigma : Data
        Evidence noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = EvidenceAccumulation()
    >>> task = EvidenceAccumulation(num_choices=4, t_evidence=3000*u.ms)
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_evidence: Duration = 2000.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        num_choices: int = 2,
        coherences: Sequence[float] = (0, 6.4, 12.8, 25.6, 51.2),
        noise_sigma: Data = _DEFAULT_NOISE_SIGMA,
        pop_per_choice: int = 10,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_evidence = t_evidence
        self.t_response = t_response
        self.num_choices = num_choices
        self.coherences = jnp.asarray(coherences, dtype=jnp.float32)
        self.noise_sigma = noise_sigma
        self.pop_per_choice = pop_per_choice
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        evid_feat = Feature(self.num_choices * self.pop_per_choice, 'evidence')
        input_features = fix_feat + evid_feat
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
                duration=self.t_evidence,
                name='evidence',
                inputs={
                    'fixation': 1.0,
                    'evidence': circular('evidence_direction', 'coherence', base_value=0.5, max_coherence=100.0)
                },
                outputs={'label': 0},
                noise={'evidence': self.noise_sigma}
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
        ctx['evidence_direction'] = directions[ctx['ground_truth']]
