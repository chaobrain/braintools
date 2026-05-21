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

"""Reasoning cognitive task classes."""

from typing import Sequence, Tuple

import brainunit as u
import jax.numpy as jnp

from ..context import Context
from ..encoder import one_hot
from ..feature import Feature
from ..phase import concat, Phase, Repeat, Fixation, Stimulus, Delay, Response, Cue, DeclarativePhase
from ..task import Task
from .._typing import Duration

__all__ = [
    'HierarchicalReasoning',
    'ProbabilisticReasoning',
]


class HierarchicalReasoning(Task):
    """
    Hierarchical Reasoning task.

    Agent must apply conditional rules:
    - If delay < threshold: Rule A (go toward flash2)
    - If delay >= threshold: Rule B (go away from flash2)

    Rules change in blocks without explicit cues.

    Structure: Fixation >> Flash1 >> Delay >> Flash2 >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_flash1 : Duration
        First flash duration (default: 100ms).
    t_delay : tuple
        (min, max) for delay duration (default: (200ms, 800ms)).
    t_flash2 : Duration
        Second flash duration (default: 100ms).
    t_response : Duration
        Response duration (default: 500ms).
    delay_threshold : float
        Threshold for rule switching in ms (default: 500.0).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = HierarchicalReasoning()
    >>> task = HierarchicalReasoning(delay_threshold=400.0)
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_flash1: Duration = 100.0 * u.ms,
        t_delay: tuple = (200.0 * u.ms, 800.0 * u.ms),
        t_flash2: Duration = 100.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        delay_threshold: float = 500.0,
        show_rule_cue: bool = True,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_flash1 = t_flash1
        self.t_delay = t_delay
        self.t_flash2 = t_flash2
        self.t_response = t_response
        self.delay_threshold = delay_threshold
        # If False, the rule is implicit (changes every 100 trials by index) and
        # the agent must infer it. If True, the rule is provided as an input
        # cue during fixation — required for learnability without feedback.
        self.show_rule_cue = show_rule_cue
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        flash_feat = Feature(2, 'flash')  # left/right
        if self.show_rule_cue:
            rule_feat = Feature(2, 'rule')
            input_features = fix_feat + flash_feat + rule_feat
        else:
            input_features = fix_feat + flash_feat
        resp_feat = Feature(2, 'response')
        output_features = fix_feat + resp_feat
        return input_features, output_features

    def _compute_variable_delay(self, ctx: Context) -> int:
        """Compute delay duration from context."""
        delay_duration = ctx.get('delay_duration', 500.0)
        dt = ctx.dt
        if hasattr(dt, 'mantissa'):
            dt_val = float(dt.mantissa)
        else:
            dt_val = float(dt)
        if dt_val == 0:
            raise ValueError("dt cannot be zero")
        return max(1, int(delay_duration / dt_val))

    def define_phases(self) -> Phase:
        # Create a custom DeclarativePhase for variable delay
        variable_delay = DeclarativePhase(
            duration=0 * u.ms,  # Will be overridden by get_duration
            name='delay',
            inputs={'fixation': 1.0},
            outputs={'label': 0}
        )
        # Override get_duration method
        variable_delay.get_duration = lambda ctx: self._compute_variable_delay(ctx)

        fixation_inputs = {'fixation': 1.0}
        if self.show_rule_cue:
            fixation_inputs['rule'] = one_hot('rule', num_classes=2)

        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs=fixation_inputs,
                outputs={'label': 0}
            ),
            Stimulus(
                duration=self.t_flash1,
                name='flash1',
                inputs={
                    'fixation': 1.0,
                    'flash': one_hot('flash1_loc')
                },
                outputs={'label': 0}
            ),
            variable_delay,
            Stimulus(
                duration=self.t_flash2,
                name='flash2',
                inputs={
                    'fixation': 1.0,
                    'flash': one_hot('flash2_loc')
                },
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
        # Convert Quantity tuple to float (in ms)
        delay_min = float(self.t_delay[0].to(u.ms).mantissa)
        delay_max = float(self.t_delay[1].to(u.ms).mantissa)

        # Trial index determines rule block (alternating every 100 trials).
        # trial_index is a Python int set by Task.sample_trial, so a Python
        # branch is correct here.
        trial_idx = int(ctx.get('trial_index', 0))
        ctx['rule'] = (trial_idx // 100) % 2  # 0 or 1

        # Sample delay
        ctx['delay_duration'] = ctx.rng.uniform(delay_min, delay_max)

        # Flash locations (left=0, right=1)
        ctx['flash1_loc'] = ctx.rng.choice(2)
        ctx['flash2_loc'] = ctx.rng.choice(2)

        # Determine correct response based on rule and delay. Both rule and
        # delay condition can be JAX values under vmap, so use jnp.where.
        short_delay = ctx['delay_duration'] < self.delay_threshold
        flash = ctx['flash2_loc']
        # Rule 0: short→toward, long→away. Rule 1: opposite.
        rule_a = jnp.where(short_delay, flash, 1 - flash)
        rule_b = jnp.where(short_delay, 1 - flash, flash)
        ctx['ground_truth'] = jnp.where(ctx['rule'] == 0, rule_a, rule_b)


class ProbabilisticReasoning(Task):
    """
    Probabilistic Reasoning task.

    Agent accumulates log-likelihood evidence from multiple cues.
    Each cue provides probabilistic evidence for one of two choices.

    Structure: Fixation >> (Cue >> Delay) * N >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_cue : Duration
        Duration of each cue (default: 100ms).
    t_delay : Duration
        Delay between cues (default: 100ms).
    num_cues : int
        Number of evidence cues (default: 8).
    t_response : Duration
        Response duration (default: 500ms).
    num_choices : int
        Number of choices (default: 2).
    cue_evidence : sequence
        Possible log-likelihood ratios (positive = choice 1)
        (default: (-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08)).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = ProbabilisticReasoning()
    >>> task = ProbabilisticReasoning(num_cues=12, t_cue=150*u.ms)
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_cue: Duration = 100.0 * u.ms,
        t_delay: Duration = 100.0 * u.ms,
        num_cues: int = 8,
        t_response: Duration = 500.0 * u.ms,
        num_choices: int = 2,
        cue_evidence: Sequence[float] = (-0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08),
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_cue = t_cue
        self.t_delay = t_delay
        self.num_cues = num_cues
        self.t_response = t_response
        self.num_choices = num_choices
        self.cue_evidence = cue_evidence
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        cue_feat = Feature(len(self.cue_evidence), 'cue')
        input_features = fix_feat + cue_feat
        resp_feat = Feature(self.num_choices, 'response')
        output_features = fix_feat + resp_feat
        return input_features, output_features

    def _cue_encoder(self, ctx: Context, feature) -> jnp.ndarray:
        """One-hot encoding of the i-th sampled cue. JIT/vmap safe."""
        # ``repeat_index`` is set by Repeat; fall back to legacy cue_index.
        idx = ctx.get('repeat_index', ctx.get('cue_index', 0))
        cue_indices = ctx['cue_indices']  # shape (num_cues,)
        cue_id = cue_indices[idx]
        result = jnp.zeros(feature.num)
        result = result.at[cue_id].set(1.0)
        return result

    def define_phases(self) -> Phase:
        cue_block = concat([
            Cue(
                duration=self.t_cue,
                name='cue',
                inputs={
                    'fixation': 1.0,
                    'cue': self._cue_encoder
                },
                outputs={'label': 0},
            ),
            Delay(
                duration=self.t_delay,
                name='cue_delay',
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
            Repeat(cue_block, self.num_cues),
            Response(
                duration=self.t_response,
                name='response',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        cue_evidence = jnp.asarray(self.cue_evidence, dtype=jnp.float32)

        # Sample cues for this trial
        ctx['cue_indices'] = ctx.rng.choice(len(cue_evidence), size=self.num_cues)
        ctx['cues'] = cue_evidence[ctx['cue_indices']]

        # Accumulate log-likelihood
        total_llr = jnp.sum(ctx['cues'])
        ctx['total_evidence'] = total_llr

        # Decision based on accumulated evidence
        ctx['ground_truth'] = jnp.where(total_llr > 0, 1, 0)
