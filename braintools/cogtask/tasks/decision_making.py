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
    'MultiSensoryIntegration',
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
    .. code-block:: python

        >>> import brainunit as u
        >>> import braintools
        >>> task = braintools.cogtask.PerceptualDecisionMaking(
        ...     t_stimulus=1500 * u.ms, num_choices=4, seed=0, dt=10.0 * u.ms)
        >>> task.num_inputs, task.num_outputs
        (17, 5)
        >>> X, Y, info = task.sample_trial(0)
        >>> X.shape, Y.shape
        ((170, 17), (170,))
        >>>
        >>> X, Y = task.batch_sample(4)  # time-first: (T, batch, features)
        >>> X.shape, Y.shape
        ((170, 4, 17), (170, 4))
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
    .. code-block:: python

        >>> import brainunit as u
        >>> import braintools
        >>> task = braintools.cogtask.PerceptualDecisionMakingDelayResponse(seed=0, dt=10.0 * u.ms)
        >>> task.num_inputs, task.num_outputs
        (9, 3)
        >>> X, Y, info = task.sample_trial(0)
        >>> X.shape, Y.shape
        ((270, 9), (270,))
        >>>
        >>> X, Y = task.batch_sample(4)  # time-first: (T, batch, features)
        >>> X.shape, Y.shape
        ((270, 4, 9), (270, 4))
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
    .. code-block:: python

        >>> import brainunit as u
        >>> import braintools
        >>> task = braintools.cogtask.ContextDecisionMaking(seed=0, dt=10.0 * u.ms)
        >>> task.num_inputs, task.num_outputs
        (19, 3)
        >>> X, Y, info = task.sample_trial(0)
        >>> X.shape, Y.shape
        ((150, 19), (150,))
        >>>
        >>> X, Y = task.batch_sample(4)  # time-first: (T, batch, features)
        >>> X.shape, Y.shape
        ((150, 4, 19), (150, 4))
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
    .. code-block:: python

        >>> import brainunit as u
        >>> import braintools
        >>> task = braintools.cogtask.SingleContextDecisionMaking(
        ...     context=0, seed=0, dt=10.0 * u.ms)
        >>> task.num_inputs, task.num_outputs
        (17, 3)
        >>> X, Y, info = task.sample_trial(0)
        >>> X.shape, Y.shape
        ((115, 17), (115,))
        >>>
        >>> X, Y = task.batch_sample(4)  # time-first: (T, batch, features)
        >>> X.shape, Y.shape
        ((115, 4, 17), (115, 4))
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
    .. code-block:: python

        >>> import brainunit as u
        >>> import braintools
        >>> task = braintools.cogtask.PulseDecisionMaking(seed=0, dt=10.0 * u.ms)
        >>> task.num_inputs, task.num_outputs
        (3, 3)
        >>> X, Y, info = task.sample_trial(0)
        >>> X.shape, Y.shape
        ((303, 3), (303,))
        >>>
        >>> X, Y = task.batch_sample(4)  # time-first: (T, batch, features)
        >>> X.shape, Y.shape
        ((303, 4, 3), (303, 4))
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


class MultiSensoryIntegration(Task):
    r"""
    Multi-sensory integration task.

    Two-alternative (or ``num_choices``-alternative) forced choice in which the
    motion direction is conveyed through two sensory modalities at once, as in
    the visual/auditory rate-discrimination paradigm of Raposo et al. [1]_.
    Both modalities point toward the same (correct) direction; the trial's
    total coherence ``coh`` is split between them, so the evidence of the two
    modalities has to be combined to reach the best decision. This is the
    ``MultiSensoryIntegration`` environment of NeuroGym [2]_.

    Structure: Fixation >> Stimulus >> Decision

    Parameters
    ----------
    t_fixation : Duration
        Fixation period (default: 300 ms, as in NeuroGym).
    t_stimulus : Duration
        Stimulus period, during which both modalities are shown
        (default: 750 ms).
    t_decision : Duration
        Decision period (default: 100 ms).
    num_choices : int
        Number of choice directions, NeuroGym's ``dim_ring`` (default: 2).
        Must be at least 2.
    coherences : sequence of float
        Total coherence levels in ``[0, 100]`` from which ``coh`` is drawn
        (default: ``(5, 15, 50)``).
    noise_sigma : Data
        Standard deviation of the Gaussian input noise added to both modality
        channels during the stimulus, in units of ``ms ** 0.5``; it is scaled
        by ``1 / sqrt(dt)`` (default: ``1.0 * u.ms ** 0.5``).
    pop_per_choice : int
        Number of cosine-tuned units per choice in each modality (default: 1,
        which reproduces NeuroGym's observation space).
    p_unisensory : float
        Probability in ``[0, 1]`` that a trial is unisensory: only one of the
        two modalities (chosen with equal probability) is shown and carries the
        whole coherence, while the other channel is silent (zero input plus
        noise). The default ``0.0`` gives NeuroGym's always-multisensory
        trials.
    **kwargs
        Forwarded to :class:`~braintools.cogtask.Task` (e.g. ``seed``,
        ``dt``, ``output_mode``).

    Raises
    ------
    ValueError
        If ``num_choices < 2``, ``pop_per_choice < 1``, ``coherences`` is
        empty or ``p_unisensory`` is outside ``[0, 1]``.

    See Also
    --------
    ContextDecisionMaking : Two modalities, but a context cue selects one.
    PerceptualDecisionMaking : A single noisy motion stimulus.

    Notes
    -----
    **Inputs** (``1 + 2 * num_choices * pop_per_choice`` channels):
    ``fixation`` (1), ``modality1`` and ``modality2``. Unit ``i`` of a modality
    with preferred direction :math:`\phi_i` receives

    .. math::

        x_i = 0.5 + \frac{c_m}{200} \cos(\phi_i - \theta^*)

    where :math:`\theta^*` is the correct direction and :math:`c_m` the
    modality's coherence. Multisensory trials draw a split ``coh_prop``
    :math:`\sim U(0, 1)` and set :math:`c_1 = coh \cdot coh\_prop` and
    :math:`c_2 = coh \cdot (1 - coh\_prop)`, so :math:`c_1 + c_2 = coh`.
    The fixation input is 1 during fixation and stimulus and 0 during the
    decision period.

    **Labels** (categorical): 0 (fixate) during fixation and stimulus,
    ``ground_truth + 1`` during the decision period; ``num_classes`` is
    ``1 + num_choices``.

    **Trial state**: ``ground_truth``, ``coherence`` (total), ``coh_prop``,
    ``mod1_coherence``, ``mod2_coherence``, ``modality_condition``
    (0 = both, 1 = only modality 1, 2 = only modality 2),
    ``mod1_present``, ``mod2_present`` and ``stimulus_direction``.

    **Differences from NeuroGym.** Phase names follow this package
    (``'decision'`` is a :class:`~braintools.cogtask.Response` phase);
    ``pop_per_choice`` and ``p_unisensory`` are additions (their defaults
    reproduce NeuroGym); noise is added to the two modality channels only,
    not to the fixation channel; there is no reward or abort signal, the task
    is supervised through the labels.

    Trials are generated with JAX, so ``batch_sample`` runs under
    ``jax.jit`` and ``vmap``.

    References
    ----------
    .. [1] Raposo, D., Sheppard, J. P., Schrater, P. R., & Churchland, A. K.
       (2012). Multisensory decision-making in rats and humans. *Journal of
       Neuroscience*, 32(11), 3726-3735.
    .. [2] Molano-Mazon, M., et al. (2022). NeuroGym: An open resource for
       developing and sharing neuroscience tasks. *PsyArXiv*.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import braintools
        >>> task = braintools.cogtask.MultiSensoryIntegration(seed=0, dt=1.0 * u.ms)
        >>> task.num_inputs, task.num_outputs
        (5, 3)
        >>> X, Y, info = task.sample_trial(0)
        >>> X.shape, Y.shape
        ((1150, 5), (1150,))
        >>> state = info['trial_state']
        >>> int(Y[-1]) == int(state['ground_truth']) + 1
        True
        >>> bool(abs(state['mod1_coherence'] + state['mod2_coherence']
        ...          - state['coherence']) < 1e-4)
        True
        >>>
        >>> X, Y = task.batch_sample(4)
        >>> X.shape, Y.shape
        ((1150, 4, 5), (1150, 4))
    """

    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_stimulus: Duration = 750.0 * u.ms,
        t_decision: Duration = 100.0 * u.ms,
        num_choices: int = 2,
        coherences: Sequence[float] = (5.0, 15.0, 50.0),
        noise_sigma: Data = _DEFAULT_NOISE_SIGMA,
        pop_per_choice: int = 1,
        p_unisensory: float = 0.0,
        **kwargs
    ):
        if int(num_choices) < 2:
            raise ValueError(f"num_choices must be >= 2, got {num_choices}.")
        if int(pop_per_choice) < 1:
            raise ValueError(f"pop_per_choice must be >= 1, got {pop_per_choice}.")
        if len(coherences) == 0:
            raise ValueError("coherences must contain at least one value.")
        if not 0.0 <= float(p_unisensory) <= 1.0:
            raise ValueError(f"p_unisensory must lie in [0, 1], got {p_unisensory}.")
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_decision = t_decision
        self.num_choices = int(num_choices)
        self.coherences = jnp.asarray(coherences, dtype=jnp.float32)
        self.noise_sigma = noise_sigma
        self.pop_per_choice = int(pop_per_choice)
        self.p_unisensory = float(p_unisensory)
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        mod1_feat = Feature(self.num_choices * self.pop_per_choice, 'modality1')
        mod2_feat = Feature(self.num_choices * self.pop_per_choice, 'modality2')
        input_features = fix_feat + mod1_feat + mod2_feat
        output_features = fix_feat + Feature(self.num_choices, 'choice')
        return input_features, output_features

    @staticmethod
    def _modality_encoder(coherence_key: str, present_key: str):
        """Cosine-tuned encoder of one modality, silenced when it is absent."""
        tuned = circular('stimulus_direction', coherence_key, base_value=0.5, max_coherence=100.0)

        def encode(ctx: Context, feature) -> jnp.ndarray:
            return tuned(ctx, feature) * ctx[present_key]

        encode.__name__ = f"modality({coherence_key})"
        return encode

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
                    'modality1': self._modality_encoder('mod1_coherence', 'mod1_present'),
                    'modality2': self._modality_encoder('mod2_coherence', 'mod2_present'),
                },
                outputs={'label': 0},
                noise={'modality1': self.noise_sigma, 'modality2': self.noise_sigma}
            ),
            Response(
                duration=self.t_decision,
                name='decision',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['ground_truth'] = ctx.rng.choice(self.num_choices)
        coherence = ctx.rng.choice(self.coherences)
        ctx['coherence'] = coherence
        # Draw every random number on every trial so the stream does not
        # depend on the trial type (keeps vmap shapes and seeds stable).
        coh_prop = ctx.rng.uniform(0.0, 1.0)
        unisensory = ctx.rng.uniform(0.0, 1.0) < self.p_unisensory
        only = ctx.rng.choice(2)  # 0 -> only modality 1, 1 -> only modality 2
        condition = jnp.where(unisensory, 1 + only, 0)
        coh_prop = jnp.where(unisensory, jnp.where(only == 0, 1.0, 0.0), coh_prop)
        ctx['modality_condition'] = condition
        ctx['coh_prop'] = coh_prop
        ctx['mod1_coherence'] = coherence * coh_prop
        ctx['mod2_coherence'] = coherence * (1.0 - coh_prop)
        ctx['mod1_present'] = jnp.where(condition == 2, 0.0, 1.0)
        ctx['mod2_present'] = jnp.where(condition == 1, 0.0, 1.0)
        directions = jnp.linspace(0, 2 * jnp.pi, self.num_choices, endpoint=False)
        ctx['stimulus_direction'] = directions[ctx['ground_truth']]
