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

"""Working memory cognitive task classes."""

from typing import Sequence, Tuple, Optional, Callable, Any

import brainunit as u
import jax.numpy as jnp

from ..context import Context
from ..encoder import one_hot, circular, scalar, von_mises, gaussian, identity, cos_sin
from ..feature import Feature
from ..label import match_label, comparison_label
from ..phase import (concat, Phase, Fixation, Stimulus, Delay, Response, Sample, Test,
                     Cue, DeclarativePhase, VariableDuration)
from ..task import Task
from .._typing import Duration, Data
from ..utils import choice

__all__ = [
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
]


class _TmpFeature:  # only providing real num of directions for def make_encoder
    def __init__(self, num):
        self.num = num


# choose encoder
def make_encoder(mode: str, key: str, *, feature_per_direction: int = 1, kappa=2.0, base_value=0.0, num_dirs=None):
    """Build the per-direction stimulus encoder used by DMS-style tasks.

    All discrete-index modes (`one_hot`, `von_mises`, `circular`) treat
    ``ctx[key]`` as an integer direction index. `scalar` treats it as a
    continuous value broadcast across all dims.
    """
    mode = mode.lower()

    if feature_per_direction == 1:
        if mode == "von_mises":
            return von_mises(key, kappa=kappa, base_value=base_value, as_index=True, num_dirs=num_dirs)
        elif mode == "one_hot":
            return one_hot(key)
        elif mode == "circular":
            return circular(key, base_value=base_value, as_index=True, num_dirs=num_dirs)
        elif mode == "scalar":
            return scalar(key)
        else:
            raise ValueError(f"Unknown mode={mode}")

    ### expanding in terms of feture_per_direction
    if mode == "one_hot":
        base_encoder = one_hot(key)
        def enc(ctx, feature):
            total_n = feature.num
            K = feature_per_direction
            if total_n % K != 0:
                raise ValueError(f"stimulus dim {total_n} not divisible by K={K}")
            base_n = total_n // K
            base = base_encoder(ctx, _TmpFeature(base_n))
            return jnp.repeat(base, K)
        return enc
    
    if mode in ("von_mises", "circular"):
        def enc(ctx, feature):
            total_n = feature.num
            K = feature_per_direction
            if total_n % K != 0:
                raise ValueError(f"stimulus dim {total_n} not divisible by K={K}")
            base_n = total_n // K

            idx = ctx.get(key, 0)
            mu = (2.0 * jnp.pi) * (idx / base_n)

            pref = jnp.linspace(0, 2.0 * jnp.pi, total_n, endpoint=False)

            if mode == "circular":
                # cosine tuning
                response = jnp.cos(pref - mu)
                # map [-1,1] to [0,1] then scale to [base_value,1]
                response = 0.5 * (response + 1.0)
                response = response * (1.0 - base_value) + base_value
                return response

            # von_mises
            effective_kappa = kappa
            raw = jnp.exp(effective_kappa * jnp.cos(pref - mu))

            # normalize to [0,1]
            lo = jnp.exp(-effective_kappa)
            hi = jnp.exp(effective_kappa)
            response = (raw - lo) / (hi - lo)

            response = response * (1.0 - base_value) + base_value
            return response

        enc.__name__ = f"{mode}_population('{key}', K={feature_per_direction})"
        return enc

    raise ValueError(f"Unknown mode={mode}")

def build_cues(cue_dim: int, non_response_cue=None, response_cue=None):
    if non_response_cue is None:
        non_response_cue = [0.0] * cue_dim
    if response_cue is None:
        response_cue = [1.0] + [0.0] * (cue_dim - 1)

    non_response_cue = jnp.asarray(non_response_cue, dtype=jnp.float32)
    response_cue = jnp.asarray(response_cue, dtype=jnp.float32)

    if non_response_cue.shape != (cue_dim,) or response_cue.shape != (cue_dim,):
        raise ValueError("Cue vectors must have shape (cue_dim,)")

    return non_response_cue, response_cue



class DelayMatchSample(Task):
    """
    Delayed Match-to-Sample (DMS) task.

    Agent must remember a sample stimulus and indicate whether
    a later comparison stimulus matches it.

    Structure: Fixation >> Sample >> Delay >> Response

    In the response phase, the comparison stimulus is presented
    together with the response cue, and the agent must decide
    whether it matches the sample.

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 300ms).
    t_sample : Duration
        Sample presentation duration (default: 500ms).
    t_delay : Duration
        Delay period duration (default: 1000ms).
    t_response : Duration
        Response duration, during which the comparison stimulus is
        shown and the agent makes its decision (default: 500ms).
    num_stimuli : int
        Number of possible stimuli (default: 8).
    match_prob : float
        Probability of a match trial (default: 0.5).
    noise_sigma : Data
        Stimulus noise level (default: 0.0 * u.ms**0.5).
    base_value : float
        Baseline activity added to encoded stimulus features
        (default: 0.0).
    feature_per_direction : int
        Number of repeated encoded features per stimulus identity
        (default: 1).
    stimulus_encoding : str
        Encoding scheme for discrete stimuli. Supported values:
        'one_hot', 'von_mises', 'circular', 'scalar'
        (default: 'von_mises').
    kappa : float
        Concentration parameter for von Mises stimulus encoding
        (default: 2.0).
    sigma : float
        Reserved for compatibility with other stimulus encoders
        (default: 0.1).
    centers : Any
        Reserved for compatibility with other stimulus encoders
        (default: None).
    cue_dim : int
        Dimensionality of the fixation/response cue vector
        (default: 1).
    non_response_cue : array-like, optional
        Cue vector used during fixation, sample, and delay phases.
        Defaults to a zero vector of length `cue_dim`.
    response_cue : array-like, optional
        Cue vector used during response phase. Defaults to
        [1, 0, ..., 0].
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = DelayMatchSample()
    >>> task = DelayMatchSample(t_delay=2000*u.ms, num_stimuli=16)
    >>> X, Y, info = task.sample_trial(0)
    """
    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_sample: Duration = 500.0 * u.ms,
        t_delay: Duration = 1000.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        num_stimuli: int = 8,
        match_prob: float = 0.5,
        noise_sigma: Data = 0.0 * u.ms ** 0.5,
        base_value: float = 0.0,
        feature_per_direction: int = 1,
        stimulus_encoding: str = 'von_mises',
        kappa: float = 2.0,
        sigma: float = 0.1,
        centers=None,
        cue_dim: int = 1,
        non_response_cue=None,
        response_cue=None,
        **kwargs,
    ):
        self.t_fixation = t_fixation
        self.t_sample = t_sample
        self.t_delay = t_delay
        self.t_response = t_response
        self.num_stimuli = num_stimuli
        self.match_prob = match_prob
        self.noise_sigma = noise_sigma
        self.base_value = base_value
        self.feature_per_direction = feature_per_direction
        self.stimulus_encoding = stimulus_encoding
        self.kappa = kappa
        self.sigma = sigma
        self.sample_encoder = make_encoder(
            self.stimulus_encoding,
            "sample_idx",
            feature_per_direction=self.feature_per_direction,
            kappa=self.kappa,
            base_value=self.base_value
        )
        self.test_encoder = make_encoder(
            self.stimulus_encoding,
            "test_idx",
            feature_per_direction=self.feature_per_direction,
            kappa=self.kappa,
            base_value=self.base_value
        )
        self.cue_dim = cue_dim
        self.non_response_cue, self.response_cue = build_cues(
            cue_dim, non_response_cue, response_cue
        )

        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(self.cue_dim, 'fixation')
        stim_feat = Feature(self.num_stimuli * self.feature_per_direction, 'stimulus')
        input_features = fix_feat + stim_feat
        output_features = fix_feat + Feature(2, 'response')
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Sample(
                duration=self.t_sample,
                inputs={
                    'fixation': self.non_response_cue,
                    'stimulus': self.sample_encoder
                },
                outputs={'label': 0},
                noise={'stimulus': self.noise_sigma}
            ),
            Delay(
                duration=self.t_delay,
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Response(
                duration=self.t_response,
                inputs={
                    'fixation': self.response_cue,
                    'stimulus': self.test_encoder
                },
                outputs={'label': match_label('is_match', match_label=1, nonmatch_label=2)},
                noise={'stimulus': self.noise_sigma}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['sample_idx'] = ctx.rng.choice(self.num_stimuli)
        ctx['is_match'] = ctx.rng.random() < self.match_prob
        ctx['test_idx'] = jnp.where(
            ctx['is_match'],
            ctx['sample_idx'],
            choice(ctx.rng, self.num_stimuli, ctx['sample_idx'])
        )

    def get_trial_meta(self, trial_state):
        return trial_state['sample_idx'], trial_state['test_idx']


class DualDelayMatchSample(Task):
    """
    Dual Delayed Match-to-Sample task.

    Agent must remember two sample stimuli and indicate which one
    matches a later comparison stimulus.

    Structure:
    Fixation >> Sample1 >> Delay1 >> Sample2 >> Delay2 >> Response

    In the response phase, the comparison stimulus is presented
    together with the response cue, and the agent must decide
    whether it matches Sample1 or Sample2.

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 300ms).
    t_sample1 : Duration
        First sample presentation duration (default: 500ms).
    t_delay1 : Duration
        First delay duration (default: 1000ms).
    t_sample2 : Duration
        Second sample presentation duration (default: 500ms).
    t_delay2 : Duration
        Second delay duration (default: 1000ms).
    t_response : Duration
        Response duration, during which the comparison stimulus is
        shown and the agent makes its decision (default: 500ms).
    num_stimuli : int
        Number of possible stimuli (default: 8).
    noise_sigma : Data
        Stimulus noise level (default: 0.0 * u.ms**0.5).
    base_value : float
        Baseline activity added to encoded stimulus features
        (default: 0.0).
    feature_per_direction : int
        Number of repeated encoded features per stimulus identity
        (default: 1).
    stimulus_encoding : str
        Encoding scheme for discrete stimuli. Supported values:
        'one_hot', 'von_mises', 'circular', 'scalar'
        (default: 'one_hot').
    kappa : float
        Concentration parameter for von Mises stimulus encoding
        (default: 2.0).
    cue_dim : int
        Dimensionality of the fixation/response cue vector
        (default: 1).
    non_response_cue : array-like, optional
        Cue vector used during fixation, sample, and delay phases.
        Defaults to a zero vector of length `cue_dim`.
    response_cue : array-like, optional
        Cue vector used during response phase. Defaults to
        [1, 0, ..., 0].
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = DualDelayMatchSample()
    >>> X, Y, info = task.sample_trial(0)
    """
    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_sample1: Duration = 500.0 * u.ms,
        t_delay1: Duration = 1000.0 * u.ms,
        t_sample2: Duration = 500.0 * u.ms,
        t_delay2: Duration = 1000.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        num_stimuli: int = 8,
        noise_sigma: Data = 0.0 * u.ms ** 0.5,
        base_value: float = 0.0,
        feature_per_direction: int = 1,
        stimulus_encoding: str = 'one_hot',
        kappa: float = 2.0,
        cue_dim: int = 1,
        non_response_cue=None,
        response_cue=None,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_sample1 = t_sample1
        self.t_delay1 = t_delay1
        self.t_sample2 = t_sample2
        self.t_delay2 = t_delay2
        self.t_response = t_response
        self.num_stimuli = num_stimuli
        self.noise_sigma = noise_sigma
        self.base_value = base_value
        self.feature_per_direction = feature_per_direction
        self.stimulus_encoding = stimulus_encoding
        self.kappa = kappa

        self.sample1_encoder = make_encoder(
            self.stimulus_encoding, "sample1_idx",
            feature_per_direction=self.feature_per_direction,
            kappa=self.kappa,
            base_value=self.base_value
        )
        self.sample2_encoder = make_encoder(
            self.stimulus_encoding, "sample2_idx",
            feature_per_direction=self.feature_per_direction,
            kappa=self.kappa,
            base_value=self.base_value
        )
        self.test_encoder = make_encoder(
            self.stimulus_encoding, "test_idx",
            feature_per_direction=self.feature_per_direction,
            kappa=self.kappa,
            base_value=self.base_value
        )

        self.cue_dim = cue_dim
        self.non_response_cue, self.response_cue = build_cues(
            cue_dim, non_response_cue, response_cue
        )

        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(self.cue_dim, 'fixation')
        stim_feat = Feature(self.num_stimuli * self.feature_per_direction, 'stimulus')
        input_features = fix_feat + stim_feat
        output_features = fix_feat + Feature(2, 'response')
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Sample(
                duration=self.t_sample1,
                name='sample1',
                inputs={
                    'fixation': self.non_response_cue,
                    'stimulus': self.sample1_encoder
                },
                outputs={'label': 0},
                noise={'stimulus': self.noise_sigma}
            ),
            Delay(
                duration=self.t_delay1,
                name='delay1',
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Sample(
                duration=self.t_sample2,
                name='sample2',
                inputs={
                    'fixation': self.non_response_cue,
                    'stimulus': self.sample2_encoder
                },
                outputs={'label': 0},
                noise={'stimulus': self.noise_sigma}
            ),
            Delay(
                duration=self.t_delay2,
                name='delay2',
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Response(
                duration=self.t_response,
                name='response',
                inputs={
                    'fixation': self.response_cue,
                    'stimulus': self.test_encoder
                },
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1},
                noise={'stimulus': self.noise_sigma}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        indices = ctx.rng.choice(self.num_stimuli, size=2, replace=False)
        ctx['sample1_idx'] = indices[0]
        ctx['sample2_idx'] = indices[1]
        ctx['ground_truth'] = ctx.rng.choice(2)
        ctx['test_idx'] = jnp.where(
            ctx['ground_truth'] == 0,
            ctx['sample1_idx'],
            ctx['sample2_idx']
        )

class DelayComparison(Task):
    """
    Delayed Comparison task.

    Agent compares the magnitudes of two stimuli separated by a delay,
    and must indicate whether the later comparison stimulus is greater
    or less than the sample stimulus.

    Structure: Fixation >> Sample >> Delay >> Response

    In the response phase, the comparison stimulus is presented
    together with the response cue, and the agent must decide
    whether test > sample or test < sample.

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_sample : Duration
        Sample presentation duration (default: 500ms).
    t_delay : Duration
        Delay duration (default: 1000ms).
    t_response : Duration
        Response duration, during which the comparison stimulus is
        shown and the agent makes its decision (default: 500ms).
    value_range : tuple
        Range (min, max) from which sample and test values are drawn
        (default: (0.0, 1.0)).
    num_features : int
        Number of stimulus features used when the encoding produces
        a population code, e.g. Gaussian encoding (default: 10).
    value_encoding : str
        Encoding scheme for continuous stimulus values. Supported
        values: 'scalar', 'gaussian', 'identity'
        (default: 'scalar').
    sigma : float
        Width parameter for Gaussian value encoding
        (default: 0.1).
    centers : Any
        Optional Gaussian centers for value encoding
        (default: None).
    noise_sigma : Data
        Stimulus noise level (default: 0.1 * u.ms**0.5).
    cue_dim : int
        Dimensionality of the fixation/response cue vector
        (default: 1).
    non_response_cue : array-like, optional
        Cue vector used during fixation, sample, and delay phases.
        Defaults to a zero vector of length `cue_dim`.
    response_cue : array-like, optional
        Cue vector used during response phase. Defaults to
        [1, 0, ..., 0].
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = DelayComparison()
    >>> X, Y, info = task.sample_trial(0)
    """
    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_sample: Duration = 500.0 * u.ms,
        t_delay: Duration = 1000.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        value_range: tuple = (0.0, 1.0),
        num_features: int = 10,
        value_encoding: str = 'scalar',   # 'scalar' | 'gaussian' | 'identity'
        sigma: float = 0.1,
        centers=None,
        noise_sigma: Data = 0.1 * u.ms ** 0.5,
        cue_dim: int = 1,
        non_response_cue=None,
        response_cue=None,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_sample = t_sample
        self.t_delay = t_delay
        self.t_response = t_response
        self.value_range = value_range
        self.num_features = num_features
        self.value_encoding = value_encoding
        self.sigma = sigma
        self.centers = centers
        self.noise_sigma = noise_sigma

        self.cue_dim = cue_dim
        self.non_response_cue, self.response_cue = build_cues(
            cue_dim, non_response_cue, response_cue
        )

        self.sample_encoder = self._make_value_encoder('sample_value')
        self.test_encoder = self._make_value_encoder('test_value')

        super().__init__(**kwargs)

    def _make_value_encoder(self, key: str):
        mode = self.value_encoding.lower()
        if mode == 'scalar':
            return scalar(key)
        elif mode == 'identity':
            return identity(key)
        elif mode == 'gaussian':
            return gaussian(key, sigma=self.sigma, centers=self.centers)
        else:
            raise ValueError(f"Unknown value_encoding={self.value_encoding}")

    def define_features(self) -> Tuple:
        fix_feat = Feature(self.cue_dim, 'fixation')

        if self.value_encoding.lower() in ('scalar', 'identity'):
            stim_dim = 1
        elif self.value_encoding.lower() == 'gaussian':
            stim_dim = self.num_features
        else:
            raise ValueError(f"Unknown value_encoding={self.value_encoding}")

        stim_feat = Feature(stim_dim, 'stimulus')
        input_features = fix_feat + stim_feat
        output_features = fix_feat + Feature(2, 'response')
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Stimulus(
                duration=self.t_sample,
                name='sample',
                inputs={
                    'fixation': self.non_response_cue,
                    'stimulus': self.sample_encoder
                },
                outputs={'label': 0},
                noise={'stimulus': self.noise_sigma}
            ),
            Delay(
                duration=self.t_delay,
                name='delay',
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Response(
                duration=self.t_response,
                name='response',
                inputs={
                    'fixation': self.response_cue,
                    'stimulus': self.test_encoder
                },
                outputs={'label': comparison_label('comparison_result', greater_label=1, less_label=2)},
                noise={'stimulus': self.noise_sigma}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        min_val, max_val = self.value_range
        ctx['sample_value'] = ctx.rng.uniform(min_val, max_val)
        ctx['test_value'] = ctx.rng.uniform(min_val, max_val)
        ctx['comparison_result'] = ctx['test_value'] > ctx['sample_value']
        ctx['ground_truth'] = jnp.where(ctx['comparison_result'], 0, 1)


class DelayMatchCategory(Task):
    """
    Delayed Match-to-Category task.

    Agent must remember the category of a sample stimulus and indicate
    whether a later comparison stimulus belongs to the same category.

    Structure: Fixation >> Sample >> Delay >> Response

    In the response phase, the comparison stimulus is presented
    together with the response cue, and the agent must decide
    whether it belongs to the same category as the sample.

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 300ms).
    t_sample : Duration
        Sample presentation duration (default: 650ms).
    t_delay : Duration
        Delay duration (default: 1000ms).
    t_response : Duration
        Response duration, during which the comparison stimulus is
        shown and the agent makes its decision (default: 500ms).
    num_categories : int
        Number of categories (default: 2).
    num_exemplars : int
        Number of exemplars per category (default: 4).
    match_prob : float
        Probability that sample and comparison stimuli belong to the
        same category (default: 0.5).
    noise_sigma : Data
        Stimulus noise level (default: 0.0 * u.ms**0.5).
    base_value : float
        Baseline activity added to encoded stimulus features
        (default: 0.0).
    feature_per_direction : int
        Number of repeated encoded features per stimulus identity
        (default: 1).
    stimulus_encoding : str
        Encoding scheme for discrete stimuli. Supported values:
        'one_hot', 'von_mises', 'circular', 'scalar'
        (default: 'one_hot').
    kappa : float
        Concentration parameter for von Mises stimulus encoding
        (default: 2.0).
    cue_dim : int
        Dimensionality of the fixation/response cue vector
        (default: 1).
    non_response_cue : array-like, optional
        Cue vector used during fixation, sample, and delay phases.
        Defaults to a zero vector of length `cue_dim`.
    response_cue : array-like, optional
        Cue vector used during response phase. Defaults to
        [1, 0, ..., 0].
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = DelayMatchCategory()
    >>> X, Y, info = task.sample_trial(0)
    """
    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_sample: Duration = 650.0 * u.ms,
        t_delay: Duration = 1000.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        num_categories: int = 2,
        num_exemplars: int = 4,
        match_prob: float = 0.5,
        noise_sigma: Data = 0.0 * u.ms ** 0.5,
        base_value: float = 0.0,
        feature_per_direction: int = 1,
        stimulus_encoding: str = 'one_hot',
        kappa: float = 2.0,
        cue_dim: int = 1,
        non_response_cue=None,
        response_cue=None,
        **kwargs,
    ):
        self.t_fixation = t_fixation
        self.t_sample = t_sample
        self.t_delay = t_delay
        self.t_response = t_response
        self.num_categories = num_categories
        self.num_exemplars = num_exemplars
        self.match_prob = match_prob
        self.noise_sigma = noise_sigma
        self.base_value = base_value
        self.feature_per_direction = feature_per_direction
        self.stimulus_encoding = stimulus_encoding
        self.kappa = kappa

        self.total_stimuli = self.num_categories * self.num_exemplars
        self.sample_encoder = make_encoder(
            self.stimulus_encoding, "sample_idx",
            feature_per_direction=self.feature_per_direction,
            kappa=self.kappa,
            base_value=self.base_value
        )
        self.test_encoder = make_encoder(
            self.stimulus_encoding, "test_idx",
            feature_per_direction=self.feature_per_direction,
            kappa=self.kappa,
            base_value=self.base_value
        )

        self.cue_dim = cue_dim
        self.non_response_cue, self.response_cue = build_cues(
            cue_dim, non_response_cue, response_cue
        )

        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(self.cue_dim, 'fixation')
        stim_feat = Feature(self.total_stimuli * self.feature_per_direction, 'stimulus')
        input_features = fix_feat + stim_feat
        output_features = fix_feat + Feature(2, 'response')
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Sample(
                duration=self.t_sample,
                name='sample',
                inputs={
                    'fixation': self.non_response_cue,
                    'stimulus': self.sample_encoder
                },
                outputs={'label': 0},
                noise={'stimulus': self.noise_sigma}
            ),
            Delay(
                duration=self.t_delay,
                name='delay',
                inputs={'fixation': self.non_response_cue},
                outputs={'label': 0}
            ),
            Response(
                duration=self.t_response,
                name='response',
                inputs={
                    'fixation': self.response_cue,
                    'stimulus': self.test_encoder
                },
                outputs={'label': match_label('is_match', match_label=1, nonmatch_label=2)},
                noise={'stimulus': self.noise_sigma}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['sample_category'] = ctx.rng.choice(self.num_categories)
        ctx['sample_exemplar'] = ctx.rng.choice(self.num_exemplars)
        ctx['sample_idx'] = ctx['sample_category'] * self.num_exemplars + ctx['sample_exemplar']

        ctx['is_match'] = ctx.rng.random() < self.match_prob
        ctx['test_category'] = jnp.where(
            ctx['is_match'],
            ctx['sample_category'],
            choice(ctx.rng, self.num_categories, ctx['sample_category'])
        )
        ctx['test_exemplar'] = ctx.rng.choice(self.num_exemplars)
        ctx['test_idx'] = ctx['test_category'] * self.num_exemplars + ctx['test_exemplar']

    def get_trial_meta(self, trial_state):
        return {
            'sample_idx': trial_state['sample_idx'],
            'sample_category': trial_state['sample_category'],
            'sample_exemplar': trial_state['sample_exemplar'],
            'test_idx': trial_state['test_idx'],
            'test_category': trial_state['test_category'],
            'test_exemplar': trial_state['test_exemplar'],
            'is_match': trial_state['is_match'],
        }

class DelayPairedAssociation(Task):
    """
    Delayed Paired Association task.

    Agent learns associations between pairs of stimuli. Given one
    item, must select its paired associate.

    Structure: Fixation >> Sample >> Delay >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 300ms).
    t_sample : Duration
        Sample duration (default: 500ms).
    t_delay : Duration
        Delay duration (default: 1000ms).
    t_response : Duration
        Response duration (default: 500ms).
    num_pairs : int
        Number of stimulus pairs (default: 4).
    noise_sigma : Data
        Stimulus noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = DelayPairedAssociation()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_sample: Duration = 500.0 * u.ms,
        t_delay: Duration = 1000.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        num_pairs: int = 4,
        noise_sigma: Data = 1.0 * u.ms ** 0.5,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_sample = t_sample
        self.t_delay = t_delay
        self.t_response = t_response
        self.num_pairs = num_pairs
        self.noise_sigma = noise_sigma
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        stim_feat = Feature(self.num_pairs * 2, 'stimulus')
        input_features = fix_feat + stim_feat
        output_features = fix_feat + Feature(self.num_pairs, 'response')
        return input_features, output_features

    def define_phases(self) -> Phase:
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            Sample(
                duration=self.t_sample,
                name='sample',
                inputs={
                    'fixation': 1.0,
                    'stimulus': one_hot('sample_idx')
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
        ctx['pair_idx'] = ctx.rng.choice(self.num_pairs)
        ctx['sample_idx'] = ctx['pair_idx']
        ctx['ground_truth'] = ctx['pair_idx']


class GoNoGo(Task):
    """
    Go/No-Go task.

    Agent must respond on 'go' trials and withhold response on 'no-go' trials.

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
    go_prob : float
        Probability of go trial (default: 0.5).
    noise_sigma : Data
        Stimulus noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = GoNoGo()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_stimulus: Duration = 500.0 * u.ms,
        t_delay: Duration = 500.0 * u.ms,
        t_response: Duration = 500.0 * u.ms,
        go_prob: float = 0.5,
        noise_sigma: Data = 1.0 * u.ms ** 0.5,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_delay = t_delay
        self.t_response = t_response
        self.go_prob = go_prob
        self.noise_sigma = noise_sigma
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        stim_feat = Feature(2, 'stimulus')  # go/nogo cues
        input_features = fix_feat + stim_feat
        output_features = fix_feat + Feature(2, 'response')
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
                outputs={'label': lambda ctx, f: ctx['ground_truth_label']}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['is_go'] = ctx.rng.random() < self.go_prob
        ctx['stimulus_class'] = jnp.where(ctx['is_go'], 0, 1)
        # Response-phase label convention:
        #   1 = "act" on go trials, 0 = "withhold" on no-go trials. Encoding
        #   no-go with label 0 (same as fixation) makes the network learn to
        #   actually suppress responding, which is what a real go/no-go task
        #   requires — rather than emitting a distinctive "no-go" symbol.
        ctx['ground_truth_label'] = jnp.where(ctx['is_go'], 1, 0)
        # Keep ground_truth for backward compat with downstream analyses.
        ctx['ground_truth'] = jnp.where(ctx['is_go'], 0, 1)


class IntervalDiscrimination(Task):
    """
    Interval Discrimination task.

    Agent compares durations of two intervals and indicates which is longer.

    Structure: Fixation >> Interval1 >> Delay >> Interval2 >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 500ms).
    t_interval1 : tuple
        (min, max) for first interval duration (default: (400ms, 800ms)).
    t_delay : Duration
        Delay duration (default: 500ms).
    t_interval2 : tuple
        (min, max) for second interval duration (default: (400ms, 800ms)).
    t_response : Duration
        Response duration (default: 500ms).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = IntervalDiscrimination()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 500.0 * u.ms,
        t_interval1: tuple = (400.0 * u.ms, 800.0 * u.ms),
        t_delay: Duration = 500.0 * u.ms,
        t_interval2: tuple = (400.0 * u.ms, 800.0 * u.ms),
        t_response: Duration = 500.0 * u.ms,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_interval1 = t_interval1
        self.t_delay = t_delay
        self.t_interval2 = t_interval2
        self.t_response = t_response
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        stim_feat = Feature(1, 'interval')
        input_features = fix_feat + stim_feat
        output_features = fix_feat + Feature(2, 'response')
        return input_features, output_features

    def define_phases(self) -> Phase:
        interval1 = VariableDuration(
            min_duration=self.t_interval1[0],
            max_duration=self.t_interval1[1],
            ctx_key='interval1_duration',
            name='interval1',
            inputs={
                'fixation': 1.0,
                'interval': 1.0
            },
            outputs={'label': 0},
        )

        interval2 = VariableDuration(
            min_duration=self.t_interval2[0],
            max_duration=self.t_interval2[1],
            ctx_key='interval2_duration',
            name='interval2',
            inputs={
                'fixation': 1.0,
                'interval': 1.0
            },
            outputs={'label': 0},
        )

        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            interval1,
            Delay(
                duration=self.t_delay,
                name='delay',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            interval2,
            Response(
                duration=self.t_response,
                name='response',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        # Convert Quantity tuples to float (in ms)
        int1_min = float(self.t_interval1[0].to(u.ms).mantissa)
        int1_max = float(self.t_interval1[1].to(u.ms).mantissa)
        int2_min = float(self.t_interval2[0].to(u.ms).mantissa)
        int2_max = float(self.t_interval2[1].to(u.ms).mantissa)

        ctx['interval1_duration'] = ctx.rng.uniform(int1_min, int1_max)
        ctx['interval2_duration'] = ctx.rng.uniform(int2_min, int2_max)
        # Ground truth: 0 if interval1 > interval2, 1 otherwise
        ctx['ground_truth'] = jnp.where(ctx['interval1_duration'] > ctx['interval2_duration'], 0, 1)


class PostDecisionWager(Task):
    """
    Post-Decision Wager task.

    Agent makes a perceptual decision, then bets on its confidence.
    High bet = confident, low bet = uncertain.

    Structure: Fixation >> Stimulus >> Delay >> Decision >> Wager

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration (default: 300ms).
    t_stimulus : Duration
        Stimulus duration (default: 1000ms).
    t_delay : Duration
        Delay duration (default: 500ms).
    t_decision : Duration
        Decision period duration (default: 500ms).
    t_wager : Duration
        Wager period duration (default: 500ms).
    num_choices : int
        Number of choices (default: 2).
    coherences : sequence
        Coherence levels (default: (0, 6.4, 12.8, 25.6, 51.2)).
    noise_sigma : Data
        Stimulus noise (default: 1.0 * u.ms**0.5).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = PostDecisionWager()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_fixation: Duration = 300.0 * u.ms,
        t_stimulus: Duration = 1000.0 * u.ms,
        t_delay: Duration = 500.0 * u.ms,
        t_decision: Duration = 500.0 * u.ms,
        t_wager: Duration = 500.0 * u.ms,
        num_choices: int = 2,
        coherences: Sequence[float] = (0, 6.4, 12.8, 25.6, 51.2),
        noise_sigma: Data = 1.0 * u.ms ** 0.5,
        **kwargs
    ):
        self.t_fixation = t_fixation
        self.t_stimulus = t_stimulus
        self.t_delay = t_delay
        self.t_decision = t_decision
        self.t_wager = t_wager
        self.num_choices = num_choices
        self.coherences = jnp.asarray(coherences, dtype=jnp.float32)
        self.noise_sigma = noise_sigma
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        stim_feat = Feature(self.num_choices * 4, 'stimulus')
        input_features = fix_feat + stim_feat
        choice_feat = Feature(self.num_choices, 'choice')
        wager_feat = Feature(2, 'wager')  # high/low
        output_features = fix_feat + choice_feat + wager_feat
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
                duration=self.t_decision,
                name='decision',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
            Response(
                duration=self.t_wager,
                name='wager',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['wager_gt'] + self.num_choices + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        ctx['ground_truth'] = ctx.rng.choice(self.num_choices)
        ctx['coherence'] = ctx.rng.choice(self.coherences)
        directions = jnp.linspace(0, 2 * jnp.pi, self.num_choices, endpoint=False)
        ctx['stimulus_direction'] = directions[ctx['ground_truth']]
        # Optimal wager depends on coherence (high coherence = should bet high)
        ctx['wager_gt'] = jnp.where(ctx['coherence'] >= 25.6, 0, 1)


class ReadySetGo(Task):
    """
    Ready-Set-Go timing task.

    Agent must produce an interval matching a measured interval.
    Ready cue, then Set cue after interval T, then agent produces interval T.

    Structure: Ready >> Interval >> Set >> Production

    Parameters
    ----------
    t_ready : Duration
        Ready cue duration (default: 500ms).
    t_set : Duration
        Set cue duration (default: 500ms).
    t_interval : tuple
        (min, max) for measured interval (default: (400ms, 800ms)).
    t_response_max : Duration
        Maximum production time (default: 1000ms).
    gain : float
        Production = gain * measured interval (default: 1.0).
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> task = ReadySetGo()
    >>> X, Y, info = task.sample_trial(0)
    """

    def __init__(
        self,
        t_ready: Duration = 500.0 * u.ms,
        t_set: Duration = 500.0 * u.ms,
        t_interval: tuple = (400.0 * u.ms, 800.0 * u.ms),
        t_response_max: Duration = 1000.0 * u.ms,
        gain: float = 1.0,
        **kwargs
    ):
        self.t_ready = t_ready
        self.t_set = t_set
        self.t_interval = t_interval
        self.t_response_max = t_response_max
        self.gain = gain
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        ready_feat = Feature(1, 'ready')
        set_feat = Feature(1, 'set')
        input_features = fix_feat + ready_feat + set_feat
        go_feat = Feature(1, 'go')
        output_features = fix_feat + go_feat
        return input_features, output_features

    def _production_label(self, ctx: Context, feature) -> jnp.ndarray:
        """Time-varying production label across the entire phase slice.

        Label is 1 ("hold") until ``go_time`` ticks have elapsed, then 2 ("go").
        Returns an int32 array of shape ``(phase_duration,)`` which
        DeclarativePhase writes into ``ctx.outputs`` directly.
        """
        duration = ctx.phase_end - ctx.phase_start
        dt = ctx.dt
        dt_val = float(dt.mantissa) if hasattr(dt, 'mantissa') else float(dt)
        go_time = ctx['produce_interval'] / dt_val
        t = jnp.arange(duration)
        return jnp.where(t < go_time, 1, 2).astype(jnp.int32)

    def define_phases(self) -> Phase:
        measure_interval = VariableDuration(
            min_duration=self.t_interval[0],
            max_duration=self.t_interval[1],
            ctx_key='measure_interval',
            name='measure',
            inputs={'fixation': 1.0},
            outputs={'label': 0},
        )

        return concat([
            Cue(
                duration=self.t_ready,
                name='ready',
                inputs={
                    'fixation': 1.0,
                    'ready': 1.0
                },
                outputs={'label': 0}
            ),
            measure_interval,
            Cue(
                duration=self.t_set,
                name='set',
                inputs={
                    'fixation': 1.0,
                    'set': 1.0
                },
                outputs={'label': 0}
            ),
            DeclarativePhase(
                duration=self.t_response_max,
                name='production',
                inputs={'fixation': 0.0},
                outputs={'label': self._production_label}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        # Convert Quantity tuple to float (in ms)
        int_min = float(self.t_interval[0].to(u.ms).mantissa)
        int_max = float(self.t_interval[1].to(u.ms).mantissa)

        ctx['measure_interval'] = ctx.rng.uniform(int_min, int_max)
        ctx['produce_interval'] = ctx['measure_interval'] * self.gain
        # ``go_time`` is computed lazily inside _production_label so we don't
        # do a Python int() on a traced JAX value (which would break vmap).


class DelayDirectionReproduction(Task):
    """
    Delay Direction Reproduction task.

    The agent observes a sample direction, maintains it over a delay period,
    and then reproduces the remembered direction during the response phase.
    The target output is a continuous 2D direction vector encoded as
    [cos(theta), sin(theta)], together with a fixation/output-gating signal.

    Structure: Fixation >> Sample >> Delay >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration before stimulus onset (default: 300ms).
    t_sample : Duration
        Sample stimulus presentation duration (default: 500ms).
    t_delay : Duration
        Delay period duration during which the direction must be maintained
        in memory (default: 1000ms).
    t_response : Duration
        Response duration during which the agent must reproduce the stored
        direction (default: 500ms).
    num_stimuli : int
        Number of discrete sample directions uniformly distributed on the
        circle (default: 8).
    noise_sigma : Data
        Input stimulus noise level (default: 0.0 * u.ms**0.5).
    base_value : float
        Baseline value used by the stimulus encoder (default: 0.0).
    feature_per_direction : int
        Number of repeated encoding features per direction. For von Mises
        encoding, the total stimulus dimension is
        num_stimuli * feature_per_direction. For cos/sin encoding, the total
        stimulus dimension is 2 * feature_per_direction (default: 1).
    IfVon : bool
        Whether to use von Mises population encoding for the input stimulus.
        If False, use repeated cos/sin encoding instead (default: True).
    kappa : float
        Concentration parameter for von Mises stimulus encoding
        (default: 2.0).
    cue_dim : int
        Dimensionality of the fixation / task-state cue input
        (default: 2).
    non_response_cue : array-like, optional
        Cue vector used during fixation, sample, and delay phases.
        If None, defaults to [1, 0, ..., 0].
    response_cue : array-like, optional
        Cue vector used during the response phase.
        If None, defaults to [0, 1, 0, ..., 0].

    Notes
    -----
    Input consists of:
    - a cue vector indicating task phase (non-response vs response),
    - a stimulus representation encoding the sample direction.

    Output consists of:
    - fixation_out: a scalar indicating whether fixation / withholding
      should be maintained,
    - direction: a 2D continuous report of the remembered sample direction
      in [cos(theta), sin(theta)] form.

    This task is a continuous-report working memory task, unlike DMS-style
    tasks that require categorical decisions.

    Examples
    --------
    >>> task = DelayedDirectionReproduction()
    >>> X, Y, info = task.sample_trial(0)
    """
    def __init__(
        self,
        t_fixation=300.0*u.ms,
        t_sample=500.0*u.ms,
        t_delay=1000.0*u.ms,
        t_response=500.0*u.ms,
        num_stimuli=8,
        noise_sigma=0.0 * u.ms**0.5,
        base_value=0.0,
        feature_per_direction=1,
        IfVon=True,
        kappa=2.0,
        cue_dim=2,
        non_response_cue=None,
        response_cue=None,
        **kwargs,
    ):
        self.t_fixation=t_fixation
        self.t_sample=t_sample
        self.t_delay=t_delay
        self.t_response=t_response
        self.num_stimuli=num_stimuli
        self.noise_sigma=noise_sigma
        self.base_value=base_value
        self.feature_per_direction=feature_per_direction
        self.IfVon=IfVon
        self.kappa=kappa

        self.cue_dim=cue_dim
        if non_response_cue is None: non_response_cue=[1.0]+[0.0]*(cue_dim-1)
        if response_cue is None: response_cue=[0.0]+[1.0]+[0.0]*(cue_dim-2)
        self.non_response_cue=jnp.asarray(non_response_cue, jnp.float32)
        self.response_cue=jnp.asarray(response_cue, jnp.float32)

        # input stimulus encoder
        if self.IfVon:
            self.sample_encoder = von_mises(
                "sample_idx", kappa=self.kappa, base_value=self.base_value,
                as_index=True, num_dirs=self.num_stimuli,
            )
        else:
            self.sample_encoder = cos_sin(
                "sample_idx",
                num_dirs=self.num_stimuli,
                repeats=self.feature_per_direction,
                base_value=self.base_value,
                map_to_01=False
            )


        super().__init__(output_mode="vector", **kwargs)

    def define_features(self):
        # INPUT
        fix_in = Feature(self.cue_dim, "fixation")
        if self.IfVon:
            stim_in = Feature(self.num_stimuli*self.feature_per_direction, "stimulus")
        else:
            stim_in = Feature(2*self.feature_per_direction, "stimulus")
        input_features = fix_in + stim_in

        # OUTPUT: fixation_out (1 dim) + direction (2 dim)
        fix_out = Feature(1, "fixation_out")
        dir_out = Feature(2, "direction")
        output_features = fix_out + dir_out
        return input_features, output_features

    def _set_target_direction(self, ctx: Context):
        idx = jnp.asarray(ctx["sample_idx"])
        theta = (2.0*jnp.pi) * (idx / float(self.num_stimuli))
        ctx["target_direction"] = jnp.asarray([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)

    def define_phases(self):
        return concat([
            Fixation(
                duration=self.t_fixation,
                inputs={"fixation": self.non_response_cue},
                outputs={
                    "fixation_out": jnp.asarray([1.0], jnp.float32),
                    "direction": jnp.zeros((2,), jnp.float32),
                },
            ),
            Sample(
                duration=self.t_sample,
                inputs={"fixation": self.non_response_cue, "stimulus": self.sample_encoder},
                outputs={
                    "fixation_out": jnp.asarray([1.0], jnp.float32),
                    "direction": jnp.zeros((2,), jnp.float32),
                },
                noise={"stimulus": self.noise_sigma},
            ),
            Delay(
                duration=self.t_delay,
                inputs={"fixation": self.non_response_cue},
                outputs={
                    "fixation_out": jnp.asarray([1.0], jnp.float32),
                    "direction": jnp.zeros((2,), jnp.float32),
                },
            ),
            Response(
                duration=self.t_response,
                inputs={"fixation": self.response_cue},
                outputs={
                    "fixation_out": jnp.asarray([0.0], jnp.float32),
                    "direction": identity("target_direction"),
                },
            ),
        ])

    def trial_init(self, ctx: Context):
        ctx["sample_idx"] = ctx.rng.choice(self.num_stimuli)
        self._set_target_direction(ctx)

    # return the sample direction index
    def get_trial_meta(self, trial_state):
        return trial_state['sample_idx']


class ImmediateDirectionReproduction(Task):
    """
    Immediate Direction Reproduction task.

    The agent reports the direction of a currently presented stimulus
    immediately, without any memory delay. The target output is a continuous
    2D direction vector encoded as [cos(theta), sin(theta)], together with
    a fixation/output-gating signal.

    Structure: Fixation >> Response

    Parameters
    ----------
    t_fixation : Duration
        Fixation duration before the go / response period (default: 300ms).
    t_go : Duration
        Response period duration during which the stimulus is presented and
        the agent must report its direction immediately (default: 500ms).
    num_stimuli : int
        Number of discrete stimulus directions uniformly distributed on the
        circle (default: 8).
    noise_sigma : Data
        Input stimulus noise level (default: 0.0 * u.ms**0.5).
    base_value : float
        Baseline value used by the stimulus encoder (default: 0.0).
    feature_per_direction : int
        Number of repeated encoding features per direction. For von Mises
        encoding, the total stimulus dimension is
        num_stimuli * feature_per_direction. For cos/sin encoding, the total
        stimulus dimension is 2 * feature_per_direction (default: 1).
    IfVon : bool
        Whether to use von Mises population encoding for the input stimulus.
        If False, use repeated cos/sin encoding instead (default: True).
    kappa : float
        Concentration parameter for von Mises stimulus encoding
        (default: 2.0).
    cue_dim : int
        Dimensionality of the fixation / task-state cue input
        (default: 2).
    non_response_cue : array-like, optional
        Cue vector used during fixation phase.
        If None, defaults to [1, 0, ..., 0].
    response_cue : array-like, optional
        Cue vector used during the response phase.
        If None, defaults to [0, 1, 0, ..., 0].

    Notes
    -----
    Input consists of:
    - a cue vector indicating task phase (fixation vs response),
    - a stimulus representation encoding the current direction.

    Output consists of:
    - fixation_out: a scalar indicating whether fixation / withholding
      should be maintained,
    - direction: a 2D continuous report of the current stimulus direction
      in [cos(theta), sin(theta)] form.

    This task serves as a no-memory control for DelayedDirectionReproduction:
    both require continuous direction report, but only the delayed version
    requires working memory across a delay period.

    Examples
    --------
    >>> task = ImmediateDirectionReproduction()
    >>> X, Y, info = task.sample_trial(0)
    """
    def __init__(
        self,
        t_fixation=300.0*u.ms,
        t_go=500.0*u.ms,
        num_stimuli=8,
        noise_sigma=0.0 * u.ms**0.5,
        base_value=0.0,
        feature_per_direction=1,
        IfVon=True,
        kappa=2.0,
        cue_dim=2,
        non_response_cue=None,
        response_cue=None,
        **kwargs,
    ):
        self.t_fixation=t_fixation
        self.t_go=t_go
        self.num_stimuli=num_stimuli
        self.noise_sigma=noise_sigma
        self.base_value=base_value
        self.feature_per_direction=feature_per_direction
        self.IfVon=IfVon
        self.kappa=kappa

        self.cue_dim=cue_dim
        if non_response_cue is None: non_response_cue=[1.0]+[0.0]*(cue_dim-1)
        if response_cue is None: response_cue=[0.0]+[1.0]+[0.0]*(cue_dim-2)
        self.non_response_cue=jnp.asarray(non_response_cue, jnp.float32)
        self.response_cue=jnp.asarray(response_cue, jnp.float32)

        # input stimulus encoder
        if self.IfVon:
            self.sample_encoder = von_mises(
                "sample_idx", kappa=self.kappa, base_value=self.base_value,
                as_index=True, num_dirs=self.num_stimuli,
            )
        else:
            self.sample_encoder = cos_sin(
                "sample_idx", num_dirs=self.num_stimuli, repeats=self.feature_per_direction,
                base_value=self.base_value, map_to_01=False
            )


        super().__init__(output_mode="vector", **kwargs)

    def define_features(self):
        # INPUT
        fix_in = Feature(self.cue_dim, "fixation")
        if self.IfVon:
            stim_in = Feature(self.num_stimuli*self.feature_per_direction, "stimulus")
        else:
            stim_in = Feature(2*self.feature_per_direction, "stimulus")
        input_features = fix_in + stim_in

        # OUTPUT: fixation_out (1 dim) + direction (2 dim)
        fix_out = Feature(1, "fixation_out")
        dir_out = Feature(2, "direction")
        output_features = fix_out + dir_out
        return input_features, output_features

    def _set_target_direction(self, ctx: Context):
        idx = jnp.asarray(ctx["sample_idx"])
        theta = (2.0*jnp.pi) * (idx / float(self.num_stimuli))
        ctx["target_direction"] = jnp.asarray([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)

    def define_phases(self):
        return concat([
            Fixation(
                duration=self.t_fixation,
                inputs={"fixation": self.non_response_cue},
                outputs={
                    "fixation_out": jnp.asarray([1.0], jnp.float32),
                    "direction": jnp.zeros((2,), jnp.float32),
                },
            ),
            Response(
                duration=self.t_go,
                inputs={"fixation": self.response_cue, "stimulus": self.sample_encoder},
                outputs={
                    "fixation_out": jnp.asarray([0.0], jnp.float32),
                    "direction": identity("target_direction"),
                },
            ),
        ])

    def trial_init(self, ctx: Context):
        ctx["sample_idx"] = ctx.rng.choice(self.num_stimuli)
        self._set_target_direction(ctx)

    # return the sample direction index
    def get_trial_meta(self, trial_state):
        return trial_state['sample_idx']


class DelayDirectionClassification(Task):
    """
    Delayed Direction Classification (DDC).

    Fixation >> Sample(direction) >> Delay >> Response(classification)

    - Stimulus: discrete direction index in [0, num_dirs-1]
    - Category: computed from direction index (default: equal-sized bins)
    - Output: categorical labels via 'label' (1..num_categories) in Response phase
    """

    def __init__(
        self,
        t_fixation=300.0 * u.ms,
        t_sample=500.0 * u.ms,
        t_delay=1000.0 * u.ms,
        t_response=500.0 * u.ms,
        num_dirs: int = 8,
        num_categories: int = 2,
        noise_sigma: Data = 0.0 * u.ms**0.5,
        base_value: float = 0.0,
        feature_per_direction: int = 1,
        IfVon: bool = True,
        kappa: float = 2.0,
        cue_dim: int = 2,
        non_response_cue=None,
        response_cue=None,
        # optional custom mapping: fn(idx:int)->cat:int in [0, num_categories-1]
        category_fn: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ):
        self.t_fixation = t_fixation
        self.t_sample = t_sample
        self.t_delay = t_delay
        self.t_response = t_response

        self.num_dirs = int(num_dirs)
        self.num_categories = int(num_categories)
        if self.num_dirs <= 0 or self.num_categories <= 1:
            raise ValueError("num_dirs must be >0 and num_categories must be >1.")

        self.noise_sigma = noise_sigma
        self.base_value = float(base_value)
        self.feature_per_direction = int(feature_per_direction)
        self.IfVon = bool(IfVon)
        self.kappa = float(kappa)

        # cue vectors (align with your DDR/IDR)
        self.cue_dim = int(cue_dim)
        if non_response_cue is None:
            non_response_cue = [1.0] + [0.0] * (self.cue_dim - 1)
        if response_cue is None:
            response_cue = [0.0] + [1.0] + [0.0] * (self.cue_dim - 2)
        self.non_response_cue = jnp.asarray(non_response_cue, jnp.float32)
        self.response_cue = jnp.asarray(response_cue, jnp.float32)
        if self.non_response_cue.shape != (self.cue_dim,) or self.response_cue.shape != (self.cue_dim,):
            raise ValueError("Cue vectors must have shape (cue_dim,)")

        # stimulus encoder
        if self.IfVon:
            # feature.num == num_dirs * feature_per_direction
            self.sample_encoder = von_mises(
                "sample_idx", kappa=self.kappa, base_value=self.base_value,
                as_index=True, num_dirs=self.num_dirs,
            )
        else:
            # feature.num == 2 * feature_per_direction
            self.sample_encoder = cos_sin(
                "sample_idx",
                num_dirs=self.num_dirs,
                repeats=self.feature_per_direction,
                base_value=self.base_value,
                map_to_01=False,
            )

        # category mapping
        if category_fn is None:
            # equal bins (works even if not divisible)
            self.category_fn = lambda idx: jnp.floor(
                jnp.asarray(idx, jnp.float32) * (self.num_categories / float(self.num_dirs))
            ).astype(jnp.int32).clip(0, self.num_categories - 1)
        else:
            self.category_fn = category_fn

        super().__init__(**kwargs)

    def define_features(self):
        # INPUT
        fix_in = Feature(self.cue_dim, "fixation")
        if self.IfVon:
            stim_in = Feature(self.num_dirs * self.feature_per_direction, "stimulus")
        else:
            stim_in = Feature(2 * self.feature_per_direction, "stimulus")
        input_features = fix_in + stim_in

        # OUTPUT: categorical classification via label; keep response feature sized num_categories
        # (This mirrors many other tasks in this file: output_features include response dims even if label drives training.)
        fix_out = Feature(1, "fixation_out")
        resp_out = Feature(self.num_categories, "response")
        output_features = fix_out + resp_out
        return input_features, output_features

    def define_phases(self):
        return concat([
            Fixation(
                duration=self.t_fixation,
                inputs={"fixation": self.non_response_cue},
                outputs={"label": 0},
            ),
            Sample(
                duration=self.t_sample,
                inputs={"fixation": self.non_response_cue, "stimulus": self.sample_encoder},
                outputs={"label": 0},
                noise={"stimulus": self.noise_sigma},
            ),
            Delay(
                duration=self.t_delay,
                inputs={"fixation": self.non_response_cue},
                outputs={"label": 0},
            ),
            Response(
                duration=self.t_response,
                inputs={"fixation": self.response_cue},
                # label in [1..num_categories]
                outputs={"label": lambda ctx, f: ctx["category"] + 1},
            ),
        ])

    def trial_init(self, ctx: Context):
        ctx["sample_idx"] = ctx.rng.choice(self.num_dirs)
        ctx["category"] = self.category_fn(ctx["sample_idx"])

        # optional convenience field
        ctx["ground_truth"] = ctx["category"]

    def get_trial_meta(self, trial_state):
        # return both for analysis
        return trial_state["sample_idx"], trial_state["category"]
    

class ImmediateDirectionClassification(Task):
    """
    Immediate Direction Classification (IDC).

    Fixation >> Response(go + stimulus) with no delay.
    """

    def __init__(
        self,
        t_fixation=300.0 * u.ms,
        t_go=500.0 * u.ms,
        num_dirs: int = 8,
        num_categories: int = 2,
        noise_sigma: Data = 0.0 * u.ms**0.5,
        base_value: float = 0.0,
        feature_per_direction: int = 1,
        IfVon: bool = True,
        kappa: float = 2.0,
        cue_dim: int = 2,
        non_response_cue=None,
        response_cue=None,
        category_fn: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ):
        self.t_fixation = t_fixation
        self.t_go = t_go

        self.num_dirs = int(num_dirs)
        self.num_categories = int(num_categories)
        if self.num_dirs <= 0 or self.num_categories <= 1:
            raise ValueError("num_dirs must be >0 and num_categories must be >1.")

        self.noise_sigma = noise_sigma
        self.base_value = float(base_value)
        self.feature_per_direction = int(feature_per_direction)
        self.IfVon = bool(IfVon)
        self.kappa = float(kappa)

        self.cue_dim = int(cue_dim)
        if non_response_cue is None:
            non_response_cue = [1.0] + [0.0] * (self.cue_dim - 1)
        if response_cue is None:
            response_cue = [0.0] + [1.0] + [0.0] * (self.cue_dim - 2)
        self.non_response_cue = jnp.asarray(non_response_cue, jnp.float32)
        self.response_cue = jnp.asarray(response_cue, jnp.float32)
        if self.non_response_cue.shape != (self.cue_dim,) or self.response_cue.shape != (self.cue_dim,):
            raise ValueError("Cue vectors must have shape (cue_dim,)")

        # stimulus encoder (same as DDC)
        if self.IfVon:
            self.sample_encoder = von_mises(
                "sample_idx", kappa=self.kappa, base_value=self.base_value,
                as_index=True, num_dirs=self.num_dirs,
            )
        else:
            self.sample_encoder = cos_sin(
                "sample_idx",
                num_dirs=self.num_dirs,
                repeats=self.feature_per_direction,
                base_value=self.base_value,
                map_to_01=False,
            )

        # category mapping
        if category_fn is None:
            self.category_fn = lambda idx: jnp.floor(
                jnp.asarray(idx, jnp.float32) * (self.num_categories / float(self.num_dirs))
            ).astype(jnp.int32).clip(0, self.num_categories - 1)
        else:
            self.category_fn = category_fn

        super().__init__(**kwargs)

    def define_features(self):
        # INPUT
        fix_in = Feature(self.cue_dim, "fixation")
        if self.IfVon:
            stim_in = Feature(self.num_dirs * self.feature_per_direction, "stimulus")
        else:
            stim_in = Feature(2 * self.feature_per_direction, "stimulus")
        input_features = fix_in + stim_in

        # OUTPUT: categorical via label
        fix_out = Feature(1, "fixation_out")
        resp_out = Feature(self.num_categories, "response")
        output_features = fix_out + resp_out
        return input_features, output_features

    def define_phases(self):
        return concat([
            Fixation(
                duration=self.t_fixation,
                inputs={"fixation": self.non_response_cue},
                outputs={"label": 0},
            ),
            Response(
                duration=self.t_go,
                inputs={"fixation": self.response_cue, "stimulus": self.sample_encoder},
                noise={"stimulus": self.noise_sigma},
                outputs={"label": lambda ctx, f: ctx["category"] + 1},
            ),
        ])

    def trial_init(self, ctx: Context):
        ctx["sample_idx"] = ctx.rng.choice(self.num_dirs)
        ctx["category"] = self.category_fn(ctx["sample_idx"])
        ctx["ground_truth"] = ctx["category"]

    def get_trial_meta(self, trial_state):
        return trial_state["sample_idx"], trial_state["category"]