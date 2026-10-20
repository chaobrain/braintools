# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Value-based (economic) decision-making task classes."""

from typing import Any, Optional, Sequence, Tuple

import brainunit as u
import jax.numpy as jnp
import numpy as np

from ..context import Context
from ..feature import Feature
from ..phase import concat, Phase, Fixation, Stimulus, Response, VariableDuration
from ..task import Task
from .._typing import Duration, Data

__all__ = [
    'EconomicDecisionMaking',
]

# Offer types of Padoa-Schioppa & Assad (2006) as used by NeuroGym, written as
# (number of drops of juice B, number of drops of juice A).
_DEFAULT_OFFERS = ((0, 1), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (6, 1), (2, 0))


class EconomicDecisionMaking(Task):
    r"""
    Economic (value-based) decision-making task.

    On each trial two offers are shown, one of juice A and one of juice B, in
    varying quantities (e.g. "1A vs 3B"); the subject must choose the offer of
    greater subjective value. Juice A is preferred: one drop of A is worth
    ``relative_value`` drops of B. This is the orbitofrontal-cortex juice
    choice paradigm of Padoa-Schioppa and Assad [1]_, as implemented by the
    ``EconomicDecisionMaking`` environment of NeuroGym [2]_ (itself adapted
    from the reward-based RNN training of Song et al. [3]_).

    Structure: Fixation >> Offer >> Decision

    Parameters
    ----------
    t_fixation : Duration
        Fixation period (default: 1500 ms).
    t_offer : Duration or tuple
        Offer period. A Quantity gives a fixed duration; a ``(min, max)``
        tuple of Quantities draws it uniformly per trial (default:
        ``(1000, 2000)`` ms, as in NeuroGym).
    t_decision : Duration
        Decision period (default: 750 ms).
    offers : sequence of (int, int)
        Offer types as ``(nB, nA)`` pairs, the number of drops of juice B and
        of juice A (default: NeuroGym's ``(0, 1), (1, 3), (1, 2), (1, 1),
        (2, 1), (3, 1), (4, 1), (6, 1), (2, 0)``).
    relative_value : float
        Value of one drop of A in drops of B, i.e. ``1A = relative_value B``
        (default: 2.2; NeuroGym's ``B_to_A = 1 / 2.2``). Must be positive.
    quantity_scale : float
        Factor applied to the drop counts in the quantity inputs (default:
        ``1 / 5``).
    noise_sigma : Data
        Standard deviation of Gaussian input noise on the quantity channels
        during the offer, in units of ``ms ** 0.5`` and scaled by
        ``1 / sqrt(dt)`` (default: 0, noise-free offers).
    **kwargs
        Forwarded to :class:`~braintools.cogtask.Task` (e.g. ``seed``,
        ``dt``).

    Raises
    ------
    ValueError
        If ``offers`` is empty, not of shape ``(K, 2)`` or negative,
        ``relative_value <= 0``, or ``t_offer`` is a malformed tuple.

    See Also
    --------
    PerceptualDecisionMaking : Two-alternative choice from sensory evidence.

    Notes
    -----
    **Inputs** (7 channels, NeuroGym's order ``fixation, a1, b1, a2, b2, n1,
    n2``): ``fixation`` (1); ``juice1`` and ``juice2`` (2 each), one-hot
    identity ``[A, B]`` of the juice in offer 1 and offer 2; ``quantity`` (2),
    ``quantity_scale`` times the drop counts ``[n1, n2]``. Which offer slot
    holds juice A is random (``juice_order`` 0: A is offer 1, 1: A is
    offer 2). The fixation input is 1 during fixation and offer and 0 during
    the decision.

    **Ground truth.** The offer values, in units of B, are

    .. math::

        V_A = \rho \, n_A, \qquad V_B = n_B,

    with :math:`\rho` = ``relative_value``. The correct choice is the offer
    of juice A if :math:`V_A > V_B` and of juice B if :math:`V_B > V_A`; a tie
    is broken at random (``tie`` is set in the trial state).

    **Labels** (categorical, 3 classes): 0 (fixate) during fixation and offer,
    ``ground_truth + 1`` during the decision, where ``ground_truth`` is the
    chosen offer slot (0 = offer 1, 1 = offer 2).

    **Trial state**: ``nA``, ``nB``, ``juice_order``, ``value_A``,
    ``value_B``, ``choose_A``, ``tie``, ``ground_truth`` and, when sampled,
    ``offer_duration`` (ms).

    With the default uniform ``t_offer`` the task runs in packed
    (variable-length) mode (see
    :meth:`~braintools.cogtask.Task.max_trial_duration`); ``batch_sample``
    runs under ``jax.jit`` and ``vmap`` either way.

    **Differences from NeuroGym.** NeuroGym's environment is trained by
    reward only (the reward is the chosen juice's value) and defines no
    ground truth; here the higher-value offer is the supervised label and
    ties are broken at random. Phase names are ``'fixation'``, ``'offer'``
    and ``'decision'`` (NeuroGym: ``'offer_on'`` for the offer). The
    optional input noise is an addition.

    References
    ----------
    .. [1] Padoa-Schioppa, C., & Assad, J. A. (2006). Neurons in the
       orbitofrontal cortex encode economic value. *Nature*, 441(7090),
       223-226.
    .. [2] Molano-Mazon, M., et al. (2022). NeuroGym: An open resource for
       developing and sharing neuroscience tasks. *PsyArXiv*.
    .. [3] Song, H. F., Yang, G. R., & Wang, X.-J. (2017). Reward-based
       training of recurrent neural networks for cognitive and value-based
       tasks. *eLife*, 6, e21492.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import braintools
        >>> task = braintools.cogtask.EconomicDecisionMaking(
        ...     seed=0, dt=10.0 * u.ms, t_offer=1000.0 * u.ms)
        >>> task.num_inputs, task.num_outputs
        (7, 3)
        >>> X, Y, info = task.sample_trial(0)
        >>> X.shape, Y.shape
        ((325, 7), (325,))
        >>> s = info['trial_state']
        >>> correct_is_A = bool(s['value_A'] > s['value_B'])
        >>> bool(s['choose_A']) == correct_is_A or bool(s['tie'])
        True
        >>> chosen_slot = int(s['juice_order']) if bool(s['choose_A']) else 1 - int(s['juice_order'])
        >>> int(Y[-1]) == chosen_slot + 1
        True
        >>>
        >>> X, Y = task.batch_sample(4)
        >>> X.shape, Y.shape
        ((325, 4, 7), (325, 4))
    """

    def __init__(
        self,
        t_fixation: Duration = 1500.0 * u.ms,
        t_offer: Any = (1000.0 * u.ms, 2000.0 * u.ms),
        t_decision: Duration = 750.0 * u.ms,
        offers: Sequence[Tuple[int, int]] = _DEFAULT_OFFERS,
        relative_value: float = 2.2,
        quantity_scale: float = 0.2,
        noise_sigma: Data = 0.0 * u.ms ** 0.5,
        **kwargs
    ):
        offers_arr: np.ndarray = np.asarray(offers, dtype=np.float32)
        if offers_arr.ndim != 2 or offers_arr.shape[0] == 0 or offers_arr.shape[1] != 2:
            raise ValueError(
                f"offers must be a non-empty sequence of (nB, nA) pairs, got shape "
                f"{offers_arr.shape}."
            )
        if np.any(offers_arr < 0):
            raise ValueError("offers must contain non-negative quantities.")
        if not float(relative_value) > 0.0:
            raise ValueError(f"relative_value must be positive, got {relative_value}.")
        if isinstance(t_offer, (tuple, list)):
            if len(t_offer) != 2:
                raise ValueError(
                    f"t_offer must be a Quantity or a (min, max) tuple, got a "
                    f"sequence of length {len(t_offer)}."
                )
            lo, hi = (float(u.Quantity(v).to_decimal(u.ms)) for v in t_offer)
            if not 0.0 < lo <= hi:
                raise ValueError(f"t_offer=(min, max) needs 0 < min <= max, got {t_offer}.")
            self._offer_range_ms: Optional[Tuple[float, float]] = (lo, hi)
        else:
            u.Quantity(t_offer).to_decimal(u.ms)  # validate the time unit
            self._offer_range_ms = None
        self.t_fixation = t_fixation
        self.t_offer = t_offer
        self.t_decision = t_decision
        self.offers = jnp.asarray(offers_arr)
        self.relative_value = float(relative_value)
        self.quantity_scale = float(quantity_scale)
        self.noise_sigma = noise_sigma
        super().__init__(**kwargs)

    def define_features(self) -> Tuple:
        fix_feat = Feature(1, 'fixation')
        input_features = (fix_feat + Feature(2, 'juice1') + Feature(2, 'juice2')
                          + Feature(2, 'quantity'))
        output_features = fix_feat + Feature(2, 'choice')
        return input_features, output_features

    @staticmethod
    def _juice_encoder(slot: int):
        """One-hot ``[A, B]`` identity of the juice in offer ``slot``."""

        def encode(ctx: Context, feature) -> jnp.ndarray:
            # juice_order 0: offer 1 is A, offer 2 is B; juice_order 1: swapped.
            is_b = jnp.where(slot == 0, ctx['juice_order'], 1 - ctx['juice_order'])
            return jnp.stack([1.0 - is_b, is_b]).astype(jnp.float32)

        encode.__name__ = f"juice{slot + 1}"
        return encode

    def _quantity_encoder(self, ctx: Context, feature) -> jnp.ndarray:
        """Scaled drop counts ``[n1, n2]`` of the two offer slots."""
        a_first = ctx['juice_order'] == 0
        n1 = jnp.where(a_first, ctx['nA'], ctx['nB'])
        n2 = jnp.where(a_first, ctx['nB'], ctx['nA'])
        return jnp.stack([n1, n2]).astype(jnp.float32) * self.quantity_scale

    def define_phases(self) -> Phase:
        offer_inputs = {
            'fixation': 1.0,
            'juice1': self._juice_encoder(0),
            'juice2': self._juice_encoder(1),
            'quantity': self._quantity_encoder,
        }
        offer_noise = {'quantity': self.noise_sigma}
        if self._offer_range_ms is None:
            offer: Phase = Stimulus(duration=self.t_offer, name='offer', inputs=offer_inputs,
                                    outputs={'label': 0}, noise=offer_noise)
        else:
            offer = VariableDuration(
                min_duration=self._offer_range_ms[0] * u.ms,
                max_duration=self._offer_range_ms[1] * u.ms,
                ctx_key='offer_duration',
                name='offer',
                inputs=offer_inputs,
                outputs={'label': 0},
                noise=offer_noise,
            )
        return concat([
            Fixation(
                duration=self.t_fixation,
                name='fixation',
                inputs={'fixation': 1.0},
                outputs={'label': 0}
            ),
            offer,
            Response(
                duration=self.t_decision,
                name='decision',
                inputs={'fixation': 0.0},
                outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}
            ),
        ])

    def trial_init(self, ctx: Context) -> None:
        offer = self.offers[ctx.rng.choice(self.offers.shape[0])]
        n_b, n_a = offer[0], offer[1]
        juice_order = ctx.rng.choice(2)
        coin = ctx.rng.choice(2)  # tie breaker, drawn on every trial
        value_a = self.relative_value * n_a
        value_b = n_b
        tie = jnp.isclose(value_a, value_b)
        choose_a = jnp.where(tie, coin == 0, value_a > value_b)
        ctx['nA'] = n_a
        ctx['nB'] = n_b
        ctx['juice_order'] = juice_order
        ctx['value_A'] = value_a
        ctx['value_B'] = value_b
        ctx['tie'] = tie
        ctx['choose_A'] = choose_a
        # Offer slot of the chosen juice: A sits in slot ``juice_order``.
        ctx['ground_truth'] = jnp.where(choose_a, juice_order, 1 - juice_order)
        if self._offer_range_ms is not None:
            ctx['offer_duration'] = ctx.rng.uniform(*self._offer_range_ms)
