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

"""Online learning rules for linear readouts: RLS, FORCE and LMS.

These optimizers do not take gradients. Each registered weight is a linear
readout ``outputs = inputs @ W`` of shape ``(n_in, n_out)``, and
``step(inputs, outputs, targets)`` updates it in place from one batch of
samples. The updates are the block equations of brainpy 2.x
(``brainpy.algorithms.online``), written as pure JAX so that ``step`` can be
called inside ``brainstate.transform.jit`` and ``brainstate.transform.for_loop``.
"""

import numbers
from typing import Dict, Hashable

import brainunit as u
import jax.numpy as jnp
from brainstate import State

from ._base import Optimizer, OptimState

__all__ = [
    'RLS',
    'FORCE',
    'LMS',
]


def _plain(x, name):
    """Return ``x`` as a JAX array, refusing physical units."""
    if isinstance(x, u.Quantity):
        if not x.is_unitless:
            raise u.UnitMismatchError(
                f'{name} must be dimensionless for online learning rules, got the unit '
                f'{x.unit}. Pass the mantissa explicitly.'
            )
        x = x.to_decimal()
    return jnp.asarray(x)


class _OnlineReadout(Optimizer):
    """Shared registration and dispatch for online readout learning rules."""
    __module__ = 'braintools.optim'

    def __init__(self):
        super().__init__()
        self.weight_states: Dict[Hashable, State] = {}

    # -- registration --------------------------------------------------------

    def register_trainable_weights(self, param_states: Dict[Hashable, State]):
        """Register the readout weights to train.

        Parameters
        ----------
        param_states : dict of {hashable : brainstate.State}
            Readout weights, each a ``State`` whose value has shape
            ``(n_in, n_out)``. Registering an existing key again replaces the
            weight and resets its optimizer state.

        Returns
        -------
        Optimizer
            ``self``, for chaining.

        Raises
        ------
        TypeError
            If ``param_states`` is not a dict, a value is not a ``State``, or a
            weight carries physical units.
        ValueError
            If a weight is not 2-D, or the same ``State`` is registered twice.
        """
        if not isinstance(param_states, dict):
            raise TypeError(
                f'param_states must be a dict of brainstate.State, got {type(param_states)}.'
            )
        for key, state in param_states.items():
            if not isinstance(state, State):
                raise TypeError(f'Weight {key!r} must be a brainstate.State, got {type(state)}.')
            value = state.value
            if isinstance(value, u.Quantity) and not value.is_unitless:
                raise TypeError(
                    f'Weight {key!r} carries the unit {value.unit}; online learning rules '
                    f'need dimensionless weights.'
                )
            if jnp.ndim(value) != 2:
                raise ValueError(
                    f'Weight {key!r} must have shape (n_in, n_out), got {jnp.shape(value)}.'
                )
            for other_key, other in self.weight_states.items():
                if other is state and other_key != key:
                    raise ValueError(
                        f'The State registered as {key!r} is already registered as {other_key!r}.'
                    )
            self.weight_states[key] = state
            self._init_weight_state(key, jnp.shape(value))
        return self

    def _init_weight_state(self, key, shape):
        """Create the per-weight optimizer state (none by default)."""

    # -- stepping ------------------------------------------------------------

    def _as_dict(self, x, name):
        if isinstance(x, dict):
            return x
        if len(self.weight_states) != 1:
            raise ValueError(
                f'{len(self.weight_states)} weights are registered, so {name} must be a dict '
                f'keyed like the registered weights ({list(self.weight_states)}).'
            )
        return {next(iter(self.weight_states)): x}

    def step(self, inputs, outputs, targets):
        """Update the registered readout weights from one batch of samples.

        Parameters
        ----------
        inputs : array_like or dict
            Presynaptic activity, shape ``(B, n_in)`` (or ``(n_in,)`` for a
            single sample). With several registered weights, a dict keyed like
            the weights; its keys may be a subset of them, and only those
            weights are updated.
        outputs : array_like, dict or None
            Current readout, shape ``(B, n_out)`` (or ``(n_out,)``), normally
            ``inputs @ W`` computed with the weights *before* this step. ``None``
            (or a ``None`` dict entry) computes ``inputs @ W`` here. A dict
            needs the same keys as ``inputs``.
        targets : array_like or dict
            Desired readout, same shape as ``outputs``; a dict needs the same
            keys as ``inputs``.

        Returns
        -------
        jax.Array or dict
            The a-priori error ``outputs - targets`` (shape of ``outputs``), or
            a dict of them when ``inputs`` is a dict.

        Raises
        ------
        ValueError
            If no weights are registered, a dict key is unknown, or the shapes
            do not match the weights.
        brainunit.UnitMismatchError
            If an array carries physical units.
        """
        if not self.weight_states:
            raise ValueError('register_trainable_weights must be called before step.')
        as_dict = isinstance(inputs, dict)
        inputs = self._as_dict(inputs, 'inputs')
        if outputs is None:
            outputs = {k: None for k in inputs}
        outputs = self._as_dict(outputs, 'outputs')
        targets = self._as_dict(targets, 'targets')
        errors = {}
        for key, x in inputs.items():
            if key not in self.weight_states:
                raise ValueError(f'Unknown weight {key!r}; registered: {list(self.weight_states)}.')
            if key not in outputs or key not in targets:
                raise ValueError(f'outputs and targets must contain the key {key!r}.')
            errors[key] = self._step_one(key, x, outputs[key], targets[key])
        return errors if as_dict else next(iter(errors.values()))

    def _step_one(self, key, x, y, t):
        state = self.weight_states[key]
        W = _plain(state.value, f'weight {key!r}')
        x = _plain(x, 'inputs')
        single = x.ndim == 1
        X = jnp.atleast_2d(x)
        if X.ndim != 2 or X.shape[1] != W.shape[0]:
            raise ValueError(
                f'inputs for {key!r} must have shape (B, {W.shape[0]}), got {x.shape}.'
            )
        Y = X @ W if y is None else jnp.atleast_2d(_plain(y, 'outputs'))
        T = jnp.atleast_2d(_plain(t, 'targets'))
        expected = (X.shape[0], W.shape[1])
        if Y.shape != expected or T.shape != expected:
            raise ValueError(
                f'outputs and targets for {key!r} must have shape {expected}, '
                f'got {Y.shape} and {T.shape}.'
            )
        E = Y - T
        state.value = W + self._delta(key, X, E).astype(W.dtype)
        return E[0] if single else E

    def _delta(self, key, X, E):
        raise NotImplementedError

    def update(self, inputs, outputs, targets):
        """Alias of :meth:`step` taking ``(inputs, outputs, targets)`` (not gradients)."""
        return self.step(inputs, outputs, targets)


class RLS(_OnlineReadout):
    r"""Recursive least squares (RLS) for linear readouts.

    For every registered weight :math:`W` (shape ``(n_in, n_out)``) RLS keeps
    an inverse-correlation matrix :math:`P` (shape ``(n_in, n_in)``, started at
    :math:`P_0 = I / \alpha`). One :meth:`step` on a batch of inputs
    :math:`X` (``(B, n_in)``) with error :math:`E = Y - T` applies the block
    update of brainpy 2.x [1]_ with a forgetting factor :math:`\lambda`:

    .. math::

        K &= P X^\top (\lambda I_B + X P X^\top)^{-1}, \\
        P &\leftarrow (P - K X P) / \lambda, \\
        W &\leftarrow W - K E.

    With :math:`\lambda = 1` and ``outputs = inputs @ W``, after any number of
    steps :math:`W` is the ridge solution
    :math:`\arg\min_W \|Y_{all} - X_{all} W\|^2 + \alpha \|W - W_0\|^2` over all
    samples seen so far, whatever the batch sizes: one step on a batch equals
    one step per sample in the same order.

    Parameters
    ----------
    alpha : float, default 1.0
        Regulariser (``> 0``); :math:`P_0 = I / \alpha`. Small ``alpha`` (a
        large :math:`P_0`) makes the first samples count fully and gives fast
        learning. Note that brainpy 2.x used the inverse convention
        (:math:`P_0 = \alpha I`).
    forgetting : float, default 1.0
        Forgetting factor :math:`\lambda \in (0, 1]`, applied once per
        :meth:`step` (so per batch, not per sample). Below 1, old batches are
        down-weighted by :math:`\lambda^{k}` after ``k`` further steps.

    Raises
    ------
    ValueError
        If ``alpha <= 0`` or ``forgetting`` is outside ``(0, 1]``.

    See Also
    --------
    FORCE : RLS applied to the readout of a recurrent network with feedback.
    LMS : first-order (gradient-descent) online rule.
    ridge_regression : the same solution in closed form, from all data at once.

    Notes
    -----
    :meth:`step` is jit-able (``brainstate.transform.jit``) and can be called
    inside ``brainstate.transform.for_loop``/``scan``: the weights and the
    :math:`P` matrices are ``State`` objects updated in place. Several weights
    can be registered; then ``inputs``, ``outputs`` and ``targets`` are dicts
    keyed like the weights (each weight is an independent readout of its own
    inputs). Weights and data must be dimensionless. Each step costs
    :math:`O(n_{in}^2 B + B^3)`.

    References
    ----------
    .. [1] S. Haykin, *Adaptive Filter Theory*, 4th ed., Prentice Hall, 2002.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> w_true = jnp.array([[1.], [-2.], [0.5]])
        >>> X = jax.random.normal(jax.random.PRNGKey(0), (200, 3))
        >>> T = X @ w_true
        >>> w = brainstate.ParamState(jnp.zeros((3, 1)))
        >>> opt = braintools.optim.RLS(alpha=1e-3)
        >>> _ = opt.register_trainable_weights({'readout': w})
        >>> def train(x, t):
        ...     return opt.step(x[None], None, t[None])
        >>> errors = brainstate.transform.for_loop(train, X, T)
        >>> errors.shape
        (200, 1, 1)
        >>> print(jnp.round(w.value[:, 0], 3))
        [ 1.  -2.   0.5]
    """
    __module__ = 'braintools.optim'

    def __init__(self, alpha: float = 1.0, *, forgetting: float = 1.0):
        if not isinstance(alpha, numbers.Real) or alpha <= 0:
            raise ValueError(f'alpha must be a positive number, got {alpha!r}.')
        if not isinstance(forgetting, numbers.Real) or not 0. < forgetting <= 1.:
            raise ValueError(f'forgetting must be in (0, 1], got {forgetting!r}.')
        super().__init__()
        self.alpha = float(alpha)
        self.forgetting = float(forgetting)
        self.P: Dict[Hashable, OptimState] = {}

    def _init_weight_state(self, key, shape):
        dtype = jnp.result_type(self.weight_states[key].value, jnp.float32)
        self.P[key] = OptimState(jnp.eye(shape[0], dtype=dtype) / self.alpha)

    def _delta(self, key, X, E):
        P = self.P[key].value
        X = X.astype(P.dtype)
        lam = self.forgetting
        k = P @ X.T  # (n_in, B)
        S = lam * jnp.eye(X.shape[0], dtype=P.dtype) + X @ k  # (B, B), symmetric
        gain = jnp.linalg.solve(S, k.T).T  # k S^{-1}, (n_in, B)
        P_new = (P - gain @ k.T) / lam  # k^T = X P since P is symmetric
        self.P[key].value = 0.5 * (P_new + P_new.T)
        return -gain @ E.astype(P.dtype)


class FORCE(RLS):
    r"""FORCE learning: RLS on the readout of a recurrent network with feedback.

    FORCE (first-order reduced and controlled error) learning [1]_ trains the
    linear readout :math:`z = W^\top r` of a chaotic recurrent network whose
    output is fed back into the network. The readout is updated by RLS at every
    (or every few) time steps, with a strong initial learning rate
    (:math:`P_0 = I / \alpha`, :math:`\alpha` of order 1), which keeps the
    error small from the start and suppresses the chaos. The update rule is
    exactly :class:`RLS`; this class provides the FORCE defaults and the
    protocol documented below.

    Parameters
    ----------
    alpha : float, default 1.0
        Regulariser; :math:`P_0 = I / \alpha`. Sussillo and Abbott use values
        from 1 to 100 (smaller means faster learning).
    forgetting : float, default 1.0
        Forgetting factor :math:`\lambda \in (0, 1]`, as in :class:`RLS`.

    Raises
    ------
    ValueError
        If ``alpha <= 0`` or ``forgetting`` is outside ``(0, 1]``.

    See Also
    --------
    RLS : the underlying update and the ``step`` signature.

    Notes
    -----
    Per time step: compute the rates :math:`r = \tanh(x)` and the readout
    :math:`z = W^\top r`, advance the network with :math:`z` fed back, then call
    ``step(r, z, target)``. Wrap the time loop in
    ``brainstate.transform.for_loop`` so that it compiles to a single XLA loop.

    References
    ----------
    .. [1] D. Sussillo and L. F. Abbott, "Generating coherent patterns of
       activity from chaotic neural networks," Neuron 63(4), 544-557, 2009.

    Examples
    --------
    Train a chaotic rate network (``g = 1.5``) to produce a sine wave, then let
    it run on its own feedback:

    .. code-block:: python

        >>> import brainstate
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> N, g, dt, tau = 200, 1.5, 1.0, 10.0
        >>> k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
        >>> J = g * jax.random.normal(k1, (N, N)) / jnp.sqrt(N)
        >>> w_fb = jax.random.uniform(k2, (N,), minval=-1., maxval=1.)
        >>> x = brainstate.HiddenState(0.5 * jax.random.normal(k3, (N,)))
        >>> w = brainstate.ParamState(jnp.zeros((N, 1)))
        >>> opt = braintools.optim.FORCE(alpha=1.0)
        >>> _ = opt.register_trainable_weights({'readout': w})
        >>>
        >>> def run(target, train):
        ...     r = jnp.tanh(x.value)
        ...     z = r @ w.value
        ...     x.value = x.value + dt / tau * (-x.value + J @ r + w_fb * z[0])
        ...     if train:
        ...         opt.step(r[None], z[None], target[None])
        ...     return z[0]
        >>>
        >>> t = jnp.arange(5000) * dt
        >>> target = jnp.sin(2 * jnp.pi * t / 100.)[:, None]
        >>> _ = brainstate.transform.for_loop(lambda y: run(y, True), target[:4000])
        >>> z = brainstate.transform.for_loop(lambda y: run(y, False), target[4000:])
        >>> bool(jnp.mean((z - target[4000:, 0]) ** 2) < 0.05)
        True
    """
    __module__ = 'braintools.optim'

    def __init__(self, alpha: float = 1.0, *, forgetting: float = 1.0):
        super().__init__(alpha, forgetting=forgetting)


class LMS(_OnlineReadout):
    r"""Least mean squares (LMS, Widrow-Hoff) rule for linear readouts.

    One :meth:`step` on a batch of inputs :math:`X` (``(B, n_in)``) with error
    :math:`E = Y - T` (``(B, n_out)``) applies the brainpy 2.x update [1]_

    .. math::

        W \leftarrow W - \eta \sum_{b=1}^{B} x_b e_b^\top = W - \eta X^\top E,

    a stochastic-gradient step on :math:`\tfrac12 \|E\|^2` (summed, not
    averaged, over the batch).

    Parameters
    ----------
    lr : float, default 0.1
        Learning rate :math:`\eta \ge 0`. With ``outputs = inputs @ W`` the rule
        is stable roughly for :math:`0 < \eta < 2 / \lambda_{max}`, with
        :math:`\lambda_{max}` the largest eigenvalue of the per-step input
        correlation :math:`X^\top X` (``|x|^2`` for a single sample).

    Raises
    ------
    ValueError
        If ``lr`` is negative.

    See Also
    --------
    RLS : second-order online rule (much faster convergence).

    Notes
    -----
    :meth:`step` is jit-able and usable inside
    ``brainstate.transform.for_loop``; see :class:`RLS` for the handling of
    several registered weights. The optimizer keeps no state besides the
    weights.

    References
    ----------
    .. [1] B. Widrow and M. E. Hoff, "Adaptive switching circuits," IRE WESCON
       Convention Record 4, 96-104, 1960.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> w_true = jnp.array([[1.], [-2.], [0.5]])
        >>> X = jax.random.normal(jax.random.PRNGKey(0), (2000, 3))
        >>> w = brainstate.ParamState(jnp.zeros((3, 1)))
        >>> opt = braintools.optim.LMS(lr=0.02)
        >>> _ = opt.register_trainable_weights({'readout': w})
        >>> def train(x):
        ...     return opt.step(x, None, x @ w_true)
        >>> _ = brainstate.transform.for_loop(train, X)
        >>> print(jnp.round(w.value[:, 0], 3))
        [ 1.  -2.   0.5]
    """
    __module__ = 'braintools.optim'

    def __init__(self, lr: float = 0.1):
        if not isinstance(lr, numbers.Real) or lr < 0:
            raise ValueError(f'lr must be a non-negative number, got {lr!r}.')
        super().__init__()
        self.lr = float(lr)

    def _delta(self, key, X, E):
        return -self.lr * (X.T @ E)
