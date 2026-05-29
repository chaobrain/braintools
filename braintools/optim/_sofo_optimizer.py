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

# SOFO ("Second-Order Forward-mode Optimization") optimizers.
# Modified from https://github.com/hennequin-lab/SOFO

from functools import wraps
from typing import Callable, Optional, Sequence, Union

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import optax
from brainstate.transform import GradientTransform

from ._optax_optimizer import OptaxOptimizer

__all__ = [
    'SOFO',
    'SOFOScan',
]


# ---------------------------------------------------------------------------
# math helpers (moved from brainstate.transform._grad_sofo, RNG via brainstate.random)
# ---------------------------------------------------------------------------

def _batch_jvp(f, W, M, has_aux=False):
    _jvp = lambda s: jax.jvp(f, (W,), (s,), has_aux=has_aux)
    return jax.vmap(_jvp)(M)


def _ggn_ce(tangents, h):
    """Generalised Gauss-Newton matrix for cross-entropy loss. ``tangents`` size (k, dim)."""
    Jgh = (tangents @ h)[:, None]
    return (tangents * h) @ tangents.T - Jgh @ Jgh.T  # (k, k)


def _ggn_mse(tangents):
    """Generalised Gauss-Newton matrix for mean-squared loss. ``tangents`` size (k, dim)."""
    return tangents @ tangents.T


def _tree_random_split(rng_key, target):
    """Split a key into one key per leaf of ``target`` (key management only)."""
    treedef = jax.tree.structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree.unflatten(treedef, keys)


def _sample_v(tangent_size, params, rng):
    """Sample a batch of globally-normalized tangent vectors matching ``params``.

    Each leaf becomes shape ``(tangent_size, *leaf.shape)``; the batch is normalized by the
    global L2 norm tangent-wise. Uses ``brainstate.random`` for number generation.
    """
    v = jax.tree.map(
        lambda x, k: brainstate.random.randn(tangent_size, *x.shape, key=k, dtype=x.dtype),
        params,
        _tree_random_split(rng, params),
    )
    l2 = jnp.sqrt(sum(jax.tree.leaves(jax.vmap(lambda v: jax.tree.map(lambda x: jnp.sum(jnp.square(x)), v))(v))))
    v = jax.tree.map(lambda x: jax.vmap(lambda a, b: a / b)(x, l2), v)
    return v


def _warp_grad_fn(fn, argnums, args, kwargs):
    """Partial-apply ``fn`` fixing all args except the one at ``argnums``.

    Reimplemented locally so braintools does not import brainstate private internals.
    """
    args = tuple(args)
    if isinstance(argnums, int):
        @wraps(fn)
        def new_fn(dyn_args):
            new_args = list(args)
            new_args[argnums] = dyn_args
            return fn(*new_args, **kwargs)

        assert argnums < len(args), f"argnum {argnums} is out of range {len(args)}"
        return new_fn, args[argnums]
    else:
        @wraps(fn)
        def new_fn(dyn_args):
            assert len(dyn_args) == len(argnums)
            new_args = list(args)
            for i, argnum in enumerate(argnums):
                new_args[argnum] = dyn_args[i]
            return fn(*new_args, **kwargs)

        argnums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        params = []
        for i in argnums:
            assert i < len(args), f"argnum {i} is out of range {len(args)}"
            params.append(args[i])
        return new_fn, params


# ---------------------------------------------------------------------------
# SOFO direction implementation (the `transform=` callable for GradientTransform)
# ---------------------------------------------------------------------------

def _sofo_grad_impl(
    fn: Callable,
    loss_fn: Callable,
    argnums: Union[int, Sequence[int]] = 0,
    has_aux: bool = False,
    return_loss: bool = False,
    tangent_size: int = 100,
    damping: float = 1e-5,
    loss: str = 'mse',
    key=None,
) -> Callable:
    """Forward pass computing the SOFO search direction. Returns ``(h, aux)`` for GradientTransform."""

    def wrapper(*args, **kwargs):
        f_partial, params = _warp_grad_fn(fn, argnums, args, kwargs)
        v = _sample_v(tangent_size, params, brainstate.random.split_key() if key is None else key)

        # tangents_out shape: (tangent_size, batch, out)
        res = _batch_jvp(f_partial, params, v, has_aux=has_aux)
        if has_aux:
            outs, tangents_out, aux = res
            aux = jax.tree.map(lambda x: x[0], aux)
        else:
            outs, tangents_out = res
        losses, vg = _batch_jvp(loss_fn, outs[0], tangents_out)

        if loss == 'mse':
            vg_gv = u.math.mean(jax.vmap(_ggn_mse, in_axes=1)(tangents_out), axis=0)
        elif loss == 'ce':
            vg_gv = u.math.mean(
                jax.vmap(_ggn_ce, in_axes=(1, 0))(tangents_out, jax.nn.softmax(outs[0], axis=-1)), axis=0
            )
        else:
            raise ValueError(f'Unknown loss function: {loss}.')

        u_, s_, _ = jnp.linalg.svd(vg_gv)
        damped_s = s_ + damping * jnp.max(s_)
        vggv_vg = (u_ / damped_s) @ (u_.T @ vg)
        h = jax.tree.map(lambda v_: jnp.einsum('i,i...->...', vggv_vg, v_), v)
        if return_loss:
            return ((h, losses[0]), aux) if has_aux else (h, losses[0])
        else:
            return (h, aux) if has_aux else h

    return wrapper


# ---------------------------------------------------------------------------
# SOFO optimizer
# ---------------------------------------------------------------------------

class SOFO(OptaxOptimizer):
    r"""Second-Order Forward-mode Optimization (SOFO) optimizer.

    SOFO computes its own search direction by sampling random tangent vectors, taking
    forward-mode JVPs through ``model`` and ``loss_fn``, building a Generalised Gauss-Newton
    matrix in the random subspace, solving a damped linear system, and projecting back to
    parameter space. The resulting direction is applied via an SGD-style optax update (so
    learning-rate scheduling, momentum, weight decay, and gradient clipping all work).

    Parameters
    ----------
    model : callable
        The network, called as ``model(inputs)`` returning predictions. Its trainable
        parameters are the ``brainstate.ParamState`` objects registered via
        :meth:`register_trainable_weights`.
    loss_fn : callable
        ``loss_fn(predictions, targets) -> scalar``.
    lr : float or LRScheduler, default 1e-3
        Learning rate.
    loss : {'mse', 'ce'}, default 'mse'
        Selects the Generalised Gauss-Newton form.
    tangent_size : int, default 100
        Number of random tangents / subspace dimension.
    damping : float, default 1e-5
        Damping on the GGN, scaled by the largest singular value.
    momentum : float, default 0.0
        Momentum for the SGD-style update.
    nesterov : bool, default False
        Whether to use Nesterov momentum.
    weight_decay : float, default 0.0
        Decoupled weight decay.
    grad_clip_norm : float, optional
        Clip the SOFO direction by global norm before the update.
    grad_clip_value : float, optional
        Clip the SOFO direction by value before the update.
    key : jax PRNG key, optional
        Random key for tangent sampling. Defaults to ``brainstate.random.split_key()`` each step.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>> import jax.numpy as jnp
        >>>
        >>> class MLP(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.l1 = brainstate.nn.Linear(8, 16)
        ...         self.l2 = brainstate.nn.Linear(16, 3)
        ...     def __call__(self, x):
        ...         import jax
        ...         return self.l2(jax.nn.relu(self.l1(x)))
        >>>
        >>> model = MLP()
        >>> loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)
        >>> opt = braintools.optim.SOFO(model, loss_fn, lr=1e-2, tangent_size=64)
        >>> opt.register_trainable_weights(model.states(brainstate.ParamState))
        >>> loss = opt.step(jnp.ones((4, 8)), jnp.zeros((4, 3)))  # doctest: +SKIP
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        model: Callable,
        loss_fn: Callable,
        lr: float = 1e-3,
        loss: str = 'mse',
        tangent_size: int = 100,
        damping: float = 1e-5,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
        key=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.loss = loss
        self.tangent_size = tangent_size
        self.damping = damping
        self.momentum = momentum
        self.nesterov = nesterov
        self.key = key
        self._grad_states = None
        super().__init__(
            tx=None, lr=lr, weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm, grad_clip_value=grad_clip_value,
        )

    def default_tx(self):
        transforms = []
        if self.grad_clip_norm is not None:
            transforms.append(optax.clip_by_global_norm(self.grad_clip_norm))
        if self.grad_clip_value is not None:
            transforms.append(optax.clip(self.grad_clip_value))
        if self.momentum > 0:
            transforms.append(optax.trace(decay=self.momentum, nesterov=self.nesterov))
        if self.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(self.weight_decay))
        transforms.append(optax.scale_by_schedule(self._lr_scheduler))
        return optax.chain(*transforms)

    def register_trainable_weights(self, param_states):
        super().register_trainable_weights(param_states)
        self._grad_states = self.param_states.to_pytree()
        return self

    def _make_grad_fn(self, targets):
        step_loss_fn = lambda preds: self.loss_fn(preds, targets)
        return GradientTransform(
            target=self.model,
            transform=_sofo_grad_impl,
            grad_states=self._grad_states,
            argnums=None,
            return_value=True,
            has_aux=False,
            transform_params=dict(
                loss=self.loss, tangent_size=self.tangent_size,
                damping=self.damping, loss_fn=step_loss_fn, key=self.key,
            ),
        )

    def _compute_direction(self, inputs, targets):
        """Return ``(grads, predictions)`` without applying the update (for tests)."""
        if self.opt_state is None:
            raise ValueError("register_trainable_weights must be called before step.")
        return self._make_grad_fn(targets)(inputs)

    def step(self, inputs, targets):
        grads, predictions = self._compute_direction(inputs, targets)
        super().step(grads)
        return self.loss_fn(predictions, targets)

    def update(self, inputs, targets):
        return self.step(inputs, targets)


# ---------------------------------------------------------------------------
# SOFOScan: recurrent SOFO optimizer (stateful one-step Module)
# ---------------------------------------------------------------------------

def _collapse_time(a):
    """Collapse the leading (time, batch) axes of ``a`` into a single sample axis."""
    return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])


class SOFOScan(OptaxOptimizer):
    r"""Recurrent Second-Order Forward-mode Optimization (SOFO) optimizer.

    Like :class:`SOFO`, but for a stateful one-step recurrent Module. The model is scanned over
    the input sequence with the hidden state ("latent") carried explicitly; forward-mode JVPs
    propagate the tangents through time automatically (``jax.jvp`` through ``lax.scan``), so the
    Generalised Gauss-Newton matrix is accumulated over every (timestep, batch) sample and solved
    once. The resulting direction is applied via the same SGD-style optax update as :class:`SOFO`.

    Parameters
    ----------
    rnn_cell : callable
        Stateful Module called as ``rnn_cell(latent, inputs) -> (new_latent, output)``, using
        ``brainstate.ParamState`` objects internally for its trainable weights.
    loss_fn : callable
        ``loss_fn(predictions, targets) -> scalar``, where both arguments have their leading
        ``(time, batch)`` axes collapsed into a single sample axis.
    lr : float or LRScheduler, default 1e-3
        Learning rate.
    loss : {'mse', 'ce'}, default 'mse'
        Selects the Generalised Gauss-Newton form.
    tangent_size : int, default 100
        Number of random tangents / subspace dimension.
    damping : float, default 1e-5
        Damping on the GGN, scaled by the largest singular value.
    momentum : float, default 0.0
        Momentum for the SGD-style update.
    nesterov : bool, default False
        Whether to use Nesterov momentum.
    weight_decay : float, default 0.0
        Decoupled weight decay.
    grad_clip_norm : float, optional
        Clip the SOFO direction by global norm before the update.
    grad_clip_value : float, optional
        Clip the SOFO direction by value before the update.
    key : jax PRNG key, optional
        Random key for tangent sampling. Defaults to ``brainstate.random.split_key()`` each step.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>> import jax.numpy as jnp
        >>>
        >>> class Cell(brainstate.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.wh = brainstate.nn.Linear(4, 4)
        ...         self.wx = brainstate.nn.Linear(3, 4)
        ...         self.wo = brainstate.nn.Linear(4, 2)
        ...     def __call__(self, latent, inp):
        ...         new_latent = self.wh(latent) + self.wx(inp)
        ...         return new_latent, self.wo(new_latent)
        >>>
        >>> cell = Cell()
        >>> loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)
        >>> opt = braintools.optim.SOFOScan(cell, loss_fn, lr=1e-2, tangent_size=64)
        >>> opt.register_trainable_weights(cell.states(brainstate.ParamState))
        >>> xs = jnp.ones((5, 4, 3)); ys = jnp.zeros((5, 4, 2)); z0 = jnp.zeros((4, 4))
        >>> loss = opt.step(z0, (xs, ys))  # doctest: +SKIP
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        rnn_cell: Callable,
        loss_fn: Callable,
        lr: float = 1e-3,
        loss: str = 'mse',
        tangent_size: int = 100,
        damping: float = 1e-5,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        grad_clip_value: Optional[float] = None,
        key=None,
    ):
        self.rnn_cell = rnn_cell
        self.loss_fn = loss_fn
        self.loss = loss
        self.tangent_size = tangent_size
        self.damping = damping
        self.momentum = momentum
        self.nesterov = nesterov
        self.key = key
        self._grad_states = None
        super().__init__(
            tx=None, lr=lr, weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm, grad_clip_value=grad_clip_value,
        )

    # reuse SOFO's SGD-style chain (same hyperparameters)
    default_tx = SOFO.default_tx

    def register_trainable_weights(self, param_states):
        super().register_trainable_weights(param_states)
        self._grad_states = self.param_states.to_pytree()
        return self

    def _scan_model(self, inputs_seq, z_init):
        def body(latent, inp):
            new_latent, output = self.rnn_cell(latent, inp)
            return new_latent, output

        _, outs = brainstate.transform.scan(body, z_init, inputs_seq)  # outs: (T, batch, ...)
        return _collapse_time(outs)  # (T * batch, ...)

    def _make_grad_fn(self, labels_seq):
        flat_targets = _collapse_time(labels_seq)
        step_loss_fn = lambda preds: self.loss_fn(preds, flat_targets)
        return GradientTransform(
            target=self._scan_model,
            transform=_sofo_grad_impl,
            grad_states=self._grad_states,
            argnums=None,
            return_value=True,
            has_aux=False,
            transform_params=dict(
                loss=self.loss, tangent_size=self.tangent_size,
                damping=self.damping, loss_fn=step_loss_fn, key=self.key,
            ),
        )

    def _compute_direction(self, z_init, batch):
        """Return ``(grads, predictions_flat)`` without applying the update (for tests)."""
        if self.opt_state is None:
            raise ValueError("register_trainable_weights must be called before step.")
        inputs_seq, labels_seq = batch
        return self._make_grad_fn(labels_seq)(inputs_seq, z_init)

    def step(self, z_init, batch):
        inputs_seq, labels_seq = batch
        grads, predictions = self._compute_direction(z_init, batch)
        super().step(grads)
        return self.loss_fn(predictions, _collapse_time(labels_seq))

    def update(self, z_init, batch):
        return self.step(z_init, batch)
