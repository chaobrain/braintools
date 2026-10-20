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

"""Vectorised multi-start minimisation of a pure, differentiable function.

:func:`multistart_minimize` runs one local optimiser (L-BFGS, Adam or gradient
descent from ``optax``) from ``S`` starting points at once: the per-start
``lax.while_loop`` is ``vmap``-ped and compiled into a single XLA program, and
every start stops on its own convergence test.
"""

from typing import Any, Callable, NamedTuple, Optional, Union

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import optax

from braintools._misc import set_module_as

__all__ = [
    'MultiStartResult',
    'multistart_minimize',
    'multistart_init',
]

_METHODS = ('lbfgs', 'adam', 'gd')

# Starting points are pulled this fraction of the interval width inside the
# bounds before the bijector is inverted. A start on a bound would otherwise map
# to an infinite unconstrained value; 1e-3 also keeps the bijector's slope (about
# 1e-3 of the width) large enough for the start to move.
_EDGE = 1e-3


class MultiStartResult(NamedTuple):
    """Result of :func:`multistart_minimize`.

    Attributes
    ----------
    x : PyTree
        Final point of every start; leaves have a leading axis of size ``S``.
    fun : jax.Array
        Final loss of every start, shape ``(S,)``.
    converged : jax.Array
        Boolean ``(S,)``: the start met the convergence test (and its loss is
        finite) before ``max_iter`` iterations.
    n_iter : jax.Array
        Integer ``(S,)``: iterations run by every start.
    best_x : PyTree
        The point of the start with the lowest finite loss (no leading axis).
    best_fun : jax.Array
        Loss of ``best_x`` (a scalar).
    best_index : jax.Array
        Index of the best start (an integer scalar).
    history : jax.Array
        Loss after every iteration, shape ``(S, max_iter)``. Entries after a
        start stopped repeat its final loss.
    """
    x: Any
    fun: jax.Array
    converged: jax.Array
    n_iter: jax.Array
    best_x: Any
    best_fun: jax.Array
    best_index: jax.Array
    history: jax.Array


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_pair(x: Any) -> bool:
    """A ``(low, high)`` bound pair (a 2-tuple of non-container values)."""
    return (
        isinstance(x, tuple) and len(x) == 2
        and not any(isinstance(e, (tuple, list, dict)) for e in x)
    )


def _is_concrete(x: Any) -> bool:
    return not isinstance(u.get_mantissa(x), jax.core.Tracer)


def _bound_mantissa(value: Any, unit: Optional[u.Unit], name: str) -> Any:
    """Express a bound in ``unit`` (the unit of its leaf) as a plain array."""
    if value is None:
        return None
    if unit is None:
        if isinstance(value, u.Quantity) and not value.dim.is_dimensionless:
            raise TypeError(
                f'{name} carries the unit {value.unit}, but the parameter it bounds has no unit.'
            )
        return jnp.asarray(u.get_mantissa(value))
    try:
        return jnp.asarray(u.Quantity(value).to_decimal(unit))
    except u.UnitMismatchError as e:
        raise TypeError(f'{name} must be expressed in a unit compatible with {unit}: {e}') from None


def _split_leaves(tree: Any):
    """Flatten ``tree`` into plain mantissa leaves, their units, and the treedef."""
    leaves, treedef = jax.tree.flatten(tree, is_leaf=u.math.is_quantity)
    mantissas, units = [], []
    for leaf in leaves:
        if isinstance(leaf, u.Quantity):
            mantissas.append(jnp.asarray(leaf.mantissa))
            units.append(leaf.unit)
        else:
            mantissas.append(jnp.asarray(leaf))
            units.append(None)
    return mantissas, units, treedef


def _join_leaves(mantissas, units, treedef):
    leaves = [m if unit is None else u.Quantity(m, unit) for m, unit in zip(mantissas, units)]
    return jax.tree.unflatten(treedef, leaves)


def _make_bijector(bound: Any, unit: Optional[u.Unit], index: int):
    """Return the brainstate transform for one bound entry, or ``None``."""
    if bound is None:
        return None
    if not _is_pair(bound):
        raise ValueError(
            f'Each bounds entry must be None or a (low, high) pair; got {bound!r} for leaf {index}.'
        )
    low = _bound_mantissa(bound[0], unit, f'Lower bound of leaf {index}')
    high = _bound_mantissa(bound[1], unit, f'Upper bound of leaf {index}')
    for side, v in (('lower', low), ('upper', high)):
        if v is not None and _is_concrete(v) and not np.all(np.isfinite(np.asarray(v))):
            raise ValueError(
                f'The {side} bound of leaf {index} must be finite; use None for an open side.'
            )
    if low is None and high is None:
        return None
    if high is None:
        return brainstate.nn.SoftplusT(low)
    if low is None:
        return brainstate.nn.NegSoftplusT(high)
    if _is_concrete(low) and _is_concrete(high) and np.any(np.asarray(low) >= np.asarray(high)):
        raise ValueError(f'Bounds of leaf {index} must satisfy low < high; got {bound!r}.')
    return brainstate.nn.SigmoidT(low, high)


def _pull_inside(x: jax.Array, t: Any) -> jax.Array:
    """Move a constrained value strictly inside the domain of transform ``t``."""
    if isinstance(t, brainstate.nn.SigmoidT):
        pad = _EDGE * t.width
        return jnp.clip(x, t.lower + pad, t.lower + t.width - pad)
    if isinstance(t, brainstate.nn.NegSoftplusT):
        return jnp.minimum(x, t.lower - _EDGE * jnp.maximum(1.0, jnp.abs(t.lower)))
    # SoftplusT
    return jnp.maximum(x, t.lower + _EDGE * jnp.maximum(1.0, jnp.abs(t.lower)))


def _resolve_bijectors(bounds: Any, units, treedef):
    if bounds is None:
        return [None] * len(units)
    if treedef.num_leaves == 1 and _is_pair(bounds):
        entries = [bounds]
    else:
        try:
            entries = treedef.flatten_up_to(bounds)
        except (ValueError, TypeError) as e:
            raise ValueError(
                '`bounds` must mirror the structure of `x0`, with a (low, high) pair '
                f'or None in place of every leaf: {e}'
            ) from None
    return [_make_bijector(b, unit, i) for i, (b, unit) in enumerate(zip(entries, units))]


def _make_optimizer(method: Any, learning_rate: Any):
    if isinstance(method, str):
        name = method.lower()
        if name == 'lbfgs':
            return optax.lbfgs(), True
        if name == 'adam':
            return optax.adam(learning_rate), False
        if name == 'gd':
            return optax.sgd(learning_rate), False
        raise ValueError(f'Unknown method {method!r}; expected one of {_METHODS} '
                         f'or an optax.GradientTransformation.')
    if isinstance(method, optax.GradientTransformation):
        return optax.with_extra_args_support(method), False
    raise TypeError(
        f'`method` must be one of {_METHODS} or an optax.GradientTransformation, '
        f'got {type(method).__name__}.'
    )


def _scalar_loss(value: Any) -> jax.Array:
    if isinstance(value, u.Quantity):
        if not value.dim.is_dimensionless:
            raise TypeError(f'`fun` must return a dimensionless scalar, got unit {value.unit}.')
        value = value.mantissa
    value = jnp.asarray(value)
    if value.ndim != 0:
        raise ValueError(f'`fun` must return a scalar, got shape {value.shape}.')
    return value


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

@set_module_as('braintools.optim')
def multistart_init(key: Any, bounds: Any, n_starts: int) -> Any:
    """Draw uniform starting points within box bounds.

    Parameters
    ----------
    key : jax.Array
        A ``jax.random`` key.
    bounds : PyTree
        A pytree whose leaves are ``(low, high)`` pairs (scalars, arrays or
        ``brainunit`` Quantities of a common unit). A bare pair describes a
        single leaf.
    n_starts : int
        Number of starting points ``S``.

    Returns
    -------
    PyTree
        The structure of ``bounds`` with every pair replaced by an array of
        shape ``(S, *broadcast(low, high).shape)`` drawn uniformly from
        ``[low, high)``; a Quantity in the unit of ``low`` when the bounds
        carry one.

    Raises
    ------
    ValueError
        If ``n_starts`` is not a positive integer.

    See Also
    --------
    multistart_minimize : Minimise from the drawn starts.

    Notes
    -----
    ``n_starts`` fixes the output shape, so it must be a Python ``int``. The
    function works under ``jax.jit`` (with ``n_starts`` static) and
    ``jax.vmap`` over ``key``.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> from braintools.optim import multistart_init
        >>> x0 = multistart_init(jax.random.PRNGKey(0), {'a': (0., 1.), 'b': (-2., 2.)}, 4)
        >>> x0['a'].shape, x0['b'].shape
        ((4,), (4,))
        >>> bool(((x0['b'] >= -2.) & (x0['b'] < 2.)).all())
        True
    """
    if isinstance(n_starts, bool) or not isinstance(n_starts, (int, np.integer)) or n_starts < 1:
        raise ValueError(f'`n_starts` must be a positive integer, got {n_starts!r}.')
    pairs, treedef = jax.tree.flatten(bounds, is_leaf=_is_pair)
    if not pairs:
        raise ValueError('`bounds` contains no (low, high) pair.')
    keys = jax.random.split(key, len(pairs))
    out = []
    for k, pair in zip(keys, pairs):
        if not _is_pair(pair):
            raise ValueError(f'Every leaf of `bounds` must be a (low, high) pair, got {pair!r}.')
        low = u.Quantity(pair[0])
        unit = low.unit
        lo = jnp.asarray(low.mantissa, dtype=brainstate.environ.dftype())
        hi = jnp.asarray(u.Quantity(pair[1]).to_decimal(unit), dtype=lo.dtype)
        shape = (int(n_starts),) + jnp.broadcast_shapes(lo.shape, hi.shape)
        draw = jax.random.uniform(k, shape, dtype=lo.dtype, minval=lo, maxval=hi)
        out.append(draw if unit.is_unitless else u.Quantity(draw, unit))
    return jax.tree.unflatten(treedef, out)


@set_module_as('braintools.optim')
def multistart_minimize(
    fun: Callable[[Any], Any],
    x0: Any,
    *,
    method: Union[str, optax.GradientTransformation] = 'lbfgs',
    max_iter: int = 500,
    tol: float = 1e-6,
    bounds: Any = None,
    learning_rate: Any = 1e-2,
) -> MultiStartResult:
    r"""Minimise a scalar function from many starting points in parallel.

    Every start runs the same local optimiser inside a ``lax.while_loop``;
    the loop is ``vmap``-ped over the starts and the whole run is compiled
    into one XLA program. A start stops when it converges, when its loss
    becomes non-finite, or after ``max_iter`` iterations, while the other
    starts keep going.

    Parameters
    ----------
    fun : callable
        ``fun(x) -> scalar``. A pure, differentiable function of one start
        ``x`` (a pytree shaped like a single slice of ``x0``). Its value must
        be a plain or dimensionless scalar.
    x0 : PyTree
        Starting points: every leaf has a leading axis of size ``S``, the
        number of starts. Leaves may be ``brainunit`` Quantities; ``fun`` then
        receives Quantities in the same units.
    method : {'lbfgs', 'adam', 'gd'} or optax.GradientTransformation, default 'lbfgs'
        Local optimiser. ``'lbfgs'`` is :func:`optax.lbfgs` with its default
        zoom line search (``learning_rate`` is ignored); ``'adam'`` and
        ``'gd'`` are :func:`optax.adam` and :func:`optax.sgd` with step size
        ``learning_rate``. Any other ``optax`` transformation can be passed
        directly.
    max_iter : int, default 500
        Maximum number of iterations per start (static).
    tol : float, default 1e-6
        Convergence tolerance. A start has converged when the largest
        absolute gradient entry (in the unconstrained space) is below
        ``tol``, or when the loss changes by a relative amount
        :math:`|f_k - f_{k-1}| \le \mathrm{tol}\,|f_{k-1}|` in one
        iteration (so a step that leaves the loss unchanged also counts).
    bounds : PyTree, optional
        Box bounds mirroring the structure of ``x0``: a ``(low, high)`` pair
        or ``None`` in place of every leaf (a bare pair for a single-leaf
        ``x0``). Bounds broadcast against one start of the leaf and may be
        Quantities in the leaf's unit. ``low`` or ``high`` may be ``None`` for
        a one-sided bound. A bounded leaf is optimised in an unconstrained
        space through :class:`brainstate.nn.SigmoidT` (``SoftplusT`` /
        ``NegSoftplusT`` for one-sided bounds) and mapped back, so iterates
        never leave the box.
    learning_rate : float or optax schedule, default 1e-2
        Step size for ``'adam'`` and ``'gd'``.

    Returns
    -------
    MultiStartResult
        ``(x, fun, converged, n_iter, best_x, best_fun, best_index,
        history)``; see :class:`MultiStartResult`.

    Raises
    ------
    ValueError
        If ``method`` is unknown, ``max_iter`` is not a positive integer,
        ``x0`` has no leaves or leaves with different leading sizes, or a
        bound is malformed (wrong structure, non-finite, ``low >= high``).
    TypeError
        If ``fun`` returns a value with a physical unit, or a bound's unit
        does not match its leaf.

    See Also
    --------
    multistart_init : Uniform starting points within bounds.
    braintools.fit.Fitter : Model fitting built on this minimiser.

    Notes
    -----
    **Convergence test.** The gradient test uses the gradient with respect
    to the *constrained* value (the chain-rule gradient divided by the slope
    of the bijector), so a start near a bound, where the sigmoid is flat,
    is not mistaken for a stationary point. At an optimum on a bound that
    gradient stays non-zero, and only the loss-change test can stop the
    start. The loss-change test is used for ``'lbfgs'`` only: its line
    search makes a negligible decrease a sign of convergence, whereas the
    per-step change of a fixed-step method (``'adam'``, ``'gd'``, a custom
    ``optax`` transformation) reflects the step size rather than the
    distance to the optimum (with a large constant offset in the loss, a
    relative test would stop such a method far from the minimum). These
    methods therefore stop on the gradient test or after ``max_iter``.
    Conversely, for ``'lbfgs'`` on a loss with a large constant offset the
    relative test can stop early; lower ``tol`` (or remove the offset) then.

    **Starts on a bound.** Starting values are moved ``1e-3`` of the
    interval width inside the bounds (``1e-3 * max(1, |bound|)`` for a
    one-sided bound) before the bijector is inverted, so starts on (or
    outside) a bound are finite and still move.

    **Non-finite losses.** A start whose loss at ``x0`` is not finite stops
    immediately; a start whose loss becomes non-finite during the run
    (it *diverged*) stops and keeps its last finite point. Either way it is
    reported with ``converged=False`` and ``n_iter < max_iter``, and it is
    never selected as the best while another start has a finite loss and
    did not diverge. If every start diverged, the lowest finite loss among
    them is selected; if no loss is finite, ``best_index`` is 0 and
    ``best_fun`` is not finite. A start that already satisfies the
    convergence test at ``x0`` runs zero iterations.

    The function contains no host synchronisation, so it can itself be
    called inside ``jax.jit`` (with the keyword options static) or
    ``jax.vmap``. It is not meant to be differentiated.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.optim import multistart_minimize
        >>> def rosenbrock(p):
        ...     return (1. - p['x']) ** 2 + 100. * (p['y'] - p['x'] ** 2) ** 2
        >>> x0 = {'x': jnp.array([-1.5, 0., 2.]), 'y': jnp.array([2., -1., 1.])}
        >>> res = multistart_minimize(rosenbrock, x0, max_iter=200)
        >>> res.converged
        Array([ True,  True,  True], dtype=bool)
        >>> print(jnp.round(res.best_x['x'], 3), jnp.round(res.best_x['y'], 3))
        1.0 1.0
        >>>
        >>> res = multistart_minimize(lambda x: jnp.sin(3. * x) + 0.1 * x ** 2,
        ...                           jnp.linspace(-2.5, 2.5, 5), bounds=(-3., 3.))
        >>> print(jnp.round(res.best_x, 3))
        -0.512
    """
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError(f'`max_iter` must be a positive integer, got {max_iter!r}.')
    max_iter = int(max_iter)
    opt, is_lbfgs = _make_optimizer(method, learning_rate)

    mantissas, units, treedef = _split_leaves(x0)
    if not mantissas:
        raise ValueError('`x0` has no leaves.')
    sizes = {m.shape[0] if m.ndim else None for m in mantissas}
    if None in sizes or len(sizes) != 1:
        raise ValueError(
            'Every leaf of `x0` must have the same leading (start) axis; got shapes '
            f'{[m.shape for m in mantissas]}.'
        )
    dtype = jnp.result_type(*[m.dtype for m in mantissas], jnp.float32)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = brainstate.environ.dftype()
    mantissas = [m.astype(dtype) for m in mantissas]
    bijectors = _resolve_bijectors(bounds, units, treedef)

    def to_x(z_leaves):
        return [m if t is None else t.forward(m) for m, t in zip(z_leaves, bijectors)]

    def loss(z_leaves):
        return _scalar_loss(fun(_join_leaves(to_x(z_leaves), units, treedef))).astype(dtype)

    z0 = [m if t is None else t.inverse(_pull_inside(m, t)) for m, t in zip(mantissas, bijectors)]
    z0 = [z.astype(dtype) for z in z0]

    value_and_grad = jax.value_and_grad(loss)
    if is_lbfgs:
        cached_value_and_grad = optax.value_and_grad_from_state(loss)
    else:
        cached_value_and_grad = None
    tol_ = jnp.asarray(tol, dtype)

    def constrained_grad(z_leaves, g_leaves):
        # d f / d x = (d f / d z) / (d x / d z) for the element-wise bijectors;
        # a saturated bijector (slope 0) gives an infinite, i.e. unconverged, entry
        out = []
        for zl, gl, t in zip(z_leaves, g_leaves, bijectors):
            if t is None:
                out.append(gl)
            else:
                slope = jax.jvp(t.forward, (zl,), (jnp.ones_like(zl),))[1]
                slope = jnp.abs(u.get_mantissa(slope))
                out.append(jnp.where(slope > 0, jnp.abs(gl) / jnp.where(slope > 0, slope, 1.), jnp.inf))
        return out

    def is_converged(z, f, f_prev, g, it):
        gx = constrained_grad(z, g)
        g_inf = jnp.max(jnp.stack([jnp.max(jnp.abs(l)) for l in gx]))
        conv = jnp.isfinite(f) & (g_inf < tol_)
        if is_lbfgs:
            rel_ok = jnp.abs(f - f_prev) <= tol_ * jnp.abs(f_prev)
            conv = conv | (jnp.isfinite(f) & (it > 0) & rel_ok)
        return conv

    def single(z):
        f, g = value_and_grad(z)
        state = opt.init(z)
        hist = jnp.full((max_iter,), f, dtype)
        conv = is_converged(z, f, f, g, 0)
        diverged = ~jnp.isfinite(f)
        done = conv | diverged
        carry0 = (z, state, f, g, jnp.asarray(0, jnp.int32), done, conv, diverged, hist)

        def cond(carry):
            it, done = carry[4], carry[5]
            return (~done) & (it < max_iter)

        def body(carry):
            z, state, f, g, it, _, _, _, hist = carry
            updates, new_state = opt.update(g, state, z, value=f, grad=g, value_fn=loss)
            new_z = optax.apply_updates(z, updates)
            if is_lbfgs:
                new_f, new_g = cached_value_and_grad(new_z, state=new_state)
            else:
                new_f, new_g = value_and_grad(new_z)
            finite = jnp.isfinite(new_f)
            conv = finite & is_converged(new_z, new_f, f, new_g, it + 1)
            done = ~finite | conv
            # keep the last finite point when the loss blows up
            keep = lambda new, old: jax.tree.map(lambda a, b: jnp.where(finite, a, b), new, old)
            z_out = keep(new_z, z)
            f_out = jnp.where(finite, new_f, f)
            g_out = keep(new_g, g)
            hist = hist.at[it].set(f_out)
            return z_out, new_state, f_out, g_out, it + 1, done, conv, ~finite, hist

        z, _, f, _, it, _, converged, diverged, hist = jax.lax.while_loop(cond, body, carry0)
        # fill the iterations after the stop with the final loss
        hist = jnp.where(jnp.arange(max_iter) < it, hist, f)
        return z, f, converged, diverged, it, hist

    @jax.jit
    def run(z0):
        return jax.vmap(single)(z0)

    z, f, converged, diverged, n_iter, hist = run(z0)
    x = _join_leaves(to_x(z), units, treedef)
    finite = jnp.isfinite(f)
    eligible = finite & ~diverged
    eligible = jnp.where(jnp.any(eligible), eligible, finite)
    best = jnp.argmin(jnp.where(eligible, f, jnp.inf))
    best_x = jax.tree.map(lambda leaf: leaf[best], x, is_leaf=u.math.is_quantity)
    return MultiStartResult(
        x=x,
        fun=f,
        converged=converged,
        n_iter=n_iter,
        best_x=best_x,
        best_fun=f[best],
        best_index=best,
        history=hist,
    )
