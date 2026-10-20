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
Adaptive-step initial value problem solver.

:func:`solve_ivp` drives the embedded Runge–Kutta steppers of
:mod:`braintools.quad` with a PI step-size controller inside a bounded,
fixed-length ``scan``, so the solve is ``jit``/``vmap`` compatible and
reverse-mode differentiable.
"""

from typing import Any, Callable, NamedTuple, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as
from ._ode_integrator import (
    ode_bs32_step,
    ode_dopri5_step,
    ode_dopri8_step,
    ode_rk23_step,
    ode_rk45_step,
    ode_rkf45_step,
)
from ._registry import get_method

__all__ = [
    'ODESolution',
    'solve_ivp',
]

# Embedded pairs: name -> (stepper, order of the embedded error estimate).
_ADAPTIVE = {
    'dopri5': (ode_dopri5_step, 4),
    'rk45_dopri': (ode_dopri5_step, 4),
    'rk45': (ode_rk45_step, 4),
    'rkf45': (ode_rkf45_step, 4),
    'rk23': (ode_rk23_step, 2),
    'bs32': (ode_bs32_step, 2),
    'dopri8': (ode_dopri8_step, 7),
    'rk87_dopri': (ode_dopri8_step, 7),
}

# Step-size controller constants (Hairer & Wanner, 1996, Sec. IV.2).
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_PI_ALPHA = 0.7  # exponent of the current error, divided by (q + 1)
_PI_BETA = 0.4  # exponent of the previous error, divided by (q + 1)


class ODESolution(NamedTuple):
    """Result of :func:`solve_ivp`.

    Attributes
    ----------
    ts : array or Quantity
        Output times. ``t_eval`` if it was given; otherwise the accepted step
        times, shape ``(max_steps + 1,)``, starting at ``t0`` and padded after
        index ``n_steps`` by repeating the last accepted time.
    ys : PyTree
        States at ``ts``, with the structure of ``y0`` and a leading axis of
        the same length as ``ts`` (padded like ``ts``).
    n_steps : jax.Array
        Number of accepted steps (int scalar).
    n_rejected : jax.Array
        Number of rejected step attempts (int scalar).
    success : jax.Array
        Boolean scalar: ``True`` if ``t1`` was reached within ``max_steps``
        attempts without the step size collapsing.
    """
    ts: Any
    ys: Any
    n_steps: jax.Array
    n_rejected: jax.Array
    success: jax.Array


def _resolve(method: str) -> Tuple[Callable, int]:
    if not isinstance(method, str):
        raise TypeError(
            f'solve_ivp needs a method name, one of {sorted(_ADAPTIVE)}, but got {type(method).__name__}.'
        )
    if method not in _ADAPTIVE:
        get_method(method, 'ode')  # raises KeyError (with close matches) for unknown names
        raise ValueError(
            f'Method {method!r} has no embedded error estimate. '
            f'Adaptive methods: {sorted(_ADAPTIVE)}.'
        )
    return _ADAPTIVE[method]


def _mantissa(x, unit, dtype):
    """Mantissa of ``x`` in ``unit``; plain numbers only for dimensionless units."""
    if isinstance(x, u.Quantity) or not unit.is_unitless:
        return jnp.asarray(u.Quantity(x).to_decimal(unit), dtype=dtype)
    return jnp.asarray(x, dtype=dtype)


def _is_concrete(*xs) -> bool:
    return not any(isinstance(x, jax.core.Tracer) for x in xs)


@set_module_as('braintools.quad')
def solve_ivp(
    f: Callable,
    t_span,
    y0,
    *args,
    method: str = 'dopri5',
    rtol: float = 1e-6,
    atol=1e-9,
    dt0=None,
    max_steps: int = 4096,
    t_eval=None,
    checkpoint: bool = True,
    **kwargs,
) -> ODESolution:
    r"""Solve an initial value problem with an adaptive embedded Runge–Kutta method.

    Integrates :math:`\dot y = f(y, t)` from ``t_span[0]`` to ``t_span[1]``
    with error control. Each attempt takes one step of an embedded pair
    (``method``), measures the error, accepts or rejects the step and picks
    the next step size with a PI controller. The attempts run in one
    fixed-length ``scan`` of ``max_steps`` iterations; once ``t1`` is
    reached the remaining iterations are skipped through a ``done`` mask.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(y, t, *args, **kwargs) -> dy/dt`` with the same
        PyTree structure (and units ``[y]/[t]``) as ``y``. Note the
        ``(y, t)`` argument order (``scipy`` uses ``(t, y)``).
    t_span : tuple of (float or Quantity)
        ``(t0, t1)``. ``t1 < t0`` integrates backwards in time. May be traced.
    y0 : PyTree
        Initial state. Leaves may be arrays or brainunit Quantities.
    *args
        Extra positional arguments forwarded to ``f``.
    method : str, optional
        Embedded pair: ``'dopri5'`` (Dormand–Prince 5(4), default),
        ``'rk45'`` (Cash–Karp 5(4)), ``'rkf45'`` (Fehlberg 4(5)), ``'rk23'``
        or ``'bs32'`` (Bogacki–Shampine 3(2)), ``'dopri8'`` (DOP853). The
        aliases ``'rk45_dopri'`` and ``'rk87_dopri'`` are accepted.
    rtol : float, optional
        Relative tolerance. Default ``1e-6``.
    atol : float, Quantity or PyTree, optional
        Absolute tolerance. A float applies to the mantissas of every leaf; a
        Quantity is converted to each leaf's unit; a PyTree matching ``y0``
        sets one tolerance per leaf. Default ``1e-9``.
    dt0 : float or Quantity, optional
        Initial step size magnitude. If ``None`` (default) it is chosen by
        the Hairer–Wanner starting-step heuristic [1]_.
    max_steps : int, optional
        Static number of step attempts (accepted plus rejected). Default
        ``4096``.
    t_eval : array or Quantity, optional
        1-D array of output times (static length, any order, duplicates
        allowed) inside ``t_span``. The steps are shortened so that every
        ``t_eval`` point is an accepted step (see Notes).
    checkpoint : bool, optional
        If ``True`` (default), the differentiated replay loop is a
        ``brainstate.transform.checkpointed_scan``, so reverse-mode memory
        no longer scales with ``max_steps`` (at roughly 30% more gradient
        compute). ``False`` stores the residuals of every one of the
        ``max_steps`` iterations. Values and gradients are identical either
        way; forward-only evaluation is unaffected.
    **kwargs
        Extra keyword arguments forwarded to ``f``.

    Returns
    -------
    ODESolution
        NamedTuple ``(ts, ys, n_steps, n_rejected, success)``. Without
        ``t_eval``, ``ts``/``ys`` hold the ``max_steps + 1`` slots of accepted
        steps (``ts[0] == t0``), padded after ``n_steps`` with the last
        accepted value; ``ys[n_steps]`` is the final state. With ``t_eval``,
        ``ts`` is ``t_eval`` and ``ys`` the interpolated states.

    Raises
    ------
    KeyError
        If ``method`` is not a registered ODE method.
    ValueError
        If ``method`` has no embedded error estimate, ``t_span`` is not a
        pair, ``max_steps < 1``, ``t_eval`` is not 1-D or (when concrete)
        lies outside ``t_span``, ``dt0`` is (concretely) zero, or ``atol``
        does not match ``y0``.
    brainunit.UnitMismatchError
        If times, tolerances or ``f``'s output have units incompatible with
        ``t_span`` and ``y0``.

    See Also
    --------
    odeint : Fixed-step trajectory driver.
    scipy.integrate.solve_ivp : The SciPy counterpart.

    Notes
    -----
    **Error norm.** The step error ``err`` returned by the embedded pair is
    measured on the mantissas as

    .. math::

        \|e\| = \sqrt{\frac{1}{N}\sum_i
        \left(\frac{e_i}{\mathrm{atol}_i + \mathrm{rtol}\,\max(|y_i|, |y^{new}_i|)}\right)^2},

    and the step is accepted when :math:`\|e\| \le 1`.

    **Step-size control.** After an accepted step the PI controller of [1]_
    (Sec. IV.2) proposes :math:`h_{new} = h \cdot s\,\|e_n\|^{-0.7/k}
    \|e_{n-1}\|^{0.4/k}` with :math:`k` one more than the order of the
    embedded error estimate, safety :math:`s = 0.9` and the factor bounded
    to ``[0.2, 10]`` (no growth right after a rejection). A rejected step is
    retried with :math:`h \cdot \max(0.2, s\,\|e\|^{-1/k})`. Non-finite
    steps are rejected. If the step size falls below ``10 * eps * |t|`` the
    solve stops with ``success=False``; the same happens when ``max_steps``
    attempts are exhausted. In both cases the outputs hold the last accepted
    state (no NaN padding).

    **Two passes.** Pass 1 runs the adaptive loop (a fixed-length ``scan``
    of ``max_steps`` attempts, with a ``cond`` skipping the iterations after
    ``done``) on stop-gradient copies of all inputs and records, per attempt,
    whether it was accepted and its step size. Pass 2 replays only the
    accepted steps (without the error estimate) on the differentiated path.
    So each accepted step is computed twice, but a rejected attempt, even
    one that produced NaN, never enters a gradient, and recomputation under
    ``checkpoint=True`` cannot change a decision.

    **Gradients.** The step sizes chosen by the controller are constants,
    i.e. gradients are those of the solution on the accepted time grid
    ("discretise-then-optimise"); the steps that land on ``t1`` or a
    ``t_eval`` point carry the dependence on those times.

    **Output at** ``t_eval``. A cubic Hermite interpolant between accepted
    steps is only :math:`O(h^4)` accurate, which for 5th- and 8th-order
    pairs is far worse than the step error. The solver therefore never steps
    over a ``t_eval`` point: a step that would cross one is shortened to land
    on it (without shrinking the next proposed step), and that step's state
    is the output, with the accuracy of the solver itself. This costs about
    one extra step per ``t_eval`` point, which counts towards
    ``max_steps``. Points beyond the last accepted time of a failed solve
    return the last accepted state.

    **Memory.** With ``t_eval`` the forward pass holds a few copies of the
    state plus the ``(len(t_eval),) + y.shape`` output, independent of
    ``max_steps``. Without ``t_eval`` the output itself has
    ``max_steps + 1`` rows, and the forward pass holds about two such
    arrays; prefer ``t_eval`` for large states. Reverse mode with
    ``checkpoint=False`` stores the step residuals (several copies of the
    state) for each of the ``max_steps`` iterations, whatever the number of
    steps actually taken; e.g. about 0.9 GiB for a ``1e4``-element float32
    state at ``max_steps=4096``, against about 7 MiB with
    ``checkpoint=True``.

    **Backwards integration** (``t1 < t0``) is supported: steps are taken
    with negative ``dt``.

    Transformations: ``jax.jit`` (``method``, ``max_steps`` and the length
    of ``t_eval`` are static; ``t_span``, ``y0``, tolerances and arguments
    may be traced), ``jax.vmap`` and forward/reverse-mode ``jax.grad`` with
    respect to ``y0``, ``args``, ``kwargs``, ``t_span`` and ``t_eval``.

    References
    ----------
    .. [1] E. Hairer and G. Wanner, *Solving Ordinary Differential Equations
       II: Stiff and Differential-Algebraic Problems*, 2nd ed., Springer,
       1996. See also E. Hairer, S. P. Nørsett and G. Wanner, *Solving
       Ordinary Differential Equations I*, 2nd ed., Springer, 1993,
       Sec. II.4 (starting step size).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.quad import solve_ivp
        >>> sol = solve_ivp(lambda y, t: -y, (0.0, 2.0), jnp.array(1.0),
        ...                 rtol=1e-6, atol=1e-8,
        ...                 t_eval=jnp.array([0.0, 1.0, 2.0]))
        >>> bool(sol.success)
        True
        >>> print([round(float(v), 5) for v in sol.ys])
        [1.0, 0.36788, 0.13534]

    Without ``t_eval`` the accepted steps are returned:

    .. code-block:: python

        >>> sol = solve_ivp(lambda y, t, k: -k * y, (0.0, 2.0), jnp.array(1.0), 3.0,
        ...                 max_steps=64)
        >>> n = int(sol.n_steps)
        >>> print(float(sol.ts[n]), round(float(sol.ys[n]), 6))
        2.0 0.002479
    """
    _step, err_order = _resolve(method)
    if not isinstance(max_steps, (int, np.integer)) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError(f'max_steps must be a positive int, but got {max_steps!r}.')
    max_steps = int(max_steps)
    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise ValueError('t_span must be a pair (t0, t1).')
    t0, t1 = t_span

    # ---- time and state in mantissa space ---------------------------------
    time_is_q = isinstance(t0, u.Quantity) or isinstance(t1, u.Quantity)
    t_unit = u.get_unit(t0) if isinstance(t0, u.Quantity) else u.get_unit(t1)
    leaves, treedef = jax.tree.flatten(y0, is_leaf=u.math.is_quantity)
    is_q = [isinstance(leaf, u.Quantity) for leaf in leaves]
    units = [u.get_unit(leaf) for leaf in leaves]
    y0m = []
    for leaf in leaves:
        m = jnp.asarray(u.get_mantissa(leaf))
        if not jnp.issubdtype(m.dtype, jnp.inexact):
            m = m.astype(brainstate.environ.dftype())
        y0m.append(m)
    dtype = jnp.result_type(*[m.dtype for m in y0m]) if y0m else brainstate.environ.dftype()
    if jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError('solve_ivp supports real-valued states only.')
    t0m = _mantissa(t0, t_unit, dtype)
    t1m = _mantissa(t1, t_unit, dtype)
    d_units = [unit / t_unit for unit in units]

    def wrap(ms):
        return treedef.unflatten([u.Quantity(m, unit) if q else m for m, unit, q in zip(ms, units, is_q)])

    def t_wrap(tm):
        return u.Quantity(tm, t_unit) if time_is_q else tm

    def make_f(f_args, f_kwargs):
        def rhs(ym, tm):
            return f_m(ym, tm, f_args, f_kwargs)

        return rhs

    def f_m(ym, tm, f_args=args, f_kwargs=kwargs):
        dy = f(wrap(ym), t_wrap(tm), *f_args, **f_kwargs)
        d_leaves = jax.tree.leaves(dy, is_leaf=u.math.is_quantity)
        if len(d_leaves) != len(ym):
            raise ValueError(
                f'f returned {len(d_leaves)} leaves, but the state has {len(ym)}.'
            )
        out = []
        for d, m, du, q in zip(d_leaves, ym, d_units, is_q):
            dm = _mantissa(d, du, m.dtype) if (q or time_is_q or isinstance(d, u.Quantity)) else \
                jnp.asarray(d, dtype=m.dtype)
            out.append(jnp.broadcast_to(dm, m.shape))
        return out

    # ---- tolerances --------------------------------------------------------
    rtol = jnp.asarray(rtol, dtype=dtype)
    if isinstance(atol, u.Quantity):
        atol_m = [_mantissa(atol, unit, dtype) for unit in units]
    else:
        a_leaves = jax.tree.leaves(atol, is_leaf=u.math.is_quantity)
        if len(a_leaves) == 1:
            a_leaves = a_leaves * len(units)
        if len(a_leaves) != len(units):
            raise ValueError(
                f'atol must be a scalar, a Quantity or a PyTree matching y0 '
                f'({len(units)} leaves), but has {len(a_leaves)} leaves.'
            )
        atol_m = [
            _mantissa(a, unit, dtype) if isinstance(a, u.Quantity) else jnp.asarray(a, dtype=dtype)
            for a, unit in zip(a_leaves, units)
        ]
    n_total = max(sum(int(np.prod(m.shape)) for m in y0m), 1)

    def err_norm(err, y, y_new):
        total = 0.
        for e, a, b, at in zip(err, y, y_new, atol_m):
            scale = at + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
            total = total + jnp.sum((e / scale) ** 2)
        return jnp.sqrt(total / n_total)

    def rms_scaled(xs, scale):
        return jnp.sqrt(sum(jnp.sum((x / s) ** 2) for x, s in zip(xs, scale)) / n_total)

    # ---- t_eval ------------------------------------------------------------
    if t_eval is not None:
        te_m = _mantissa(t_eval, t_unit, dtype)
        if te_m.ndim != 1:
            raise ValueError(f't_eval must be 1-D, but has shape {te_m.shape}.')
        if _is_concrete(te_m, t0m, t1m):
            lo, hi = sorted([float(t0m), float(t1m)])
            tol = 1e-9 * max(abs(lo), abs(hi), 1.0)
            if te_m.size and (float(te_m.min()) < lo - tol or float(te_m.max()) > hi + tol):
                raise ValueError(f't_eval must lie inside t_span, i.e. within [{lo}, {hi}].')

    span = t1m - t0m
    direction = jnp.where(span >= 0, 1.0, -1.0).astype(dtype)

    # ---- initial step ------------------------------------------------------
    if dt0 is None:
        def initial_step():
            f0 = f_m(y0m, t0m)
            scale = [at + rtol * jnp.abs(m) for at, m in zip(atol_m, y0m)]
            d0 = rms_scaled(y0m, scale)
            d1 = rms_scaled(f0, scale)
            small = (d0 < 1e-5) | (d1 < 1e-5)
            h0 = jnp.where(small, 1e-6, 0.01 * d0 / jnp.where(small, 1.0, d1))
            h0 = jnp.minimum(h0, jnp.abs(span))
            h0_safe = jnp.where(h0 > 0, h0, 1.0)
            y1 = [m + direction * h0 * k for m, k in zip(y0m, f0)]
            f1 = f_m(y1, t0m + direction * h0)
            d2 = rms_scaled([a - b for a, b in zip(f1, f0)], scale) / h0_safe
            dmax = jnp.maximum(d1, d2)
            h1 = jnp.where(
                dmax <= 1e-15,
                jnp.maximum(1e-6, h0 * 1e-3),
                (0.01 / jnp.where(dmax <= 1e-15, 1.0, dmax)) ** (1.0 / (err_order + 1)),
            )
            return jnp.minimum(jnp.minimum(100 * h0, h1), jnp.abs(span))

        habs0 = jax.lax.stop_gradient(initial_step()).astype(dtype)
    else:
        habs0 = jnp.abs(_mantissa(dt0, t_unit, dtype))
        if _is_concrete(habs0) and float(habs0) == 0.0:
            raise ValueError('dt0 must be non-zero.')
        habs0 = jax.lax.stop_gradient(habs0)

    # ---- stopping points: the sorted t_eval (clipped to t1), then t1 -----------
    span_abs = jnp.abs(span)
    n_eval = 0 if t_eval is None else te_m.shape[0]
    if n_eval:
        tau_e = (te_m - t0m) * direction
        order = jnp.argsort(tau_e)
        tau_e_sorted, te_sorted = tau_e[order], te_m[order]
        beyond = tau_e_sorted >= span_abs
        stops_tau = jnp.concatenate([jnp.where(beyond, span_abs, tau_e_sorted), span_abs[None]])
        stops_t = jnp.concatenate([jnp.where(beyond, t1m, te_sorted), t1m[None]])
    else:
        stops_tau, stops_t = span_abs[None], t1m[None]
    n_stops = stops_tau.shape[0]

    # ---- pass 1: step-size control (no gradients) ------------------------------
    # The adaptive loop runs on stop-gradient copies of everything, recording
    # per attempt whether it was accepted, whether it landed on a stopping
    # point and the step size. Non-finite partials of rejected attempts can
    # therefore never reach a gradient, and pass 2 replays fixed decisions.
    eps = float(jnp.finfo(dtype).eps)
    k_exp = float(err_order + 1)
    sg = jax.lax.stop_gradient

    def sg_arrays(tree):
        return jax.tree.map(lambda x: sg(x) if isinstance(x, jax.Array) else x, tree)

    f_ctrl = make_f(sg_arrays(args), sg_arrays(kwargs))
    stops_tau_c, stops_t_c, t0c, t1c, span_c = sg(stops_tau), sg(stops_t), sg(t0m), sg(t1m), sg(span_abs)
    direction_c = sg(direction)

    def attempt(carry):
        t, y, habs, err_old, done, failed, n_acc, n_rej, prev_rej = carry
        tau_t = (t - t0c) * direction_c
        # never step over the next stopping point
        j = jnp.minimum(jnp.searchsorted(stops_tau_c, tau_t, side='right'), n_stops - 1)
        target_tau, target_t = stops_tau_c[j], stops_t_c[j]
        is_stop = habs >= target_tau - tau_t
        is_last = is_stop & (target_tau >= span_c)
        h = jnp.where(is_stop, target_t - t, direction_c * habs)
        with brainstate.environ.context(dt=h):
            y_new, err = _step(f_ctrl, y, t, return_error=True)
        en = err_norm(err, y, y_new)
        finite = jnp.isfinite(en)
        for leaf in y_new:
            finite = finite & jnp.all(jnp.isfinite(leaf))
        en = jnp.where(finite, en, jnp.inf)
        accept = en <= 1.0
        en_safe = jnp.maximum(en, 1e-10)
        fac_acc = _SAFETY * en_safe ** (-_PI_ALPHA / k_exp) * err_old ** (_PI_BETA / k_exp)
        fac_acc = jnp.clip(fac_acc, _MIN_FACTOR, jnp.where(prev_rej, 1.0, _MAX_FACTOR))
        fac_rej = jnp.clip(_SAFETY * en_safe ** (-1.0 / k_exp), _MIN_FACTOR, 1.0)
        h_abs = jnp.abs(h)
        habs_new = jnp.where(
            accept,
            # a step shortened to land on a stopping point does not shrink the proposal
            jnp.where(is_stop, jnp.maximum(habs, h_abs * fac_acc), h_abs * fac_acc),
            h_abs * fac_rej,
        ).astype(dtype)
        t_c = jnp.where(accept, jnp.where(is_stop, target_t, t + h), t)
        y_c = [jnp.where(accept, a, b).astype(b.dtype) for a, b in zip(y_new, y)]
        done_new = accept & is_last
        tiny = 10 * eps * jnp.maximum(jnp.abs(t_c), jnp.abs(t1c))
        failed_new = (~done_new) & (habs_new <= tiny)
        new_carry = (
            t_c, y_c, habs_new,
            jnp.where(accept, jnp.maximum(en, 1e-4), err_old).astype(dtype),
            done_new, failed_new,
            n_acc + accept.astype(n_acc.dtype),
            n_rej + (~accept).astype(n_rej.dtype),
            ~accept,
        )
        return new_carry, (accept, is_stop, j.astype(jnp.int32), habs)

    def skip(carry):
        return carry, (jnp.asarray(False), jnp.asarray(False), jnp.asarray(0, jnp.int32), carry[2])

    def control(carry, _):
        done, failed = carry[4], carry[5]
        return brainstate.transform.cond(done | failed, skip, attempt, carry)

    zero = jnp.asarray(0, dtype=jnp.int32)
    init = (
        t0c, [sg(m) for m in y0m], habs0, jnp.asarray(1.0, dtype=dtype),
        span_c == 0, jnp.asarray(False), zero, zero, jnp.asarray(False),
    )
    final, decisions = brainstate.transform.scan(control, init, None, length=max_steps)
    _, _, _, _, done, failed, n_acc, n_rej, _ = final
    success = done & ~failed

    # ---- pass 2: replay the accepted steps on the differentiated path ----------
    def advance(carry, take, is_stop, j, habs):
        t, y, buf = carry
        target_t = stops_t[j]
        h = jnp.where(is_stop, target_t - t, direction * habs)
        # Under vmap ``cond`` becomes ``select``; a zero step keeps the lanes that
        # do not take a step finite.
        with brainstate.environ.context(dt=jnp.where(take, h, jnp.zeros_like(h))):
            y_next = _step(f_m, y, t)
        y_next = [a.astype(b.dtype) for a, b in zip(y_next, y)]
        t_next = jnp.where(is_stop, target_t, t + h)
        if n_eval:
            # write the state into the output slot of the t_eval point it landed on
            idx = jnp.where(is_stop & (j < n_eval), j, n_eval)
            buf = [b.at[idx].set(v, mode='drop') for b, v in zip(buf, y_next)]
        return (t_next, y_next, buf), None if t_eval is not None else (t_next, y_next)

    def hold(carry, take, is_stop, j, habs):
        return carry, None if t_eval is not None else (carry[0], carry[1])

    def replay(carry, xs):
        take = xs[0]
        return brainstate.transform.cond(take, advance, hold, carry, *xs)

    buf0 = [jnp.broadcast_to(m, (n_eval,) + m.shape) for m in y0m] if n_eval else None
    scan = brainstate.transform.checkpointed_scan if checkpoint else brainstate.transform.scan
    (t_f, y_f, buf), outs = scan(replay, (t0m, y0m, buf0), decisions)

    if t_eval is None:
        # compact the accepted steps into slots 0..max_steps
        acc = decisions[0]
        ts_it, ys_it = outs
        csum = jnp.cumsum(acc.astype(jnp.int32))
        kk = jnp.minimum(jnp.arange(max_steps + 1), n_acc)
        src = jnp.where(kk == 0, 0, jnp.searchsorted(csum, kk, side='left') + 1)

        def slots(first, stacked):
            return jnp.concatenate([first[None], stacked], axis=0)[src]

        ts_s = slots(t0m, ts_it)
        ys_s = [slots(a, b) for a, b in zip(y0m, ys_it)]
        return ODESolution(t_wrap(ts_s), wrap(ys_s), n_acc, n_rej, success)

    if not n_eval:
        return ODESolution(t_wrap(te_m), wrap([jnp.zeros((0,) + m.shape, m.dtype) for m in y0m]),
                           n_acc, n_rej, success)

    # ---- states at t_eval ----------------------------------------------------
    # Every t_eval point inside the reached range is an accepted step, written
    # into the slot of the first of its duplicates (points at t0 keep y0).
    # Points beyond the last accepted time of a failed solve get the last state.
    first = jnp.searchsorted(stops_tau[:n_eval], stops_tau[:n_eval], side='left')
    unreached = stops_tau[:n_eval] > (t_f - t0m) * direction
    inverse = jnp.argsort(order)

    def gather(b, last):
        shape = (-1,) + (1,) * last.ndim
        vals = jnp.where(jnp.reshape(unreached, shape), last[None], b[first])
        return vals[inverse]

    ys_e = [gather(b, last) for b, last in zip(buf, y_f)]
    return ODESolution(t_wrap(te_m), wrap(ys_e), n_acc, n_rej, success)
