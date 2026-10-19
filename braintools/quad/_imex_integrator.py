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
Implicit–explicit (IMEX) time integrators for split ODEs.

These steppers solve ODEs split into nonstiff (explicit) and stiff (implicit)
components:

    y' = f_exp(y, t, ...) + f_imp(y, t, ...).

All steppers operate on arbitrary JAX PyTrees and use the global time step
``dt`` from ``brainstate.environ``. The implicit stage equations

.. math::

   Y = c + a\,\Delta t\, f_{\mathrm{imp}}(Y, t^*)

are solved with Newton's method on the flattened state. For states with up to
256 elements the Jacobian of ``f_imp`` is formed with ``jax.jacfwd`` and the
linear system is solved directly, so one Newton iteration is exact when
``f_imp`` is linear in ``y`` and the steppers are stable for arbitrarily stiff
linear implicit parts. Larger states use matrix-free Newton–Krylov steps
(GMRES on Jacobian-vector products, restart 20, at most 200 Krylov
iterations), so the cost grows with the number of Krylov iterations rather
than with ``n**3``. Elementwise implicit parts (for example a population of
uncoupled neurons) converge in one Krylov iteration; strongly coupled and
very stiff implicit parts may need more Newton iterations (``max_iter``).
"""

import math
from typing import Callable, Any

import brainstate
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from braintools._misc import set_module_as, tree_map

__all__ = [
    'imex_euler_step',
    'imex_ars222_step',
    'imex_cnab_step',
]

DT = brainstate.typing.ArrayLike
PyTree = brainstate.typing.PyTree
F = Callable[[PyTree, DT, Any], PyTree]


# States with at most this many elements use a dense Jacobian and direct solve;
# larger states use matrix-free GMRES on Jacobian-vector products.
_DENSE_NEWTON_MAX_SIZE = 256


def _newton_solve(residual, y0: PyTree, max_iter: int = 2) -> PyTree:
    """Solve ``residual(Y) = 0`` by Newton's method, starting from ``y0``.

    The state is flattened to a vector. For small states the Jacobian of the
    flattened residual is formed with ``jax.jacfwd`` and each update solves the
    dense system. For larger states each update is computed matrix-free with
    GMRES on Jacobian-vector products (``jax.jvp``), which costs a few
    evaluations of the residual per Krylov iteration and never forms the
    ``n x n`` Jacobian.
    """
    if max_iter < 1:
        raise ValueError(f'max_iter must be at least 1, got {max_iter}.')
    v, unravel = ravel_pytree(y0)

    def r_flat(x):
        return ravel_pytree(residual(unravel(x)))[0]

    dense = v.size <= _DENSE_NEWTON_MAX_SIZE
    tol = float(jnp.finfo(v.dtype).eps) ** 0.75
    for _ in range(max_iter):
        if dense:
            r = r_flat(v)
            v = v - jnp.linalg.solve(jax.jacfwd(r_flat)(v), r)
        else:
            r, jvp = jax.linearize(r_flat, v)
            dv, _ = jax.scipy.sparse.linalg.gmres(jvp, r, tol=tol, atol=0.0, restart=20, maxiter=10)
            v = v - dv
    return unravel(v)


def _stage_residual(c, a_dt, f_imp, t_stage, args, kwargs):
    """Residual of ``Y = c + a_dt * f_imp(Y, t_stage)``."""

    def residual(Y):
        return tree_map(lambda Yv, cv, fv: Yv - cv - a_dt * fv, Y, c, f_imp(Y, t_stage, *args, **kwargs))

    return residual


@set_module_as('braintools.quad')
def imex_euler_step(
    f_exp: F,
    f_imp: F,
    y: PyTree,
    t: DT,
    *args,
    max_iter: int = 2,
    **kwargs,
) -> PyTree:
    """
    First-order IMEX Euler step (explicit + drift-implicit).

    Solves

    ``y_{n+1} = y_n + dt * f_exp(y_n, t_n) + dt * f_imp(y_{n+1}, t_{n+1})``

    for ``y_{n+1}`` with Newton's method.

    Parameters
    ----------
    f_exp : callable
        Explicit (nonstiff) part ``f_exp(y, t, *args)``.
    f_imp : callable
        Implicit (stiff) part ``f_imp(y, t, *args)`` evaluated implicitly at
        the end of the step.
    y : PyTree
        Current state.
    t : float or brainunit.Quantity
        Current time.
    *args
        Extra arguments forwarded to ``f_exp`` and ``f_imp``.
    max_iter : int, default 2
        Newton iterations for the implicit solve. One iteration is exact when
        ``f_imp`` is linear in ``y``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    dt = brainstate.environ.get_dt()
    rhs_exp = f_exp(y, t, *args, **kwargs)
    c = tree_map(lambda y0, fe: y0 + dt * fe, y, rhs_exp)
    return _newton_solve(_stage_residual(c, dt, f_imp, t + dt, args, kwargs), c, max_iter=max_iter)


@set_module_as('braintools.quad')
def imex_ars222_step(
    f_exp: F,
    f_imp: F,
    y: PyTree,
    t: DT,
    *args,
    max_iter: int = 2,
    **kwargs,
) -> PyTree:
    """
    ARS(2,2,2) IMEX Runge–Kutta step (Ascher, Ruuth & Spiteri, 1997).

    Second-order, L-stable, stiffly accurate IMEX Runge–Kutta scheme with
    ``γ = 1 - 1/sqrt(2)`` and ``δ = 1 - 1/(2γ)``:

    Explicit tableau (c = [0, γ, 1])::

        0 | 0      0      0
        γ | γ      0      0
        1 | δ      1-δ    0
        --+--------------------
          | δ      1-δ    0

    Implicit tableau (c = [0, γ, 1])::

        0 | 0      0      0
        γ | 0      γ      0
        1 | 0      1-γ    γ
        --+--------------------
          | 0      1-γ    γ

    The two implicit stages are solved with Newton's method; the new state is
    the last stage.

    Parameters
    ----------
    f_exp, f_imp, y, t, *args
        As in :func:`imex_euler_step`.
    max_iter : int, default 2
        Newton iterations per implicit stage. One iteration is exact when
        ``f_imp`` is linear in ``y``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    dt = brainstate.environ.get_dt()
    gamma = 1.0 - 1.0 / math.sqrt(2.0)
    delta = 1.0 - 1.0 / (2.0 * gamma)

    # Stage 2: Y2 = y + dt*γ*fE(y) + dt*γ*fI(Y2)
    fE1 = f_exp(y, t, *args, **kwargs)
    c2 = tree_map(lambda y0, fe1: y0 + dt * gamma * fe1, y, fE1)
    Y2 = _newton_solve(_stage_residual(c2, gamma * dt, f_imp, t + gamma * dt, args, kwargs), c2, max_iter=max_iter)

    # Stage 3: Y3 = y + dt*(δ fE(y) + (1-δ) fE(Y2)) + dt*((1-γ) fI(Y2) + γ fI(Y3))
    fE2 = f_exp(Y2, t + gamma * dt, *args, **kwargs)
    fI2 = f_imp(Y2, t + gamma * dt, *args, **kwargs)
    c3 = tree_map(
        lambda y0, fe1, fe2, fi2: y0 + dt * (delta * fe1 + (1.0 - delta) * fe2 + (1.0 - gamma) * fi2),
        y, fE1, fE2, fI2,
    )
    return _newton_solve(_stage_residual(c3, gamma * dt, f_imp, t + dt, args, kwargs), Y2, max_iter=max_iter)


@set_module_as('braintools.quad')
def imex_cnab_step(
    f_exp: F,
    f_imp: F,
    y: PyTree,
    y_prev: PyTree,
    t: DT,
    *args,
    max_iter: int = 2,
    **kwargs,
) -> PyTree:
    """
    CNAB (Crank–Nicolson / Adams–Bashforth) IMEX step (second order).

    Advances using explicit AB2 for the nonstiff part and trapezoidal rule for
    the stiff part::

        y_{n+1} = y_n + dt * [ 3/2 f_exp(y_n, t_n) - 1/2 f_exp(y_{n-1}, t_{n-1}) ]
                      + dt/2 * [ f_imp(y_{n+1}, t_{n+1}) + f_imp(y_n, t_n) ]

    This is a two-step method: start it with a one-step method (for example
    :func:`imex_ars222_step`) to obtain ``y_prev`` at the first step.

    Parameters
    ----------
    f_exp, f_imp : callable
        Explicit and implicit functions.
    y : PyTree
        Current state at time ``t``.
    y_prev : PyTree
        Previous state at time ``t - dt``.
    t : float or brainunit.Quantity
        Current time.
    *args
        Extra arguments forwarded to ``f_exp`` and ``f_imp``.
    max_iter : int, default 2
        Newton iterations for the implicit solve. One iteration is exact when
        ``f_imp`` is linear in ``y``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    dt = brainstate.environ.get_dt()

    fE_n = f_exp(y, t, *args, **kwargs)
    fE_nm1 = f_exp(y_prev, t - dt, *args, **kwargs)
    fI_n = f_imp(y, t, *args, **kwargs)
    c = tree_map(
        lambda y0, fe, fem1, fi: y0 + dt * (1.5 * fe - 0.5 * fem1) + 0.5 * dt * fi,
        y, fE_n, fE_nm1, fI_n
    )
    return _newton_solve(_stage_residual(c, 0.5 * dt, f_imp, t + dt, args, kwargs), c, max_iter=max_iter)
