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
``dt`` from ``brainstate.environ``. Implicit solves are performed by simple
fixed‑point iterations by default; for difficult problems you can increase the
number of iterations or substitute a more robust nonlinear solver externally.
"""

from typing import Callable, Any

import brainstate

from braintools._misc import set_module_as, tree_map

__all__ = [
    'imex_euler_step',
    'imex_ars222_step',
    'imex_cnab_step',
]

DT = brainstate.typing.ArrayLike
PyTree = brainstate.typing.PyTree
F = Callable[[PyTree, DT, Any], PyTree]


def _fixed_point(update, y0: PyTree, max_iter: int = 2) -> PyTree:
    y = y0
    for _ in range(max_iter):
        y = update(y)
    return y


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

    by fixed-point iteration.

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
        Fixed‑point iterations for the implicit solve.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    dt = brainstate.environ.get_dt()
    rhs_exp = f_exp(y, t, *args, **kwargs)

    def G(Y):
        return tree_map(lambda y0, fe, fi: y0 + dt * (fe + fi), y, rhs_exp, f_imp(Y, t + dt, *args, **kwargs))

    return _fixed_point(G, y, max_iter=max_iter)


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
    ARS(2,2,2) IMEX Runge–Kutta step (Ascher–Ruuth–Spiteri).

    Two-stage, second‑order IMEX RK with explicit and implicit tableaus:

    Explicit A^E, b^E
      c_E = [0, 1]
      A^E = [[0, 0],
             [1, 0]]
      b^E = [1 - γ, γ]

    Implicit A^I, b^I
      c_I = [γ, 1]
      A^I = [[γ,      0],
             [1-γ,  γ]]
      b^I = [1 - γ, γ]

    with ``γ = 1 - 1/sqrt(2)``.

    Implementation solves stage equations by fixed‑point iteration.

    Parameters
    ----------
    f_exp, f_imp, y, t, *args
        As usual.
    max_iter : int, default 2
        Fixed‑point iterations for the implicit stage solves.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    import math

    dt = brainstate.environ.get_dt()
    gamma = 1.0 - 1.0 / math.sqrt(2.0)

    # Stage 1 (implicit only)
    def G1(Y1):
        return tree_map(lambda y0, fi: y0 + dt * gamma * fi, y, f_imp(Y1, t + gamma * dt, *args, **kwargs))

    Y1 = _fixed_point(G1, y, max_iter=max_iter)

    # Stage 2 (explicit from stage 1, implicit on itself)
    fE1 = f_exp(Y1, t, *args, **kwargs)  # c_E1 = 0
    fI1 = f_imp(Y1, t + gamma * dt, *args, **kwargs)

    def G2(Y2):
        return tree_map(
            lambda y0, fe1, fi1, fi2: y0 + dt * (fe1 + (1.0 - gamma) * fi1 + gamma * fi2),
            y, fE1, fI1, f_imp(Y2, t + dt, *args, **kwargs)
        )

    Y2 = _fixed_point(G2, y, max_iter=max_iter)

    # Combine stages
    fE2 = f_exp(Y2, t + dt, *args, **kwargs)
    fI2 = f_imp(Y2, t + dt, *args, **kwargs)

    y_next = tree_map(
        lambda y0, fe1, fe2, fi1, fi2: y0 + dt * (
            (1.0 - gamma) * fe1 + gamma * fe2 + (1.0 - gamma) * fi1 + gamma * fi2),
        y, fE1, fE2, fI1, fI2
    )
    return y_next


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
    the stiff part:

    ``y_{n+1} = y_n + dt * [ 3/2 f_exp(y_n, t_n) - 1/2 f_exp(y_{n-1}, t_{n-1}) ]
                   + dt/2 * [ f_imp(y_{n+1}, t_{n+1}) + f_imp(y_n, t_n) ]``

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
        Fixed‑point iterations for the implicit corrector.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    dt = brainstate.environ.get_dt()

    # Explicit AB2 predictor
    fE_n = f_exp(y, t, *args, **kwargs)
    fE_nm1 = f_exp(y_prev, t - dt, *args, **kwargs)
    fI_n = f_imp(y, t, *args, **kwargs)

    y_pred = tree_map(
        lambda y0, fe, fem1, fi: y0 + dt * (1.5 * fe - 0.5 * fem1) + 0.5 * dt * fi,
        y, fE_n, fE_nm1, fI_n
    )

    # Implicit CN corrector: y_{n+1} = y_pred + 0.5 dt * f_imp(y_{n+1})
    def G(Y):
        return tree_map(lambda yp, fi: yp + 0.5 * dt * fi, y_pred, f_imp(Y, t + dt, *args, **kwargs))

    return _fixed_point(G, y_pred, max_iter=max_iter)
