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
Stochastic differential equation (SDE) one-step integrators.

This module provides compact steppers for integrating SDEs inside simulation
loops. The steppers operate on arbitrary JAX PyTrees, making them suitable for
state containers used across BrainState.

Available steppers
------------------
- ``sde_euler_step``: Euler–Maruyama (Ito) scheme; strong order 0.5.
- ``sde_milstein_step``: Milstein scheme (Ito/Stratonovich); strong order 1.0.
- ``sde_expeuler_step``: Exponential Euler using linearized drift plus diffusion.

Notes
-----
All steppers use the global time step ``dt`` from ``brainstate.environ`` and
draw Gaussian noise using ``brainstate.random``. Noise is applied per PyTree
leaf and scaled by ``sqrt(dt)``.
"""

from typing import Callable, Union

import brainstate
import brainunit as u
import jax.numpy as jnp

from braintools._misc import set_module_as, tree_map, randn_like

__all__ = [
    'sde_euler_step',
    'sde_milstein_step',
    'sde_expeuler_step',
    'sde_heun_step',
    'sde_tamed_euler_step',
    'sde_implicit_euler_step',
    'sde_srk2_step',
    'sde_srk3_step',
    'sde_srk4_step',
]

DT = Union[float, u.Quantity]
DF = Callable[[brainstate.typing.PyTree, DT, ...], brainstate.typing.PyTree]
DG = Callable[[brainstate.typing.PyTree, DT, ...], brainstate.typing.PyTree]


@set_module_as('braintools.quad')
def sde_euler_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    sde_type: str = 'ito',
    **kwargs,
):
    r"""One Euler–Maruyama step for Ito SDEs.

    This integrates an Ito SDE of the form

    .. math:: dy = f(y, t)\,dt + g(y, t)\,dW,

    where ``f`` is the drift and ``g`` is the diffusion, using
    ``y_{n+1} = y_n + f(y_n, t_n) dt + g(y_n, t_n) dW_n`` with
    ``dW_n ~ Normal(0, dt)`` applied per PyTree leaf.

    Parameters
    ----------
    df : Callable[[PyTree, ArrayLike, ...], PyTree]
        Drift function ``f(y, t, *args)`` returning a PyTree matching ``y``.
    dg : Callable[[PyTree, ArrayLike, ...], PyTree]
        Diffusion function ``g(y, t, *args)`` returning a PyTree matching ``y``.
    y : PyTree
        Current state.
    t : float or brainunit.Quantity
        Current time (scalar or array broadcastable with ``y`` leaves).
    *args
        Extra arguments passed to ``df`` and ``dg``.
    sde_type : {'ito'}, optional
        Interpretation of the SDE. Only ``'ito'`` is supported, by default 'ito'.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}`` with the same tree structure as ``y``.

    See Also
    --------
    sde_milstein_step : Milstein scheme (strong order 1.0).
    sde_expeuler_step : Exponential Euler with linearized drift.

    Notes
    -----
    - Strong order 0.5, weak order 1.0.
    - Uses ``dt = brainstate.environ.get_dt()`` and Gaussian noise scaled by
      ``sqrt(dt)`` via ``brainstate.random.randn_like`` for each leaf of ``y``.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    assert sde_type in ['ito']

    dt = brainstate.environ.get_dt()
    dt_sqrt = jnp.sqrt(dt)
    y_bars = tree_map(
        lambda y0, drift, diffusion: y0 + drift * dt + diffusion * randn_like(y0) * dt_sqrt,
        y,
        df(y, t, *args, **kwargs),
        dg(y, t, *args, **kwargs),
    )
    return y_bars


@set_module_as('braintools.quad')
def sde_milstein_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    sde_type: str = 'ito',
    **kwargs,
):
    r"""One Milstein step for Ito or Stratonovich SDEs.

    This integrates an SDE of the form

    .. math:: dy = f(y, t)\,dt + g(y, t)\,dW,

    using the Milstein scheme. In Ito form, the update is

    .. math::
        y_{n+1} = y_n + f_n dt + g_n dW_n + \tfrac{1}{2} g_n \partial_y g_n (dW_n^2 - dt),

    while for Stratonovich (``sde_type='stra'``) the last term uses ``dW_n^2``
    instead of ``(dW_n^2 - dt)``. The directional derivative ``\partial_y g`` is
    approximated here by a finite difference using an intermediate evaluation.

    Parameters
    ----------
    df : Callable[[PyTree, ArrayLike, ...], PyTree]
        Drift function ``f(y, t, *args)``.
    dg : Callable[[PyTree, ArrayLike, ...], PyTree]
        Diffusion function ``g(y, t, *args)``.
    y : PyTree
        Current state.
    t : float or brainunit.Quantity
        Current time.
    *args
        Extra arguments forwarded to ``df`` and ``dg``.
    sde_type : {'ito', 'stra'}, optional
        Interpretation of the SDE: Ito or Stratonovich (``'stra'``).

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.

    See Also
    --------
    sde_euler_step : Euler–Maruyama scheme.
    sde_expeuler_step : Exponential Euler with linearized drift.

    Notes
    -----
    - Strong order 1.0 (Ito), offering higher accuracy than Euler–Maruyama.
    - The derivative term is realized via a finite-difference correction using an
      auxiliary diffusion evaluation at an intermediate state.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    assert sde_type in ['ito', 'stra']

    dt = brainstate.environ.get_dt()
    dt_sqrt = u.math.sqrt(dt)

    # drift values
    drifts = df(y, t, *args, **kwargs)

    # diffusion values
    diffusions = dg(y, t, *args, **kwargs)

    # intermediate results
    y_bars = tree_map(lambda y0, drift, diffusion: y0 + drift * dt + diffusion * dt_sqrt, y, drifts, diffusions)
    diffusion_bars = dg(y_bars, t, *args, **kwargs)

    # integral results
    def f_integral(y0, drift, diffusion, diffusion_bar):
        noise = randn_like(y0) * dt_sqrt
        noise_p2 = (noise ** 2 - dt) if sde_type == 'ito' else noise ** 2
        minus = (diffusion_bar - diffusion) / 2 / dt_sqrt
        return y0 + drift * dt + diffusion * noise + minus * noise_p2

    integrals = tree_map(f_integral, y, drifts, diffusions, diffusion_bars)
    return integrals


@set_module_as('braintools.quad')
def sde_expeuler_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.ArrayLike,
    t: DT,
    *args,
    **kwargs,
):
    r"""One Exponential Euler step for SDEs with linearized drift.

    The drift ``f`` is locally linearized and integrated exactly over one step
    via the exponential relative function, while the diffusion term from ``g``
    is added in Euler form with Gaussian noise scaled by ``sqrt(dt)``.

    Parameters
    ----------
    df : Callable[[PyTree, ArrayLike, ...], PyTree]
        Drift function ``f(y, t, *args)``. Its value and vector–Jacobian product
        are used internally for the exponential update.
    dg : Callable[[PyTree, ArrayLike, ...], PyTree]
        Diffusion function ``g(y, t, *args)``.
    y : PyTree
        Current state. Must have a floating dtype.
    t : float or brainunit.Quantity
        Current time.
    *args
        Extra arguments forwarded to ``df`` and ``dg``. The first extra argument
        is used solely to determine the shape for ``random.randn_like`` when
        sampling the diffusion noise.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}`` with the same structure as ``y``.

    See Also
    --------
    sde_euler_step : Euler–Maruyama scheme.
    sde_milstein_step : Milstein scheme (strong order 1.0).

    Notes
    -----
    - Uses ``dt = brainstate.environ.get('dt')`` for the step size.
    - Requires floating dtypes for ``y`` (float16/32/64 or bfloat16).
    - Unit consistency is validated using ``brainunit``; a mismatch between the
      drift update and diffusion units raises a ``ValueError``.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    if u.math.get_dtype(y) not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
        raise ValueError(
            f'The input data type should be float64, float32, float16, or bfloat16 '
            f'when using Exponential Euler method. But we got {y.dtype}.'
        )

    # drift
    dt = brainstate.environ.get('dt')
    linear, derivative = brainstate.transform.vector_grad(df, argnums=0, return_value=True)(y, t, *args, **kwargs)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    phi = u.math.exprel(dt * linear)
    x_next = y + dt * phi * derivative

    # diffusion
    diffusion_part = dg(y, t, *args, **kwargs) * u.math.sqrt(dt) * randn_like(args[0])
    if u.get_dim(x_next) != u.get_dim(diffusion_part):
        drift_unit = u.get_unit(x_next)
        time_unit = u.get_unit(dt)
        raise ValueError(
            f"Drift unit is {drift_unit}, "
            f"expected diffusion unit is {drift_unit / time_unit ** 0.5}, "
            f"but we got {u.get_unit(diffusion_part)}."
        )
    x_next += diffusion_part
    return x_next


@set_module_as('braintools.quad')
def sde_heun_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    sde_type: str = 'ito',
    **kwargs,
):
    r"""Stochastic Heun (predictor–corrector) step.

    Implements a predictor–corrector scheme. For Stratonovich SDEs, both drift
    and diffusion are averaged between the predictor and corrector; for Itô SDEs
    only the drift is averaged while diffusion is evaluated at the start.

    Predictor
    ---------
    ``y* = y + f(y, t) dt + g(y, t) dW``

    Corrector
    ---------
    - Itô: ``y_{n+1} = y + 0.5 (f(y, t) + f(y*, t+dt)) dt + g(y, t) dW``
    - Stratonovich: ``y_{n+1} = y + 0.5 (f(y, t) + f(y*, t+dt)) dt + 0.5 (g(y, t) + g(y*, t+dt)) dW``

    Parameters
    ----------
    df : callable
        Drift function ``f(y, t, *args)``.
    dg : callable
        Diffusion function ``g(y, t, *args)``.
    y : PyTree
        Current state.
    t : float or brainunit.Quantity
        Current time.
    *args
        Extra arguments forwarded to ``df`` and ``dg``.
    sde_type : {'ito', 'stra'}, optional
        Interpretation of the SDE.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    assert sde_type in ['ito', 'stra']

    dt = brainstate.environ.get_dt()
    dt_sqrt = u.math.sqrt(dt)

    # evaluate at start
    f0 = df(y, t, *args, **kwargs)
    g0 = dg(y, t, *args, **kwargs)

    # shared Brownian increment dW for all stages
    dW = tree_map(lambda y0: randn_like(y0) * dt_sqrt, y)

    # predictor state
    y_pred = tree_map(lambda y0, a, b, z: y0 + a * dt + b * z, y, f0, g0, dW)

    # evaluate at end
    f1 = df(y_pred, t + dt, *args, **kwargs)
    if sde_type == 'stra':
        g1 = dg(y_pred, t + dt, *args, **kwargs)
        g_use = tree_map(lambda a, b: 0.5 * (a + b), g0, g1)
    else:
        g_use = g0

    f_use = tree_map(lambda a, b: 0.5 * (a + b), f0, f1)
    y_next = tree_map(lambda y0, a, b, z: y0 + a * dt + b * z, y, f_use, g_use, dW)
    return y_next


@set_module_as('braintools.quad')
def sde_tamed_euler_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs,
):
    r"""Tamed Euler–Maruyama step (drift taming for superlinear growth).

    Applies componentwise taming to the drift to prevent explosion when the
    drift exhibits superlinear growth:

    ``y_{n+1} = y_n + [f(y_n, t_n) / (1 + dt * |f(y_n, t_n)|)] dt + g(y_n, t_n) dW_n``.

    Parameters
    ----------
    df : callable
        Drift function ``f(y, t, *args)``.
    dg : callable
        Diffusion function ``g(y, t, *args)``.
    y : PyTree
        Current state.
    t : float or brainunit.Quantity
        Current time.
    *args
        Extra arguments forwarded to ``df`` and ``dg``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.

    Notes
    -----
    - Taming is performed elementwise via ``f / (1 + dt * |f|)`` on each leaf.
    - Uses Brownian increment ``dW ~ Normal(0, dt)`` per leaf.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    dt = brainstate.environ.get_dt()
    dt_sqrt = u.math.sqrt(dt)

    f0 = df(y, t, *args, **kwargs)
    g0 = dg(y, t, *args, **kwargs)

    f_tamed = tree_map(lambda a: a / (1.0 + dt * u.math.abs(a)), f0)
    y_next = tree_map(
        lambda y0, a, b: y0 + a * dt + b * randn_like(y0) * dt_sqrt,
        y, f_tamed, g0
    )
    return y_next


@set_module_as('braintools.quad')
def sde_implicit_euler_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    max_iter: int = 2,
    **kwargs,
):
    r"""Implicit (drift-implicit) Euler–Maruyama step via fixed-point iteration.

    Solves ``y_{n+1} = y_n + f(y_{n+1}, t_{n+1}) dt + g(y_n, t_n) dW`` using a
    few fixed-point iterations starting from an explicit predictor.

    Parameters
    ----------
    df : callable
        Drift function ``f(y, t, *args)``.
    dg : callable
        Diffusion function ``g(y, t, *args)``.
    y : PyTree
        Current state.
    t : float or brainunit.Quantity
        Current time.
    *args
        Extra arguments forwarded to ``df`` and ``dg``.
    max_iter : int, default 2
        Number of fixed-point iterations for the implicit corrector.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.

    Notes
    -----
    - Uses a simple Picard iteration; for stiff problems increase ``max_iter``
      or provide a more robust nonlinear solver.
    - Diffusion is treated explicitly with ``g(y_n, t_n) dW``.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    assert max_iter >= 1

    dt = brainstate.environ.get_dt()
    dt_sqrt = u.math.sqrt(dt)

    # Explicit pieces at start
    g0 = dg(y, t, *args, **kwargs)
    dW = tree_map(lambda y0: randn_like(y0) * dt_sqrt, y)
    diff_inc = tree_map(lambda b, z: b * z, g0, dW)

    # Predictor (explicit Euler)
    y_pred = tree_map(lambda y0, a, inc: y0 + a * dt + inc, y, df(y, t, *args, **kwargs), diff_inc)

    # Fixed-point iterations on the drift term at t+dt
    y_k = y_pred
    for _ in range(max_iter):
        y_k = tree_map(lambda y0, inc, fnew: y0 + inc + dt * fnew, y, diff_inc, df(y_k, t + dt, *args, **kwargs))

    return y_k


def _brownian_like(y, dt):
    dt_sqrt = u.math.sqrt(dt)
    return tree_map(lambda y0: randn_like(y0) * dt_sqrt, y)


@set_module_as('braintools.quad')
def sde_srk2_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs,
):
    r"""Stochastic Runge–Kutta 2 (Heun) for Stratonovich SDEs.

    Applies the deterministic RK2 (Heun) tableau to the combined Stratonovich
    increment ``f dt + g dW`` using a single Brownian increment ``dW`` shared
    across stages:

    - ``k1 = f(y, t) dt + g(y, t) dW``
    - ``k2 = f(y + k1, t + dt) dt + g(y + k1, t + dt) dW``
    - ``y_{n+1} = y + 0.5 (k1 + k2)``

    Parameters
    ----------
    df, dg, y, t, *args
        Same semantics as other steppers. Interprets the SDE in Stratonovich
        sense.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    dt = brainstate.environ.get_dt()
    dW = _brownian_like(y, dt)

    k1 = tree_map(lambda a, b, z: a * dt + b * z, df(y, t, *args, **kwargs), dg(y, t, *args, **kwargs), dW)
    y2 = tree_map(lambda y0, k: y0 + k, y, k1)
    k2 = tree_map(lambda a, b, z: a * dt + b * z, df(y2, t + dt, *args, **kwargs), dg(y2, t + dt, *args, **kwargs), dW)
    y_next = tree_map(lambda y0, a, b: y0 + 0.5 * (a + b), y, k1, k2)
    return y_next


@set_module_as('braintools.quad')
def sde_srk3_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs,
):
    r"""Stochastic Runge–Kutta 3 (Stratonovich; Heun-RK3).

    Uses a classic 3-stage RK3 scheme on the Stratonovich increment
    ``F(y,t) = f(y,t) dt + g(y,t) dW`` with a single shared ``dW``:

    - ``k1 = F(y, t)``
    - ``k2 = F(y + 0.5 k1, t + 0.5 dt)``
    - ``k3 = F(y - k1 + 2 k2, t + dt)``
    - ``y_{n+1} = y + (k1 + 4 k2 + k3) / 6``

    Parameters
    ----------
    df, dg, y, t, *args
        As usual; Stratonovich interpretation assumed.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    dt = brainstate.environ.get_dt()
    dW = _brownian_like(y, dt)

    def F(y_, t_):
        return tree_map(lambda a, b, z: a * dt + b * z, df(y_, t_, *args, **kwargs), dg(y_, t_, *args, **kwargs), dW)

    k1 = F(y, t)
    y2 = tree_map(lambda y0, k: y0 + 0.5 * k, y, k1)
    k2 = F(y2, t + 0.5 * dt)
    y3 = tree_map(lambda y0, kk1, kk2: y0 - kk1 + 2.0 * kk2, y, k1, k2)
    k3 = F(y3, t + dt)
    y_next = tree_map(lambda y0, a, b, c: y0 + (a + 4.0 * b + c) / 6.0, y, k1, k2, k3)
    return y_next


@set_module_as('braintools.quad')
def sde_srk4_step(
    df: DF,
    dg: DG,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs,
):
    r"""Stochastic Runge–Kutta 4 (Stratonovich; classical RK4).

    Applies the classical 4-stage RK4 tableau to the Stratonovich increment
    ``F(y,t) = f(y,t) dt + g(y,t) dW`` with a single shared Brownian ``dW``:

    - ``k1 = F(y, t)``
    - ``k2 = F(y + 0.5 k1, t + 0.5 dt)``
    - ``k3 = F(y + 0.5 k2, t + 0.5 dt)``
    - ``k4 = F(y + k3, t + dt)``
    - ``y_{n+1} = y + (k1 + 2 k2 + 2 k3 + k4)/6``

    Notes
    -----
    - Suitable for Stratonovich SDEs. For Itô SDEs, conversions require
      additional correction terms (not included here).

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    assert callable(df), 'The drift function should be callable.'
    assert callable(dg), 'The diffusion function should be callable.'
    dt = brainstate.environ.get_dt()
    dW = _brownian_like(y, dt)

    def F(y_, t_):
        return tree_map(lambda a, b, z: a * dt + b * z, df(y_, t_, *args, **kwargs), dg(y_, t_, *args, **kwargs), dW)

    k1 = F(y, t)
    y2 = tree_map(lambda y0, k: y0 + 0.5 * k, y, k1)
    k2 = F(y2, t + 0.5 * dt)
    y3 = tree_map(lambda y0, k: y0 + 0.5 * k, y, k2)
    k3 = F(y3, t + 0.5 * dt)
    y4 = tree_map(lambda y0, k: y0 + k, y, k3)
    k4 = F(y4, t + dt)
    y_next = tree_map(lambda y0, a, b, c, d: y0 + (a + 2.0 * b + 2.0 * c + d) / 6.0, y, k1, k2, k3, k4)
    return y_next
