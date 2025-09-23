# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
Ordinary differential equation (ODE) one-step integrators.

Compact, JAX-friendly steppers that operate on arbitrary PyTrees and use the
global time step ``dt`` from ``brainstate.environ``. Methods include Euler and
Runge–Kutta families as well as an exponential Euler variant for stiff linear
parts.
"""

from typing import Callable, Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

__all__ = [
    'ode_euler_step',
    'ode_rk2_step',
    'ode_rk3_step',
    'ode_rk4_step',
    'ode_expeuler_step',
    'ode_midpoint_step',
    'ode_heun_step',
    'ode_rk4_38_step',
    'ode_rk45_step',
]

DT = brainstate.typing.ArrayLike
ODE = Callable[[brainstate.typing.PyTree, float | u.Quantity, ...], brainstate.typing.PyTree]


def tree_map(f: Callable[..., Any], tree: Any, *rest: Any):
    return jax.tree.map(f, tree, *rest, is_leaf=u.math.is_quantity)


def ode_euler_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args
):
    r"""
    Explicit Euler step for ordinary differential equations.

    Implements a single forward Euler step for ODEs of the form

    .. math::

        \frac{dy}{dt} = f(y, t), \qquad y_{n+1} = y_n + \Delta t\, f(y_n, t_n).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree`` that computes
        the time-derivative at ``(y, t)``.
    y : PyTree
        Current state at time ``t``. Any JAX-compatible pytree.
    t : float or brainunit.Quantity
        Current time. If a quantity, units may propagate through derivatives.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}`` after one Euler step.

    See Also
    --------
    ode_rk2_step : Second-order Runge–Kutta.
    ode_rk4_step : Fourth-order Runge–Kutta.
    ode_expeuler_step : Exponential Euler step.

    Notes
    -----
    - First-order accurate with local truncation error :math:`\mathcal{O}(\Delta t)`.
    - Uses ``dt = brainstate.environ.get_dt()`` as the step size.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    return tree_map(lambda x, _k1: x + dt * _k1, y, k1)


def ode_rk2_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args
):
    r"""
    Second-order Runge–Kutta (RK2) step for ODEs.

    The classical RK2 (Heun/midpoint) method performs two function evaluations:

    .. math::

        k_1 = f(y_n, t_n),\quad
        k_2 = f\big(y_n + \Delta t\,k_1,\ t_n + \Delta t\big),\\
        y_{n+1} = y_n + \tfrac{\Delta t}{2}\,(k_1 + k_2).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK2 step.

    Notes
    -----
    Second-order accurate with local truncation error :math:`\mathcal{O}(\Delta t^2)`.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(tree_map(lambda x, k: x + dt * k, y, k1), t + dt, *args)
    return tree_map(lambda x, _k1, _k2: x + dt / 2 * (_k1 + _k2), y, k1, k2)


def ode_rk3_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args
):
    r"""
    Third-order Runge–Kutta (RK3) step for ODEs.

    A common RK3 scheme uses three stages:

    .. math::

        k_1 = f(y_n, t_n),\quad
        k_2 = f\big(y_n + \tfrac{\Delta t}{2}k_1,\ t_n + \tfrac{\Delta t}{2}\big),\\
        k_3 = f\big(y_n - \Delta t\,k_1 + 2\Delta t\,k_2,\ t_n + \Delta t\big),\\
        y_{n+1} = y_n + \tfrac{\Delta t}{6}(k_1 + 4k_2 + k_3).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK3 step.

    Notes
    -----
    Third-order accurate with local truncation error :math:`\mathcal{O}(\Delta t^3)`.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args)
    k3 = f(tree_map(lambda x, k1_val, k2_val: x - dt * k1_val + 2 * dt * k2_val, y, k1, k2), t + dt, *args)
    return tree_map(lambda x, _k1, _k2, _k3: x + dt / 6 * (_k1 + 4 * _k2 + _k3), y, k1, k2, k3)


def ode_rk4_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args
):
    r"""
    Classical fourth-order Runge–Kutta (RK4) step for ODEs.

    The standard RK4 scheme uses four stages:

    .. math::

        k_1 = f(y_n, t_n),\\
        k_2 = f\big(y_n + \tfrac{\Delta t}{2}k_1,\ t_n + \tfrac{\Delta t}{2}\big),\\
        k_3 = f\big(y_n + \tfrac{\Delta t}{2}k_2,\ t_n + \tfrac{\Delta t}{2}\big),\\
        k_4 = f\big(y_n + \Delta t\,k_3,\ t_n + \Delta t\big),\\
        y_{n+1} = y_n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4).

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK4 step.

    Notes
    -----
    Fourth-order accurate with local truncation error :math:`\mathcal{O}(\Delta t^4)`.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args)
    k3 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k2), t + dt / 2, *args)
    k4 = f(tree_map(lambda x, k: x + dt * k, y, k3), t + dt, *args)
    return tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt / 6 * (_k1 + 2 * _k2 + 2 * _k3 + _k4),
        y, k1, k2, k3, k4
    )


def ode_expeuler_step(
    f: ODE,
    y: brainstate.typing.ArrayLike,
    t: DT,
    *args
):
    r"""
    One-step Exponential Euler method for ODEs with linearized drift.

    Examples
    --------

    >>> def fun(x, t):
    ...     return -x
    >>> x = 1.0
    >>> exp_euler_step(fun, x， 0.)

    If the variable ( $x$ ) has units of ( $[X]$ ), then the drift term ( $\text{drift_fn}(x)$ ) should
    have units of ( $[X]/[T]$ ), where ( $[T]$ ) is the unit of time.

    If the variable ( x ) has units of ( [X] ), then the diffusion term ( \text{diffusion_fn}(x) )
    should have units of ( [X]/\sqrt{[T]} ).

    Parameters
    ----------
    f : callable
        Drift function ``f(y, t, *args)`` used in the exponential update.
    y : PyTree
        Current state. Must have a floating dtype.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}``.
    """
    assert callable(f), 'The input function should be callable.'
    if u.math.get_dtype(y) not in [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16]:
        raise ValueError(
            f'The input data type should be float64, float32, float16, or bfloat16 '
            f'when using Exponential Euler method. But we got {y.dtype}.'
        )
    dt = brainstate.environ.get('dt')
    linear, derivative = brainstate.transform.vector_grad(f, argnums=0, return_value=True)(y, t, *args)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    phi = u.math.exprel(dt * linear)
    x_next = y + dt * phi * derivative
    return x_next


def ode_midpoint_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args
):
    """
    Second-order Runge-Kutta (midpoint) step for ODEs.

    Uses the explicit midpoint variant:

    - k1 = f(y, t)
    - k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    - y_{n+1} = y + dt*k2

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK2-midpoint step.

    See Also
    --------
    ode_rk2_step : Heun/modified Euler variant of RK2.
    ode_rk4_step : Classical fourth-order Runge-Kutta.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    y_mid = tree_map(lambda x, k: x + (dt * 0.5) * k, y, k1)
    k2 = f(y_mid, t + dt * 0.5, *args)
    return tree_map(lambda x, _k2: x + dt * _k2, y, k2)


def ode_heun_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args
):
    """
    Third-order Runge-Kutta (Heun's RK3) step for ODEs.

    Coefficients (c,a,b):
    - c = [0, 1/3, 2/3]
    - a21 = 1/3; a32 = 2/3
    - b = [1/4, 0, 3/4]

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK3 (Heun) step.

    See Also
    --------
    ode_rk3_step : A different third-order RK scheme.
    ode_rk4_step : Classical fourth-order Runge-Kutta.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    y2 = tree_map(lambda x, k: x + (dt * (1.0 / 3.0)) * k, y, k1)
    k2 = f(y2, t + dt * (1.0 / 3.0), *args)
    y3 = tree_map(lambda x, k: x + (dt * (2.0 / 3.0)) * k, y, k2)
    k3 = f(y3, t + dt * (2.0 / 3.0), *args)
    return tree_map(lambda x, _k1, _k3: x + dt * ((1.0 / 4.0) * _k1 + (3.0 / 4.0) * _k3), y, k1, k3)


def ode_rk4_38_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args
):
    """
    Fourth-order Runge-Kutta (3/8-rule) step for ODEs.

    Butcher tableau:
    - c = [0, 1/3, 2/3, 1]
    - a21 = 1/3
    - a31 = -1/3, a32 = 1
    - a41 = 1, a42 = -1, a43 = 1
    - b = [1/8, 3/8, 3/8, 1/8]

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state after one RK4 (3/8 rule) step.

    See Also
    --------
    ode_rk4_step : Classical RK4 (1/6, 1/3, 1/3, 1/6 weights).
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    y2 = tree_map(lambda x, k: x + (dt * (1.0 / 3.0)) * k, y, k1)
    k2 = f(y2, t + dt * (1.0 / 3.0), *args)
    y3 = tree_map(lambda x, _k1, _k2: x + dt * ((-1.0 / 3.0) * _k1 + 1.0 * _k2), y, k1, k2)
    k3 = f(y3, t + dt * (2.0 / 3.0), *args)
    y4 = tree_map(lambda x, _k1, _k2, _k3: x + dt * (1.0 * _k1 + (-1.0) * _k2 + 1.0 * _k3), y, k1, k2, k3)
    k4 = f(y4, t + dt, *args)
    return tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt * (
            (1.0 / 8.0) * _k1 + (3.0 / 8.0) * _k2 + (3.0 / 8.0) * _k3 + (1.0 / 8.0) * _k4),
        y, k1, k2, k3, k4
    )


def ode_rk45_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    return_error: bool = False,
):
    """
    One step of the Cash-Karp embedded Runge-Kutta 4(5) method.

    Computes a 5th-order solution and a 4th-order embedded solution using six
    stages. Optionally returns a PyTree error estimate ``y5 - y4`` for adaptive
    step-size controllers.

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state at time ``t``.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.
    return_error : bool, default False
        If True, also return a PyTree error estimate ``(y5 - y4)``.

    Returns
    -------
    PyTree or tuple
        The updated state (5th-order). If ``return_error`` is True, returns
        ``(y_next, error_estimate)`` where both are PyTrees matching ``y``.

    Notes
    -----
    Butcher tableau (c, a, b5, b4):
    - c = [0, 1/5, 3/10, 3/5, 1, 7/8]
    - a21 = 1/5
    - a31 = 3/40,  a32 = 9/40
    - a41 = 3/10,  a42 = -9/10, a43 = 6/5
    - a51 = -11/54, a52 = 5/2, a53 = -70/27, a54 = 35/27
    - a61 = 1631/55296, a62 = 175/512, a63 = 575/13824, a64 = 44275/110592, a65 = 253/4096
    - b5  = [37/378, 0, 250/621, 125/594, 0, 512/1771]
    - b4  = [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]
    """
    dt = brainstate.environ.get_dt()

    k1 = f(y, t, *args)
    y2 = tree_map(lambda x, a: x + dt * (1.0 / 5.0) * a, y, k1)
    k2 = f(y2, t + dt * (1.0 / 5.0), *args)

    y3 = tree_map(lambda x, _k1, _k2: x + dt * ((3.0 / 40.0) * _k1 + (9.0 / 40.0) * _k2), y, k1, k2)
    k3 = f(y3, t + dt * (3.0 / 10.0), *args)

    y4 = tree_map(
        lambda x, _k1, _k2, _k3: x + dt * ((3.0 / 10.0) * _k1 + (-9.0 / 10.0) * _k2 + (6.0 / 5.0) * _k3),
        y, k1, k2, k3)
    k4 = f(y4, t + dt * (3.0 / 5.0), *args)

    y5 = tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt * (
            (-11.0 / 54.0) * _k1 + (5.0 / 2.0) * _k2 + (-70.0 / 27.0) * _k3 + (35.0 / 27.0) * _k4),
        y, k1, k2, k3, k4
    )
    k5 = f(y5, t + dt * 1.0, *args)

    y6 = tree_map(
        lambda x, _k1, _k2, _k3, _k4, _k5: x + dt * (
            (1631.0 / 55296.0) * _k1 +
            (175.0 / 512.0) * _k2 +
            (575.0 / 13824.0) * _k3 +
            (44275.0 / 110592.0) * _k4 +
            (253.0 / 4096.0) * _k5
        ),
        y, k1, k2, k3, k4, k5
    )
    k6 = f(y6, t + dt * (7.0 / 8.0), *args)

    # 5th-order solution
    y5th = tree_map(
        lambda x, _k1, _k3, _k4, _k6: x + dt * (
            (37.0 / 378.0) * _k1 + (250.0 / 621.0) * _k3 + (125.0 / 594.0) * _k4 + (512.0 / 1771.0) * _k6
        ),
        y, k1, k3, k4, k6
    )

    if not return_error:
        return y5th

    # 4th-order solution (embedded)
    y4th = tree_map(
        lambda x, _k1, _k3, _k4, _k5, _k6: x + dt * (
            (2825.0 / 27648.0) * _k1 + (18575.0 / 48384.0) * _k3 + (13525.0 / 55296.0) * _k4 + (
            277.0 / 14336.0) * _k5 + (1.0 / 4.0) * _k6
        ),
        y, k1, k3, k4, k5, k6
    )
    err = tree_map(lambda a, b: a - b, y5th, y4th)
    return y5th, err
