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
Ordinary differential equation (ODE) one-step integrators.

Compact, JAX-friendly steppers that operate on arbitrary PyTrees and use the
global time step ``dt`` from ``brainstate.environ``. Methods include Euler and
Runge–Kutta families as well as an exponential Euler variant for stiff linear
parts.
"""

from typing import Callable

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as, tree_map

__all__ = [
    'ode_euler_step',
    'ode_rk2_step',
    'ode_rk3_step',
    'ode_rk4_step',
    'ode_expeuler_step',
    'ode_midpoint_step',
    'ode_heun_step',
    'ode_rk4_38_step',
    'ode_rk45_step',  # Cash–Karp 4(5)
    'ode_rk23_step',  # Bogacki–Shampine 2(3)
    'ode_dopri5_step', 'ode_rk45_dopri_step',  # Dormand–Prince 5(4)
    'ode_rkf45_step',  # RK–Fehlberg 4(5)
    'ode_ssprk33_step',  # SSPRK(3,3)
    'ode_dopri8_step', 'ode_rk87_dopri_step',  # Dormand–Prince 8(7) (DOP853)
    'ode_bs32_step',  # Bogacki–Shampine 3(2) alias
    'ode_ralston2_step',  # Ralston RK2
    'ode_ralston3_step',  # Ralston RK3
]

DT = brainstate.typing.ArrayLike
ODE = Callable[[brainstate.typing.PyTree, float | u.Quantity, ...], brainstate.typing.PyTree]


@set_module_as('braintools.quad')
def ode_euler_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    return tree_map(lambda x, _k1: x + dt * _k1, y, k1)


@set_module_as('braintools.quad')
def ode_rk2_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
):
    r"""
    Second-order Runge–Kutta (RK2) step for ODEs.

    The classical RK2 (Heun/midpoint) method performs two function evaluations:

    .. math::

        k_1 = f(y_n, t_n),\\
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    k2 = f(tree_map(lambda x, k: x + dt * k, y, k1), t + dt, *args, **kwargs)
    return tree_map(lambda x, _k1, _k2: x + dt / 2 * (_k1 + _k2), y, k1, k2)


@set_module_as('braintools.quad')
def ode_rk3_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
):
    r"""
    Third-order Runge–Kutta (RK3) step for ODEs.

    A common RK3 scheme uses three stages:

    .. math::

        k_1 = f(y_n, t_n),\\
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    k2 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args, **kwargs)
    k3 = f(tree_map(lambda x, k1_val, k2_val: x - dt * k1_val + 2 * dt * k2_val, y, k1, k2), t + dt, *args, **kwargs)
    return tree_map(lambda x, _k1, _k2, _k3: x + dt / 6 * (_k1 + 4 * _k2 + _k3), y, k1, k2, k3)


@set_module_as('braintools.quad')
def ode_rk4_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    k2 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args, **kwargs)
    k3 = f(tree_map(lambda x, k: x + dt / 2 * k, y, k2), t + dt / 2, *args, **kwargs)
    k4 = f(tree_map(lambda x, k: x + dt * k, y, k3), t + dt, *args, **kwargs)
    return tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt / 6 * (_k1 + 2 * _k2 + 2 * _k3 + _k4),
        y, k1, k2, k3, k4
    )


@set_module_as('braintools.quad')
def ode_expeuler_step(
    f: ODE,
    y: brainstate.typing.ArrayLike,
    t: DT,
    *args,
    **kwargs
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
    linear, derivative = brainstate.transform.vector_grad(f, argnums=0, return_value=True)(y, t, *args, **kwargs)
    linear = u.Quantity(u.get_mantissa(linear), u.get_unit(derivative) / u.get_unit(linear))
    phi = u.math.exprel(dt * linear)
    x_next = y + dt * phi * derivative
    return x_next


@set_module_as('braintools.quad')
def ode_midpoint_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    y_mid = tree_map(lambda x, k: x + (dt * 0.5) * k, y, k1)
    k2 = f(y_mid, t + dt * 0.5, *args, **kwargs)
    return tree_map(lambda x, _k2: x + dt * _k2, y, k2)


@set_module_as('braintools.quad')
def ode_heun_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, k: x + (dt * (1.0 / 3.0)) * k, y, k1)
    k2 = f(y2, t + dt * (1.0 / 3.0), *args, **kwargs)
    y3 = tree_map(lambda x, k: x + (dt * (2.0 / 3.0)) * k, y, k2)
    k3 = f(y3, t + dt * (2.0 / 3.0), *args, **kwargs)
    return tree_map(lambda x, _k1, _k3: x + dt * ((1.0 / 4.0) * _k1 + (3.0 / 4.0) * _k3), y, k1, k3)


@set_module_as('braintools.quad')
def ode_rk4_38_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, k: x + (dt * (1.0 / 3.0)) * k, y, k1)
    k2 = f(y2, t + dt * (1.0 / 3.0), *args, **kwargs)
    y3 = tree_map(lambda x, _k1, _k2: x + dt * ((-1.0 / 3.0) * _k1 + 1.0 * _k2), y, k1, k2)
    k3 = f(y3, t + dt * (2.0 / 3.0), *args, **kwargs)
    y4 = tree_map(lambda x, _k1, _k2, _k3: x + dt * (1.0 * _k1 + (-1.0) * _k2 + 1.0 * _k3), y, k1, k2, k3)
    k4 = f(y4, t + dt, *args, **kwargs)
    return tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt * (
            (1.0 / 8.0) * _k1 + (3.0 / 8.0) * _k2 + (3.0 / 8.0) * _k3 + (1.0 / 8.0) * _k4),
        y, k1, k2, k3, k4
    )


@set_module_as('braintools.quad')
def ode_rk45_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    return_error: bool = False,
    **kwargs,
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
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()

    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, a: x + dt * (1.0 / 5.0) * a, y, k1)
    k2 = f(y2, t + dt * (1.0 / 5.0), *args, **kwargs)

    y3 = tree_map(lambda x, _k1, _k2: x + dt * ((3.0 / 40.0) * _k1 + (9.0 / 40.0) * _k2), y, k1, k2)
    k3 = f(y3, t + dt * (3.0 / 10.0), *args, **kwargs)

    y4 = tree_map(
        lambda x, _k1, _k2, _k3: x + dt * ((3.0 / 10.0) * _k1 + (-9.0 / 10.0) * _k2 + (6.0 / 5.0) * _k3),
        y, k1, k2, k3)
    k4 = f(y4, t + dt * (3.0 / 5.0), *args, **kwargs)

    y5 = tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt * (
            (-11.0 / 54.0) * _k1 + (5.0 / 2.0) * _k2 + (-70.0 / 27.0) * _k3 + (35.0 / 27.0) * _k4),
        y, k1, k2, k3, k4
    )
    k5 = f(y5, t + dt * 1.0, *args, **kwargs)

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
    k6 = f(y6, t + dt * (7.0 / 8.0), *args, **kwargs)

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


@set_module_as('braintools.quad')
def ode_rk23_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    return_error: bool = False,
    **kwargs,
):
    """
    Bogacki–Shampine embedded Runge–Kutta 2(3) step (RK23).

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
        If True, also return a PyTree error estimate ``(y3 - y2)``.

    Returns
    -------
    PyTree or tuple
        The updated state (3rd-order). If ``return_error`` is True, returns
        ``(y_next, error_estimate)``.
    """
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()

    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, a: x + dt * 0.5 * a, y, k1)
    k2 = f(y2, t + dt * 0.5, *args, **kwargs)

    y3 = tree_map(lambda x, _k1, _k2: x + dt * (0.0 * _k1 + 0.75 * _k2), y, k1, k2)
    k3 = f(y3, t + dt * 0.75, *args, **kwargs)

    # 3rd-order solution (no k4 in combination)
    y3rd = tree_map(
        lambda x, _k1, _k2, _k3: x + dt * ((2.0 / 9.0) * _k1 + (1.0 / 3.0) * _k2 + (4.0 / 9.0) * _k3),
        y, k1, k2, k3
    )

    if not return_error:
        return y3rd

    # Compute k4 at full step for 2nd-order embedded estimate
    k4 = f(y3rd, t + dt, *args, **kwargs)
    y2nd = tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt * (
            (7.0 / 24.0) * _k1 + (1.0 / 4.0) * _k2 + (1.0 / 3.0) * _k3 + (1.0 / 8.0) * _k4),
        y, k1, k2, k3, k4
    )
    err = tree_map(lambda a, b: a - b, y3rd, y2nd)
    return y3rd, err


@set_module_as('braintools.quad')
def ode_dopri5_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    return_error: bool = False,
    **kwargs,
):
    """
    Dormand–Prince embedded Runge–Kutta 5(4) step (DOPRI5/ode45).

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
        ``(y_next, error_estimate)``.
    """
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()

    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, a: x + dt * (1.0 / 5.0) * a, y, k1)
    k2 = f(y2, t + dt * (1.0 / 5.0), *args, **kwargs)

    y3 = tree_map(lambda x, _k1, _k2: x + dt * ((3.0 / 40.0) * _k1 + (9.0 / 40.0) * _k2), y, k1, k2)
    k3 = f(y3, t + dt * (3.0 / 10.0), *args, **kwargs)

    y4 = tree_map(
        lambda x, _k1, _k2, _k3: x + dt * ((44.0 / 45.0) * _k1 + (-56.0 / 15.0) * _k2 + (32.0 / 9.0) * _k3),
        y, k1, k2, k3
    )
    k4 = f(y4, t + dt * (4.0 / 5.0), *args, **kwargs)

    y5 = tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt * ((19372.0 / 6561.0) * _k1 +
                                                (-25360.0 / 2187.0) * _k2 +
                                                (64448.0 / 6561.0) * _k3 +
                                                (-212.0 / 729.0) * _k4),
        y, k1, k2, k3, k4
    )
    k5 = f(y5, t + dt * (8.0 / 9.0), *args, **kwargs)

    y6 = tree_map(
        lambda x, _k1, _k2, _k3, _k4, _k5: x + dt * ((9017.0 / 3168.0) * _k1 +
                                                     (-355.0 / 33.0) * _k2 +
                                                     (46732.0 / 5247.0) * _k3 +
                                                     (49.0 / 176.0) * _k4 +
                                                     (-5103.0 / 18656.0) * _k5),
        y, k1, k2, k3, k4, k5
    )
    k6 = f(y6, t + dt * 1.0, *args, **kwargs)

    # Compute k7 stage at t+dt
    y7 = tree_map(
        lambda x, _k1, _k3, _k4, _k5, _k6: x + dt * ((35.0 / 384.0) * _k1 +
                                                     (500.0 / 1113.0) * _k3 +
                                                     (125.0 / 192.0) * _k4 +
                                                     (-2187.0 / 6784.0) * _k5 +
                                                     (11.0 / 84.0) * _k6),
        y, k1, k3, k4, k5, k6
    )
    k7 = f(y7, t + dt, *args, **kwargs)

    # 5th-order solution (b5)
    y5th = tree_map(
        lambda x, _k1, _k3, _k4, _k5, _k6: x + dt * ((35.0 / 384.0) * _k1 +
                                                     (500.0 / 1113.0) * _k3 +
                                                     (125.0 / 192.0) * _k4 +
                                                     (-2187.0 / 6784.0) * _k5 +
                                                     (11.0 / 84.0) * _k6),
        y, k1, k3, k4, k5, k6
    )

    if not return_error:
        return y5th

    # 4th-order embedded (b4) uses k7
    y4th = tree_map(lambda x, _k1, _k3, _k4, _k5, _k6, _k7: x + dt * ((5179.0 / 57600.0) * _k1 +
                                                                      (7571.0 / 16695.0) * _k3 +
                                                                      (393.0 / 640.0) * _k4 +
                                                                      (-92097.0 / 339200.0) * _k5 +
                                                                      (187.0 / 2100.0) * _k6 +
                                                                      (1.0 / 40.0) * _k7),
                    y, k1, k3, k4, k5, k6, k7
                    )
    err = tree_map(lambda a, b: a - b, y5th, y4th)
    return y5th, err


ode_rk45_dopri_step = ode_dopri5_step


@set_module_as('braintools.quad')
def ode_rkf45_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    return_error: bool = False,
    **kwargs,
):
    """
    Runge–Kutta–Fehlberg 4(5) embedded step (RKF45).

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
        ``(y_next, error_estimate)``.
    """
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()

    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, a: x + dt * (1.0 / 4.0) * a, y, k1)
    k2 = f(y2, t + dt * (1.0 / 4.0), *args, **kwargs)

    y3 = tree_map(
        lambda x, _k1, _k2: x + dt * ((3.0 / 32.0) * _k1 + (9.0 / 32.0) * _k2),
        y, k1, k2
    )
    k3 = f(y3, t + dt * (3.0 / 8.0), *args, **kwargs)

    y4 = tree_map(
        lambda x, _k1, _k2, _k3: x + dt * (
            (1932.0 / 2197.0) * _k1 + (-7200.0 / 2197.0) * _k2 + (7296.0 / 2197.0) * _k3),
        y, k1, k2, k3
    )
    k4 = f(y4, t + dt * (12.0 / 13.0), *args, **kwargs)

    y5 = tree_map(
        lambda x, _k1, _k2, _k3, _k4: x + dt * (
            (439.0 / 216.0) * _k1 + (-8.0) * _k2 + (3680.0 / 513.0) * _k3 + (-845.0 / 4104.0) * _k4),
        y, k1, k2, k3, k4
    )
    k5 = f(y5, t + dt * 1.0, *args, **kwargs)

    y6 = tree_map(
        lambda x, _k1, _k2, _k3, _k4, _k5: x + dt * (
            (-8.0 / 27.0) * _k1 + 2.0 * _k2 + (-3544.0 / 2565.0) * _k3 + (1859.0 / 4104.0) * _k4 + (
            -11.0 / 40.0) * _k5),
        y, k1, k2, k3, k4, k5
    )
    k6 = f(y6, t + dt * 0.5, *args, **kwargs)

    # 5th-order solution
    y5th = tree_map(
        lambda x, _k1, _k3, _k4, _k5, _k6: x + dt * (
            (16.0 / 135.0) * _k1 + (6656.0 / 12825.0) * _k3 + (28561.0 / 56430.0) * _k4 + (-9.0 / 50.0) * _k5 + (
            2.0 / 55.0) * _k6),
        y, k1, k3, k4, k5, k6
    )

    if not return_error:
        return y5th

    # 4th-order embedded
    y4th = tree_map(
        lambda x, _k1, _k3, _k4, _k5: x + dt * (
            (25.0 / 216.0) * _k1 + (1408.0 / 2565.0) * _k3 + (2197.0 / 4104.0) * _k4 + (-1.0 / 5.0) * _k5),
        y, k1, k3, k4, k5
    )
    err = tree_map(lambda a, b: a - b, y5th, y4th)
    return y5th, err


@set_module_as('braintools.quad')
def ode_ssprk33_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
):
    """
    Strong-stability-preserving RK(3,3) (Shu–Osher) step.

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
        The updated state after one SSPRK(3,3) step.
    """
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()

    k1 = f(y, t, *args, **kwargs)
    y1 = tree_map(lambda x, a: x + dt * a, y, k1)

    k2 = f(y1, t + dt, *args, **kwargs)
    y2 = tree_map(lambda x, y1_, a2: (3.0 / 4.0) * x + (1.0 / 4.0) * (y1_ + dt * a2), y, y1, k2)

    k3 = f(y2, t + dt * 0.5, *args, **kwargs)
    y3 = tree_map(lambda x, y2_, a3: (1.0 / 3.0) * x + (2.0 / 3.0) * (y2_ + dt * a3), y, y2, k3)
    return y3


@set_module_as('braintools.quad')
def ode_bs32_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    return_error: bool = False,
    **kwargs,
):
    r"""
    Bogacki–Shampine 3(2) (BS32) embedded one-step method.

    Alias of ``ode_rk23_step`` using the 3(2) naming convention. Produces a
    3rd-order solution with a 2nd-order embedded error estimate.

    Parameters
    ----------
    f, y, t, *args, return_error
        Same as for ``ode_rk23_step``.

    Returns
    -------
    PyTree or tuple
        3rd-order updated state. If ``return_error=True``, returns
        ``(y_next, error_estimate)``.
    """
    return ode_rk23_step(f, y, t, *args, return_error=return_error, **kwargs)


@set_module_as('braintools.quad')
def ode_ralston2_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
):
    r"""
    Ralston's 2nd-order Runge–Kutta method (minimized truncation error).

    Butcher tableau
    ----------------
    - c = [0, 2/3]
    - a21 = 2/3
    - b = [1/4, 3/4]

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
        The updated state after one Ralston RK2 step.
    """
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, a: x + dt * (2.0 / 3.0) * a, y, k1)
    k2 = f(y2, t + dt * (2.0 / 3.0), *args, **kwargs)
    return tree_map(lambda x, _k1, _k2: x + dt * ((1.0 / 4.0) * _k1 + (3.0 / 4.0) * _k2), y, k1, k2)


@set_module_as('braintools.quad')
def ode_ralston3_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    **kwargs
):
    r"""
    Ralston's 3rd-order Runge–Kutta method (optimized RK3).

    Butcher tableau
    ----------------
    - c = [0, 1/2, 3/4]
    - a21 = 1/2; a31 = 0, a32 = 3/4
    - b = [2/9, 1/3, 4/9]

    Note
    ----
    This RK3 equals the 3rd-order solution of the Bogacki–Shampine 3(2) pair;
    see ``ode_rk23_step``/``ode_bs32_step`` for the embedded variant.

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
        The updated state after one Ralston RK3 step.
    """
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args, **kwargs)
    y2 = tree_map(lambda x, a: x + dt * 0.5 * a, y, k1)
    k2 = f(y2, t + dt * 0.5, *args, **kwargs)
    y3 = tree_map(lambda x, a: x + dt * 0.75 * a, y, k2)
    k3 = f(y3, t + dt * 0.75, *args, **kwargs)
    return tree_map(lambda x, _k1, _k2, _k3: x + dt * ((2.0 / 9.0) * _k1 + (1.0 / 3.0) * _k2 + (4.0 / 9.0) * _k3),
                    y, k1, k2, k3)


@set_module_as('braintools.quad')
def ode_dopri8_step(
    f: ODE,
    y: brainstate.typing.PyTree,
    t: DT,
    *args,
    return_error: bool = False,
    **kwargs,
):
    """
    Dormand–Prince 8(7) (DOP853) one-step integrator with error estimate.

    Implements the explicit 8th-order Dormand–Prince method with an embedded
    lower-order estimator as used in DOP853. Coefficients are taken from the
    standard tableau and applied per PyTree leaf with broadcasting.

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(y, t, *args) -> PyTree``.
    y : PyTree
        Current state.
    t : float or brainunit.Quantity
        Current time.
    *args
        Additional positional arguments forwarded to ``f``.
    return_error : bool, default False
        If True, also return an error estimate PyTree computed from the
        embedded formulas.

    Returns
    -------
    PyTree or tuple
        8th-order solution after one step. If ``return_error`` is True,
        returns ``(y_next, error_estimate)``.

    Notes
    -----
    Uses ``dt = brainstate.environ.get_dt()`` for the step size. Error estimate
    follows the DOP853 strategy combining 5th- and 3rd-order differences.
    """
    assert callable(f), 'The input function should be callable.'
    dt = brainstate.environ.get_dt()

    # Time fractions for stages (first 12 entries are used for stages k0..k11)
    C = [
        0.0,
        0.0526001519587677318785587544488,
        0.0789002279381515978178381316732,
        0.11835034190722739672675719751,
        0.28164965809277260327324280249,
        0.333333333333333333333333333333,
        0.25,
        0.307692307692307692307692307692,
        0.651282051282051282051282051282,
        0.6,
        0.857142857142857142857142857142,
        1.0,
    ]

    # Non-zero A-coefficients per stage s: list of (j, a_sj) with j < s
    A = [
        [],
        [(0, 5.26001519587677318785587544488e-2)],
        [(0, 1.97250569845378994544595329183e-2),
         (1, 5.91751709536136983633785987549e-2)],
        [(0, 2.95875854768068491816892993775e-2),
         (2, 8.87627564304205475450678981324e-2)],
        [(0, 2.41365134159266685502369798665e-1),
         (2, -8.84549479328286085344864962717e-1),
         (3, 9.24834003261792003115737966543e-1)],
        [(0, 3.7037037037037037037037037037e-2),
         (3, 1.70828608729473871279604482173e-1),
         (4, 1.25467687566822425016691814123e-1)],
        [(0, 3.7109375e-2),
         (3, 1.70252211019544039314978060272e-1),
         (4, 6.02165389804559606850219397283e-2),
         (5, -1.7578125e-2)],
        [(0, 3.70920001185047927108779319836e-2),
         (3, 1.70383925712239993810214054705e-1),
         (4, 1.07262030446373284651809199168e-1),
         (5, -1.53194377486244017527936158236e-2),
         (6, 8.27378916381402288758473766002e-3)],
        [
            (0, 6.24110958716075717114429577812e-1),
            (3, -3.36089262944694129406857109825),
            (4, -8.68219346841726006818189891453e-1),
            (5, 2.75920996994467083049415600797e1),
            (6, 2.01540675504778934086186788979e1),
            (7, -4.34898841810699588477366255144e1)
        ],
        [
            (0, 4.77662536438264365890433908527e-1),
            (3, -2.48811461997166764192642586468),
            (4, -5.90290826836842996371446475743e-1),
            (5, 2.12300514481811942347288949897e1),
            (6, 1.52792336328824235832596922938e1),
            (7, -3.32882109689848629194453265587e1),
            (8, -2.03312017085086261358222928593e-2)
        ],
        [
            (0, -9.3714243008598732571704021658e-1),
            (3, 5.18637242884406370830023853209),
            (4, 1.09143734899672957818500254654),
            (5, -8.14978701074692612513997267357),
            (6, -1.85200656599969598641566180701e1),
            (7, 2.27394870993505042818970056734e1),
            (8, 2.49360555267965238987089396762),
            (9, -3.0467644718982195003823669022)
        ],
        [
            (0, 2.27331014751653820792359768449),
            (3, -1.05344954667372501984066689879e1),
            (4, -2.00087205822486249909675718444),
            (5, -1.79589318631187989172765950534e1),
            (6, 2.79488845294199600508499808837e1),
            (7, -2.85899827713502369474065508674),
            (8, -8.87285693353062954433549289258),
            (9, 1.23605671757943030647266201528e1),
            (10, 6.43392746015763530355970484046e-1)
        ],
    ]

    # 8th-order weights (B) corresponding to A[12, :12]
    B = [
        5.42937341165687622380535766363e-2,
        0.0, 0.0, 0.0, 0.0,
        4.45031289275240888144113950566,
        1.89151789931450038304281599044,
        -5.8012039600105847814672114227,
        3.1116436695781989440891606237e-1,
        -1.52160949662516078556178806805e-1,
        2.01365400804030348374776537501e-1,
        4.47106157277725905176885569043e-2,
    ]

    # Embedded error coefficients (length N_STAGES+1 with last entry for f_new)
    E3 = [0.0] * 13
    for j, bj in enumerate(B):
        E3[j] = bj
    E3[0] -= 0.244094488188976377952755905512
    E3[8] -= 0.733846688281611857341361741547
    E3[11] -= 0.0220588235294117647058823529412

    E5 = [0.0] * 13
    E5[0] = 0.01312004499419488073250102996
    E5[5] = -1.225156446376204440720569753
    E5[6] = -0.4957589496572501915214079952
    E5[7] = 1.664377182454986536961530415
    E5[8] = -0.3503288487499736816886487290
    E5[9] = 0.3341791187130174790297318841
    E5[10] = 0.08192320648511571246570742613
    E5[11] = -0.02235530786388629525884427845

    def affine_sum(base, coeffs, ks):
        return jax.tree.map(
            lambda b, *leaves: b + dt * sum(c * v for c, v in zip(coeffs, leaves)),
            base, *ks,
            is_leaf=u.math.is_quantity
        )

    # Stages K[0..11], plus K[12] = f(y8, t+dt)
    K = [None] * 13
    K[0] = f(y, t, *args, **kwargs)

    for s in range(1, 12):
        row = A[s]
        if row:
            coeffs = [c for (_, c) in row]
            ks = [K[j] for (j, _) in row]
            y_s = affine_sum(y, coeffs, ks)
        else:
            y_s = y
        t_s = t + C[s] * dt
        K[s] = f(y_s, t_s, *args, **kwargs)

    # 8th-order solution and final stage
    y8 = affine_sum(y, B, K[:12])
    K[12] = f(y8, t + dt, *args, **kwargs)

    if not return_error:
        return y8

    def lincomb(weights):
        return jax.tree.map(
            lambda *leaves: sum(w * v for w, v in zip(weights, leaves)), *K,
            is_leaf=u.math.is_quantity
        )

    err5 = lincomb(E5)
    err3 = lincomb(E3)

    def err_leaf(e5, e3):
        e5_abs = u.math.abs(e5)
        e3_abs = u.math.abs(e3)
        denom = u.math.sqrt(e5_abs * e5_abs + (0.1 * e3_abs) * (0.1 * e3_abs))
        corr = u.math.where(denom > 0, e5_abs / denom, u.math.ones_like(denom))
        return dt * e5 * corr

    err = jax.tree.map(err_leaf, err5, err3, is_leaf=u.math.is_quantity)
    return y8, err


ode_rk87_dopri_step = ode_dopri8_step
