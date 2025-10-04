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
Delay differential equation (DDE) one-step integrators.

These steppers solve delay differential equations of the form:

    y'(t) = f(t, y(t), y(t-τ₁), y(t-τ₂), ..., *args)

where y(t-τᵢ) represents the delayed state at time t-τᵢ.

All steppers operate on arbitrary JAX PyTrees and use the global time step
``dt`` from ``brainstate.environ``. History interpolation is handled by
user-provided history functions that can look up past states.

Key concepts:
- History function: callable that returns y(t-delay) for any requested time
- Multiple delays: DDEs can have multiple delay terms
- Initial conditions: require specification of solution over delay interval
"""

from typing import Callable, Union, Sequence

import brainstate

from braintools._misc import set_module_as, tree_map

__all__ = [
    'dde_euler_step',
    'dde_heun_step',
    'dde_rk4_step',
    'dde_euler_pc_step',
    'dde_heun_pc_step',
]

DT = brainstate.typing.ArrayLike
PyTree = brainstate.typing.PyTree
HistoryFn = Callable[[DT], PyTree]
DDE = Callable[[DT, PyTree, PyTree, ...], PyTree]


@set_module_as('braintools.quad')
def dde_euler_step(
    f: DDE,
    y: PyTree,
    t: DT,
    history_fn: HistoryFn,
    delays: Union[DT, Sequence[DT]],
    *args,
    **kwargs
) -> PyTree:
    """
    Explicit Euler step for delay differential equations.
    
    Implements a single forward Euler step for DDEs of the form:
    
        dy/dt = f(t, y(t), y(t-τ), *args)
        
    or with multiple delays:
    
        dy/dt = f(t, y(t), y(t-τ₁), y(t-τ₂), ..., *args)

    Parameters
    ----------
    f : callable
        Right-hand side function ``f(t, y, y_delayed_1, y_delayed_2, ..., *args)``
        that computes the time-derivative. The delayed terms are passed as
        separate arguments in the order of the delays list.
    y : PyTree
        Current state at time ``t``.
    t : float or brainunit.Quantity
        Current time.
    history_fn : callable
        Function that returns the solution at past times: ``history_fn(t_past) -> PyTree``.
        Should handle interpolation for non-grid times.
    delays : float, brainunit.Quantity, or sequence thereof
        Delay value(s) τ. If a sequence, multiple delayed terms y(t-τᵢ) will
        be passed to f in order.
    *args
        Additional arguments forwarded to ``f``.

    Returns
    -------
    PyTree
        The updated state ``y_{n+1}`` after one Euler step.
        
    Examples
    --------
    Single delay case:
    
    >>> def f(t, y, y_delayed):
    ...     return -y + y_delayed  # Simple delayed feedback
    >>> y_next = dde_euler_step(f, y, t, history_fn, delay=1.0)
    
    Multiple delays case:
    
    >>> def f(t, y, y_delay1, y_delay2):
    ...     return -y + 0.5*y_delay1 + 0.3*y_delay2
    >>> y_next = dde_euler_step(f, y, t, history_fn, delays=[1.0, 2.0])
    """
    dt = brainstate.environ.get_dt()

    # Handle single delay vs multiple delays
    if not isinstance(delays, (list, tuple)):
        delays = [delays]

    # Get delayed states
    delayed_states = [history_fn(t - delay) for delay in delays]

    # Compute derivative
    dydt = f(t, y, *delayed_states, *args, **kwargs)

    # Euler update
    return tree_map(lambda y_val, dy_val: y_val + dt * dy_val, y, dydt)


@set_module_as('braintools.quad')
def dde_heun_step(
    f: DDE,
    y: PyTree,
    t: DT,
    history_fn: HistoryFn,
    delays: Union[DT, Sequence[DT]],
    *args,
    **kwargs
) -> PyTree:
    """
    Heun's method (improved Euler) for delay differential equations.
    
    Second-order Runge-Kutta method for DDEs. Uses Euler predictor followed
    by trapezoidal corrector.

    Parameters
    ----------
    f, y, t, history_fn, delays, *args
        Same as dde_euler_step.

    Returns
    -------
    PyTree
        The updated state after one Heun step (second-order accurate).
        
    Notes
    -----
    The method computes:
    1. Predictor: y₁ = y₀ + h*f(t₀, y₀, y(t₀-τ))
    2. Corrector: y₁ = y₀ + h/2*[f(t₀, y₀, y(t₀-τ)) + f(t₁, y₁, y(t₁-τ))]
    """
    dt = brainstate.environ.get_dt()

    # Handle single delay vs multiple delays
    if not isinstance(delays, (list, tuple)):
        delays = [delays]

    # Get delayed states at current time
    delayed_states_t = [history_fn(t - delay) for delay in delays]

    # Predictor step (Euler)
    k1 = f(t, y, *delayed_states_t, *args, **kwargs)
    y_pred = tree_map(lambda y_val, k1_val: y_val + dt * k1_val, y, k1)

    # Get delayed states at next time (for corrector)
    delayed_states_t1 = [history_fn(t + dt - delay) for delay in delays]

    # Corrector step
    k2 = f(t + dt, y_pred, *delayed_states_t1, *args, **kwargs)

    # Combine predictor and corrector
    return tree_map(
        lambda y_val, k1_val, k2_val: y_val + 0.5 * dt * (k1_val + k2_val),
        y, k1, k2
    )


@set_module_as('braintools.quad')
def dde_rk4_step(
    f: DDE,
    y: PyTree,
    t: DT,
    history_fn: HistoryFn,
    delays: Union[DT, Sequence[DT]],
    *args,
    **kwargs
) -> PyTree:
    """
    Fourth-order Runge-Kutta method for delay differential equations.
    
    Classic RK4 extended to handle delayed terms.

    Parameters
    ----------
    f, y, t, history_fn, delays, *args
        Same as dde_euler_step.

    Returns
    -------
    PyTree
        The updated state after one RK4 step (fourth-order accurate).
        
    Notes
    -----
    Uses the standard RK4 tableau with delayed terms evaluated at the
    appropriate times for each stage.
    """
    dt = brainstate.environ.get_dt()

    # Handle single delay vs multiple delays
    if not isinstance(delays, (list, tuple)):
        delays = [delays]

    # Stage 1
    delayed_1 = [history_fn(t - delay) for delay in delays]
    k1 = f(t, y, *delayed_1, *args, **kwargs)

    # Stage 2  
    y2 = tree_map(lambda y_val, k1_val: y_val + 0.5 * dt * k1_val, y, k1)
    delayed_2 = [history_fn(t + 0.5 * dt - delay) for delay in delays]
    k2 = f(t + 0.5 * dt, y2, *delayed_2, *args, **kwargs)

    # Stage 3
    y3 = tree_map(lambda y_val, k2_val: y_val + 0.5 * dt * k2_val, y, k2)
    delayed_3 = [history_fn(t + 0.5 * dt - delay) for delay in delays]
    k3 = f(t + 0.5 * dt, y3, *delayed_3, *args, **kwargs)

    # Stage 4
    y4 = tree_map(lambda y_val, k3_val: y_val + dt * k3_val, y, k3)
    delayed_4 = [history_fn(t + dt - delay) for delay in delays]
    k4 = f(t + dt, y4, *delayed_4, *args, **kwargs)

    # Combine stages
    return tree_map(
        lambda y_val, k1_val, k2_val, k3_val, k4_val:
        y_val + dt / 6.0 * (k1_val + 2 * k2_val + 2 * k3_val + k4_val),
        y, k1, k2, k3, k4
    )


@set_module_as('braintools.quad')
def dde_euler_pc_step(
    f: DDE,
    y: PyTree,
    t: DT,
    history_fn: HistoryFn,
    delays: Union[DT, Sequence[DT]],
    *args,
    max_iter: int = 3,
    **kwargs
) -> PyTree:
    """
    Euler predictor-corrector method for delay differential equations.
    
    Uses explicit Euler as predictor and implicit Euler as corrector,
    with fixed-point iteration to solve the implicit equation.

    Parameters
    ----------
    f, y, t, history_fn, delays, *args
        Same as dde_euler_step.
    max_iter : int, default 3
        Maximum number of corrector iterations.

    Returns
    -------
    PyTree
        The updated state after predictor-corrector step.
        
    Notes
    -----
    This method can be more stable for stiff DDEs compared to explicit methods.
    The corrector equation is:
        y_{n+1} = y_n + h*f(t_{n+1}, y_{n+1}, y(t_{n+1}-τ))
    """
    dt = brainstate.environ.get_dt()

    # Handle single delay vs multiple delays
    if not isinstance(delays, (list, tuple)):
        delays = [delays]

    # Predictor step (explicit Euler)
    delayed_states_t = [history_fn(t - delay) for delay in delays]
    k_pred = f(t, y, *delayed_states_t, *args, **kwargs)
    y_pred = tree_map(lambda y_val, k_val: y_val + dt * k_val, y, k_pred)

    # Corrector iterations (implicit Euler)
    y_corr = y_pred
    delayed_states_t1 = [history_fn(t + dt - delay) for delay in delays]

    for _ in range(max_iter):
        k_corr = f(t + dt, y_corr, *delayed_states_t1, *args, **kwargs)
        y_corr = tree_map(lambda y_val, k_val: y_val + dt * k_val, y, k_corr)

    return y_corr


@set_module_as('braintools.quad')
def dde_heun_pc_step(
    f: DDE,
    y: PyTree,
    t: DT,
    history_fn: HistoryFn,
    delays: Union[DT, Sequence[DT]],
    *args,
    max_iter: int = 3,
    **kwargs
) -> PyTree:
    """
    Heun predictor-corrector method for delay differential equations.
    
    Uses explicit Heun as predictor and implicit trapezoidal rule as corrector.

    Parameters
    ----------
    f, y, t, history_fn, delays, *args
        Same as dde_euler_step.
    max_iter : int, default 3
        Maximum number of corrector iterations.

    Returns
    -------
    PyTree
        The updated state after Heun predictor-corrector step.
        
    Notes
    -----
    Higher-order predictor-corrector method that combines the stability
    of implicit methods with good accuracy. The corrector equation is:
        y_{n+1} = y_n + h/2*[f(t_n, y_n, y(t_n-τ)) + f(t_{n+1}, y_{n+1}, y(t_{n+1}-τ))]
    """
    dt = brainstate.environ.get_dt()

    # Handle single delay vs multiple delays
    if not isinstance(delays, (list, tuple)):
        delays = [delays]

    # Predictor step (explicit Heun)
    y_pred = dde_heun_step(f, y, t, history_fn, delays, *args, **kwargs)

    # Corrector iterations (implicit trapezoidal)
    y_corr = y_pred
    delayed_states_t = [history_fn(t - delay) for delay in delays]
    delayed_states_t1 = [history_fn(t + dt - delay) for delay in delays]

    k1 = f(t, y, *delayed_states_t, *args, **kwargs)

    for _ in range(max_iter):
        k2 = f(t + dt, y_corr, *delayed_states_t1, *args, **kwargs)
        y_corr = tree_map(
            lambda y_val, k1_val, k2_val: y_val + 0.5 * dt * (k1_val + k2_val),
            y, k1, k2
        )

    return y_corr
