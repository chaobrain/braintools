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
Lightweight One-Step Integrators for ODEs, SDEs, DDEs, and IMEX Systems.

This module provides a comprehensive collection of compact, JAX-friendly stepping
functions for numerical integration of differential equations. All steppers operate
directly on JAX PyTrees and use the global time step ``dt`` from ``brainstate.environ``,
making them ideal for simulation loops with minimal boilerplate.

**Key Features:**

- **Ordinary Differential Equations (ODEs)**: Euler, Runge-Kutta families, adaptive methods
- **Stochastic Differential Equations (SDEs)**: Euler-Maruyama, Milstein, stochastic RK
- **Implicit-Explicit (IMEX)**: Split methods for stiff/nonstiff systems
- **Delay Differential Equations (DDEs)**: Methods with history interpolation
- **PyTree Compatible**: Works with arbitrary nested state structures
- **Unit-Aware**: Full integration with BrainUnit for physical quantities
- **JAX-Optimized**: JIT-compatible, vectorizable, and differentiable
- **Trajectory drivers**: :func:`odeint` (fixed step, one compiled ``scan``)
  and :func:`solve_ivp` (adaptive step size with error control)
- **Method registry**: :func:`get_method`, :func:`register_method` and
  :func:`list_methods` look steppers up by short name (``'rk4'``, ``'dopri5'``, ...)

**Quick Start - ODE Trajectories** (one compiled ``scan``, no Python loop):

.. code-block:: python

    >>> import jax.numpy as jnp
    >>> from braintools.quad import odeint, solve_ivp
    >>> def f(y, t):
    ...     return -y + jnp.sin(t)
    >>> ts, ys = odeint(f, jnp.array(0.0), 0.0, 1.0, 0.01, method='rk4')
    >>> ts.shape, ys.shape
    ((101,), (101,))
    >>> sol = solve_ivp(f, (0.0, 1.0), jnp.array(0.0), t_eval=jnp.array([0.5, 1.0]))
    >>> print(f'{float(ys[-1]):.5f} {float(sol.ys[-1]):.5f}')
    0.33452 0.33452

The single-step functions below are the building blocks: ``odeint`` accepts
any registered ODE stepper by name (``method='euler'``, ``'rk4'``, ...).

**Quick Start - SDE Integration:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.quad import sde_euler_step, sde_milstein_step

    # Set global time step (dimensionless for this simple scalar SDE)
    bst.environ.set(dt=0.1)

    # Define SDE: dy = -y*dt + 0.5*dW
    def drift(y, t):
        return -y

    def diffusion(y, t):
        return 0.5

    # Drive a stepper with a compiled scan (not a Python loop)
    def simulate(step, y0, n_steps):
        def body(y, i):
            y = step(drift, diffusion, y, i * bst.environ.get_dt())
            return y, y
        _, ys = bst.transform.scan(body, y0, jnp.arange(n_steps))
        return ys

    run = bst.transform.jit(simulate, static_argnums=(0, 2))
    ys_em = run(sde_euler_step, jnp.array(1.0), 1000)       # Euler-Maruyama
    ys_mil = run(sde_milstein_step, jnp.array(1.0), 1000)   # Milstein

**ODE Integrators:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import (
        ode_euler_step, ode_rk2_step, ode_rk3_step, ode_rk4_step,
        ode_midpoint_step, ode_heun_step, ode_rk4_38_step,
        ode_expeuler_step, ode_dopri5_step, ode_rk23_step
    )

    bst.environ.set(dt=0.01 * u.ms)

    # A simple leaky-integrator neuron model
    def neuron_ode(V, t, I_ext=0.0 * u.nA):
        tau = 20.0 * u.ms
        V_rest = -65.0 * u.mV
        R = 10.0 * u.Mohm
        return (V_rest - V + R * I_ext) / tau

    V = -65.0 * u.mV
    t = 0.0 * u.ms

    # First-order methods
    V = ode_euler_step(neuron_ode, V, t, I_ext=0.5 * u.nA)

    # Second-order methods
    V = ode_rk2_step(neuron_ode, V, t, I_ext=0.5 * u.nA)
    V = ode_midpoint_step(neuron_ode, V, t, I_ext=0.5 * u.nA)

    # Third-order methods
    V = ode_rk3_step(neuron_ode, V, t, I_ext=0.5 * u.nA)
    V = ode_heun_step(neuron_ode, V, t, I_ext=0.5 * u.nA)  # Heun's RK3

    # Fourth-order methods
    V = ode_rk4_step(neuron_ode, V, t, I_ext=0.5 * u.nA)
    V = ode_rk4_38_step(neuron_ode, V, t, I_ext=0.5 * u.nA)

    # Adaptive methods (embedded Runge-Kutta)
    V = ode_rk23_step(neuron_ode, V, t, I_ext=0.5 * u.nA)  # Bogacki-Shampine
    V = ode_dopri5_step(neuron_ode, V, t, I_ext=0.5 * u.nA)  # Dormand-Prince

    # Exponential Euler takes a single drift function and linearizes it
    # internally (well suited to stiff, near-linear dynamics).
    V = ode_expeuler_step(neuron_ode, V, t, I_ext=0.5 * u.nA)

**SDE Integrators:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import (
        sde_euler_step, sde_milstein_step,
        sde_expeuler_step, sde_heun_step,
        sde_srk2_step, sde_srk3_step, sde_tamed_euler_step
    )

    bst.environ.set(dt=0.01 * u.ms)

    # Stochastic neuron with current noise. The steppers forward extra kwargs to
    # *both* drift and diffusion, so both accept **kwargs. A diffusion coefficient
    # has units of [state] / sqrt([time]).
    def drift(V, t, I_mean=0.0 * u.nA, **kwargs):
        tau = 20.0 * u.ms
        V_rest = -65.0 * u.mV
        R = 10.0 * u.Mohm
        return (V_rest - V + R * I_mean) / tau

    def diffusion(V, t, noise_sigma=0.1, **kwargs):
        return noise_sigma * u.mV / u.ms ** 0.5

    V = -65.0 * u.mV
    t = 0.0 * u.ms

    # Euler-Maruyama (strong order 0.5)
    V = sde_euler_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

    # Milstein (strong order 1.0)
    V = sde_milstein_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

    # Heun's method (Ito: strong order 0.5; sde_type='stra': strong order 1.0)
    V = sde_heun_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

    # Stochastic Runge-Kutta methods (Stratonovich, strong order 1.0)
    V = sde_srk2_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)
    V = sde_srk3_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

    # Tamed Euler (for superlinearly growing drifts)
    V = sde_tamed_euler_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

    # Exponential Euler (drift is linearized internally; signature is
    # sde_expeuler_step(drift, diffusion, y, t, *args))
    V = sde_expeuler_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

**IMEX Integrators:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import (
        imex_euler_step, imex_ars222_step, imex_cnab_step
    )

    bst.environ.set(dt=0.01 * u.ms)

    # Split the leaky-integrator neuron V' = (V_rest - V + R*I)/tau into a
    # nonstiff input/reversal part (explicit) and the stiff decay -V/tau
    # (implicit). Both parts are rates with units [state]/[time]. The steppers
    # forward extra kwargs to both parts, so both accept **kwargs.

    # Explicit (nonstiff) part
    def f_explicit(V, t, I_ext=0.0 * u.nA, **kwargs):
        tau = 20.0 * u.ms
        V_rest = -65.0 * u.mV
        R = 10.0 * u.Mohm
        return (V_rest + R * I_ext) / tau

    # Implicit (stiff) part
    def f_implicit(V, t, **kwargs):
        tau = 20.0 * u.ms
        return -V / tau

    V = -65.0 * u.mV
    t = 0.0 * u.ms

    # First-order IMEX Euler
    V = imex_euler_step(f_explicit, f_implicit, V, t, I_ext=0.5 * u.nA)

    # Second-order ARS(2,2,2) method
    V = imex_ars222_step(f_explicit, f_implicit, V, t, I_ext=0.5 * u.nA)

    # Crank-Nicolson + Adams-Bashforth (multistep: also needs the previous
    # state y_{n-1}; on the first step pass the current state)
    V_prev = V
    V = imex_cnab_step(f_explicit, f_implicit, V, V_prev, t, I_ext=0.5 * u.nA)

**DDE Integrators:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.quad import (
        dde_euler_step, dde_heun_step, dde_rk4_step,
        dde_euler_pc_step, dde_heun_pc_step
    )

    bst.environ.set(dt=0.1)

    # Delayed feedback system: dy/dt = -y(t) + tanh(y(t - delay))
    delay = 5.0
    dt = bst.environ.get_dt()
    n_hist = int(round(delay / dt)) + 1

    # DDE right-hand side
    def f(t, y, y_delayed):
        return -y + jnp.tanh(y_delayed)

    # One step. The history is a ring buffer carried through the scan:
    # buf[k] holds y(t - delay + k * dt). The lookup takes the nearest sample,
    # clamped to the buffer; replace it with interpolation for production use.
    def body(carry, i):
        y, buf = carry
        t = i * dt

        def history_fn(t_past):
            k = jnp.clip(jnp.round((t_past - (t - delay)) / dt).astype(int), 0, n_hist - 1)
            return buf[k]

        y_new = dde_euler_step(f, y, t, history_fn, delays=delay)
        # Or use higher-order / predictor-corrector methods:
        # y_new = dde_heun_step(f, y, t, history_fn, delays=delay)
        # y_new = dde_rk4_step(f, y, t, history_fn, delays=delay)
        # y_new = dde_euler_pc_step(f, y, t, history_fn, delays=delay)
        buf = jnp.roll(buf, -1).at[-1].set(y_new)
        return (y_new, buf), y_new

    # Constant initial history y = 0.1; 100 steps in one compiled scan
    y0 = jnp.asarray(0.1)
    _, ys = bst.transform.scan(body, (y0, jnp.full(n_hist, 0.1)), jnp.arange(100))

    # Multiple delays: pass a list, size the buffer for the longest delay, and
    # give the right-hand side one delayed argument per delay
    def f_multi(t, y, y_delay1, y_delay2):
        return -y + 0.5 * jnp.tanh(y_delay1) + 0.3 * jnp.sin(y_delay2)
    # y_new = dde_euler_step(f_multi, y, t, history_fn, delays=[5.0, 10.0])

**PyTree State Integration:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import ode_rk4_step, sde_euler_step

    bst.environ.set(dt=0.01 * u.ms)

    # State as a PyTree (dictionary) with mixed physical units
    state = {
        'V': -65.0 * u.mV,
        'Ca': 0.1 * u.uM,
    }

    # ODE for the PyTree state (each leaf is a rate [state]/[time])
    def neuron_dynamics(state, t, I_ext=0.0 * u.nA):
        V, Ca = state['V'], state['Ca']
        tau_V = 20.0 * u.ms
        tau_Ca = 50.0 * u.ms
        R = 10.0 * u.Mohm

        dV = (-65.0 * u.mV - V + R * I_ext) / tau_V
        dCa = (-Ca + 0.1 * u.uM) / tau_Ca

        return {'V': dV, 'Ca': dCa}

    # Integration preserves PyTree structure
    state = ode_rk4_step(neuron_dynamics, state, 0.0 * u.ms, I_ext=1.0 * u.nA)

    # SDE with PyTree state (diffusion units are [state]/sqrt([time]))
    def drift(state, t):
        return neuron_dynamics(state, t, I_ext=0.5 * u.nA)

    def diffusion(state, t):
        return {
            'V': 0.1 * u.mV / u.ms ** 0.5,
            'Ca': 0.01 * u.uM / u.ms ** 0.5,
        }

    state = sde_euler_step(drift, diffusion, state, 0.0 * u.ms)

**Adaptive Time Stepping:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.quad import (
        ode_rk23_step, ode_rk45_step, ode_dopri5_step, ode_dopri8_step
    )

    bst.environ.set(dt=0.01)

    # Pass return_error=True to embedded methods to also get an error estimate.
    def f(y, t):
        return -y + jnp.sin(10 * t)

    y = 1.0
    t = 0.0

    # RK23 (Bogacki-Shampine 2(3))
    y_new, err = ode_rk23_step(f, y, t, return_error=True)

    # RK45 (Cash-Karp 4(5)) and DOPRI5 (Dormand-Prince 5(4))
    y_new = ode_rk45_step(f, y, t)
    y_new = ode_dopri5_step(f, y, t)  # alias: ode_rk45_dopri_step

    # DOP853 (Dormand-Prince 8(7)) - high accuracy
    y_new = ode_dopri8_step(f, y, t)

**Strong Stability Preserving Methods:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.quad import ode_ssprk33_step

    bst.environ.set(dt=0.001)

    # SSPRK(3,3) - third-order SSP Runge-Kutta
    # Useful for problems with discontinuities or shocks
    def f(y, t):
        # Some hyperbolic PDE discretization
        return -jnp.roll(y, 1) + y

    y = jnp.ones(100)
    t = 0.0

    y = ode_ssprk33_step(f, y, t)

"""

# Trajectory drivers and the method registry
from ._odeint import odeint
from ._registry import (
    register_method,
    get_method,
    list_methods,
)
from ._solve_ivp import (
    ODESolution,
    solve_ivp,
)

# ODE integrators
from ._ode_integrator import (
    ode_euler_step,
    ode_rk2_step,
    ode_rk3_step,
    ode_rk4_step,
    ode_expeuler_step,
    ode_midpoint_step,
    ode_heun_step,
    ode_rk4_38_step,
    ode_rk45_step,
    ode_rk23_step,
    ode_dopri5_step,
    ode_rk45_dopri_step,
    ode_rkf45_step,
    ode_ssprk33_step,
    ode_dopri8_step,
    ode_rk87_dopri_step,
    ode_bs32_step,
    ode_ralston2_step,
    ode_ralston3_step,
)

# SDE integrators
from ._sde_integrator import (
    sde_euler_step,
    sde_milstein_step,
    sde_expeuler_step,
    sde_heun_step,
    sde_tamed_euler_step,
    sde_implicit_euler_step,
    sde_srk2_step,
    sde_srk3_step,
    sde_srk4_step,
)

# IMEX integrators
from ._imex_integrator import (
    imex_euler_step,
    imex_ars222_step,
    imex_cnab_step,
)

# DDE integrators
from ._dde_integrator import (
    dde_euler_step,
    dde_heun_step,
    dde_rk4_step,
    dde_euler_pc_step,
    dde_heun_pc_step,
)

__all__ = [
    # Trajectory drivers and the method registry
    'odeint',
    'solve_ivp',
    'ODESolution',
    'register_method',
    'get_method',
    'list_methods',

    # ODE integrators - Basic methods
    'ode_euler_step',
    'ode_rk2_step',
    'ode_rk3_step',
    'ode_rk4_step',
    'ode_expeuler_step',
    'ode_midpoint_step',
    'ode_heun_step',
    'ode_rk4_38_step',

    # ODE integrators - Adaptive/embedded methods
    'ode_rk45_step',
    'ode_rk23_step',
    'ode_dopri5_step',
    'ode_rk45_dopri_step',
    'ode_rkf45_step',
    'ode_ssprk33_step',
    'ode_dopri8_step',
    'ode_rk87_dopri_step',
    'ode_bs32_step',
    'ode_ralston2_step',
    'ode_ralston3_step',

    # SDE integrators
    'sde_euler_step',
    'sde_milstein_step',
    'sde_expeuler_step',
    'sde_heun_step',
    'sde_tamed_euler_step',
    'sde_implicit_euler_step',
    'sde_srk2_step',
    'sde_srk3_step',
    'sde_srk4_step',

    # IMEX integrators
    'imex_euler_step',
    'imex_ars222_step',
    'imex_cnab_step',

    # DDE integrators
    'dde_euler_step',
    'dde_heun_step',
    'dde_rk4_step',
    'dde_euler_pc_step',
    'dde_heun_pc_step',
]
