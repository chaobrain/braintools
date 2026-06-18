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

**Quick Start - ODE Integration:**

.. code-block:: python

    import brainstate as bst
    import jax.numpy as jnp
    from braintools.quad import ode_euler_step, ode_rk4_step

    # Set global time step (dimensionless for this simple scalar ODE)
    bst.environ.set(dt=0.01)

    # Define ODE: dy/dt = -y + sin(t)
    def f(y, t):
        return -y + jnp.sin(t)

    # Simple Euler integration
    y = 0.0
    t = 0.0
    for _ in range(100):
        y = ode_euler_step(f, y, t)
        t += bst.environ.get_dt()

    # Higher accuracy with RK4
    y = 0.0
    t = 0.0
    for _ in range(100):
        y = ode_rk4_step(f, y, t)
        t += bst.environ.get_dt()

**Quick Start - SDE Integration:**

.. code-block:: python

    import brainstate as bst
    from braintools.quad import sde_euler_step, sde_milstein_step

    # Set global time step (dimensionless for this simple scalar SDE)
    bst.environ.set(dt=0.1)

    # Define SDE: dy = -y*dt + 0.5*dW
    def drift(y, t):
        return -y

    def diffusion(y, t):
        return 0.5

    # Euler-Maruyama integration
    y = 1.0
    t = 0.0
    for _ in range(1000):
        y = sde_euler_step(drift, diffusion, y, t)
        t += bst.environ.get_dt()

    # Higher accuracy with Milstein
    y = 1.0
    t = 0.0
    for _ in range(1000):
        y = sde_milstein_step(drift, diffusion, y, t)
        t += bst.environ.get_dt()

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

    # Heun's method (strong order 0.5, better weak order)
    V = sde_heun_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

    # Stochastic Runge-Kutta methods
    V = sde_srk2_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)
    V = sde_srk3_step(drift, diffusion, V, t, I_mean=0.5 * u.nA)

    # Tamed Euler (for stiff SDEs)
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
    from collections import deque
    from braintools.quad import (
        dde_euler_step, dde_heun_step, dde_rk4_step,
        dde_euler_pc_step, dde_heun_pc_step
    )

    bst.environ.set(dt=0.1)

    # Delayed feedback system: dy/dt = -y(t) + tanh(y(t - delay))
    delay = 5.0
    dt = bst.environ.get_dt()
    n_hist = int(delay / dt) + 1

    # History buffers seeded over the delay interval (constant IC y = 0.1)
    history = deque([0.1] * n_hist, maxlen=n_hist)
    times = deque([-delay + i * dt for i in range(n_hist)], maxlen=n_hist)

    # History lookup: nearest stored sample, clamped to the buffer.
    # Replace with proper interpolation for production use.
    def history_fn(t_past):
        idx = int(round((t_past - times[0]) / dt))
        idx = min(max(idx, 0), len(history) - 1)
        return history[idx]

    # DDE right-hand side
    def f(t, y, y_delayed):
        return -y + jnp.tanh(y_delayed)

    # Integration loop
    y = 0.1
    t = 0.0
    for _ in range(100):
        # Euler method for DDEs
        y_new = dde_euler_step(f, y, t, history_fn, delays=delay)

        # Or use higher-order / predictor-corrector methods:
        # y_new = dde_heun_step(f, y, t, history_fn, delays=delay)
        # y_new = dde_rk4_step(f, y, t, history_fn, delays=delay)
        # y_new = dde_euler_pc_step(f, y, t, history_fn, delays=delay)

        # Update history
        history.append(y_new)
        times.append(t)
        y = y_new
        t += dt

    # Multiple delays example
    def f_multi(t, y, y_delay1, y_delay2):
        return -y + 0.5 * jnp.tanh(y_delay1) + 0.3 * jnp.sin(y_delay2)

    y_new = dde_euler_step(f_multi, y, t, history_fn, delays=[5.0, 10.0])

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
