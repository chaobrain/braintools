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
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import ode_euler_step, ode_rk4_step

    # Set global time step
    bst.environ.set(dt=0.01 * u.ms)

    # Define ODE: dy/dt = -y + sin(t)
    def f(y, t):
        return -y + jnp.sin(t)

    # Simple Euler integration
    y = 0.0
    t = 0.0 * u.ms
    for _ in range(100):
        y = ode_euler_step(f, y, t)
        t += bst.environ.get_dt()

    # Higher accuracy with RK4
    y = 0.0
    t = 0.0 * u.ms
    for _ in range(100):
        y = ode_rk4_step(f, y, t)
        t += bst.environ.get_dt()

**Quick Start - SDE Integration:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import sde_euler_step, sde_milstein_step

    # Set global time step
    bst.environ.set(dt=0.1 * u.ms)

    # Define SDE: dy = -y*dt + 0.5*dW
    def drift(y, t):
        return -y

    def diffusion(y, t):
        return 0.5

    # Euler-Maruyama integration
    y = 1.0
    t = 0.0 * u.ms
    for _ in range(1000):
        y = sde_euler_step(drift, diffusion, y, t)
        t += bst.environ.get_dt()

    # Higher accuracy with Milstein
    y = 1.0
    t = 0.0 * u.ms
    for _ in range(1000):
        y = sde_milstein_step(drift, diffusion, y, t)
        t += bst.environ.get_dt()

**ODE Integrators:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import (
        ode_euler_step, ode_rk2_step, ode_rk4_step,
        ode_midpoint_step, ode_heun_step,
        ode_rk4_38_step, ode_expeuler_step,
        ode_dopri5_step, ode_rk23_step
    )

    bst.environ.set(dt=0.01 * u.ms)

    # Define a simple neuron model
    def neuron_ode(V, t, I_ext=0.0):
        tau = 20.0 * u.ms
        V_rest = -65.0 * u.mV
        R = 10.0 * u.MOhm
        return (V_rest - V + R * I_ext) / tau

    V = -65.0 * u.mV
    t = 0.0 * u.ms

    # First-order methods
    V = ode_euler_step(neuron_ode, V, t, I_ext=0.5 * u.nA)

    # Second-order methods
    V = ode_rk2_step(neuron_ode, V, t, I_ext=0.5 * u.nA)
    V = ode_midpoint_step(neuron_ode, V, t, I_ext=0.5 * u.nA)
    V = ode_heun_step(neuron_ode, V, t, I_ext=0.5 * u.nA)

    # Fourth-order methods
    V = ode_rk4_step(neuron_ode, V, t, I_ext=0.5 * u.nA)
    V = ode_rk4_38_step(neuron_ode, V, t, I_ext=0.5 * u.nA)

    # Adaptive methods (embedded Runge-Kutta)
    V = ode_rk23_step(neuron_ode, V, t, I_ext=0.5 * u.nA)  # Bogacki-Shampine
    V = ode_dopri5_step(neuron_ode, V, t, I_ext=0.5 * u.nA)  # Dormand-Prince

    # Exponential Euler (for stiff linear parts)
    def linear_coeff(V, t):
        tau = 20.0 * u.ms
        return -1.0 / tau

    def nonlinear_part(V, t, I_ext=0.0):
        tau = 20.0 * u.ms
        V_rest = -65.0 * u.mV
        R = 10.0 * u.MOhm
        return (V_rest + R * I_ext) / tau

    V = ode_expeuler_step(linear_coeff, nonlinear_part, V, t, I_ext=0.5 * u.nA)

**SDE Integrators:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import (
        sde_euler_step, sde_milstein_step,
        sde_expeuler_step, sde_heun_step,
        sde_srk2_step, sde_tamed_euler_step
    )

    bst.environ.set(dt=0.01 * u.ms)

    # Stochastic neuron with current noise
    def drift(V, t, I_mean=0.0):
        tau = 20.0 * u.ms
        V_rest = -65.0 * u.mV
        R = 10.0 * u.MOhm
        return (V_rest - V + R * I_mean) / tau

    def diffusion(V, t, noise_sigma=0.1):
        return noise_sigma * u.mV / u.ms

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

    # Exponential Euler (linearized drift)
    def linear_drift(V, t):
        tau = 20.0 * u.ms
        return -1.0 / tau

    def nonlinear_drift(V, t, I_mean=0.0):
        tau = 20.0 * u.ms
        V_rest = -65.0 * u.mV
        R = 10.0 * u.MOhm
        return (V_rest + R * I_mean) / tau

    V = sde_expeuler_step(linear_drift, nonlinear_drift, diffusion, V, t)

**IMEX Integrators:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import (
        imex_euler_step, imex_ars222_step, imex_cnab_step
    )

    bst.environ.set(dt=0.01 * u.ms)

    # Split system: stiff linear + nonstiff nonlinear
    # Example: V' = -V/tau (stiff) + f(V) (nonstiff)

    # Explicit (nonstiff) part
    def f_explicit(V, t, I_ext=0.0):
        V_rest = -65.0 * u.mV
        R = 10.0 * u.MOhm
        return V_rest + R * I_ext

    # Implicit (stiff) part
    def f_implicit(V, t):
        tau = 20.0 * u.ms
        return -V / tau

    V = -65.0 * u.mV
    t = 0.0 * u.ms

    # First-order IMEX Euler
    V = imex_euler_step(f_explicit, f_implicit, V, t, I_ext=0.5 * u.nA)

    # Second-order ARS(2,2,2) method
    V = imex_ars222_step(f_explicit, f_implicit, V, t, I_ext=0.5 * u.nA)

    # Crank-Nicolson + Adams-Bashforth
    V = imex_cnab_step(f_explicit, f_implicit, V, t, I_ext=0.5 * u.nA)

**DDE Integrators:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from collections import deque
    from braintools.quad import (
        dde_euler_step, dde_heun_step, dde_rk4_step,
        dde_euler_pc_step, dde_heun_pc_step
    )

    bst.environ.set(dt=0.1 * u.ms)

    # Delayed feedback system: dy/dt = -y(t) + tanh(y(t-τ))
    delay = 5.0 * u.ms

    # Simple history storage
    history = deque(maxlen=int(delay / bst.environ.get_dt()) + 1)
    times = deque(maxlen=int(delay / bst.environ.get_dt()) + 1)

    # Initialize history
    y = 0.1
    for i in range(len(history)):
        history.append(y)
        times.append(-delay + i * bst.environ.get_dt())

    # History function (linear interpolation)
    def history_fn(t_past):
        # Find closest stored values and interpolate
        # (simplified example - use proper interpolation in practice)
        idx = min(max(0, int((t_past - times[0]) / bst.environ.get_dt())), len(history)-1)
        return history[idx]

    # DDE right-hand side
    def f(t, y, y_delayed):
        return -y + jnp.tanh(y_delayed)

    # Integration loop
    t = 0.0 * u.ms
    for _ in range(100):
        # Euler method for DDEs
        y_new = dde_euler_step(f, y, t, history_fn, delays=delay)

        # Or use higher-order methods
        # y_new = dde_heun_step(f, y, t, history_fn, delays=delay)
        # y_new = dde_rk4_step(f, y, t, history_fn, delays=delay)

        # Or predictor-corrector methods
        # y_new = dde_euler_pc_step(f, y, t, history_fn, delays=delay)

        # Update history
        history.append(y_new)
        times.append(t)
        y = y_new
        t += bst.environ.get_dt()

    # Multiple delays example
    def f_multi(t, y, y_delay1, y_delay2):
        return -y + 0.5 * jnp.tanh(y_delay1) + 0.3 * jnp.sin(y_delay2)

    delays = [5.0 * u.ms, 10.0 * u.ms]
    y_new = dde_euler_step(f_multi, y, t, history_fn, delays=delays)

**PyTree State Integration:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import ode_rk4_step, sde_euler_step

    bst.environ.set(dt=0.01 * u.ms)

    # Complex state as PyTree (dictionary)
    state = {
        'V': -65.0 * u.mV,
        'w': 0.0,
        'Ca': 0.1 * u.uM,
    }

    # ODE for complex state
    def neuron_dynamics(state, t, I_ext=0.0):
        V, w, Ca = state['V'], state['w'], state['Ca']
        tau_V = 20.0 * u.ms
        tau_w = 100.0 * u.ms
        tau_Ca = 50.0 * u.ms

        dV = (-V + 65 * u.mV + 10 * u.MOhm * I_ext - w) / tau_V
        dw = (-w + 0.5 * (V + 65 * u.mV)) / tau_w
        dCa = (-Ca + 0.1 * u.uM) / tau_Ca

        return {'V': dV, 'w': dw, 'Ca': dCa}

    # Integration preserves PyTree structure
    state = ode_rk4_step(neuron_dynamics, state, 0.0 * u.ms, I_ext=1.0 * u.nA)

    # SDE with PyTree state
    def drift(state, t):
        return neuron_dynamics(state, t, I_ext=0.5 * u.nA)

    def diffusion(state, t):
        return {
            'V': 0.1 * u.mV / u.ms,
            'w': 0.0,
            'Ca': 0.01 * u.uM / u.ms,
        }

    state = sde_euler_step(drift, diffusion, state, 0.0 * u.ms)

**Adaptive Time Stepping:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import ode_rk23_step, ode_rk45_step, ode_dopri5_step

    bst.environ.set(dt=0.01 * u.ms)

    # Embedded RK methods return both solutions for error estimation
    def f(y, t):
        return -y + jnp.sin(10 * t)

    y = 1.0
    t = 0.0 * u.ms

    # RK23 (Bogacki-Shampine 2(3))
    y_new = ode_rk23_step(f, y, t)

    # RK45 (Cash-Karp or Dormand-Prince 4(5))
    y_new = ode_rk45_step(f, y, t)
    y_new = ode_dopri5_step(f, y, t)  # Same as ode_rk45_dopri_step

    # DOP853 (Dormand-Prince 8(7)) - high accuracy
    y_new = ode_dopri8_step(f, y, t)

**Strong Stability Preserving Methods:**

.. code-block:: python

    import brainstate as bst
    import brainunit as u
    import jax.numpy as jnp
    from braintools.quad import ode_ssprk33_step

    bst.environ.set(dt=0.001 * u.ms)

    # SSPRK(3,3) - third-order SSP Runge-Kutta
    # Useful for problems with discontinuities or shocks
    def f(y, t):
        # Some hyperbolic PDE discretization
        return -jnp.roll(y, 1) + y

    y = jnp.ones(100)
    t = 0.0 * u.ms

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
