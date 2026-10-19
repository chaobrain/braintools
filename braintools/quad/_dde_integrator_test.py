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


import math

import brainstate
import brainunit as u
import jax.numpy as jnp
import jax.tree
import numpy as np
import pytest

import braintools


def test_dde_euler_single_delay():
    """Test DDE Euler method with single delay term."""

    # Simple DDE: y'(t) = -y(t) + y(t-τ)
    def f(t, y, y_delayed):
        return -y + y_delayed

    # Create simple history function (constant initial condition)
    def history_fn(t):
        return 1.0 if t <= 0 else 1.0  # Constant history

    y0 = 1.0
    t0 = 0.0
    delay = 0.5
    dt = 0.1

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.dde_euler_step(f, y0, t0, history_fn, delay)

    # Basic sanity check - result should be a number
    assert isinstance(y1, (float, jnp.ndarray))
    assert not jnp.isnan(y1)


def test_dde_euler_multiple_delays():
    """Test DDE Euler method with multiple delay terms."""

    # DDE with two delays: y'(t) = -y(t) + 0.5*y(t-τ₁) + 0.3*y(t-τ₂)
    def f(t, y, y_delay1, y_delay2):
        return -y + 0.5 * y_delay1 + 0.3 * y_delay2

    def history_fn(t):
        return 1.0

    y0 = 1.0
    t0 = 0.0
    delays = [0.5, 1.0]
    dt = 0.1

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.dde_euler_step(f, y0, t0, history_fn, delays)

    assert isinstance(y1, (float, jnp.ndarray))
    assert not jnp.isnan(y1)


def test_dde_heun_method():
    """Test DDE Heun method."""

    def f(t, y, y_delayed):
        return -0.5 * y + y_delayed

    def history_fn(t):
        return 1.0

    y0 = 1.0
    t0 = 0.0
    delay = 0.5
    dt = 0.05

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.dde_heun_step(f, y0, t0, history_fn, delay)

    assert isinstance(y1, (float, jnp.ndarray))
    assert not jnp.isnan(y1)


def test_dde_rk4_method():
    """Test DDE RK4 method."""

    def f(t, y, y_delayed):
        return -y + 0.8 * y_delayed

    def history_fn(t):
        return 1.0

    y0 = 1.0
    t0 = 0.0
    delay = 0.3
    dt = 0.05

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.dde_rk4_step(f, y0, t0, history_fn, delay)

    assert isinstance(y1, (float, jnp.ndarray))
    assert not jnp.isnan(y1)


def test_dde_predictor_corrector_methods():
    """Test DDE predictor-corrector methods."""

    def f(t, y, y_delayed):
        return -2.0 * y + y_delayed

    def history_fn(t):
        return math.exp(-t) if t <= 0 else 1.0

    y0 = 1.0
    t0 = 0.0
    delay = 0.2
    dt = 0.02

    with brainstate.environ.context(dt=dt):
        # Test Euler P-C
        y1_pc = braintools.quad.dde_euler_pc_step(f, y0, t0, history_fn, delay)
        assert isinstance(y1_pc, (float, jnp.ndarray))
        assert not jnp.isnan(y1_pc)

        # Test Heun P-C
        y1_heun_pc = braintools.quad.dde_heun_pc_step(f, y0, t0, history_fn, delay)
        assert isinstance(y1_heun_pc, (float, jnp.ndarray))
        assert not jnp.isnan(y1_heun_pc)


def test_dde_pytree_structure_preserved():
    """Test that DDE integrators preserve PyTree structure."""

    def f(t, y, y_delayed):
        return jax.tree.map(lambda a, b: -0.5 * a + 0.8 * b, y, y_delayed)

    def history_fn(t):
        return {
            'x': jnp.array([1.0, 0.5]),
            'y': {'z': jnp.ones((2, 2))}
        }

    y0 = {
        'x': jnp.array([1.0, 0.5]),
        'y': {'z': jnp.ones((2, 2))}
    }
    t0 = 0.0
    delay = 0.1
    dt = 0.05

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.dde_euler_step(f, y0, t0, history_fn, delay)

    # Check structure preservation
    assert set(y1.keys()) == {'x', 'y'}
    assert y1['x'].shape == (2,)
    assert y1['y']['z'].shape == (2, 2)
    assert not jnp.any(jnp.isnan(y1['x']))
    assert not jnp.any(jnp.isnan(y1['y']['z']))


def test_dde_linear_stability():
    """Test DDE methods on a simple linear DDE with known behavior."""

    # Test case: y'(t) = -y(t) + 0.5*y(t-1)
    # With constant initial condition y(t) = 1 for t <= 0
    def f(t, y, y_delayed):
        return -y + 0.5 * y_delayed

    def history_fn(t):
        return 1.0

    y0 = 1.0
    t0 = 0.0
    delay = 1.0
    dt = 0.1

    with brainstate.environ.context(dt=dt):
        # Take several steps with different methods
        y_euler = y0
        y_heun = y0
        y_rk4 = y0

        for i in range(5):
            t = t0 + i * dt
            y_euler = braintools.quad.dde_euler_step(f, y_euler, t, history_fn, delay)
            y_heun = braintools.quad.dde_heun_step(f, y_heun, t, history_fn, delay)
            y_rk4 = braintools.quad.dde_rk4_step(f, y_rk4, t, history_fn, delay)

        # All methods should produce reasonable values (not NaN/Inf)
        assert jnp.isfinite(y_euler)
        assert jnp.isfinite(y_heun)
        assert jnp.isfinite(y_rk4)

        # Higher-order methods should be different from Euler
        assert abs(y_heun - y_euler) > 1e-10
        assert abs(y_rk4 - y_euler) > 1e-10


# ==============================================================================
# Edge-case / branch / unit-invariance tests. A recurring invariant is *scale
# invariance*: integrating ``dy/dt = -y / tau`` with units (mV, ms) must give a
# mantissa identical to the dimensionless version (same RNG seed for SDEs).
# ==============================================================================


# ===========================================================================
# DDE: all methods, reduction to ODE, units, multiple delays
# ===========================================================================
DDE_METHODS = [
    'dde_euler_step', 'dde_heun_step', 'dde_rk4_step',
    'dde_euler_pc_step', 'dde_heun_pc_step',
]


@pytest.mark.parametrize('name', DDE_METHODS)
def test_dde_independent_of_history_when_delay_unused(name):
    # If f ignores the delayed term, the result must not depend on the history
    # values: history may only influence the output through f's delayed argument.
    method = getattr(braintools.quad, name)

    def f_dde(t, y, y_delayed):
        return -y  # ignores y_delayed

    def hist_a(t_past):
        return 123.0

    def hist_b(t_past):
        return -999.0

    with brainstate.environ.context(dt=0.05):
        ya = method(f_dde, 1.0, 0.0, hist_a, 0.5)
        yb = method(f_dde, 1.0, 0.0, hist_b, 0.5)
    assert np.isfinite(float(ya))
    assert np.allclose(float(ya), float(yb))


def test_dde_euler_matches_explicit_euler_when_delay_unused():
    # The plain forward-Euler DDE predictor (no corrector) reduces exactly to the
    # explicit ODE Euler step when f ignores the delayed term.
    def f_dde(t, y, y_delayed):
        return -y

    def f_ode(y, t):
        return -y

    def history_fn(t_past):
        return 123.0

    with brainstate.environ.context(dt=0.05):
        y_dde = braintools.quad.dde_euler_step(f_dde, 1.0, 0.0, history_fn, 0.5)
        y_euler = braintools.quad.ode_euler_step(f_ode, 1.0, 0.0)
    assert np.allclose(float(y_dde), float(y_euler))


def test_dde_euler_with_units():
    tau = 10.0 * u.ms

    def f(t, y, y_delayed):
        return (-y + y_delayed) / tau

    def history_fn(t_past):
        return -65.0 * u.mV

    with brainstate.environ.context(dt=0.1 * u.ms):
        v1 = braintools.quad.dde_euler_step(f, -65.0 * u.mV, 0.0 * u.ms, history_fn, 1.0 * u.ms)
    assert u.get_unit(v1) == u.mV
    assert np.isfinite(u.get_mantissa(v1))


def test_dde_multiple_delays_and_pytree():
    def f(t, y, yd1, yd2):
        return jnp.tanh(yd1) + 0.5 * jnp.sin(yd2) - y

    def history_fn(t_past):
        return jnp.array([0.1, 0.2])

    with brainstate.environ.context(dt=0.05):
        y1 = braintools.quad.dde_rk4_step(
            f, jnp.array([0.1, 0.2]), 0.0, history_fn, [0.5, 1.0]
        )
    assert y1.shape == (2,)
    assert np.all(np.isfinite(np.asarray(y1)))


def test_dde_pc_iterations_change_result():
    # More corrector iterations should change (refine) the predictor-corrector result.
    def f(t, y, y_delayed):
        return -3.0 * y + 0.5 * y_delayed

    def history_fn(t_past):
        return 1.0

    with brainstate.environ.context(dt=0.1):
        y1 = braintools.quad.dde_euler_pc_step(f, 1.0, 0.0, history_fn, 0.5, max_iter=1)
        y5 = braintools.quad.dde_euler_pc_step(f, 1.0, 0.0, history_fn, 0.5, max_iter=5)
    assert not np.allclose(y1, y5)


@pytest.mark.parametrize('name', ['dde_euler_pc_step', 'dde_heun_pc_step'])
def test_dde_pc_accepts_list_of_delays(name):
    # The predictor-corrector methods must accept multiple delays passed as a list,
    # not only a single scalar delay.
    method = getattr(braintools.quad, name)

    def f(t, y, yd1, yd2):
        return -y + 0.3 * yd1 + 0.2 * yd2

    def history_fn(t_past):
        return 1.0

    with brainstate.environ.context(dt=0.1):
        y_list = method(f, 1.0, 0.0, history_fn, [0.5, 1.0], max_iter=3)
    assert np.isfinite(float(y_list))
