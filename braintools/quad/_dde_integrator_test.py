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
import jax.numpy as jnp
import jax.tree

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
