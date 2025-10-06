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
import jax
import jax.numpy as jnp
import numpy as np

import braintools


def _exact_linear(y0, a, dt):
    return y0 * math.exp(a * dt)


def _one_step_err(method_fn, dt, order=None, a=-1.3, y0=1.0):
    def f(y, t):
        return a * y

    with brainstate.environ.context(dt=dt):
        y1 = method_fn(f, y0, 0.0)
    return abs(y1 - _exact_linear(y0, a, dt))


def test_accuracy_improves_with_smaller_dt():
    methods = {
        'rk2': (braintools.quad.ode_rk2_step, 2),
        'midpoint': (braintools.quad.ode_midpoint_step, 2),
        'rk3': (braintools.quad.ode_rk3_step, 3),
        'heun_rk3': (braintools.quad.ode_heun_step, 3),
        'rk4': (braintools.quad.ode_rk4_step, 4),
        'rk4_38': (braintools.quad.ode_rk4_38_step, 4),
        'ssprk33': (braintools.quad.ode_ssprk33_step, 3),
    }

    h = 0.1
    for name, (fn, order) in methods.items():
        e_h = _one_step_err(fn, h)
        e_h2 = _one_step_err(fn, h / 2)
        # Expect error reduction when halving step; for p>=2 it should drop a lot
        assert e_h2 < e_h, f"{name}: error did not decrease when halving dt"


def test_embedded_pairs_return_error_and_reasonable_magnitude():
    a = -0.7

    def f(y, t):
        return a * y

    y0 = 1.0
    h = 0.05

    embedded_methods = [
        braintools.quad.ode_rk45_step,  # Cash–Karp
        braintools.quad.ode_rk45_dopri_step,  # Dormand–Prince 5(4)
        braintools.quad.ode_rkf45_step,  # Fehlberg
        braintools.quad.ode_rk23_step,  # Bogacki–Shampine 2(3)
    ]

    for fn in embedded_methods:
        with brainstate.environ.context(dt=h):
            y1, err = fn(f, y0, 0.0, return_error=True)
        actual = abs(y1 - _exact_linear(y0, a, h))
        est = abs(err)
        # Error estimate and actual error should be within an order of magnitude
        if est > 0:
            ratio = actual / est
            assert 0.0 <= ratio < 20.0


def test_dopri8_is_very_accurate_for_moderate_dt():
    a = -1.0

    def f(y, t):
        return a * y

    y0 = 1.0
    h = 0.1

    with brainstate.environ.context(dt=h):
        y1 = braintools.quad.ode_rk87_dopri_step(f, y0, 0.0)

    exact = _exact_linear(y0, a, h)
    # Very small error expected for 8th order
    assert abs(y1 - exact) < 1e-8


def test_alias_bs32_matches_rk23():
    a = -0.4

    def f(y, t):
        return a * y

    y0 = 1.0
    h = 0.02

    with brainstate.environ.context(dt=h):
        y_bs, e_bs = braintools.quad.ode_bs32_step(f, y0, 0.0, return_error=True)
        y_23, e_23 = braintools.quad.ode_rk23_step(f, y0, 0.0, return_error=True)

    assert np.allclose(y_bs, y_23)
    assert np.allclose(e_bs, e_23)


def test_tree_structure_preserved():
    # y' = a*y elementwise on a PyTree (dict of arrays)
    a = -0.3

    def f(tree, t):
        return jax.tree.map(lambda x: a * x, tree)

    tree0 = {
        'x': jnp.array([1.0, 2.0, 3.0]),
        'y': {
            'z': jnp.array([[1.0, 0.0], [0.0, 1.0]])
        }
    }
    h = 0.05

    with brainstate.environ.context(dt=h):
        out = braintools.quad.ode_rk4_step(f, tree0, 0.0)

    assert set(out.keys()) == {'x', 'y'}
    assert out['x'].shape == (3,)
    assert out['y']['z'].shape == (2, 2)


def test_ode_integrators_scalar_linear():
    # y' = a*y, a=1.0, y0 = 1.0
    a = 1.0

    def f(y, t):
        return a * y

    y0 = 1.0
    dt = 0.1

    with brainstate.environ.context(dt=dt):
        y_euler = braintools.quad.ode_euler_step(f, y0, 0.0)
        y_rk2 = braintools.quad.ode_rk2_step(f, y0, 0.0)
        y_rk3 = braintools.quad.ode_rk3_step(f, y0, 0.0)
        y_rk4 = braintools.quad.ode_rk4_step(f, y0, 0.0)

    # True solution: y = e^{a dt}
    y_true = np.exp(a * dt)
    assert np.allclose(y_euler, 1 + a * dt)
    assert abs(y_rk2 - y_true) < abs(y_euler - y_true)
    assert abs(y_rk3 - y_true) < abs(y_rk2 - y_true)
    assert abs(y_rk4 - y_true) < abs(y_rk3 - y_true)


def test_ode_integrators_vector_tree():
    # y' = A y for vector y
    A = jnp.array([[0.0, 1.0], [-1.0, 0.0]])  # rotation system

    def f(y, t):
        return A @ y

    y0 = jnp.array([1.0, 0.0])
    dt = 0.01

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.ode_rk4_step(f, y0, 0.0)

    assert y1.shape == (2,)
