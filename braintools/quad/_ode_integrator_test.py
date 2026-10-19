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
import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


# ==============================================================================
# Edge-case / branch / unit-invariance tests. A recurring invariant is *scale
# invariance*: integrating ``dy/dt = -y / tau`` with units (mV, ms) must give a
# mantissa identical to the dimensionless version (same RNG seed for SDEs).
# ==============================================================================


# ===========================================================================
# ODE: Ralston methods (previously untested) + order checks
# ===========================================================================
def _linear_one_step_err(method, dt, a=-1.3, y0=1.0):
    def f(y, t):
        return a * y

    with brainstate.environ.context(dt=dt):
        y1 = method(f, y0, 0.0)
    return abs(float(y1) - y0 * math.exp(a * dt))


def test_ralston2_is_second_order():
    e1 = _linear_one_step_err(braintools.quad.ode_ralston2_step, 0.1)
    e2 = _linear_one_step_err(braintools.quad.ode_ralston2_step, 0.05)
    # 2nd order local error ~ dt^3; halving dt reduces error by ~8x (allow margin)
    assert e2 < e1 / 4.0


def test_ralston3_is_third_order():
    e1 = _linear_one_step_err(braintools.quad.ode_ralston3_step, 0.1)
    e2 = _linear_one_step_err(braintools.quad.ode_ralston3_step, 0.05)
    assert e2 < e1 / 8.0


def test_ralston3_equals_rk23_third_order_solution():
    # The Ralston RK3 tableau equals the 3rd-order solution of Bogacki-Shampine.
    def f(y, t):
        return -1.3 * y

    with brainstate.environ.context(dt=0.1):
        r3 = braintools.quad.ode_ralston3_step(f, 1.0, 0.0)
        r23 = braintools.quad.ode_rk23_step(f, 1.0, 0.0)
    assert np.allclose(r3, r23)


def test_dopri8_return_error_is_small_and_finite():
    def f(y, t):
        return -1.0 * y

    with brainstate.environ.context(dt=0.1):
        y1, err = braintools.quad.ode_dopri8_step(f, 1.0, 0.0, return_error=True)
    assert np.isfinite(float(y1)) and np.isfinite(float(err))
    assert abs(float(err)) < 1e-6  # 8th order: tiny local error estimate


# ===========================================================================
# ODE: scale invariance across all deterministic methods
# ===========================================================================
ODE_METHODS = [
    'ode_euler_step', 'ode_rk2_step', 'ode_rk3_step', 'ode_rk4_step',
    'ode_midpoint_step', 'ode_heun_step', 'ode_rk4_38_step',
    'ode_ralston2_step', 'ode_ralston3_step', 'ode_ssprk33_step',
    'ode_rk45_step', 'ode_rk23_step', 'ode_dopri5_step', 'ode_rkf45_step',
    'ode_dopri8_step', 'ode_expeuler_step',
]


@pytest.mark.parametrize('name', ODE_METHODS)
def test_ode_unit_scale_invariance(name):
    method = getattr(braintools.quad, name)

    def f_dimless(y, t):
        return -y / 10.0

    def f_unit(v, t):
        return -v / (10.0 * u.ms)

    with brainstate.environ.context(dt=0.1):
        yd = method(f_dimless, -65.0, 0.0)
    with brainstate.environ.context(dt=0.1 * u.ms):
        yu = method(f_unit, -65.0 * u.mV, 0.0 * u.ms)

    assert u.get_unit(yu) == u.mV
    assert np.allclose(u.get_mantissa(yu), float(yd), rtol=1e-6)


# ==============================================================================
# Regression tests for bugs found in the 2026-06-18 ``braintools.quad`` audit.
# Each reproduces a confirmed bug and asserts the *correct* behaviour.
# ==============================================================================


# ---------------------------------------------------------------------------
# Bug 1: ode_expeuler_step divided by the Jacobian unit instead of the state
# unit, raising "exprel requires a dimensionless x" for unitful states.
# Exponential Euler is exact for linear ODEs: y1 = y0 * exp(A*dt).
# ---------------------------------------------------------------------------
def test_ode_expeuler_step_with_units():
    tau = 10.0 * u.ms

    def f(v, t):
        return -v / tau

    v0 = -65.0 * u.mV
    with brainstate.environ.context(dt=0.1 * u.ms):
        v1 = braintools.quad.ode_expeuler_step(f, v0, 0.0 * u.ms)

    expected = -65.0 * math.exp(-0.01)  # A*dt = -(1/10ms)*0.1ms = -0.01
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), expected, rtol=1e-6)


def test_ode_expeuler_step_dimensionless_unchanged():
    # Guard: the fix must not change the dimensionless result.
    def f(x, t):
        return -x

    with brainstate.environ.context(dt=0.1):
        x1 = braintools.quad.ode_expeuler_step(f, 1.0, 0.0)
    assert np.allclose(x1, math.exp(-0.1), rtol=1e-6)


# ---------------------------------------------------------------------------
# Bug 14: the primary (propagated) solution of an embedded method must be
# identical whether or not the error estimate is requested. This characterises
# the FSAL refactor of ode_dopri5_step (the extra stage must not change y).
# ---------------------------------------------------------------------------
def test_embedded_primary_matches_return_error_branch():
    a = -0.9

    def f(y, t):
        return a * y

    methods = [
        braintools.quad.ode_rk45_step,
        braintools.quad.ode_dopri5_step,
        braintools.quad.ode_rkf45_step,
        braintools.quad.ode_rk23_step,
        braintools.quad.ode_dopri8_step,
    ]
    with brainstate.environ.context(dt=0.1):
        for fn in methods:
            y_plain = fn(f, 1.0, 0.0)
            y_err, _ = fn(f, 1.0, 0.0, return_error=True)
            assert np.allclose(y_plain, y_err), fn.__name__


# ---------------------------------------------------------------------------
# Bug 15: a disallowed (non-float) input must raise a clean ValueError, not an
# AttributeError while formatting the message (Python scalars have no .dtype).
# ---------------------------------------------------------------------------
def test_ode_expeuler_step_rejects_integer_scalar():
    def f(x, t):
        return -x

    with brainstate.environ.context(dt=0.1):
        try:
            braintools.quad.ode_expeuler_step(f, 1, 0.0)  # python int
        except ValueError:
            pass
        except Exception as e:  # pragma: no cover - explicit failure path
            raise AssertionError(f"expected ValueError, got {type(e).__name__}: {e}")
        else:
            raise AssertionError("expected ValueError for integer input")
