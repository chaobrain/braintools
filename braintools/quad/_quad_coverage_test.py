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

"""Comprehensive edge-case / branch / unit-invariance tests for ``braintools.quad``.

A recurring, strong invariant used here is *scale invariance*: integrating the
linear problem ``dy/dt = -y / tau`` with units (mV, ms) must give a mantissa
identical to the dimensionless version (with the same RNG seed for SDEs), because
units only rescale the algebra. This exercises every method's unit handling.
"""

import math

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintools


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


# ===========================================================================
# SDE: branch coverage (Stratonovich, invalid types) + scale invariance
# ===========================================================================
def test_milstein_stratonovich_differs_from_ito():
    def df(y, t):
        return 0.1 * y

    def dg(y, t):
        return 0.3 * y

    with brainstate.environ.context(dt=0.05):
        brainstate.random.seed(1)
        mi = braintools.quad.sde_milstein_step(df, dg, 1.0, 0.0, sde_type='ito')
        brainstate.random.seed(1)
        ms = braintools.quad.sde_milstein_step(df, dg, 1.0, 0.0, sde_type='stra')
    # Same noise, different Ito/Stratonovich correction -> different result
    assert not np.allclose(mi, ms)


def test_heun_stratonovich_runs():
    def df(y, t):
        return -0.5 * y

    def dg(y, t):
        return 0.2 * y

    with brainstate.environ.context(dt=0.05):
        y1 = braintools.quad.sde_heun_step(df, dg, 1.0, 0.0, sde_type='stra')
    assert np.isfinite(float(y1))


@pytest.mark.parametrize('name', ['sde_euler_step', 'sde_milstein_step', 'sde_heun_step'])
def test_sde_invalid_type_raises(name):
    method = getattr(braintools.quad, name)

    def df(y, t):
        return -y

    def dg(y, t):
        return 0.1

    with brainstate.environ.context(dt=0.1):
        with pytest.raises(AssertionError):
            method(df, dg, 1.0, 0.0, sde_type='nonsense')


SDE_METHODS = [
    'sde_euler_step', 'sde_milstein_step', 'sde_heun_step',
    'sde_tamed_euler_step', 'sde_implicit_euler_step',
    'sde_srk2_step', 'sde_srk3_step', 'sde_srk4_step', 'sde_expeuler_step',
]


@pytest.mark.parametrize('name', SDE_METHODS)
def test_sde_unit_scale_invariance(name):
    method = getattr(braintools.quad, name)
    sigma = 0.5

    def df_dimless(y, t):
        return -y / 10.0

    def dg_dimless(y, t):
        return sigma

    def df_unit(v, t):
        return -v / (10.0 * u.ms)

    def dg_unit(v, t):
        return sigma * u.mV / u.ms ** 0.5

    brainstate.random.seed(42)
    with brainstate.environ.context(dt=0.1):
        yd = method(df_dimless, dg_dimless, -65.0, 0.0)
    brainstate.random.seed(42)
    with brainstate.environ.context(dt=0.1 * u.ms):
        yu = method(df_unit, dg_unit, -65.0 * u.mV, 0.0 * u.ms)

    assert u.get_unit(yu) == u.mV
    assert np.allclose(u.get_mantissa(yu), float(yd), rtol=1e-5)


def test_sde_expeuler_diffusion_unit_mismatch_raises():
    # Drift has unit mV/ms (state mV); diffusion with a wrong unit must be caught.
    def df(v, t):
        return -v / (10.0 * u.ms)

    def dg(v, t):
        return 0.5 * u.mV  # wrong: should be mV / sqrt(ms)

    with brainstate.environ.context(dt=0.1 * u.ms):
        with pytest.raises(ValueError):
            braintools.quad.sde_expeuler_step(df, dg, -65.0 * u.mV, 0.0 * u.ms)


def test_implicit_euler_converges_for_contractive_drift():
    # Picard iteration converges when the drift is contractive (|a*dt| < 1).
    a = -2.0

    def df(y, t):
        return a * y

    def dg(y, t):
        return 0.0

    dt = 0.1
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_implicit_euler_step(df, dg, 1.0, 0.0, max_iter=60)
    assert np.allclose(float(y1), 1.0 / (1.0 - a * dt), rtol=1e-4)


# ===========================================================================
# IMEX: scale invariance + structure
# ===========================================================================
def test_imex_euler_unit_scale_invariance():
    def fe_d(y, t):
        return -y / 50.0

    def fi_d(y, t):
        return -y / 10.0

    def fe_u(v, t):
        return -v / (50.0 * u.ms)

    def fi_u(v, t):
        return -v / (10.0 * u.ms)

    with brainstate.environ.context(dt=0.1):
        yd = braintools.quad.imex_euler_step(fe_d, fi_d, -65.0, 0.0, max_iter=10)
    with brainstate.environ.context(dt=0.1 * u.ms):
        yu = braintools.quad.imex_euler_step(fe_u, fi_u, -65.0 * u.mV, 0.0 * u.ms, max_iter=10)
    assert u.get_unit(yu) == u.mV
    assert np.allclose(u.get_mantissa(yu), float(yd), rtol=1e-6)


def test_imex_ars222_and_cnab_with_units_finite():
    def fe(v, t):
        return (-65.0 * u.mV) / (20.0 * u.ms)

    def fi(v, t):
        return -v / (20.0 * u.ms)

    v = -60.0 * u.mV
    with brainstate.environ.context(dt=0.1 * u.ms):
        v_ars = braintools.quad.imex_ars222_step(fe, fi, v, 0.0 * u.ms, max_iter=5)
        v_cnab = braintools.quad.imex_cnab_step(fe, fi, v, v, 0.0 * u.ms, max_iter=5)
    assert u.get_unit(v_ars) == u.mV and np.isfinite(u.get_mantissa(v_ars))
    assert u.get_unit(v_cnab) == u.mV and np.isfinite(u.get_mantissa(v_cnab))


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
