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


def test_imex_euler_linear_split_formula():
    # y' = a*y + b*y, split a -> explicit, b -> implicit
    a = -0.3
    b = -2.0

    def f_exp(y, t):
        return a * y

    def f_imp(y, t):
        return b * y

    y0 = 1.2
    dt = 0.05

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.imex_euler_step(f_exp, f_imp, y0, 0.0, max_iter=10)

    # IMEX Euler exact update for linear split
    y1_formula = (y0 + dt * a * y0) / (1.0 - dt * b)
    assert np.allclose(y1, y1_formula, rtol=1e-6)


def test_imex_cnab_linear_split_formula():
    # CNAB: AB2 for explicit, trapezoidal for implicit (second order)
    a = 0.8
    b = -3.0

    def f_exp(y, t):
        return a * y

    def f_imp(y, t):
        return b * y

    y0 = 0.7
    dt = 0.02
    # Provide y_{-1} consistent with exact y(t) = y0 * exp((a+b)t)
    y_prev = y0 * math.exp(-(a + b) * dt)

    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.imex_cnab_step(f_exp, f_imp, y0, y_prev, 0.0, max_iter=10)

    # Closed form for one CNAB step on linear split
    num = y0 * (1.0 + 1.5 * a * dt + 0.5 * b * dt) - 0.5 * a * dt * y_prev
    den = (1.0 - 0.5 * b * dt)
    y1_formula = num / den
    assert np.allclose(y1, y1_formula, rtol=1e-6)


def test_imex_ars222_accuracy_order():
    # ARS(2,2,2) is 2nd order; halving dt should significantly reduce error
    a = -1.1
    b = -5.0

    def f_exp(y, t):
        return a * y

    def f_imp(y, t):
        return b * y

    y0 = 1.0

    def exact(y, dt):
        return y * math.exp((a + b) * dt)

    def one_step_err(h):
        with brainstate.environ.context(dt=h):
            y1 = braintools.quad.imex_ars222_step(f_exp, f_imp, y0, 0.0, max_iter=10)
        return abs(y1 - exact(y0, h))

    e_h = one_step_err(0.1)
    e_h2 = one_step_err(0.05)
    assert e_h2 < 0.6 * e_h


def test_imex_reductions_when_parts_zero():
    # b=0 -> explicit schemes; a=0 -> implicit schemes
    a = 0.9
    b = -4.0

    def f_exp(y, t):
        return a * y

    def f_imp(y, t):
        return b * y

    y0 = 0.5
    dt = 0.03
    with brainstate.environ.context(dt=dt):
        # b=0: IMEX Euler equals explicit Euler
        y_e_imex = braintools.quad.imex_euler_step(f_exp, lambda y, t: 0.0, y0, 0.0, max_iter=10)
        assert np.allclose(y_e_imex, y0 + dt * a * y0, rtol=1e-6)

        # b=0: ARS(2,2,2) reduces to its explicit tableau (c2 = gamma, weights [delta, 1-delta])
        import math as _m
        gamma = 1.0 - 1.0 / _m.sqrt(2.0)
        delta = 1.0 - 1.0 / (2.0 * gamma)
        k1 = a * y0
        k2 = a * (y0 + dt * gamma * k1)
        y_rk2_gamma = y0 + dt * (delta * k1 + (1.0 - delta) * k2)
        y_ars = braintools.quad.imex_ars222_step(f_exp, lambda y, t: 0.0, y0, 0.0, max_iter=10)
        assert np.allclose(y_ars, y_rk2_gamma, rtol=1e-6)

        # a=0: IMEX Euler equals implicit Euler
        y_i_imex = braintools.quad.imex_euler_step(lambda y, t: 0.0, f_imp, y0, 0.0, max_iter=10)
        y_i_formula = y0 / (1.0 - dt * b)
        assert np.allclose(y_i_imex, y_i_formula, rtol=1e-6)

        # a=0: CNAB equals trapezoidal rule when y_prev=y0
        # If a=0, AB2 part is zero; CN corrector reduces to trapezoidal.
        y_cnab = braintools.quad.imex_cnab_step(lambda y, t: 0.0, f_imp, y0, y0, 0.0, max_iter=10)
        # Solve y1 = y0 + dt/2 (b*y1 + b*y0)
        y_tr = (y0 + 0.5 * dt * b * y0) / (1.0 - 0.5 * dt * b)
        assert np.allclose(y_cnab, y_tr, rtol=1e-6)


def test_imex_pytree_structure_preserved():
    a = -0.2
    b = -3.0

    def f_exp(tree, t):
        return jax.tree.map(lambda x: a * x, tree)

    def f_imp(tree, t):
        return jax.tree.map(lambda x: b * x, tree)

    y0 = {
        'x': jnp.array([1.0, 2.0, 3.0]),
        'y': {'z': jnp.ones((2, 2))}
    }
    dt = 0.04
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.imex_ars222_step(f_exp, f_imp, y0, 0.0, max_iter=10)

    assert set(y1.keys()) == {'x', 'y'}
    assert y1['x'].shape == (3,)
    assert y1['y']['z'].shape == (2, 2)


def _solve_forced_decay(step, k, dt, T=1.0):
    # y' = sin(t) - k y, y(0) = 1; CNAB is started with one ARS(2,2,2) step
    fe = lambda y, t: jnp.sin(t)
    fi = lambda y, t: -k * y
    n = int(round(T / dt))
    y = jnp.asarray(1.0)
    y_prev = y
    with brainstate.environ.context(dt=dt):
        for i in range(n):
            if step is braintools.quad.imex_cnab_step:
                if i == 0:
                    y_new = braintools.quad.imex_ars222_step(fe, fi, y, 0.0)
                else:
                    y_new = step(fe, fi, y, y_prev, i * dt)
                y_prev, y = y, y_new
            else:
                y = step(fe, fi, y, i * dt)
    return float(y)


def _forced_decay_exact(k, T=1.0):
    A = k / (k ** 2 + 1)
    B = -1 / (k ** 2 + 1)
    return (1 - B) * np.exp(-k * T) + A * np.sin(T) + B * np.cos(T)


@pytest.mark.parametrize('name, order', [('imex_euler_step', 1), ('imex_ars222_step', 2), ('imex_cnab_step', 2)])
def test_imex_global_convergence_order(name, order):
    step = getattr(braintools.quad, name)
    errs = [abs(_solve_forced_decay(step, 2.0, dt) - _forced_decay_exact(2.0)) for dt in [0.05, 0.025]]
    assert np.log2(errs[0] / errs[1]) > order - 0.3


@pytest.mark.parametrize('name', ['imex_euler_step', 'imex_ars222_step', 'imex_cnab_step'])
def test_imex_stable_for_stiff_implicit_part(name):
    # k * dt = 10: the former fixed-point iteration diverged here
    step = getattr(braintools.quad, name)
    y = _solve_forced_decay(step, 1000.0, 0.01)
    assert np.isfinite(y)
    assert abs(y - _forced_decay_exact(1000.0)) < 1e-4


def test_imex_newton_nonlinear_implicit_part():
    # implicit Euler on y' = -y^3 must satisfy its own stage equation
    dt = 0.5
    y0 = jnp.asarray(2.0)
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.imex_euler_step(lambda y, t: 0.0 * y, lambda y, t: -y ** 3, y0, 0.0, max_iter=8)
    assert abs(float(y1 - y0 + dt * y1 ** 3)) < 1e-5


def test_imex_max_iter_must_be_positive():
    with brainstate.environ.context(dt=0.1):
        with pytest.raises(ValueError):
            braintools.quad.imex_euler_step(lambda y, t: y, lambda y, t: -y, jnp.asarray(1.0), 0.0, max_iter=0)


def test_imex_newton_krylov_large_state():
    # 1000-point periodic diffusion with k * dt ~ 4e3: above the dense-size
    # threshold, so the step uses matrix-free GMRES; one implicit Euler step
    # on a Fourier mode has a closed form
    n = 1000
    lap = lambda y, t: 1e4 * (jnp.roll(y, 1) - 2 * y + jnp.roll(y, -1))
    x = jnp.linspace(0.0, 1.0, n, endpoint=False)
    y0 = jnp.sin(2 * jnp.pi * x)
    lam = 1e4 * (2 * np.cos(2 * np.pi / n) - 2)
    dt = 0.1
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.imex_euler_step(lambda y, t: jnp.zeros_like(y), lap, y0, 0.0)
    np.testing.assert_allclose(y1, y0 / (1 - dt * lam), atol=1e-5)


# ==============================================================================
# Edge-case / branch / unit-invariance tests. A recurring invariant is *scale
# invariance*: integrating ``dy/dt = -y / tau`` with units (mV, ms) must give a
# mantissa identical to the dimensionless version (same RNG seed for SDEs).
# ==============================================================================


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
