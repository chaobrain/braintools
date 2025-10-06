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
import numpy as np

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

        # b=0: ARS reduces to explicit 2-stage RK with weights [1-gamma, gamma]
        import math as _m
        gamma = 1.0 - 1.0 / _m.sqrt(2.0)
        k1 = a * y0
        k2 = a * (y0 + dt * k1)
        y_rk2_gamma = y0 + dt * ((1.0 - gamma) * k1 + gamma * k2)
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
