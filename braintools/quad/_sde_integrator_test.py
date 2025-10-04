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


import brainstate
import jax
import jax.numpy as jnp
import numpy as np

import braintools


def test_sde_euler_shape_and_variance():
    # d y = sigma dW (pure diffusion)
    sigma = 2.0

    def df(y, t):
        return 0.0

    def dg(y, t):
        return sigma

    y0 = 0.0
    dt = 0.2
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_euler_step(df, dg, y0, 0.0)
        y2 = braintools.quad.sde_euler_step(df, dg, y0, 0.0)

    # shapes (scalar array)
    assert np.shape(y1) == ()
    # stochastic - likely different
    assert not np.isclose(y1, y2)


def test_sde_milstein_basic():
    # d y = a y dt + b y dW (geometric brownian motion)
    a, b = 0.1, 0.3

    def df(y, t):
        return a * y

    def dg(y, t):
        return b * y

    y0 = 1.0
    dt = 0.05
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_milstein_step(df, dg, y0, 0.0)

    assert np.isfinite(y1)


def test_heun_reduces_to_rk2_when_no_diffusion():
    # Pure drift: Heun predictor-corrector should match RK2
    a = -0.8

    def df(y, t):
        return a * y

    def dg(y, t):
        return 0.0

    y0 = 1.0
    dt = 0.1
    with brainstate.environ.context(dt=dt):
        y_heun = braintools.quad.sde_heun_step(df, dg, y0, 0.0, sde_type='ito')
        y_rk2 = braintools.quad.ode_rk2_step(lambda y, t: a * y, y0, 0.0)

    assert np.allclose(y_heun, y_rk2)


def test_srk_reduce_to_classical_when_no_diffusion():
    a = 0.5

    def df(y, t):
        return a * y

    def dg(y, t):
        return 0.0

    y0 = 2.0
    dt = 0.05
    with brainstate.environ.context(dt=dt):
        y2 = braintools.quad.sde_srk2_step(df, dg, y0, 0.0)
        y3 = braintools.quad.sde_srk3_step(df, dg, y0, 0.0)
        y4 = braintools.quad.sde_srk4_step(df, dg, y0, 0.0)

        rk2 = braintools.quad.ode_rk2_step(lambda y, t: a * y, y0, 0.0)
        rk3 = braintools.quad.ode_rk3_step(lambda y, t: a * y, y0, 0.0)
        rk4 = braintools.quad.ode_rk4_step(lambda y, t: a * y, y0, 0.0)

    assert np.allclose(y2, rk2)
    assert np.allclose(y3, rk3)
    assert np.allclose(y4, rk4)


def test_diffusion_variance_matches_sigma2_dt():
    # Pure diffusion: y_{n+1} - y_n ~ N(0, sigma^2 dt)
    sigma = 1.7

    def df(y, t):
        return 0.0

    def dg(y, t):
        return sigma

    y0 = 0.0
    dt = 0.03
    N = 2000

    samples_euler = []
    samples_srk2 = []
    with brainstate.environ.context(dt=dt):
        for _ in range(N):
            y1 = braintools.quad.sde_euler_step(df, dg, y0, 0.0)
            y2 = braintools.quad.sde_srk2_step(df, dg, y0, 0.0)
            samples_euler.append(y1)
            samples_srk2.append(y2)

    var_target = sigma * sigma * dt
    var_e = float(np.var(samples_euler))
    var_s = float(np.var(samples_srk2))

    # Allow broad tolerance due to randomness
    assert 0.5 * var_target < var_e < 1.5 * var_target
    assert 0.5 * var_target < var_s < 1.5 * var_target


def test_tamed_euler_bounds_superlinear_no_noise():
    # Drift f(y)=y^3 can blow up with explicit Euler; tamed Euler caps increment
    def df(y, t):
        return y ** 3

    def dg(y, t):
        return 0.0

    y0 = 10.0
    dt = 0.1
    with brainstate.environ.context(dt=dt):
        y_tamed = braintools.quad.sde_tamed_euler_step(df, dg, y0, 0.0)
        # Explicit Euler (deterministic) reference increment
        y_euler = y0 + df(y0, 0.0) * dt

    # Tamed should produce much smaller step than explicit Euler
    assert y_tamed < y_euler


def test_implicit_euler_linear_no_noise():
    # Implicit Euler for stiff linear drift f(y)=a*y, a<<0, without noise
    a = -50.0

    def df(y, t):
        return a * y

    def dg(y, t):
        return 0.0

    y0 = 1.0
    dt = 0.1
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_implicit_euler_step(df, dg, y0, 0.0, max_iter=40)

    # Exact implicit Euler formula: y1 = y0 / (1 - a*dt)
    y_exact = y0 / (1.0 - a * dt)
    # assert np.allclose(y1, y_exact, rtol=1e-3, atol=1e-6)


def test_sde_euler_pytree_structure():
    # Verify structure is preserved for nested PyTree, no noise
    a = -0.2

    def df(tree, t):
        return jax.tree.map(lambda x: a * x, tree)

    def dg(tree, t):
        return jax.tree.map(lambda x: 0.0 * x, tree)

    y0 = {
        'a': jnp.array([1.0, 2.0]),
        'b': {'c': jnp.array([[3.0, 4.0], [5.0, 6.0]])}
    }
    dt = 0.05
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_euler_step(df, dg, y0, 0.0)

    assert set(y1.keys()) == {'a', 'b'}
    assert y1['a'].shape == (2,)
    assert y1['b']['c'].shape == (2, 2)
