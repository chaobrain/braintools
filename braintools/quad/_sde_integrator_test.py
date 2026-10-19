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
    # Implicit Euler for linear drift f(y)=a*y, without noise.
    #
    # ``sde_implicit_euler_step`` solves the implicit stage y1 = y0 + dt*a*y1 by
    # fixed-point (Picard) iteration y1^{k+1} = y0 + dt*a*y1^{k}. This iteration is
    # a contraction iff |dt*a| < 1, so it converges only for non-stiff coefficients.
    # We therefore use a = -2 (|dt*a| = 0.2 < 1); a stiff a (e.g. -50) would make
    # the iteration DIVERGE away from the exact solution, no matter how large
    # max_iter is -- see the docstring note on ``sde_implicit_euler_step``.
    a = -2.0

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
    assert np.allclose(y1, y_exact, rtol=1e-3, atol=1e-6)


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


@pytest.mark.parametrize('n', [8, 4096])
def test_sde_implicit_euler_solves_stiff_drift(n):
    # dy = -k y dt + dW with zero noise amplitude: the drift-implicit step must
    # give y / (1 + k dt) even for k dt >> 1, for both the dense (n=8) and the
    # matrix-free (n=4096) Newton paths
    k, dt = 1000.0, 0.01
    y0 = jnp.linspace(1.0, 2.0, n)
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_implicit_euler_step(
            lambda y, t: -k * y, lambda y, t: jnp.zeros_like(y), y0, 0.0)
    np.testing.assert_allclose(y1, y0 / (1 + k * dt), rtol=1e-5)


def test_sde_implicit_euler_superlinear_drift_stays_finite():
    brainstate.random.seed(0)
    dt = 0.4
    with brainstate.environ.context(dt=dt):
        def body(y, i):
            return braintools.quad.sde_implicit_euler_step(
                lambda y, t: y - y ** 3, lambda y, t: 2.0 * jnp.ones_like(y), y, i * dt), None

        y, _ = brainstate.transform.scan(body, jnp.zeros(500), jnp.arange(50))
    assert np.all(np.isfinite(np.asarray(y)))
    assert np.max(np.abs(np.asarray(y))) < 10


# ==============================================================================
# Edge-case / branch / unit-invariance tests. A recurring invariant is *scale
# invariance*: integrating ``dy/dt = -y / tau`` with units (mV, ms) must give a
# mantissa identical to the dimensionless version (same RNG seed for SDEs).
# ==============================================================================


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


def _milstein_strong_errors(sde_type, monkeypatch):
    """Strong errors on geometric Brownian motion with a shared Brownian path."""
    from braintools.quad import _sde_integrator as S
    mu, sig, T, M, n_fine = 1.5, 1.0, 1.0, 2000, 2 ** 8
    rng = np.random.default_rng(0)
    dW_fine = rng.normal(size=(n_fine, M)) * np.sqrt(T / n_fine)
    W = dW_fine.sum(0)
    drift_exp = mu - 0.5 * sig ** 2 if sde_type == 'ito' else mu
    exact = np.exp(drift_exp * T + sig * W)
    errors = []
    for k in [16, 8, 4]:
        dt = T * k / n_fine
        normals = iter(dW_fine.reshape(-1, k, M).sum(1) / np.sqrt(dt))
        monkeypatch.setattr(S, 'randn_like', lambda y: jnp.asarray(next(normals)))
        y = jnp.ones(M)
        with brainstate.environ.context(dt=dt):
            for j in range(n_fine // k):
                y = braintools.quad.sde_milstein_step(lambda y, t: mu * y, lambda y, t: sig * y, y, j * dt,
                                                      sde_type=sde_type)
        errors.append(np.mean(np.abs(np.asarray(y) - exact)))
    return errors


@pytest.mark.parametrize('sde_type', ['ito', 'stra'])
def test_milstein_strong_order_one(sde_type, monkeypatch):
    # Halving dt must roughly halve the strong error (order 1). The Stratonovich
    # branch used to keep the drift in its support point and converged at 1/2.
    errors = _milstein_strong_errors(sde_type, monkeypatch)
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(orders) > 0.8, orders


# ==============================================================================
# Regression tests for bugs found in the 2026-06-18 ``braintools.quad`` audit.
# Each reproduces a confirmed bug and asserts the *correct* behaviour.
# ==============================================================================


# ---------------------------------------------------------------------------
# Bug 2 + 3: sde_expeuler_step used randn_like(args[0]) (crashes with no extra
# args) and the wrong unit. With zero diffusion it must equal exp-Euler drift.
# ---------------------------------------------------------------------------
def test_sde_expeuler_step_no_extra_args():
    def df(x, t):
        return -x

    def dg(x, t):
        return 0.0

    with brainstate.environ.context(dt=0.1):
        x1 = braintools.quad.sde_expeuler_step(df, dg, 1.0, 0.0)
    # Zero diffusion -> deterministic exp-Euler drift = exp(-0.1)
    assert np.allclose(x1, math.exp(-0.1), rtol=1e-6)


def test_sde_expeuler_step_with_units():
    tau = 10.0 * u.ms

    def df(v, t):
        return -v / tau

    def dg(v, t):
        return 0.0 * u.mV / u.ms ** 0.5

    v0 = -65.0 * u.mV
    with brainstate.environ.context(dt=0.1 * u.ms):
        v1 = braintools.quad.sde_expeuler_step(df, dg, v0, 0.0 * u.ms)
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), -65.0 * math.exp(-0.01), rtol=1e-6)


# ---------------------------------------------------------------------------
# Bug 4: sde_euler_step used jnp.sqrt(dt), which rejects unitful dt.
# With zero diffusion the step is deterministic Euler.
# ---------------------------------------------------------------------------
def test_sde_euler_step_with_units():
    tau = 10.0 * u.ms

    def df(v, t):
        return -v / tau

    def dg(v, t):
        return 0.0 * u.mV / u.ms ** 0.5

    v0 = -65.0 * u.mV
    with brainstate.environ.context(dt=0.1 * u.ms):
        v1 = braintools.quad.sde_euler_step(df, dg, v0, 0.0 * u.ms)
    expected = -65.0 + (65.0 / 10.0) * 0.1  # v0 + (-v0/tau)*dt
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Bug 5: sde_tamed_euler_step added a dimensionless 1 to a unitful dt*|f|.
# ---------------------------------------------------------------------------
def test_sde_tamed_euler_step_with_units():
    tau = 10.0 * u.ms

    def df(v, t):
        return -v / tau

    def dg(v, t):
        return 0.0 * u.mV / u.ms ** 0.5

    v0 = -65.0 * u.mV
    dt = 0.1
    with brainstate.environ.context(dt=dt * u.ms):
        v1 = braintools.quad.sde_tamed_euler_step(df, dg, v0, 0.0 * u.ms)
    f_val = 65.0 / 10.0  # mantissa of -v0/tau in mV/ms
    tamed = f_val / (1.0 + dt * abs(f_val))
    expected = -65.0 + tamed * dt
    assert u.get_unit(v1) == u.mV
    assert np.allclose(u.get_mantissa(v1), expected, rtol=1e-6)


def test_sde_tamed_euler_step_dimensionless_unchanged():
    # Guard: dimensionless taming result must be unchanged by the fix.
    def df(y, t):
        return y ** 3

    def dg(y, t):
        return 0.0

    y0 = 10.0
    dt = 0.1
    with brainstate.environ.context(dt=dt):
        y1 = braintools.quad.sde_tamed_euler_step(df, dg, y0, 0.0)
    f_val = y0 ** 3
    expected = y0 + (f_val / (1.0 + dt * abs(f_val))) * dt
    assert np.allclose(y1, expected, rtol=1e-6)


def test_sde_expeuler_step_rejects_integer_scalar():
    def df(x, t):
        return -x

    def dg(x, t):
        return 0.0

    with brainstate.environ.context(dt=0.1):
        try:
            braintools.quad.sde_expeuler_step(df, dg, 1, 0.0)  # python int
        except ValueError:
            pass
        except Exception as e:  # pragma: no cover - explicit failure path
            raise AssertionError(f"expected ValueError, got {type(e).__name__}: {e}")
        else:
            raise AssertionError("expected ValueError for integer input")
