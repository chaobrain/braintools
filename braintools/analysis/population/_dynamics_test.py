# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from braintools.analysis.population._dynamics import (
    AvalancheResult,
    avalanches,
    branching_ratio,
    granger_causality,
    kuramoto_order_parameter,
)


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


# ----------------------------------------------------------------------------
# kuramoto_order_parameter
# ----------------------------------------------------------------------------

class TestKuramoto:
    def test_reference_numpy(self):
        theta = np.random.default_rng(0).uniform(-np.pi, np.pi, (20, 7))
        z = np.mean(np.exp(1j * theta), axis=-1)
        r, psi = kuramoto_order_parameter(theta, return_phase=True)
        np.testing.assert_allclose(r, np.abs(z), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(psi, np.angle(z), rtol=1e-5, atol=1e-5)
        assert r.shape == (20,)

    def test_locked_and_splay(self):
        locked = jnp.full((5, 4), 1.3)
        np.testing.assert_allclose(kuramoto_order_parameter(locked), 1., atol=1e-6)
        _, psi = kuramoto_order_parameter(locked, return_phase=True)
        np.testing.assert_allclose(psi, 1.3, atol=1e-6)
        splay = jnp.tile(2 * jnp.pi * jnp.arange(6) / 6, (3, 1))
        np.testing.assert_allclose(kuramoto_order_parameter(splay), 0., atol=1e-6)

    def test_axis(self):
        theta = np.random.default_rng(1).uniform(0, 2 * np.pi, (4, 9))
        np.testing.assert_allclose(
            kuramoto_order_parameter(theta, axis=0),
            np.abs(np.mean(np.exp(1j * theta), axis=0)), rtol=1e-5, atol=1e-6,
        )

    def test_units(self):
        theta = np.random.default_rng(2).uniform(0, 2 * np.pi, (3, 5))
        plain = kuramoto_order_parameter(theta)
        np.testing.assert_allclose(kuramoto_order_parameter(theta * u.radian), plain, rtol=1e-6)
        deg = np.degrees(theta) * u.degree
        np.testing.assert_allclose(kuramoto_order_parameter(deg), plain, rtol=1e-5, atol=1e-6)
        with pytest.raises(TypeError):
            kuramoto_order_parameter(theta * u.ms)

    def test_integer_input(self):
        assert kuramoto_order_parameter(jnp.zeros((2, 3), dtype=jnp.int32)).dtype.kind == 'f'

    def test_jit_vmap(self):
        theta = jax.random.uniform(jax.random.key(0), (6, 10, 5), maxval=2 * jnp.pi)
        ref = np.abs(np.mean(np.exp(1j * np.asarray(theta)), axis=-1))
        np.testing.assert_allclose(jax.jit(kuramoto_order_parameter)(theta[0]), ref[0], rtol=1e-5)
        out = jax.vmap(kuramoto_order_parameter)(theta)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_grad(self, x64):
        theta = jnp.asarray(np.random.default_rng(3).uniform(0, 1, (8,)))
        check_grads(lambda t: kuramoto_order_parameter(t), (theta,), order=1, modes=['rev'])

    def test_precision(self, x64):
        theta = np.random.default_rng(4).uniform(0, 2 * np.pi, (5, 6))
        r64 = kuramoto_order_parameter(jnp.asarray(theta, dtype=jnp.float64))
        r32 = kuramoto_order_parameter(jnp.asarray(theta, dtype=jnp.float32))
        assert r64.dtype == jnp.float64 and r32.dtype == jnp.float32
        np.testing.assert_allclose(r32, r64, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------
# granger_causality
# ----------------------------------------------------------------------------

def _ar_bivariate(n=4000, seed=0):
    """x drives y: x_t = .5 x_{t-1} + e, y_t = .4 y_{t-1} + .6 x_{t-1} + e."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 2))
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + e[t, 0]
        y[t] = 0.4 * y[t - 1] + 0.6 * x[t - 1] + e[t, 1]
    return np.stack([x, y], axis=1)


def _var_chain(n=3000, seed=1):
    """VAR(2) chain x -> y -> z (no direct x -> z)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 3))
    s = np.zeros((n, 3))
    for t in range(2, n):
        s[t, 0] = 0.5 * s[t - 1, 0] - 0.2 * s[t - 2, 0] + e[t, 0]
        s[t, 1] = 0.3 * s[t - 1, 1] + 0.7 * s[t - 1, 0] + e[t, 1]
        s[t, 2] = 0.3 * s[t - 1, 2] + 0.5 * s[t - 2, 1] + e[t, 2]
    return s


def _numpy_granger(x, p):
    """Reference: OLS with intercept, restricted models drop the source columns."""
    x = np.asarray(x, dtype=np.float64)
    n_time, n_var = x.shape
    y = x[p:]
    lagged = [x[p - k:n_time - k] for k in range(1, p + 1)]

    def resid_var(cols):
        d = np.concatenate([lag[:, cols] for lag in lagged] + [np.ones((n_time - p, 1))], axis=1)
        coef = np.linalg.lstsq(d, y, rcond=None)[0]
        return np.mean((y - d @ coef) ** 2, axis=0)

    full = resid_var(list(range(n_var)))
    gc = np.zeros((n_var, n_var))
    for j in range(n_var):
        restricted = resid_var([k for k in range(n_var) if k != j])
        for i in range(n_var):
            if i != j:
                gc[i, j] = np.log(restricted[i] / full[i])
    return gc


class TestGranger:
    def test_numpy_reference(self, x64):
        s = _var_chain()
        for p in (1, 2, 3):
            np.testing.assert_allclose(granger_causality(s, p), _numpy_granger(s, p),
                                       rtol=1e-8, atol=1e-10)

    def test_statsmodels_reference(self, x64):
        pytest.importorskip('statsmodels')
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.ar_model import AutoReg
        s = _ar_bivariate()
        gc = granger_causality(s, 2)
        full = VAR(s).fit(2, trend='c').resid  # OLS with intercept, as here
        for target, source in ((1, 0), (0, 1)):
            restricted = AutoReg(s[:, target], 2, trend='c').fit().resid
            ref = np.log(np.mean(restricted ** 2) / np.mean(full[:, target] ** 2))
            np.testing.assert_allclose(gc[target, source], ref, rtol=1e-8, atol=1e-10)

    def test_elephant_golden(self, x64):
        # elephant 1.2.1 (numpy 2.4.6):
        #   elephant.causality.granger.pairwise_granger(_ar_bivariate(), max_order=1,
        #                                               information_criterion='aic')
        #   -> Causality(directional_causality_x_y=0.38, directional_causality_y_x=0.0, ...)
        # Elephant fits by Yule-Walker and rounds to 2 significant figures.
        gc = granger_causality(_ar_bivariate(), 1)
        assert abs(float(gc[1, 0]) - 0.38) < 0.006
        assert abs(float(gc[0, 1]) - 0.0) < 0.005

    def test_analytic_bivariate_ar1(self, x64):
        scipy_linalg = pytest.importorskip('scipy.linalg')
        # Population value for order 1: full residual variance is 1 (the noise),
        # restricted is gamma_y(0) - gamma_y(1)^2 / gamma_y(0) from the stationary
        # covariance of the VAR(1) (Lyapunov equation).
        a = np.array([[0.5, 0.], [0.6, 0.4]])
        cov0 = scipy_linalg.solve_discrete_lyapunov(a, np.eye(2))
        cov1 = a @ cov0
        expected = np.log(cov0[1, 1] - cov1[1, 1] ** 2 / cov0[1, 1])
        gc = granger_causality(_ar_bivariate(n=4000, seed=7), 1)
        assert abs(float(gc[1, 0]) - expected) < 0.06
        assert float(gc[0, 1]) < 0.005
        np.testing.assert_array_equal(np.diag(gc), 0.)

    def test_conditional(self, x64):
        s = _var_chain()
        gc = granger_causality(s, 2)
        assert gc[1, 0] > 0.2 and gc[2, 1] > 0.1
        assert gc[2, 0] < 0.005  # x -> z is mediated by y
        bivariate = granger_causality(s[:, [0, 2]], 3)
        assert bivariate[1, 0] > 0.02  # without conditioning, x seems to drive z

    def test_collinear_no_nan(self):
        s = _ar_bivariate(n=500)
        c = np.stack([s[:, 0], s[:, 0], 2 * s[:, 0], s[:, 1]], axis=1)
        gc = granger_causality(c, 2)
        assert np.all(np.isfinite(gc))
        np.testing.assert_allclose(gc[3, :3], 0., atol=1e-3)  # redundant copies
        const = granger_causality(jnp.ones((50, 3)), 2)
        np.testing.assert_allclose(const, 0.)
        assert np.all(np.isfinite(granger_causality(jnp.zeros((30, 2)), 1)))

    def test_units(self):
        s = _ar_bivariate(n=300)
        np.testing.assert_allclose(granger_causality(s * u.mV, 1), granger_causality(s, 1),
                                   rtol=1e-5, atol=1e-6)

    def test_jit_vmap(self):
        trials = jnp.stack([jnp.asarray(_ar_bivariate(n=400, seed=k), dtype=jnp.float32)
                            for k in range(3)])
        f = jax.jit(granger_causality, static_argnums=1)
        np.testing.assert_allclose(f(trials[0], 1), granger_causality(trials[0], 1),
                                   rtol=1e-5, atol=1e-6)
        out = jax.vmap(lambda s: granger_causality(s, 1))(trials)
        assert out.shape == (3, 2, 2)
        for k in range(3):
            np.testing.assert_allclose(out[k], granger_causality(trials[k], 1),
                                       rtol=1e-4, atol=1e-5)
        assert np.all(out[:, 1, 0] > 0.2)

    def test_grad(self, x64):
        s = jnp.asarray(_ar_bivariate(n=60, seed=3))
        g = jax.grad(lambda z: granger_causality(z, 1)[1, 0])(s)
        assert g.shape == s.shape and np.all(np.isfinite(g))
        check_grads(lambda z: granger_causality(z, 1)[1, 0], (s,), order=1, modes=['rev'],
                    atol=1e-4, rtol=1e-4)

    def test_precision(self, x64):
        s = _var_chain(n=1000)
        g64 = granger_causality(jnp.asarray(s, dtype=jnp.float64), 2)
        g32 = granger_causality(jnp.asarray(s, dtype=jnp.float32), 2)
        assert g64.dtype == jnp.float64 and g32.dtype == jnp.float32
        np.testing.assert_allclose(g32, g64, atol=2e-4)

    def test_memory_bounded(self):
        # The restricted fits must not materialise the (M, N p + 1) design matrix
        # (let alone N copies of it, ~300 MB here before the fix): temporaries are
        # bounded by a few copies of the data plus O(P^2) for the Gram matrices.
        n_time, n_var, order = 20000, 16, 5
        compiled = jax.jit(granger_causality, static_argnums=1).lower(
            jax.ShapeDtypeStruct((n_time, n_var), jnp.float32), order).compile()
        try:
            stats = compiled.memory_analysis()
        except Exception:  # pragma: no cover - backend without memory analysis
            stats = None
        if stats is None:  # pragma: no cover
            pytest.skip('memory_analysis is not available on this backend')
        data_bytes = n_time * n_var * 4
        n_reg = n_var * order + 1
        bound = 4 * data_bytes + 4 * n_reg ** 2 * 8
        assert stats.temp_size_in_bytes < bound, stats.temp_size_in_bytes
        assert bound < (n_time - order) * n_reg * 4  # i.e. less than one design matrix

    def test_errors(self):
        with pytest.raises(ValueError, match='shape'):
            granger_causality(jnp.zeros(10), 1)
        with pytest.raises(ValueError, match='exceed'):
            granger_causality(jnp.zeros((3, 2)), 3)
        with pytest.raises(ValueError):
            granger_causality(jnp.zeros((10, 2)), 0)
        with pytest.raises(TypeError):
            granger_causality(jnp.zeros((10, 2)), 1.5)
        with pytest.raises(TypeError):
            granger_causality(jnp.zeros((10, 2)), True)


# ----------------------------------------------------------------------------
# avalanches
# ----------------------------------------------------------------------------

def _numpy_avalanches(counts, threshold=0):
    sizes, durations = [], []
    size = dur = 0
    for c in counts:
        if c > threshold:
            size += c - threshold
            dur += 1
        elif dur:
            sizes.append(size)
            durations.append(dur)
            size = dur = 0
    if dur:
        sizes.append(size)
        durations.append(dur)
    return np.array(sizes), np.array(durations, dtype=int)


def _check(res, counts, threshold=0):
    sizes, durations = _numpy_avalanches(np.asarray(counts), threshold)
    n = len(sizes)
    assert int(res.n) == n
    np.testing.assert_allclose(res.sizes[:n], sizes, rtol=1e-6)
    np.testing.assert_array_equal(res.durations[:n], durations)
    np.testing.assert_array_equal(res.sizes[n:], 0)
    np.testing.assert_array_equal(res.durations[n:], 0)


class TestAvalanches:
    def test_example(self):
        res = avalanches(jnp.array([2, 1, 0, 0, 3, 0, 1, 4, 2]), max_avalanches=4)
        assert isinstance(res, AvalancheResult)
        np.testing.assert_array_equal(res.sizes, [3, 3, 7, 0])
        np.testing.assert_array_equal(res.durations, [2, 1, 3, 0])
        assert int(res.n) == 3

    def test_random_reference(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            counts = rng.poisson(0.8, size=200)
            res = avalanches(counts)
            assert res.sizes.shape == (101,)
            _check(res, counts)

    def test_threshold(self):
        counts = np.random.default_rng(1).poisson(2.0, size=100).astype(np.float32)
        _check(avalanches(counts, threshold=1.5), counts, threshold=1.5)
        _check(avalanches(counts, threshold=2), counts, threshold=2)

    def test_start_and_end_inside(self):
        counts = jnp.array([1, 2, 0, 3, 0, 0, 5])
        res = avalanches(counts)
        np.testing.assert_array_equal(res.sizes[:3], [3, 3, 5])
        np.testing.assert_array_equal(res.durations[:3], [2, 1, 1])
        assert int(res.n) == 3

    def test_all_zero(self):
        res = avalanches(jnp.zeros(10, dtype=jnp.int32))
        assert int(res.n) == 0
        np.testing.assert_array_equal(res.sizes, 0)
        np.testing.assert_array_equal(res.durations, 0)

    def test_no_quiet_bins(self):
        res = avalanches(jnp.array([1, 3, 2, 5]))
        assert int(res.n) == 1
        assert int(res.sizes[0]) == 11 and int(res.durations[0]) == 4

    def test_default_capacity_is_enough(self):
        for t in (1, 2, 5, 6):
            counts = jnp.arange(t) % 2 == 0  # alternating, starting active
            res = avalanches(counts.astype(jnp.int32))
            assert res.sizes.shape == (t // 2 + 1,)
            assert int(res.n) == (t + 1) // 2

    def test_overflow_clipped(self):
        res = avalanches(jnp.array([1, 0, 2, 0, 3, 0, 4]), max_avalanches=2)
        assert int(res.n) == 2
        np.testing.assert_array_equal(res.sizes, [1, 2])

    def test_units(self):
        counts = jnp.array([1., 0., 2.])
        res = avalanches(counts * u.UNITLESS)
        assert int(res.n) == 2
        with pytest.raises(TypeError):
            avalanches(counts * u.ms)

    def test_from_spike_trains(self):
        from braintools.spike import SpikeTrains
        raster = jnp.zeros((10, 4)).at[jnp.array([1, 2, 6]), jnp.array([0, 3, 1])].set(1.)
        st = SpikeTrains.from_dense(raster, dt=1.)
        res = avalanches(st.bin(1.).sum(axis=1))
        assert int(res.n) == 2
        np.testing.assert_array_equal(res.sizes[:2], [2, 1])

    def test_jit_vmap(self):
        counts = jnp.asarray(np.random.default_rng(2).poisson(0.7, size=(4, 50)))
        res = jax.jit(avalanches, static_argnums=2)(counts[0], 0, 30)
        _check(res, counts[0])
        batched = jax.vmap(lambda c: avalanches(c))(counts)
        assert batched.sizes.shape == (4, 26)
        for k in range(4):
            _check(AvalancheResult(batched.sizes[k], batched.durations[k], batched.n[k]), counts[k])

    def test_errors(self):
        with pytest.raises(ValueError):
            avalanches(jnp.zeros((3, 2)))
        with pytest.raises(ValueError):
            avalanches(jnp.zeros(4), max_avalanches=0)
        with pytest.raises(TypeError):
            avalanches(jnp.zeros(4), max_avalanches=2.5)
        with pytest.raises(TypeError):
            avalanches(jnp.zeros(4), max_avalanches=True)


# ----------------------------------------------------------------------------
# branching_ratio
# ----------------------------------------------------------------------------

class TestBranchingRatio:
    def test_ratio_reference(self):
        counts = np.random.default_rng(0).poisson(1.0, size=300).astype(float)
        a, d = counts[:-1], counts[1:]
        ref = np.mean(d[a > 0] / a[a > 0])
        np.testing.assert_allclose(branching_ratio(counts), ref, rtol=1e-5)
        np.testing.assert_allclose(branching_ratio(jnp.array([1., 2., 4., 0., 3., 3.])), 1.25)

    def test_regression_reference(self):
        stats = pytest.importorskip('scipy.stats')
        counts = np.random.default_rng(1).poisson(3.0, size=300).astype(float)
        ref = stats.linregress(counts[:-1], counts[1:]).slope
        np.testing.assert_allclose(branching_ratio(counts, 'regression'), ref, rtol=1e-4, atol=1e-6)

    def test_branching_process(self, x64):
        # n_{t+1} ~ Poisson(m n_t + h): the lag-1 regression slope estimates m.
        rng = np.random.default_rng(2)
        m, h = 0.9, 2.0
        n = np.zeros(20000)
        n[0] = h / (1 - m)
        for t in range(1, n.size):
            n[t] = rng.poisson(m * n[t - 1] + h)
        assert abs(float(branching_ratio(n, 'regression')) - m) < 0.02

    def test_undefined_is_nan(self):
        assert np.isnan(branching_ratio(jnp.zeros(5)))
        assert np.isnan(branching_ratio(jnp.zeros(5), 'regression'))
        assert np.isnan(branching_ratio(jnp.full(5, 3.), 'regression'))
        assert float(branching_ratio(jnp.array([0., 2., 0.]))) == 0.

    def test_int_and_unitless(self):
        counts = jnp.array([1, 2, 4, 0, 3, 3])
        np.testing.assert_allclose(branching_ratio(counts), 1.25)
        np.testing.assert_allclose(branching_ratio(counts * u.UNITLESS), 1.25)

    def test_jit_vmap(self):
        counts = jnp.asarray(np.random.default_rng(3).poisson(2.0, size=(5, 40)), dtype=jnp.float32)
        for method in ('ratio', 'regression'):
            f = jax.jit(branching_ratio, static_argnums=1)
            np.testing.assert_allclose(f(counts[0], method), branching_ratio(counts[0], method),
                                       rtol=1e-6)
            out = jax.vmap(lambda c: branching_ratio(c, method))(counts)
            ref = [branching_ratio(c, method) for c in counts]
            np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_grad(self, x64):
        counts = jnp.asarray(np.random.default_rng(4).uniform(0.5, 3., size=20))
        for method in ('ratio', 'regression'):
            check_grads(lambda c: branching_ratio(c, method), (counts,), order=1, modes=['rev'])
        g = jax.grad(branching_ratio)(jnp.array([0., 1., 2., 0.]))
        assert np.all(np.isfinite(g))

    def test_errors(self):
        with pytest.raises(ValueError):
            branching_ratio(jnp.ones(5), method='mr')
        with pytest.raises(ValueError):
            branching_ratio(jnp.ones(1))
        with pytest.raises(ValueError):
            branching_ratio(jnp.ones((4, 2)))
