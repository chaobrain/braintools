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

import functools
import warnings

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.analysis.spectral import power_spectral_density
from braintools.analysis.spike import van_rossum_distance
from braintools.fit._losses import (
    correlation_loss, cv_loss, fc_correlation_loss, fc_rmse, fcd_ks, fcd_wasserstein,
    ks_distance, mae, mse, psd_log_distance, rate_loss, rmse, van_rossum_loss,
    wasserstein_1d,
)
from braintools.spike import SpikeTrains
from braintools.surrogate import sigmoid


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _signals(seed=0, shape=(120, 5)):
    rng = _rng(seed)
    a = rng.standard_normal(shape)
    b = 0.6 * a + 0.8 * rng.standard_normal(shape)
    return jnp.asarray(a, dtype=jnp.float32), jnp.asarray(b, dtype=jnp.float32)


@pytest.fixture
def bm():
    """brainmass.objectives; its braintools.metric calls emit deprecation warnings."""
    brainmass = pytest.importorskip('brainmass')
    return brainmass.objectives


def _bm_call(fn, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return fn(*args)


# ----------------------------------------------------------------------------
# mse / rmse / mae
# ----------------------------------------------------------------------------

class TestPointwise:
    def test_reference_values(self):
        a, b = _signals()
        d = np.asarray(a) - np.asarray(b)
        np.testing.assert_allclose(mse(a, b), np.mean(d ** 2), rtol=1e-5)
        np.testing.assert_allclose(rmse(a, b), np.sqrt(np.mean(d ** 2)), rtol=1e-5)
        np.testing.assert_allclose(mae(a, b), np.mean(np.abs(d)), rtol=1e-5)

    def test_rmse_matches_brainmass(self, bm):
        a, b = _signals()
        np.testing.assert_allclose(rmse(a, b), bm.timeseries_rmse()(a, b), rtol=1e-6)

    @pytest.mark.parametrize('loss', [mse, rmse, mae])
    def test_identical_is_zero(self, loss):
        a, _ = _signals()
        assert float(loss(a, a)) == 0.

    @pytest.mark.parametrize('loss', [mse, rmse, mae])
    def test_jit_and_vmap(self, loss):
        a, b = _signals(shape=(4, 30, 3))
        out = jax.vmap(jax.jit(loss))(a, b)
        ref = jnp.stack([loss(a[i], b[i]) for i in range(4)])
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_grad(self):
        a, b = _signals(shape=(20, 3))
        g = jax.grad(mse)(a, b)
        np.testing.assert_allclose(g, 2 * (a - b) / a.size, rtol=1e-5)
        g = jax.grad(mae)(a, b)
        np.testing.assert_allclose(g, np.sign(a - b) / a.size, rtol=1e-6)
        g = jax.grad(rmse)(a, b)
        np.testing.assert_allclose(g, (a - b) / (a.size * rmse(a, b)), rtol=1e-4)

    def test_rmse_grad_at_perfect_fit_is_zero(self):
        a, _ = _signals(shape=(10, 2))
        g = jax.grad(rmse)(a, a)
        assert np.all(np.isfinite(g)) and np.all(g == 0)

    def test_units(self):
        a, b = _signals(shape=(10, 2))
        ref = mse(a, b)
        # target in volts is converted to the prediction's unit (mV).
        np.testing.assert_allclose(mse(a * u.mV, (b * 1e-3) * u.volt), ref, rtol=1e-5)
        np.testing.assert_allclose(rmse(a * u.mV, b * u.mV), rmse(a, b), rtol=1e-6)
        np.testing.assert_allclose(mae(a * u.volt, (b * 1e3) * u.mV), mae(a, b), rtol=1e-5)
        # unitless Quantities count as plain arrays
        np.testing.assert_allclose(mse(u.Quantity(a), b), ref, rtol=1e-6)

    def test_unit_errors(self):
        a, b = _signals(shape=(10, 2))
        with pytest.raises(TypeError, match='both carry compatible units'):
            mse(a * u.mV, b)
        with pytest.raises(TypeError, match='both carry compatible units'):
            rmse(a, b * u.mV)
        with pytest.raises(u.UnitMismatchError):
            mae(a * u.mV, b * u.second)

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match='same shape'):
            mse(jnp.zeros(3), jnp.zeros(4))

    def test_integer_input(self):
        assert float(mse(jnp.array([1, 2]), jnp.array([1, 4]))) == 2.


# ----------------------------------------------------------------------------
# correlation_loss
# ----------------------------------------------------------------------------

class TestCorrelationLoss:
    def test_reference_flattened(self):
        a, b = _signals()
        r = np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1]
        np.testing.assert_allclose(correlation_loss(a, b), 1 - r, rtol=1e-5)

    def test_reference_per_channel(self):
        a, b = _signals()
        rs = [np.corrcoef(np.asarray(a)[:, i], np.asarray(b)[:, i])[0, 1] for i in range(5)]
        np.testing.assert_allclose(correlation_loss(a, b, per_channel=True),
                                   1 - np.mean(rs), rtol=1e-5)

    def test_invariances_and_bounds(self):
        a, _ = _signals()
        np.testing.assert_allclose(correlation_loss(3 * a + 2, a), 0., atol=1e-6)
        np.testing.assert_allclose(correlation_loss(-a, a), 2., atol=1e-6)
        # units are not compared (scale invariant)
        np.testing.assert_allclose(correlation_loss(a * u.mV, a * u.nA), 0., atol=1e-6)

    def test_constant_input(self):
        a, _ = _signals(shape=(20,))
        c = jnp.ones(20)
        assert float(correlation_loss(c, a)) == 1.
        g = jax.grad(correlation_loss)(c, a)
        assert np.all(np.isfinite(g))
        g = jax.grad(functools.partial(correlation_loss, per_channel=True))(
            jnp.ones((20, 2)), jnp.ones((20, 2)))
        assert np.all(np.isfinite(g))

    def test_jit_vmap(self):
        a, b = _signals(shape=(3, 40, 2))
        f = jax.jit(functools.partial(correlation_loss, per_channel=True))
        out = jax.vmap(f)(a, b)
        ref = jnp.stack([correlation_loss(a[i], b[i], per_channel=True) for i in range(3)])
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_grad_matches_finite_difference(self, x64):
        rng = _rng(1)
        a = jnp.asarray(rng.standard_normal(15))
        b = jnp.asarray(rng.standard_normal(15))
        g = jax.grad(correlation_loss)(a, b)
        h = 1e-6
        e = jnp.zeros(15).at[3].set(h)
        fd = (correlation_loss(a + e, b) - correlation_loss(a - e, b)) / (2 * h)
        np.testing.assert_allclose(g[3], fd, rtol=1e-5)

    def test_errors(self):
        with pytest.raises(ValueError, match='same shape'):
            correlation_loss(jnp.zeros(3), jnp.zeros(4))
        with pytest.raises(ValueError, match='time axis'):
            correlation_loss(jnp.float32(1.), jnp.float32(2.), per_channel=True)


# ----------------------------------------------------------------------------
# functional connectivity
# ----------------------------------------------------------------------------

def _np_fc(x):
    fc = np.corrcoef(np.asarray(x, dtype=float).T)
    iu = np.triu_indices(fc.shape[0], k=1)
    return fc[iu]


class TestFC:
    def test_fc_correlation_reference(self):
        a, b = _signals()
        r = np.corrcoef(_np_fc(a), _np_fc(b))[0, 1]
        np.testing.assert_allclose(fc_correlation_loss(a, b), 1 - r, rtol=1e-4, atol=1e-6)

    def test_fc_correlation_matches_brainmass(self, bm):
        a, b = _signals()
        ref = _bm_call(bm.fc_corr(as_loss=True), a, b)
        np.testing.assert_allclose(fc_correlation_loss(a, b), ref, rtol=1e-6, atol=1e-7)

    def test_fc_rmse_reference(self):
        a, b = _signals()
        d = _np_fc(a) - _np_fc(b)
        np.testing.assert_allclose(fc_rmse(a, b), np.sqrt(np.mean(d ** 2)), rtol=1e-4)

    def test_fc_rmse_relation_to_brainmass(self, bm):
        a, b = _signals()
        n = a.shape[1]
        ref = _bm_call(bm.fc_rmse(), a, b)
        np.testing.assert_allclose(ref ** 2, fc_rmse(a, b) ** 2 * (n - 1) / n, rtol=1e-5)

    @pytest.mark.parametrize('loss', [fc_correlation_loss, fc_rmse])
    def test_identical_and_different_lengths(self, loss):
        a, b = _signals()
        np.testing.assert_allclose(loss(a, a), 0., atol=1e-6)
        assert np.isfinite(float(loss(a, b[:80])))
        # Quantities: magnitudes are used (scale invariant)
        np.testing.assert_allclose(loss(a * u.mV, b), loss(a, b), rtol=1e-6)

    @pytest.mark.parametrize('loss', [fc_correlation_loss, fc_rmse])
    def test_jit_vmap_grad(self, loss):
        a, b = _signals(shape=(3, 60, 4))
        out = jax.vmap(jax.jit(loss))(a, b)
        ref = jnp.stack([loss(a[i], b[i]) for i in range(3)])
        np.testing.assert_allclose(out, ref, rtol=1e-5)
        g = jax.grad(loss)(a[0], b[0])
        assert np.all(np.isfinite(g)) and np.any(g != 0)

    def test_fc_rmse_grad_at_perfect_fit(self):
        a, _ = _signals(shape=(40, 3))
        g = jax.grad(fc_rmse)(a, a)
        assert np.all(np.isfinite(g))

    @pytest.mark.parametrize('loss', [fc_correlation_loss, fc_rmse])
    def test_errors(self, loss):
        with pytest.raises(ValueError, match='n_time, n_signals'):
            loss(jnp.zeros(10), jnp.zeros((10, 2)))
        with pytest.raises(ValueError, match='at least 2 signals'):
            loss(jnp.zeros((10, 1)), jnp.zeros((10, 1)))
        with pytest.raises(ValueError, match='same number of regions'):
            loss(jnp.ones((10, 2)), jnp.ones((10, 3)))


# ----------------------------------------------------------------------------
# ks_distance / wasserstein_1d
# ----------------------------------------------------------------------------

class TestDistributionDistances:
    def test_ks_matches_scipy_ks_2samp(self):
        stats = pytest.importorskip('scipy.stats')
        rng = _rng(2)
        grid = np.arange(10.)
        s1 = rng.integers(0, 10, 50)
        s2 = rng.integers(2, 10, 70)
        p = np.bincount(s1, minlength=10).astype(np.float32)
        q = np.bincount(s2, minlength=10).astype(np.float32)
        ref = stats.ks_2samp(grid[s1], grid[s2]).statistic
        np.testing.assert_allclose(ks_distance(p, q), ref, rtol=1e-5)

    def test_wasserstein_matches_scipy(self):
        stats = pytest.importorskip('scipy.stats')
        rng = _rng(3)
        x = np.linspace(-1., 1., 30)
        p = rng.random(30)
        q = rng.random(30)
        ref = stats.wasserstein_distance(x, x, u_weights=p, v_weights=q)
        np.testing.assert_allclose(wasserstein_1d(jnp.asarray(p, jnp.float32), jnp.asarray(q, jnp.float32),
                                                  jnp.asarray(x, jnp.float32)), ref, rtol=1e-4)

    def test_match_brainmass(self, bm):
        rng = _rng(4)
        x = jnp.linspace(0., 2., 40)
        p = jnp.asarray(rng.random(40), jnp.float32)
        q = jnp.asarray(rng.random(40), jnp.float32)
        assert float(ks_distance(p, q)) == float(bm.ks_distance(p, q))
        assert float(wasserstein_1d(p, q, x)) == float(bm.wasserstein_1d(p, q, x))

    def test_normalisation_and_identity(self):
        p = jnp.array([1., 2., 3.])
        assert float(ks_distance(p, 5 * p)) == 0.
        assert float(wasserstein_1d(p, 5 * p, jnp.arange(3.))) == 0.

    def test_units(self):
        p = jnp.array([1., 2., 3.])
        q = jnp.array([3., 2., 1.])
        ref = ks_distance(p, q)
        np.testing.assert_allclose(ks_distance(p / u.mV, (q * 1e3) / u.volt), ref, rtol=1e-6)
        w = wasserstein_1d(p, q, jnp.arange(3.) * u.mV)
        assert isinstance(w, u.Quantity)
        np.testing.assert_allclose(w.to_decimal(u.mV), wasserstein_1d(p, q, jnp.arange(3.)), rtol=1e-6)
        with pytest.raises(TypeError):
            ks_distance(p / u.mV, q)

    def test_jit_vmap_grad(self):
        rng = _rng(5)
        p = jnp.asarray(rng.random((4, 20)), jnp.float32)
        q = jnp.asarray(rng.random((4, 20)), jnp.float32)
        x = jnp.linspace(0., 1., 20)
        out = jax.vmap(jax.jit(ks_distance))(p, q)
        np.testing.assert_allclose(out, [ks_distance(p[i], q[i]) for i in range(4)], rtol=1e-6)
        out = jax.vmap(jax.jit(wasserstein_1d), in_axes=(0, 0, None))(p, q, x)
        np.testing.assert_allclose(out, [wasserstein_1d(p[i], q[i], x) for i in range(4)], rtol=1e-6)
        g = jax.grad(wasserstein_1d)(p[0], q[0], x)
        assert np.all(np.isfinite(g)) and np.any(g != 0)
        g = jax.grad(ks_distance)(p[0], q[0])
        assert np.all(np.isfinite(g)) and np.any(g != 0)

    def test_errors(self):
        with pytest.raises(ValueError, match='1-D'):
            ks_distance(jnp.ones((2, 2)), jnp.ones((2, 2)))
        with pytest.raises(ValueError, match='same shape'):
            ks_distance(jnp.ones(3), jnp.ones(4))
        with pytest.raises(ValueError, match='at least 2 points'):
            wasserstein_1d(jnp.ones(1), jnp.ones(1), jnp.ones(1))


# ----------------------------------------------------------------------------
# FCD losses
# ----------------------------------------------------------------------------

def _fcd_signals(seed=0, shape=(150, 5)):
    rng = _rng(seed)
    # slowly drifting coupling makes the FCD non-trivial
    base = rng.standard_normal(shape)
    common = np.cumsum(rng.standard_normal(shape[0]))[:, None] * 0.2
    a = base + common * np.linspace(0, 1, shape[1])[None]
    b = rng.standard_normal(shape)
    return jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)


class TestFCD:
    @pytest.mark.parametrize('kw', [
        {},
        dict(window_size=20, step_size=4),
        dict(window_size=25, step_size=5, bw_method=0.3, n_diag=2),
        dict(window_size=30, step_size=6, midpoints=jnp.linspace(-1., 1., 41)),
    ])
    def test_match_brainmass(self, bm, kw):
        a, b = _fcd_signals()
        for ours, theirs in ((fcd_ks, bm.fcd_ks), (fcd_wasserstein, bm.fcd_wasserstein)):
            ref = _bm_call(theirs(**kw), a, b)
            np.testing.assert_allclose(ours(a, b, **kw), ref, rtol=1e-6, atol=1e-7)
            assert float(ref) > 0

    def test_reference_by_hand(self):
        stats = pytest.importorskip('scipy.stats')
        a, b = _fcd_signals()

        def density(x):
            from braintools.analysis.population import functional_connectivity_dynamics
            fcd = np.asarray(functional_connectivity_dynamics(x, window_size=30, step_size=5),
                             dtype=float)
            vals = fcd[np.triu_indices(fcd.shape[0], k=1)]
            mid = np.linspace(-0.99, 0.99, 100)
            d = stats.gaussian_kde(vals).evaluate(mid)
            return d / (d.sum() * (mid[1] - mid[0]))

        dp, dq = density(a), density(b)
        cp, cq = np.cumsum(dp) / np.sum(dp), np.cumsum(dq) / np.sum(dq)
        np.testing.assert_allclose(fcd_ks(a, b), np.max(np.abs(cp - cq)), rtol=1e-3)
        np.testing.assert_allclose(fcd_wasserstein(a, b),
                                   np.sum(np.abs(cp - cq)) * (1.98 / 99), rtol=1e-3)

    @pytest.mark.parametrize('loss', [fcd_ks, fcd_wasserstein])
    def test_identical_is_zero(self, loss):
        a, _ = _fcd_signals()
        assert float(loss(a, a)) == 0.

    @pytest.mark.parametrize('loss', [fcd_ks, fcd_wasserstein])
    def test_jit_vmap(self, loss):
        a, b = _fcd_signals(shape=(100, 4))
        a2, b2 = _fcd_signals(seed=1, shape=(100, 4))
        f = jax.jit(functools.partial(loss, window_size=20, step_size=5))
        out = jax.vmap(f)(jnp.stack([a, a2]), jnp.stack([b, b2]))
        ref = [loss(a, b, window_size=20, step_size=5), loss(a2, b2, window_size=20, step_size=5)]
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-7)

    def test_wasserstein_grad(self):
        a, b = _fcd_signals(shape=(100, 4))
        g = jax.grad(functools.partial(fcd_wasserstein, window_size=20, step_size=5))(a, b)
        assert np.all(np.isfinite(g)) and np.any(g != 0)

    @pytest.mark.parametrize('loss', [fcd_ks, fcd_wasserstein])
    def test_errors(self, loss):
        a, b = _fcd_signals(shape=(40, 3))
        with pytest.raises(ValueError, match='KDE needs at least 2'):
            loss(a, b, window_size=30, step_size=5, n_diag=2)  # 3 windows -> 1 value
        with pytest.raises(ValueError, match='KDE needs at least 2'):
            loss(a, b[:20], window_size=30)            # too short: no window at all
        with pytest.raises(ValueError, match='n_time, n_signals'):
            loss(a[:, 0], b)
        loss(a, b, window_size=30, step_size=5, n_diag=1)  # 3 values: fine

    @staticmethod
    def _temp_mib(fn, shape):
        x = jax.ShapeDtypeStruct(shape, jnp.float32)
        compiled = jax.jit(fn).lower(x, x).compile()
        try:
            analysis = compiled.memory_analysis()
        except Exception:  # pragma: no cover - backend without the API
            analysis = None
        if analysis is None or not hasattr(analysis, 'temp_size_in_bytes'):  # pragma: no cover
            pytest.skip('memory_analysis() is not available on this backend')
        return analysis.temp_size_in_bytes / 2 ** 20

    @pytest.mark.parametrize('loss', [fcd_ks, fcd_wasserstein])
    def test_temp_memory_bounded(self, loss):
        # T = 5000, 20 regions, window 30 / step 5: 995 windows, ~5e5 FCD values.
        # Evaluating the KDE of all values on all 100 midpoints at once needed
        # 192 MiB (forward) and 588 MiB (gradient) of temporaries; only compiled.
        f = functools.partial(loss, window_size=30, step_size=5)
        assert self._temp_mib(f, (5000, 20)) < 48
        if loss is fcd_wasserstein:
            assert self._temp_mib(jax.grad(f), (5000, 20)) < 64

    def test_chunked_kde_matches_single_block(self, bm):
        import braintools.fit._losses as mod
        a, b = _fcd_signals()
        kw = dict(window_size=20, step_size=4, midpoints=jnp.linspace(-1., 1., 37))
        old = mod._KDE_BLOCK_ELEMENTS
        mod._KDE_BLOCK_ELEMENTS = 5000   # a few midpoints per block, with a remainder
        try:
            for ours, theirs in ((fcd_ks, bm.fcd_ks), (fcd_wasserstein, bm.fcd_wasserstein)):
                ref = _bm_call(theirs(**kw), a, b)
                np.testing.assert_allclose(ours(a, b, **kw), ref, rtol=1e-6, atol=1e-7)
                np.testing.assert_allclose(jax.jit(functools.partial(ours, **kw))(a, b), ref,
                                           rtol=1e-6, atol=1e-7)
            g_chunk = jax.grad(functools.partial(fcd_wasserstein, **kw))(a, b)
        finally:
            mod._KDE_BLOCK_ELEMENTS = old
        g_full = jax.grad(functools.partial(fcd_wasserstein, **kw))(a, b)
        np.testing.assert_allclose(g_chunk, g_full, rtol=1e-4, atol=1e-7)


class TestFCDErrorCount:
    def test_n_diag_two_with_three_windows(self):
        a, b = _fcd_signals(shape=(40, 3))
        with pytest.raises(ValueError, match='1 values'):
            fcd_ks(a, b, window_size=30, step_size=5, n_diag=2)


# ----------------------------------------------------------------------------
# psd_log_distance
# ----------------------------------------------------------------------------

def _tones(n=1000, fs=1000., seed=0):
    t = np.arange(n) / fs
    rng = _rng(seed)
    low = np.sin(2 * np.pi * 10 * t)
    high = 0.5 * np.sin(2 * np.pi * 120 * t)
    noise = 0.1 * rng.standard_normal(n)
    return low, high, noise


class TestPSDLogDistance:
    def test_matches_scipy_welch(self, x64):
        signal = pytest.importorskip('scipy.signal')
        low, high, noise = _tones()
        p, t = low + high + noise, low + noise
        nper = 200
        _, pp = signal.welch(p, fs=1000., window=np.hanning(nper), nperseg=nper,
                             detrend=False, scaling='density')
        _, pt = signal.welch(t, fs=1000., window=np.hanning(nper), nperseg=nper,
                             detrend=False, scaling='density')
        ref = np.mean((np.log10(pp) - np.log10(pt)) ** 2)
        out = psd_log_distance(jnp.asarray(p), jnp.asarray(t), 1e-3, nper)
        np.testing.assert_allclose(out, ref, rtol=1e-8)

    def test_band_restriction(self):
        low, high, noise = _tones()
        p = jnp.asarray(low + high + noise, jnp.float32)
        t = jnp.asarray(low + noise, jnp.float32)
        full = psd_log_distance(p, t, 1e-3, 200)
        in_band = psd_log_distance(p, t, 1e-3, 200, band=(0., 60.))
        out_band = psd_log_distance(p, t, 1e-3, 200, band=(100., 140.))
        assert float(in_band) < 0.05 * float(full)
        assert float(out_band) > float(full)
        # reference: the mean over the selected frequency bins only
        f, pp = power_spectral_density(p, 1e-3, nperseg=200)
        _, pt = power_spectral_density(t, 1e-3, nperseg=200)
        m = np.asarray((f >= 100.) & (f <= 140.))
        ref = np.mean((np.log10(np.asarray(pp)[m]) - np.log10(np.asarray(pt)[m])) ** 2)
        np.testing.assert_allclose(out_band, ref, rtol=1e-4)
        # band as Quantities
        np.testing.assert_allclose(
            psd_log_distance(p, t, 1. * u.ms, 200, band=(0.1 * u.kHz, 140. * u.Hz)),
            out_band, rtol=1e-6)

    def test_gain_and_identity(self):
        low, _, noise = _tones()
        x = jnp.asarray(low + noise, jnp.float32)
        assert float(psd_log_distance(x, x, 1e-3, 100)) == 0.
        np.testing.assert_allclose(psd_log_distance(2 * x, x, 1e-3, 100),
                                   np.log10(4.) ** 2, rtol=1e-4)

    def test_units_and_default_nperseg(self):
        low, high, noise = _tones()
        p = jnp.asarray(low + high + noise, jnp.float32)
        t = jnp.asarray(low + noise, jnp.float32)
        ref = psd_log_distance(p, t, 1e-3)
        np.testing.assert_allclose(psd_log_distance(p, t, 1. * u.ms, 125), ref, rtol=1e-6)
        np.testing.assert_allclose(psd_log_distance(p * u.mV, (t * 1e-3) * u.volt, 1. * u.ms),
                                   ref, rtol=1e-3)
        with pytest.raises(TypeError, match='compatible units'):
            psd_log_distance(p * u.mV, t, 1e-3)
        with pytest.raises(TypeError, match='must be a time'):
            psd_log_distance(p, t, 1. * u.mV)
        with pytest.raises(TypeError, match='frequencies'):
            psd_log_distance(p, t, 1e-3, band=(1. * u.ms, 2. * u.ms))

    def test_multichannel(self):
        low, high, noise = _tones()
        p = jnp.asarray(np.stack([low + high, low + noise], axis=1), jnp.float32)
        t = jnp.asarray(np.stack([low + noise, low + noise], axis=1), jnp.float32)
        both = psd_log_distance(p, t, 1e-3, 200)
        first = psd_log_distance(p[:, 0], t[:, 0], 1e-3, 200)
        np.testing.assert_allclose(both, first / 2, rtol=1e-5)

    def test_jit_vmap_grad(self):
        low, high, noise = _tones(n=400)
        p = jnp.asarray(np.stack([low + high + noise, low + 2 * noise]), jnp.float32)
        t = jnp.asarray(np.stack([low + noise, low + noise]), jnp.float32)
        f = jax.jit(functools.partial(psd_log_distance, dt=1e-3, nperseg=100, band=(5., 200.)))
        out = jax.vmap(f)(p, t)
        ref = [psd_log_distance(p[i], t[i], 1e-3, 100, band=(5., 200.)) for i in range(2)]
        np.testing.assert_allclose(out, ref, rtol=1e-5)
        g = jax.grad(f)(p[0], t[0])
        assert np.all(np.isfinite(g)) and np.any(g != 0)

    def test_zero_power_floor(self):
        z = jnp.zeros(256)
        assert float(psd_log_distance(z, z, 1e-3, 64)) == 0.
        assert np.isfinite(float(psd_log_distance(z, jnp.ones(256), 1e-3, 64, eps=1e-6)))

    def test_errors(self):
        x = jnp.ones(256)
        with pytest.raises(ValueError, match='none of the Welch'):
            psd_log_distance(x, x, 1e-3, 64, band=(1., 2.))
        with pytest.raises(ValueError, match='n_time'):
            psd_log_distance(jnp.ones((4, 4, 4)), jnp.ones((4, 4, 4)), 1e-3)
        with pytest.raises(ValueError, match='same shape'):
            psd_log_distance(x, x[:100], 1e-3)


# ----------------------------------------------------------------------------
# spike losses
# ----------------------------------------------------------------------------

class TestRateLoss:
    def _rasters(self):
        pred = jnp.zeros((1000, 4)).at[::100].set(1.)                 # 10 Hz at 1 ms
        target = jnp.zeros((1000, 4)).at[::50, :2].set(1.)            # 20, 20, 0, 0 Hz
        return pred, target

    def test_reference(self):
        pred, target = self._rasters()
        assert float(rate_loss(pred, target, 1.)) == pytest.approx(0.)
        assert float(rate_loss(pred, target, 1., per_neuron=True)) == pytest.approx(
            np.mean([100., 100., 100., 100.]))
        target2 = jnp.zeros((500, 4)).at[::25].set(1.)                # 40 Hz, different T
        assert float(rate_loss(pred, target2, 1.)) == pytest.approx(900.)

    def test_1d_raster(self):
        a = jnp.zeros(200).at[::20].set(1.)                           # 50 Hz at 1 ms
        assert float(rate_loss(a, jnp.zeros(200), 1. * u.ms)) == pytest.approx(2500.)

    def test_dt_units_and_environ(self):
        pred, target = self._rasters()
        target = target.at[:, 2:].set(0.)
        ref = rate_loss(pred, target * 0, 1.)
        np.testing.assert_allclose(ref, 100., rtol=1e-6)
        np.testing.assert_allclose(rate_loss(pred, target * 0, 1e-3 * u.second), ref, rtol=1e-6)
        with brainstate.environ.context(dt=1. * u.ms):
            np.testing.assert_allclose(rate_loss(pred, target * 0), ref, rtol=1e-6)
        with pytest.raises(ValueError, match='brainstate.environ'):
            rate_loss(pred, target)
        with pytest.raises(TypeError, match='must be a time'):
            rate_loss(pred, target, 1. * u.mV)

    def test_spike_trains(self):
        pred, target = self._rasters()
        st = SpikeTrains.from_dense(target, dt=1. * u.ms)
        np.testing.assert_allclose(rate_loss(pred, st, 1., per_neuron=True),
                                   rate_loss(pred, target, 1., per_neuron=True), rtol=1e-6)
        st_s = SpikeTrains.from_dense(target, dt=1e-3 * u.second)
        st_plain = SpikeTrains.from_dense(target, dt=1.)               # ms
        np.testing.assert_allclose(rate_loss(st_s, st_plain), 0., atol=1e-6)
        sp = SpikeTrains.from_dense(pred, dt=1.)
        np.testing.assert_allclose(rate_loss(sp, st), rate_loss(pred, target, 1.), atol=1e-6)

    def test_jit_vmap_grad(self):
        pred, target = self._rasters()
        f = jax.jit(functools.partial(rate_loss, dt=1.))
        batch = jnp.stack([pred, 2 * pred])
        out = jax.vmap(f, in_axes=(0, None))(batch, target)
        np.testing.assert_allclose(out, [rate_loss(pred, target, 1.), rate_loss(2 * pred, target, 1.)],
                                   rtol=1e-6)
        g = jax.grad(f)(2 * pred, target)
        # d/ds (mean_p - mean_t)^2 = 2 (mean_p - mean_t) / (T N dt_s)
        np.testing.assert_allclose(g, 2 * (20. - 10.) / (1000 * 4 * 1e-3), rtol=1e-5)
        # dt traced
        out = jax.jit(rate_loss)(pred, target, 1.)
        np.testing.assert_allclose(out, rate_loss(pred, target, 1.), rtol=1e-6)

    def test_errors(self):
        with pytest.raises(ValueError, match='same number of neurons'):
            rate_loss(jnp.zeros((10, 2)), jnp.zeros((10, 3)), 1., per_neuron=True)
        with pytest.raises(ValueError, match=r'\(T,\) or \(T, N\)'):
            rate_loss(jnp.zeros((2, 10, 2)), jnp.zeros((10, 2)), 1.)
        with pytest.raises(TypeError, match='unitless'):
            rate_loss(jnp.zeros(10) * u.mV, jnp.zeros(10), 1.)


def _np_cv(train):
    isi = np.diff(np.asarray(train, dtype=float))
    return np.std(isi) / np.mean(isi) if isi.size >= 2 else np.nan


class TestCVLoss:
    TRAINS_P = [[1., 3., 4., 8., 9.5, 15.], [2., 5., 11., 12.], [7.], []]
    TRAINS_T = [[1., 2., 4.], [0.5, 2.0], [1., 5., 9., 13.]]

    def test_reference_and_nan_neurons_ignored(self):
        sp = SpikeTrains.from_list([np.asarray(t) for t in self.TRAINS_P], t_start=0., t_stop=20.)
        st = SpikeTrains.from_list([np.asarray(t) for t in self.TRAINS_T], t_start=0., t_stop=20.)
        mp = np.nanmean([_np_cv(t) for t in self.TRAINS_P])
        mt = np.nanmean([_np_cv(t) for t in self.TRAINS_T])
        np.testing.assert_allclose(cv_loss(sp, st), (mp - mt) ** 2, rtol=1e-5)
        assert float(cv_loss(sp, sp)) == 0.

    def test_scipy_variation(self):
        stats = pytest.importorskip('scipy.stats')
        sp = SpikeTrains.from_list([np.asarray(self.TRAINS_P[0])], t_start=0., t_stop=20.)
        st = SpikeTrains.from_list([np.asarray(self.TRAINS_P[1])], t_start=0., t_stop=20.)
        ref = (stats.variation(np.diff(self.TRAINS_P[0])) - stats.variation(np.diff(self.TRAINS_P[1]))) ** 2
        np.testing.assert_allclose(cv_loss(sp, st), ref, rtol=1e-5)

    def test_all_nan_gives_nan(self):
        silent = SpikeTrains.from_dense(jnp.zeros((10, 3)), dt=1.)
        regular = SpikeTrains.from_dense(jnp.zeros((10, 1)).at[::3].set(1.), dt=1.)
        assert np.isnan(float(cv_loss(silent, regular)))

    def test_units(self):
        sp = SpikeTrains.from_list([np.asarray(t) * u.ms for t in self.TRAINS_P[:2]],
                                   t_start=0. * u.ms, t_stop=20. * u.ms)
        st = SpikeTrains.from_list([np.asarray(t) * u.second for t in self.TRAINS_T],
                                   t_start=0. * u.second, t_stop=20. * u.second)
        mp = np.mean([_np_cv(t) for t in self.TRAINS_P[:2]])
        mt = np.nanmean([_np_cv(t) for t in self.TRAINS_T])
        np.testing.assert_allclose(cv_loss(sp, st), (mp - mt) ** 2, rtol=1e-5)

    def test_jit_vmap(self):
        rng = _rng(6)
        rasters = jnp.asarray(rng.random((3, 200, 4)) < 0.1, jnp.float32)
        target = SpikeTrains.from_dense(rasters[0], dt=1.)

        def f(r):
            return cv_loss(SpikeTrains.from_dense(r, dt=1., capacity=200), target)

        out = jax.vmap(jax.jit(f))(rasters)
        ref = [cv_loss(SpikeTrains.from_dense(rasters[i], dt=1.), target) for i in range(3)]
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-8)
        assert float(out[0]) == pytest.approx(0., abs=1e-10)

    def test_grad_wrt_spike_times(self):
        times = jnp.array([[1., 3., 4., 8., jnp.inf], [2., 5., 11., jnp.inf, jnp.inf],
                           [7., jnp.inf, jnp.inf, jnp.inf, jnp.inf]])
        counts = jnp.array([4, 3, 1])
        target = SpikeTrains.from_list([np.array([1., 2., 4.])], t_start=0., t_stop=20.)

        def f(tm):
            return cv_loss(SpikeTrains(tm, counts, 0., 20.), target)

        g = jax.grad(f)(times)
        valid = np.isfinite(np.asarray(times))
        assert np.all(np.isfinite(np.asarray(g)))
        assert np.any(np.asarray(g)[valid] != 0)
        assert np.all(np.asarray(g)[2] == 0)    # the nan-CV neuron carries no gradient

    def test_type_error(self):
        st = SpikeTrains.from_dense(jnp.zeros((10, 1)), dt=1.)
        with pytest.raises(TypeError, match='from_dense'):
            cv_loss(jnp.zeros((10, 1)), st)
        with pytest.raises(TypeError, match='target'):
            cv_loss(st, jnp.zeros((10, 1)))


class TestVanRossumLoss:
    def _rasters(self, seed=7, shape=(400, 5), p=0.03):
        rng = _rng(seed)
        a = (rng.random(shape) < p).astype(np.float64)
        b = (rng.random(shape) < p).astype(np.float64)
        return a, b

    @pytest.mark.parametrize('tau', [0.5, 2., 10.])
    def test_matches_event_based_distance_on_grid(self, x64, tau):
        dt = 0.1
        a, b = self._rasters()
        T = a.shape[0]
        out = van_rossum_loss(jnp.asarray(a), jnp.asarray(b), tau, dt, reduction='none')
        for n in range(a.shape[1]):
            ta = np.nonzero(a[:, n])[0] * dt
            tb = np.nonzero(b[:, n])[0] * dt
            ref = van_rossum_distance(jnp.asarray(ta), jnp.asarray(tb), tau=tau, t_max=T * dt)
            np.testing.assert_allclose(out[n], ref, rtol=1e-9, atol=1e-12)

    def test_off_grid_spikes_up_to_discretisation(self, x64):
        dt, tau, T = 0.05, 3., 600
        rng = _rng(8)
        ta = np.sort(rng.uniform(0, T * dt - 10, 6))
        tb = np.sort(rng.uniform(0, T * dt - 10, 7))
        a = np.zeros(T)
        a[np.round(ta / dt).astype(int)] = 1.
        b = np.zeros(T)
        b[np.round(tb / dt).astype(int)] = 1.
        out = van_rossum_loss(jnp.asarray(a), jnp.asarray(b), tau, dt, reduction='none')
        ref = van_rossum_distance(jnp.asarray(ta), jnp.asarray(tb), tau=tau, t_max=T * dt)
        assert abs(float(out) - float(ref)) < 0.05 * float(ref)

    def test_units_and_conventions(self):
        a, b = self._rasters()
        a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
        ref = van_rossum_loss(a, b, 2., 0.1)
        np.testing.assert_allclose(van_rossum_loss(a, b, 2. * u.ms, 0.1 * u.ms), ref, rtol=1e-6)
        np.testing.assert_allclose(van_rossum_loss(a, b, 2e-3 * u.second, 100. * u.us), ref,
                                   rtol=1e-5)
        with brainstate.environ.context(dt=0.1 * u.ms):
            np.testing.assert_allclose(van_rossum_loss(a, b, 2. * u.ms), ref, rtol=1e-6)
        with pytest.raises(ValueError, match='brainstate.environ'):
            van_rossum_loss(a, b, 2.)
        with pytest.raises(TypeError, match='`tau` must be a time'):
            van_rossum_loss(a, b, 2. * u.mV, 0.1)
        np.testing.assert_allclose(van_rossum_loss(a, b, u.Quantity(2.), 0.1), ref, rtol=1e-6)

    def test_reductions(self):
        a, b = self._rasters()
        a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
        per = van_rossum_loss(a, b, 2., 0.1, reduction='none')
        assert per.shape == (5,)
        np.testing.assert_allclose(van_rossum_loss(a, b, 2., 0.1), per.mean(), rtol=1e-6)
        np.testing.assert_allclose(van_rossum_loss(a, b, 2., 0.1, reduction='sum'), per.sum(),
                                   rtol=1e-6)
        sq = van_rossum_loss(a, b, 2., 0.1, reduction='none', squared=True)
        np.testing.assert_allclose(sq, per ** 2, rtol=1e-5)
        one = van_rossum_loss(a[:, 0], b[:, 0], 2., 0.1, reduction='none')
        assert one.shape == ()
        np.testing.assert_allclose(one, per[0], rtol=1e-6)

    def test_identical_is_zero_with_finite_grad(self):
        a, _ = self._rasters()
        a = jnp.asarray(a, jnp.float32)
        assert float(van_rossum_loss(a, a, 2., 0.1)) == 0.
        g = jax.grad(van_rossum_loss)(a, a, 2., 0.1)
        assert np.all(np.isfinite(g)) and np.all(g == 0)

    def test_grad_through_surrogate_spikes(self):
        rng = _rng(9)
        v = jnp.asarray(rng.standard_normal((200, 3)), jnp.float32)
        a, _ = self._rasters(shape=(200, 3), p=0.05)
        target = jnp.asarray(a, jnp.float32)

        def f(v):
            return van_rossum_loss(sigmoid(v - 1.), target, 2., 0.1)

        g = jax.grad(f)(v)
        assert np.all(np.isfinite(g)) and np.any(g != 0)
        g_tau = jax.grad(lambda tau: van_rossum_loss(sigmoid(v - 1.), target, tau, 0.1))(2.)
        assert np.isfinite(float(g_tau)) and float(g_tau) != 0

    def test_grad_matches_finite_difference(self, x64):
        a, b = self._rasters(shape=(100, 2), p=0.1)
        a = jnp.asarray(a) * 0.7
        b = jnp.asarray(b)
        g = jax.grad(van_rossum_loss)(a, b, 2., 0.1)
        h = 1e-6
        e = jnp.zeros_like(a).at[10, 1].set(h)
        fd = (van_rossum_loss(a + e, b, 2., 0.1) - van_rossum_loss(a - e, b, 2., 0.1)) / (2 * h)
        np.testing.assert_allclose(g[10, 1], fd, rtol=1e-5)

    def test_jit_vmap(self):
        a, b = self._rasters(shape=(3, 100, 2), p=0.1)
        a, b = jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)
        f = jax.jit(functools.partial(van_rossum_loss, tau=2. * u.ms, dt=0.1 * u.ms))
        out = jax.vmap(f)(a, b)
        np.testing.assert_allclose(out, [van_rossum_loss(a[i], b[i], 2., 0.1) for i in range(3)],
                                   rtol=1e-5)
        # tau traced
        out = jax.jit(van_rossum_loss)(a[0], b[0], 2., 0.1)
        np.testing.assert_allclose(out, van_rossum_loss(a[0], b[0], 2., 0.1), rtol=1e-6)

    def test_errors(self):
        z = jnp.zeros((10, 2))
        with pytest.raises(ValueError, match='reduction'):
            van_rossum_loss(z, z, 1., 0.1, reduction='max')
        with pytest.raises(ValueError, match='same shape'):
            van_rossum_loss(z, jnp.zeros((10, 3)), 1., 0.1)
        with pytest.raises(ValueError, match=r'\(T,\) or \(T, N\)'):
            van_rossum_loss(jnp.zeros((2, 10, 2)), jnp.zeros((2, 10, 2)), 1., 0.1)
        with pytest.raises(TypeError, match='unitless'):
            van_rossum_loss(z * u.mV, z, 1., 0.1)
