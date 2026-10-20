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

"""Tests for ``braintools.analysis.spectral._spectral``.

Reference values come from ``scipy.signal`` (``spectrogram``, ``welch``,
``sosfiltfilt``, ``hilbert``), ``scipy.integrate.trapezoid`` and ``mne``
(``psd_array_multitaper``, ``tfr_array_morlet``), computed in the tests
themselves, plus NumPy re-implementations of PLV / PLI / wPLI.
"""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from braintools.analysis.spectral._spectral import (
    analytic_signal,
    band_power,
    bandpass,
    morlet_transform,
    multitaper_psd,
    pli,
    plv,
    spectrogram,
    wpli,
)

ss = pytest.importorskip('scipy.signal')
si = pytest.importorskip('scipy.integrate')

FS = 500.0
DT = 1.0 / FS


@pytest.fixture
def x64():
    """Enable float64 for one test and restore the previous setting."""
    prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


def _noise(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def _rel_err(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


# ---------------------------------------------------------------------------
# spectrogram
# ---------------------------------------------------------------------------

class TestSpectrogram:
    @pytest.mark.parametrize('T', [401, 400])
    @pytest.mark.parametrize('nperseg,noverlap', [(64, 20), (65, 0), (50, None)])
    def test_matches_scipy(self, x64, T, nperseg, noverlap):
        x = _noise((T, 3))
        f, t, sxx = spectrogram(x, DT, nperseg, noverlap)
        kw = {} if noverlap is None else {'noverlap': noverlap}
        f2, t2, s2 = ss.spectrogram(x, FS, window='hann', nperseg=nperseg, axis=0, **kw)
        assert sxx.shape == (f2.size, t2.size, 3)
        np.testing.assert_allclose(f, f2)
        np.testing.assert_allclose(t, t2)
        np.testing.assert_allclose(np.moveaxis(np.asarray(sxx), 1, -1), s2, rtol=1e-10, atol=1e-14)

    def test_window_and_detrend_options(self, x64):
        x = _noise((300,), 1) + 3.0
        f, t, sxx = spectrogram(x, DT, 64, 16, window=('tukey', 0.25), detrend=False)
        _, _, s2 = ss.spectrogram(x, FS, window=('tukey', 0.25), nperseg=64, noverlap=16, detrend=False)
        np.testing.assert_allclose(sxx, s2, rtol=1e-10, atol=1e-14)
        win = np.hamming(64)
        _, _, s3 = spectrogram(x, DT, 64, 16, window=win, detrend=None)
        _, _, s4 = ss.spectrogram(x, FS, window=win, nperseg=64, noverlap=16, detrend=False)
        np.testing.assert_allclose(s3, s4, rtol=1e-10, atol=1e-14)

    def test_float32_close_to_scipy(self):
        x = _noise((256, 2), 2).astype(np.float32)
        _, _, sxx = spectrogram(x, DT, 64, 32)
        assert sxx.dtype == jnp.float32
        _, _, s2 = ss.spectrogram(x.astype(np.float64), FS, window='hann', nperseg=64, noverlap=32, axis=0)
        assert _rel_err(np.moveaxis(np.asarray(sxx), 1, -1), s2) < 1e-5

    def test_peak_frequency(self):
        t = np.arange(1000) * DT
        x = np.sin(2 * np.pi * 50 * t)
        f, _, sxx = spectrogram(x, DT, 100, 50)
        assert np.all(np.asarray(f)[np.argmax(np.asarray(sxx), axis=0)] == 50.0)

    def test_units(self):
        x = _noise((200, 2), 3)
        f, t, sxx = spectrogram(x * u.mV, 2.0 * u.ms, 50, 10)
        f2, t2, s2 = spectrogram(x, 2e-3, 50, 10)
        assert isinstance(sxx, u.Quantity)
        assert sxx.unit == u.mV ** 2 * u.second
        assert isinstance(t, u.Quantity)
        np.testing.assert_allclose(t.to_decimal(u.second), t2)
        np.testing.assert_allclose(f, f2)
        np.testing.assert_allclose(sxx.to_decimal(u.mV ** 2 * u.second), s2, rtol=1e-6)
        assert not isinstance(t2, u.Quantity) and not isinstance(s2, u.Quantity)

    def test_jit_and_vmap(self):
        x = _noise((4, 200), 4).astype(np.float32)
        f = jax.jit(lambda s: spectrogram(s, DT, 50, 25)[2])
        ref = np.stack([np.asarray(spectrogram(s, DT, 50, 25)[2]) for s in x])
        np.testing.assert_allclose(f(x[0]), ref[0], rtol=1e-5)
        out = jax.vmap(lambda s: spectrogram(s, DT, 50, 25)[2])(x)
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_grad(self, x64):
        x = jnp.asarray(_noise((80,), 5))
        w = jnp.asarray(_noise((17, 7), 6))
        check_grads(lambda s: jnp.sum(spectrogram(s, DT, 32, 24)[2] * w), (x,), order=1, modes=['rev'])

    def test_errors(self):
        x = np.zeros(50)
        with pytest.raises(ValueError, match='fewer than nperseg'):
            spectrogram(x, DT, 64)
        with pytest.raises(ValueError, match='noverlap'):
            spectrogram(x, DT, 32, 32)
        with pytest.raises(ValueError, match='nperseg'):
            spectrogram(x, DT, 0)
        with pytest.raises(ValueError, match='detrend'):
            spectrogram(x, DT, 32, detrend='linear')
        with pytest.raises(ValueError, match='window'):
            spectrogram(x, DT, 32, window=np.ones(31))
        with pytest.raises(ValueError, match='dt'):
            spectrogram(x, 0.0, 32)
        with pytest.raises(ValueError, match='dt'):
            spectrogram(x, None, 32)
        with pytest.raises(ValueError, match='time axis'):
            spectrogram(np.float32(1.0), DT, 1)

    def test_integer_input(self):
        x = np.arange(64) % 5
        _, _, sxx = spectrogram(x, DT, 32, 0)
        assert jnp.issubdtype(sxx.dtype, jnp.floating)


# ---------------------------------------------------------------------------
# multitaper_psd
# ---------------------------------------------------------------------------

class TestMultitaperPSD:
    @pytest.mark.parametrize('T', [301, 300])
    @pytest.mark.parametrize('remove_dc', [True, False])
    def test_matches_mne(self, x64, T, remove_dc):
        mne = pytest.importorskip('mne')
        x = _noise((T, 3), 7) + 0.5
        nw = 3.0
        # mne keeps int(2 * nw) tapers when low_bias=False.
        f, p = multitaper_psd(x, DT, nw=nw, n_tapers=int(2 * nw), remove_dc=remove_dc)
        pm, fm = mne.time_frequency.psd_array_multitaper(
            x.T, FS, bandwidth=2 * nw * FS / T, adaptive=False, low_bias=False,
            normalization='full', remove_dc=remove_dc, verbose=False)
        np.testing.assert_allclose(f, fm)
        np.testing.assert_allclose(np.asarray(p).T, pm, rtol=1e-9, atol=1e-14)

    def test_default_tapers_numpy_reference(self, x64):
        dpss = pytest.importorskip('scipy.signal.windows').dpss
        x = _noise((257,), 8)
        nw = 2.5
        f, p = multitaper_psd(x, DT, nw=nw)
        k = 4  # default: int(2 * nw) - 1
        tapers, ratios = dpss(257, nw, k, sym=False, norm=2, return_ratios=True)
        xc = x - x.mean()
        spec = np.abs(np.fft.rfft(tapers * xc[None], axis=1)) ** 2
        ref = (ratios[:, None] * spec).sum(0) / ratios.sum() * 2 / FS
        ref[0] /= 2
        np.testing.assert_allclose(p, ref, rtol=1e-10)

    def test_white_noise_level(self):
        # One-sided PSD of unit-variance white noise is 2 / fs.
        x = _noise((4000,), 9).astype(np.float32)
        _, p = multitaper_psd(x, DT, nw=4.0)
        assert abs(float(jnp.mean(p[10:-10])) * FS / 2 - 1.0) < 0.05

    def test_units(self):
        x = _noise((128,), 10)
        _, p = multitaper_psd(x * u.mV, 2 * u.ms, nw=2.0)
        _, p2 = multitaper_psd(x, 2e-3, nw=2.0)
        assert p.unit == u.mV ** 2 * u.second
        np.testing.assert_allclose(p.to_decimal(u.mV ** 2 * u.second), p2, rtol=1e-6)

    def test_jit_vmap_grad(self, x64):
        x = jnp.asarray(_noise((3, 64), 11))
        ref = np.stack([np.asarray(multitaper_psd(s, DT, nw=2.0)[1]) for s in x])
        np.testing.assert_allclose(jax.jit(lambda s: multitaper_psd(s, DT, nw=2.0)[1])(x[0]), ref[0], rtol=1e-10)
        np.testing.assert_allclose(jax.vmap(lambda s: multitaper_psd(s, DT, nw=2.0)[1])(x), ref, rtol=1e-10)
        w = jnp.asarray(_noise((33,), 12))
        check_grads(lambda s: jnp.sum(multitaper_psd(s, DT, nw=2.0)[1] * w), (x[0],), order=1, modes=['rev'])

    def test_errors(self):
        x = np.zeros(20)
        with pytest.raises(ValueError, match='nw'):
            multitaper_psd(x, DT, nw=0.4)
        with pytest.raises(ValueError, match='nw'):
            multitaper_psd(x, DT, nw=10.0)
        with pytest.raises(ValueError, match='n_tapers'):
            multitaper_psd(x, DT, nw=2.0, n_tapers=0)
        with pytest.raises(ValueError, match='n_tapers'):
            multitaper_psd(x, DT, nw=2.0, n_tapers=21)


# ---------------------------------------------------------------------------
# morlet_transform
# ---------------------------------------------------------------------------

class TestMorlet:
    @pytest.mark.parametrize('T', [301, 300])
    @pytest.mark.parametrize('zero_mean', [True, False])
    def test_matches_mne(self, x64, T, zero_mean):
        mne = pytest.importorskip('mne')
        x = _noise((T, 2), 13)
        freqs = np.array([8.0, 20.0, 45.0, 250.0])
        n_cycles = np.array([3.0, 5.0, 7.0, 7.0])
        out = morlet_transform(x, DT, freqs, n_cycles, zero_mean=zero_mean)
        ref = mne.time_frequency.tfr_array_morlet(
            x.T[None], FS, freqs, n_cycles=n_cycles, zero_mean=zero_mean,
            output='complex', verbose=False)[0]  # (n_chans, n_freqs, T)
        assert out.shape == (4, T, 2)
        np.testing.assert_allclose(np.moveaxis(np.asarray(out), -1, 0), ref, rtol=1e-9, atol=1e-12)

    def test_1d_scalar_freq_and_float32(self):
        mne = pytest.importorskip('mne')
        x = _noise((200,), 14).astype(np.float32)
        out = morlet_transform(x, DT, 30.0, n_cycles=4.0)
        assert out.shape == (1, 200) and out.dtype == jnp.complex64
        ref = mne.time_frequency.tfr_array_morlet(
            x.astype(np.float64)[None, None], FS, np.array([30.0]), n_cycles=4.0,
            output='complex', verbose=False)[0, 0]
        assert _rel_err(out, ref) < 1e-5

    def test_peak_frequency(self):
        t = np.arange(1000) * DT
        x = np.sin(2 * np.pi * 20 * t)
        tfr = morlet_transform(x, DT, [10.0, 20.0, 40.0])
        power = np.mean(np.abs(np.asarray(tfr))[:, 300:700] ** 2, axis=1)
        assert int(np.argmax(power)) == 1

    def test_units(self):
        x = _noise((200,), 15)
        out = morlet_transform(x * u.mV, 2 * u.ms, [20.0], 3.0)
        out2 = morlet_transform(x, 2e-3, [20.0], 3.0)
        assert out.unit == u.mV
        np.testing.assert_allclose(out.to_decimal(u.mV), out2, rtol=1e-6)

    def test_jit_vmap_grad(self, x64):
        x = jnp.asarray(_noise((3, 120), 16))
        fn = lambda s: morlet_transform(s, DT, [25.0, 40.0], 3.0)
        ref = np.stack([np.asarray(fn(s)) for s in x])
        np.testing.assert_allclose(jax.jit(fn)(x[0]), ref[0], rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(jax.vmap(fn)(x), ref, rtol=1e-10, atol=1e-14)
        check_grads(lambda s: jnp.sum(jnp.abs(fn(s)) ** 2), (x[0],), order=1, modes=['rev'])

    def test_errors(self):
        x = np.zeros(100)
        with pytest.raises(ValueError, match='freqs'):
            morlet_transform(x, DT, [0.0])
        with pytest.raises(ValueError, match='freqs'):
            morlet_transform(x, DT, [251.0])
        with pytest.raises(ValueError, match='freqs'):
            morlet_transform(x, DT, [])
        with pytest.raises(ValueError, match='n_cycles'):
            morlet_transform(x, DT, [20.0, 30.0], n_cycles=[3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match='n_cycles'):
            morlet_transform(x, DT, [20.0], n_cycles=0.0)
        with pytest.raises(ValueError, match='longer than the signal'):
            morlet_transform(x, DT, [5.0], n_cycles=7.0)


# ---------------------------------------------------------------------------
# bandpass
# ---------------------------------------------------------------------------

class TestBandpass:
    @pytest.mark.parametrize('T', [301, 300])
    @pytest.mark.parametrize('order,band', [(4, (8.0, 30.0)), (2, (1.0, 100.0)), (5, (60.0, 240.0))])
    def test_matches_scipy(self, x64, T, order, band):
        x = _noise((T, 3), 17)
        y = bandpass(x, DT, band, order=order)
        sos = ss.butter(order, band, btype='band', fs=FS, output='sos')
        ref = ss.sosfiltfilt(sos, x, axis=0)
        assert y.shape == x.shape
        assert _rel_err(y, ref) < 1e-5
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-10)

    def test_1d_and_3d(self, x64):
        sos = ss.butter(4, (10.0, 40.0), btype='band', fs=FS, output='sos')
        x1 = _noise((150,), 18)
        np.testing.assert_allclose(bandpass(x1, DT, (10.0, 40.0)), ss.sosfiltfilt(sos, x1), atol=1e-10)
        x3 = _noise((150, 2, 2), 19)
        np.testing.assert_allclose(bandpass(x3, DT, (10.0, 40.0)), ss.sosfiltfilt(sos, x3, axis=0), atol=1e-10)

    def test_float32(self):
        x = _noise((400, 2), 20).astype(np.float32)
        y = bandpass(x, DT, (8.0, 30.0))
        assert y.dtype == jnp.float32
        sos = ss.butter(4, (8.0, 30.0), btype='band', fs=FS, output='sos')
        assert _rel_err(y, ss.sosfiltfilt(sos, x.astype(np.float64), axis=0)) < 1e-3

    def test_padlen_boundary(self, x64):
        # order=4 band-pass: 4 sections, ntaps = 9, padlen = 27 (scipy's rule).
        sos = ss.butter(4, (10.0, 40.0), btype='band', fs=FS, output='sos')
        with pytest.raises(ValueError):
            ss.sosfiltfilt(sos, np.zeros(27))
        with pytest.raises(ValueError, match='padlen=27'):
            bandpass(np.zeros(27), DT, (10.0, 40.0))
        x = _noise((28,), 21)
        np.testing.assert_allclose(bandpass(x, DT, (10.0, 40.0)), ss.sosfiltfilt(sos, x), atol=1e-10)

    def test_units(self):
        x = _noise((200,), 22)
        y = bandpass(x * u.mV, 2 * u.ms, (10.0, 40.0))
        assert isinstance(y, u.Quantity) and y.unit == u.mV
        np.testing.assert_allclose(y.to_decimal(u.mV), bandpass(x, 2e-3, (10.0, 40.0)), rtol=1e-5, atol=1e-6)

    def test_jit_vmap_grad(self, x64):
        x = jnp.asarray(_noise((3, 100), 23))
        fn = lambda s: bandpass(s, DT, (10.0, 60.0), order=3)
        ref = np.stack([np.asarray(fn(s)) for s in x])
        np.testing.assert_allclose(jax.jit(fn)(x[0]), ref[0], atol=1e-12)
        np.testing.assert_allclose(jax.vmap(fn)(x), ref, atol=1e-12)
        w = jnp.asarray(_noise((100,), 24))
        check_grads(lambda s: jnp.sum(fn(s) * w), (x[0],), order=1, modes=['rev'])

    @pytest.mark.parametrize('band', [(0.0, 30.0), (-1.0, 30.0), (10.0, 250.0), (10.0, 300.0),
                                      (30.0, 10.0), (10.0,), 'ab'])
    def test_bad_band(self, band):
        with pytest.raises(ValueError, match='band'):
            bandpass(np.zeros(200), DT, band)

    def test_bad_order(self):
        with pytest.raises(ValueError, match='order'):
            bandpass(np.zeros(200), DT, (10.0, 30.0), order=0)


# ---------------------------------------------------------------------------
# analytic_signal
# ---------------------------------------------------------------------------

class TestAnalyticSignal:
    @pytest.mark.parametrize('T', [1, 2, 7, 8, 101, 100])
    def test_matches_scipy(self, x64, T):
        x = _noise((T, 3), 25)
        np.testing.assert_allclose(analytic_signal(x), ss.hilbert(x, axis=0), atol=1e-12)

    @pytest.mark.parametrize('axis', [1, -1])
    def test_other_axis(self, x64, axis):
        x = _noise((4, 31), 26)
        np.testing.assert_allclose(analytic_signal(x, axis=axis), ss.hilbert(x, axis=axis), atol=1e-12)

    def test_float32_and_envelope(self):
        t = np.arange(1000) * 1e-3
        z = analytic_signal(np.cos(2 * np.pi * 10 * t).astype(np.float32))
        assert z.dtype == jnp.complex64
        np.testing.assert_allclose(np.abs(z), 1.0, atol=1e-4)
        np.testing.assert_allclose(np.real(z), np.cos(2 * np.pi * 10 * t), atol=1e-5)

    def test_units(self):
        x = _noise((50,), 27)
        z = analytic_signal(x * u.mV)
        assert z.unit == u.mV
        np.testing.assert_allclose(z.to_decimal(u.mV), analytic_signal(x), rtol=1e-6)

    def test_jit_vmap_grad(self, x64):
        x = jnp.asarray(_noise((3, 33), 28))
        ref = np.stack([np.asarray(analytic_signal(s)) for s in x])
        np.testing.assert_allclose(jax.jit(analytic_signal)(x[0]), ref[0], atol=1e-14)
        np.testing.assert_allclose(jax.vmap(analytic_signal)(x), ref, atol=1e-14)
        check_grads(lambda s: jnp.sum(jnp.abs(analytic_signal(s)) ** 2), (x[0],), order=1, modes=['rev'])

    def test_errors(self):
        with pytest.raises(ValueError, match='real'):
            analytic_signal(np.ones(4) + 1j)
        with pytest.raises(ValueError, match='dimension'):
            analytic_signal(np.float32(1.0))
        with pytest.raises(ValueError, match='length 0'):
            analytic_signal(np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# band_power
# ---------------------------------------------------------------------------

def _welch_band(x, nperseg, band, relative=False):
    f, p = ss.welch(x, FS, nperseg=nperseg, axis=0)
    m = (f >= band[0]) & (f <= band[1])
    bp = si.trapezoid(p[m], f[m], axis=0)
    return bp / si.trapezoid(p, f, axis=0) if relative else bp


class TestBandPower:
    @pytest.mark.parametrize('T', [601, 600])
    @pytest.mark.parametrize('relative', [False, True])
    def test_matches_scipy(self, x64, T, relative):
        x = _noise((T, 3), 29)
        bp = band_power(x, DT, (8.0, 30.0), nperseg=128, relative=relative)
        assert bp.shape == (3,)
        np.testing.assert_allclose(bp, _welch_band(x, 128, (8.0, 30.0), relative), rtol=1e-10)

    @pytest.mark.parametrize('T', [200, 600])
    def test_default_nperseg(self, x64, T):
        x = _noise((T,), 30)
        # scipy's welch default is nperseg=256, shrunk (with a warning) to T if shorter.
        np.testing.assert_allclose(band_power(x, DT, (20.0, 60.0)),
                                   _welch_band(x, min(256, T), (20.0, 60.0)), rtol=1e-10)

    def test_sinusoid_power(self):
        t = np.arange(4000) * 1e-3
        x = np.sin(2 * np.pi * 10 * t)
        assert abs(float(band_power(x, 1e-3, (5.0, 15.0), nperseg=500)) - 0.5) < 1e-2
        assert abs(float(band_power(x, 1e-3, (5.0, 15.0), nperseg=500, relative=True)) - 1.0) < 1e-3

    def test_units(self):
        x = _noise((300,), 31)
        bp = band_power(x * u.mV, 2 * u.ms, (10.0, 50.0), nperseg=100)
        assert bp.unit == u.mV ** 2
        np.testing.assert_allclose(bp.to_decimal(u.mV ** 2), band_power(x, 2e-3, (10.0, 50.0), nperseg=100), rtol=1e-6)
        rel = band_power(x * u.mV, 2 * u.ms, (10.0, 50.0), nperseg=100, relative=True)
        assert not isinstance(rel, u.Quantity)

    def test_zero_signal_relative_is_nan(self):
        assert np.isnan(float(band_power(np.zeros(300), DT, (10.0, 50.0), nperseg=100, relative=True)))

    def test_jit_vmap_grad(self, x64):
        x = jnp.asarray(_noise((3, 200), 32))
        fn = lambda s: band_power(s, DT, (10.0, 80.0), nperseg=64)
        ref = np.stack([np.asarray(fn(s)) for s in x])
        np.testing.assert_allclose(jax.jit(fn)(x[0]), ref[0], rtol=1e-12)
        np.testing.assert_allclose(jax.vmap(fn)(x), ref, rtol=1e-12)
        check_grads(fn, (x[0],), order=1, modes=['rev'])

    def test_errors(self):
        x = np.zeros(300)
        with pytest.raises(ValueError, match='Nyquist'):
            band_power(x, DT, (10.0, 250.0))
        with pytest.raises(ValueError, match='band'):
            band_power(x, DT, (0.0, 30.0))
        with pytest.raises(ValueError, match='Fewer than two frequency bins'):
            band_power(x, DT, (10.0, 11.0), nperseg=64)
        with pytest.raises(ValueError, match='fewer than nperseg'):
            band_power(x, DT, (10.0, 30.0), nperseg=400)


# ---------------------------------------------------------------------------
# plv / pli / wpli
# ---------------------------------------------------------------------------

def _np_phase_measures(x, y, band=None):
    if band is not None:
        sos = ss.butter(4, band, btype='band', fs=FS, output='sos')
        x = ss.sosfiltfilt(sos, x, axis=0)
        y = ss.sosfiltfilt(sos, y, axis=0)
    zx = ss.hilbert(x, axis=0)
    zy = ss.hilbert(y, axis=0)
    s = zx * np.conj(zy)
    plv_ = np.abs(np.mean(np.exp(1j * (np.angle(zx) - np.angle(zy))), axis=0))
    pli_ = np.abs(np.mean(np.sign(np.imag(s)), axis=0))
    wpli_ = np.abs(np.mean(np.imag(s), axis=0)) / np.mean(np.abs(np.imag(s)), axis=0)
    return plv_, pli_, wpli_


class TestPhaseSynchrony:
    @pytest.mark.parametrize('band', [None, (8.0, 30.0)])
    def test_numpy_reference(self, x64, band):
        x = _noise((301, 4), 33)
        y = 0.6 * np.roll(x, 3, axis=0) + 0.8 * _noise((301, 4), 34)
        ref = _np_phase_measures(x, y, band)
        for fn, r in zip((plv, pli, wpli), ref):
            np.testing.assert_allclose(fn(x, y, DT, band), r, rtol=1e-9, atol=1e-12)

    def test_identical_signals(self):
        x = _noise((200, 3), 35)
        np.testing.assert_allclose(plv(x, x), 1.0, atol=1e-6)
        np.testing.assert_array_equal(pli(x, x), 0.0)
        assert np.all(np.isnan(np.asarray(wpli(x, x))))  # undefined: Im S == 0

    @pytest.mark.parametrize('lag', [np.pi / 4, -np.pi / 3])
    def test_fixed_phase_lag(self, lag):
        t = np.arange(1000) * 1e-3
        x = np.cos(2 * np.pi * 10 * t)
        y = np.cos(2 * np.pi * 10 * t - lag)
        assert abs(float(plv(x, y)) - 1.0) < 1e-5
        assert float(pli(x, y)) == 1.0
        assert abs(float(wpli(x, y)) - 1.0) < 1e-5

    def test_band_option_rejects_bad_input(self):
        x = _noise((200,), 36)
        with pytest.raises(ValueError, match='dt'):
            plv(x, x, band=(8.0, 30.0))
        with pytest.raises(ValueError, match='Nyquist'):
            pli(x, x, DT, band=(8.0, 250.0))
        with pytest.raises(ValueError, match='same shape'):
            wpli(x, x[:-1])

    def test_units(self):
        x = _noise((200,), 37)
        y = _noise((200,), 38)
        for fn in (plv, pli, wpli):
            np.testing.assert_allclose(fn(x * u.mV, y * u.volt, 2 * u.ms, (10.0, 40.0)),
                                       fn(x, y, 2e-3, (10.0, 40.0)), rtol=1e-4, atol=1e-6)

    def test_jit_and_vmap(self, x64):
        x = jnp.asarray(_noise((3, 150), 39))
        y = jnp.asarray(_noise((3, 150), 40))
        for fn in (plv, pli, wpli):
            g = lambda a, b: fn(a, b, DT, (10.0, 60.0))
            ref = np.stack([np.asarray(g(a, b)) for a, b in zip(x, y)])
            np.testing.assert_allclose(jax.jit(g)(x[0], y[0]), ref[0], rtol=1e-10)
            np.testing.assert_allclose(jax.vmap(g)(x, y), ref, rtol=1e-10)

    def test_grad(self, x64):
        x = jnp.asarray(_noise((120, 2), 41))
        y = jnp.asarray(_noise((120, 2), 42))
        check_grads(lambda a, b: jnp.sum(plv(a, b)), (x, y), order=1, modes=['rev'], atol=1e-4, rtol=1e-4)
        check_grads(lambda a, b: jnp.sum(wpli(a, b)), (x, y), order=1, modes=['rev'], atol=1e-4, rtol=1e-4)
        # PLI is piecewise constant: its gradient is zero almost everywhere.
        g = jax.grad(lambda a: jnp.sum(pli(a, y)))(x)
        np.testing.assert_array_equal(g, 0.0)
