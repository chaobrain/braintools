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

# -*- coding: utf-8 -*-

"""JAX-native spectral, filtering and phase-synchrony analysis.

Every function takes the signal with **time on axis 0** (``(T, ...)``) and a
sampling interval ``dt`` that is either a plain number in **seconds** or a
:class:`brainunit.Quantity`. Frequencies are returned in Hz as plain arrays.

Everything that determines an output shape (``nperseg``, ``noverlap``,
``freqs``, ``band``, ``order``, ``nw``, ``n_tapers``, ``dt``) is a static Python
value. Filter coefficients (Butterworth SOS), DPSS tapers, windows and Morlet
wavelets are designed on the host with NumPy/SciPy from those static values;
all signal processing runs in JAX, so the functions work under ``jax.jit`` and
``jax.vmap`` and are differentiable with respect to the signal.
"""

from typing import Optional, Sequence, Tuple, Union

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as onp
import scipy.fft
import scipy.signal

from braintools._misc import set_module_as

__all__ = [
    'spectrogram',
    'multitaper_psd',
    'morlet_transform',
    'bandpass',
    'analytic_signal',
    'band_power',
    'plv',
    'pli',
    'wpli',
]

Dt = Union[float, u.Quantity]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dt_seconds(dt: Dt) -> float:
    """Return the (static, positive) sampling interval in seconds."""
    if dt is None:
        raise ValueError('`dt` (the sampling interval) is required.')
    if isinstance(dt, u.Quantity):
        value = float(dt.to_decimal(u.second))
    else:
        value = float(dt)
    if not onp.isfinite(value) or value <= 0:
        raise ValueError(f'`dt` must be a positive finite sampling interval, got {dt}.')
    return value


def _split(x) -> Tuple[jax.Array, Optional[u.Unit]]:
    """Split ``x`` into a floating JAX magnitude and its unit (``None`` if plain)."""
    unit = None
    if isinstance(x, u.Quantity):
        unit = x.unit
        x = x.mantissa
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.result_type(float))
    return x, unit


def _wrap(mag: jax.Array, unit) -> Union[jax.Array, u.Quantity]:
    """Re-attach ``unit`` to ``mag`` (plain array if ``unit`` is ``None``/unitless)."""
    if unit is None or unit.is_unitless:
        return mag
    return u.Quantity(mag, unit=unit)


def _real_dtype(x: jax.Array):
    return jnp.finfo(x.dtype).dtype


def _check_time_axis(x: jax.Array, name: str = 'x') -> int:
    if x.ndim < 1:
        raise ValueError(f'`{name}` must have a time axis 0; got a scalar.')
    return x.shape[0]


def _check_band(band, fs: float) -> Tuple[float, float]:
    """Validate ``band = (low, high)`` in Hz: ``0 < low < high < fs / 2``."""
    try:
        low, high = (float(b) for b in band)
    except (TypeError, ValueError):
        raise ValueError(f'`band` must be a pair (low, high) in Hz, got {band!r}.') from None
    nyq = fs / 2.0
    if not (0.0 < low < high):
        raise ValueError(f'`band` must satisfy 0 < low < high, got ({low}, {high}).')
    if high >= nyq:
        raise ValueError(
            f'`band` upper edge {high} Hz must be below the Nyquist frequency {nyq} Hz.'
        )
    return low, high


def _bcast(v, ndim: int, axis: int = 0):
    """Reshape the 1-D ``v`` so that it broadcasts along ``axis`` of an ``ndim`` array."""
    shape = [1] * ndim
    shape[axis] = -1
    return jnp.reshape(v, shape)


def _get_window(window, nperseg: int) -> onp.ndarray:
    if isinstance(window, (str, tuple)):
        return onp.asarray(scipy.signal.get_window(window, nperseg), dtype=onp.float64)
    win = onp.asarray(window, dtype=onp.float64)
    if win.ndim != 1 or win.shape[0] != nperseg:
        raise ValueError(f'`window` array must have shape ({nperseg},), got {win.shape}.')
    return win


def _stft_density(x, fs, nperseg, noverlap, window, detrend):
    """One-sided periodogram density of each segment: returns ``(freqs, starts, P (S, F, ...))``.

    Mirrors ``scipy.signal._spectral_helper`` with ``scaling='density'``,
    ``mode='psd'``, ``nfft=nperseg``, ``boundary=None`` and ``padded=False``.
    """
    n_time = _check_time_axis(x)
    nperseg = int(nperseg)
    if nperseg < 1:
        raise ValueError(f'`nperseg` must be a positive integer, got {nperseg}.')
    if n_time < nperseg:
        raise ValueError(
            f'The signal has {n_time} samples along axis 0, fewer than nperseg={nperseg}.'
        )
    noverlap = int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError(f'`noverlap` must satisfy 0 <= noverlap < nperseg, got {noverlap}.')
    if detrend not in ('constant', False, None):
        raise ValueError(f"`detrend` must be 'constant', False or None, got {detrend!r}.")
    win = _get_window(window, nperseg)

    step = nperseg - noverlap
    starts = onp.arange(0, n_time - nperseg + 1, step)
    idx = starts[:, None] + onp.arange(nperseg)[None, :]  # (S, nperseg), static
    segs = x[idx]  # (S, nperseg, ...)
    if detrend == 'constant':
        segs = segs - jnp.mean(segs, axis=1, keepdims=True)
    dtype = _real_dtype(x)
    segs = segs * _bcast(jnp.asarray(win, dtype=dtype), segs.ndim, axis=1)
    spec = jnp.fft.rfft(segs, axis=1)  # (S, F, ...)
    scale = 1.0 / (fs * float(onp.sum(win ** 2)))
    factor = onp.full(nperseg // 2 + 1, 2.0)
    factor[0] = 1.0
    if nperseg % 2 == 0:
        factor[-1] = 1.0
    factor = _bcast(jnp.asarray(factor * scale, dtype=dtype), spec.ndim, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2) * factor
    freqs = onp.fft.rfftfreq(nperseg, 1.0 / fs)
    return freqs, starts, power


def _sosfilt(sos: onp.ndarray, x: jax.Array, zi: jax.Array) -> jax.Array:
    """Cascade of direct-form-II-transposed biquads, run as one ``lax.scan`` over time.

    ``sos`` is a static ``(n_sections, 6)`` host array, ``x`` has time on axis 0
    and ``zi`` has shape ``(n_sections, 2, *x.shape[1:])``.
    """
    coefs = [tuple(float(c) / float(row[3]) for c in row) for row in sos]

    def step(z, xt):
        v = xt
        new = []
        for s, (b0, b1, b2, _, a1, a2) in enumerate(coefs):
            y = b0 * v + z[s, 0]
            new.append(jnp.stack([b1 * v - a1 * y + z[s, 1], b2 * v - a2 * y]))
            v = y
        return jnp.stack(new), v

    _, y = jax.lax.scan(step, zi, x)
    return y


def _cross_spectrum(x, y, dt, band):
    """Analytic cross-spectrum ``z_x conj(z_y)`` along axis 0 (optionally band-passed)."""
    xm, _ = _split(x)
    ym, _ = _split(y)
    if xm.shape != ym.shape:
        raise ValueError(f'`x` and `y` must have the same shape, got {xm.shape} and {ym.shape}.')
    _check_time_axis(xm)
    if band is not None:
        if dt is None:
            raise ValueError('`dt` is required when `band` is given.')
        xm = bandpass(xm, dt, band)
        ym = bandpass(ym, dt, band)
    zx = analytic_signal(xm)
    zy = analytic_signal(ym)
    return zx, zy


def _imag_cross(zx, zy):
    """``Im(z_x conj(z_y)) = |z_x||z_y| sin(phi_x - phi_y)`` in polar form.

    The polar form is exactly zero for identical signals, whereas the
    Cartesian product can leave an FMA rounding residue of either sign.
    """
    return jnp.abs(zx) * jnp.abs(zy) * jnp.sin(jnp.angle(zx) - jnp.angle(zy))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@set_module_as('braintools.analysis.spectral')
def spectrogram(
    x: brainstate.typing.ArrayLike,
    dt: Dt,
    nperseg: int,
    noverlap: Optional[int] = None,
    window: Union[str, tuple, brainstate.typing.ArrayLike] = 'hann',
    detrend: Union[str, bool, None] = 'constant',
) -> Tuple[jax.Array, Union[jax.Array, u.Quantity], Union[jax.Array, u.Quantity]]:
    r"""Compute the short-time Fourier power spectrogram along time axis 0.

    The signal is cut into segments of ``nperseg`` samples that overlap by
    ``noverlap`` samples. Each segment is mean-detrended, multiplied by the
    window and Fourier transformed. The one-sided power spectral density of
    every segment is returned, scaled as in ``scipy.signal.spectrogram`` with
    ``scaling='density'`` and ``mode='psd'``.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        Real signal of shape ``(T, ...)``; time is axis 0. A
        :class:`brainunit.Quantity` is accepted.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a plain number). Static.
    nperseg : int
        Segment length in samples. Static. Must not exceed ``T``.
    noverlap : int, optional
        Overlap between segments in samples, ``0 <= noverlap < nperseg``.
        Static. Defaults to ``nperseg // 8`` (the SciPy default).
    window : str, tuple or array_like, default='hann'
        Window specification passed to ``scipy.signal.get_window`` (periodic
        windows, as in SciPy), or an explicit window of length ``nperseg``.
    detrend : {'constant', False, None}, default='constant'
        ``'constant'`` removes the mean of every segment (the SciPy default);
        ``False``/``None`` disables detrending.

    Returns
    -------
    freqs : jax.Array
        Frequencies in Hz, shape ``(F,)`` with ``F = nperseg // 2 + 1``.
    times : jax.Array or brainunit.Quantity
        Segment centres, shape ``(S,)`` with
        ``S = (T - noverlap) // (nperseg - noverlap)``. Seconds as a plain array,
        or a Quantity in seconds if ``dt`` is a Quantity.
    Sxx : jax.Array or brainunit.Quantity
        Power spectral density, shape ``(F, S, ...)``. If ``x`` is a Quantity
        with unit ``U``, ``Sxx`` has unit ``U**2 * second`` (i.e. ``U**2/Hz``).

    Raises
    ------
    ValueError
        If ``T < nperseg``, ``noverlap`` is out of range, ``detrend`` is
        unknown or ``dt`` is not positive.

    See Also
    --------
    band_power : Welch PSD integrated over a frequency band.
    multitaper_psd : Multitaper power spectral density.
    morlet_transform : Complex Morlet wavelet time-frequency transform.

    Notes
    -----
    The result equals ``scipy.signal.spectrogram(x, fs, window, nperseg,
    noverlap, axis=0)`` (shape ``(F, ..., S)``) with the segment axis moved to
    position 1. Unlike SciPy, a signal shorter than ``nperseg`` raises instead
    of silently shrinking ``nperseg``.

    The function is jittable and vmappable (all shape-determining arguments are
    static) and differentiable with respect to ``x``.

    References
    ----------
    .. [1] Welch, P. "The use of fast Fourier transform for the estimation of
           power spectra: a method based on time averaging over short, modified
           periodograms." IEEE Transactions on Audio and Electroacoustics 15.2
           (1967): 70-73.

    Examples
    --------
    The spectrogram of a 50 Hz sinusoid peaks at 50 Hz in every segment
    (periodogram scaling as in Welch [1]_):

    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import spectrogram
        >>> t = jnp.arange(1000) * 1e-3
        >>> x = jnp.sin(2 * jnp.pi * 50 * t)
        >>> freqs, times, Sxx = spectrogram(x, dt=1 * u.ms, nperseg=200, noverlap=100)
        >>> Sxx.shape
        (101, 9)
        >>> float(freqs[jnp.argmax(Sxx[:, 0])])
        50.0
        >>> print(times[:2])
        [0.1 0.2] s
    """
    xm, unit = _split(x)
    fs = 1.0 / _dt_seconds(dt)
    nperseg = int(nperseg)
    if noverlap is None:
        noverlap = nperseg // 8
    freqs, starts, power = _stft_density(xm, fs, nperseg, noverlap, window, detrend)
    sxx = jnp.moveaxis(power, 0, 1)  # (F, S, ...)
    times = jnp.asarray((starts + nperseg / 2.0) / fs, dtype=_real_dtype(xm))
    if isinstance(dt, u.Quantity):
        times = u.Quantity(times, unit=u.second)
    out_unit = None if unit is None else unit ** 2 * u.second
    return jnp.asarray(freqs, dtype=_real_dtype(xm)), times, _wrap(sxx, out_unit)


@set_module_as('braintools.analysis.spectral')
def multitaper_psd(
    x: brainstate.typing.ArrayLike,
    dt: Dt,
    nw: float = 4.0,
    n_tapers: Optional[int] = None,
    *,
    remove_dc: bool = True,
) -> Tuple[jax.Array, Union[jax.Array, u.Quantity]]:
    r"""Estimate the power spectral density with DPSS (Slepian) multitapers.

    The signal is multiplied by ``n_tapers`` discrete prolate spheroidal
    sequences [1]_ with time-half-bandwidth product ``nw``. The one-sided
    periodograms of the tapered signals are averaged with weights equal to the
    taper concentration ratios ``lambda_k`` (non-adaptive weighting, [2]_):

    .. math::

        S(f) = \frac{2}{f_s} \frac{\sum_k \lambda_k |X_k(f)|^2}{\sum_k \lambda_k},

    where the factor 2 is not applied at DC and (even ``T``) Nyquist.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        Real signal of shape ``(T, ...)``; time is axis 0. A
        :class:`brainunit.Quantity` is accepted.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a plain number). Static.
    nw : float, default=4.0
        Time-half-bandwidth product ``NW``; the frequency resolution
        (full bandwidth) is ``2 * nw / (T * dt)`` Hz. Static,
        ``0.5 <= nw < T / 2``.
    n_tapers : int, optional
        Number of tapers ``K``, ``1 <= K <= T``. Static. Defaults to
        ``int(2 * nw) - 1`` (the tapers with good spectral concentration).
    remove_dc : bool, default=True
        Subtract the mean of each channel before tapering (as MNE does).

    Returns
    -------
    freqs : jax.Array
        Frequencies in Hz, shape ``(T // 2 + 1,)``.
    psd : jax.Array or brainunit.Quantity
        Power spectral density, shape ``(T // 2 + 1, ...)``. If ``x`` has unit
        ``U``, the result has unit ``U**2 * second`` (``U**2/Hz``).

    Raises
    ------
    ValueError
        If ``nw`` or ``n_tapers`` is out of range, or ``dt`` is not positive.

    See Also
    --------
    spectrogram : Short-time Fourier power spectrogram.
    band_power : Welch PSD integrated over a frequency band.

    Notes
    -----
    The tapers are periodic (``sym=False``) L2-normalised DPSS windows from
    ``scipy.signal.windows.dpss``, designed on the host. The result equals

    .. code-block:: python

        mne.time_frequency.psd_array_multitaper(
            x.T, sfreq=fs, bandwidth=2 * nw * fs / T, adaptive=False,
            low_bias=False, normalization='full', remove_dc=True)

    when ``n_tapers = int(2 * nw)``: with ``low_bias=False`` MNE keeps
    ``int(2 * nw)`` tapers, whereas the default here drops the last, poorly
    concentrated one.

    The function is jittable and vmappable and differentiable with respect to
    ``x``.

    References
    ----------
    .. [1] Slepian, D. "Prolate spheroidal wave functions, Fourier analysis, and
           uncertainty - V: the discrete case." Bell System Technical Journal
           57.5 (1978): 1371-1430.
    .. [2] Percival, D. B., and Walden, A. T. Spectral Analysis for Physical
           Applications. Cambridge University Press, 1993.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import multitaper_psd
        >>> t = jnp.arange(1000) * 1e-3
        >>> x = jnp.sin(2 * jnp.pi * 40 * t)
        >>> freqs, psd = multitaper_psd(x, dt=1e-3, nw=3.0)
        >>> psd.shape
        (501,)
        >>> float(freqs[jnp.argmax(psd)])
        40.0
    """
    xm, unit = _split(x)
    fs = 1.0 / _dt_seconds(dt)
    n_time = _check_time_axis(xm)
    nw = float(nw)
    if not 0.5 <= nw < n_time / 2.0:
        raise ValueError(f'`nw` must satisfy 0.5 <= nw < T / 2 = {n_time / 2}, got {nw}.')
    if n_tapers is None:
        n_tapers = max(1, int(2 * nw) - 1)
    n_tapers = int(n_tapers)
    if not 1 <= n_tapers <= n_time:
        raise ValueError(f'`n_tapers` must satisfy 1 <= n_tapers <= T = {n_time}, got {n_tapers}.')
    tapers, ratios = scipy.signal.windows.dpss(
        n_time, nw, n_tapers, sym=False, norm=2, return_ratios=True
    )
    tapers = onp.atleast_2d(tapers)
    weights = onp.atleast_1d(ratios) / onp.sum(ratios)

    dtype = _real_dtype(xm)
    if remove_dc:
        xm = xm - jnp.mean(xm, axis=0, keepdims=True)
    tap = jnp.asarray(tapers, dtype=dtype).reshape((n_tapers, n_time) + (1,) * (xm.ndim - 1))
    spec = jnp.fft.rfft(tap * xm[None], axis=1)  # (K, F, ...)
    power = spec.real ** 2 + spec.imag ** 2
    psd = jnp.tensordot(jnp.asarray(weights, dtype=dtype), power, axes=(0, 0))  # (F, ...)
    factor = onp.full(n_time // 2 + 1, 2.0 / fs)
    factor[0] = 1.0 / fs
    if n_time % 2 == 0:
        factor[-1] = 1.0 / fs
    psd = psd * _bcast(jnp.asarray(factor, dtype=dtype), psd.ndim)
    freqs = jnp.asarray(onp.fft.rfftfreq(n_time, 1.0 / fs), dtype=dtype)
    out_unit = None if unit is None else unit ** 2 * u.second
    return freqs, _wrap(psd, out_unit)


@set_module_as('braintools.analysis.spectral')
def morlet_transform(
    x: brainstate.typing.ArrayLike,
    dt: Dt,
    freqs: Sequence[float],
    n_cycles: Union[float, Sequence[float]] = 7.0,
    *,
    zero_mean: bool = True,
) -> Union[jax.Array, u.Quantity]:
    r"""Compute the complex Morlet wavelet transform along time axis 0.

    For each frequency :math:`f` with :math:`\sigma_t = n_{cycles} / (2\pi f)`,
    the wavelet [1]_

    .. math::

        W_f(t) = c \, \left(e^{2\pi i f t} - \kappa\right) e^{-t^2 / (2\sigma_t^2)},
        \qquad |t| < 5\sigma_t,

    is sampled at ``dt`` (with a sample at exactly ``t = 0``), where
    :math:`\kappa = e^{-2(\pi f \sigma_t)^2}` if ``zero_mean`` else 0, and
    :math:`c` makes :math:`\|W_f\|_2 = \sqrt{2}` (MNE's normalisation). The
    signal is convolved with each wavelet (``'same'`` mode, the output aligned
    with the input samples, zero padding outside the signal) via the FFT.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        Real signal of shape ``(T, ...)``; time is axis 0. A
        :class:`brainunit.Quantity` is accepted.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a plain number). Static.
    freqs : sequence of float
        Wavelet frequencies in Hz, ``0 < f <= fs / 2``. Static.
    n_cycles : float or sequence of float, default=7.0
        Number of cycles of each wavelet (one value, or one per frequency).
        Static.
    zero_mean : bool, default=True
        Subtract the Gaussian-weighted offset so each wavelet has zero mean
        (the default of ``mne.time_frequency.tfr_array_morlet``).

    Returns
    -------
    tfr : jax.Array or brainunit.Quantity
        Complex coefficients of shape ``(n_freqs, T, ...)``; ``abs(tfr)**2`` is
        the power and ``angle(tfr)`` the phase. Carries the unit of ``x``.

    Raises
    ------
    ValueError
        If a frequency is not in ``(0, fs / 2]``, ``n_cycles`` has the wrong
        length or is not positive, or the longest wavelet is longer than ``T``
        (as in MNE).

    See Also
    --------
    spectrogram : Short-time Fourier power spectrogram.
    analytic_signal : Hilbert-transform analytic signal.

    Notes
    -----
    The result equals
    ``mne.time_frequency.tfr_array_morlet(x.T[None], sfreq=fs, freqs=freqs,
    n_cycles=n_cycles, zero_mean=zero_mean, output='complex')[0]`` with the
    channel axis moved behind the time axis. The wavelets are designed on the
    host and their FFTs (length ``next_fast_len(T + L_max - 1)``, which avoids
    circular wrap-around) are constants; the convolution runs in JAX.

    The function is jittable and vmappable and differentiable with respect to
    ``x``.

    References
    ----------
    .. [1] Tallon-Baudry, C., Bertrand, O., Delpuech, C., and Pernier, J.
           "Stimulus specificity of phase-locked and non-phase-locked 40 Hz
           visual responses in human." Journal of Neuroscience 16.13 (1996):
           4240-4249.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import morlet_transform
        >>> t = jnp.arange(2000) * 1e-3
        >>> x = jnp.sin(2 * jnp.pi * 20 * t)
        >>> tfr = morlet_transform(x, dt=1e-3, freqs=[10.0, 20.0, 40.0])
        >>> tfr.shape
        (3, 2000)
        >>> power = jnp.mean(jnp.abs(tfr[:, 500:1500]) ** 2, axis=1)
        >>> int(jnp.argmax(power))
        1
    """
    xm, unit = _split(x)
    dt_s = _dt_seconds(dt)
    fs = 1.0 / dt_s
    n_time = _check_time_axis(xm)
    freqs_np: onp.ndarray = onp.atleast_1d(onp.asarray(freqs, dtype=onp.float64))
    if freqs_np.ndim != 1 or freqs_np.size == 0:
        raise ValueError('`freqs` must be a non-empty 1-D sequence of frequencies.')
    if onp.any(freqs_np <= 0) or onp.any(freqs_np > fs / 2.0):
        raise ValueError(
            f'All `freqs` must lie in (0, fs/2] = (0, {fs / 2}] Hz, got {freqs_np.tolist()}.'
        )
    cycles: onp.ndarray = onp.atleast_1d(onp.asarray(n_cycles, dtype=onp.float64))
    if cycles.size == 1:
        cycles = onp.full(freqs_np.shape, cycles[0])
    if cycles.shape != freqs_np.shape:
        raise ValueError('`n_cycles` must be a scalar or have one value per frequency.')
    if onp.any(cycles <= 0):
        raise ValueError('`n_cycles` must be positive.')

    wavelets = []
    for f, nc in zip(freqs_np, cycles):  # host-side design of static constants
        sigma_t = nc / (2.0 * onp.pi * f)
        t = onp.arange(0.0, 5.0 * sigma_t, 1.0 / fs)
        t = onp.r_[-t[::-1], t[1:]]
        osc = onp.exp(2.0j * onp.pi * f * t)
        if zero_mean:
            osc = osc - onp.exp(-2.0 * (onp.pi * f * sigma_t) ** 2)
        w = osc * onp.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
        wavelets.append(w / (onp.sqrt(0.5) * onp.linalg.norm(w)))
    max_len = max(w.size for w in wavelets)
    if max_len > n_time:
        raise ValueError(
            f'At least one of the wavelets is longer than the signal ({max_len} > '
            f'{n_time} samples). Use a longer signal or fewer cycles.'
        )
    nfft = scipy.fft.next_fast_len(n_time + max_len - 1)
    kernel: onp.ndarray = onp.zeros((freqs_np.size, nfft), dtype=onp.complex128)
    for i, w in enumerate(wavelets):
        half = (w.size - 1) // 2  # w.size is odd; centre sample is t = 0
        kernel[i, :half + 1] = w[half:]
        kernel[i, nfft - half:] = w[:half]
    kernel_fft = onp.fft.fft(kernel, axis=1)

    cdtype = jnp.result_type(xm.dtype, jnp.complex64)
    xf = jnp.fft.fft(xm, n=nfft, axis=0)  # (nfft, ...)
    kf = jnp.asarray(kernel_fft, dtype=cdtype).reshape(
        (freqs_np.size, nfft) + (1,) * (xm.ndim - 1)
    )
    out = jnp.fft.ifft(kf * xf[None], axis=1)[:, :n_time]
    return _wrap(out, unit)


@set_module_as('braintools.analysis.spectral')
def bandpass(
    x: brainstate.typing.ArrayLike,
    dt: Dt,
    band: Tuple[float, float],
    order: int = 4,
) -> Union[jax.Array, u.Quantity]:
    r"""Zero-phase Butterworth band-pass filter along time axis 0.

    A Butterworth band-pass of order ``order`` (an IIR filter of order
    ``2 * order``) is designed on the host as second-order sections with
    ``scipy.signal.butter(order, band, btype='band', fs=fs, output='sos')``.
    The signal is filtered forward and backward [1]_, which squares the
    magnitude response and cancels the phase, exactly as
    ``scipy.signal.sosfiltfilt`` with its defaults:

    - odd extension of ``padlen = 3 * (2 * n_sections + 1 - n_zero)`` samples
      at both ends, where ``n_zero`` counts sections with a trivial last
      coefficient;
    - initial states ``sosfilt_zi(sos) * x[0]`` for the forward pass and
      ``sosfilt_zi(sos) * y[-1]`` for the backward pass.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        Real signal of shape ``(T, ...)``; time is axis 0. A
        :class:`brainunit.Quantity` is accepted.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a plain number). Static.
    band : tuple of float
        Pass band ``(low, high)`` in Hz with ``0 < low < high < fs / 2``.
        Static.
    order : int, default=4
        Butterworth order (number of second-order sections). Static.

    Returns
    -------
    y : jax.Array or brainunit.Quantity
        Filtered signal with the shape, dtype and unit of ``x``.

    Raises
    ------
    ValueError
        If a band edge is ``<= 0`` or ``>= fs / 2``, ``order < 1``, or
        ``T <= padlen`` (the signal is too short for the edge padding, the
        same condition under which SciPy raises).

    See Also
    --------
    analytic_signal : Hilbert-transform analytic signal.
    plv : Phase-locking value, optionally after band-passing.

    Notes
    -----
    Both passes run as a single ``jax.lax.scan`` over time that cascades all
    sections at each step (direct form II transposed), so the cost is
    sequential in ``T``. The function is jittable, vmappable and (being linear)
    differentiable with respect to ``x``. In float32, low cut-offs relative to
    ``fs`` lose precision; use float64 for very narrow or very low bands.

    References
    ----------
    .. [1] Gustafsson, F. "Determining the initial states in forward-backward
           filtering." IEEE Transactions on Signal Processing 44.4 (1996):
           988-992.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import bandpass
        >>> t = jnp.arange(2000) * 1e-3
        >>> x = jnp.sin(2 * jnp.pi * 10 * t) + jnp.sin(2 * jnp.pi * 100 * t)
        >>> y = bandpass(x, dt=1e-3, band=(5.0, 20.0))
        >>> y.shape
        (2000,)
        >>> bool(jnp.max(jnp.abs(y[500:1500] - jnp.sin(2 * jnp.pi * 10 * t[500:1500]))) < 0.05)
        True
    """
    xm, unit = _split(x)
    fs = 1.0 / _dt_seconds(dt)
    n_time = _check_time_axis(xm)
    low, high = _check_band(band, fs)
    order = int(order)
    if order < 1:
        raise ValueError(f'`order` must be a positive integer, got {order}.')
    sos = scipy.signal.butter(order, (low, high), btype='band', fs=fs, output='sos')
    n_sections = sos.shape[0]
    ntaps = 2 * n_sections + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    padlen = 3 * ntaps
    if n_time <= padlen:
        raise ValueError(
            f'The signal has {n_time} samples along axis 0; bandpass needs more than '
            f'padlen={padlen} samples for order={order}.'
        )
    zi = scipy.signal.sosfilt_zi(sos)  # (n_sections, 2)
    dtype = _real_dtype(xm)
    zi = jnp.asarray(zi, dtype=dtype).reshape((n_sections, 2) + (1,) * (xm.ndim - 1))

    # odd extension (scipy.signal._arraytools.odd_ext)
    left = 2 * xm[:1] - xm[padlen:0:-1]
    right = 2 * xm[-1:] - xm[-2:-(padlen + 2):-1]
    ext = jnp.concatenate([left, xm, right], axis=0)

    y = _sosfilt(sos, ext, zi * ext[0][None, None])
    y = _sosfilt(sos, y[::-1], zi * y[-1][None, None])[::-1]
    return _wrap(y[padlen:-padlen], unit)


@set_module_as('braintools.analysis.spectral')
def analytic_signal(
    x: brainstate.typing.ArrayLike,
    axis: int = 0,
) -> Union[jax.Array, u.Quantity]:
    r"""Compute the analytic signal by the FFT Hilbert transform.

    The analytic signal is :math:`x_a = x + i\,\mathcal{H}[x]` [1]_. It is
    obtained by zeroing the negative-frequency half of the spectrum and
    doubling the positive half (DC and, for even length, the Nyquist bin are
    kept as is), exactly as ``scipy.signal.hilbert``.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        Real signal. A :class:`brainunit.Quantity` is accepted.
    axis : int, default=0
        Axis along which to transform (time). Static.

    Returns
    -------
    xa : jax.Array or brainunit.Quantity
        Complex analytic signal of the same shape (and unit) as ``x``;
        ``abs(xa)`` is the amplitude envelope and ``angle(xa)`` the
        instantaneous phase.

    Raises
    ------
    ValueError
        If ``x`` is complex or the transform axis has length 0.

    See Also
    --------
    bandpass : Zero-phase Butterworth band-pass filter.
    plv : Phase-locking value between two signals.

    Notes
    -----
    The function is jittable, vmappable and (being linear) differentiable with
    respect to ``x``.

    References
    ----------
    .. [1] Marple, S. L. "Computing the discrete-time analytic signal via FFT."
           IEEE Transactions on Signal Processing 47.9 (1999): 2600-2603.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import analytic_signal
        >>> t = jnp.arange(1000) * 1e-3
        >>> z = analytic_signal(jnp.cos(2 * jnp.pi * 10 * t))
        >>> bool(jnp.allclose(jnp.abs(z), 1.0, atol=1e-4))
        True
    """
    xm, unit = _split(x)
    if jnp.iscomplexobj(xm):
        raise ValueError('`x` must be real.')
    if xm.ndim == 0:
        raise ValueError('`x` must have at least one dimension.')
    axis = int(axis) % xm.ndim
    n = xm.shape[axis]
    if n == 0:
        raise ValueError('The transform axis of `x` has length 0.')
    h = onp.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    spec = jnp.fft.fft(xm, axis=axis)
    out = jnp.fft.ifft(spec * _bcast(jnp.asarray(h, dtype=_real_dtype(xm)), xm.ndim, axis), axis=axis)
    return _wrap(out, unit)


@set_module_as('braintools.analysis.spectral')
def band_power(
    x: brainstate.typing.ArrayLike,
    dt: Dt,
    band: Tuple[float, float],
    nperseg: Optional[int] = None,
    relative: bool = False,
) -> Union[jax.Array, u.Quantity]:
    r"""Integrate the Welch power spectral density over a frequency band.

    The PSD is estimated with Welch's method [1]_ exactly as
    ``scipy.signal.welch`` with its defaults (periodic Hann window, 50 %
    overlap, mean detrending, density scaling, mean average), and integrated
    with the trapezoid rule over the frequency bins ``low <= f <= high``.

    Parameters
    ----------
    x : brainstate.typing.ArrayLike
        Real signal of shape ``(T, ...)``; time is axis 0. A
        :class:`brainunit.Quantity` is accepted.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a plain number). Static.
    band : tuple of float
        ``(low, high)`` in Hz with ``0 < low < high < fs / 2``. Static.
    nperseg : int, optional
        Welch segment length. Static. Defaults to ``min(256, T)`` (as SciPy).
    relative : bool, default=False
        If True, divide by the power integrated over all frequencies.

    Returns
    -------
    power : jax.Array or brainunit.Quantity
        Band power of shape ``x.shape[1:]``. Absolute power has unit ``U**2``
        when ``x`` has unit ``U``; relative power is dimensionless and ``nan``
        for an all-constant signal (zero total power).

    Raises
    ------
    ValueError
        If a band edge is ``<= 0`` or ``>= fs / 2``, ``T < nperseg``, or fewer
        than two frequency bins fall inside ``band`` (increase ``nperseg``).

    See Also
    --------
    spectrogram : Short-time Fourier power spectrogram.
    multitaper_psd : Multitaper power spectral density.
    braintools.analysis.spectral.power_spectral_density : Welch PSD.

    Notes
    -----
    Equivalent to ``f, p = scipy.signal.welch(x, fs, nperseg=nperseg, axis=0)``
    followed by ``scipy.integrate.trapezoid(p[m], f[m], axis=0)`` with
    ``m = (f >= low) & (f <= high)``. Jittable, vmappable and differentiable
    with respect to ``x``.

    References
    ----------
    .. [1] Welch, P. "The use of fast Fourier transform for the estimation of
           power spectra: a method based on time averaging over short, modified
           periodograms." IEEE Transactions on Audio and Electroacoustics 15.2
           (1967): 70-73.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import band_power
        >>> t = jnp.arange(4000) * 1e-3
        >>> x = jnp.sin(2 * jnp.pi * 10 * t)
        >>> round(float(band_power(x, dt=1e-3, band=(5.0, 15.0), nperseg=500)), 2)
        0.5
        >>> round(float(band_power(x, dt=1e-3, band=(5.0, 15.0), nperseg=500, relative=True)), 2)
        1.0
    """
    xm, unit = _split(x)
    fs = 1.0 / _dt_seconds(dt)
    n_time = _check_time_axis(xm)
    low, high = _check_band(band, fs)
    if nperseg is None:
        nperseg = min(256, n_time)
    nperseg = int(nperseg)
    freqs, _, power = _stft_density(xm, fs, nperseg, nperseg // 2, 'hann', 'constant')
    psd = jnp.mean(power, axis=0)  # (F, ...)
    idx = onp.nonzero((freqs >= low) & (freqs <= high))[0]
    if idx.size < 2:
        raise ValueError(
            f'Fewer than two frequency bins fall in band ({low}, {high}) Hz with '
            f'resolution {fs / nperseg} Hz; increase `nperseg`.'
        )
    dtype = _real_dtype(xm)
    bp = jnp.trapezoid(psd[idx], jnp.asarray(freqs[idx], dtype=dtype), axis=0)
    if relative:
        total = jnp.trapezoid(psd, jnp.asarray(freqs, dtype=dtype), axis=0)
        return bp / total
    return _wrap(bp, None if unit is None else unit ** 2)


@set_module_as('braintools.analysis.spectral')
def plv(
    x: brainstate.typing.ArrayLike,
    y: brainstate.typing.ArrayLike,
    dt: Optional[Dt] = None,
    band: Optional[Tuple[float, float]] = None,
) -> jax.Array:
    r"""Compute the phase-locking value between two continuous signals.

    The instantaneous phases :math:`\phi_x, \phi_y` are the angles of the
    analytic signals (:func:`analytic_signal`) and [1]_

    .. math::

        \mathrm{PLV} = \left| \frac{1}{T} \sum_{t} e^{i(\phi_x(t) - \phi_y(t))} \right|.

    Parameters
    ----------
    x, y : brainstate.typing.ArrayLike
        Real signals of identical shape ``(T, ...)``; time is axis 0.
        Quantities are accepted (phases are scale-invariant).
    dt : float or brainunit.Quantity, optional
        Sampling interval (seconds if a plain number). Required with ``band``.
    band : tuple of float, optional
        If given, both signals are first filtered with :func:`bandpass`
        (``order=4``) in ``band`` Hz. Phase is only meaningful for narrow-band
        signals, so either pass ``band`` or pre-filter the inputs.

    Returns
    -------
    plv : jax.Array
        Values in ``[0, 1]`` of shape ``x.shape[1:]``; 1 means a constant phase
        difference.

    Raises
    ------
    ValueError
        If the shapes differ, ``band`` is given without ``dt``, or ``band`` is
        invalid (see :func:`bandpass`).

    See Also
    --------
    pli : Phase-lag index (insensitive to zero-lag coupling).
    wpli : Weighted phase-lag index.
    braintools.analysis.spike.phase_locking_value : Spike-field locking. That
        function measures how consistently the *spikes* of each neuron fall at
        the same phase of an external reference oscillation of a given
        frequency (vector strength of spike phases), whereas this function
        measures phase locking between two *continuous* signals sampled at
        every time step (Lachaux et al. 1999).

    Notes
    -----
    Jittable and vmappable. Differentiable with respect to the signals wherever
    the analytic amplitudes are non-zero and ``PLV > 0``.

    References
    ----------
    .. [1] Lachaux, J.-P., Rodriguez, E., Martinerie, J., and Varela, F. J.
           "Measuring phase synchrony in brain signals." Human Brain Mapping
           8.4 (1999): 194-208.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import plv
        >>> t = jnp.arange(1000) * 1e-3
        >>> x = jnp.cos(2 * jnp.pi * 10 * t)
        >>> y = jnp.cos(2 * jnp.pi * 10 * t - jnp.pi / 3)
        >>> round(float(plv(x, y)), 4)
        1.0
    """
    zx, zy = _cross_spectrum(x, y, dt, band)
    dphi = jnp.angle(zx) - jnp.angle(zy)
    return jnp.abs(jnp.mean(jnp.exp(1j * dphi), axis=0))


@set_module_as('braintools.analysis.spectral')
def pli(
    x: brainstate.typing.ArrayLike,
    y: brainstate.typing.ArrayLike,
    dt: Optional[Dt] = None,
    band: Optional[Tuple[float, float]] = None,
) -> jax.Array:
    r"""Compute the phase-lag index between two continuous signals.

    With :math:`S(t) = z_x(t)\,\overline{z_y(t)}` the cross-spectrum of the
    analytic signals, the phase-lag index [1]_ is

    .. math::

        \mathrm{PLI} = \left| \frac{1}{T} \sum_t \operatorname{sign}
        \left(\operatorname{Im} S(t)\right) \right|
        = \left| \langle \operatorname{sign}(\sin(\phi_x - \phi_y)) \rangle \right|.

    It ignores zero-lag (and :math:`\pi`-lag) coupling, which volume conduction
    produces, so identical signals have ``PLI = 0``.

    Parameters
    ----------
    x, y : brainstate.typing.ArrayLike
        Real signals of identical shape ``(T, ...)``; time is axis 0.
    dt : float or brainunit.Quantity, optional
        Sampling interval (seconds if a plain number). Required with ``band``.
    band : tuple of float, optional
        If given, both signals are first filtered with :func:`bandpass`.

    Returns
    -------
    pli : jax.Array
        Values in ``[0, 1]`` of shape ``x.shape[1:]``.

    Raises
    ------
    ValueError
        If the shapes differ, ``band`` is given without ``dt``, or ``band`` is
        invalid.

    See Also
    --------
    plv : Phase-locking value.
    wpli : Weighted phase-lag index.

    Notes
    -----
    Jittable and vmappable. Not differentiable: the sign makes it piecewise
    constant, so its gradient is zero almost everywhere.

    References
    ----------
    .. [1] Stam, C. J., Nolte, G., and Daffertshofer, A. "Phase lag index:
           assessment of functional connectivity from multi channel EEG and MEG
           with diminished bias from common sources." Human Brain Mapping 28.11
           (2007): 1178-1193.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import pli
        >>> t = jnp.arange(1000) * 1e-3
        >>> x = jnp.cos(2 * jnp.pi * 10 * t)
        >>> float(pli(x, x))
        0.0
        >>> float(pli(x, jnp.cos(2 * jnp.pi * 10 * t - jnp.pi / 4)))
        1.0
    """
    zx, zy = _cross_spectrum(x, y, dt, band)
    dphi = jnp.angle(zx) - jnp.angle(zy)
    return jnp.abs(jnp.mean(jnp.sign(jnp.sin(dphi)), axis=0))


@set_module_as('braintools.analysis.spectral')
def wpli(
    x: brainstate.typing.ArrayLike,
    y: brainstate.typing.ArrayLike,
    dt: Optional[Dt] = None,
    band: Optional[Tuple[float, float]] = None,
) -> jax.Array:
    r"""Compute the weighted phase-lag index between two continuous signals.

    With :math:`S(t) = z_x(t)\,\overline{z_y(t)}` the cross-spectrum of the
    analytic signals, the weighted phase-lag index [1]_ is

    .. math::

        \mathrm{wPLI} = \frac{\left| \langle \operatorname{Im} S(t) \rangle \right|}
                             {\langle \left| \operatorname{Im} S(t) \right| \rangle},

    i.e. the PLI with each sample weighted by the magnitude of the imaginary
    cross-spectrum, which down-weights samples near zero lag.

    Parameters
    ----------
    x, y : brainstate.typing.ArrayLike
        Real signals of identical shape ``(T, ...)``; time is axis 0.
    dt : float or brainunit.Quantity, optional
        Sampling interval (seconds if a plain number). Required with ``band``.
    band : tuple of float, optional
        If given, both signals are first filtered with :func:`bandpass`.

    Returns
    -------
    wpli : jax.Array
        Values in ``[0, 1]`` of shape ``x.shape[1:]``. ``nan`` where the
        imaginary cross-spectrum is identically zero (e.g. identical signals),
        since the index is then undefined.

    Raises
    ------
    ValueError
        If the shapes differ, ``band`` is given without ``dt``, or ``band`` is
        invalid.

    See Also
    --------
    plv : Phase-locking value.
    pli : Phase-lag index.

    Notes
    -----
    Jittable and vmappable. Differentiable with respect to the signals wherever
    the denominator is non-zero.

    References
    ----------
    .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., and
           Pennartz, C. M. A. "An improved index of phase-synchronization for
           electrophysiological data in the presence of volume-conduction,
           noise and sample-size bias." NeuroImage 55.4 (2011): 1548-1565.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.analysis.spectral import wpli
        >>> t = jnp.arange(1000) * 1e-3
        >>> x = jnp.cos(2 * jnp.pi * 10 * t)
        >>> round(float(wpli(x, jnp.cos(2 * jnp.pi * 10 * t - jnp.pi / 4))), 4)
        1.0
    """
    zx, zy = _cross_spectrum(x, y, dt, band)
    im = _imag_cross(zx, zy)
    num = jnp.abs(jnp.mean(im, axis=0))
    den = jnp.mean(jnp.abs(im), axis=0)
    return jnp.where(den > 0, num / jnp.where(den > 0, den, 1.0), jnp.nan)
