# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Optional, Tuple, Union

import brainstate
import brainunit as u
import jax
from jax import numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'unitary_LFP',
    'power_spectral_density',
    'coherence_analysis',
    'phase_amplitude_coupling',
    'theta_gamma_coupling',
    'current_source_density',
    'spectral_entropy',
    'lfp_phase_coherence',
]


def _segment_starts(n_time: int, nperseg: int, noverlap: int):
    """Return the list of (static) segment start indices for Welch segmentation."""
    step = max(1, nperseg - noverlap)
    starts = list(range(0, n_time - nperseg + 1, step))
    return starts if starts else [0]


def _to_seconds(dt: Union[float, u.Quantity]) -> float:
    """Return the sampling interval in seconds (Quantity -> seconds, float as-is)."""
    if isinstance(dt, u.Quantity):
        return float(dt.to_decimal(u.second))
    return float(dt)


def _analytic_band(fft_vals: jax.Array, freqs: jax.Array, low: float, high: float) -> jax.Array:
    """Reconstruct the band-limited analytic signal from a full FFT.

    Keeping only the **positive** in-band frequency bins (and doubling them) yields a
    complex analytic signal whose ``angle`` is the instantaneous phase and whose
    ``abs`` is the amplitude envelope. The earlier symmetric ``|f|`` mask produced an
    almost-real band-passed signal whose phase was degenerate (≈ 0 or π).
    """
    # Broadcast the per-frequency mask/factor against the frequency axis (axis 0),
    # supporting both 1-D and (n_time, ...) inputs.
    shape = (-1,) + (1,) * (fft_vals.ndim - 1)
    mask = ((freqs >= low) & (freqs <= high)).reshape(shape)
    # Double only strictly positive in-band bins. DC (f == 0) must NOT be doubled
    # in the analytic-signal construction; doubling it distorts the phase of any
    # band that includes 0 Hz.
    factor = jnp.where(freqs == 0.0, 1.0, 2.0).reshape(shape)
    band = jnp.where(mask, fft_vals * factor, 0.0 + 0.0j)
    return jnp.fft.ifft(band, axis=0)


@set_module_as('braintools.metric')
def unitary_LFP(
    times: brainstate.typing.ArrayLike,
    spikes: brainstate.typing.ArrayLike,
    spike_type: str,
    xmax: brainstate.typing.ArrayLike = 0.2,
    ymax: brainstate.typing.ArrayLike = 0.2,
    va: brainstate.typing.ArrayLike = 200.,
    lambda_: brainstate.typing.ArrayLike = 0.2,
    sig_i: brainstate.typing.ArrayLike = 2.1,
    sig_e: brainstate.typing.ArrayLike = 2.1 * 1.5,
    location: str = 'soma layer',
    seed: brainstate.typing.SeedOrKey = None
) -> jax.Array:
    r"""Calculate unitary local field potentials (uLFP) from spike train data.

    Computes the contribution of spiking neurons to local field potentials using
    a kernel-based method. This approach models the spatial distribution of neurons,
    axonal conduction delays, and layer-specific amplitude scaling to estimate
    the LFP signal recorded at an electrode positioned at the center of the
    neural population.

    The method implements a biophysically-motivated model where each spike
    contributes to the LFP through a Gaussian kernel with amplitude and delay
    determined by the neuron's distance from the recording electrode:

    .. math::

        \text{uLFP}(t) = \sum_{i,s} A_i \exp\left(-\frac{(t - t_s - \delta_i)^2}{2\sigma^2}\right)

    where :math:`A_i` is the distance-dependent amplitude, :math:`t_s` is the
    spike time, :math:`\delta_i` is the conduction delay, and :math:`\sigma`
    is the kernel width (different for excitatory and inhibitory neurons).

    Parameters
    ----------
    times : brainstate.typing.ArrayLike
        Time points of the recording with shape ``(n_time_steps,)``, in
        **milliseconds**.
    spikes : brainstate.typing.ArrayLike
        Binary spike matrix with shape ``(n_time_steps, n_neurons)`` where
        non-zero values indicate spike occurrences.
    spike_type : {'exc', 'inh'}
        Type of neurons generating the spikes:

        - ``'exc'``: Excitatory neurons (positive contribution)
        - ``'inh'``: Inhibitory neurons (can be positive or negative depending on layer)

    xmax : float, default=0.2
        Spatial extent of the neuron population in the x-dimension (mm).
        Neurons are randomly distributed within a rectangle of size
        ``xmax × ymax`` centered at the electrode position.
    ymax : float, default=0.2
        Spatial extent of the neuron population in the y-dimension (mm).
    va : float, default=200.0
        Axonal conduction velocity in mm/s. Determines the delay between
        spike occurrence and its contribution to the LFP. Typical values
        range from 100-500 mm/s for cortical neurons.
    lambda_ : float, default=0.2
        Spatial decay constant in mm. Controls how quickly the LFP amplitude
        decreases with distance from the electrode.
    sig_i : float, default=2.1
        Standard deviation of the inhibitory neuron kernel in ms.
    sig_e : float, default=3.15
        Standard deviation of the excitatory neuron kernel in ms
        (default ``2.1 * 1.5``).
    location : {'soma layer', 'deep layer', 'superficial layer', 'surface'}, default='soma layer'
        Recording electrode location relative to the cortical layers:

        - ``'soma layer'``: At the soma level (excitatory: +0.48, inhibitory: +3.0)
        - ``'deep layer'``: Below soma layer (excitatory: -0.16, inhibitory: -0.2)
        - ``'superficial layer'``: Above soma layer (excitatory: +0.24, inhibitory: -1.2)
        - ``'surface'`` (alias ``'surface layer'``): At cortical surface
          (excitatory: -0.08, inhibitory: +0.3)

    seed : brainstate.typing.SeedOrKey, optional
        Random seed for reproducible neuron positioning.

    Returns
    -------
    jax.Array
        Unitary LFP signal with shape ``(n_time_steps,)`` representing the
        contribution of the specified neuron population to the local field
        potential, in microvolts (μV).

    Raises
    ------
    ValueError
        If ``spike_type`` is not ``'exc'`` or ``'inh'``, if ``spikes`` is not 2-D,
        or if ``times`` and ``spikes`` have incompatible first axes.
    NotImplementedError
        If ``location`` is not one of the supported options.

    Notes
    -----
    This implementation focuses on spike-triggered LFP contributions and does
    not account for subthreshold synaptic currents, dendritic active channels,
    volume conduction from distant sources, or frequency-dependent propagation.

    The per-neuron peak delay is :math:`\delta_i = 10.4\,\text{ms} + d_i / v_a`,
    where :math:`d_i` is the neuron-electrode distance (mm) and :math:`v_a` the
    axonal velocity (mm/s); the ``d_i / v_a`` term (seconds) is converted to
    milliseconds so it is commensurate with the 10.4 ms peak constant of
    Telenczuk et al. (2020).

    This function runs on host (concrete) arrays — it uses ``jnp.where(spikes)``
    (data-dependent output shape), so it is not ``jit``/``vmap``-compatible.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> brainstate.random.seed(42)
        >>> n_time, n_exc = 1000, 100
        >>> times = jnp.arange(n_time) * 0.1  # ms
        >>> exc_spikes = (brainstate.random.random((n_time, n_exc)) < 0.02).astype(float)
        >>> lfp = braintools.metric.unitary_LFP(times, exc_spikes, 'exc', seed=42)
        >>> lfp.shape
        (1000,)

    References
    ----------
    .. [1] Telenczuk, Bartosz, Maria Telenczuk, and Alain Destexhe.
           "A kernel-based method to calculate local field potentials from
           networks of spiking neurons." Journal of Neuroscience Methods
           344 (2020): 108871. https://doi.org/10.1016/j.jneumeth.2020.108871
    .. [2] Einevoll, Gaute T., et al. "Modelling and analysis of local field
           potentials for studying the function of cortical circuits."
           Nature Reviews Neuroscience 14.11 (2013): 770-785.
    """
    if spike_type not in ('exc', 'inh'):
        raise ValueError(f'"spike_type" should be "exc" or "inh", but we got {spike_type!r}.')
    spikes = jnp.asarray(spikes)
    times = jnp.asarray(times)
    if spikes.ndim != 2:
        raise ValueError('"spikes" should be a matrix with shape (num_time, num_neuron), '
                         f'but we got {spikes.shape}.')
    if times.shape[0] != spikes.shape[0]:
        raise ValueError('"times" and "spikes" should be consistent along the first axis, '
                         f'but we got {times.shape[0]} != {spikes.shape[0]}.')

    # Distributing cells in a 2D grid.
    rng = brainstate.random.RandomState(seed)
    num_neuron = spikes.shape[1]
    pos_xs, pos_ys = rng.rand(2, num_neuron) * jnp.array([[xmax], [ymax]])
    pos_xs, pos_ys = jnp.asarray(pos_xs), jnp.asarray(pos_ys)

    # Distance of each neuron to the electrode at the population center (mm).
    xe, ye = xmax / 2, ymax / 2
    dist = jnp.sqrt((pos_xs - xe) ** 2 + (pos_ys - ye) ** 2)

    # Layer-/type-specific amplitude.
    if location == 'soma layer':
        amp_e, amp_i = 0.48, 3.
    elif location == 'deep layer':
        amp_e, amp_i = -0.16, -0.2
    elif location == 'superficial layer':
        amp_e, amp_i = 0.24, -1.2
    elif location in ('surface layer', 'surface'):
        amp_e, amp_i = -0.08, 0.3
    else:
        raise NotImplementedError(
            f"Unknown location {location!r}. Choose from 'soma layer', 'deep layer', "
            "'superficial layer', or 'surface'."
        )
    A = jnp.exp(-dist / lambda_) * (amp_e if spike_type == 'exc' else amp_i)

    # Peak delay (ms): 10.4 ms constant + distance[mm] / velocity[mm/s] converted s -> ms.
    delay = 10.4 + (dist / va) * 1e3

    # LFP calculation: sum a Gaussian bump per spike.
    iis, ids = jnp.where(spikes)
    tts = times[iis] + delay[ids]
    amp = A[ids]
    tau = (2 * sig_e * sig_e) if spike_type == 'exc' else (2 * sig_i * sig_i)
    return brainstate.transform.for_loop(
        lambda t: jnp.sum(amp * jnp.exp(-(t - tts) ** 2 / tau)), times
    )


@set_module_as('braintools.metric')
def power_spectral_density(
    lfp: brainstate.typing.ArrayLike,
    dt: Union[float, u.Quantity],
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None
) -> Tuple[jax.Array, jax.Array]:
    """Estimate the one-sided power spectral density (PSD) using Welch's method.

    The signal is split into overlapping Hann-windowed segments; the periodogram of
    each segment is averaged. Power is normalized by ``fs * sum(window**2)`` and the
    one-sided spectrum doubles every bin except DC (and Nyquist for even-length
    segments), so the PSD integrates to the signal variance.

    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal with shape ``(n_time,)`` or ``(n_time, n_channels)``.
        ``brainunit.Quantity`` inputs are accepted (the magnitude is used).
    dt : float or brainunit.Quantity
        Sampling interval. If a float, it is taken to be in **seconds** (so ``fs``
        is in Hz); if a ``Quantity``, it is converted to seconds.
    nperseg : int, optional
        Length of each segment. Default: ``n_time // 8``.
    noverlap : int, optional
        Number of points to overlap between segments. Default: ``nperseg // 2``.
    freq_range : tuple of float, optional
        ``(f_min, f_max)`` in Hz to retain. If None, returns all frequencies.

    Returns
    -------
    freqs : jax.Array
        One-sided sample frequencies in Hz, shape ``(nperseg // 2 + 1,)``.
    psd : jax.Array
        Power spectral density: shape ``(n_freqs,)`` for 1-D input or
        ``(n_freqs, n_channels)`` for 2-D input.

    Notes
    -----
    Supplying ``freq_range`` uses boolean-mask indexing (data-dependent output
    length), so that path is not ``jit``-compatible; the full-spectrum path is.

    The window is ``jnp.hanning``, which is the *symmetric* Hann window;
    ``scipy.signal.welch`` uses a *periodic* (``sym=False``) Hann window by
    default, so PSD values differ marginally when comparing against scipy.
    """
    dt = _to_seconds(dt)
    lfp = jnp.asarray(u.get_magnitude(lfp))
    squeeze = lfp.ndim == 1
    if squeeze:
        lfp = lfp[:, None]
    n_time, n_channels = lfp.shape

    if nperseg is None:
        nperseg = max(1, n_time // 8)
    nperseg = int(min(nperseg, n_time))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(noverlap, nperseg - 1)) if nperseg > 1 else 0

    fs = 1.0 / dt
    window = jnp.hanning(nperseg)
    win_power = jnp.sum(window ** 2)
    starts = _segment_starts(n_time, nperseg, noverlap)

    segs = jnp.stack([lfp[s:s + nperseg] for s in starts], axis=0)  # (n_seg, nperseg, n_ch)
    segs = segs * window[None, :, None]
    spectra = jnp.fft.rfft(segs, axis=1)                           # (n_seg, n_freq, n_ch)
    psd = jnp.mean(jnp.abs(spectra) ** 2, axis=0) / (fs * win_power)

    # One-sided scaling: double every bin except DC (and Nyquist for even nperseg).
    psd = psd.at[1:].multiply(2.0)
    if nperseg % 2 == 0:
        psd = psd.at[-1].divide(2.0)

    freqs = jnp.fft.rfftfreq(nperseg, dt)

    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        psd = psd[mask]
        if freqs.size == 0:
            freqs = jnp.array([freq_range[0]])
            psd = jnp.zeros((1, n_channels))

    return freqs, (psd[:, 0] if squeeze else psd)


@set_module_as('braintools.metric')
def coherence_analysis(
    lfp1: brainstate.typing.ArrayLike,
    lfp2: brainstate.typing.ArrayLike,
    dt: Union[float, u.Quantity],
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None
) -> Tuple[jax.Array, jax.Array]:
    r"""Compute the magnitude-squared coherence between two LFP signals (Welch).

    Coherence is estimated by averaging cross- and auto-spectra over overlapping
    Hann-windowed segments:

    .. math::

        C_{xy}(f) = \frac{|\langle P_{xy}(f) \rangle|^2}
                         {\langle P_{xx}(f) \rangle \, \langle P_{yy}(f) \rangle}

    Parameters
    ----------
    lfp1, lfp2 : brainstate.typing.ArrayLike
        LFP signals with shape ``(n_time,)``.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a float; converted if a ``Quantity``).
    nperseg : int, optional
        Length of each segment. Default: ``n_time // 8``.
    noverlap : int, optional
        Overlap between segments. Default: ``nperseg // 2``.
    freq_range : tuple of float, optional
        ``(f_min, f_max)`` in Hz to retain. If None, returns all frequencies.

    Returns
    -------
    freqs : jax.Array
        One-sided sample frequencies in Hz.
    coherence : jax.Array
        Magnitude-squared coherence in ``[0, 1]``.

    Notes
    -----
    Averaging over segments is **essential**: with a single segment the estimator
    is identically 1 at every frequency. The default ``nperseg = n_time // 8`` with
    50% overlap yields ~15 segments. Choose ``nperseg`` so at least two segments
    fit, otherwise the result is degenerate.
    """
    dt = _to_seconds(dt)
    lfp1 = jnp.asarray(u.get_magnitude(lfp1))
    lfp2 = jnp.asarray(u.get_magnitude(lfp2))
    n_time = lfp1.shape[0]

    if nperseg is None:
        nperseg = max(1, n_time // 8)
    nperseg = int(min(nperseg, n_time))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(noverlap, nperseg - 1)) if nperseg > 1 else 0

    window = jnp.hanning(nperseg)
    starts = _segment_starts(n_time, nperseg, noverlap)
    s1 = jnp.stack([lfp1[s:s + nperseg] for s in starts], axis=0) * window[None, :]
    s2 = jnp.stack([lfp2[s:s + nperseg] for s in starts], axis=0) * window[None, :]

    f1 = jnp.fft.rfft(s1, axis=1)
    f2 = jnp.fft.rfft(s2, axis=1)
    p12 = jnp.mean(f1 * jnp.conj(f2), axis=0)
    p11 = jnp.mean(jnp.abs(f1) ** 2, axis=0)
    p22 = jnp.mean(jnp.abs(f2) ** 2, axis=0)
    # Scale-invariant guard: bins with zero power (denominator 0) yield 0 (their
    # cross-power is also 0). Avoids the absolute ``+ 1e-15`` floor, which biased
    # genuinely-coherent bins toward 0 for amplitude-downscaled signals.
    denom = p11 * p22
    coherence = jnp.abs(p12) ** 2 / jnp.where(denom > 0, denom, 1.0)
    coherence = jnp.clip(coherence, 0.0, 1.0)

    freqs = jnp.fft.rfftfreq(nperseg, dt)
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        coherence = coherence[mask]
        # Mirror power_spectral_density: never return empty arrays for an
        # out-of-range band.
        if freqs.size == 0:
            freqs = jnp.array([freq_range[0]])
            coherence = jnp.zeros((1,))
    return freqs, coherence


@set_module_as('braintools.metric')
def phase_amplitude_coupling(
    lfp: brainstate.typing.ArrayLike,
    dt: Union[float, u.Quantity],
    phase_freq_range: Tuple[float, float] = (4, 8),
    amplitude_freq_range: Tuple[float, float] = (30, 100),
    n_bins: int = 18
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""Compute phase-amplitude coupling (PAC) via the Tort modulation index.

    The low-frequency phase and high-frequency amplitude are extracted from the
    **analytic** band-limited signals (Hilbert construction in the frequency
    domain). The amplitude is binned by phase and the modulation index is the
    Kullback-Leibler divergence of the amplitude distribution from uniform,
    normalized by ``log(n_bins)``.

    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal with shape ``(n_time,)``.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a float; converted if a ``Quantity``).
    phase_freq_range : tuple of float, default=(4, 8)
        Low-frequency band (Hz) for phase extraction, e.g. theta.
    amplitude_freq_range : tuple of float, default=(30, 100)
        High-frequency band (Hz) for amplitude extraction, e.g. gamma.
    n_bins : int, default=18
        Number of phase bins.

    Returns
    -------
    modulation_index : jax.Array
        Scalar modulation index in ``[0, 1]`` (0 = no coupling).
    phase_bins : jax.Array
        Phase bin centers, shape ``(n_bins,)``.
    mean_amplitudes : jax.Array
        Mean high-frequency amplitude in each phase bin, shape ``(n_bins,)``.
    """
    dt = _to_seconds(dt)
    lfp = jnp.asarray(u.get_magnitude(lfp))
    n_fft = lfp.shape[0]
    freqs = jnp.fft.fftfreq(n_fft, dt)
    fft_lfp = jnp.fft.fft(lfp)

    phase_signal = _analytic_band(fft_lfp, freqs, phase_freq_range[0], phase_freq_range[1])
    instantaneous_phase = jnp.angle(phase_signal)

    amp_signal = _analytic_band(fft_lfp, freqs, amplitude_freq_range[0], amplitude_freq_range[1])
    instantaneous_amplitude = jnp.abs(amp_signal)

    # Bin amplitudes by phase using segment sums (jit-safe; empty bins -> 0).
    phase_bins = jnp.linspace(-jnp.pi, jnp.pi, n_bins + 1)
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    bin_idx = jnp.clip(jnp.digitize(instantaneous_phase, phase_bins) - 1, 0, n_bins - 1)
    sums = jax.ops.segment_sum(instantaneous_amplitude, bin_idx, num_segments=n_bins)
    counts = jax.ops.segment_sum(jnp.ones_like(instantaneous_amplitude), bin_idx, num_segments=n_bins)
    mean_amplitudes = sums / jnp.maximum(counts, 1.0)

    # Tort modulation index: normalized KL divergence from the uniform distribution.
    p = mean_amplitudes / (jnp.sum(mean_amplitudes) + 1e-12)
    p = jnp.where(p > 0, p, 1e-12)
    entropy = -jnp.sum(p * jnp.log(p))
    max_entropy = jnp.log(n_bins)
    modulation_index = (max_entropy - entropy) / max_entropy

    return modulation_index, bin_centers, mean_amplitudes


@set_module_as('braintools.metric')
def theta_gamma_coupling(
    lfp: brainstate.typing.ArrayLike,
    dt: Union[float, u.Quantity]
) -> jax.Array:
    """Compute theta-gamma coupling strength using standard frequency bands.

    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal with shape ``(n_time,)``.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a float; converted if a ``Quantity``).

    Returns
    -------
    coupling_strength : jax.Array
        Scalar theta-gamma (4-8 Hz phase, 30-80 Hz amplitude) modulation index.
    """
    return phase_amplitude_coupling(
        lfp, dt,
        phase_freq_range=(4, 8),  # Theta band
        amplitude_freq_range=(30, 80)  # Gamma band
    )[0]


@set_module_as('braintools.metric')
def current_source_density(
    lfp_laminar: brainstate.typing.ArrayLike,
    electrode_spacing: Union[float, u.Quantity],
    conductivity: float = 1.0,
    axis: int = -1
) -> jax.Array:
    r"""Compute current source density (CSD) from laminar LFP recordings.

    The CSD is the (negative, conductivity-scaled) second spatial derivative of the
    potential along the electrode axis:

    .. math::

        \text{CSD}(z) \approx -\sigma\, \frac{\phi(z+h) - 2\phi(z) + \phi(z-h)}{h^2}

    Parameters
    ----------
    lfp_laminar : brainstate.typing.ArrayLike
        Laminar LFP data. The electrode axis (selected by ``axis``) must be ordered
        from superficial to deep and have at least 3 electrodes.
    electrode_spacing : float or brainunit.Quantity
        Spacing :math:`h` between adjacent electrodes. A float is taken to be in
        **mm**; a ``Quantity`` is converted to mm. The CSD is therefore expressed in
        ``[lfp] / mm^2`` (scaled by ``conductivity``).
    conductivity : float, default=1.0
        Tissue conductivity :math:`\sigma` (assumed constant).
    axis : int, default=-1
        Axis along which electrodes are arranged. Use ``axis=0`` for
        channels-first ``(n_electrodes, n_time)`` data, or the default ``axis=-1``
        for ``(n_time, n_electrodes)`` data.

    Returns
    -------
    csd : jax.Array
        Current source density with the electrode axis reduced by 2 (the boundary
        electrodes are dropped).

    Raises
    ------
    ValueError
        If fewer than 3 electrodes are present along ``axis``.
    """
    if isinstance(electrode_spacing, u.Quantity):
        spacing = float(electrode_spacing.to_decimal(u.mm))
    else:
        spacing = float(electrode_spacing)
    lfp_laminar = jnp.asarray(u.get_magnitude(lfp_laminar))

    n_elec = lfp_laminar.shape[axis]
    if n_elec < 3:
        raise ValueError(
            f"current_source_density requires at least 3 electrodes along axis {axis}, "
            f"but got {n_elec}."
        )

    x = jnp.moveaxis(lfp_laminar, axis, -1)
    csd = -conductivity * (x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]) / (spacing ** 2)
    return jnp.moveaxis(csd, -1, axis)


@set_module_as('braintools.metric')
def spectral_entropy(
    lfp: brainstate.typing.ArrayLike,
    dt: Union[float, u.Quantity],
    freq_range: Tuple[float, float] = (1, 100)
) -> jax.Array:
    """Compute the normalized spectral entropy of an LFP signal.

    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal with shape ``(n_time,)`` (multi-channel input is averaged into a
        single spectrum before the entropy is computed).
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a float; converted if a ``Quantity``).
    freq_range : tuple of float, default=(1, 100)
        Frequency range (Hz) for the entropy calculation.

    Returns
    -------
    entropy : jax.Array
        Normalized spectral entropy in ``[0, 1]`` (0 = most regular / peaked,
        1 = flat / most random).
    """
    freqs, psd = power_spectral_density(lfp, dt, freq_range=freq_range)
    psd = jnp.asarray(psd)
    if psd.ndim > 1:
        psd = jnp.mean(psd, axis=-1)

    n = psd.shape[0] if psd.ndim > 0 else 1
    if n <= 1:
        return jnp.asarray(0.0)

    total = jnp.sum(psd)
    # Fall back to a uniform distribution when the band carries no power.
    psd_norm = jnp.where(total > 0, psd / jnp.where(total > 0, total, 1.0), 1.0 / n)
    psd_norm = jnp.maximum(psd_norm, 1e-12)
    entropy = -jnp.sum(psd_norm * jnp.log2(psd_norm))
    return entropy / jnp.log2(n)


@set_module_as('braintools.metric')
def lfp_phase_coherence(
    lfp_signals: brainstate.typing.ArrayLike,
    dt: Union[float, u.Quantity],
    freq_band: Tuple[float, float] = (8, 12)
) -> jax.Array:
    r"""Compute pairwise phase coherence between LFP channels in a frequency band.

    For each channel the band-limited analytic signal is built (Hilbert
    construction), and the pairwise phase-locking value

    .. math::

        \text{PLV}_{ij} = \left| \frac{1}{T} \sum_t e^{\,i(\phi_i(t) - \phi_j(t))} \right|

    is returned for every channel pair.

    Parameters
    ----------
    lfp_signals : brainstate.typing.ArrayLike
        Multiple LFP signals with shape ``(n_time, n_channels)``.
    dt : float or brainunit.Quantity
        Sampling interval (seconds if a float; converted if a ``Quantity``).
    freq_band : tuple of float, default=(8, 12)
        Frequency band (Hz) for phase extraction (e.g. the alpha band).

    Returns
    -------
    phase_coherence_matrix : jax.Array
        Symmetric coherence matrix with shape ``(n_channels, n_channels)`` and
        values in ``[0, 1]`` (diagonal exactly 1).

    Notes
    -----
    The phase of a channel that carries negligible power in ``freq_band`` is
    undefined. Such channels are detected (in-band power negligible relative to
    their broadband power) and their off-diagonal coherence is reported as ``0``
    rather than a spurious value (a zero band-limited signal would otherwise
    appear perfectly phase-locked).
    """
    dt = _to_seconds(dt)
    lfp_signals = jnp.asarray(u.get_magnitude(lfp_signals))
    if lfp_signals.ndim != 2:
        raise ValueError(
            "lfp_phase_coherence expects a 2-D array of shape (n_time, n_channels), "
            f"but got shape {lfp_signals.shape}."
        )
    n_time, n_channels = lfp_signals.shape

    freqs = jnp.fft.fftfreq(n_time, dt)
    mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])  # positive band -> analytic
    fft = jnp.fft.fft(lfp_signals, axis=0)
    analytic = jnp.fft.ifft(jnp.where(mask[:, None], fft, 0.0 + 0.0j) * 2.0, axis=0)
    phases = jnp.angle(analytic)                       # (n_time, n_channels)

    z = jnp.exp(1j * phases)                           # unit phasors
    # (i, j) = mean_t conj(z_i) z_j = mean_t exp(i(phi_j - phi_i)); |.| is the PLV.
    coherence = jnp.abs(jnp.conj(z).T @ z) / n_time
    # PLV is the magnitude of a normalized sum of unit phasors, so it is
    # mathematically bounded in [0, 1]. Clip to remove float32 roundoff that
    # can push near-coherent pairs marginally above 1.0.
    coherence = jnp.clip(coherence, 0.0, 1.0)

    # A channel with negligible power in ``freq_band`` has a degenerate (~0)
    # analytic signal whose phase is undefined -- ``angle(0) == 0`` would make
    # every such channel look perfectly phase-locked. Detect those channels
    # (in-band power negligible relative to broadband power) and report their
    # off-diagonal coherence as 0 instead of a spurious value.
    in_band_power = jnp.mean(jnp.abs(analytic) ** 2, axis=0)      # (n_channels,)
    total_power = jnp.mean(lfp_signals ** 2, axis=0)              # (n_channels,)
    valid = in_band_power > 1e-6 * jnp.maximum(total_power, 1e-30)
    coherence = jnp.where(valid[:, None] & valid[None, :], coherence, 0.0)

    # A channel is perfectly phase-locked with itself, so the diagonal is
    # exactly 1 by definition; set it explicitly to remove float32 drift that
    # can leave it marginally below 1.0.
    coherence = coherence.at[jnp.diag_indices(n_channels)].set(1.0)
    return coherence
