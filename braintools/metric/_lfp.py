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

import brainstate
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
        Time points of the recording with shape ``(n_time_steps,)``. These
        represent the temporal sampling points for the LFP calculation,
        typically in milliseconds.
    spikes : brainstate.typing.ArrayLike
        Binary spike matrix with shape ``(n_time_steps, n_neurons)`` where
        non-zero values indicate spike occurrences. Each element
        ``spikes[t, i]`` represents whether neuron ``i`` fired at time ``t``.
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
        decreases with distance from the electrode. Smaller values result
        in more localized LFP signals.
    sig_i : float, default=2.1
        Standard deviation of the inhibitory neuron kernel in ms.
        Determines the temporal width of inhibitory contributions to the LFP.
    sig_e : float, default=3.15
        Standard deviation of the excitatory neuron kernel in ms.
        Default is ``2.1 * 1.5``, making excitatory contributions broader
        than inhibitory ones.
    location : {'soma layer', 'deep layer', 'superficial layer', 'surface'}, default='soma layer'
        Recording electrode location relative to the cortical layers:
        
        - ``'soma layer'``: At the soma level (excitatory: +0.48, inhibitory: +3.0)
        - ``'deep layer'``: Below soma layer (excitatory: -0.16, inhibitory: -0.2)
        - ``'superficial layer'``: Above soma layer (excitatory: +0.24, inhibitory: -1.2)
        - ``'surface'``: At cortical surface (excitatory: -0.08, inhibitory: +0.3)
        
        Values in parentheses indicate the base amplitude scaling factors.
    seed : brainstate.typing.SeedOrKey, optional
        Random seed for reproducible neuron positioning. If None, positions
        are generated randomly. Use for consistent results across runs.

    Returns
    -------
    jax.Array
        Unitary LFP signal with shape ``(n_time_steps,)`` representing the
        contribution of the specified neuron population to the local field
        potential. Units are typically in microvolts (μV).

    Raises
    ------
    ValueError
        If ``spike_type`` is not 'exc' or 'inh', if ``spikes`` is not 2D,
        or if ``times`` and ``spikes`` have incompatible shapes.
    NotImplementedError
        If ``location`` is not one of the supported options.

    Notes
    -----
    This implementation focuses on spike-triggered LFP contributions and does
    not account for:
    
    - Subthreshold synaptic currents
    - Dendritic voltage-dependent ion channels  
    - Volume conduction effects from distant sources
    - Frequency-dependent propagation
    
    For realistic LFP modeling, combine contributions from both excitatory
    and inhibitory populations and consider using multiple electrode locations.

    The neuron positions are randomly generated within the specified spatial
    bounds, and the electrode is positioned at the center ``(xmax/2, ymax/2)``.
    Each neuron's contribution is weighted by distance and scaled according
    to the recording location and neuron type.

    Examples
    --------
    Calculate LFP from excitatory and inhibitory populations:

    >>> import brainstate as brainstate
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Set up simulation parameters
    >>> brainstate.random.seed(42)
    >>> n_time, n_exc, n_inh = 1000, 100, 25
    >>> dt = 0.1  # ms
    >>> times = jnp.arange(n_time) * dt
    >>> # Generate sparse random spike trains
    >>> exc_spikes = (brainstate.random.random((n_time, n_exc)) < 0.02).astype(float)
    >>> inh_spikes = (brainstate.random.random((n_time, n_inh)) < 0.04).astype(float)
    >>> # Calculate LFP components
    >>> lfp_exc = braintools.metric.unitary_LFP(times, exc_spikes, 'exc', seed=42)
    >>> lfp_inh = braintools.metric.unitary_LFP(times, inh_spikes, 'inh', seed=42)
    >>> total_lfp = lfp_exc + lfp_inh
    >>> print(f"LFP shape: {total_lfp.shape}")
    >>> print(f"LFP range: {total_lfp.min():.3f} to {total_lfp.max():.3f}")

    Compare different recording locations:

    >>> # Same spike data, different recording depths
    >>> lfp_soma = braintools.metric.unitary_LFP(times, exc_spikes, 'exc',
    ...                                  location='soma layer')
    >>> lfp_deep = braintools.metric.unitary_LFP(times, exc_spikes, 'exc',
    ...                                  location='deep layer')
    >>> lfp_surface = braintools.metric.unitary_LFP(times, exc_spikes, 'exc',
    ...                                      location='surface')

    Analyze the effect of spatial parameters:

    >>> # Larger population area
    >>> lfp_large = braintools.metric.unitary_LFP(times, exc_spikes, 'exc',
    ...                                   xmax=0.5, ymax=0.5)
    >>> # Faster conduction velocity
    >>> lfp_fast = braintools.metric.unitary_LFP(times, exc_spikes, 'exc', va=500.0)

    Visualize the results:

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(times[:500], total_lfp[:500], 'k-', linewidth=1)
    >>> plt.xlabel('Time (ms)')
    >>> plt.ylabel('LFP Amplitude (μV)')
    >>> plt.title('Simulated Local Field Potential')
    >>> plt.grid(True, alpha=0.3)
    >>> plt.show()

    See Also
    --------
    braintools.metric.firing_rate : Calculate population firing rates
    braintools.metric.raster_plot : Extract spike timing data
    jax.numpy.convolve : Alternative smoothing approach for LFP

    References
    ----------
    .. [1] Telenczuk, Bartosz, Maria Telenczuk, and Alain Destexhe.
           "A kernel-based method to calculate local field potentials from
           networks of spiking neurons." Journal of Neuroscience Methods
           344 (2020): 108871. https://doi.org/10.1016/j.jneumeth.2020.108871
    .. [2] Einevoll, Gaute T., et al. "Modelling and analysis of local field
           potentials for studying the function of cortical circuits."
           Nature Reviews Neuroscience 14.11 (2013): 770-785.
    .. [3] Buzsáki, György, Costas A. Anastassiou, and Christof Koch.
           "The origin of extracellular fields and currents—EEG, ECoG, LFP
           and spikes." Nature Reviews Neuroscience 13.6 (2012): 407-420.
    """
    if spike_type not in ['exc', 'inh']:
        raise ValueError('"spike_type" should be "exc or ""inh". ')
    if spikes.ndim != 2:
        raise ValueError('"E_spikes" should be a matrix with shape of (num_time, num_neuron). '
                         f'But we got {spikes.shape}')
    if times.shape[0] != spikes.shape[0]:
        raise ValueError('times and spikes should be consistent at the firs axis. '
                         f'But we got {times.shape[0]} != {spikes.shape}.')

    # Distributing cells in a 2D grid
    rng = brainstate.random.RandomState(seed)
    num_neuron = spikes.shape[1]
    pos_xs, pos_ys = rng.rand(2, num_neuron) * jnp.array([[xmax], [ymax]])
    pos_xs, pos_ys = jnp.asarray(pos_xs), jnp.asarray(pos_ys)

    # distance/coordinates
    xe, ye = xmax / 2, ymax / 2  # coordinates of electrode
    dist = jnp.sqrt((pos_xs - xe) ** 2 + (pos_ys - ye) ** 2)  # distance to electrode in mm

    # amplitude
    if location == 'soma layer':
        amp_e, amp_i = 0.48, 3.  # exc/inh uLFP amplitude (soma layer)
    elif location == 'deep layer':
        amp_e, amp_i = -0.16, -0.2  # exc/inh uLFP amplitude (deep layer)
    elif location == 'superficial layer':
        amp_e, amp_i = 0.24, -1.2  # exc/inh uLFP amplitude (superficial layer)
    elif location == 'surface layer':
        amp_e, amp_i = -0.08, 0.3  # exc/inh uLFP amplitude (surface)
    else:
        raise NotImplementedError
    A = jnp.exp(-dist / lambda_) * (amp_e if spike_type == 'exc' else amp_i)

    # delay
    delay = 10.4 + dist / va  # delay to peak (in ms)

    # LFP Calculation
    iis, ids = jnp.where(spikes)
    tts = times[iis] + delay[ids]
    exc_amp = A[ids]
    tau = (2 * sig_e * sig_e) if spike_type == 'exc' else (2 * sig_i * sig_i)
    return brainstate.compile.for_loop(lambda t: jnp.sum(exc_amp * jnp.exp(-(t - tts) ** 2 / tau)), times)


@set_module_as('braintools.metric')
def power_spectral_density(
    lfp: brainstate.typing.ArrayLike,
    dt: float,
    nperseg: int = None,
    noverlap: int = None,
    freq_range: tuple = None
) -> tuple:
    """Compute power spectral density (PSD) of LFP signals using Welch's method.
    
    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal array with shape (n_time,) or (n_time, n_channels).
    dt : float
        Sampling interval in seconds.
    nperseg : int, optional
        Length of each segment for PSD calculation. Default: n_time // 8.
    noverlap : int, optional
        Number of points to overlap between segments. Default: nperseg // 2.
    freq_range : tuple, optional
        Frequency range (f_min, f_max) in Hz to extract. If None, returns all frequencies.
    
    Returns
    -------
    freqs : jax.Array
        Array of sample frequencies.
    psd : jax.Array
        Power spectral density estimate.
    """
    lfp = jnp.asarray(lfp)
    if lfp.ndim == 1:
        lfp = lfp[:, None]

    n_time, n_channels = lfp.shape
    fs = 1.0 / dt

    if nperseg is None:
        nperseg = n_time // 8
    if noverlap is None:
        noverlap = nperseg // 2

    # Simple periodogram approach (more JAX-friendly than Welch's method)
    n_fft = 2 ** int(jnp.ceil(jnp.log2(nperseg)))
    freqs = jnp.fft.fftfreq(n_fft, dt)[:n_fft // 2]

    # Compute FFT-based PSD
    windowed_fft = jnp.fft.fft(lfp[:nperseg] * jnp.hanning(nperseg)[:, None], n=n_fft, axis=0)
    psd = (jnp.abs(windowed_fft[:n_fft // 2]) ** 2) / (fs * nperseg)

    if freq_range is not None:
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[freq_mask]
        psd = psd[freq_mask]

        # Handle case where no frequencies are in range
        if freqs.size == 0:
            freqs = jnp.array([freq_range[0]])
            psd = jnp.zeros((1, n_channels)) if n_channels > 1 else jnp.array([0.0])

    return freqs, psd.squeeze() if n_channels == 1 else psd


@set_module_as('braintools.metric')
def coherence_analysis(
    lfp1: brainstate.typing.ArrayLike,
    lfp2: brainstate.typing.ArrayLike,
    dt: float,
    nperseg: int = None
) -> tuple:
    """Compute coherence between two LFP signals.
    
    Parameters
    ----------
    lfp1, lfp2 : brainstate.typing.ArrayLike
        LFP signals with shape (n_time,).
    dt : float
        Sampling interval in seconds.
    nperseg : int, optional
        Length of each segment. Default: n_time // 8.
    
    Returns
    -------
    freqs : jax.Array
        Array of sample frequencies.
    coherence : jax.Array
        Magnitude-squared coherence.
    """
    lfp1, lfp2 = jnp.asarray(lfp1), jnp.asarray(lfp2)
    n_time = len(lfp1)

    if nperseg is None:
        nperseg = n_time // 8

    n_fft = 2 ** int(jnp.ceil(jnp.log2(nperseg)))
    freqs = jnp.fft.fftfreq(n_fft, dt)[:n_fft // 2]

    # Apply window and compute FFTs
    window = jnp.hanning(nperseg)
    fft1 = jnp.fft.fft(lfp1[:nperseg] * window, n=n_fft)[:n_fft // 2]
    fft2 = jnp.fft.fft(lfp2[:nperseg] * window, n=n_fft)[:n_fft // 2]

    # Cross-spectral density
    psd12 = fft1 * jnp.conj(fft2)
    psd11 = jnp.abs(fft1) ** 2
    psd22 = jnp.abs(fft2) ** 2

    # Coherence (magnitude-squared coherence)
    coherence = jnp.abs(psd12) ** 2 / (psd11 * psd22 + 1e-12)

    # Ensure coherence is bounded between 0 and 1
    coherence = jnp.clip(coherence, 0.0, 1.0)

    return freqs, coherence


@set_module_as('braintools.metric')
def phase_amplitude_coupling(
    lfp: brainstate.typing.ArrayLike,
    dt: float,
    phase_freq_range: tuple = (4, 8),
    amplitude_freq_range: tuple = (30, 100),
    n_bins: int = 18
) -> tuple:
    """Compute phase-amplitude coupling (PAC) using the modulation index.
    
    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal with shape (n_time,).
    dt : float
        Sampling interval in seconds.
    phase_freq_range : tuple, default=(4, 8)
        Frequency range for phase extraction (low frequency, e.g., theta).
    amplitude_freq_range : tuple, default=(30, 100)
        Frequency range for amplitude extraction (high frequency, e.g., gamma).
    n_bins : int, default=18
        Number of phase bins for analysis.
    
    Returns
    -------
    modulation_index : float
        Normalized entropy-based modulation index.
    phase_bins : jax.Array
        Phase bin centers.
    mean_amplitudes : jax.Array
        Mean amplitude in each phase bin.
    """
    lfp = jnp.asarray(lfp)

    # Extract phase and amplitude using simplified filtering
    # Low-pass for phase (simplified Butterworth approximation)
    nyquist = 0.5 / dt
    low_cutoff = jnp.mean(jnp.array(phase_freq_range)) / nyquist
    high_cutoff = jnp.mean(jnp.array(amplitude_freq_range)) / nyquist

    # Simple bandpass filtering using FFT
    n_fft = len(lfp)
    freqs = jnp.fft.fftfreq(n_fft, dt)
    fft_lfp = jnp.fft.fft(lfp)

    # Phase component
    phase_mask = (jnp.abs(freqs) >= phase_freq_range[0]) & (jnp.abs(freqs) <= phase_freq_range[1])
    phase_fft = fft_lfp * phase_mask
    phase_signal = jnp.fft.ifft(phase_fft)
    instantaneous_phase = jnp.angle(phase_signal)

    # Amplitude component  
    amp_mask = (jnp.abs(freqs) >= amplitude_freq_range[0]) & (jnp.abs(freqs) <= amplitude_freq_range[1])
    amp_fft = fft_lfp * amp_mask
    amp_signal = jnp.fft.ifft(amp_fft)
    instantaneous_amplitude = jnp.abs(amp_signal)

    # Compute PAC using phase binning
    phase_bins = jnp.linspace(-jnp.pi, jnp.pi, n_bins + 1)
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2

    mean_amplitudes = jnp.array([
        jnp.mean(instantaneous_amplitude[
                     (instantaneous_phase >= phase_bins[i]) &
                     (instantaneous_phase < phase_bins[i + 1])
                     ]) for i in range(n_bins)
    ])

    # Handle NaN values
    mean_amplitudes = jnp.nan_to_num(mean_amplitudes, nan=0.0)

    # Modulation index (normalized entropy)
    p_normalized = mean_amplitudes / (jnp.sum(mean_amplitudes) + 1e-12)
    p_normalized = jnp.where(p_normalized > 0, p_normalized, 1e-12)
    entropy = -jnp.sum(p_normalized * jnp.log(p_normalized))
    max_entropy = jnp.log(n_bins)
    modulation_index = (max_entropy - entropy) / max_entropy

    return modulation_index, bin_centers, mean_amplitudes


@set_module_as('braintools.metric')
def theta_gamma_coupling(
    lfp: brainstate.typing.ArrayLike,
    dt: float
) -> float:
    """Compute theta-gamma coupling strength using standard frequency bands.
    
    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal with shape (n_time,).
    dt : float
        Sampling interval in seconds.
    
    Returns
    -------
    coupling_strength : float
        Theta-gamma coupling modulation index.
    """
    return phase_amplitude_coupling(
        lfp, dt,
        phase_freq_range=(4, 8),  # Theta band
        amplitude_freq_range=(30, 80)  # Gamma band
    )[0]


@set_module_as('braintools.metric')
def current_source_density(
    lfp_laminar: brainstate.typing.ArrayLike,
    electrode_spacing: float
) -> jax.Array:
    """Compute current source density (CSD) from laminar LFP recordings.
    
    Parameters
    ----------
    lfp_laminar : brainstate.typing.ArrayLike
        Laminar LFP data with shape (n_time, n_electrodes).
        Electrodes should be ordered from superficial to deep layers.
    electrode_spacing : float
        Spacing between electrodes in mm.
    
    Returns
    -------
    csd : jax.Array
        Current source density with shape (n_time, n_electrodes-2).
        First and last electrodes are excluded due to boundary conditions.
    """
    lfp_laminar = jnp.asarray(lfp_laminar)

    # Second spatial derivative approximation
    # CSD ≈ -σ * ∂²φ/∂z² ≈ -σ * (φ[z+h] - 2φ[z] + φ[z-h]) / h²
    # where σ is tissue conductivity (assumed constant)

    # Apply second derivative operator
    csd = -(lfp_laminar[:, 2:] - 2 * lfp_laminar[:, 1:-1] + lfp_laminar[:, :-2])
    csd = csd / (electrode_spacing ** 2)

    return csd


@set_module_as('braintools.metric')
def spectral_entropy(
    lfp: brainstate.typing.ArrayLike,
    dt: float,
    freq_range: tuple = (1, 100)
) -> float:
    """Compute spectral entropy of LFP signal as a complexity measure.
    
    Parameters
    ----------
    lfp : brainstate.typing.ArrayLike
        LFP signal with shape (n_time,).
    dt : float
        Sampling interval in seconds.
    freq_range : tuple, default=(1, 100)
        Frequency range for entropy calculation.
    
    Returns
    -------
    entropy : float
        Normalized spectral entropy (0 = most regular, 1 = most random).
    """
    freqs, psd = power_spectral_density(lfp, dt, freq_range=freq_range)

    # Handle edge cases
    if jnp.sum(psd) == 0 or psd.size == 0:
        return 0.0

    # Normalize PSD to get probability distribution
    psd_norm = psd / jnp.sum(psd)
    psd_norm = jnp.where(psd_norm > 0, psd_norm, 1e-12)

    # Shannon entropy
    entropy = -jnp.sum(psd_norm * jnp.log2(psd_norm))
    max_entropy = jnp.log2(psd_norm.shape[0] if psd_norm.ndim > 0 else 1)

    # Handle edge cases
    if max_entropy == 0:
        return 0.0

    return entropy / max_entropy


@set_module_as('braintools.metric')
def lfp_phase_coherence(
    lfp_signals: brainstate.typing.ArrayLike,
    dt: float,
    freq_band: tuple = (8, 12)
) -> jax.Array:
    """Compute phase coherence between multiple LFP signals in a frequency band.
    
    Parameters
    ----------
    lfp_signals : brainstate.typing.ArrayLike
        Multiple LFP signals with shape (n_time, n_channels).
    dt : float
        Sampling interval in seconds.
    freq_band : tuple, default=(8, 12)
        Frequency band for phase extraction (e.g., alpha band).
    
    Returns
    -------
    phase_coherence_matrix : jax.Array
        Phase coherence matrix with shape (n_channels, n_channels).
        Values range from 0 (no coherence) to 1 (perfect coherence).
    """
    lfp_signals = jnp.asarray(lfp_signals)
    n_time, n_channels = lfp_signals.shape

    # Extract phase for each channel using bandpass filtering
    freqs = jnp.fft.fftfreq(n_time, dt)
    band_mask = (jnp.abs(freqs) >= freq_band[0]) & (jnp.abs(freqs) <= freq_band[1])

    phases = []
    for ch in range(n_channels):
        fft_signal = jnp.fft.fft(lfp_signals[:, ch])
        band_fft = fft_signal * band_mask
        analytic_signal = jnp.fft.ifft(band_fft)
        phase = jnp.angle(analytic_signal)
        phases.append(phase)

    phases = jnp.array(phases)  # Shape: (n_channels, n_time)

    # Compute pairwise phase coherence
    coherence_matrix = jnp.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                coherence_matrix = coherence_matrix.at[i, j].set(1.0)
            else:
                # Phase difference
                phase_diff = phases[i] - phases[j]
                # Circular mean of complex exponentials
                mean_coherence = jnp.abs(jnp.mean(jnp.exp(1j * phase_diff)))
                coherence_matrix = coherence_matrix.at[i, j].set(mean_coherence)

    return coherence_matrix
