# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

    >>> import brainstate as bst
    >>> import jax.numpy as jnp
    >>> import braintools as bt
    >>> # Set up simulation parameters
    >>> bst.random.seed(42)
    >>> n_time, n_exc, n_inh = 1000, 100, 25
    >>> dt = 0.1  # ms
    >>> times = jnp.arange(n_time) * dt
    >>> # Generate sparse random spike trains
    >>> exc_spikes = (bst.random.random((n_time, n_exc)) < 0.02).astype(float)
    >>> inh_spikes = (bst.random.random((n_time, n_inh)) < 0.04).astype(float)
    >>> # Calculate LFP components
    >>> lfp_exc = bt.metric.unitary_LFP(times, exc_spikes, 'exc', seed=42)
    >>> lfp_inh = bt.metric.unitary_LFP(times, inh_spikes, 'inh', seed=42)
    >>> total_lfp = lfp_exc + lfp_inh
    >>> print(f"LFP shape: {total_lfp.shape}")
    >>> print(f"LFP range: {total_lfp.min():.3f} to {total_lfp.max():.3f}")

    Compare different recording locations:

    >>> # Same spike data, different recording depths
    >>> lfp_soma = bt.metric.unitary_LFP(times, exc_spikes, 'exc', 
    ...                                  location='soma layer')
    >>> lfp_deep = bt.metric.unitary_LFP(times, exc_spikes, 'exc', 
    ...                                  location='deep layer')
    >>> lfp_surface = bt.metric.unitary_LFP(times, exc_spikes, 'exc', 
    ...                                      location='surface')

    Analyze the effect of spatial parameters:

    >>> # Larger population area
    >>> lfp_large = bt.metric.unitary_LFP(times, exc_spikes, 'exc',
    ...                                   xmax=0.5, ymax=0.5)
    >>> # Faster conduction velocity
    >>> lfp_fast = bt.metric.unitary_LFP(times, exc_spikes, 'exc', va=500.0)

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
