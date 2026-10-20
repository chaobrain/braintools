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

from typing import Union

import brainstate
import brainunit as u
import jax
from jax import numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'unitary_LFP',
    'current_source_density',
]


@set_module_as('braintools.analysis.field')
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
    neural population [1]_ [2]_.

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
        >>> lfp = braintools.analysis.field.unitary_LFP(times, exc_spikes, 'exc', seed=42)
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


@set_module_as('braintools.analysis.field')
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
