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

from typing import Union

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as onp

from braintools._misc import set_module_as

__all__ = [
    'raster_plot',
    'firing_rate',
]


@set_module_as('braintools.metric')
def raster_plot(
    sp_matrix: brainstate.typing.ArrayLike,
    times: brainstate.typing.ArrayLike
):
    """Extract spike times and neuron indices for raster plot visualization.

    A raster plot displays the spiking activity of a population of neurons over time,
    where each row represents a neuron and each dot or line indicates a spike occurrence.
    This function extracts the necessary data (neuron indices and corresponding spike
    times) from a spike matrix to create such visualizations.

    Parameters
    ----------
    sp_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)`` where non-zero values
        indicate spike occurrences. Each element ``sp_matrix[t, i]`` represents the
        spike activity of neuron ``i`` at time step ``t``.
    times : brainstate.typing.ArrayLike
        Time points corresponding to each row of the spike matrix with shape
        ``(n_time_steps,)``. These represent the actual time values for each
        time step in the simulation.

    Returns
    -------
    neuron_indices : numpy.ndarray
        Array of neuron indices where spikes occurred. Each index corresponds
        to a neuron that fired at the corresponding time in ``spike_times``.
    spike_times : numpy.ndarray
        Array of spike times corresponding to each spike event. These are the
        actual time values when spikes occurred, extracted from the ``times`` array.

    Examples
    --------
    Create a simple spike matrix and extract raster data:

    >>> import numpy as np
    >>> import braintools as bt
    >>> # Create sample spike data (3 neurons, 10 time steps)
    >>> spikes = np.array([
    ...     [0, 1, 0],  # t=0: neuron 1 spikes
    ...     [1, 0, 0],  # t=1: neuron 0 spikes  
    ...     [0, 0, 1],  # t=2: neuron 2 spikes
    ...     [0, 1, 1],  # t=3: neurons 1,2 spike
    ...     [0, 0, 0],  # t=4: no spikes
    ... ])
    >>> times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # Time in seconds
    >>> neuron_ids, spike_times = bt.metric.raster_plot(spikes, times)
    >>> print("Neuron indices:", neuron_ids)
    >>> print("Spike times:", spike_times)

    Use the results for matplotlib visualization:

    >>> import matplotlib.pyplot as plt
    >>> neuron_ids, spike_times = bt.metric.raster_plot(spikes, times)
    >>> plt.scatter(spike_times, neuron_ids, marker='|', s=50)
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Neuron Index')
    >>> plt.title('Raster Plot')
    >>> plt.show()

    Notes
    -----
    The function uses ``numpy.where`` to find non-zero elements in the spike matrix,
    making it efficient for sparse spike data. The returned arrays have the same
    length and can be directly used for scatter plots or other visualizations.

    See Also
    --------
    braintools.metric.firing_rate : Calculate population firing rates
    matplotlib.pyplot.scatter : For creating raster plot visualizations
    """
    times = onp.asarray(times)
    elements = onp.where(sp_matrix > 0.)
    index = elements[1]
    time = times[elements[0]]
    return index, time


@set_module_as('braintools.metric')
def firing_rate(
    spikes: brainstate.typing.ArrayLike,
    width: Union[float, u.Quantity],
    dt: Union[float, u.Quantity] = None
):
    r"""Calculate the smoothed population firing rate from spike data.

    Computes the time-varying population firing rate by averaging spike counts
    across neurons and applying temporal smoothing with a rectangular window.
    This provides a measure of the overall activity level of the neural population
    over time.

    The instantaneous firing rate at time :math:`t` is calculated as:

    .. math::

        r(t) = \frac{1}{N} \sum_{i=1}^{N} s_i(t)

    where :math:`N` is the number of neurons and :math:`s_i(t)` is the spike
    indicator for neuron :math:`i` at time :math:`t`. The rate is then smoothed
    using a rectangular window:

    .. math::

        \bar{r}(t) = \frac{1}{T} \int_{t-T/2}^{t+T/2} r(\tau) d\tau

    where :math:`T` is the window width.

    Parameters
    ----------
    spikes : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)`` where each element
        indicates spike occurrence (typically 0 or 1). Non-zero values represent
        spikes at the corresponding time step and neuron.
    width : float or brainunit.Quantity
        Width of the smoothing window. If a float, interpreted as time units
        consistent with ``dt``. If a brainunit.Quantity, should have time dimensions
        (e.g., milliseconds). Larger values produce more smoothing.
    dt : float or brainunit.Quantity, optional
        Time step between successive samples in the spike matrix. If None,
        uses the default time step from the brainstate environment
        (``brainstate.environ.get_dt()``).

    Returns
    -------
    numpy.ndarray
        Smoothed population firing rate with shape ``(n_time_steps,)``.
        Values are in Hz (spikes per second) when using appropriate time units.
        The smoothing may introduce edge effects at the beginning and end
        of the time series.

    Examples
    --------
    Calculate firing rate from spike data:

    >>> import numpy as np
    >>> import brainunit as u
    >>> import braintools as bt
    >>> # Create sample spike data (100 time steps, 50 neurons)
    >>> np.random.seed(42)
    >>> spikes = (np.random.random((100, 50)) < 0.1).astype(float)
    >>> dt = 0.1 * u.ms  # 0.1 ms time steps
    >>> window_width = 5 * u.ms  # 5 ms smoothing window
    >>> rates = bt.metric.firing_rate(spikes, window_width, dt)
    >>> print(f"Rate shape: {rates.shape}")
    >>> print(f"Mean rate: {np.mean(rates):.2f} Hz")

    Compare different smoothing window sizes:

    >>> narrow_rates = bt.metric.firing_rate(spikes, 2*u.ms, dt)
    >>> wide_rates = bt.metric.firing_rate(spikes, 10*u.ms, dt)
    >>> # narrow_rates will be more variable, wide_rates more smooth

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> time = np.arange(len(rates)) * float(dt.to_decimal(u.ms))
    >>> plt.plot(time, rates, label='Population rate')
    >>> plt.xlabel('Time (ms)')
    >>> plt.ylabel('Firing rate (Hz)')
    >>> plt.title('Population Firing Rate')
    >>> plt.show()

    Notes
    -----
    This method is adapted from the Brian2 simulator and uses convolution
    with a rectangular window for smoothing. The window size is automatically
    adjusted to be odd-sized for symmetric smoothing.

    Edge effects occur at the beginning and end of the time series due to
    the convolution operation. For critical applications, consider using
    alternative boundary conditions or trimming the results.

    The function converts brainunit.Quantity objects to appropriate numerical
    values when necessary, ensuring compatibility with the JAX computation backend.

    See Also
    --------
    braintools.metric.raster_plot : Extract spike times for visualization
    numpy.convolve : Underlying convolution operation for smoothing
    jax.numpy.mean : Population averaging operation

    References
    ----------
    .. [1] Stimberg, Marcel, Romain Brette, and Dan FM Goodman. 
           "Brian 2, an intuitive and efficient neural simulator." 
           Elife 8 (2019): e47314.
    """
    dt = brainstate.environ.get_dt() if (dt is None) else dt
    width1 = int(width / 2 / dt) * 2 + 1
    window = u.math.ones(width1) / width
    if isinstance(window, u.Quantity):
        window = window.to_decimal(u.Hz)
    return jnp.convolve(jnp.mean(spikes, axis=1), window, mode='same')
