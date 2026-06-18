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
import jax.numpy as jnp
import numpy as onp

from braintools._misc import set_module_as

__all__ = [
    'raster_plot',
    'firing_rate',
    'victor_purpura_distance',
    'van_rossum_distance',
    'spike_train_synchrony',
    'burst_synchrony_index',
    'phase_locking_value',
    'spike_time_tiling_coefficient',
    'correlation_index',
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
    >>> import braintools as braintools
    >>> # Create sample spike data (3 neurons, 10 time steps)
    >>> spikes = np.array([
    ...     [0, 1, 0],  # t=0: neuron 1 spikes
    ...     [1, 0, 0],  # t=1: neuron 0 spikes  
    ...     [0, 0, 1],  # t=2: neuron 2 spikes
    ...     [0, 1, 1],  # t=3: neurons 1,2 spike
    ...     [0, 0, 0],  # t=4: no spikes
    ... ])
    >>> times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # Time in seconds
    >>> neuron_ids, spike_times = braintools.metric.raster_plot(spikes, times)
    >>> print("Neuron indices:", neuron_ids)
    >>> print("Spike times:", spike_times)

    Use the results for matplotlib visualization:

    >>> import matplotlib.pyplot as plt
    >>> neuron_ids, spike_times = braintools.metric.raster_plot(spikes, times)
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
    # Locate spikes from the (unit-stripped) matrix, but index ``times`` directly so
    # that ``brainunit.Quantity`` time axes keep their units in the returned array.
    elements = onp.where(onp.asarray(u.get_magnitude(sp_matrix)) > 0.)
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
        consistent with ``dt`` (assumed **seconds** when both are plain floats,
        so the rate comes out in Hz). If a brainunit.Quantity, should have time
        dimensions (e.g., milliseconds). Larger values produce more smoothing.
    dt : float or brainunit.Quantity, optional
        Time step between successive samples in the spike matrix. If None,
        uses the default time step from the brainstate environment
        (``brainstate.environ.get_dt()``). A plain float is assumed to be in
        seconds.

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
    >>> import braintools as braintools
    >>> # Create sample spike data (100 time steps, 50 neurons)
    >>> np.random.seed(42)
    >>> spikes = (np.random.random((100, 50)) < 0.1).astype(float)
    >>> dt = 0.1 * u.ms  # 0.1 ms time steps
    >>> window_width = 5 * u.ms  # 5 ms smoothing window
    >>> rates = braintools.metric.firing_rate(spikes, window_width, dt)
    >>> print(f"Rate shape: {rates.shape}")
    >>> print(f"Mean rate: {np.mean(rates):.2f} Hz")

    Compare different smoothing window sizes:

    >>> narrow_rates = braintools.metric.firing_rate(spikes, 2*u.ms, dt)
    >>> wide_rates = braintools.metric.firing_rate(spikes, 10*u.ms, dt)
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

    The result is always expressed in **Hz**. When ``width``/``dt`` are
    :class:`brainunit.Quantity` objects the conversion is exact for any time unit.
    When they are plain floats they are assumed to be in **seconds**, so the
    returned rate is ``mean_spike_fraction / window_duration_seconds``. Pass
    ``width`` and ``dt`` in the *same* time unit.

    ``width`` and ``dt`` must be concrete (static) values: the window length is
    computed with Python ``int``, so this function is usable inside ``jit`` only
    when ``width``/``dt`` are static.

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
    # Number of samples in the (odd-sized) smoothing window. ``width / dt`` must be
    # dimensionless; ``u.get_magnitude`` strips the unit when both are Quantities.
    width1 = int(u.get_magnitude(width / 2 / dt)) * 2 + 1
    # Normalize by the *realized* window duration (``width1 * dt``) rather than the
    # nominal ``width`` so the kernel integrates spike counts to a rate over exactly
    # the samples that are used.
    duration = width1 * dt
    inv_duration = 1.0 / duration
    if isinstance(inv_duration, u.Quantity):
        # Quantity time step -> express the rate in Hz (spikes per second).
        inv_duration = inv_duration.to_decimal(u.Hz)
    # Float ``dt``/``width`` are taken to be in seconds, so ``1 / duration`` is Hz.
    window = jnp.ones(width1) * inv_duration
    rate = jnp.mean(jnp.asarray(u.get_magnitude(spikes)), axis=1)
    return jnp.convolve(rate, window, mode='same')


@set_module_as('braintools.metric')
def victor_purpura_distance(
    spike_times_1: brainstate.typing.ArrayLike,
    spike_times_2: brainstate.typing.ArrayLike,
    cost_factor: float = 1.0
):
    r"""Calculate Victor-Purpura distance between two spike trains.
    
    The Victor-Purpura distance quantifies the dissimilarity between two spike trains
    by computing the minimum cost to transform one spike train into another through
    spike insertions, deletions, and temporal shifts.
    
    The distance is computed as:
    
    .. math::
    
        D_{VP} = \min \sum_{ops} c_{op}
        
    where the cost of moving a spike by time :math:`\Delta t` is :math:`q|\Delta t|`,
    insertion/deletion costs are 1, and :math:`q` is the cost factor.
    
    Parameters
    ----------
    spike_times_1 : brainstate.typing.ArrayLike
        First spike train as array of spike times.
    spike_times_2 : brainstate.typing.ArrayLike
        Second spike train as array of spike times.
    cost_factor : float, default=1.0
        Cost factor :math:`q` for temporal shifts. Higher values penalize 
        temporal differences more heavily.
    
    Returns
    -------
    float
        Victor-Purpura distance between the two spike trains.

    Notes
    -----
    This function runs on host (concrete) arrays: it fills a dynamic-programming
    table with Python loops and ``len`` and returns a Python ``float``, so it is not
    ``jit``/``vmap``/``grad``-compatible. ``cost_factor`` and the spike times must
    use consistent units (the distance has units of ``cost_factor * time + count``).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes1 = jnp.array([1.0, 2.0, 3.0])
        >>> spikes2 = jnp.array([1.1, 2.1, 3.1])
        >>> d = braintools.metric.victor_purpura_distance(spikes1, spikes2, cost_factor=10.0)
        >>> bool(d >= 0.0)
        True

    References
    ----------
    .. [1] Victor, Jonathan D., and Keith P. Purpura. "Nature and precision of 
           temporal coding in visual cortex: a metric-space analysis." 
           Journal of neurophysiology 76.2 (1996): 1310-1326.
    """
    spikes1 = onp.asarray(u.get_magnitude(spike_times_1), dtype=float)
    spikes2 = onp.asarray(u.get_magnitude(spike_times_2), dtype=float)
    cost_factor = float(u.get_magnitude(cost_factor))

    n1, n2 = spikes1.shape[0], spikes2.shape[0]

    # Handle empty spike trains
    if n1 == 0:
        return float(n2)
    if n2 == 0:
        return float(n1)

    # Dynamic programming on host (NumPy) arrays with plain Python arithmetic.
    # dp[i, j] = minimum cost to transform spikes1[:i] into spikes2[:j].
    # Using JAX arrays here would force a host<->device round-trip for every
    # ``.at[].set`` and ``dp[...]`` read, making the routine O(n^2) *device*
    # operations; NumPy keeps it a fast in-memory fill.
    dp = onp.empty((n1 + 1, n2 + 1), dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n1 + 1):
        dp[i, 0] = i  # Delete all spikes in train 1
    for j in range(1, n2 + 1):
        dp[0, j] = j  # Insert all spikes in train 2

    for i in range(1, n1 + 1):
        s1 = spikes1[i - 1]
        for j in range(1, n2 + 1):
            # Match spike i with spike j (temporal shift), delete, or insert.
            match_cost = dp[i - 1, j - 1] + cost_factor * abs(s1 - spikes2[j - 1])
            delete_cost = dp[i - 1, j] + 1.0
            insert_cost = dp[i, j - 1] + 1.0
            dp[i, j] = min(match_cost, delete_cost, insert_cost)

    return float(dp[n1, n2])


@set_module_as('braintools.metric')
def van_rossum_distance(
    spike_times_1: brainstate.typing.ArrayLike,
    spike_times_2: brainstate.typing.ArrayLike,
    tau: float = 1.0,
    t_max: float = None
):
    r"""Calculate van Rossum distance between two spike trains.
    
    The van Rossum distance measures dissimilarity between spike trains by
    convolving each with an exponential kernel and computing the Euclidean
    distance between the resulting continuous functions.
    
    Each spike train is convolved with kernel :math:`K(t) = \frac{1}{\tau}e^{-t/\tau}H(t)`
    where :math:`H(t)` is the Heaviside step function. The distance is:
    
    .. math::
    
        D_{vR} = \sqrt{\int_0^{T} [f_1(t) - f_2(t)]^2 dt}
        
    where :math:`f_i(t)` is the convolved spike train.
    
    Parameters
    ----------
    spike_times_1 : brainstate.typing.ArrayLike
        First spike train as array of spike times.
    spike_times_2 : brainstate.typing.ArrayLike
        Second spike train as array of spike times.
    tau : float, default=1.0
        Time constant of the exponential kernel. Larger values emphasize
        longer-term dependencies.
    t_max : float, optional
        Maximum time to consider. If None, uses maximum spike time + 5*tau.
        
    Returns
    -------
    float
        van Rossum distance between the two spike trains.

    Notes
    -----
    Convention: each spike train is convolved with the *rate-normalized* kernel
    :math:`K(t) = \frac{1}{\tau} e^{-t/\tau} H(t)` and the distance is
    :math:`\sqrt{\int (f_1 - f_2)^2\, dt}` (no extra :math:`1/\tau` prefactor on the
    integral). Other texts use :math:`K(t) = e^{-t/\tau}` with a :math:`1/\tau`
    integral prefactor; both differ only by an overall scale. The integral is
    truncated at ``t_max`` (default ``max(spike_time) + 5*tau``) and discretized at
    ``dt = tau / 20``; the ``5*tau`` tail captures >99% of the exponential.

    This function runs on host (concrete) arrays (Python loop over spikes, ``len``)
    and returns a Python ``float``, so it is not ``jit``/``vmap``/``grad``-compatible.
    ``tau``, ``t_max`` and the spike times must use consistent time units.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes1 = jnp.array([1.0, 3.0, 5.0])
        >>> spikes2 = jnp.array([1.2, 3.2, 5.2])
        >>> d = braintools.metric.van_rossum_distance(spikes1, spikes2, tau=0.5)
        >>> bool(d >= 0.0)
        True

    References
    ----------
    .. [1] van Rossum, Mark CW. "A novel spike distance." 
           Neural computation 13.4 (2001): 751-763.
    """
    spikes1 = jnp.asarray(spike_times_1)
    spikes2 = jnp.asarray(spike_times_2)

    # Determine time window
    if t_max is None:
        all_spikes = jnp.concatenate([spikes1, spikes2])
        if len(all_spikes) == 0:
            return 0.0
        t_max = jnp.max(all_spikes) + 5 * tau

    # Create time grid
    dt = tau / 20  # Fine temporal resolution
    t_grid = jnp.arange(0, t_max + dt, dt)

    def convolve_spikes(spike_times):
        """Convolve spike train with exponential kernel."""
        if len(spike_times) == 0:
            return jnp.zeros_like(t_grid)

        # For each time point, sum contributions from all spikes
        response = jnp.zeros_like(t_grid)
        for spike_time in spike_times:
            # Exponential kernel starting from spike time
            mask = t_grid >= spike_time
            kernel = jnp.where(mask,
                               (1.0 / tau) * jnp.exp(-(t_grid - spike_time) / tau),
                               0.0)
            response = response + kernel
        return response

    # Convolve both spike trains
    f1 = convolve_spikes(spikes1)
    f2 = convolve_spikes(spikes2)

    # Compute Euclidean distance
    diff = f1 - f2
    distance_squared = jnp.sum(diff ** 2) * dt

    return float(jnp.sqrt(distance_squared))


@set_module_as('braintools.metric')
def spike_train_synchrony(
    spike_matrix: brainstate.typing.ArrayLike,
    window_size: float,
    dt: float = None
):
    r"""Calculate spike train synchrony using the SPIKE-synchronization measure.
    
    This measure quantifies the degree of synchronization between multiple spike trains
    by counting coincident events within sliding time windows and normalizing by the
    total number of possible coincidences.
    
    The synchrony index is the average over neuron pairs of a symmetric
    coincidence ratio:

    .. math::

        S = \frac{1}{N_{pairs}} \sum_{i < j} \frac{C_{i \to j} + C_{j \to i}}{N_i + N_j}

    where :math:`C_{i \to j}` is the number of spikes in train :math:`i` that have at
    least one spike of train :math:`j` within half the coincidence window, and
    :math:`N_i` is the number of spikes in train :math:`i`. Counting coincidences in
    both directions and normalizing by the total spike count guarantees
    :math:`S \in [0, 1]`.

    Parameters
    ----------
    spike_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)`` where non-zero values
        indicate spike occurrences.
    window_size : float or brainunit.Quantity
        Full width of the coincidence-detection window. Two spikes are coincident
        when they are at most ``window_size / 2`` apart. Must use the same time unit
        as ``dt``.
    dt : float or brainunit.Quantity, optional
        Time step between successive samples. If None, uses brainstate default.

    Returns
    -------
    float
        Spike train synchrony index between 0 (no synchrony) and 1 (perfect synchrony).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Create synchronized spikes
    >>> spikes = jnp.zeros((100, 5))
    >>> spikes = spikes.at[20:25, :].set(1)  # Synchronized burst
    >>> synchrony = braintools.metric.spike_train_synchrony(spikes, window_size=10.0)
    >>> print(f"Synchrony: {synchrony:.3f}")
    
    References
    ----------
    .. [1] Kreuz, Thomas, et al. "Measuring spike train synchrony." 
           Journal of neuroscience methods 165.1 (2007): 151-161.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = onp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape

    if n_neurons < 2:
        return 0.0

    # Half-coincidence window in samples. ``window_size`` and ``dt`` must share the
    # same time unit; only their magnitudes are compared.
    half_steps = u.get_magnitude(window_size) / (2.0 * u.get_magnitude(dt))

    # Spike sample indices for each neuron.
    spike_idx_list = [onp.where(spikes[:, i] > 0)[0] for i in range(n_neurons)]

    total_synchrony = 0.0
    n_pairs = 0
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            idx_i = spike_idx_list[i]
            idx_j = spike_idx_list[j]
            n_i, n_j = len(idx_i), len(idx_j)
            if n_i == 0 or n_j == 0:
                continue

            # A spike is "coincident" if the *other* train has any spike within the
            # half-window. Counting coincidences symmetrically in both directions and
            # normalizing by the total spike count bounds the ratio to [0, 1] (the
            # earlier ``min(N_i, N_j)`` normalization with one-directional counting
            # could exceed 1).
            within = onp.abs(idx_i[:, None] - idx_j[None, :]) <= half_steps
            coin_i = int(onp.any(within, axis=1).sum())
            coin_j = int(onp.any(within, axis=0).sum())
            total_synchrony += (coin_i + coin_j) / (n_i + n_j)
            n_pairs += 1

    return float(total_synchrony / n_pairs) if n_pairs > 0 else 0.0


@set_module_as('braintools.metric')
def burst_synchrony_index(
    spike_matrix: brainstate.typing.ArrayLike,
    burst_threshold: int = 3,
    max_isi: float = 100.0,
    dt: float = None
):
    r"""Calculate burst synchrony index based on co-occurring burst events.
    
    This measure identifies burst events in each spike train and quantifies
    the synchronization of these bursts across the population.
    
    A burst is defined as a sequence of at least ``burst_threshold`` spikes
    with inter-spike intervals ≤ ``max_isi``.
    
    Parameters
    ----------
    spike_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)``.
    burst_threshold : int, default=3
        Minimum number of spikes required to constitute a burst.
    max_isi : float or brainunit.Quantity, default=100.0
        Maximum inter-spike interval within a burst. Must use the same time unit
        as ``dt``.
    dt : float or brainunit.Quantity, optional
        Time step between successive samples. If None, uses brainstate default.

    Returns
    -------
    float
        Burst synchrony index between 0 (no burst synchrony) and 1 (perfect burst synchrony).

    Notes
    -----
    This function runs on host (concrete) arrays (Python loops, ``len``), so it is
    not ``jit``/``vmap``/``grad``-compatible.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes = jnp.zeros((1000, 10))
        >>> for start in [100, 300, 600]:
        ...     for i in range(10):
        ...         spikes = spikes.at[start:start + 5, i].set(1)
        >>> sync_idx = braintools.metric.burst_synchrony_index(spikes, max_isi=5.0, dt=1.0)
        >>> bool(0.0 <= sync_idx <= 1.0)
        True
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = onp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape
    dt_m = u.get_magnitude(dt)
    max_isi_m = u.get_magnitude(max_isi)

    def detect_bursts(spike_train):
        """Detect burst events in a single spike train."""
        spike_times = onp.where(spike_train > 0)[0] * dt_m
        if len(spike_times) < burst_threshold:
            return []

        bursts = []
        current_burst = [spike_times[0]]

        for i in range(1, len(spike_times)):
            isi = spike_times[i] - spike_times[i - 1]
            if isi <= max_isi_m:
                current_burst.append(spike_times[i])
            else:
                if len(current_burst) >= burst_threshold:
                    bursts.append((current_burst[0], current_burst[-1]))
                current_burst = [spike_times[i]]

        # Check final burst
        if len(current_burst) >= burst_threshold:
            bursts.append((current_burst[0], current_burst[-1]))

        return bursts

    # Detect bursts for all neurons
    all_bursts = []
    for i in range(n_neurons):
        bursts = detect_bursts(spikes[:, i])
        for start, end in bursts:
            all_bursts.append((i, start, end))

    if len(all_bursts) == 0:
        return 0.0

    # Count synchronized bursts
    synchronous_bursts = 0
    total_bursts = len(all_bursts)

    for i, (neuron1, start1, end1) in enumerate(all_bursts):
        overlapping_neurons = {neuron1}

        for j, (neuron2, start2, end2) in enumerate(all_bursts):
            if i != j and neuron1 != neuron2:
                # Check for temporal overlap. ``>= 0`` (inclusive) counts bursts
                # that touch at a single instant -- the strongest possible
                # synchrony -- which a strict ``> 0`` would wrongly exclude.
                overlap = min(end1, end2) - max(start1, start2)
                if overlap >= 0:
                    overlapping_neurons.add(neuron2)

        # If burst involves multiple neurons, it's synchronous
        if len(overlapping_neurons) > 1:
            synchronous_bursts += 1

    return float(synchronous_bursts / total_bursts) if total_bursts > 0 else 0.0


@set_module_as('braintools.metric')
def phase_locking_value(
    spike_matrix: brainstate.typing.ArrayLike,
    reference_freq: float,
    dt: Union[float, u.Quantity] = None
):
    r"""Calculate phase-locking value (PLV) for spike synchronization.
    
    The PLV measures the consistency of phase relationships between spike trains
    and a reference oscillation, indicating rhythmic synchronization.
    
    For each spike, the phase relative to the reference oscillation is computed,
    and the PLV is the magnitude of the mean resultant vector:
    
    .. math::
    
        PLV = \left|\frac{1}{N}\sum_{k=1}^{N} e^{i\phi_k}\right|
        
    where :math:`\phi_k` is the phase of the k-th spike.
    
    Parameters
    ----------
    spike_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)``.
    reference_freq : float
        Reference frequency for phase computation (in Hz).
    dt : float or brainunit.Quantity, optional
        Time step between successive samples. If None, uses the brainstate
        default. A plain float is assumed to be in **seconds** (so that it is
        consistent with ``reference_freq`` in Hz); a ``brainunit.Quantity`` is
        converted to seconds.
        
    Returns
    -------
    jnp.ndarray
        Phase-locking values for each neuron. Shape ``(n_neurons,)``.
        Values range from 0 (no phase locking) to 1 (perfect phase locking).

    Notes
    -----
    This computes the **vector strength** (Rayleigh resultant length) of each
    neuron's spikes relative to an *external* reference oscillation of frequency
    ``reference_freq``. It is therefore a spike–field locking measure, **not** the
    Lachaux et al. (1999) pairwise PLV between two continuous signals.

    Vector strength is biased upward for small spike counts: a neuron with a single
    spike trivially yields ``PLV = 1``, and a handful of spikes can give a large
    value by chance. Interpret values with caution when spike counts are low, and
    consider a Rayleigh-style bias correction or a minimum-spike threshold.

    This function runs on host (concrete) arrays (Python loop over neurons,
    boolean-mask indexing), so it is not ``jit``/``vmap``/``grad``-compatible.

    References
    ----------
    .. [1] Lachaux, Jean-Philippe, et al. "Measuring phase synchrony in brain
           signals." Human brain mapping 8.4 (1999): 194-208.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> n_time, n_neurons = 1000, 5
        >>> spikes = jnp.zeros((n_time, n_neurons))
        >>> freq, dt = 10.0, 0.001  # 10 Hz reference, 1 ms step
        >>> # Place one spike per 10 Hz cycle for neuron 0 (strong locking).
        >>> idx = (jnp.arange(10) / freq / dt).astype(int)
        >>> spikes = spikes.at[idx, 0].set(1.0)
        >>> plv = braintools.metric.phase_locking_value(spikes, freq, dt)
        >>> plv.shape
        (5,)
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    # ``reference_freq`` is in Hz, so the time base must be in seconds. A
    # ``brainunit.Quantity`` ``dt`` (e.g. the ``brainstate.environ.get_dt()``
    # default, or ``0.1 * u.ms``) is converted to seconds; a plain float is
    # assumed to already be in seconds.
    if isinstance(dt, u.Quantity):
        dt = float(dt.to_decimal(u.second))
    spikes = jnp.asarray(spike_matrix)
    n_time, n_neurons = spikes.shape

    # Create time vector (in seconds)
    times = jnp.arange(n_time) * dt

    # Reference phase signal
    reference_phase = 2 * jnp.pi * reference_freq * times

    plv_values = jnp.zeros(n_neurons)

    for i in range(n_neurons):
        spike_indices = jnp.where(spikes[:, i] > 0)[0]

        if len(spike_indices) == 0:
            plv_values = plv_values.at[i].set(0.0)
            continue

        # Get phases at spike times
        spike_phases = reference_phase[spike_indices]

        # Compute mean resultant vector
        complex_phases = jnp.exp(1j * spike_phases)
        mean_vector = jnp.mean(complex_phases)
        plv = jnp.abs(mean_vector)

        plv_values = plv_values.at[i].set(plv)

    return plv_values


@set_module_as('braintools.metric')
def spike_time_tiling_coefficient(
    spike_matrix: brainstate.typing.ArrayLike,
    dt: float = None,
    tau: float = 0.005
):
    r"""Calculate Spike Time Tiling Coefficient (STTC).
    
    STTC measures synchrony between spike trains while controlling for firing rate
    differences. It's based on the proportion of spikes that fall within a temporal
    window around spikes in the other train.
    
    The STTC is computed as:
    
    .. math::
    
        STTC = \frac{1}{2}\left(\frac{P_A - T_B}{1 - P_A T_B} + \frac{P_B - T_A}{1 - P_B T_A}\right)
        
    where :math:`P_A` is the proportion of spikes in train A that have a spike from
    train B within time :math:`\tau`, and :math:`T_A` is the proportion of total
    time covered by windows around spikes in train A.
    
    Parameters
    ----------
    spike_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)``.
    dt : float or brainunit.Quantity, optional
        Time step between successive samples. If None, uses brainstate default.
    tau : float or brainunit.Quantity, default=0.005
        Half-width of the temporal window for coincidence detection. Must use the
        same time unit as ``dt`` (default assumes seconds).

    Returns
    -------
    jnp.ndarray
        STTC matrix with shape ``(n_neurons, n_neurons)``. Diagonal elements are 1.
        Values range from -1 to 1, where 1 indicates perfect synchrony.

    Notes
    -----
    The time-coverage terms :math:`T_A`, :math:`T_B` use the **union** of the
    per-spike :math:`[\,s - \tau,\ s + \tau\,]` windows, so overlapping windows are
    not double-counted (the defining quantity in Cutts & Eglen, 2014). Each STTC
    term is guarded independently against a zero denominator.

    This function runs on host (concrete) arrays (Python loops, ``len``,
    boolean-mask indexing), so it is not ``jit``/``vmap``/``grad``-compatible.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes = jnp.zeros((1000, 3))
        >>> sync_times = [100, 300, 500, 700]
        >>> for t in sync_times:
        ...     spikes = spikes.at[t:t + 3, :].set(1)
        >>> sttc = braintools.metric.spike_time_tiling_coefficient(spikes, dt=1.0, tau=2.0)
        >>> sttc.shape
        (3, 3)
    
    References
    ----------
    .. [1] Cutts, Catherine S., and Stephen J. Eglen. "Detecting pairwise correlations 
           in spike trains: an objective comparison of methods and application to the 
           retina." Journal of Neuroscience 34.43 (2014): 14288-14303.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = onp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape

    # Coincidence half-window in samples. ``tau`` and ``dt`` must share a time unit;
    # only their magnitudes are used.
    tau_steps = int(round(u.get_magnitude(tau) / u.get_magnitude(dt)))

    def covered_fraction(idx):
        # Fraction of the recording covered by the *union* of [s - tau, s + tau]
        # windows around each spike. Summing per-spike window lengths (the previous
        # behaviour) double-counts overlapping windows and inflates T; the union is
        # the quantity defined by Cutts & Eglen (2014).
        covered = onp.zeros(n_time, dtype=bool)
        for s in idx:
            lo = max(0, s - tau_steps)
            hi = min(n_time, s + tau_steps + 1)
            covered[lo:hi] = True
        return covered.sum() / n_time

    sttc_matrix = onp.eye(n_neurons)  # Diagonal is 1 by definition
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            idx_i = onp.where(spikes[:, i] > 0)[0]
            idx_j = onp.where(spikes[:, j] > 0)[0]
            n_i, n_j = len(idx_i), len(idx_j)

            if n_i == 0 or n_j == 0:
                sttc_matrix[i, j] = sttc_matrix[j, i] = 0.0
                continue

            within = onp.abs(idx_i[:, None] - idx_j[None, :]) <= tau_steps
            P_A = onp.any(within, axis=1).sum() / n_i  # spikes in i near a j spike
            P_B = onp.any(within, axis=0).sum() / n_j  # spikes in j near an i spike
            T_A = covered_fraction(idx_i)
            T_B = covered_fraction(idx_j)

            # Guard each term independently: a single degenerate denominator should
            # zero only its own term, not the whole coefficient.
            term1 = (P_A - T_B) / (1.0 - P_A * T_B) if (1.0 - P_A * T_B) != 0 else 0.0
            term2 = (P_B - T_A) / (1.0 - P_B * T_A) if (1.0 - P_B * T_A) != 0 else 0.0
            sttc_value = 0.5 * (term1 + term2)

            sttc_matrix[i, j] = sttc_matrix[j, i] = sttc_value

    return jnp.asarray(sttc_matrix)


@set_module_as('braintools.metric')
def correlation_index(
    spike_matrix: brainstate.typing.ArrayLike,
    window_size: float,
    dt: float = None
):
    r"""Calculate correlation index for spike train synchrony.
    
    The correlation index measures the strength of pairwise correlations in spike
    trains by computing the average Pearson correlation coefficient between binned
    spike counts.

    The index is computed as:

    .. math::

        CI = \frac{1}{N(N-1)} \sum_{i \neq j} \rho_{ij}

    where :math:`\rho_{ij}` is the Pearson correlation coefficient between
    the binned spike counts of neurons i and j.

    Parameters
    ----------
    spike_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)``.
    window_size : float or brainunit.Quantity
        Size of the time windows used to bin spikes. Must use the same time unit
        as ``dt``.
    dt : float or brainunit.Quantity, optional
        Time step between successive samples. If None, uses brainstate default.

    Returns
    -------
    float
        Correlation index representing the average pairwise correlation.
        Values range from -1 to 1, where positive values indicate synchrony.

    Notes
    -----
    This is the **mean pairwise Pearson correlation** of binned spike counts. It is
    distinct from the Wong–Meister/Mastronarde "correlation index", which is a
    coincidence-rate ratio (expected-vs-observed near-coincident firings) with a
    different range and interpretation. Use this function when a normalized
    [-1, 1] linear-correlation summary is desired.

    This function runs on host (concrete) arrays — it uses Python loops and
    :func:`numpy.corrcoef`, so it is not ``jit``/``vmap``/``grad``-compatible.

    References
    ----------
    .. [1] Pearson, Karl. "Notes on regression and inheritance in the case of two
           parents." Proceedings of the Royal Society of London 58 (1895): 240-242.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import braintools
        >>> # Two correlated trains and one independent train.
        >>> base = (brainstate.random.rand(1000, 1) < 0.1).astype(float)
        >>> noise = (brainstate.random.rand(1000, 1) < 0.1).astype(float)
        >>> third = (brainstate.random.rand(1000, 1) < 0.1).astype(float)
        >>> import jax.numpy as jnp
        >>> spikes = jnp.concatenate([base, jnp.clip(base + noise, 0, 1), third], axis=1)
        >>> ci = braintools.metric.correlation_index(spikes, window_size=50.0, dt=1.0)
        >>> bool(-1.0 <= ci <= 1.0)
        True
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape

    if n_neurons < 2:
        return 0.0

    # Bin size in samples (``window_size`` and ``dt`` must share a time unit).
    bin_size = max(1, int(u.get_magnitude(window_size) / u.get_magnitude(dt)))
    # Only whole bins are used; any trailing samples that do not fill a complete
    # bin are discarded so every bin has the same width.
    n_bins = n_time // bin_size

    if n_bins < 2:
        return 0.0

    # Bin spike counts
    binned_spikes = jnp.zeros((n_bins, n_neurons))
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        binned_spikes = binned_spikes.at[i, :].set(jnp.sum(spikes[start_idx:end_idx, :], axis=0))

    # Calculate pairwise correlations
    correlations = []
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            # Compute Pearson correlation
            x, y = binned_spikes[:, i], binned_spikes[:, j]

            # Handle case where one or both series have zero variance
            x_var = jnp.var(x)
            y_var = jnp.var(y)

            if x_var == 0 or y_var == 0:
                corr = 0.0
            else:
                corr = jnp.corrcoef(x, y)[0, 1]
                # Handle NaN case
                corr = jnp.where(jnp.isnan(corr), 0.0, corr)

            correlations.append(corr)

    # Clamp to the documented [-1, 1] range (guards against float round-off in
    # ``corrcoef`` pushing the mean marginally outside the interval).
    return float(jnp.clip(jnp.mean(jnp.array(correlations)), -1.0, 1.0))
