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
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Similar spike trains
    >>> spikes1 = jnp.array([1.0, 2.0, 3.0])
    >>> spikes2 = jnp.array([1.1, 2.1, 3.1])
    >>> distance = braintools.metric.victor_purpura_distance(spikes1, spikes2, cost_factor=10.0)
    >>> print(f"VP distance: {distance:.3f}")
    
    References
    ----------
    .. [1] Victor, Jonathan D., and Keith P. Purpura. "Nature and precision of 
           temporal coding in visual cortex: a metric-space analysis." 
           Journal of neurophysiology 76.2 (1996): 1310-1326.
    """
    spikes1 = jnp.asarray(spike_times_1)
    spikes2 = jnp.asarray(spike_times_2)

    n1, n2 = len(spikes1), len(spikes2)

    # Handle empty spike trains
    if n1 == 0:
        return float(n2)
    if n2 == 0:
        return float(n1)

    # Dynamic programming matrix
    # dp[i][j] = minimum cost to transform spikes1[:i] to spikes2[:j]
    dp = jnp.full((n1 + 1, n2 + 1), jnp.inf)

    # Base cases
    dp = dp.at[0, 0].set(0.0)
    for i in range(1, n1 + 1):
        dp = dp.at[i, 0].set(i)  # Delete all spikes in train 1
    for j in range(1, n2 + 1):
        dp = dp.at[0, j].set(j)  # Insert all spikes in train 2

    # Fill the DP matrix
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            # Cost to match spike i with spike j (temporal shift)
            shift_cost = cost_factor * jnp.abs(spikes1[i - 1] - spikes2[j - 1])
            match_cost = dp[i - 1, j - 1] + shift_cost

            # Cost to delete spike i
            delete_cost = dp[i - 1, j] + 1.0

            # Cost to insert spike j
            insert_cost = dp[i, j - 1] + 1.0

            # Take minimum cost
            dp = dp.at[i, j].set(jnp.min(jnp.array([match_cost, delete_cost, insert_cost])))

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
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> spikes1 = jnp.array([1.0, 3.0, 5.0])
    >>> spikes2 = jnp.array([1.2, 3.2, 5.2])
    >>> distance = braintools.metric.van_rossum_distance(spikes1, spikes2, tau=0.5)
    >>> print(f"van Rossum distance: {distance:.3f}")
    
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
    
    The synchrony index is computed as:
    
    .. math::
    
        S = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{C_{ij}}{min(N_i, N_j)}
        
    where :math:`C_{ij}` is the number of coincidences between trains i and j,
    and :math:`N_i` is the number of spikes in train i.
    
    Parameters
    ----------
    spike_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)`` where non-zero values
        indicate spike occurrences.
    window_size : float
        Size of the coincidence detection window.
    dt : float, optional
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
    spikes = jnp.asarray(spike_matrix)
    n_time, n_neurons = spikes.shape

    if n_neurons < 2:
        return 0.0

    # Convert window size to time steps
    window_steps = int(window_size / dt)

    # Extract spike times for each neuron
    spike_times_list = []
    for i in range(n_neurons):
        spike_indices = jnp.where(spikes[:, i] > 0)[0]
        spike_times_list.append(spike_indices * dt)

    total_synchrony = 0.0
    n_pairs = 0

    # Calculate pairwise synchrony
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            spikes_i = spike_times_list[i]
            spikes_j = spike_times_list[j]

            if len(spikes_i) == 0 or len(spikes_j) == 0:
                continue

            # Count coincidences
            coincidences = 0
            for spike_time_i in spikes_i:
                # Check if any spike in j is within window
                time_diffs = jnp.abs(spikes_j - spike_time_i)
                if jnp.any(time_diffs <= window_size / 2):
                    coincidences += 1

            # Normalize by minimum number of spikes
            min_spikes = min(len(spikes_i), len(spikes_j))
            if min_spikes > 0:
                pair_synchrony = coincidences / min_spikes
                total_synchrony += pair_synchrony
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
    max_isi : float, default=100.0
        Maximum inter-spike interval within a burst (in time units).
    dt : float, optional
        Time step between successive samples. If None, uses brainstate default.
        
    Returns
    -------
    float
        Burst synchrony index between 0 (no burst synchrony) and 1 (perfect burst synchrony).
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Create spike matrix with bursts
    >>> spikes = jnp.zeros((1000, 10))
    >>> # Add synchronized bursts
    >>> for start in [100, 300, 600]:
    >>>     for i in range(10):
    >>>         spikes = spikes.at[start:start+5, i].set(1)
    >>> sync_idx = braintools.metric.burst_synchrony_index(spikes)
    >>> print(f"Burst synchrony: {sync_idx:.3f}")
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(spike_matrix)
    n_time, n_neurons = spikes.shape

    def detect_bursts(spike_train):
        """Detect burst events in a single spike train."""
        spike_times = jnp.where(spike_train > 0)[0] * dt
        if len(spike_times) < burst_threshold:
            return []

        bursts = []
        current_burst = [spike_times[0]]

        for i in range(1, len(spike_times)):
            isi = spike_times[i] - spike_times[i - 1]
            if isi <= max_isi:
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
                # Check for temporal overlap
                overlap = min(end1, end2) - max(start1, start2)
                if overlap > 0:
                    overlapping_neurons.add(neuron2)

        # If burst involves multiple neurons, it's synchronous
        if len(overlapping_neurons) > 1:
            synchronous_bursts += 1

    return float(synchronous_bursts / total_bursts) if total_bursts > 0 else 0.0


@set_module_as('braintools.metric')
def phase_locking_value(
    spike_matrix: brainstate.typing.ArrayLike,
    reference_freq: float,
    dt: float = None
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
    dt : float, optional
        Time step between successive samples. If None, uses brainstate default.
        
    Returns
    -------
    jnp.ndarray
        Phase-locking values for each neuron. Shape ``(n_neurons,)``.
        Values range from 0 (no phase locking) to 1 (perfect phase locking).
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Create phase-locked spikes
    >>> n_time, n_neurons = 1000, 5
    >>> spikes = jnp.zeros((n_time, n_neurons))
    >>> freq = 10.0  # 10 Hz reference
    >>> dt = 0.001   # 1 ms
    >>> # Add spikes at preferred phases
    >>> for i in range(n_neurons):
    >>>     phase_pref = i * jnp.pi / 4  # Different preferred phases
    >>>     for cycle in range(10):
    >>>         t_spike = int((cycle / freq + phase_pref / (2*jnp.pi*freq)) / dt)
    >>>         if t_spike < n_time:
    >>>             spikes = spikes.at[t_spike, i].set(1)
    >>> plv = braintools.metric.phase_locking_value(spikes, freq, dt)
    >>> print(f"PLV: {plv}")
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(spike_matrix)
    n_time, n_neurons = spikes.shape

    # Create time vector
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
    dt : float, optional
        Time step between successive samples. If None, uses brainstate default.
    tau : float, default=0.005
        Half-width of the temporal window for coincidence detection (in seconds).
        
    Returns
    -------
    jnp.ndarray
        STTC matrix with shape ``(n_neurons, n_neurons)``. Diagonal elements are 1.
        Values range from -1 to 1, where 1 indicates perfect synchrony.
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Create correlated spike trains
    >>> spikes = jnp.zeros((1000, 3))
    >>> # Add some synchronized spikes
    >>> sync_times = [100, 300, 500, 700]
    >>> for t in sync_times:
    >>>     spikes = spikes.at[t:t+3, :].set(1)
    >>> sttc = braintools.metric.spike_time_tiling_coefficient(spikes)
    >>> print(f"STTC matrix:\\n{sttc}")
    
    References
    ----------
    .. [1] Cutts, Catherine S., and Stephen J. Eglen. "Detecting pairwise correlations 
           in spike trains: an objective comparison of methods and application to the 
           retina." Journal of Neuroscience 34.43 (2014): 14288-14303.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(spike_matrix)
    n_time, n_neurons = spikes.shape

    # Total recording time
    T_total = n_time * dt

    # Convert tau to time steps
    tau_steps = int(tau / dt)

    sttc_matrix = jnp.eye(n_neurons)  # Diagonal is 1 by definition

    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            spikes_i = jnp.where(spikes[:, i] > 0)[0]
            spikes_j = jnp.where(spikes[:, j] > 0)[0]

            n_i, n_j = len(spikes_i), len(spikes_j)

            if n_i == 0 or n_j == 0:
                sttc_matrix = sttc_matrix.at[i, j].set(0.0)
                sttc_matrix = sttc_matrix.at[j, i].set(0.0)
                continue

            # Calculate P_A: proportion of spikes in i that have spike in j within tau
            coincident_i = 0
            for spike_i in spikes_i:
                # Check if any spike in j is within tau_steps
                time_diffs = jnp.abs(spikes_j - spike_i)
                if jnp.any(time_diffs <= tau_steps):
                    coincident_i += 1
            P_A = coincident_i / n_i

            # Calculate P_B: proportion of spikes in j that have spike in i within tau
            coincident_j = 0
            for spike_j in spikes_j:
                time_diffs = jnp.abs(spikes_i - spike_j)
                if jnp.any(time_diffs <= tau_steps):
                    coincident_j += 1
            P_B = coincident_j / n_j

            # Calculate T_A: proportion of time covered by windows around spikes in i
            covered_time_i = 0
            for spike_i in spikes_i:
                window_start = max(0, spike_i - tau_steps)
                window_end = min(n_time - 1, spike_i + tau_steps)
                covered_time_i += (window_end - window_start + 1)
            T_A = min(1.0, covered_time_i * dt / T_total)

            # Calculate T_B: proportion of time covered by windows around spikes in j
            covered_time_j = 0
            for spike_j in spikes_j:
                window_start = max(0, spike_j - tau_steps)
                window_end = min(n_time - 1, spike_j + tau_steps)
                covered_time_j += (window_end - window_start + 1)
            T_B = min(1.0, covered_time_j * dt / T_total)

            # Calculate STTC
            if P_A * T_B < 1 and P_B * T_A < 1:
                term1 = (P_A - T_B) / (1 - P_A * T_B)
                term2 = (P_B - T_A) / (1 - P_B * T_A)
                sttc_value = 0.5 * (term1 + term2)
            else:
                sttc_value = 0.0  # Avoid division by zero

            sttc_matrix = sttc_matrix.at[i, j].set(sttc_value)
            sttc_matrix = sttc_matrix.at[j, i].set(sttc_value)

    return sttc_matrix


@set_module_as('braintools.metric')
def correlation_index(
    spike_matrix: brainstate.typing.ArrayLike,
    window_size: float,
    dt: float = None
):
    r"""Calculate correlation index for spike train synchrony.
    
    The correlation index measures the strength of pairwise correlations in spike
    trains by computing correlation coefficients between binned spike counts.
    
    The index is computed as:
    
    .. math::
    
        CI = \frac{1}{N(N-1)} \sum_{i \neq j} \rho_{ij}
        
    where :math:`\rho_{ij}` is the Pearson correlation coefficient between
    the binned spike counts of neurons i and j.
    
    Parameters
    ----------
    spike_matrix : brainstate.typing.ArrayLike
        Spike matrix with shape ``(n_time_steps, n_neurons)``.
    window_size : float
        Size of time windows for binning spikes (in time units).
    dt : float, optional
        Time step between successive samples. If None, uses brainstate default.
        
    Returns
    -------
    float
        Correlation index representing average pairwise correlation.
        Values range from -1 to 1, where positive values indicate synchrony.
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import braintools as braintools
    >>> # Create correlated spike trains
    >>> spikes = (jnp.random.random((1000, 10)) < 0.1).astype(float)
    >>> # Add some correlation by copying spikes between neurons
    >>> spikes = spikes.at[:, 1].set(spikes[:, 0] * 0.7 + spikes[:, 1] * 0.3)
    >>> ci = braintools.metric.correlation_index(spikes, window_size=50.0)
    >>> print(f"Correlation index: {ci:.3f}")
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(spike_matrix)
    n_time, n_neurons = spikes.shape

    if n_neurons < 2:
        return 0.0

    # Bin size in time steps
    bin_size = int(window_size / dt)
    n_bins = n_time // bin_size

    if n_bins < 2:
        return 0.0

    # Bin spike counts
    binned_spikes = jnp.zeros((n_bins, n_neurons))
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, n_time)
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

    return float(jnp.mean(jnp.array(correlations)))
