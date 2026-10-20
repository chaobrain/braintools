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

from typing import Optional, Union

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as onp
from jax import lax

from braintools._misc import set_module_as
from braintools.spike._spike_trains import _is_traced

__all__ = [
    'firing_rate',
    'victor_purpura_distance',
    'van_rossum_distance',
    'spike_train_synchrony',
    'burst_synchrony_index',
    'phase_locking_value',
    'spike_time_tiling_coefficient',
    'correlation_index',
]


def _in_unit(x, unit) -> float:
    """Magnitude of a time ``x`` in ``unit``.

    A ``brainunit.Quantity`` is converted; a plain number is taken to be
    already in ``unit`` (the function's documented convention for floats).
    """
    if isinstance(x, u.Quantity):
        return float(x.to_decimal(unit))
    return float(x)


def _steps(a, b, unit) -> float:
    """Ratio ``a / b`` of two times, converting Quantities to ``unit`` first."""
    return _in_unit(a, unit) / _in_unit(b, unit)


def _floor_ratio(r: float) -> int:
    """``floor(r)``, but treat values within 1e-9 of an integer as that integer."""
    n = round(r)
    return int(n) if abs(r - n) <= 1e-9 * max(1.0, abs(r)) else int(onp.floor(r))


def _float_array(x) -> jax.Array:
    """``x`` as a JAX array of the default float dtype (floats keep their dtype)."""
    x = jnp.asarray(u.get_magnitude(x))
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(jnp.result_type(float))
    return x


def _dilate(fired, half_width: int, xp):
    """Mark the samples that have a spike at most ``half_width`` samples away.

    ``fired`` is a boolean ``(T, N)`` raster; the result has the same shape and
    is ``True`` at ``t`` when some ``fired[s]`` with ``|s - t| <= half_width``
    is ``True``. Computed with a cumulative sum (``O(T N)``) by ``xp``
    (``numpy`` on host, ``jax.numpy`` when traced). A negative ``half_width``
    gives an all-``False`` raster.
    """
    n_time = fired.shape[0]
    if half_width < 0:
        return xp.zeros(fired.shape, dtype=bool)
    half_width = min(half_width, n_time)
    zero = xp.zeros((1,) + tuple(fired.shape[1:]), dtype=xp.int32)
    csum = xp.concatenate([zero, xp.cumsum(fired, axis=0, dtype=xp.int32)], axis=0)
    t = onp.arange(n_time)
    hi = onp.minimum(t + half_width + 1, n_time)
    lo = onp.maximum(t - half_width, 0)
    return (csum[hi] - csum[lo]) > 0


@set_module_as('braintools.analysis.spike')
def firing_rate(
    spikes: brainstate.typing.ArrayLike,
    width: Union[float, u.Quantity],
    dt: Optional[Union[float, u.Quantity]] = None
):
    r"""Calculate the smoothed population firing rate from spike data.

    Computes the time-varying population firing rate by averaging spike counts
    across neurons and applying temporal smoothing with a rectangular window, as in
    Brian 2 [1]_.
    This provides a measure of the overall activity level of the neural population
    over time [1]_.

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
    >>> rates = braintools.analysis.spike.firing_rate(spikes, window_width, dt)
    >>> print(f"Rate shape: {rates.shape}")
    Rate shape: (100,)
    >>> print(f"Mean rate: {np.mean(rates):.2f} Hz")
    Mean rate: 917.45 Hz

    Compare different smoothing window sizes:

    >>> narrow_rates = braintools.analysis.spike.firing_rate(spikes, 2*u.ms, dt)
    >>> wide_rates = braintools.analysis.spike.firing_rate(spikes, 10*u.ms, dt)
    >>> # narrow_rates will be more variable, wide_rates more smooth

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> time = np.arange(len(rates)) * float(dt.to_decimal(u.ms))
    >>> _ = plt.plot(time, rates, label='Population rate')
    >>> _ = plt.xlabel('Time (ms)')
    >>> _ = plt.ylabel('Firing rate (Hz)')
    >>> _ = plt.title('Population Firing Rate')
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
    braintools.visualize.raster_plot : Plot spike times as a raster
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
    width1 = _floor_ratio(_steps(width, dt, u.second) / 2) * 2 + 1
    # Normalize by the *realized* window duration (``width1 * dt``) rather than the
    # nominal ``width`` so the kernel integrates spike counts to a rate over exactly
    # the samples that are used.
    # Float ``dt``/``width`` are taken to be in seconds and Quantities are
    # converted, so ``1 / duration`` is in Hz.
    duration = width1 * _in_unit(dt, u.second)
    window = jnp.ones(width1) / duration
    rate = jnp.mean(jnp.asarray(u.get_magnitude(spikes)), axis=1)
    return jnp.convolve(rate, window, mode='same')


@set_module_as('braintools.analysis.spike')
def victor_purpura_distance(
    spike_times_1: brainstate.typing.ArrayLike,
    spike_times_2: brainstate.typing.ArrayLike,
    cost_factor: float = 1.0
):
    r"""Calculate Victor-Purpura distance between two spike trains.

    The Victor-Purpura distance quantifies the dissimilarity between two spike trains
    by computing the minimum cost to transform one spike train into another through
    spike insertions, deletions, and temporal shifts [1]_.

    The distance is computed as:

    .. math::

        D_{VP} = \min \sum_{ops} c_{op}

    where the cost of moving a spike by time :math:`\Delta t` is :math:`q|\Delta t|`,
    insertion/deletion costs are 1, and :math:`q` is the cost factor.

    Parameters
    ----------
    spike_times_1 : brainstate.typing.ArrayLike
        First spike train as a 1-D array of spike times. Entries equal to
        ``+inf`` are padding and are treated as absent spikes, so a row of
        ``SpikeTrains.times`` can be passed directly.
    spike_times_2 : brainstate.typing.ArrayLike
        Second spike train, same convention.
    cost_factor : float, default=1.0
        Cost factor :math:`q` for temporal shifts. Higher values penalize
        temporal differences more heavily.

    Returns
    -------
    jax.Array
        Victor-Purpura distance, a 0-d array in the float dtype of the
        inputs (``float32`` unless ``jax_enable_x64`` is on).

    Raises
    ------
    ValueError
        If a spike train is not 1-D.

    See Also
    --------
    van_rossum_distance : Kernel-based distance between two spike trains.
    isi_distance : Time-resolved ISI-distance between two populations.
    spike_distance : Time-resolved SPIKE-distance between two populations.

    Notes
    -----
    The dynamic programme fills the ``(n_1 + 1) x (n_2 + 1)`` cost table with a
    :func:`jax.lax.scan` over its rows and an inner scan over its columns; a
    ``+inf`` entry leaves the row (or column) unchanged, which is the same as
    removing that spike. The function therefore works under ``jax.jit`` and
    ``jax.vmap`` (e.g. over the rows of a padded ``(N, K)`` array), and is
    differentiable almost everywhere with respect to the spike times and
    ``cost_factor`` (the distance is piecewise linear in them). It takes
    ``O(n_1 n_2)`` sequential steps.

    ``cost_factor`` and the spike times must use consistent units (the distance
    has units of ``cost_factor * time + count``); the magnitudes of Quantities
    are used as given.

    Before version 0.5 this function returned a Python ``float``; ``float(d)``
    still works on the 0-d array.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes1 = jnp.array([1.0, 2.0, 3.0])
        >>> spikes2 = jnp.array([1.1, 2.1, 3.1])
        >>> d = braintools.analysis.spike.victor_purpura_distance(spikes1, spikes2, cost_factor=10.0)
        >>> round(float(d), 4)
        3.0
        >>>
        >>> # Padded rows (``+inf`` = no spike), e.g. ``SpikeTrains.times``, under vmap.
        >>> a = jnp.array([[1.0, 2.0, jnp.inf], [1.0, jnp.inf, jnp.inf]])
        >>> b = jnp.array([[1.0, 2.0, 3.0], [jnp.inf, jnp.inf, jnp.inf]])
        >>> jax.vmap(braintools.analysis.spike.victor_purpura_distance)(a, b)
        Array([1., 1.], dtype=float32)

    References
    ----------
    .. [1] Victor, Jonathan D., and Keith P. Purpura. "Nature and precision of
           temporal coding in visual cortex: a metric-space analysis."
           Journal of neurophysiology 76.2 (1996): 1310-1326.
    """
    spikes1 = _float_array(spike_times_1)
    spikes2 = _float_array(spike_times_2)
    if spikes1.ndim != 1 or spikes2.ndim != 1:
        raise ValueError(
            f'victor_purpura_distance expects 1-D spike-time arrays, got shapes '
            f'{spikes1.shape} and {spikes2.shape}; use jax.vmap for batches.'
        )
    dtype = jnp.result_type(spikes1, spikes2)
    spikes1 = spikes1.astype(dtype)
    spikes2 = spikes2.astype(dtype)
    q = jnp.asarray(u.get_magnitude(cost_factor), dtype=dtype)
    valid1 = spikes1 != jnp.inf
    valid2 = spikes2 != jnp.inf
    # Padding is replaced by a finite value before any arithmetic ("double
    # where"): its cost is discarded below anyway, and ``q * |s - inf|`` would
    # otherwise give ``0 * inf = nan`` gradients for ``q`` and the spike times.
    spikes1 = jnp.where(valid1, spikes1, 0.0)
    spikes2 = jnp.where(valid2, spikes2, 0.0)

    # dp[i, j] = minimum cost to transform spikes1[:i] into spikes2[:j].
    # Row 0: insert every (valid) spike of train 2.
    row0 = jnp.concatenate([
        jnp.zeros((1,), dtype), jnp.cumsum(valid2.astype(dtype))
    ])

    def row_step(prev, xs):
        s, valid = xs
        first = prev[0] + 1.0  # delete every spike of train 1 so far

        def col_step(left, ys):
            diag, up, t, valid_t = ys
            # Match spike i with spike j (temporal shift), delete, or insert.
            match_cost = diag + q * jnp.abs(s - t)
            cost = jnp.minimum(jnp.minimum(match_cost, up + 1.0), left + 1.0)
            new = jnp.where(valid_t, cost, left)
            return new, new

        _, rest = lax.scan(col_step, first, (prev[:-1], prev[1:], spikes2, valid2))
        row = jnp.concatenate([first[None], rest])
        return jnp.where(valid, row, prev), None

    last_row, _ = lax.scan(row_step, row0, (spikes1, valid1))
    return last_row[-1]


@set_module_as('braintools.analysis.spike')
def van_rossum_distance(
    spike_times_1: brainstate.typing.ArrayLike,
    spike_times_2: brainstate.typing.ArrayLike,
    tau: float = 1.0,
    t_max: Optional[float] = None
):
    r"""Calculate van Rossum distance between two spike trains.

    The van Rossum distance measures dissimilarity between spike trains by
    convolving each with an exponential kernel and computing the Euclidean
    distance between the resulting continuous functions [1]_.

    Each spike train is convolved with kernel :math:`K(t) = \frac{1}{\tau}e^{-t/\tau}H(t)`
    where :math:`H(t)` is the Heaviside step function. The distance is:

    .. math::

        D_{vR} = \sqrt{\int_0^{T} [f_1(t) - f_2(t)]^2 dt}

    where :math:`f_i(t)` is the convolved spike train.

    Parameters
    ----------
    spike_times_1 : brainstate.typing.ArrayLike
        First spike train as a 1-D array of spike times. Entries equal to
        ``+inf`` are padding and are ignored, so a row of
        ``SpikeTrains.times`` can be passed directly.
    spike_times_2 : brainstate.typing.ArrayLike
        Second spike train, same convention.
    tau : float, default=1.0
        Time constant of the exponential kernel. Larger values emphasize
        longer-term dependencies.
    t_max : float, optional
        End :math:`T` of the integration window. If None, uses the largest
        (non-padding) spike time + ``5 * tau``; this default depends on the
        data, so ``t_max`` must be given when the spike times are traced
        (inside ``jax.jit``, ``jax.vmap`` or ``jax.grad``).

    Returns
    -------
    jax.Array
        van Rossum distance, a 0-d array (0 when both trains are empty and
        ``t_max`` is None).

    Raises
    ------
    ValueError
        If ``t_max`` is None while the spike times are traced.

    See Also
    --------
    victor_purpura_distance : Edit distance between two spike trains.
    spike_distance : Time-resolved SPIKE-distance between two populations.

    Notes
    -----
    Convention: each spike train is convolved with the *rate-normalized* kernel
    :math:`K(t) = \frac{1}{\tau} e^{-t/\tau} H(t)` and the distance is
    :math:`\sqrt{\int (f_1 - f_2)^2\, dt}` (no extra :math:`1/\tau` prefactor on the
    integral). Other texts use :math:`K(t) = e^{-t/\tau}` with a :math:`1/\tau`
    integral prefactor; both differ only by an overall scale.

    The integral over :math:`[0, T]` is evaluated in **closed form** (one term per
    spike pair), so there is no quadrature error; only the deliberate truncation at
    ``t_max`` (default ``max(spike_time) + 5*tau``, whose ``5*tau`` tail leaves out
    less than :math:`e^{-10} \approx 5\cdot10^{-5}` of the exponential) remains.
    Pass a larger ``t_max`` to reduce that further. Spike pairs are evaluated with
    vectorized array ops rather than a Python loop over spikes.

    With ``t_max`` given, the function works under ``jax.jit`` and ``jax.vmap``
    and is differentiable with respect to the spike times, ``tau`` and
    ``t_max`` (except where the distance is 0). ``tau``, ``t_max`` and the
    spike times must use consistent time units.

    Before version 0.5 this function returned a Python ``float``; ``float(d)``
    still works on the 0-d array.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes1 = jnp.array([1.0, 3.0, 5.0])
        >>> spikes2 = jnp.array([1.2, 3.2, 5.2])
        >>> d = braintools.analysis.spike.van_rossum_distance(spikes1, spikes2, tau=0.5)
        >>> round(float(d), 4)
        1.4022
        >>>
        >>> # Under jit, give ``t_max``; ``+inf`` entries are padding.
        >>> f = jax.jit(lambda a, b: braintools.analysis.spike.van_rossum_distance(
        ...     a, b, tau=0.5, t_max=10.0))
        >>> d = f(jnp.array([1.0, 3.0, 5.0, jnp.inf]), spikes2)
        >>> round(float(d), 4)
        1.4022

    References
    ----------
    .. [1] van Rossum, Mark CW. "A novel spike distance."
           Neural computation 13.4 (2001): 751-763.
    """
    spikes1 = jnp.asarray(spike_times_1)
    spikes2 = jnp.asarray(spike_times_2)
    valid1 = spikes1 != jnp.inf
    valid2 = spikes2 != jnp.inf

    # Determine time window (default: max spike time + 5*tau)
    if t_max is None:
        if _is_traced(spikes1, spikes2):
            raise ValueError(
                'van_rossum_distance: `t_max` is required when the spike times are '
                'traced (inside jax.jit, jax.vmap or jax.grad), because the default '
                '`max(spike_time) + 5 * tau` depends on the data. Pass it explicitly, '
                'e.g. `van_rossum_distance(a, b, tau=tau, t_max=t_stop + 5 * tau)`.'
            )
        all_spikes = jnp.concatenate([spikes1[valid1], spikes2[valid2]])
        if all_spikes.size == 0:
            return jnp.asarray(0.0)
        t_max = float(jnp.max(all_spikes)) + 5.0 * tau

    def pair_integral(times_a, valid_a, times_b, valid_b):
        r"""Exact ``\int_0^{T} f_a(t) f_b(t)\, dt`` for the exponential kernel.

        With :math:`f_a(t) = \frac{1}{\tau}\sum_k e^{-(t-a_k)/\tau} H(t-a_k)`,
        the product of two such responses integrates in closed form to one term
        per spike pair:

        .. math::

            \frac{1}{2\tau}\left[
                e^{(a+b-2m)/\tau} - e^{(a+b-2T)/\tau}
            \right], \qquad m = \max(0, a, b),

        for :math:`m < T` (a pair contributes nothing once :math:`m \ge T`, since
        the integrand vanishes on :math:`[0, T]`). Padding entries are
        replaced by 0 before the arithmetic and masked out of the sum, so they
        produce neither ``nan`` values nor ``nan`` gradients.
        """
        a = jnp.where(valid_a, times_a, 0.0)[:, None]
        b = jnp.where(valid_b, times_b, 0.0)[None, :]
        m = jnp.maximum(jnp.maximum(0.0, a), b)
        terms = (
            jnp.exp((a + b - 2.0 * m) / tau)
            - jnp.exp((a + b - 2.0 * t_max) / tau)
        )
        keep = valid_a[:, None] & valid_b[None, :] & (m < t_max)
        return jnp.sum(jnp.where(keep, terms, 0.0))

    distance_squared = (1.0 / (2.0 * tau)) * (
        pair_integral(spikes1, valid1, spikes1, valid1)
        + pair_integral(spikes2, valid2, spikes2, valid2)
        - 2.0 * pair_integral(spikes1, valid1, spikes2, valid2)
    )
    # Round-off can push an exactly-zero distance marginally negative.
    return jnp.sqrt(jnp.maximum(distance_squared, 0.0))


@set_module_as('braintools.analysis.spike')
def spike_train_synchrony(
    spike_matrix: brainstate.typing.ArrayLike,
    window_size: float,
    dt: Optional[float] = None
):
    r"""Calculate spike train synchrony using the SPIKE-synchronization measure.

    This measure quantifies the degree of synchronization between multiple spike trains
    by counting coincident events within sliding time windows and normalizing by the
    total number of possible coincidences [1]_.

    The synchrony index is the average over neuron pairs of a symmetric
    coincidence ratio:

    .. math::

        S = \frac{1}{N_{pairs}} \sum_{i < j} \frac{C_{i \to j} + C_{j \to i}}{N_i + N_j}

    where :math:`C_{i \to j}` is the number of spikes in train :math:`i` that have at
    least one spike of train :math:`j` within half the coincidence window, and
    :math:`N_i` is the number of spikes in train :math:`i`. Counting coincidences in
    both directions and normalizing by the total spike count guarantees
    :math:`S \in [0, 1]`. Pairs with a silent neuron are left out of the
    average.

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
        Time step between successive samples. If None, uses the default from
        ``brainstate.environ.get_dt()``. A plain float is assumed to be in
        **milliseconds** (the ``brainstate`` convention), so it must be expressed
        in the same unit as ``window_size``/``max_isi``/``tau``; a
        ``brainunit.Quantity`` is converted automatically.

    Returns
    -------
    jax.Array
        Spike train synchrony index, a 0-d array between 0 (no synchrony) and
        1 (perfect synchrony). 0 when fewer than two neurons, or no pair of
        active neurons, are present.

    Notes
    -----
    The coincidences of all pairs are counted at once: a cumulative sum marks
    the samples within ``window_size / 2`` of a spike of each neuron, a
    matrix product counts, for every ordered pair, the spikes of one neuron
    on the marked samples of the other, and the pair ratios are evaluated with
    a ``vmap`` over the ``N (N - 1) / 2`` upper-triangle pairs. Memory is
    ``O(T N + N^2)``.

    ``window_size`` and ``dt`` must be concrete; the spike matrix may be
    traced, so the function works under ``jax.jit`` and ``jax.vmap`` (e.g.
    over trials). It is not differentiable (it counts spikes).

    Before version 0.5 this function returned a Python ``float``;
    ``float(s)`` still works on the 0-d array.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools as braintools
        >>> # Create synchronized spikes
        >>> spikes = jnp.zeros((100, 5))
        >>> spikes = spikes.at[20:25, :].set(1)  # Synchronized burst
        >>> synchrony = braintools.analysis.spike.spike_train_synchrony(spikes, window_size=10.0, dt=1.0)
        >>> print(f"Synchrony: {synchrony:.3f}")
        Synchrony: 1.000

    References
    ----------
    .. [1] Kreuz, Thomas, et al. "Measuring spike train synchrony."
           Journal of neuroscience methods 165.1 (2007): 151-161.
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape
    float_dtype = jnp.result_type(float)

    if n_neurons < 2:
        return jnp.asarray(0.0, dtype=float_dtype)

    # Half-coincidence window in samples (Quantities converted, floats in ms).
    # Sample offsets are integers, so ``|d| <= half_steps`` <=> ``|d| <= floor(half_steps)``.
    half_steps = _steps(window_size, dt, u.ms) / 2.0
    half_width = -1 if half_steps < 0 else int(min(onp.floor(half_steps), n_time))

    fired = spikes > 0
    near = _dilate(fired, half_width, jnp)
    # coin[i, j]: spikes of neuron i with at least one spike of neuron j nearby.
    coin = jnp.matmul(fired.T.astype(jnp.int32), near.astype(jnp.int32))
    counts = jnp.sum(fired, axis=0, dtype=jnp.int32)

    def pair(i, j):
        n_i, n_j = counts[i], counts[j]
        active = (n_i > 0) & (n_j > 0)
        # Symmetric counting normalized by the total spike count keeps the ratio
        # in [0, 1] (a one-directional ``min(N_i, N_j)`` normalization could exceed 1).
        total = (coin[i, j] + coin[j, i]).astype(float_dtype)
        ratio = total / jnp.where(active, n_i + n_j, 1).astype(float_dtype)
        return jnp.where(active, ratio, 0.0), active

    rows, cols = onp.triu_indices(n_neurons, k=1)
    ratios, active = jax.vmap(pair)(jnp.asarray(rows), jnp.asarray(cols))
    n_pairs = jnp.sum(active)
    mean = jnp.sum(ratios) / jnp.maximum(n_pairs, 1).astype(float_dtype)
    return jnp.where(n_pairs > 0, mean, 0.0).astype(float_dtype)


@set_module_as('braintools.analysis.spike')
def burst_synchrony_index(
    spike_matrix: brainstate.typing.ArrayLike,
    burst_threshold: int = 3,
    max_isi: float = 100.0,
    dt: Optional[float] = None
):
    r"""Calculate burst synchrony index based on co-occurring burst events.

    This measure identifies burst events in each spike train and quantifies
    the synchronization of these bursts across the population.

    A burst is defined as a sequence of at least ``burst_threshold`` spikes
    with inter-spike intervals ≤ ``max_isi``. The index is the fraction of
    bursts that overlap in time (endpoints included) with a burst of another
    neuron.

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
        Time step between successive samples. If None, uses the default from
        ``brainstate.environ.get_dt()``. A plain float is assumed to be in
        **milliseconds** (the ``brainstate`` convention), so it must be expressed
        in the same unit as ``window_size``/``max_isi``/``tau``; a
        ``brainunit.Quantity`` is converted automatically.

    Returns
    -------
    float
        Burst synchrony index between 0 (no burst synchrony) and 1 (perfect burst synchrony).

    Notes
    -----
    The bursts of all neurons are found at once from the sorted spike list
    (a new burst starts at every neuron change or ISI above ``max_isi``), and
    the overlapping bursts are counted with two binary searches over the
    sorted burst starts and ends, in ``O(S log S)`` for ``S`` spikes. Bursts
    of the same neuron never overlap, so only other neurons' bursts are
    counted.

    This function runs on host (concrete) arrays with NumPy and returns a
    Python ``float``, so it is not ``jit``/``vmap``/``grad``-compatible.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes = jnp.zeros((1000, 10))
        >>> for start in [100, 300, 600]:
        ...     spikes = spikes.at[start:start + 5, :].set(1)
        >>> sync_idx = braintools.analysis.spike.burst_synchrony_index(spikes, max_isi=5.0, dt=1.0)
        >>> sync_idx
        1.0
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = onp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape
    dt_m = _in_unit(dt, u.ms)
    max_isi_m = _in_unit(max_isi, u.ms)

    # All spikes, ordered by neuron and then by time.
    neuron, step = onp.nonzero(spikes.T > 0)
    if neuron.size == 0:
        return 0.0
    spike_times = step * dt_m

    # A new burst candidate starts at the first spike of every neuron and after
    # every ISI above ``max_isi``.
    starts: onp.ndarray = onp.ones(neuron.size, dtype=bool)
    starts[1:] = (neuron[1:] != neuron[:-1]) | ((spike_times[1:] - spike_times[:-1]) > max_isi_m)
    first = onp.flatnonzero(starts)
    last: onp.ndarray = onp.append(first[1:] - 1, neuron.size - 1)
    is_burst = (last - first + 1) >= burst_threshold
    burst_start = spike_times[first[is_burst]]
    burst_end = spike_times[last[is_burst]]
    total_bursts = int(burst_start.size)
    if total_bursts == 0:
        return 0.0

    # Two bursts overlap when ``min(end1, end2) - max(start1, start2) >= 0``
    # (inclusive: bursts touching at a single instant count). The number of
    # bursts overlapping burst b is #{start' <= end_b} - #{end' < start_b};
    # it includes b itself and no other burst of the same neuron, because the
    # bursts of one neuron are separated by an ISI above ``max_isi``.
    n_overlap = (
        onp.searchsorted(onp.sort(burst_start), burst_end, side='right')
        - onp.searchsorted(onp.sort(burst_end), burst_start, side='left')
        - 1
    )
    synchronous_bursts = int(onp.count_nonzero(n_overlap > 0))
    return float(synchronous_bursts / total_bursts)


@set_module_as('braintools.analysis.spike')
def phase_locking_value(
    spike_matrix: brainstate.typing.ArrayLike,
    reference_freq: float,
    dt: Optional[Union[float, u.Quantity]] = None
):
    r"""Calculate phase-locking value (PLV) for spike synchronization.
    
    The PLV measures the consistency of phase relationships between spike trains
    and a reference oscillation, indicating rhythmic synchronization [1]_.
    
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

    See Also
    --------
    braintools.analysis.spectral.plv : Phase-locking value between two continuous signals (Lachaux et al. 1999).

    Notes
    -----
    This computes the **vector strength** (Rayleigh resultant length) of each
    neuron's spikes relative to an *external* reference oscillation of frequency
    ``reference_freq``. It is therefore a spike–field locking measure, **not** the
    Lachaux et al. (1999) [1]_ pairwise PLV between two continuous signals.

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
        >>> plv = braintools.analysis.spike.phase_locking_value(spikes, freq, dt)
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


@set_module_as('braintools.analysis.spike')
def spike_time_tiling_coefficient(
    spike_matrix: brainstate.typing.ArrayLike,
    dt: Optional[float] = None,
    tau: float = 5.0
):
    r"""Calculate Spike Time Tiling Coefficient (STTC).

    STTC measures synchrony between spike trains while controlling for firing rate
    differences. It's based on the proportion of spikes that fall within a temporal
    window around spikes in the other train [1]_.

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
        Time step between successive samples. If None, uses the default from
        ``brainstate.environ.get_dt()``. A plain float is assumed to be in
        **milliseconds** (the ``brainstate`` convention), so it must be expressed
        in the same unit as ``window_size``/``max_isi``/``tau``; a
        ``brainunit.Quantity`` is converted automatically.
    tau : float or brainunit.Quantity, default=5.0
        Half-width of the temporal window for coincidence detection. A plain
        float is in milliseconds (like ``dt``); a Quantity is converted. The
        default is 5 ms.

    Returns
    -------
    jnp.ndarray
        STTC matrix with shape ``(n_neurons, n_neurons)``. Diagonal elements are 1.
        Values range from -1 to 1, where 1 indicates perfect synchrony. A pair
        involving a silent neuron has STTC 0.

    Notes
    -----
    The time-coverage terms :math:`T_A`, :math:`T_B` use the **union** of the
    per-spike :math:`[\,s - \tau,\ s + \tau\,]` windows, so overlapping windows are
    not double-counted (the defining quantity in Cutts & Eglen, 2014 [1]_). Each STTC
    term is guarded independently against a zero denominator.

    All pairs are computed at once: a cumulative sum marks the samples covered
    by each neuron's windows, and one matrix product counts, for every ordered
    pair, the spikes of one neuron inside the other's windows. Memory is
    ``O(T N + N^2)``.

    This function runs on host (concrete) arrays with NumPy, so it is not
    ``jit``/``vmap``/``grad``-compatible.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> spikes = jnp.zeros((1000, 3))
        >>> sync_times = [100, 300, 500, 700]
        >>> for t in sync_times:
        ...     spikes = spikes.at[t:t + 3, :].set(1)
        >>> sttc = braintools.analysis.spike.spike_time_tiling_coefficient(spikes, dt=1.0, tau=2.0)
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

    # Coincidence half-window in samples (Quantities converted, floats in ms).
    tau_steps = int(round(_steps(tau, dt, u.ms)))

    fired = spikes > 0
    # covered[t, n]: t lies in the union of the [s - tau, s + tau] windows of
    # neuron n (clipped to the recording). Summing per-spike window lengths
    # instead would double-count overlapping windows.
    covered = _dilate(fired, tau_steps, onp)
    # near[i, j]: spikes of neuron i within tau of a spike of neuron j.
    near = fired.T.astype(onp.int64) @ covered.astype(onp.int64)
    counts = fired.sum(axis=0)
    coverage = covered.sum(axis=0)

    sttc_matrix = onp.eye(n_neurons)  # Diagonal is 1 by definition
    i, j = onp.triu_indices(n_neurons, k=1)
    active = (counts[i] > 0) & (counts[j] > 0)
    i, j = i[active], j[active]
    P_A = near[i, j] / counts[i]  # spikes in i near a j spike
    P_B = near[j, i] / counts[j]  # spikes in j near an i spike
    T_A = coverage[i] / n_time
    T_B = coverage[j] / n_time

    # Guard each term independently: a single degenerate denominator should
    # zero only its own term, not the whole coefficient.
    den1 = 1.0 - P_A * T_B
    den2 = 1.0 - P_B * T_A
    term1 = onp.where(den1 != 0, (P_A - T_B) / onp.where(den1 != 0, den1, 1.0), 0.0)
    term2 = onp.where(den2 != 0, (P_B - T_A) / onp.where(den2 != 0, den2, 1.0), 0.0)
    sttc_value = 0.5 * (term1 + term2)
    sttc_matrix[i, j] = sttc_value
    sttc_matrix[j, i] = sttc_value
    return jnp.asarray(sttc_matrix)


@set_module_as('braintools.analysis.spike')
def correlation_index(
    spike_matrix: brainstate.typing.ArrayLike,
    window_size: float,
    dt: Optional[float] = None
):
    r"""Calculate correlation index for spike train synchrony.

    The correlation index measures the strength of pairwise correlations in spike
    trains by computing the average Pearson correlation coefficient between binned
    spike counts [1]_.

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
        Time step between successive samples. If None, uses the default from
        ``brainstate.environ.get_dt()``. A plain float is assumed to be in
        **milliseconds** (the ``brainstate`` convention), so it must be expressed
        in the same unit as ``window_size``/``max_isi``/``tau``; a
        ``brainunit.Quantity`` is converted automatically.

    Returns
    -------
    float
        Correlation index representing the average pairwise correlation.
        Values range from -1 to 1, where positive values indicate synchrony.
        A pair with a constant (zero-variance) binned train contributes 0;
        fewer than two neurons or two whole bins give 0.

    Notes
    -----
    This is the **mean pairwise Pearson correlation** of binned spike counts. It is
    distinct from the Wong–Meister/Mastronarde "correlation index", which is a
    coincidence-rate ratio (expected-vs-observed near-coincident firings) with a
    different range and interpretation. Use this function when a normalized
    [-1, 1] linear-correlation summary is desired.

    Only whole bins are used; trailing samples that do not fill a bin are
    dropped. The bins are summed with one reshape and the pair correlations
    are evaluated with a ``vmap`` over the ``N (N - 1) / 2`` upper-triangle
    pairs. The result is returned as a Python ``float``, so the function is
    not ``jit``/``vmap``/``grad``-compatible.

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
        >>> ci = braintools.analysis.spike.correlation_index(spikes, window_size=50.0, dt=1.0)
        >>> bool(-1.0 <= ci <= 1.0)
        True
    """
    dt = brainstate.environ.get_dt() if dt is None else dt
    spikes = jnp.asarray(u.get_magnitude(spike_matrix))
    n_time, n_neurons = spikes.shape

    if n_neurons < 2:
        return 0.0

    # Bin size in samples (``window_size`` and ``dt`` must share a time unit).
    bin_size = max(1, _floor_ratio(_steps(window_size, dt, u.ms)))
    # Only whole bins are used; any trailing samples that do not fill a complete
    # bin are discarded so every bin has the same width.
    n_bins = n_time // bin_size

    if n_bins < 2:
        return 0.0

    # Bin spike counts
    binned_spikes = jnp.sum(
        spikes[:n_bins * bin_size].reshape(n_bins, bin_size, n_neurons), axis=1
    ).astype(jnp.result_type(float))

    def pair(i, j):
        # Pearson correlation; a series with zero variance contributes 0.
        x, y = binned_spikes[:, i], binned_spikes[:, j]
        degenerate = (jnp.var(x) == 0) | (jnp.var(y) == 0)
        corr = jnp.corrcoef(x, y)[0, 1]
        corr = jnp.where(jnp.isnan(corr), 0.0, corr)
        return jnp.where(degenerate, 0.0, corr)

    rows, cols = onp.triu_indices(n_neurons, k=1)
    correlations = jax.vmap(pair)(jnp.asarray(rows), jnp.asarray(cols))

    # Clamp to the documented [-1, 1] range (guards against float round-off in
    # ``corrcoef`` pushing the mean marginally outside the interval).
    return float(jnp.clip(jnp.mean(correlations), -1.0, 1.0))
