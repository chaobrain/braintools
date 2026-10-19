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

"""Spike-triggered average of a sampled signal."""

from typing import Any, Sequence, Tuple

import jax.numpy as jnp

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains, _split, _wrap
from ._isi import _EPS_FACTOR, _bins_for, _check_trains, _safe_div, _static_time
from ._psth import _parse_window

__all__ = [
    'spike_triggered_average',
]


@set_module_as('braintools.analysis.spike')
def spike_triggered_average(
    trains: SpikeTrains,
    signal: Any,
    dt: Any,
    window: Sequence[Any],
) -> Tuple[Any, Any]:
    r"""Average a sampled signal around the spikes of every neuron.

    For neuron :math:`n` with usable spikes :math:`t_1, \dots, t_m`,

    .. math::

        \mathrm{STA}_n[l] = \frac{1}{m} \sum_{k=1}^{m} x[s_k + l],
        \qquad l = 0, \dots, L - 1,

    where :math:`s_k = \lfloor (t_k - t_\text{before} - t_0) / dt \rfloor`
    is the first sample of the window of spike :math:`k` and
    :math:`L = \lceil (t_\text{before} + t_\text{after}) / dt \rceil`.

    Parameters
    ----------
    trains : SpikeTrains
        Spike trains with ``N`` neurons.
    signal : array-like or Quantity
        Signal of shape ``(T, ...)``: sample ``i`` is at time
        ``trains.t_start + i * dt``. Shared by all neurons.
    dt : float or Quantity
        Sampling interval, static and positive. A plain number is in
        milliseconds, like the other spike-time arguments.
    window : tuple of (float or Quantity)
        ``(t_before, t_after)``: the average spans ``[-t_before, t_after)``
        around each spike (a positive ``t_before`` reaches before the
        spike). Static; plain numbers are in milliseconds.

    Returns
    -------
    sta : jax.Array or Quantity
        Averages of shape ``(N, L) + signal.shape[1:]``, a Quantity when
        ``signal`` is one. ``nan`` for a neuron without a usable spike.
    lags : jax.Array or Quantity
        Nominal lag of every sample, ``-t_before + l * dt``, shape ``(L,)``;
        a Quantity when the spike times carry units, else milliseconds.

    Raises
    ------
    TypeError
        If ``trains`` is not a ``SpikeTrains``, or ``dt``/``window`` are
        traced.
    ValueError
        If ``signal`` is a scalar, ``dt`` is not positive, or the window has
        non-positive length.

    See Also
    --------
    psth : Event-aligned spike histograms.
    braintools.spike.SpikeTrains.from_dense : Build the input from a raster.

    Notes
    -----
    A spike is used only if its whole window lies inside the signal,
    ``t - t_before >= t_0`` and ``t + t_after <= t_0 + T dt``; other spikes,
    e.g. near the start of the recording, are excluded [1]_.

    This matches ``elephant.sta.spike_triggered_average(signal, trains,
    window=(-t_before, t_after))``: Elephant gives the window as signed
    ``(start, stop)`` offsets, while here ``t_before`` is the (positive)
    time before the spike.

    The windows are gathered with ``O(N K L D)`` memory, where ``D`` is the
    number of signal channels. Works under ``jax.jit`` and ``jax.vmap``
    with static ``dt`` and ``window``, and is differentiable with respect to
    ``signal``.

    References
    ----------
    .. [1] Schwartz, O., Pillow, J. W., Rust, N. C., & Simoncelli, E. P.
           (2006). Spike-triggered neural characterization. Journal of
           Vision, 6(4), 13.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import spike_triggered_average
        >>> signal = jnp.arange(10.) * u.mV
        >>> raster = jnp.zeros((10, 1)).at[jnp.array([1, 4, 6]), 0].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> sta, lags = spike_triggered_average(st, signal, dt=1. * u.ms,
        ...                                     window=(2. * u.ms, 1. * u.ms))
        >>> sta     # the spike at 1 ms is excluded
        Quantity([[3. 4. 5.]], "mV")
        >>> lags
        Quantity([-2. -1.  0.], "ms")
    """
    _check_trains(trains)
    unit = trains._unit
    times = trains._times_m
    dtype = times.dtype
    step = _static_time(dt, unit, 'dt')
    if step <= 0:
        raise ValueError(f'`dt` must be positive, got {dt!r}.')
    t_before, t_after = _parse_window(window, unit)
    n_lags = _bins_for(t_before + t_after, step, dtype, 'window / dt')

    sig_m, sig_unit = _split(signal)
    sig = jnp.asarray(sig_m)
    if sig.ndim == 0:
        raise ValueError('`signal` must have shape (T, ...), got a scalar.')
    if not jnp.issubdtype(sig.dtype, jnp.floating):
        sig = sig.astype(jnp.result_type(float))
    n_steps = sig.shape[0]
    flat = sig.reshape(n_steps, -1)

    t0 = trains._t_start_m
    t_end = t0 + n_steps * step
    mask = trains.mask
    safe = jnp.where(mask, times, t0)
    start_time = safe - t_before
    eps = jnp.finfo(dtype).eps
    tol = _EPS_FACTOR * eps * jnp.maximum(jnp.maximum(jnp.abs(safe), jnp.abs(t0)), step)
    # first sample of each window, floor() corrected for round-off on the grid
    q = (start_time - t0) / step
    first = jnp.floor(q)
    first = jnp.where(start_time - t0 >= (first + 1) * step - tol, first + 1, first)
    first = first.astype(jnp.int32)
    usable = (
        mask
        & (start_time >= t0 - tol)
        & (safe + t_after <= t_end + tol)
        & (first >= 0)
        & (first + n_lags <= n_steps)
    )
    idx = jnp.clip(first[..., None] + jnp.arange(n_lags, dtype=jnp.int32), 0, max(n_steps - 1, 0))
    windows = flat[idx]  # (N, K, L, D)
    total = jnp.sum(jnp.where(usable[..., None, None], windows, 0), axis=1)  # (N, L, D)
    n_used = jnp.sum(usable, axis=-1).astype(total.dtype)[:, None, None]
    sta = _safe_div(total, n_used, n_used > 0)
    sta = sta.reshape((trains.n_neurons, n_lags) + sig.shape[1:])
    lags = (-t_before + jnp.arange(n_lags, dtype=dtype) * step).astype(dtype)
    return _wrap(sta, sig_unit), _wrap(lags, unit)
