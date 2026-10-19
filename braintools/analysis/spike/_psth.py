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

"""Event-aligned peri-stimulus time histogram of a :class:`~braintools.spike.SpikeTrains`."""

from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains, _wrap
from ._isi import (
    _EPS_FACTOR, _bins_for, _check_trains, _seconds_per, _static_time, _time_value,
)

__all__ = [
    'psth',
]

# Upper bound on the number of elements of the per-step temporaries (edge
# times and searchsorted results). Work is split into batches of this size so
# that memory does not grow with the product of all problem dimensions.
_BATCH_ELEMENTS = 2 ** 21


def _batch_size(n_items: int, per_item: int) -> int:
    """Static batch size so that ``batch * per_item <= _BATCH_ELEMENTS`` (at least 1)."""
    return max(1, min(max(n_items, 1), _BATCH_ELEMENTS // max(per_item, 1)))


def _window_counts(row: jax.Array, anchors: jax.Array, rel_edges: jax.Array, scale: float) -> jax.Array:
    """Count the spikes of one sorted, ``+inf``-padded row in bins around anchors.

    Bin ``i`` around anchor ``a`` is the half-open interval
    ``[a + rel_edges[i], a + rel_edges[i + 1])``. Spikes within a few machine
    epsilons below an edge are counted as lying on it, so grid-aligned times
    that carry float round-off land in the right bin.

    Parameters
    ----------
    row : jax.Array
        Sorted spike times of shape ``(K,)``, padded with ``+inf``.
    anchors : jax.Array
        Finite anchor times of shape ``(A,)``.
    rel_edges : jax.Array
        Bin edges relative to the anchors, shape ``(B + 1,)``, increasing.
    scale : float
        Bin width, used for the round-off tolerance.

    Returns
    -------
    jax.Array
        ``int32`` counts of shape ``(A, B)``.
    """
    edges = anchors[:, None] + rel_edges[None, :]
    eps = jnp.finfo(edges.dtype).eps
    tol = _EPS_FACTOR * eps * jnp.maximum(jnp.abs(edges), scale)
    below = jnp.searchsorted(row, (edges - tol).ravel(), side='left').reshape(edges.shape)
    return jnp.diff(below, axis=-1).astype(jnp.int32)


def _parse_window(window: Sequence[Any], unit, name: str = 'window') -> Tuple[float, float]:
    """Static ``(t_before, t_after)`` in ``unit``; the span must be positive."""
    if not isinstance(window, (tuple, list)) or len(window) != 2:
        raise ValueError(f'`{name}` must be a pair (t_before, t_after), got {window!r}.')
    t_before = _static_time(window[0], unit, f'{name}[0]')
    t_after = _static_time(window[1], unit, f'{name}[1]')
    if t_before + t_after <= 0:
        raise ValueError(
            f'`{name}` = (t_before, t_after) spans [-t_before, t_after) around each '
            f'event and must have positive length; got {window!r}.'
        )
    return t_before, t_after


@set_module_as('braintools.analysis.spike')
def psth(
    trains: SpikeTrains,
    events: Any,
    window: Sequence[Any],
    bin_size: Any,
    *,
    rate: bool = True,
) -> Tuple[jax.Array, Any]:
    r"""Compute the event-aligned peri-stimulus time histogram (PSTH).

    The spikes of every neuron are aligned to each event and counted in
    consecutive bins covering the window ``[-t_before, t_after)`` around the
    event, then summed over the events.

    Parameters
    ----------
    trains : SpikeTrains
        Spike trains with ``N`` neurons.
    events : array-like or Quantity
        Event (trial onset) times of shape ``(E,)``, in the time frame of the
        spike trains. Non-finite entries (e.g. ``+inf`` padding) are ignored,
        so a variable number of events fits a static shape. A plain number is
        in milliseconds.
    window : tuple of (float or Quantity)
        ``(t_before, t_after)``: the histogram spans ``[-t_before, t_after)``
        relative to each event, i.e. ``t_before`` is the time *before* the
        event (a positive value reaches into the past). Static. Plain numbers
        are in milliseconds.
    bin_size : float or Quantity
        Bin width, static and positive. A plain number is in milliseconds.
    rate : bool, default True
        If True, return the trial-averaged rate in Hz, ``count / (E_valid *
        bin_size)``, with ``E_valid`` the number of finite events. If False,
        return the summed spike counts.

    Returns
    -------
    values : jax.Array
        Rates in Hz (float) or counts (``int32``), shape ``(n_bins, N)``, with
        ``n_bins = ceil((t_before + t_after) / bin_size)``. The rate is
        ``nan`` when there is no finite event.
    bin_edges : jax.Array or Quantity
        Edges relative to the event, shape ``(n_bins + 1,)``, starting at
        ``-t_before``; a Quantity when the spike times carry units, otherwise
        plain milliseconds.

    Raises
    ------
    TypeError
        If ``trains`` is not a ``SpikeTrains``, or ``window``/``bin_size`` are
        traced.
    ValueError
        If ``events`` is not 1-D, ``bin_size`` is not positive, or the window
        has non-positive length.

    See Also
    --------
    cross_correlogram : Lag histograms with spikes as the alignment events.
    braintools.spike.SpikeTrains.bin : Binning from the recording start.

    Notes
    -----
    Bin ``i`` is the half-open interval ``[edge_i, edge_{i+1})``; a spike
    exactly on an edge belongs to the later bin. When ``bin_size`` does not
    divide the window, the last bin is still a full bin and ends after
    ``t_after``. Windows extending beyond the recording are not corrected:
    the missing part simply contributes no spikes.

    Spike counts are found with ``searchsorted`` on the sorted spike times,
    in ``O(E N n_bins log K)`` time. The events are processed in batches of
    a static size chosen so that each step's temporaries hold at most about
    :math:`2^{21}` elements (at least one event), and the counts are
    accumulated with ``lax.scan``; peak memory is therefore
    ``O(N K + N n_bins + 2^21)`` rather than ``O(N E n_bins)``. Under an
    outer ``jax.vmap`` the per-step memory is multiplied by the batch size of
    that ``vmap``. Works under ``jax.jit`` and ``jax.vmap``
    (e.g. over trials or sessions) with static ``window`` and ``bin_size``;
    the counts are not differentiable.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import psth
        >>> raster = jnp.zeros((100, 1)).at[jnp.array([20, 21, 61, 63]), 0].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> events = jnp.array([20., 60., jnp.inf]) * u.ms   # last event is padding
        >>> counts, edges = psth(st, events, window=(2. * u.ms, 4. * u.ms),
        ...                      bin_size=2. * u.ms, rate=False)
        >>> counts[:, 0]
        Array([0, 3, 1], dtype=int32)
        >>> edges
        Quantity([-2.  0.  2.  4.], "ms")
        >>> rates, _ = psth(st, events, window=(2. * u.ms, 4. * u.ms), bin_size=2. * u.ms)
        >>> rates[:, 0]
        Array([  0., 750., 250.], dtype=float32)
    """
    _check_trains(trains)
    unit = trains._unit
    times = trains._times_m
    dtype = times.dtype
    t_before, t_after = _parse_window(window, unit)
    b = _static_time(bin_size, unit, 'bin_size')
    if b <= 0:
        raise ValueError(f'`bin_size` must be positive, got {bin_size!r}.')
    n_bins = _bins_for(t_before + t_after, b, dtype, 'n_bins')
    rel_edges = (-t_before + jnp.arange(n_bins + 1, dtype=dtype) * b).astype(dtype)

    ev = jnp.asarray(_time_value(events, unit, 'events'), dtype=dtype)
    if ev.ndim != 1:
        raise ValueError(f'`events` must have shape (E,), got shape {ev.shape}.')
    valid = jnp.isfinite(ev)
    anchors = jnp.where(valid, ev, 0).astype(dtype)

    # Scan over batches of events, accumulating (N, n_bins) counts, so the
    # (N, E, n_bins) per-event counts are never materialised at once.
    n_events = ev.shape[0]
    batch = _batch_size(n_events, times.shape[0] * (n_bins + 1))
    n_steps = -(-n_events // batch)
    pad = n_steps * batch - n_events
    chunks = jnp.concatenate([anchors, jnp.zeros(pad, dtype)]).reshape(n_steps, batch)
    keep = jnp.concatenate([valid, jnp.zeros(pad, bool)]).reshape(n_steps, batch)

    def step(acc, chunk):
        anchors_c, keep_c = chunk
        per_event = jax.vmap(lambda row: _window_counts(row, anchors_c, rel_edges, b))(times)  # (N, batch, B)
        return acc + jnp.sum(jnp.where(keep_c[None, :, None], per_event, 0), axis=1, dtype=jnp.int32), None

    init = jnp.zeros((times.shape[0], n_bins), dtype=jnp.int32)
    counts = jax.lax.scan(step, init, (chunks, keep))[0].T
    if rate:
        n_valid = jnp.sum(valid).astype(dtype)
        per_second = 1. / (b * _seconds_per(unit))  # host float64, so 1 / 2 ms is exactly 500 Hz
        values = counts.astype(dtype) * per_second / n_valid
    else:
        values = counts
    return values, _wrap(rel_edges, unit)
