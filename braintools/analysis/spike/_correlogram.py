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

"""Spike-time cross- and auto-correlograms of :class:`~braintools.spike.SpikeTrains`."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains, _wrap
from ._isi import _EPS_FACTOR, _bins_for, _check_trains, _static_time, _time_value
from ._psth import _batch_size

__all__ = [
    'cross_correlogram',
    'auto_correlogram',
]


def _lag_grid(unit, dtype, bin_size: Any, max_lag: Any):
    """Static half-bin count ``M``, bin width and relative edges/centres of the lag bins."""
    b = _static_time(bin_size, unit, 'bin_size')
    if b <= 0:
        raise ValueError(f'`bin_size` must be positive, got {bin_size!r}.')
    m = _static_time(max_lag, unit, 'max_lag')
    if m < 0:
        raise ValueError(f'`max_lag` must be non-negative, got {max_lag!r}.')
    half = _bins_for(m, b, dtype, 'max_lag / bin_size')
    k = jnp.arange(-half, half + 1, dtype=dtype)
    edges = ((jnp.arange(2 * half + 2, dtype=dtype) - half - 0.5) * b).astype(dtype)
    return half, b, edges, (k * b).astype(dtype)


def _lag_counts(row_b, row_a, valid_a, edges, half: int, scale: float):
    """Lag histogram of one neuron pair, with a tie rule symmetric about lag 0.

    ``edges`` are the ``2 * half + 2`` bin edges ``(i - half - 1/2) * scale``.
    A lag on a positive edge goes to the bin above it, a lag on a negative
    edge to the bin below it, i.e. always to the bin farther from 0. Lags
    within a few machine epsilons of an edge count as lying on it.
    """
    anchors = jnp.where(valid_a, row_a, 0)

    def n_before(rel, side, sign):
        at = anchors[:, None] + rel[None, :]
        tol = _EPS_FACTOR * jnp.finfo(at.dtype).eps * jnp.maximum(jnp.abs(at), scale)
        found = jnp.searchsorted(row_b, (at + sign * tol).ravel(), side=side)
        return found.reshape(at.shape)

    # negative edges: number of b-spikes with lag <= edge; positive: lag < edge
    cum = jnp.concatenate([n_before(edges[:half + 1], 'right', 1.),
                           n_before(edges[half + 1:], 'left', -1.)], axis=1)
    per_spike = jnp.diff(cum, axis=1)
    return jnp.sum(jnp.where(valid_a[:, None], per_spike, 0), axis=0, dtype=jnp.int32)


@set_module_as('braintools.analysis.spike')
def cross_correlogram(
    a: SpikeTrains,
    b: SpikeTrains,
    bin_size: Any,
    max_lag: Any,
) -> Tuple[jax.Array, Any]:
    r"""Compute the spike-time cross-correlogram of paired neurons.

    For every neuron :math:`n`, counts the pairs of spikes
    :math:`(t^a_k, t^b_{k'})` of neuron :math:`n` of ``a`` and neuron
    :math:`n` of ``b`` by their lag :math:`\tau = t^b_{k'} - t^a_k`, in
    ``n_bins = 2 * ceil(max_lag / bin_size) + 1`` bins centred on the lags
    :math:`j \cdot \text{bin\_size}`, :math:`j = -M, \dots, M`. A positive
    lag means the spike of ``b`` follows the spike of ``a``.

    Parameters
    ----------
    a, b : SpikeTrains
        Spike trains with the same number of neurons ``N``; neuron ``n`` of
        ``a`` is paired with neuron ``n`` of ``b`` (row-wise pairs). The
        times of ``b`` are converted to the unit of ``a``.
    bin_size : float or Quantity
        Width of the lag bins, static and positive. A plain number is in
        milliseconds.
    max_lag : float or Quantity
        Largest lag of interest, static and non-negative. A plain number is
        in milliseconds.

    Returns
    -------
    counts : jax.Array
        ``int32`` pair counts of shape ``(N, n_bins)``.
    lags : jax.Array or Quantity
        Bin centres ``(-M, ..., M) * bin_size``, shape ``(n_bins,)``; a
        Quantity in the unit of ``a`` when it carries units, otherwise plain
        milliseconds.

    Raises
    ------
    TypeError
        If ``a`` or ``b`` is not a ``SpikeTrains``, or ``bin_size``/``max_lag``
        are traced.
    ValueError
        If the neuron counts differ, ``bin_size`` is not positive or
        ``max_lag`` is negative.

    See Also
    --------
    auto_correlogram : The correlogram of a train with itself.
    psth : Event-aligned histograms with the same counting.
    braintools.spike.SpikeTrains.select : Pick neurons to pair up.

    Notes
    -----
    Bin :math:`j` collects the lags whose nearest bin centre is
    :math:`j b`. A lag exactly half-way between two centres,
    :math:`\pm (j + \tfrac12) b`, goes to the bin *farther* from lag 0 on
    both sides, so the correlogram of a train with itself is exactly
    mirror-symmetric. Bin :math:`j > 0` is :math:`[(j - \tfrac12) b,
    (j + \tfrac12) b)`, bin :math:`-j` is :math:`(-(j + \tfrac12) b,
    -(j - \tfrac12) b]`, and the centre bin is the open interval
    :math:`(-\tfrac12 b, \tfrac12 b)`. Together the bins cover a little
    more than ``[-max_lag, max_lag]`` [1]_. Lags within a few machine
    epsilons of a half-way point count as lying on it, so grid-aligned spike
    times with float round-off are binned consistently.

    ``elephant.spike_train_correlation.cross_correlation_histogram`` works on
    *binned* trains and counts the difference of bin indices. Both agree
    when the spike times lie on the ``bin_size`` grid; for off-grid spikes
    this function bins the exact time differences instead, so it does not
    depend on where the bin grid starts.

    For every spike of ``a``, the spikes of ``b`` before each bin edge are
    counted with ``searchsorted``, in ``O(N K_a n_bins log K_b)`` time. The
    neurons are processed with ``jax.lax.map`` in batches of a static size
    chosen so that a batch's temporaries hold at most about :math:`2^{21}`
    elements (at least one neuron, ``K_a (n_bins + 1)`` elements), so peak
    memory is ``O(N (K_a + K_b) + max(2^{21}, K_a n_bins))`` instead of
    ``O(N K_a n_bins)``. Under an outer ``jax.vmap`` the per-batch memory is
    multiplied by the size of that ``vmap``. The function works under ``jax.jit`` and ``jax.vmap`` with static
    ``bin_size`` and ``max_lag``; the counts are not differentiable. For all
    pairs of neurons, vmap over index pairs with
    :meth:`~braintools.spike.SpikeTrains.select` (see the example).

    References
    ----------
    .. [1] Perkel, D. H., Gerstein, G. L., & Moore, G. P. (1967). Neuronal
           spike trains and stochastic point processes. II. Simultaneous
           spike trains. Biophysical Journal, 7(4), 419-440.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import cross_correlogram
        >>> raster = jnp.zeros((20, 3)).at[jnp.array([2, 10]), 0].set(1)
        >>> raster = raster.at[jnp.array([3, 11]), 1].set(1).at[5, 2].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> counts, lags = cross_correlogram(st.select([0]), st.select([1]),
        ...                                  bin_size=1. * u.ms, max_lag=2. * u.ms)
        >>> counts
        Array([[0, 0, 0, 2, 0]], dtype=int32)
        >>> lags
        Quantity([-2. -1.  0.  1.  2.], "ms")
        >>>
        >>> # all pairs i < j, by vmapping over index pairs
        >>> i, j = jnp.triu_indices(3, k=1)
        >>> pairs = jax.vmap(lambda i, j: cross_correlogram(
        ...     st.select(i), st.select(j), 1. * u.ms, 2. * u.ms)[0][0])(i, j)
        >>> pairs.shape
        (3, 5)
    """
    _check_trains(a, 'a')
    _check_trains(b, 'b')
    if a.n_neurons != b.n_neurons:
        raise ValueError(
            f'`a` and `b` must have the same number of neurons (row-wise pairs), got '
            f'{a.n_neurons} and {b.n_neurons}; use SpikeTrains.select to pair neurons.'
        )
    unit = a._unit
    ta = a._times_m
    dtype = ta.dtype
    tb = jnp.asarray(_time_value(b.times, unit, 'b.times'), dtype=dtype)
    half, width, edges, centres = _lag_grid(unit, dtype, bin_size, max_lag)

    def one(args):
        row_a, valid_a, row_b = args
        return _lag_counts(row_b, row_a, valid_a, edges, half, width)

    batch = _batch_size(a.n_neurons, a.capacity * edges.shape[0])
    if a.n_neurons == 0:
        counts = jnp.zeros((0, centres.shape[0]), dtype=jnp.int32)
    else:
        counts = jax.lax.map(one, (ta, a.mask, tb), batch_size=batch)
    return counts, _wrap(centres, unit)


@set_module_as('braintools.analysis.spike')
def auto_correlogram(
    trains: SpikeTrains,
    bin_size: Any,
    max_lag: Any,
) -> Tuple[jax.Array, Any]:
    r"""Compute the spike-time auto-correlogram of every neuron.

    This is :func:`cross_correlogram` of ``trains`` with itself, with the
    zero-lag pairs of a spike with itself removed. Distinct spikes at the
    same time are still counted at lag 0.

    Parameters
    ----------
    trains : SpikeTrains
        Spike trains with ``N`` neurons.
    bin_size : float or Quantity
        Width of the lag bins, static and positive. A plain number is in
        milliseconds.
    max_lag : float or Quantity
        Largest lag of interest, static and non-negative. A plain number is
        in milliseconds.

    Returns
    -------
    counts : jax.Array
        ``int32`` pair counts of shape ``(N, n_bins)`` with
        ``n_bins = 2 * ceil(max_lag / bin_size) + 1``; symmetric about the
        centre bin.
    lags : jax.Array or Quantity
        Bin centres, shape ``(n_bins,)``.

    Raises
    ------
    TypeError
        If ``trains`` is not a ``SpikeTrains``.
    ValueError
        As for :func:`cross_correlogram`.

    See Also
    --------
    cross_correlogram

    Notes
    -----
    Equal to ``elephant.spike_train_correlation.cross_correlation_histogram``
    of a binned train with itself minus the spike count in the centre bin,
    for grid-aligned spikes. Works under ``jax.jit`` and ``jax.vmap``; not
    differentiable.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import auto_correlogram
        >>> raster = jnp.zeros((10, 1)).at[jnp.array([1, 3, 4]), 0].set(1)
        >>> st = SpikeTrains.from_dense(raster, dt=1.)
        >>> counts, lags = auto_correlogram(st, bin_size=1., max_lag=3.)
        >>> counts
        Array([[1, 1, 1, 0, 1, 1, 1]], dtype=int32)
        >>> lags
        Array([-3., -2., -1.,  0.,  1.,  2.,  3.], dtype=float32)
    """
    _check_trains(trains)
    counts, lags = cross_correlogram(trains, trains, bin_size, max_lag)
    half = counts.shape[-1] // 2
    counts = counts.at[:, half].add(-trains.counts)
    return counts, lags
