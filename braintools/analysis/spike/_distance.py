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

"""Time-resolved spike-train distances of Kreuz et al.: ISI- and SPIKE-distance.

Both measures compare two spike trains through a profile that is defined at
every instant of the recording and average that profile over time. The
profiles are piecewise constant (ISI) or piecewise linear (SPIKE) between the
merged spike times of the two trains, so the time averages are computed
exactly, interval by interval, without a time grid.

The edge conventions follow PySpike 0.9 [3]_: auxiliary spikes at the
recording bounds, empty trains replaced by the two bounds, duplicate spike
times removed.
"""

from typing import Any, Optional

import brainunit as u
import jax
import jax.numpy as jnp

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains, _split

__all__ = [
    'isi_distance',
    'spike_distance',
]


# ----------------------------------------------------------------------------
# input handling
# ----------------------------------------------------------------------------

def _window_time(value: Any, unit: Optional[u.Unit], name: str, where: str) -> Any:
    """Express a window bound as a plain value in the unit of the spike times.

    Plain numbers are milliseconds (the ``brainstate`` convention): they are
    used as they are for unitless trains (whose times are milliseconds) and
    converted from milliseconds to ``unit`` for Quantity trains. A Quantity is
    converted to ``unit`` (to milliseconds for unitless trains).
    """
    target = u.ms if unit is None else unit
    if isinstance(value, u.Quantity) and not value.unit.is_unitless:
        try:
            return value.to_decimal(target)
        except Exception as e:  # incompatible dimension
            raise TypeError(f'{where}: `{name}` must be a time, got {value!r}.') from e
    value = _split(value)[0]
    if unit is None or unit == u.ms:
        return value
    return value * float(u.Quantity(1., unit=u.ms).to_decimal(unit))


def _resolve_pair(a: Any, b: Any, t_start: Any, t_stop: Any, where: str):
    """Validate the two trains and return their times in a common unit.

    Returns ``(times_a, counts_a, times_b, counts_b, t0, t1)`` as plain arrays,
    with ``b`` converted to the unit of ``a``.
    """
    for name, st in (('a', a), ('b', b)):
        if not isinstance(st, SpikeTrains):
            raise TypeError(
                f'{where}: `{name}` must be a braintools.spike.SpikeTrains, got '
                f'{type(st).__name__}. Build one with SpikeTrains.from_dense, '
                f'from_events or from_list.'
            )
        if st.times.ndim != 2:
            raise ValueError(
                f'{where}: `{name}` must be an unbatched SpikeTrains with times of shape '
                f'(n_neurons, capacity), got shape {st.times.shape}; use jax.vmap for '
                f'batches of trials.'
            )
    if a.n_neurons != b.n_neurons:
        raise ValueError(
            f'{where}: `a` and `b` are compared row by row and must have the same number '
            f'of neurons, got {a.n_neurons} and {b.n_neurons}.'
        )
    unit = a._unit
    unit_b = b._unit
    if (unit is None) != (unit_b is None):
        raise TypeError(
            f'{where}: one SpikeTrains carries time units and the other does not '
            f'({unit} vs {unit_b}); build both with the same kind of times.'
        )
    if unit is None or unit_b == unit:
        times_b, b0, b1 = b._times_m, b._t_start_m, b._t_stop_m
    else:
        times_b = b.times.to_decimal(unit)
        b0 = b.t_start.to_decimal(unit)
        b1 = b.t_stop.to_decimal(unit)
    times_a = a._times_m
    dtype = jnp.result_type(times_a, times_b)
    times_a = jnp.asarray(times_a, dtype=dtype)
    times_b = jnp.asarray(times_b, dtype=dtype)
    # PySpike's ``reconcile_spike_trains``: the common interval spans both trains.
    if t_start is None:
        t0 = jnp.minimum(jnp.asarray(a._t_start_m, dtype), jnp.asarray(b0, dtype))
    else:
        t0 = jnp.asarray(_window_time(t_start, unit, 't_start', where), dtype=dtype)
    if t_stop is None:
        t1 = jnp.maximum(jnp.asarray(a._t_stop_m, dtype), jnp.asarray(b1, dtype))
    else:
        t1 = jnp.asarray(_window_time(t_stop, unit, 't_stop', where), dtype=dtype)
    if t0.ndim != 0 or t1.ndim != 0:
        raise ValueError(f'{where}: `t_start` and `t_stop` must be scalars.')
    return times_a, a.counts, times_b, b.counts, t0, t1


def _clean_row(x: jax.Array, n: jax.Array, t0: jax.Array, t1: jax.Array):
    """Apply PySpike's spike-train conventions to one padded row.

    Spikes outside ``[t0, t1]`` and repeated spike times are dropped, and an
    empty train is replaced by the two auxiliary spikes ``[t0, t1]``
    (PySpike's ``SpikeTrain.get_spikes_non_empty``). Returns the sorted row,
    padded with ``+inf`` to length ``max(K, 2)``, and its number of spikes.
    """
    if x.shape[0] < 2:
        x = jnp.concatenate([x, jnp.full((2 - x.shape[0],), jnp.inf, x.dtype)])
    k = x.shape[0]
    idx = jnp.arange(k)
    prev = jnp.concatenate([jnp.full((1,), -jnp.inf, x.dtype), x[:-1]])
    keep = (idx < n) & (x >= t0) & (x <= t1) & (x != prev)
    y = jnp.sort(jnp.where(keep, x, jnp.inf))
    m = jnp.sum(keep, dtype=jnp.int32)
    empty = m == 0
    y = jnp.where(empty & (idx == 0), t0, y)
    y = jnp.where(empty & (idx == 1), t1, y)
    m = jnp.where(empty, 2, m)
    return y, m


class _Train:
    """Per-row quantities of a cleaned train shared by both distances."""

    def __init__(self, y: jax.Array, m: jax.Array, t0: jax.Array, t1: jax.Array):
        k = y.shape[0]
        self.y, self.m, self.k = y, m, k
        self.multi = m > 1
        self.first = y[0]
        second = jnp.where(self.multi, y[1], y[0])
        self.last = y[jnp.clip(m - 1, 0, k - 1)]
        prelast = y[jnp.clip(m - 2, 0, k - 1)]
        # ISI before the first spike and after the last one (PySpike edge
        # correction): the longer of the edge gap and the neighbouring ISI.
        self.isi_first = jnp.where(
            self.multi, jnp.maximum(self.first - t0, second - self.first), self.first - t0
        )
        self.isi_last = jnp.where(
            self.multi, jnp.maximum(t1 - self.last, self.last - prelast), t1 - self.last
        )
        # Auxiliary spikes (Kreuz et al. 2015): mirror the first/last ISI
        # outside the recording, but never inside it.
        self.aux0 = jnp.where(self.multi, jnp.minimum(t0, 2 * self.first - second), t0)
        self.aux1 = jnp.where(self.multi, jnp.maximum(t1, 2 * self.last - prelast), t1)

    def locate(self, left: jax.Array):
        """Index of the segment containing each interval that starts at ``left``.

        Returns ``(k, k_prev, k_next)`` where ``k`` is the number of spikes
        ``<= left`` (0 = before the first spike, ``m`` = after the last one).
        """
        k = jnp.searchsorted(self.y, left, side='right')
        return k, jnp.clip(k - 1, 0, self.k - 1), jnp.clip(k, 0, self.k - 1)

    def isi_at(self, left: jax.Array):
        """ISI of the segment containing each interval starting at ``left``."""
        k, kp, kn = self.locate(left)
        middle = self.y[kn] - self.y[kp]
        return jnp.where(k == 0, self.isi_first, jnp.where(k >= self.m, self.isi_last, middle))


def _min_dist(t: jax.Array, other: _Train) -> jax.Array:
    """Distance from each ``t`` to the nearest spike or auxiliary spike of ``other``."""
    k = other.k
    j = jnp.searchsorted(other.y, t, side='left')
    below = jnp.where(j > 0, other.y[jnp.clip(j - 1, 0, k - 1)], jnp.inf)
    above = jnp.where(j < other.m, other.y[jnp.clip(j, 0, k - 1)], jnp.inf)
    d = jnp.minimum(jnp.abs(t - other.aux0), jnp.abs(t - other.aux1))
    return jnp.minimum(d, jnp.minimum(jnp.abs(t - below), jnp.abs(t - above)))


def _intervals(ya: jax.Array, yb: jax.Array, t0: jax.Array, t1: jax.Array):
    """Merged intervals ``[left, right)`` of both trains and the bounds, and their widths."""
    bounds = jnp.stack([t0, t1])
    points = jnp.sort(jnp.concatenate([bounds, ya, yb]))
    left, right = points[:-1], points[1:]
    finite = jnp.isfinite(right)
    width = jnp.where(finite, right - left, 0.)
    left = jnp.where(finite, left, t0)
    right = jnp.where(finite, right, t0)
    return left, right, width


def _average(total: jax.Array, t0: jax.Array, t1: jax.Array) -> jax.Array:
    duration = t1 - t0
    return jnp.where(duration > 0, total / jnp.where(duration > 0, duration, 1), jnp.nan)


def _isi_row(xa, na, xb, nb, t0, t1):
    ya, ma = _clean_row(xa, na, t0, t1)
    yb, mb = _clean_row(xb, nb, t0, t1)
    ta, tb = _Train(ya, ma, t0, t1), _Train(yb, mb, t0, t1)
    left, _, width = _intervals(ya, yb, t0, t1)
    nu_a, nu_b = ta.isi_at(left), tb.isi_at(left)
    big = jnp.maximum(nu_a, nu_b)
    used = width > 0
    profile = jnp.abs(nu_a - nu_b) / jnp.where(used, big, 1)
    total = jnp.sum(jnp.where(used, profile * width, 0.))
    return _average(total, t0, t1)


def _spike_segments(tr: _Train, other: _Train, left: jax.Array, t0, t1):
    """Previous/following spike, their distances to ``other`` and the ISI of each segment."""
    dist = _min_dist(tr.y, other)                          # (K,)
    dist_end = _min_dist(t1[None], other)[0]
    k, kp, kn = tr.locate(left)
    before = k == 0
    after = k >= tr.m
    # A single spike exactly at ``t_start``: PySpike keeps ``t_start`` as the
    # previous spike and ``t_stop`` as the following one, whose distance is
    # measured like a real spike's.
    lone_start = (tr.m == 1) & (tr.first == t0)
    d_last = dist[jnp.clip(tr.m - 1, 0, tr.k - 1)]
    t_prev = jnp.where(before, tr.aux0, jnp.where(after, tr.last, tr.y[kp]))
    t_next = jnp.where(
        before, tr.first, jnp.where(after, jnp.where(lone_start, t1, tr.aux1), tr.y[kn])
    )
    d_prev = jnp.where(before, dist[0], jnp.where(after, d_last, dist[kp]))
    d_next = jnp.where(
        before, dist[0], jnp.where(after, jnp.where(lone_start, dist_end, d_last), dist[kn])
    )
    isi = jnp.where(before, tr.isi_first, jnp.where(after, tr.isi_last, tr.y[kn] - tr.y[kp]))
    return t_prev, t_next, d_prev, d_next, isi


def _spike_row(xa, na, xb, nb, t0, t1):
    ya, ma = _clean_row(xa, na, t0, t1)
    yb, mb = _clean_row(xb, nb, t0, t1)
    ta, tb = _Train(ya, ma, t0, t1), _Train(yb, mb, t0, t1)
    left, right, width = _intervals(ya, yb, t0, t1)
    used = width > 0
    pa, fa, dpa, dfa, isi_a = _spike_segments(ta, tb, left, t0, t1)
    pb, fb, dpb, dfb, isi_b = _spike_segments(tb, ta, left, t0, t1)
    isi_a = jnp.where(used, isi_a, 1)
    isi_b = jnp.where(used, isi_b, 1)
    mean_isi = 0.5 * (isi_a + isi_b)

    def profile(t):
        # Local distance of each train's spikes to the other train, linearly
        # interpolated between its previous and following spike.
        s_a = (dpa * (fa - t) + dfa * (t - pa)) / isi_a
        s_b = (dpb * (fb - t) + dfb * (t - pb)) / isi_b
        return 0.5 * (s_a * isi_b + s_b * isi_a) / (mean_isi * mean_isi)

    # The profile is linear on each merged interval: the trapezoid rule is exact.
    area = 0.5 * (profile(left) + profile(right)) * width
    total = jnp.sum(jnp.where(used, area, 0.))
    return _average(total, t0, t1)


def _rowwise(row_fn, a, b, t_start, t_stop, where):
    xa, na, xb, nb, t0, t1 = _resolve_pair(a, b, t_start, t_stop, where)
    return jax.vmap(row_fn, in_axes=(0, 0, 0, 0, None, None))(xa, na, xb, nb, t0, t1)


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.spike')
def isi_distance(
    a: SpikeTrains,
    b: SpikeTrains,
    t_start: Optional[Any] = None,
    t_stop: Optional[Any] = None,
) -> jax.Array:
    r"""Compute the ISI-distance between two populations, neuron by neuron.

    The ISI-distance of Kreuz et al. [1]_ compares the local firing rates of
    two spike trains. At every instant :math:`t`, let :math:`\nu_a(t)` and
    :math:`\nu_b(t)` be the inter-spike intervals that contain :math:`t` in
    each train. The ISI-profile is

    .. math::

        I(t) = \frac{|\nu_a(t) - \nu_b(t)|}{\max(\nu_a(t), \nu_b(t))},

    and the distance is its time average over the recording,

    .. math::

        D_I = \frac{1}{t_{stop} - t_{start}} \int_{t_{start}}^{t_{stop}} I(t)\, dt .

    It is 0 for identical trains and at most 1.

    Parameters
    ----------
    a, b : SpikeTrains
        The two populations, with the same number of neurons ``N`` (their
        capacities may differ). Neuron ``i`` of ``a`` is compared with neuron
        ``i`` of ``b``. Both must carry compatible time units, or both none.
    t_start, t_stop : float or Quantity, optional
        Averaging interval. A Quantity is converted to the unit of the spike
        times; a plain number is in milliseconds (the ``brainstate``
        convention), so it is used as is for unitless trains and converted
        from milliseconds for trains that carry units. Defaults to
        ``min(a.t_start, b.t_start)`` and ``max(a.t_stop, b.t_stop)``, as
        PySpike does. Spikes outside the interval are ignored.

    Returns
    -------
    jax.Array
        ISI-distances, shape ``(N,)``, dimensionless. ``nan`` when
        ``t_stop <= t_start``.

    Raises
    ------
    TypeError
        If ``a`` or ``b`` is not a ``SpikeTrains``, if only one carries
        units, or if ``t_start``/``t_stop`` is a Quantity that is not a time.
    ValueError
        If the numbers of neurons differ, or a ``SpikeTrains`` is batched.

    See Also
    --------
    spike_distance : The SPIKE-distance, sensitive to spike timing.
    victor_purpura_distance : Edit distance between two spike trains.
    braintools.spike.SpikeTrains : The padded spike-time container.

    Notes
    -----
    The profile is piecewise constant between the merged spike times of the
    two trains, so the integral is computed exactly as a sum over those
    intervals. The edge conventions are those of PySpike 0.9 [3]_ (Kreuz et
    al. [2]_), so the values agree with ``pyspike.isi_distance`` to
    round-off:

    - before the first spike, the ISI is ``max(s_0 - t_start, s_1 - s_0)``,
      and after the last spike ``max(t_stop - s_{n-1}, s_{n-1} - s_{n-2})``;
      with a single spike these are ``s_0 - t_start`` and
      ``t_stop - s_0``;
    - an empty train is replaced by spikes at ``t_start`` and ``t_stop``, so
      two empty trains are at distance 0 and an empty train against a train
      with one spike at ``t_start + T/2`` is at distance 0.5;
    - repeated spike times within a train count once.

    One deviation: PySpike returns ``nan`` when both trains have a single
    spike exactly at ``t_stop``; here the zero-length interval after it is
    ignored and the distance is 0.

    The function works under ``jax.jit`` and ``jax.vmap`` (over trials, or
    over neurons to build a pair matrix, see the examples). It is not
    differentiable in a useful sense: the profile is piecewise constant in
    the spike times. Memory and time are ``O(N (K_a + K_b) log(K_a + K_b))``.

    References
    ----------
    .. [1] Kreuz, T., Haas, J. S., Morelli, A., Abarbanel, H. D. I., & Politi,
           A. (2007). Measuring spike train synchrony. *Journal of
           Neuroscience Methods*, 165(1), 151-161.
    .. [2] Kreuz, T., Mulansky, M., & Bozanic, N. (2015). SPIKY: a graphical
           user interface for monitoring spike train synchrony. *Journal of
           Neurophysiology*, 113(9), 3432-3445.
    .. [3] Mulansky, M., & Kreuz, T. (2016). PySpike - A Python library for
           analyzing spike train synchrony. *SoftwareX*, 5, 183-189.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import isi_distance
        >>> a = SpikeTrains.from_list([[1., 5., 9.], [2., 4., 6., 8.]], t_start=0., t_stop=10.)
        >>> b = SpikeTrains.from_list([[1., 5., 9.], []], t_start=0., t_stop=10.)
        >>> isi_distance(a, b)
        Array([0. , 0.8], dtype=float32)
        >>>
        >>> # From a dense raster (time on axis 0), with units.
        >>> import brainunit as u
        >>> raster = jnp.zeros((100, 2)).at[::10, 0].set(1.).at[::20, 1].set(1.)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> d = isi_distance(st.select(0), st.select(1))
        >>> np.round(np.asarray(d), 4)
        array([0.5], dtype=float32)
        >>>
        >>> # All pairs of neurons of one population: vmap over the first neuron.
        >>> n = st.n_neurons
        >>> pair = jax.vmap(lambda i: isi_distance(st.select(jnp.full((n,), i)), st))
        >>> np.round(np.asarray(pair(jnp.arange(n))), 4)
        array([[0. , 0.5],
               [0.5, 0. ]], dtype=float32)
    """
    return _rowwise(_isi_row, a, b, t_start, t_stop, 'isi_distance')


@set_module_as('braintools.analysis.spike')
def spike_distance(
    a: SpikeTrains,
    b: SpikeTrains,
    t_start: Optional[Any] = None,
    t_stop: Optional[Any] = None,
) -> jax.Array:
    r"""Compute the SPIKE-distance between two populations, neuron by neuron.

    The SPIKE-distance of Kreuz et al. [1]_ compares spike timing. For a
    train :math:`x` at time :math:`t`, let :math:`t_P^x \le t < t_F^x` be the
    previous and following spikes, :math:`x_{ISI} = t_F^x - t_P^x`, and
    :math:`\Delta t_P^x`, :math:`\Delta t_F^x` the distances of those spikes
    to the nearest spike of the other train. The local spike-time difference
    is interpolated between them,

    .. math::

        S_x(t) = \frac{\Delta t_P^x (t_F^x - t) + \Delta t_F^x (t - t_P^x)}{x_{ISI}},

    and the profile weights both trains by their local ISIs,

    .. math::

        S(t) = \frac{S_a(t)\, b_{ISI} + S_b(t)\, a_{ISI}}
                    {2 \langle x_{ISI} \rangle^2},
        \qquad \langle x_{ISI} \rangle = \frac{a_{ISI} + b_{ISI}}{2}.

    The distance is the time average of :math:`S(t)` over the recording. It is
    0 for identical trains and at most 1.

    Parameters
    ----------
    a, b : SpikeTrains
        The two populations, with the same number of neurons ``N`` (their
        capacities may differ). Neuron ``i`` of ``a`` is compared with neuron
        ``i`` of ``b``. Both must carry compatible time units, or both none.
    t_start, t_stop : float or Quantity, optional
        Averaging interval. A Quantity is converted to the unit of the spike
        times; a plain number is in milliseconds (the ``brainstate``
        convention), so it is used as is for unitless trains and converted
        from milliseconds for trains that carry units. Defaults to
        ``min(a.t_start, b.t_start)`` and ``max(a.t_stop, b.t_stop)``, as
        PySpike does. Spikes outside the interval are ignored.

    Returns
    -------
    jax.Array
        SPIKE-distances, shape ``(N,)``, dimensionless. ``nan`` when
        ``t_stop <= t_start``.

    Raises
    ------
    TypeError
        If ``a`` or ``b`` is not a ``SpikeTrains``, if only one carries
        units, or if ``t_start``/``t_stop`` is a Quantity that is not a time.
    ValueError
        If the numbers of neurons differ, or a ``SpikeTrains`` is batched.

    See Also
    --------
    isi_distance : The ISI-distance, sensitive to local firing rates.
    van_rossum_distance : Kernel-based distance between two spike trains.
    braintools.spike.SpikeTrains : The padded spike-time container.

    Notes
    -----
    :math:`S(t)` is linear on every interval between the merged spike times
    of the two trains, so the time average is integrated exactly with the
    trapezoid rule over those intervals.

    The edge conventions are those of PySpike 0.9 [3]_, which follow Kreuz et
    al. [2]_, so the values agree with ``pyspike.spike_distance`` to
    round-off:

    - auxiliary spikes are placed at
      ``min(t_start, s_0 - (s_1 - s_0))`` and
      ``max(t_stop, s_{n-1} + (s_{n-1} - s_{n-2}))`` (at ``t_start`` and
      ``t_stop`` for a single spike); they serve as the previous/following
      spikes at the edges and as candidates for the nearest spike of the
      other train;
    - before the first and after the last spike, :math:`S_x` is held at the
      distance of that first or last spike, and the edge ISIs are those of
      :func:`isi_distance`;
    - a train whose only spike is at ``t_start`` uses ``t_stop`` as its
      following spike;
    - an empty train is replaced by spikes at ``t_start`` and ``t_stop``;
      repeated spike times within a train count once.

    One deviation: PySpike returns ``nan`` when both trains have a single
    spike exactly at ``t_stop``; here the distance is 0.

    The function works under ``jax.jit`` and ``jax.vmap`` (over trials, or
    over neurons to build a pair matrix, see the examples). Gradients with
    respect to the spike times are not supported (the nearest-neighbour
    search is piecewise). Memory and time are
    ``O(N (K_a + K_b) log(K_a + K_b))``.

    References
    ----------
    .. [1] Kreuz, T., Chicharro, D., Houghton, C., Andrzejak, R. G., &
           Mormann, F. (2013). Monitoring spike train synchrony. *Journal of
           Neurophysiology*, 109(5), 1457-1472.
    .. [2] Kreuz, T., Mulansky, M., & Bozanic, N. (2015). SPIKY: a graphical
           user interface for monitoring spike train synchrony. *Journal of
           Neurophysiology*, 113(9), 3432-3445.
    .. [3] Mulansky, M., & Kreuz, T. (2016). PySpike - A Python library for
           analyzing spike train synchrony. *SoftwareX*, 5, 183-189.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from braintools.spike import SpikeTrains
        >>> from braintools.analysis.spike import spike_distance
        >>> a = SpikeTrains.from_list([[1., 5., 9.], [3.]], t_start=0., t_stop=10.)
        >>> b = SpikeTrains.from_list([[1., 5., 9.], [6.]], t_start=0., t_stop=10.)
        >>> np.round(np.asarray(spike_distance(a, b)), 4)
        array([0.    , 0.5566], dtype=float32)
        >>>
        >>> # From a dense raster (time on axis 0), with units.
        >>> import brainunit as u
        >>> raster = jnp.zeros((100, 2)).at[::10, 0].set(1.).at[5::10, 1].set(1.)
        >>> st = SpikeTrains.from_dense(raster, dt=1. * u.ms)
        >>> d = spike_distance(st.select(0), st.select(1))
        >>> d.shape
        (1,)
        >>>
        >>> # All pairs of neurons of one population: vmap over the first neuron.
        >>> n = st.n_neurons
        >>> pair = jax.vmap(lambda i: spike_distance(st.select(jnp.full((n,), i)), st))
        >>> pair(jnp.arange(n)).shape
        (2, 2)
    """
    return _rowwise(_spike_row, a, b, t_start, t_stop, 'spike_distance')
