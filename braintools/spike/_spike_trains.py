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

"""A padded, fixed-shape container for the spike times of a population."""

import math
import operator
from typing import Any, List, Optional, Sequence, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from ._convert import flatten_spikes

__all__ = [
    'SpikeTrains',
]

# Times on a grid (``t_start + i * dt``) carry float round-off, so bin edges are
# compared with a tolerance of a few machine epsilons of the values involved:
# a spike on an edge must not land one bin too early, and a duration that is a
# multiple of ``bin_size`` must not gain an extra bin. The tolerance scales with
# the magnitude of the times (not with the bin index), so spikes a few float
# steps away from an edge are still binned by their actual value.
_EPS_FACTOR = 4


# ----------------------------------------------------------------------------
# unit and tracing helpers
# ----------------------------------------------------------------------------

def _is_traced(*values: Any) -> bool:
    """Whether any of ``values`` is a JAX tracer (inside jit / vmap / grad)."""
    for v in values:
        if isinstance(v, u.Quantity):
            v = v.mantissa
        if isinstance(v, jax.core.Tracer):
            return True
    return False


def _split(x: Any) -> Tuple[Any, Optional[u.Unit]]:
    """Split ``x`` into ``(mantissa, unit)``; ``unit`` is None for unitless input."""
    if isinstance(x, u.Quantity):
        if x.unit.is_unitless:
            return x.mantissa, None
        return x.mantissa, x.unit
    return x, None


def _in_unit(value: Any, unit: Optional[u.Unit], name: str) -> Any:
    """Express ``value`` as a plain number/array in ``unit`` (None = unitless)."""
    if isinstance(value, u.Quantity):
        if value.unit.is_unitless:
            if unit is None:
                return value.mantissa
        elif unit is None:
            raise TypeError(
                f'`{name}` has unit {value.unit}, but the spike times are unitless. '
                f'Pass a plain number, or build the SpikeTrains with times/dt in units.'
            )
        else:
            return value.to_decimal(unit)
    if unit is not None:
        raise TypeError(
            f'`{name}` must be a Quantity compatible with {unit}, because the spike '
            f'times carry units; got the plain value {value!r}.'
        )
    return value


def _wrap(mantissa: Any, unit: Optional[u.Unit]) -> Any:
    """Attach ``unit`` to ``mantissa`` (no-op for unitless data)."""
    if unit is None:
        return mantissa
    return u.Quantity(mantissa, unit=unit)


def _float_dtype():
    return jnp.result_type(float)


def _as_float(x: Any) -> jax.Array:
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(_float_dtype())
    return x


def _static_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f'`{name}` must be an integer, got {value!r}.')
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(
            f'`{name}` must be a static Python integer (it fixes an array shape), '
            f'got {value!r}.'
        ) from None
    if value < minimum:
        raise ValueError(f'`{name}` must be >= {minimum}, got {value}.')
    return value


def _ceil_ratio(numerator: float, denominator: float, dtype: Any) -> int:
    """``ceil(numerator / denominator)`` that ignores round-off of ``dtype`` values."""
    ratio = float(numerator) / float(denominator)
    eps = float(jnp.finfo(dtype).eps)
    return int(math.ceil(ratio - _EPS_FACTOR * eps * max(1.0, abs(ratio))))


def _resolve_capacity(capacity, counts, traced: bool, where: str) -> int:
    """Apply the capacity policy shared by the constructors.

    - ``capacity=None`` with concrete data: ``max(1, max count)``;
    - ``capacity=None`` with traced data: ``ValueError``;
    - explicit capacity with concrete data that does not fit: ``ValueError``.
    """
    if capacity is None:
        if traced:
            raise ValueError(
                f'SpikeTrains.{where}: `capacity` is required when the input is traced '
                f'(inside jit, vmap or grad), because it fixes the padded shape '
                f'(n_neurons, capacity) at trace time. Pass `capacity=` an upper bound on '
                f'the number of spikes of any neuron (for dense input, the number of time '
                f'steps T is always enough), e.g. `SpikeTrains.{where}(..., capacity=100)`.'
            )
        max_count = int(np.max(np.asarray(counts), initial=0))
        return max(1, max_count)
    capacity = _static_int(capacity, 'capacity', 1)
    if not traced:
        counts_np = np.asarray(counts)
        if counts_np.size and int(counts_np.max()) > capacity:
            worst = int(np.argmax(counts_np))
            raise ValueError(
                f'SpikeTrains.{where}: neuron {worst} has {int(counts_np[worst])} spikes, '
                f'more than capacity={capacity}. Increase `capacity`, or pass '
                f'`capacity=None` to size it from the data.'
            )
    return capacity


def _environ_dt():
    try:
        return brainstate.environ.get_dt()
    except KeyError:
        raise ValueError(
            '`dt` was not given and `brainstate.environ` has no time step; pass `dt=` '
            'explicitly or set it with `brainstate.environ.set(dt=...)`.'
        ) from None


# ----------------------------------------------------------------------------
# the container
# ----------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class SpikeTrains:
    r"""Spike times of a population in a padded, fixed-shape layout.

    Neuron :math:`n` has ``counts[n]`` spikes, stored in ascending order in
    ``times[n, :counts[n]]``; the remaining ``capacity - counts[n]`` entries
    of the row are padding with value ``+inf``. Because every array has a
    static shape, a ``SpikeTrains`` can be created, transformed and consumed
    inside ``jax.jit`` and ``jax.vmap``. It is a JAX pytree whose leaves are
    all five fields.

    Parameters
    ----------
    times : array-like or Quantity
        Spike times with shape ``(n_neurons, capacity)``, sorted ascending per
        neuron and padded with ``+inf``. A ``brainunit`` Quantity when the time
        unit is known, otherwise a plain array.
    counts : array-like
        Number of valid entries of every row, shape ``(n_neurons,)``; stored
        as ``int32``.
    t_start : float or Quantity
        Start of the recording, in the unit of ``times``.
    t_stop : float or Quantity
        End of the recording, in the unit of ``times``.
    overflow : bool, default False
        Whether some neuron had more than ``capacity`` spikes, so that spikes
        were dropped (only possible for traced input, see below).

    Attributes
    ----------
    times : jax.Array or Quantity
        Padded spike times, shape ``(n_neurons, capacity)``.
    counts : jax.Array
        Valid entries per neuron, shape ``(n_neurons,)``, ``int32``.
    t_start, t_stop : jax.Array or Quantity
        Scalars in the unit of ``times``. Spike times lie in
        ``[t_start, t_stop]``.
    overflow : jax.Array
        Scalar ``bool``.

    See Also
    --------
    braintools.visualize.spike_raster : Plot a ``SpikeTrains`` as a raster.

    Notes
    -----
    **Capacity and overflow.** The capacity (the padded row length ``K``) is
    part of the array shapes, so it must be known when a function is traced.

    - With concrete (non-traced) input and ``capacity=None``, the capacity is
      computed as ``max(1, largest spike count)``.
    - With traced input (inside ``jit``, ``vmap`` or ``grad``),
      ``capacity=None`` raises ``ValueError``; pass an upper bound instead.
    - With concrete input, a neuron with more than ``capacity`` spikes raises
      ``ValueError``.
    - With traced input, the first ``capacity`` spikes of every neuron are
      kept, ``counts`` is clipped to ``capacity`` and ``overflow`` is set to
      ``True``. Call :meth:`check` on the result outside the transform to
      turn the flag into an error.

    **Units.** When ``dt`` (for :meth:`from_dense`) or the input times carry
    ``brainunit`` units, ``times``, ``t_start`` and ``t_stop`` are Quantities
    in that unit, and time arguments of the methods (``bin_size``, ``t0``,
    ...) must be Quantities with compatible units. Without units, plain
    numbers are used throughout.

    **Transforms.** All methods work under ``jit`` and ``vmap`` except the
    host-only ones: :meth:`from_list`, :meth:`to_events` and :meth:`to_list`
    (their results are ragged) and, for traced data, the default
    ``t_start``/``t_stop`` of :meth:`from_events`. Methods whose output shape
    depends on ``t_start``/``t_stop`` (:meth:`to_dense`, :meth:`bin`) need
    those values to be concrete, or an explicit ``n_steps``/``n_bins``.
    Methods assume an unbatched object; inside ``vmap`` every trial is
    unbatched.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.spike import SpikeTrains
        >>> spikes = jnp.array([[1, 0],
        ...                     [0, 0],
        ...                     [1, 1],
        ...                     [1, 0]])
        >>> st = SpikeTrains.from_dense(spikes, dt=1. * u.ms)
        >>> st.n_neurons, st.capacity
        (2, 3)
        >>> st.counts
        Array([3, 1], dtype=int32)
        >>> st.times
        Quantity([[ 0.  2.  3.]
                  [ 2. inf inf]], "ms")
        >>> st.bin(2. * u.ms)
        Array([[1, 0],
               [2, 1]], dtype=int32)
        >>>
        >>> bool((st.to_dense(dt=1. * u.ms) == spikes.astype(bool)).all())
        True
    """

    __module__ = 'braintools.spike'

    def __init__(
        self,
        times: Any,
        counts: Any,
        t_start: Any,
        t_stop: Any,
        overflow: Any = False,
    ):
        times_m, unit = _split(times)
        times_m = _as_float(times_m)
        if times_m.ndim != 2:
            raise ValueError(
                f'`times` must have shape (n_neurons, capacity), got shape {times_m.shape}.'
            )
        counts = jnp.asarray(counts, dtype=jnp.int32)
        if counts.shape != times_m.shape[:1]:
            raise ValueError(
                f'`counts` must have shape ({times_m.shape[0]},) to match `times`, '
                f'got shape {counts.shape}.'
            )
        t_start = jnp.asarray(_in_unit(t_start, unit, 't_start'), dtype=times_m.dtype)
        t_stop = jnp.asarray(_in_unit(t_stop, unit, 't_stop'), dtype=times_m.dtype)
        if t_start.ndim != 0 or t_stop.ndim != 0:
            raise ValueError('`t_start` and `t_stop` must be scalars.')
        self._set(times_m, counts, t_start, t_stop, jnp.asarray(overflow, dtype=bool), unit)

    def _set(self, times_m, counts, t_start_m, t_stop_m, overflow, unit):
        self.times = _wrap(times_m, unit)
        self.counts = counts
        self.t_start = _wrap(t_start_m, unit)
        self.t_stop = _wrap(t_stop_m, unit)
        self.overflow = overflow

    @classmethod
    def _make(cls, times_m, counts, t_start_m, t_stop_m, overflow, unit) -> 'SpikeTrains':
        """Build from mantissas without validation (internal fast path)."""
        obj = object.__new__(cls)
        obj._set(times_m, counts, t_start_m, t_stop_m, overflow, unit)
        return obj

    # -- pytree ----------------------------------------------------------------

    def tree_flatten(self):
        """Flatten into the five leaves (``times, counts, t_start, t_stop, overflow``)."""
        return (self.times, self.counts, self.t_start, self.t_stop, self.overflow), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Rebuild from :meth:`tree_flatten` output without validating the leaves."""
        obj = object.__new__(cls)
        obj.times, obj.counts, obj.t_start, obj.t_stop, obj.overflow = children
        return obj

    # -- internal accessors ----------------------------------------------------

    @property
    def _unit(self) -> Optional[u.Unit]:
        return _split(self.times)[1]

    @property
    def _times_m(self):
        return _split(self.times)[0]

    @property
    def _t_start_m(self):
        return _split(self.t_start)[0]

    @property
    def _t_stop_m(self):
        return _split(self.t_stop)[0]

    def _require_concrete(self, method: str) -> None:
        if _is_traced(self.times, self.counts, self.t_start, self.t_stop):
            raise TypeError(
                f'SpikeTrains.{method}() is host-only: its result is ragged, so it cannot '
                f'run inside jit/vmap. Call it on a concrete SpikeTrains.'
            )

    # -- properties ------------------------------------------------------------

    @property
    def n_neurons(self) -> int:
        """int: Number of neurons ``N`` (static)."""
        return self.times.shape[-2]

    @property
    def capacity(self) -> int:
        """int: Padded number of spike slots per neuron ``K`` (static)."""
        return self.times.shape[-1]

    @property
    def mask(self) -> jax.Array:
        """jax.Array: ``(N, K)`` bool, True for valid (non-padding) entries."""
        return jnp.arange(self.capacity, dtype=jnp.int32) < self.counts[..., None]

    @property
    def duration(self) -> Any:
        """float or Quantity: ``t_stop - t_start``."""
        return self.t_stop - self.t_start

    # -- constructors ----------------------------------------------------------

    @classmethod
    def from_dense(
        cls,
        spikes: Any,
        dt: Any = None,
        t_start: Any = 0.,
        capacity: Optional[int] = None,
    ) -> 'SpikeTrains':
        """Build from a dense spike raster with time on axis 0.

        Every nonzero entry ``spikes[i, n]`` is one spike of neuron ``n`` at
        time ``t_start + i * dt``; ``t_stop = t_start + T * dt``.

        Parameters
        ----------
        spikes : array-like
            Raster with shape ``(T, N)`` (bool or numeric).
        dt : float or Quantity, optional
            Time step. Defaults to ``brainstate.environ.get_dt()``. A Quantity
            makes the result carry its unit.
        t_start : float or Quantity, default 0
            Time of row 0. Must be a Quantity when ``dt`` is one, unless it is
            zero.
        capacity : int, optional
            Padded number of spike slots per neuron. Required when ``spikes``
            is traced; see the class notes for the overflow policy.

        Returns
        -------
        SpikeTrains
            The spike trains. Works under ``jit`` and ``vmap`` when
            ``capacity`` is given.

        Raises
        ------
        ValueError
            If ``spikes`` is not 2-D, if ``capacity`` is None for traced input,
            or if a neuron has more than ``capacity`` spikes (concrete input).
        TypeError
            If ``t_start`` and ``dt`` disagree about units.

        Examples
        --------
        .. code-block:: python

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_dense(jnp.array([[0, 1], [1, 1]]), dt=0.5)
            >>> st.times
            Array([[0.5, inf],
                   [0. , 0.5]], dtype=float32)
            >>>
            >>> trials = jnp.zeros((8, 100, 3)).at[:, ::10, 0].set(1.)
            >>> batched = jax.vmap(lambda s: SpikeTrains.from_dense(s, dt=0.1, capacity=10))(trials)
            >>> batched.times.shape
            (8, 3, 10)
        """
        if dt is None:
            dt = _environ_dt()
        dt_m, unit = _split(dt)
        if unit is not None and not isinstance(t_start, u.Quantity):
            if _is_traced(t_start) or np.any(np.asarray(t_start) != 0):
                raise TypeError(
                    f'`t_start` must be a Quantity compatible with {unit} because `dt` '
                    f'is a Quantity; got the plain value {t_start!r}.'
                )
            t0 = 0.
        else:
            t0 = _in_unit(t_start, unit, 't_start')

        spikes = jnp.asarray(spikes)
        if spikes.ndim != 2:
            raise ValueError(f'`spikes` must have shape (T, N), got shape {spikes.shape}.')
        n_steps, n_neurons = spikes.shape
        fired = spikes != 0
        counts = jnp.sum(fired, axis=0, dtype=jnp.int32)
        traced = _is_traced(spikes)
        k = _resolve_capacity(capacity, counts, traced, 'from_dense')

        # The (j+1)-th spike of a neuron is at the first step where its running
        # spike count reaches j+1: a binary search per neuron, O(T N + N K log T).
        running = jnp.cumsum(fired, axis=0, dtype=jnp.int32)
        targets = jnp.arange(1, k + 1, dtype=jnp.int32)
        steps = jax.vmap(lambda c: jnp.searchsorted(c, targets, side='left'), in_axes=1)(running)

        fdt = _float_dtype()
        t0 = jnp.asarray(t0, dtype=fdt)
        kept = jnp.minimum(counts, k)
        valid = jnp.arange(k, dtype=jnp.int32) < kept[:, None]
        times_m = jnp.where(valid, t0 + steps.astype(fdt) * dt_m, jnp.inf).astype(fdt)
        t_stop = jnp.asarray(t0 + n_steps * dt_m, dtype=fdt)
        overflow = jnp.any(counts > k)
        return cls._make(times_m, kept, t0, t_stop, overflow, unit)

    @classmethod
    def from_events(
        cls,
        times: Any,
        neuron_ids: Any,
        n_neurons: int,
        t_start: Any = None,
        t_stop: Any = None,
        capacity: Optional[int] = None,
    ) -> 'SpikeTrains':
        """Build from flat event arrays ``(times, neuron_ids)``.

        Events may come in any order; they are sorted per neuron.

        Parameters
        ----------
        times : array-like or Quantity
            Spike times, shape ``(M,)``.
        neuron_ids : array-like of int
            Neuron index of every spike, shape ``(M,)``, in ``[0, n_neurons)``.
        n_neurons : int
            Number of neurons ``N`` (static).
        t_start, t_stop : float or Quantity, optional
            Recording bounds, in units compatible with ``times``. Default to
            the minimum and maximum of ``times`` (0 for no events), which
            requires concrete ``times``.
        capacity : int, optional
            Padded number of spike slots per neuron. Required when any input
            is traced; see the class notes for the overflow policy.

        Returns
        -------
        SpikeTrains
            The spike trains. Works under ``jit`` and ``vmap`` when
            ``capacity``, ``t_start`` and ``t_stop`` are given.

        Raises
        ------
        ValueError
            If the shapes differ; if ``capacity``, ``t_start`` or ``t_stop`` is
            missing for traced input; or, for concrete input, if a neuron id is
            out of range, a time lies outside ``[t_start, t_stop]``, or a
            neuron has more than ``capacity`` spikes.

        Notes
        -----
        For traced input, events with an out-of-range neuron id or a time
        outside ``[t_start, t_stop]`` are dropped instead of raising.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_events(jnp.array([3., 1., 2.]) * u.ms,
            ...                              jnp.array([0, 0, 1]), n_neurons=2)
            >>> st.times
            Quantity([[ 1.  3.]
                      [ 2. inf]], "ms")
            >>> st.t_start, st.t_stop
            (Quantity(1., "ms"), Quantity(3., "ms"))
        """
        times_m, unit = _split(times)
        times_m = _as_float(times_m).ravel()
        ids = jnp.asarray(neuron_ids)
        if ids.size and not jnp.issubdtype(ids.dtype, jnp.integer):
            raise TypeError(f'`neuron_ids` must be integers, got dtype {ids.dtype}.')
        ids = ids.ravel().astype(jnp.int32)
        if ids.shape != times_m.shape:
            raise ValueError(
                f'`times` has {times_m.shape[0]} entries but `neuron_ids` has {ids.shape[0]}.'
            )
        n = _static_int(n_neurons, 'n_neurons', 0)
        traced = _is_traced(times, neuron_ids, t_start, t_stop)
        fdt = times_m.dtype

        def _bound(value, name, reduce):
            if value is not None:
                return jnp.asarray(_in_unit(value, unit, name), dtype=fdt)
            if _is_traced(times):
                raise ValueError(
                    f'SpikeTrains.from_events: `{name}` defaults to the {reduce.__name__} of '
                    f'`times`, which needs concrete times; pass `{name}=` explicitly '
                    f'inside jit/vmap.'
                )
            host = np.asarray(times_m)
            return jnp.asarray(reduce(host) if host.size else 0., dtype=fdt)

        t0 = _bound(t_start, 't_start', np.min)
        t1 = _bound(t_stop, 't_stop', np.max)

        id_ok = (ids >= 0) & (ids < n)
        time_ok = (times_m >= t0) & (times_m <= t1)
        if not traced:
            if not bool(jnp.all(id_ok)):
                bad = np.asarray(ids)[~np.asarray(id_ok)]
                raise ValueError(
                    f'neuron_ids must lie in [0, {n}); got {bad[:5].tolist()}.'
                )
            if not bool(jnp.all(time_ok)):
                raise ValueError(
                    'some spike times lie outside [t_start, t_stop] '
                    f'= [{float(t0)}, {float(t1)}]; widen the bounds or filter the events.'
                )
        valid = id_ok & time_ok
        # Invalid events go to a sentinel neuron ``n``, sorted after all valid ones
        # and dropped by the out-of-bounds scatter below.
        ids_eff = jnp.where(valid, ids, n)
        counts_all = jnp.bincount(ids_eff, length=n + 1).astype(jnp.int32)
        counts = counts_all[:n]
        k = _resolve_capacity(capacity, counts, traced, 'from_events')

        order = jnp.lexsort((times_m, ids_eff))  # by neuron, then by time
        ids_sorted = ids_eff[order]
        times_sorted = times_m[order]
        starts = jnp.cumsum(counts_all) - counts_all
        rank = jnp.arange(ids_sorted.shape[0], dtype=jnp.int32) - starts[ids_sorted]
        slot = jnp.where(rank < k, rank, k)
        out = jnp.full((n, k), jnp.inf, dtype=fdt).at[ids_sorted, slot].set(times_sorted, mode='drop')
        overflow = jnp.any(counts > k)
        return cls._make(out, jnp.minimum(counts, k), t0, t1, overflow, unit)

    @classmethod
    def from_list(
        cls,
        trains: Sequence[Any],
        t_start: Any = None,
        t_stop: Any = None,
        capacity: Optional[int] = None,
    ) -> 'SpikeTrains':
        """Build from a ragged list with one spike-time array per neuron (host-only).

        Parameters
        ----------
        trains : sequence of array-like or Quantity
            ``trains[n]`` holds the spike times of neuron ``n``; empty trains
            are allowed. Either all non-empty trains are Quantities with
            compatible units, or none is.
        t_start, t_stop : float or Quantity, optional
            Recording bounds; default to the earliest and latest spike.
        capacity : int, optional
            Padded number of spike slots per neuron; defaults to the largest
            train length (at least 1).

        Returns
        -------
        SpikeTrains
            The spike trains.

        Raises
        ------
        TypeError
            If Quantity and plain trains are mixed.
        ValueError
            As for :meth:`from_events`.

        Examples
        --------
        .. code-block:: python

            >>> import numpy as np
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_list([np.array([0.5, 0.1]), np.array([]), [0.3]],
            ...                            t_start=0., t_stop=1.)
            >>> st.counts
            Array([2, 0, 1], dtype=int32)
            >>> [t.tolist() for t in st.to_list()]
            [[0.10000000149011612, 0.5], [], [0.30000001192092896]]
        """
        trains = list(trains)
        unit = None
        for t in trains:
            _, t_unit = _split(t)
            if t_unit is not None:
                unit = t_unit
                break
        mantissas = []
        for i, t in enumerate(trains):
            if isinstance(t, u.Quantity) and not t.unit.is_unitless:
                arr = t.to_decimal(unit)
            else:
                if unit is not None and np.size(t) > 0:
                    raise TypeError(
                        f'trains[{i}] is a plain array but other trains carry the unit '
                        f'{unit}; pass all trains as Quantities.'
                    )
                arr = _split(t)[0]
            mantissas.append(np.atleast_1d(np.asarray(arr, dtype=float)).ravel())
        times, ids = flatten_spikes(mantissas)
        return cls.from_events(
            _wrap(times, unit), ids, len(trains),
            t_start=t_start, t_stop=t_stop, capacity=capacity,
        )

    # -- converters ------------------------------------------------------------

    def to_dense(self, dt: Any = None, n_steps: Optional[int] = None) -> jax.Array:
        """Convert to a dense boolean raster with time on axis 0.

        Spike ``t`` goes to row ``round((t - t_start) / dt)``; spikes that
        share a row merge, and spikes on rows ``>= n_steps`` are dropped.

        Parameters
        ----------
        dt : float or Quantity, optional
            Time step, compatible with the spike times. Defaults to
            ``brainstate.environ.get_dt()``.
        n_steps : int, optional
            Number of rows ``T``. Defaults to
            ``round((t_stop - t_start) / dt)``, plus one row if a spike lies
            exactly at ``t_stop`` (as with the default ``t_stop`` of
            :meth:`from_events`), so that no spike is dropped. The default
            needs concrete values; pass it explicitly under ``jit``/``vmap``.

        Returns
        -------
        jax.Array
            Boolean raster of shape ``(T, N)``.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_events(jnp.array([0., 2.]), jnp.array([1, 0]),
            ...                              n_neurons=2, t_start=0., t_stop=3.)
            >>> st.to_dense(dt=1.).astype(int)
            Array([[0, 1],
                   [0, 0],
                   [1, 0]], dtype=int32)
        """
        if dt is None:
            dt = _environ_dt()
        unit = self._unit
        dt_m = _in_unit(dt, unit, 'dt')
        t0, t1 = self._t_start_m, self._t_stop_m
        if n_steps is None:
            if _is_traced(t0, t1, dt_m):
                raise ValueError(
                    'SpikeTrains.to_dense: the number of rows depends on t_start/t_stop, '
                    'which are traced here; pass `n_steps=` explicitly.'
                )
            n_steps = int(np.round((float(t1) - float(t0)) / float(dt_m)))
            if not _is_traced(self._times_m, self.counts):
                rows = np.round((np.asarray(self._times_m) - float(t0)) / float(dt_m))
                rows = rows[np.asarray(self.mask)]
                if rows.size:
                    n_steps = max(n_steps, int(rows.max()) + 1)
        n_steps = _static_int(n_steps, 'n_steps', 0)
        mask = self.mask
        row = jnp.round(jnp.where(mask, (self._times_m - t0) / dt_m, -1.))
        ok = mask & (row >= 0) & (row < n_steps)
        row = jnp.where(ok, row, n_steps).astype(jnp.int32)
        col = jnp.broadcast_to(jnp.arange(self.n_neurons, dtype=jnp.int32)[:, None], row.shape)
        dense = jnp.zeros((n_steps, self.n_neurons), dtype=bool)
        return dense.at[row, col].set(True, mode='drop')

    def _host_trains(self) -> List[np.ndarray]:
        times = np.asarray(self._times_m)
        counts = np.asarray(self.counts)
        valid = np.arange(times.shape[1]) < counts[:, None]
        return np.split(times[valid], np.cumsum(counts)[:-1]) if counts.size else []

    def to_events(self) -> Tuple[Any, np.ndarray]:
        """Return flat ``(times, neuron_ids)`` sorted by time (host-only).

        Ties in time are ordered by neuron index.

        Returns
        -------
        times : numpy.ndarray or Quantity
            Spike times, shape ``(M,)``; a Quantity when the trains carry units.
        neuron_ids : numpy.ndarray
            Neuron index of every spike, shape ``(M,)``.

        Raises
        ------
        TypeError
            If called on a traced ``SpikeTrains`` (inside ``jit``/``vmap``).

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_dense(jnp.array([[0, 1], [1, 1]]), dt=1.)
            >>> times, ids = st.to_events()
            >>> times.tolist(), ids.tolist()
            ([0.0, 1.0, 1.0], [1, 0, 1])
        """
        self._require_concrete('to_events')
        times, ids = flatten_spikes(self._host_trains())
        order = np.lexsort((ids, times))
        return _wrap(times[order], self._unit), ids[order]

    def to_list(self) -> List[Any]:
        """Return one spike-time array per neuron (host-only).

        Returns
        -------
        list of numpy.ndarray or Quantity
            ``N`` arrays of lengths ``counts``; Quantities when the trains
            carry units.

        Raises
        ------
        TypeError
            If called on a traced ``SpikeTrains`` (inside ``jit``/``vmap``).

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_dense(jnp.array([[0, 1], [1, 1]]), dt=1.)
            >>> [t.tolist() for t in st.to_list()]
            [[1.0], [0.0, 1.0]]
        """
        self._require_concrete('to_list')
        unit = self._unit
        return [_wrap(t, unit) for t in self._host_trains()]

    # -- transforms ------------------------------------------------------------

    def bin(self, bin_size: Any, n_bins: Optional[int] = None) -> jax.Array:
        """Count spikes in consecutive bins starting at ``t_start``.

        Bin ``b`` covers ``[t_start + b * bin_size, t_start + (b + 1) * bin_size)``.
        When ``bin_size`` does not divide the duration, the last bin is
        partial. A spike exactly at ``t_stop`` is counted in the last bin when
        that bin reaches ``t_stop``; spikes past the last bin are ignored.

        Parameters
        ----------
        bin_size : float or Quantity
            Bin width, compatible with the spike times.
        n_bins : int, optional
            Number of bins. Defaults to ``ceil(duration / bin_size)`` (at
            least 1 when there are spikes, so a zero-length recording keeps
            them), which needs concrete ``t_start``/``t_stop``/``bin_size``;
            pass it explicitly under ``jit``/``vmap``. Spikes beyond the last
            bin are ignored.

        Returns
        -------
        jax.Array
            Spike counts of shape ``(n_bins, N)``, ``int32``.

        Raises
        ------
        ValueError
            If ``bin_size`` is not positive, or ``n_bins`` is missing for
            traced bounds.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_dense(jnp.ones((5, 1)), dt=1.)
            >>> st.bin(2.)[:, 0]
            Array([2, 2, 1], dtype=int32)
        """
        unit = self._unit
        b = _in_unit(bin_size, unit, 'bin_size')
        t0, t1 = self._t_start_m, self._t_stop_m
        if not _is_traced(b) and float(np.asarray(b)) <= 0:
            raise ValueError(f'`bin_size` must be positive, got {bin_size!r}.')
        if n_bins is None:
            if _is_traced(t0, t1, b):
                raise ValueError(
                    'SpikeTrains.bin: the number of bins depends on t_start/t_stop, '
                    'which are traced here; pass `n_bins=` explicitly.'
                )
            n_bins = max(0, _ceil_ratio(float(t1) - float(t0), float(np.asarray(b)),
                                        self._times_m.dtype))
            if n_bins == 0 and not _is_traced(self.counts) and int(np.sum(self.counts)) > 0:
                n_bins = 1
        n_bins = _static_int(n_bins, 'n_bins', 0)
        mask = self.mask
        times = jnp.where(mask, self._times_m, t0)
        eps = jnp.finfo(times.dtype).eps
        tol = _EPS_FACTOR * eps * jnp.maximum(jnp.maximum(jnp.abs(times), jnp.abs(t0)), jnp.abs(b))
        idx = jnp.floor((times - t0) / b)
        # Correct the floor by comparing against the actual bin edges.
        idx = jnp.where(times < t0 + idx * b - tol, idx - 1, idx)
        idx = jnp.where(times >= t0 + (idx + 1) * b - tol, idx + 1, idx)
        # A spike exactly at t_stop closes the last bin instead of opening a new
        # one, but only when the last bin actually reaches t_stop.
        reaches_stop = t0 + n_bins * b >= t1 - tol
        idx = jnp.where(mask & (idx == n_bins) & (times <= t1) & reaches_stop, n_bins - 1, idx)
        ok = mask & (idx >= 0) & (idx < n_bins)
        idx = jnp.where(ok, idx, n_bins).astype(jnp.int32)
        col = jnp.broadcast_to(jnp.arange(self.n_neurons, dtype=jnp.int32)[:, None], idx.shape)
        counts = jnp.zeros((n_bins, self.n_neurons), dtype=jnp.int32)
        return counts.at[idx, col].add(1, mode='drop')

    def window(self, t0: Any, t1: Any) -> 'SpikeTrains':
        """Restrict to the spikes in ``[t0, t1)``, keeping the capacity.

        The bounds are clipped to the recording, so the result has
        ``t_start = max(t0, t_start)`` and ``t_stop = min(t1, t_stop)``
        (never below the new ``t_start``). A spike exactly at ``t0`` is kept;
        a spike exactly at ``t1`` is dropped, except that a spike at ``t_stop``
        is kept when ``t1`` extends past ``t_stop``. ``overflow`` is carried
        over.

        Parameters
        ----------
        t0, t1 : float or Quantity
            Window bounds, compatible with the spike times.

        Returns
        -------
        SpikeTrains
            The windowed trains, with the same ``(N, K)`` shape.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_dense(jnp.ones((6, 1)), dt=1.)
            >>> w = st.window(2., 4.)
            >>> w.times, w.counts
            (Array([[ 2.,  3., inf, inf, inf, inf]], dtype=float32), Array([2], dtype=int32))
        """
        unit = self._unit
        times_m = self._times_m
        lo_t = jnp.maximum(jnp.asarray(_in_unit(t0, unit, 't0'), times_m.dtype), self._t_start_m)
        t1_m = jnp.asarray(_in_unit(t1, unit, 't1'), times_m.dtype)
        hi_t = jnp.maximum(jnp.minimum(t1_m, self._t_stop_m), lo_t)
        # The recording is closed at t_stop, so a window reaching past it keeps a
        # spike at exactly t_stop; otherwise the window is half-open.
        past_stop = t1_m > self._t_stop_m
        mask = self.mask
        lo = jnp.sum(mask & (times_m < lo_t), axis=-1, dtype=jnp.int32)
        hi = jnp.sum(mask & ((times_m < hi_t) | (past_stop & (times_m <= hi_t))),
                     axis=-1, dtype=jnp.int32)
        counts = hi - lo
        slots = jnp.arange(self.capacity, dtype=jnp.int32)
        src = jnp.clip(lo[:, None] + slots, 0, self.capacity - 1)
        shifted = jnp.take_along_axis(times_m, src, axis=-1)
        new_times = jnp.where(slots < counts[:, None], shifted, jnp.inf)
        return type(self)._make(new_times, counts, lo_t, hi_t, self.overflow, unit)

    def select(self, idx: Any) -> 'SpikeTrains':
        """Return the subset of neurons ``idx``.

        Parameters
        ----------
        idx : int, slice, sequence of int or bool mask
            Neurons to keep, in the given order. An integer keeps a single
            neuron (the result still has a neuron axis). Boolean masks need
            concrete values.

        Returns
        -------
        SpikeTrains
            The selected neurons, same capacity and bounds. ``overflow`` is
            carried over (it is not tracked per neuron).

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> st = SpikeTrains.from_dense(jnp.eye(3), dt=1.)
            >>> st.select([2, 0]).times
            Array([[2.],
                   [0.]], dtype=float32)
        """
        if not isinstance(idx, slice):
            idx = jnp.asarray(idx)
            if idx.ndim == 0:
                # A single neuron keeps the neuron axis, whatever the index type.
                idx = idx[None]
        return type(self)._make(
            self._times_m[idx], self.counts[idx], self._t_start_m, self._t_stop_m,
            self.overflow, self._unit,
        )

    def check(self) -> 'SpikeTrains':
        """Validate the container and raise if spikes were dropped.

        In eager mode this raises on ``overflow`` and checks the layout
        invariants (counts in ``[0, K]``, valid times sorted and within
        ``[t_start, t_stop]``, padding ``+inf``). On a traced object it cannot
        raise and returns ``self`` unchanged; call it on the output of the
        transform instead.

        Returns
        -------
        SpikeTrains
            ``self``, to allow chaining.

        Raises
        ------
        ValueError
            If ``overflow`` is set or an invariant is violated.

        Examples
        --------
        .. code-block:: python

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from braintools.spike import SpikeTrains
            >>> make = jax.jit(lambda s: SpikeTrains.from_dense(s, dt=1., capacity=2))
            >>> st = make(jnp.ones((3, 1)))
            >>> bool(st.overflow)
            True
            >>> try:
            ...     st.check()
            ... except ValueError:
            ...     print('overflow')
            overflow
        """
        if _is_traced(self.times, self.counts, self.t_start, self.t_stop, self.overflow):
            return self
        if bool(np.any(np.asarray(self.overflow))):
            raise ValueError(
                f'SpikeTrains overflow: some neuron had more than capacity={self.capacity} '
                f'spikes, and the extra spikes were dropped. Rebuild with a larger `capacity`.'
            )
        times = np.asarray(self._times_m)
        counts = np.asarray(self.counts)
        t0 = np.asarray(self._t_start_m)
        t1 = np.asarray(self._t_stop_m)
        if np.any(counts < 0) or np.any(counts > self.capacity):
            raise ValueError(f'counts must lie in [0, {self.capacity}].')
        if np.any(t0 > t1):
            raise ValueError('t_start must not exceed t_stop.')
        mask = np.arange(self.capacity) < counts[..., None]
        if np.any(~np.isposinf(times[~mask])):
            raise ValueError('padding entries (beyond counts) must be +inf.')
        lo = np.expand_dims(t0, (-1, -2))
        hi = np.expand_dims(t1, (-1, -2))
        if np.any(mask & ~((times >= lo) & (times <= hi))):
            raise ValueError('valid spike times must lie in [t_start, t_stop].')
        if np.any(mask[..., 1:] & (times[..., 1:] < times[..., :-1])):
            raise ValueError('spike times must be sorted ascending per neuron.')
        return self

    def __repr__(self) -> str:
        shape = f'n_neurons={self.n_neurons}, capacity={self.capacity}'
        if _is_traced(self.times, self.counts, self.t_start, self.t_stop, self.overflow):
            return f'SpikeTrains({shape}, traced)'
        n_spikes = int(np.sum(np.asarray(self.counts)))
        return (
            f'SpikeTrains({shape}, n_spikes={n_spikes}, t_start={self.t_start}, '
            f't_stop={self.t_stop}, overflow={bool(np.any(np.asarray(self.overflow)))})'
        )
