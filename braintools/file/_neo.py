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

"""Convert between :class:`~braintools.spike.SpikeTrains` and ``neo.SpikeTrain`` lists.

``neo`` and ``quantities`` are optional dependencies
(``pip install braintools[neo]``); they are imported only when the functions
are called.
"""

import math
from typing import Any, List, Optional, Sequence, Tuple

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains
from ._hdf5 import _import_optional

__all__ = [
    'to_neo',
    'from_neo',
]

# Annotation set on every exported ``neo.SpikeTrain`` when the source
# ``SpikeTrains`` had dropped spikes (``overflow=True``).
_OVERFLOW_ANNOTATION = 'braintools_overflow'


def _seconds_per(unit: u.Unit) -> float:
    return float(u.Quantity(1.0, unit=unit).to_decimal(u.second))


def _time_unit(trains: SpikeTrains, where: str) -> u.Unit:
    """Return the time unit of ``trains``; raise if it has none or is not a time."""
    unit = trains._unit
    if unit is None:
        raise TypeError(
            f'{where}: the SpikeTrains has no time unit, so its times cannot be given a '
            f'physical meaning. Build it with Quantity times or dt (e.g. `dt=0.1 * u.ms`).'
        )
    if unit.dim != u.second.dim:
        raise TypeError(f'{where}: the SpikeTrains times have unit {unit}, which is not a time.')
    return unit


def _check_unbatched(trains: SpikeTrains, where: str) -> None:
    if not isinstance(trains, SpikeTrains):
        raise TypeError(f'{where}: expected a SpikeTrains, got {type(trains).__name__}.')
    if np.ndim(trains._times_m) != 2:
        raise ValueError(
            f'{where}: expected an unbatched SpikeTrains with times of shape '
            f'(n_neurons, capacity), got shape {tuple(np.shape(trains._times_m))}.'
        )


def _pq_unit(pq, unit: u.Unit) -> Tuple[Any, float]:
    """Map a brainunit time unit to a ``quantities`` unit.

    Returns ``(pq_unit, scale)`` where ``scale`` multiplies mantissas in
    ``unit`` to express them in ``pq_unit``. The same-named ``quantities``
    unit is used when it exists and has the same size; otherwise seconds.
    """
    seconds = _seconds_per(unit)
    try:
        candidate = pq.Quantity(1.0, str(unit))
        same = math.isclose(float(candidate.rescale(pq.s).magnitude), seconds, rel_tol=1e-12)
    except (LookupError, ValueError, TypeError, AttributeError):
        same = False
    if same:
        return candidate.units, 1.0
    return pq.s, seconds


def _bu_unit(pq, pq_unit) -> Tuple[u.Unit, float]:
    """Map a ``quantities`` time unit to a brainunit unit.

    Returns ``(unit, scale)`` where ``scale`` multiplies magnitudes in
    ``pq_unit`` to express them in ``unit``. Falls back to seconds.
    """
    try:
        seconds = float(pq.Quantity(1.0, pq_unit).rescale(pq.s).magnitude)
    except ValueError:
        raise TypeError(f'from_neo: the spike times have unit {pq_unit}, which is not a time.') from None
    name = pq_unit.dimensionality.string
    try:
        unit = u.parse_unit(name)
        if unit.dim == u.second.dim and math.isclose(_seconds_per(unit), seconds, rel_tol=1e-12):
            return unit, 1.0
    except (ValueError, KeyError, TypeError):
        pass
    return u.second, seconds


@set_module_as('braintools.file')
def to_neo(trains: SpikeTrains) -> List[Any]:
    """Convert a :class:`~braintools.spike.SpikeTrains` to a list of ``neo.SpikeTrain``.

    One ``neo.SpikeTrain`` is created per neuron, holding its valid spikes
    (padding is dropped), with the population's ``t_start`` and ``t_stop``.

    Parameters
    ----------
    trains : SpikeTrains
        Unbatched spike trains whose times carry a ``brainunit`` time unit.

    Returns
    -------
    list of neo.SpikeTrain
        ``trains.n_neurons`` spike trains, in neuron order. Their unit is the
        same-named ``quantities`` unit (``ms``, ``s``, ``us``, ...) or, if
        there is none, seconds. If ``trains.overflow`` is set, every train
        carries the annotation ``braintools_overflow=True``.

    Raises
    ------
    TypeError
        If ``trains`` is not a ``SpikeTrains``, is traced (inside
        ``jax.jit``/``jax.vmap``), or its times have no time unit.
    ValueError
        If ``trains`` is batched (times not 2-D).
    ImportError
        If ``neo`` or ``quantities`` is missing (``pip install braintools[neo]``).

    See Also
    --------
    from_neo : The inverse conversion.
    braintools.spike.SpikeTrains.to_list : Ragged per-neuron arrays.

    Notes
    -----
    Host-only: the output is ragged, so this cannot run under ``jax.jit`` or
    ``jax.vmap``. Unitless trains are rejected rather than assumed to be in
    seconds or milliseconds.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.file._neo import to_neo
        >>> from braintools.spike import SpikeTrains
        >>> st = SpikeTrains.from_list([np.array([1., 3.]) * u.ms, np.array([]) * u.ms],
        ...                            t_start=0. * u.ms, t_stop=5. * u.ms)
        >>> neo_trains = to_neo(st)
        >>> len(neo_trains)
        2
        >>> neo_trains[0].magnitude.tolist(), str(neo_trains[0].units.dimensionality)
        ([1.0, 3.0], 'ms')
        >>> float(neo_trains[1].t_stop), neo_trains[1].size
        (5.0, 0)
    """
    neo = _import_optional('neo', 'neo')
    pq = _import_optional('quantities', 'neo')
    _check_unbatched(trains, 'to_neo')
    unit = _time_unit(trains, 'to_neo')
    per_neuron = trains.to_list()  # raises TypeError for traced trains
    pq_unit, scale = _pq_unit(pq, unit)
    t_start = float(np.asarray(trains._t_start_m)) * scale
    t_stop = float(np.asarray(trains._t_stop_m)) * scale
    overflow = bool(np.asarray(trains.overflow))
    out = []
    for times in per_neuron:
        mags = np.asarray(times.mantissa, dtype=np.float64) * scale
        st = neo.SpikeTrain(mags, units=pq_unit, t_start=t_start, t_stop=t_stop)
        if overflow:
            st.annotate(**{_OVERFLOW_ANNOTATION: True})
        out.append(st)
    return out


@set_module_as('braintools.file')
def from_neo(
    spiketrains: Sequence[Any],
    *,
    capacity: Optional[int] = None,
    t_start: Any = None,
    t_stop: Any = None,
) -> SpikeTrains:
    """Build a :class:`~braintools.spike.SpikeTrains` from ``neo.SpikeTrain`` objects.

    Each ``neo.SpikeTrain`` becomes one neuron. By default the population's
    ``t_start`` and ``t_stop`` are the earliest ``t_start`` and the latest
    ``t_stop`` of the inputs.

    Parameters
    ----------
    spiketrains : sequence of neo.SpikeTrain
        The spike trains, one per neuron (for example
        ``segment.spiketrains``). Trains may use different time units; all
        are converted to the unit of the first train.
    capacity : int, optional
        Padded number of spike slots per neuron. Defaults to the longest
        train (at least 1).
    t_start, t_stop : Quantity, optional
        Recording bounds overriding the defaults above, as ``brainunit``
        time Quantities. They are needed to keep the bounds of a population
        with zero neurons: ``to_neo`` then returns an empty list, which
        cannot carry them. For an empty ``spiketrains``, ``t_start``
        defaults to 0 and ``t_stop`` to ``t_start``.

    Returns
    -------
    SpikeTrains
        Spike times as Quantities in the brainunit equivalent of the first
        train's unit (seconds if there is no equivalent), sorted per neuron.
        ``overflow`` is True if any input carries the annotation
        ``braintools_overflow=True`` (as written by :func:`to_neo`). An empty
        sequence gives zero neurons, in the unit of ``t_start`` (or
        ``t_stop``) if given and otherwise with ``t_start = t_stop = 0 s``.

    Raises
    ------
    ValueError
        If a train has more than ``capacity`` spikes, or a spike lies outside
        explicit ``t_start``/``t_stop``.
    TypeError
        If an element is not a ``neo.SpikeTrain`` or its unit is not a time,
        or ``t_start``/``t_stop`` is not a time Quantity.
    ImportError
        If ``neo`` or ``quantities`` is missing (``pip install braintools[neo]``).

    See Also
    --------
    to_neo : The inverse conversion.
    braintools.spike.SpikeTrains.from_list : Build from ragged arrays.

    Notes
    -----
    Host-only (not under ``jax.jit``/``jax.vmap``). The times are converted
    to JAX's default float dtype, so with ``jax_enable_x64`` off they are
    rounded to float32.

    Examples
    --------
    .. code-block:: python

        >>> import neo
        >>> import quantities as pq
        >>> from braintools.file._neo import from_neo
        >>> trains = [neo.SpikeTrain([3., 1.] * pq.ms, t_start=0. * pq.ms, t_stop=4. * pq.ms),
        ...           neo.SpikeTrain([0.002] * pq.s, t_start=0. * pq.s, t_stop=0.005 * pq.s)]
        >>> st = from_neo(trains)
        >>> st.times
        Quantity([[ 1.  3.]
                  [ 2. inf]], "ms")
        >>> st.t_start, st.t_stop
        (Quantity(0., "ms"), Quantity(5., "ms"))
        >>>
        >>> import brainunit as u
        >>> empty = from_neo([], t_start=0. * u.ms, t_stop=5. * u.ms)
        >>> empty.n_neurons, empty.t_stop
        (0, Quantity(5., "ms"))
    """
    neo = _import_optional('neo', 'neo')
    pq = _import_optional('quantities', 'neo')
    spiketrains = list(spiketrains)
    for i, st in enumerate(spiketrains):
        if not isinstance(st, neo.SpikeTrain):
            raise TypeError(
                f'from_neo: spiketrains[{i}] is a {type(st).__name__}, not a neo.SpikeTrain.'
            )
    given = [b for b in (t_start, t_stop) if b is not None]
    for b in given:
        if not isinstance(b, u.Quantity):
            raise TypeError(
                f'from_neo: `t_start`/`t_stop` must be a brainunit Quantity with a time '
                f'unit (e.g. `5. * u.ms`); got the plain value {b!r}.'
            )
        if b.unit.dim != u.second.dim:
            raise TypeError(f'from_neo: `t_start`/`t_stop` has unit {b.unit}, which is not a time.')
    if not spiketrains:
        unit = given[0].unit if given else u.second
        t0 = t_start.to_decimal(unit) if t_start is not None else 0.
        t1 = t_stop.to_decimal(unit) if t_stop is not None else t0
        return SpikeTrains(
            jnp.zeros((0, 1)) * unit, jnp.zeros(0, dtype=jnp.int32), t0 * unit, t1 * unit,
        )
    pq_unit = spiketrains[0].units
    unit, scale = _bu_unit(pq, pq_unit)

    def _mags(x):
        return np.asarray(x.rescale(pq_unit).magnitude, dtype=np.float64) * scale

    times = [u.Quantity(np.sort(_mags(st).ravel()), unit=unit) for st in spiketrains]
    if t_start is None:
        t_start = min(float(_mags(st.t_start)) for st in spiketrains) * unit
    if t_stop is None:
        t_stop = max(float(_mags(st.t_stop)) for st in spiketrains) * unit
    result = SpikeTrains.from_list(times, t_start=t_start, t_stop=t_stop, capacity=capacity)
    overflow = any(bool(st.annotations.get(_OVERFLOW_ANNOTATION, False)) for st in spiketrains)
    if overflow:
        result = SpikeTrains(result.times, result.counts, result.t_start, result.t_stop,
                             overflow=True)
    return result
