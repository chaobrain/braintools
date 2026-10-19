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

"""Read and write spike trains and time series in Neurodata Without Borders (NWB) files.

``pynwb`` is an optional dependency (``pip install braintools[nwb]``); it is
imported only when the functions are called.
"""

import datetime
import os
import uuid
import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import brainunit as u
import jax
import numpy as np

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains
from ._hdf5 import _atomic_target, _check_target, _import_optional
from ._neo import _check_unbatched, _seconds_per, _time_unit

__all__ = [
    'save_nwb_units',
    'load_nwb_units',
    'save_nwb_timeseries',
    'load_nwb_timeseries',
]

PathLike = Union[str, os.PathLike]

# Custom Units-table column written only when the source SpikeTrains had dropped
# spikes (``overflow=True``); every row holds the same value.
_OVERFLOW_COLUMN = 'braintools_overflow'

# ``TimeIntervals`` table (in the file's ``intervals`` group) with one row
# ``[t_start, t_stop]``: the recording bounds of the saved SpikeTrains. Unlike
# the per-unit ``obs_intervals``, it survives a population with zero neurons.
_BOUNDS_TABLE = 'braintools_recording'

# Unit strings that mean "no physical unit" in NWB files.
_NO_UNIT = {'', 'n/a', 'none', 'unitless', 'dimensionless', 'a.u.', 'no unit'}

# NWB-schema spellings (full plural names) mapped to brainunit display names.
_NWB_UNIT_ALIASES = {
    'volt': 'V', 'volts': 'V',
    'ampere': 'A', 'amperes': 'A', 'amp': 'A', 'amps': 'A',
    'second': 's', 'seconds': 's',
    'siemens': 'S',
    'ohm': 'ohm', 'ohms': 'ohm',
    'hertz': 'Hz',
    'meter': 'm', 'meters': 'm', 'metre': 'm', 'metres': 'm',
    'degree': 'degree', 'degrees': 'degree',
    'radian': 'rad', 'radians': 'rad',
}


def _session_start(session_start_time: Optional[datetime.datetime]) -> datetime.datetime:
    if session_start_time is None:
        return datetime.datetime.now(datetime.timezone.utc)
    return session_start_time


def _write_nwb(pynwb, path: str, nwbfile) -> None:
    with _atomic_target(path) as tmp:
        with pynwb.NWBHDF5IO(tmp, 'w') as io:
            io.write(nwbfile)


def _open_for_read(path: PathLike) -> str:
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'No such file: "{path}".')
    return path


def _ragged(units, column: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(flat_values, row_lengths)`` of a ragged Units-table column."""
    index = units[column]
    flat = np.asarray(index.target.data[:])
    ends = np.asarray(index.data[:], dtype=np.int64)
    lengths = np.diff(np.concatenate([[0], ends]))
    return flat, lengths


def _parse_nwb_unit(text: str) -> Optional[u.Unit]:
    """Parse an NWB unit string; ``None`` for "no unit" or unparseable strings."""
    key = text.strip()
    if key.lower() in _NO_UNIT:
        return None
    key = _NWB_UNIT_ALIASES.get(key.lower(), key)
    try:
        return u.parse_unit(key)
    except (ValueError, KeyError, TypeError):
        warnings.warn(
            f'load_nwb_timeseries: cannot interpret the NWB unit string {text!r} as a '
            f'brainunit unit; returning the data as a plain array.',
            UserWarning,
            stacklevel=3,
        )
        return None


# ----------------------------------------------------------------------------
# spike trains <-> Units table
# ----------------------------------------------------------------------------

@set_module_as('braintools.file')
def save_nwb_units(
    path: PathLike,
    trains: SpikeTrains,
    *,
    session_description: str,
    identifier: str,
    session_start_time: Optional[datetime.datetime] = None,
    unit_ids: Optional[Sequence[int]] = None,
    overwrite: bool = False,
) -> None:
    """Write a :class:`~braintools.spike.SpikeTrains` to the Units table of a new NWB file.

    Each neuron becomes one row of the NWB ``units`` table. Its ``spike_times``
    are stored in seconds (the NWB convention), converted from the trains'
    time unit, and its ``obs_intervals`` is ``[[t_start, t_stop]]`` in
    seconds. The recording bounds are also written as the single row of a
    ``TimeIntervals`` table ``braintools_recording`` in the file's
    ``intervals`` group, so they round-trip even for zero neurons.

    Parameters
    ----------
    path : str or os.PathLike
        Target ``.nwb`` file. It is written to a temporary file next to it and
        moved into place only when writing succeeded.
    trains : SpikeTrains
        Unbatched spike trains whose times carry a ``brainunit`` time unit.
    session_description : str
        Description of the recording session (required by NWB).
    identifier : str
        Unique identifier of the file (required by NWB).
    session_start_time : datetime.datetime, optional
        Start of the session; NWB requires a timezone-aware value. Defaults
        to the current UTC time.
    unit_ids : sequence of int, optional
        Unique ids of the ``n_neurons`` rows. Defaults to ``0, ..., N-1``.
    overwrite : bool, default False
        Whether to replace an existing file at ``path``.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False.
    TypeError
        If ``trains`` is not a ``SpikeTrains``, is traced, or has no time
        unit (NWB stores seconds, so unitless times are rejected rather than
        assumed to be in some unit).
    ValueError
        If ``trains`` is batched, or ``unit_ids`` has the wrong length or
        repeated ids.
    ImportError
        If ``pynwb`` is not installed (``pip install braintools[nwb]``).

    See Also
    --------
    load_nwb_units : Read the Units table back.
    save_nwb_timeseries : Write a regularly sampled signal.

    Notes
    -----
    Host-only (not under ``jax.jit``/``jax.vmap``). Times are converted to
    float64 seconds before writing. If ``trains.overflow`` is set, a boolean
    column ``braintools_overflow`` (True in every row) is added so that
    :func:`load_nwb_units` restores the flag; otherwise the file contains
    only standard NWB columns. A population with zero neurons is written as
    an empty Units table.

    Examples
    --------
    .. code-block:: python

        >>> import os, tempfile
        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.file._nwb import save_nwb_units, load_nwb_units
        >>> from braintools.spike import SpikeTrains
        >>> st = SpikeTrains.from_list([np.array([1.5, 4.]) * u.ms, np.array([]) * u.ms],
        ...                            t_start=0. * u.ms, t_stop=5. * u.ms)
        >>> path = os.path.join(tempfile.mkdtemp(), 'units.nwb')
        >>> save_nwb_units(path, st, session_description='demo', identifier='demo-1')
        >>> back = load_nwb_units(path)
        >>> back.counts
        Array([2, 0], dtype=int32)
        >>> back.t_stop
        Quantity(0.005, "s")
    """
    pynwb = _import_optional('pynwb', 'nwb')
    _check_unbatched(trains, 'save_nwb_units')
    unit = _time_unit(trains, 'save_nwb_units')
    per_neuron = trains.to_list()  # raises TypeError for traced trains
    n = len(per_neuron)
    if unit_ids is None:
        ids = list(range(n))
    else:
        ids = [int(i) for i in unit_ids]
        if len(ids) != n:
            raise ValueError(f'`unit_ids` has {len(ids)} entries, but there are {n} neurons.')
        if len(set(ids)) != n:
            raise ValueError('`unit_ids` must be unique.')
    path = _check_target(path, overwrite)

    to_s = _seconds_per(unit)
    t_start = float(np.asarray(trains._t_start_m, dtype=np.float64)) * to_s
    t_stop = float(np.asarray(trains._t_stop_m, dtype=np.float64)) * to_s
    overflow = bool(np.asarray(trains.overflow))

    nwbfile = pynwb.NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=_session_start(session_start_time),
    )
    if n == 0:
        nwbfile.units = pynwb.misc.Units(name='units', description='Spike times (no units).')
    else:
        extra = {}
        if overflow:
            nwbfile.add_unit_column(
                _OVERFLOW_COLUMN,
                'True if braintools dropped spikes because a neuron exceeded the '
                'SpikeTrains capacity (the trains are incomplete).',
            )
            extra[_OVERFLOW_COLUMN] = True
        for uid, times in zip(ids, per_neuron):
            seconds = np.asarray(times.mantissa, dtype=np.float64) * to_s
            nwbfile.add_unit(
                spike_times=seconds, obs_intervals=[[t_start, t_stop]], id=uid, **extra,
            )
    bounds = pynwb.epoch.TimeIntervals(
        name=_BOUNDS_TABLE,
        description='Recording bounds [t_start, t_stop] of the braintools SpikeTrains in the '
                    'units table, in seconds.',
    )
    bounds.add_interval(start_time=t_start, stop_time=t_stop)
    nwbfile.add_time_intervals(bounds)
    _write_nwb(pynwb, path, nwbfile)


@set_module_as('braintools.file')
def load_nwb_units(path: PathLike, *, capacity: Optional[int] = None) -> SpikeTrains:
    """Read the Units table of an NWB file into a :class:`~braintools.spike.SpikeTrains`.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.nwb`` file.
    capacity : int, optional
        Padded number of spike slots per neuron. Defaults to the largest
        spike count (at least 1).

    Returns
    -------
    SpikeTrains
        One neuron per row of the Units table, in row order, with times as
        Quantities in seconds. ``t_start``/``t_stop`` are the earliest start
        and the latest end of the ``braintools_recording`` intervals table
        written by :func:`save_nwb_units` and of the ``obs_intervals``
        column, widened if needed to contain every spike. Without either,
        they are the earliest and latest spike (0 s if there are none). ``overflow`` is restored from
        the ``braintools_overflow`` column written by :func:`save_nwb_units`
        (False if absent).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file has no Units table, or a neuron has more than
        ``capacity`` spikes.
    ImportError
        If ``pynwb`` is not installed (``pip install braintools[nwb]``).

    See Also
    --------
    save_nwb_units : Write a Units table.
    braintools.file.from_neo : Import spike trains from ``neo``.

    Notes
    -----
    Host-only (not under ``jax.jit``/``jax.vmap``). Unit ids are not
    returned; read them with ``pynwb`` if needed. Times are converted to
    JAX's default float dtype (float32 unless ``jax_enable_x64`` is on).

    Examples
    --------
    .. code-block:: python

        >>> import os, tempfile
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.file._nwb import save_nwb_units, load_nwb_units
        >>> from braintools.spike import SpikeTrains
        >>> st = SpikeTrains.from_dense(jnp.array([[1, 0], [0, 1], [1, 1]]), dt=1. * u.ms)
        >>> path = os.path.join(tempfile.mkdtemp(), 'units.nwb')
        >>> save_nwb_units(path, st, session_description='demo', identifier='demo-2')
        >>> back = load_nwb_units(path)
        >>> back.times.to_decimal(u.ms).round(6)
        Array([[0., 2.],
               [1., 2.]], dtype=float32)
    """
    pynwb = _import_optional('pynwb', 'nwb')
    path = _open_for_read(path)
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwbfile = io.read()
        units = nwbfile.units
        if units is None:
            raise ValueError(f'"{path}" has no Units table.')
        n = len(units)
        colnames = set(units.colnames)
        if 'spike_times' in colnames and n:
            flat, lengths = _ragged(units, 'spike_times')
        else:
            flat, lengths = np.zeros(0), np.zeros(n, dtype=np.int64)
        if 'obs_intervals' in colnames and n:
            intervals = np.asarray(_ragged(units, 'obs_intervals')[0], dtype=np.float64)
            intervals = intervals.reshape(-1, 2)
        else:
            intervals = np.zeros((0, 2))
        recording = nwbfile.intervals.get(_BOUNDS_TABLE) if nwbfile.intervals else None
        if recording is not None and len(recording):
            rows = np.stack([np.asarray(recording['start_time'][:], dtype=np.float64),
                             np.asarray(recording['stop_time'][:], dtype=np.float64)], axis=1)
            intervals = np.concatenate([rows, intervals])
        overflow = (
            _OVERFLOW_COLUMN in colnames and n > 0
            and bool(np.any(np.asarray(units[_OVERFLOW_COLUMN].data[:])))
        )

    flat = np.asarray(flat, dtype=np.float64).ravel()
    ids = np.repeat(np.arange(n), lengths)
    lows = np.concatenate([intervals[:, 0], flat])
    highs = np.concatenate([intervals[:, 1], flat])
    t_start = float(np.min(lows)) if lows.size else 0.
    t_stop = float(np.max(highs)) if highs.size else 0.
    trains = SpikeTrains.from_events(
        flat * u.second, ids, n, t_start=t_start * u.second, t_stop=t_stop * u.second,
        capacity=capacity,
    )
    if overflow:
        trains = SpikeTrains(trains.times, trains.counts, trains.t_start, trains.t_stop,
                             overflow=True)
    return trains


# ----------------------------------------------------------------------------
# regularly sampled signals <-> TimeSeries
# ----------------------------------------------------------------------------

@set_module_as('braintools.file')
def save_nwb_timeseries(
    path: PathLike,
    data: Any,
    dt: Any,
    name: str,
    unit: Optional[Union[u.Unit, str]] = None,
    *,
    description: str = 'no description',
    session_description: str = 'braintools time series',
    identifier: Optional[str] = None,
    session_start_time: Optional[datetime.datetime] = None,
    overwrite: bool = False,
) -> None:
    """Write a regularly sampled signal as an NWB ``TimeSeries`` in a new file.

    The series is added to the file's ``acquisition`` group, with time on
    axis 0, sampling rate ``1 / dt`` in Hz and starting time 0 s.

    Parameters
    ----------
    path : str or os.PathLike
        Target ``.nwb`` file, written atomically.
    data : array-like or Quantity
        Samples with time on axis 0, shape ``(T, ...)``.
    dt : float or Quantity
        Sampling interval. A Quantity must have a time unit; a plain number
        is in seconds (the NWB time unit).
    name : str
        Name of the ``TimeSeries``.
    unit : brainunit.Unit or str, optional
        Physical unit to store. If ``data`` is a Quantity, it is converted to
        ``unit`` (a ``str`` is parsed with ``brainunit.parse_unit``), and the
        default is the unit of ``data``. If ``data`` is a plain array, ``unit``
        is only recorded as its string; the default is ``"n/a"`` (no unit).
    description : str, default 'no description'
        Description of the series.
    session_description : str, default 'braintools time series'
        Description of the session (required by NWB).
    identifier : str, optional
        Unique file identifier; a random UUID by default.
    session_start_time : datetime.datetime, optional
        Timezone-aware session start; the current UTC time by default.
    overwrite : bool, default False
        Whether to replace an existing file at ``path``.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False.
    TypeError
        If ``dt`` has a non-time unit or ``data`` is a JAX tracer.
    ValueError
        If ``dt`` is not positive, or ``data`` is 0-d.
    brainunit.UnitMismatchError
        If ``data`` cannot be converted to ``unit``.
    ImportError
        If ``pynwb`` is not installed (``pip install braintools[nwb]``).

    See Also
    --------
    load_nwb_timeseries : Read the series back.
    save_nwb_units : Write spike trains.

    Notes
    -----
    Host-only (not under ``jax.jit``/``jax.vmap``). Each call creates a new
    file holding one series.

    Examples
    --------
    .. code-block:: python

        >>> import os, tempfile
        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.file._nwb import save_nwb_timeseries, load_nwb_timeseries
        >>> path = os.path.join(tempfile.mkdtemp(), 'v.nwb')
        >>> v = np.array([[-65., -64.], [-63., -60.], [-55., -70.]]) * u.mV
        >>> save_nwb_timeseries(path, v, 0.1 * u.ms, 'membrane_potential')
        >>> data, dt = load_nwb_timeseries(path, 'membrane_potential')
        >>> data
        Quantity([[-65. -64.]
                  [-63. -60.]
                  [-55. -70.]], "mV")
        >>> dt.to_decimal(u.ms)
        0.1
    """
    pynwb = _import_optional('pynwb', 'nwb')

    # sampling interval, in seconds
    if isinstance(dt, u.Quantity):
        if dt.unit.dim != u.second.dim:
            raise TypeError(f'`dt` must be a time, got a Quantity with unit {dt.unit}.')
        dt_s = float(np.asarray(dt.to_decimal(u.second)))
    else:
        dt_s = float(np.asarray(dt))
    if not dt_s > 0:
        raise ValueError(f'`dt` must be positive, got {dt!r}.')

    # samples and their unit string
    if isinstance(data, u.Quantity):
        target = u.parse_unit(unit) if isinstance(unit, str) else unit
        if target is None:
            target = data.unit
        values = data.to_decimal(target)
        unit_str = str(target)
    else:
        values = data
        unit_str = 'n/a' if unit is None else str(unit)
    if isinstance(values, jax.core.Tracer):
        raise TypeError('save_nwb_timeseries: `data` is a JAX tracer; save concrete arrays.')
    values = np.asarray(values)
    if values.ndim == 0:
        raise ValueError('`data` must have a time axis (axis 0); got a 0-d value.')
    path = _check_target(path, overwrite)

    nwbfile = pynwb.NWBFile(
        session_description=session_description,
        identifier=uuid.uuid4().hex if identifier is None else identifier,
        session_start_time=_session_start(session_start_time),
    )
    nwbfile.add_acquisition(pynwb.TimeSeries(
        name=name, data=values, unit=unit_str, rate=1.0 / dt_s, starting_time=0.0,
        description=description,
    ))
    _write_nwb(pynwb, path, nwbfile)


@set_module_as('braintools.file')
def load_nwb_timeseries(path: PathLike, name: str) -> Tuple[Any, Any]:
    """Read a regularly sampled NWB ``TimeSeries`` by name.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.nwb`` file.
    name : str
        Name of the ``TimeSeries``. The ``acquisition`` group is searched
        first, then the rest of the file (processing modules, stimulus, ...).

    Returns
    -------
    data : numpy.ndarray or Quantity
        The samples, time on axis 0, with the series' ``conversion`` and
        ``offset`` applied (``data * conversion + offset``). A Quantity when
        the NWB unit string is a unit brainunit can parse (NWB spellings
        such as ``"volts"`` are recognised); a plain array for ``"n/a"`` and
        similar, or with a ``UserWarning`` for an unknown unit string.
    dt : Quantity
        Sampling interval in seconds, from the series' ``rate``, or from
        uniformly spaced ``timestamps``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If no ``TimeSeries`` called ``name`` exists (the message lists the
        available names).
    ValueError
        If the series has irregular timestamps (no single ``dt``).
    ImportError
        If ``pynwb`` is not installed (``pip install braintools[nwb]``).

    See Also
    --------
    save_nwb_timeseries : Write a series.

    Notes
    -----
    Host-only (not under ``jax.jit``/``jax.vmap``). The data is read fully
    into memory.

    Examples
    --------
    .. code-block:: python

        >>> import os, tempfile
        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.file._nwb import save_nwb_timeseries, load_nwb_timeseries
        >>> path = os.path.join(tempfile.mkdtemp(), 'i.nwb')
        >>> save_nwb_timeseries(path, np.arange(4.), 1e-3, 'counts')
        >>> data, dt = load_nwb_timeseries(path, 'counts')
        >>> data, dt
        (array([0., 1., 2., 3.]), Quantity(0.001, "s"))
    """
    pynwb = _import_optional('pynwb', 'nwb')
    path = _open_for_read(path)
    with pynwb.NWBHDF5IO(path, 'r') as io:
        nwbfile = io.read()
        ts = nwbfile.acquisition.get(name)
        if not isinstance(ts, pynwb.TimeSeries):
            matches = [o for o in nwbfile.objects.values()
                       if isinstance(o, pynwb.TimeSeries) and o.name == name]
            if not matches:
                available = sorted({o.name for o in nwbfile.objects.values()
                                    if isinstance(o, pynwb.TimeSeries)})
                raise KeyError(
                    f'No TimeSeries named {name!r} in "{path}". Available: {available}.'
                )
            ts = matches[0]
        values = np.asarray(ts.data[:])
        conversion = float(ts.conversion)
        offset = float(getattr(ts, 'offset', 0.0) or 0.0)
        unit_str = str(ts.unit)
        rate = ts.rate
        timestamps = None if rate is not None else np.asarray(ts.timestamps[:], dtype=np.float64)

    if rate is not None:
        dt_s = 1.0 / float(rate)
    else:
        steps = np.diff(timestamps)
        if steps.size == 0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=0.):
            raise ValueError(
                f'TimeSeries {name!r} has irregular timestamps (or fewer than two), so it has '
                f'no single sampling interval `dt`.'
            )
        dt_s = float(steps[0])
    if conversion != 1.0 or offset != 0.0:
        values = values * conversion + offset
    unit = _parse_nwb_unit(unit_str)
    data = values if unit is None else u.Quantity(values, unit=unit)
    return data, dt_s * u.second
