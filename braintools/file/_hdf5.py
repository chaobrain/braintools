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

"""Save and load nested pytrees of arrays, Quantities and spike trains in HDF5.

``h5py`` is an optional dependency (``pip install braintools[hdf5]``); it is
imported only when :func:`save_hdf5` or :func:`load_hdf5` is called.
"""

import importlib
import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Union
from urllib.parse import unquote

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as
from braintools.spike._spike_trains import SpikeTrains

__all__ = [
    'save_hdf5',
    'load_hdf5',
]

PathLike = Union[str, os.PathLike]

# Every object written by ``save_hdf5`` carries this attribute; its value names
# the Python type to rebuild. User keys only ever become group/dataset names,
# never attribute names, so they cannot collide with these.
_TYPE = '__bt_type__'
_MANTISSA = '__bt_mantissa__'
_FORMAT = '__bt_format__'
_VERSION = '__bt_version__'
_FORMAT_NAME = 'braintools.hdf5'
_FORMAT_VERSION = 1

# Unit metadata: the display string (parsed with ``brainunit.parse_unit``) and
# the exact numeric definition, used when the string cannot be parsed back to
# the very same unit.
_UNIT_STR = 'unit'
_UNIT_DIM = 'unit_dim'
_UNIT_SCALE = 'unit_scale'
_UNIT_BASE = 'unit_base'
_UNIT_FACTOR = 'unit_factor'

_SPIKE_FIELDS = ('times', 'counts', 't_start', 't_stop', 'overflow')

_PY_SCALARS = {bool: 'bool', int: 'int', float: 'float', complex: 'complex'}
_PY_SCALAR_TYPES = {v: k for k, v in _PY_SCALARS.items()}

# Python ints are stored as int64, or as uint64 above the int64 range.
_INT_MIN = -2 ** 63
_INT_MAX = 2 ** 64 - 1


# ----------------------------------------------------------------------------
# helpers shared by the optional-format modules
# ----------------------------------------------------------------------------

def _import_optional(module: str, extra: str):
    """Import ``module`` or raise ``ImportError`` with the ``pip`` extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f'This function requires the optional package `{module}`. Install it with '
            f'`pip install braintools[{extra}]`.'
        ) from e


def _check_target(path: PathLike, overwrite: bool) -> str:
    """Return ``path`` as a string; raise ``FileExistsError`` if it exists and ``overwrite`` is False."""
    path = os.fspath(path)
    if not overwrite and os.path.exists(path):
        raise FileExistsError(
            f'"{path}" already exists. Pass `overwrite=True` to replace it.'
        )
    return path


@contextmanager
def _atomic_target(path: str):
    """Yield a temporary path next to ``path`` and move it onto ``path`` on success.

    A failed write leaves any existing file at ``path`` untouched and removes
    the temporary file.
    """
    root, ext = os.path.splitext(path)
    tmp = f'{root}.tmp-{uuid.uuid4().hex}{ext}'  # keep the extension (pynwb checks it)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _encode_unit(unit: u.Unit) -> Dict[str, Any]:
    """Describe ``unit`` as HDF5-storable attributes (string plus exact numbers)."""
    return {
        _UNIT_STR: str(unit),
        _UNIT_DIM: np.asarray(unit.dim._dims, dtype=np.float64),
        _UNIT_SCALE: float(unit.scale),
        _UNIT_BASE: float(unit.base),
        _UNIT_FACTOR: float(unit.factor),
    }


def _decode_unit(attrs) -> u.Unit:
    """Rebuild the unit written by :func:`_encode_unit`.

    The parsed display string is preferred because it keeps the display name
    (``mV / ms`` rather than ``10.0^-3 * m^2 kg s^-4 A^-1``); it is used only
    when it equals the exact numeric definition.
    """
    dims = np.asarray(attrs[_UNIT_DIM], dtype=np.float64)
    if np.all(dims == np.round(dims)):
        dims = dims.astype(np.int64)
    exact = u.Unit(
        dim=u.Dimension(dims),
        scale=_as_number(attrs[_UNIT_SCALE]),
        base=float(attrs[_UNIT_BASE]),
        factor=float(attrs[_UNIT_FACTOR]),
    )
    try:
        parsed = u.parse_unit(str(attrs[_UNIT_STR]))
    except (ValueError, KeyError, TypeError):
        return exact
    return parsed if parsed == exact else exact


def _as_number(x: Any):
    x = float(x)
    return int(x) if x.is_integer() else x


# ----------------------------------------------------------------------------
# name escaping
# ----------------------------------------------------------------------------

def _escape(key: str) -> str:
    """Turn a dict key into a valid HDF5 link name (reversible with :func:`_unescape`).

    ``%`` and the characters HDF5 forbids in names (``/`` and NUL) are
    percent-encoded; the reserved names ``.`` and the empty string become
    ``%2E`` and a lone ``%`` (which escaping never produces otherwise).
    """
    if key == '':
        return '%'
    name = key.replace('%', '%25').replace('/', '%2F').replace('\x00', '%00')
    return '%2E' if name == '.' else name


def _unescape(name: str) -> str:
    if name == '%':
        return ''
    return unquote(name)


# ----------------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------------

def _host_array(x: Any, where: str) -> np.ndarray:
    if isinstance(x, jax.core.Tracer):
        raise TypeError(f'save_hdf5: the value at {where} is a JAX tracer; save concrete arrays.')
    arr = np.asarray(x)
    if arr.dtype.kind == 'V':  # extension float types such as bfloat16
        raise TypeError(
            f'save_hdf5: the array at {where} has dtype {arr.dtype}, which HDF5 cannot store '
            f'natively; cast it to a standard dtype first (e.g. bfloat16 -> float32).'
        )
    if arr.dtype.kind in 'OUS':
        raise TypeError(
            f'save_hdf5: the array at {where} has dtype {arr.dtype}, which is not supported; '
            f'use numeric or bool arrays (store strings as Python `str`).'
        )
    return arr


def _write_dataset(group, name: str, data: Any, kind: str, **attrs):
    ds = group.create_dataset(name, data=data)
    ds.attrs[_TYPE] = kind
    for k, v in attrs.items():
        ds.attrs[k] = v
    return ds


def _write(h5py, group, name: str, obj: Any, where: str) -> None:
    """Write ``obj`` as child ``name`` of ``group``."""
    if obj is None:
        _write_dataset(group, name, h5py.Empty('f'), 'none')
    elif isinstance(obj, SpikeTrains):
        sub = group.create_group(name, track_order=True)
        _write_spike_trains(sub, obj, where)
    elif isinstance(obj, u.Quantity):
        mantissa = obj.mantissa
        if isinstance(mantissa, jax.Array):
            mantissa_kind = 'jax'
        elif type(mantissa) in _PY_SCALARS:
            mantissa_kind = 'python'
        else:
            mantissa_kind = 'numpy'
        _write_dataset(
            group, name, _host_array(mantissa, where), 'quantity',
            **{_MANTISSA: mantissa_kind},
            **_encode_unit(obj.unit),
        )
    elif isinstance(obj, dict):
        sub = group.create_group(name, track_order=True)
        _write_dict(h5py, sub, obj, where)
    elif isinstance(obj, (list, tuple)):
        sub = group.create_group(name, track_order=True)
        _write_sequence(h5py, sub, obj, where)
    elif isinstance(obj, str):
        ds = group.create_dataset(name, data=obj, dtype=h5py.string_dtype())
        ds.attrs[_TYPE] = 'str'
    elif type(obj) in _PY_SCALARS:
        if type(obj) is int and not _INT_MIN <= obj <= _INT_MAX:
            raise OverflowError(
                f'save_hdf5: the int at {where} does not fit in a 64-bit integer '
                f'(allowed range [-2**63, 2**64 - 1]); store it as a str or float instead.'
            )
        _write_dataset(group, name, obj, _PY_SCALARS[type(obj)])
    elif isinstance(obj, jax.Array):
        _write_dataset(group, name, _host_array(obj, where), 'jax')
    elif isinstance(obj, np.generic):
        _write_dataset(group, name, _host_array(obj, where), 'numpy_scalar')
    elif isinstance(obj, np.ndarray):
        _write_dataset(group, name, _host_array(obj, where), 'numpy')
    else:
        raise TypeError(
            f'save_hdf5: cannot store the value of type {type(obj).__name__} at {where}. '
            f'Supported: dict (str keys), list, tuple, None, bool/int/float/complex/str, '
            f'numpy and JAX arrays, brainunit Quantities and SpikeTrains.'
        )


def _write_dict(h5py, group, obj: dict, where: str) -> None:
    group.attrs[_TYPE] = 'dict'
    for key, value in obj.items():
        if not isinstance(key, str):
            raise TypeError(
                f'save_hdf5: dict keys must be str; got {key!r} ({type(key).__name__}) at {where}.'
            )
        _write(h5py, group, _escape(key), value, f'{where}[{key!r}]')


def _write_sequence(h5py, group, obj, where: str) -> None:
    group.attrs[_TYPE] = 'tuple' if isinstance(obj, tuple) else 'list'
    for i, value in enumerate(obj):
        _write(h5py, group, str(i), value, f'{where}[{i}]')


def _write_spike_trains(group, trains: SpikeTrains, where: str) -> None:
    group.attrs[_TYPE] = 'spike_trains'
    unit = trains._unit
    if unit is not None:
        for k, v in _encode_unit(unit).items():
            group.attrs[k] = v
    values = (trains._times_m, trains.counts, trains._t_start_m, trains._t_stop_m, trains.overflow)
    for field, value in zip(_SPIKE_FIELDS, values):
        group.create_dataset(field, data=_host_array(value, f'{where}.{field}'))


# ----------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------

def _read(h5py, obj) -> Any:
    kind = obj.attrs.get(_TYPE, None)
    if isinstance(obj, h5py.Group):
        if kind == 'spike_trains':
            return _read_spike_trains(obj)
        if kind in ('list', 'tuple'):
            items = sorted(obj.items(), key=lambda kv: int(kv[0]))
            values = [_read(h5py, child) for _, child in items]
            return tuple(values) if kind == 'tuple' else values
        # 'dict', or a group written by another program
        return {_unescape(name): _read(h5py, child) for name, child in obj.items()}

    if kind == 'none':
        return None
    if kind == 'str' or (kind is None and h5py.check_string_dtype(obj.dtype) is not None):
        return obj.asstr()[()]
    data = obj[()]
    if isinstance(data, h5py.Empty):
        return None
    if kind in _PY_SCALAR_TYPES:
        return _PY_SCALAR_TYPES[kind](data)
    if kind == 'quantity':
        mantissa_kind = obj.attrs.get(_MANTISSA)
        if mantissa_kind == 'jax':
            mantissa = jnp.asarray(data)
        elif mantissa_kind == 'python':
            mantissa = np.asarray(data).item()
        else:
            mantissa = np.asarray(data)
        return u.Quantity(mantissa, unit=_decode_unit(obj.attrs))
    if kind == 'jax':
        return jnp.asarray(data)
    if kind == 'numpy_scalar':
        return data
    # 'numpy', or a dataset written by another program
    return np.asarray(data)


def _read_spike_trains(group) -> SpikeTrains:
    unit = _decode_unit(group.attrs) if _UNIT_STR in group.attrs else None
    times, counts, t_start, t_stop, overflow = (jnp.asarray(group[f][()]) for f in _SPIKE_FIELDS)
    return SpikeTrains._make(times, counts.astype(jnp.int32), t_start, t_stop, overflow.astype(bool), unit)


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

@set_module_as('braintools.file')
def save_hdf5(path: PathLike, tree: Any, *, overwrite: bool = False) -> None:
    """Save a nested tree of arrays, Quantities and spike trains to an HDF5 file.

    Containers become HDF5 groups and leaves become datasets, so the file can
    also be browsed with standard HDF5 tools (``h5ls``, HDFView, ``h5py``).
    Every object carries a small type attribute that lets :func:`load_hdf5`
    rebuild exactly the Python types that were saved.

    Parameters
    ----------
    path : str or os.PathLike
        Target file. The file is written to a temporary file next to it and
        moved into place only when writing succeeded.
    tree : Any
        The data to save. Supported values, nested arbitrarily:

        - ``dict`` with ``str`` keys (any characters, including ``/``; keys
          are escaped into valid HDF5 names and restored on load), ``list``
          and ``tuple``;
        - numpy arrays and numpy scalars, JAX arrays (of any shape, including
          0-d, with numeric or bool dtype);
        - Python ``bool``, ``int``, ``float``, ``complex``, ``str`` and
          ``None``;
        - ``brainunit.Quantity`` (the mantissa is the dataset; the unit is
          stored as attributes);
        - :class:`braintools.spike.SpikeTrains` (a group with the datasets
          ``times``, ``counts``, ``t_start``, ``t_stop`` and ``overflow``,
          and the time unit as attributes).
    overwrite : bool, default False
        Whether to replace an existing file at ``path``.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is False.
    TypeError
        If the tree contains an unsupported value (for example a non-``str``
        dict key, an object or string array, a ``bfloat16`` array, or a JAX
        tracer).
    OverflowError
        If a Python ``int`` lies outside ``[-2**63, 2**64 - 1]``.
    ImportError
        If ``h5py`` is not installed (``pip install braintools[hdf5]``).

    See Also
    --------
    load_hdf5 : Read the file back.
    msgpack_save : Checkpoint brainstate model states.

    Notes
    -----
    This is a host-side I/O function: call it on concrete values, not inside
    ``jax.jit`` or ``jax.vmap``.

    **Type fidelity.** dtypes and shapes are kept, and so are the container
    types, the dict key order, whether an array was a numpy or a JAX array,
    and whether a Quantity mantissa was a numpy or a JAX array. Named tuples
    are saved as their fields and loaded as plain tuples.

    **Units.** A unit is stored both as its display string (for example
    ``"mV / ms"``) and as its exact definition (SI dimension exponents,
    scale, base and factor). Loading parses the string with
    ``brainunit.parse_unit`` and falls back to the exact definition, so even
    units without a display name round-trip exactly.

    **None.** ``None`` is stored as an empty dataset and loaded as ``None``.

    Examples
    --------
    .. code-block:: python

        >>> import os, tempfile
        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.file._hdf5 import save_hdf5, load_hdf5
        >>> path = os.path.join(tempfile.mkdtemp(), 'run.h5')
        >>> tree = {'v/soma': np.zeros(3) * u.mV, 'meta': {'seed': 42, 'tag': None}}
        >>> save_hdf5(path, tree)
        >>> out = load_hdf5(path)
        >>> out['v/soma']
        Quantity([0. 0. 0.], "mV")
        >>> out['meta']
        {'seed': 42, 'tag': None}
    """
    h5py = _import_optional('h5py', 'hdf5')
    path = _check_target(path, overwrite)
    with _atomic_target(path) as tmp:
        with h5py.File(tmp, 'w', track_order=True) as f:
            f.attrs[_FORMAT] = _FORMAT_NAME
            f.attrs[_VERSION] = _FORMAT_VERSION
            if isinstance(tree, dict):
                _write_dict(h5py, f, tree, 'tree')
            elif isinstance(tree, (list, tuple)):
                _write_sequence(h5py, f, tree, 'tree')
            else:
                f.attrs[_TYPE] = 'leaf'
                _write(h5py, f, 'value', tree, 'tree')


@set_module_as('braintools.file')
def load_hdf5(path: PathLike) -> Any:
    """Load a tree saved with :func:`save_hdf5`.

    Parameters
    ----------
    path : str or os.PathLike
        The HDF5 file to read.

    Returns
    -------
    Any
        The saved tree, with the original container types, keys, dtypes,
        units and ``SpikeTrains`` objects (including their ``overflow``
        flag). Files written by other programs are read too: groups become
        dicts and datasets become numpy arrays (strings become ``str``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ImportError
        If ``h5py`` is not installed (``pip install braintools[hdf5]``).

    See Also
    --------
    save_hdf5 : Write a tree.

    Notes
    -----
    Everything is read into memory. Host-side only (not under ``jax.jit``).

    Examples
    --------
    .. code-block:: python

        >>> import os, tempfile
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> from braintools.file._hdf5 import save_hdf5, load_hdf5
        >>> from braintools.spike import SpikeTrains
        >>> path = os.path.join(tempfile.mkdtemp(), 'spikes.h5')
        >>> st = SpikeTrains.from_dense(jnp.array([[1, 0], [0, 0], [1, 0]]), dt=1. * u.ms)
        >>> save_hdf5(path, {'trains': st, 'labels': ('exc', 'inh')})
        >>> out = load_hdf5(path)
        >>> out['trains'].times
        Quantity([[ 0.  2.]
                  [inf inf]], "ms")
        >>> out['labels']
        ('exc', 'inh')
    """
    h5py = _import_optional('h5py', 'hdf5')
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'No such file: "{path}".')
    with h5py.File(path, 'r') as f:
        if f.attrs.get(_TYPE) == 'leaf':
            return _read(h5py, f['value'])
        return _read(h5py, f)
