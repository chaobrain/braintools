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

"""Checkpointing helper functions.

This module is rewritten from the Flax APIs (https://github.com/google/flax).
"""

import enum
import os
import sys
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import brainstate
import brainunit as u
import jax
import numpy as np

try:
    import msgpack
except (ModuleNotFoundError, ImportError):
    msgpack = None

__all__ = [
    'msgpack_from_state_dict',
    'msgpack_to_state_dict',
    'msgpack_register_serialization',
    'msgpack_save',
    'msgpack_load',
    'AsyncManager',
    'AlreadyExistsError',
    'InvalidCheckpointPath',
]


class AlreadyExistsError(Exception):
    """Raised when a rename would overwrite an existing file.

    Pass ``overwrite=True`` to allow replacing the existing file.
    """

    def __init__(self, path):
        super().__init__(f'Trying overwrite an existing file: "{path}".')


class InvalidCheckpointPath(Exception):
    """Raised for an invalid checkpoint target or corrupt checkpoint data.

    This is raised when saving with ``overwrite=False`` to a path that already
    exists, or when loading data that cannot be decoded as a valid checkpoint.
    """

    def __init__(self, path):
        super().__init__(f'Invalid checkpoint at "{path}".')


# msgpack has a hard limit of 2**31 - 1 bytes per object leaf.  To circumvent
# this limit for giant arrays (e.g. embedding tables), we traverse the tree
# and break up arrays near the limit into flattened array chunks.
# True limit is 2**31 - 1, but leave a margin for encoding padding.
MAX_CHUNK_SIZE = 2 ** 30

# Files larger than this are read with multiple threads in ``msgpack_load``;
# smaller files use a single sequential read (the parallel path's extra file
# opens are pure overhead below this size).
PARALLEL_READ_BUF_SIZE = 128 << 20  # 128 MB

# Type alias for mismatch handling modes
MismatchMode = Literal['error', 'warn', 'ignore']

_STATE_DICT_REGISTRY: Dict[Any, Any] = {}


def _validate_mismatch(mismatch: str) -> None:
    """Validate mismatch parameter value.

    Args:
        mismatch: The mismatch mode to validate

    Raises:
        ValueError: If mismatch is not one of 'error', 'warn', 'ignore'
    """
    if mismatch not in ('error', 'warn', 'ignore'):
        raise ValueError(
            f"Invalid mismatch mode: '{mismatch}'. "
            "Must be one of 'error', 'warn', or 'ignore'."
        )


def _handle_mismatch(condition: bool, msg: str, mismatch: MismatchMode) -> None:
    """Handle mismatch conditions based on the mismatch mode.

    Args:
        condition: If True, the mismatch will be handled
        msg: The error/warning message to use
        mismatch: How to handle the mismatch ('error', 'warn', 'ignore')

    Raises:
        ValueError: If condition is True and mismatch is 'error'
    """
    if condition:
        if mismatch == 'error':
            raise ValueError(msg)
        elif mismatch == 'warn':
            warnings.warn(msg, UserWarning)
        # For 'ignore', do nothing


class _ErrorContext(threading.local):
    """Context for deserialization error messages."""

    def __init__(self):
        self.path = []


_error_context = _ErrorContext()


@contextmanager
def _record_path(name):
    try:
        _error_context.path.append(name)
        yield
    finally:
        _error_context.path.pop()


def check_msgpack():
    if msgpack is None:
        raise ModuleNotFoundError('\nPlease install msgpack via:\n'
                                  '> pip install msgpack')


def current_path():
    """Current state_dict path during deserialization for error messages."""
    return '/'.join(_error_context.path)


class _NamedTuple:
    """Fake type marker for namedtuple for registry."""
    pass


def _is_namedtuple(x):
    """Duck typing test for namedtuple factory-generated objects."""
    return isinstance(x, tuple) and hasattr(x, '_fields')


def _lookup_registry(ty):
    """Return the most-specific registered base class for ``ty`` (or None).

    When several registered types match (e.g. a user class that subclasses
    another registered class), the most derived one wins, using the length of
    the MRO as a specificity proxy. This avoids the first-registered handler
    shadowing a more specific one.
    """
    candidates = [t for t in _STATE_DICT_REGISTRY if issubclass(ty, t)]
    if not candidates:
        return None
    return max(candidates, key=lambda t: len(t.__mro__))


def msgpack_from_state_dict(
    target: Any,
    state: Any,
    name: str = '.',
    mismatch: MismatchMode = 'error'
):
    """Restores the state of the given target using a state dict.

    This function takes the current target as an argument. This lets us know the
    exact structure of the target, and lets us assert that array leaves keep
    their shape (see ``mismatch``).

    In practice, none of the leaf values in `target` are actually used. Only the
    tree structure and shapes.

    .. note::
        For :class:`brainstate.State` leaves the restore is performed
        *in place*: the template ``State`` object passed in ``target`` has its
        ``.value`` mutated and is returned (rather than copied). Pass a throwaway
        template if you need to keep the original untouched.

    Args:
      target: the object of which the state should be restored.
      state: a dictionary generated by `to_state_dict` with the desired new
             state for `target`.
      name: name of branch taken, used to improve deserialization error messages.
      mismatch: How to handle mismatches between target and state dict (including
                array shape mismatches).
                'error' (default): raise ValueError on mismatch
                'warn': issue warning and skip mismatched keys
                'ignore': silently skip mismatched keys
    Returns:
      A copy of the object with the restored state (except for ``State`` leaves,
      which are restored in place; see note above).
    """
    _validate_mismatch(mismatch)
    ty = _NamedTuple if _is_namedtuple(target) else type(target)
    registered = _lookup_registry(ty)
    if registered is None:
        return state
    ty_from_state_dict = _STATE_DICT_REGISTRY[registered][1]
    with _record_path(name):
        return ty_from_state_dict(target, state, mismatch)


def msgpack_to_state_dict(target) -> Dict[str, Any]:
    """
    Returns a dictionary with the state of the given target.
    """
    ty = _NamedTuple if _is_namedtuple(target) else type(target)
    registered = _lookup_registry(ty)
    if registered is None:
        return target

    ty_to_state_dict = _STATE_DICT_REGISTRY[registered][0]
    state_dict = ty_to_state_dict(target)
    if isinstance(state_dict, dict):
        for key in state_dict.keys():
            if not isinstance(key, str):
                raise TypeError(
                    f'A state dict must only have string keys, '
                    f'got key {key!r} of type {type(key).__name__}.'
                )
    return state_dict


def msgpack_register_serialization(
    ty,
    ty_to_state_dict,
    ty_from_state_dict,
    override: bool = False
):
    """Register a type for serialization.

    Args:
      ty: the type to be registered
      ty_to_state_dict: a function that takes an instance of ty and
        returns its state as a dictionary.
      ty_from_state_dict: a function that takes an instance of ty and
        a state dict, and returns a copy of the instance with the restored state.
      override: override a previously registered serialization handler
        (default: False).
    """
    if ty in _STATE_DICT_REGISTRY and not override:
        raise ValueError(f'a serialization handler for "{ty.__name__}"'
                         ' is already registered')
    _STATE_DICT_REGISTRY[ty] = (ty_to_state_dict, ty_from_state_dict)


def _list_state_dict(xs: List[Any]) -> Dict[str, Any]:
    return {
        str(i): msgpack_to_state_dict(x)
        for i, x in enumerate(xs)
    }


def _restore_list(xs, state_dict: Dict[str, Any], mismatch: MismatchMode = 'error') -> List[Any]:
    msg = ('The size of the list and the state dict do not match,'
           f' got {len(xs)} and {len(state_dict)} '
           f'at path {current_path()}')
    _handle_mismatch(len(state_dict) != len(xs), msg, mismatch)

    ys = []
    max_len = min(len(xs), len(state_dict))
    for i in range(max_len):
        y = msgpack_from_state_dict(xs[i], state_dict[str(i)], name=str(i), mismatch=mismatch)
        ys.append(y)

    # If target is longer, pad with original values
    for i in range(max_len, len(xs)):
        ys.append(xs[i])

    return ys


msgpack_register_serialization(
    list,
    _list_state_dict,
    _restore_list,
)
msgpack_register_serialization(
    tuple,
    _list_state_dict,
    lambda xs, state_dict, mismatch='error': tuple(_restore_list(list(xs), state_dict, mismatch))
)


def _dict_state_dict(xs: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(xs, brainstate.util.FlattedDict):
        xs = xs.to_nest()

    try:
        str_keys = set(str(k) for k in xs.keys())
    except TypeError as e:
        raise ValueError(f'Dict contains unhashable keys: {e}') from e

    if len(str_keys) != len(xs):
        # Provide detailed error showing which keys collide
        str_to_keys = {}
        for k in xs.keys():
            sk = str(k)
            if sk in str_to_keys:
                str_to_keys[sk].append(k)
            else:
                str_to_keys[sk] = [k]

        collisions = {sk: keys for sk, keys in str_to_keys.items() if len(keys) > 1}
        raise ValueError(
            f'Dict keys do not have a unique string representation. '
            f'Collisions: {collisions}'
        )

    return {
        str(key): msgpack_to_state_dict(value)
        for key, value in xs.items()
    }


def _restore_dict(xs, states: Dict[str, Any], mismatch: MismatchMode = 'error') -> Dict[str, Any]:
    if isinstance(xs, brainstate.util.FlattedDict):
        xs = xs.to_nest()
    if isinstance(states, brainstate.util.FlattedDict):
        states = states.to_nest()
    diff = set(map(str, xs.keys())).difference(states.keys())
    msg = ('The target dict keys and state dict keys do not match,'
           f' target dict contains keys {diff} which are not present in state dict '
           f'at path {current_path()}')
    _handle_mismatch(bool(diff), msg, mismatch)

    result = {}
    for key, value in xs.items():
        str_key = str(key)
        if str_key in states:
            result[key] = msgpack_from_state_dict(value, states[str_key], name=str_key, mismatch=mismatch)
        else:
            # Keep original value if key is missing from state_dict
            result[key] = value

    return result


msgpack_register_serialization(dict, _dict_state_dict, _restore_dict)


def _namedtuple_state_dict(nt) -> Dict[str, Any]:
    return {key: msgpack_to_state_dict(getattr(nt, key)) for key in nt._fields}


def _restore_namedtuple(xs, state_dict: Dict[str, Any], mismatch: MismatchMode = 'error'):
    """Rebuild namedtuple from serialized dict."""
    nt_keys = set(xs._fields)
    # Legacy ``{'name', 'fields', 'values'}`` envelope form. Only reinterpret it
    # when the target itself is not a namedtuple whose fields are exactly those
    # three names, otherwise a genuine ``NT(name, fields, values)`` would be
    # misread and fail to restore.
    if (set(state_dict.keys()) == {'name', 'fields', 'values'}
            and nt_keys != {'name', 'fields', 'values'}):
        state_dict = {state_dict['fields'][str(i)]: state_dict['values'][str(i)]
                      for i in range(len(state_dict['fields']))}

    sd_keys = set(state_dict.keys())

    msg = ('The field names of the state dict and the named tuple do not match,'
           f' got {sd_keys} and {nt_keys} at path {current_path()}')
    _handle_mismatch(sd_keys != nt_keys, msg, mismatch)

    fields = {}
    for field in xs._fields:
        if field in state_dict:
            fields[field] = msgpack_from_state_dict(getattr(xs, field), state_dict[field], name=field,
                                                    mismatch=mismatch)
        else:
            # Keep original value if field is missing from state_dict
            fields[field] = getattr(xs, field)

    try:
        return type(xs)(**fields)
    except TypeError as e:
        raise TypeError(
            f"Failed to reconstruct namedtuple {type(xs).__name__} at path {current_path()}: {e}. "
            f"Ensure the namedtuple class definition is available."
        ) from e


msgpack_register_serialization(
    _NamedTuple,
    _namedtuple_state_dict,
    _restore_namedtuple
)


def _quantity_dict_state(x: u.Quantity) -> Dict[str, Any]:
    """Convert Quantity to state dict.

    Returns:
        Dict containing mantissa (array) and unit information (scale, base, dim, factor)
    """
    return {
        'mantissa': x.mantissa,
        'scale': x.unit.scale,
        'base': x.unit.base,
        'dim': x.unit.dim._dims,
        'factor': x.unit.factor,
    }


def _restore_quantity(x: u.Quantity, state_dict: Dict, mismatch: MismatchMode = 'error') -> u.Quantity:
    unit = u.Unit(
        dim=u.Dimension(state_dict['dim']),
        scale=state_dict['scale'],
        base=state_dict['base'],
        factor=state_dict['factor']
    )
    msg = f'Unit mismatch: expected {x.unit}, got {unit} at path {current_path()}'
    _handle_mismatch(x.unit != unit, msg, mismatch)
    # For 'ignore' and 'warn', use the loaded unit
    return u.Quantity(state_dict['mantissa'], unit=unit)


msgpack_register_serialization(u.Quantity, _quantity_dict_state, _restore_quantity)


def _brainstate_dict_state(x: brainstate.State) -> Dict[str, Any]:
    return msgpack_to_state_dict(x.value)


def _restore_brainstate(x: brainstate.State, state_dict: Dict, mismatch: MismatchMode = 'error') -> brainstate.State:
    """Restore brainstate.State from state dict by mutating its value in-place.

    This function mutates the State object's value attribute rather than creating
    a new State object, which is consistent with how State objects are used
    throughout the codebase.

    Args:
        x: Template State object to restore (will be mutated)
        state_dict: Serialized state dictionary
        mismatch: How to handle mismatches

    Returns:
        The same State object with restored value (for chaining)
    """
    x.value = msgpack_from_state_dict(x.value, state_dict, mismatch=mismatch)
    return x


msgpack_register_serialization(brainstate.State, _brainstate_dict_state, _restore_brainstate)

# NOTE: Only ``args`` and ``keywords`` are serialized; the wrapped function
# ``x.func`` is taken from the *target* template on restore. A ``Partial`` can
# therefore only be restored against a target carrying the correct function, and
# loading without a target yields the raw ``{'args', 'keywords'}`` dict.
msgpack_register_serialization(
    jax.tree_util.Partial,
    lambda x: {
        "args": msgpack_to_state_dict(x.args),
        "keywords": msgpack_to_state_dict(x.keywords),
    },
    lambda x, sd, mismatch='error': jax.tree_util.Partial(
        x.func,
        *msgpack_from_state_dict(x.args, sd["args"], mismatch=mismatch),
        **msgpack_from_state_dict(x.keywords, sd["keywords"], mismatch=mismatch)
    )
)


def _identity_state_dict(x):
    """State-dict for array leaves: the array is itself the serialized leaf."""
    return x


def _restore_array(x, state, mismatch: MismatchMode = 'error'):
    """Validate that a restored array matches the template's shape.

    The loaded value ``state`` replaces the template leaf ``x``. For
    ``mismatch='error'`` a shape change raises ``ValueError``; otherwise it
    warns ('warn') or is silently accepted ('ignore') and the loaded array is
    returned as-is. Dtype is intentionally not enforced, so loading into a
    template of a different dtype (a common coercion) is allowed.
    """
    if hasattr(x, 'shape') and hasattr(state, 'shape'):
        _handle_mismatch(
            tuple(x.shape) != tuple(state.shape),
            f'Array shape mismatch: expected {tuple(x.shape)}, got {tuple(state.shape)} '
            f'at path {current_path()}',
            mismatch,
        )
    return state


msgpack_register_serialization(np.ndarray, _identity_state_dict, _restore_array)
msgpack_register_serialization(jax.Array, _identity_state_dict, _restore_array)


# On-the-wire / disk serialization format

# We encode state-dicts via msgpack, using its custom type extension.
# https://github.com/msgpack/msgpack/blob/master/spec.md
#
# - ndarrays and DeviceArrays are serialized to nested msgpack-encoded string
#   of (shape-tuple, dtype-name (e.g. 'float32'), row-major array-bytes).
#   Note: only simple ndarray types are supported, no objects or fields.
#
# - native complex scalars are converted to nested msgpack-encoded tuples
#   (real, imag).


def _ndarray_to_bytes(arr) -> bytes:
    """Save ndarray to simple msgpack encoding."""
    if isinstance(arr, jax.Array):
        arr = np.array(arr)
    if arr.dtype.hasobject or arr.dtype.isalignedstruct:
        raise ValueError('Object and structured dtypes not supported '
                         'for serialization of ndarrays.')
    tpl = (arr.shape, arr.dtype.name, arr.tobytes('C'))
    return msgpack.packb(tpl, use_bin_type=True)


def _dtype_from_name(name: str):
    """Handle JAX bfloat16 dtype correctly."""
    if name == b'bfloat16':
        return jax.numpy.bfloat16
    else:
        return np.dtype(name)


def _ndarray_from_bytes(data: bytes) -> np.ndarray:
    """Load ndarray from simple msgpack encoding."""
    shape, dtype_name, buffer = msgpack.unpackb(data, raw=True)
    return np.frombuffer(buffer,
                         dtype=_dtype_from_name(dtype_name),
                         count=-1,
                         offset=0).reshape(shape, order='C')


class _MsgpackExtType(enum.IntEnum):
    """Messagepack custom type ids."""
    ndarray = 1
    native_complex = 2
    npscalar = 3


def _msgpack_ext_pack(x):
    """Messagepack encoders for custom types."""
    # TODO: Array here only work when they are fully addressable.
    #       If they are not fully addressable, use the GDA path for checkpointing.
    if isinstance(x, (np.ndarray, jax.Array)):
        return msgpack.ExtType(_MsgpackExtType.ndarray, _ndarray_to_bytes(x))
    if issubclass(type(x), np.generic):
        # pack scalar as ndarray
        return msgpack.ExtType(
            _MsgpackExtType.npscalar,
            _ndarray_to_bytes(np.asarray(x))
        )
    elif isinstance(x, complex):
        return msgpack.ExtType(
            _MsgpackExtType.native_complex,
            msgpack.packb((x.real, x.imag))
        )
    return x


def _msgpack_ext_unpack(code, data):
    """Messagepack decoders for custom types."""
    if code == _MsgpackExtType.ndarray:
        return _ndarray_from_bytes(data)
    elif code == _MsgpackExtType.native_complex:
        complex_tuple = msgpack.unpackb(data)
        return complex(complex_tuple[0], complex_tuple[1])
    elif code == _MsgpackExtType.npscalar:
        ar = _ndarray_from_bytes(data)
        return ar[()]  # unpack ndarray to scalar
    return msgpack.ExtType(code, data)


def _np_convert_in_place(d):
    """Convert any jax devicearray leaves to numpy arrays in place.

    Note: This function modifies nested dictionaries in place. Top-level
    non-dict values cannot be converted in place and will remain unchanged.

    Args:
        d: Dictionary or value to convert
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, jax.Array):
                d[k] = np.array(v)
            elif isinstance(v, dict):
                _np_convert_in_place(v)
    elif isinstance(d, jax.Array):
        return np.array(d)
    return d


def _tuple_to_dict(tpl):
    """Convert tuple to dict with string indices as keys."""
    return {str(x): y for x, y in enumerate(tpl)}


def _dict_to_tuple(dct):
    """Convert dict with string indices to tuple."""
    return tuple(dct[str(i)] for i in range(len(dct)))


def _chunk(arr) -> Dict[str, Any]:
    """Convert array to a canonical dictionary of chunked arrays."""
    itemsize = arr.dtype.itemsize
    if itemsize == 0:
        raise ValueError(f"Cannot chunk array with zero itemsize dtype: {arr.dtype}")

    chunksize = max(1, int(MAX_CHUNK_SIZE / itemsize))

    data = {'__msgpack_chunked_array__': True,
            'shape': _tuple_to_dict(arr.shape)}
    flatarr = arr.reshape(-1)
    chunks = [flatarr[i:i + chunksize] for i in range(0, flatarr.size, chunksize)]
    data['chunks'] = _tuple_to_dict(chunks)
    return data


def _unchunk(data: Dict[str, Any]):
    """Convert canonical dictionary of chunked arrays back into array."""
    assert '__msgpack_chunked_array__' in data
    shape = _dict_to_tuple(data['shape'])
    flatarr = np.concatenate(_dict_to_tuple(data['chunks']))
    return flatarr.reshape(shape)


def _chunk_array_leaves_in_place(d):
    """Convert oversized array leaves to safe chunked form in place.

    Note: This function modifies nested dictionaries in place. Top-level
    non-dict values cannot be converted in place and will remain unchanged.

    Args:
        d: Dictionary or value to convert
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                if v.size * v.dtype.itemsize > MAX_CHUNK_SIZE:
                    d[k] = _chunk(v)
            elif isinstance(v, dict):
                _chunk_array_leaves_in_place(v)
    elif isinstance(d, np.ndarray):
        if d.size * d.dtype.itemsize > MAX_CHUNK_SIZE:
            return _chunk(d)
    return d


def _unchunk_array_leaves_in_place(d):
    """Convert chunked array leaves back into array leaves, in place."""
    if isinstance(d, dict):
        if '__msgpack_chunked_array__' in d:
            return _unchunk(d)
        else:
            for k, v in d.items():
                if isinstance(v, dict) and '__msgpack_chunked_array__' in v:
                    d[k] = _unchunk(v)
                elif isinstance(v, dict):
                    _unchunk_array_leaves_in_place(v)
    return d


def _msgpack_serialize(pytree, in_place: bool = False) -> bytes:
    """Save data structure to bytes in msgpack format.

    Low-level function that only supports python trees with array leaves,
    for custom objects use `to_bytes`.  It splits arrays above MAX_CHUNK_SIZE into
    multiple chunks.

    Args:
      pytree: python tree of dict, list, tuple with python primitives
        and array leaves.
      in_place: boolean specifyng if pytree should be modified in place.

    Returns:
      msgpack-encoded bytes of pytree.
    """
    if not in_place:
        pytree = jax.tree_util.tree_map(lambda x: x, pytree)
    pytree = _np_convert_in_place(pytree)
    pytree = _chunk_array_leaves_in_place(pytree)
    return msgpack.packb(pytree, default=_msgpack_ext_pack, strict_types=True)


def _msgpack_restore(encoded_pytree: bytes, max_size: Optional[int] = None):
    """Restore data structure from bytes in msgpack format.

    Low-level function that only supports python trees with array leaves,
    for custom objects use `from_bytes`.

    Args:
      encoded_pytree: msgpack-encoded bytes of python tree.
      max_size: Optional maximum allowed size in bytes. ``None`` (default)
        means no limit, which is appropriate for checkpoints with large
        (chunked) arrays. When set, data larger than this raises ``ValueError``.

    Returns:
      Python tree of dict, list, tuple with python primitive
      and array leaves.

    Raises:
      ValueError: If data exceeds max_size
      InvalidCheckpointPath: If the msgpack data is corrupt or invalid.
    """
    if max_size is not None and len(encoded_pytree) > max_size:
        raise ValueError(
            f"Checkpoint data too large: {len(encoded_pytree)} bytes "
            f"exceeds maximum {max_size} bytes"
        )

    try:
        state_dict = msgpack.unpackb(encoded_pytree, ext_hook=_msgpack_ext_unpack, raw=False)
    except (msgpack.exceptions.ExtraData,
            msgpack.exceptions.UnpackException,
            msgpack.exceptions.BufferFull,
            ValueError,
            TypeError) as e:
        raise InvalidCheckpointPath(f"Corrupt or invalid checkpoint data: {e}") from e
    return _unchunk_array_leaves_in_place(state_dict)


def _from_bytes(target, encoded_bytes: bytes, mismatch: MismatchMode = 'error',
                max_size: Optional[int] = None):
    """Restore optimizer or other object from msgpack-serialized state-dict.

    Args:
      target: template object with state-dict registrations that matches
        the structure being deserialized from `encoded_bytes`.
      encoded_bytes: msgpack serialized object structurally isomorphic to
        `target`.  Typically, a model or optimizer.
      mismatch: How to handle mismatches between target and state dict.
                'error' (default): raise ValueError on mismatch
                'warn': issue warning and skip mismatched keys
                'ignore': silently skip mismatched keys
      max_size: Optional maximum decoded size in bytes (``None`` = no limit).

    Returns:
      A new object structurally isomorphic to `target` containing the updated
      leaf data from saved data.
    """
    _validate_mismatch(mismatch)
    state_dict = _msgpack_restore(encoded_bytes, max_size=max_size)
    return msgpack_from_state_dict(target, state_dict, mismatch=mismatch)


def _to_bytes(target) -> bytes:
    """Save optimizer or other object as msgpack-serialized state-dict.

    Args:
      target: template object with state-dict registrations to be
        serialized to msgpack format.  Typically, a model or optimizer.

    Returns:
      Bytes of msgpack-encoded state-dict of `target` object.
    """
    state_dict = msgpack_to_state_dict(target)
    return _msgpack_serialize(state_dict, in_place=True)


def _rename_fn(src, dst, overwrite=False):
    """Rename file from src to dst, with overwrite control.

    Uses os.replace() for atomic rename on both Unix and Windows (Python 3.3+).

    Args:
        src: Source file path
        dst: Destination file path
        overwrite: If False, raise AlreadyExistsError when dst exists
    """
    if not os.path.exists(src):
        return

    if not overwrite and os.path.exists(dst):
        raise AlreadyExistsError(dst)

    try:
        os.replace(src, dst)  # Atomic on both platforms
    except OSError:
        # Fallback for edge cases
        if overwrite and os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        os.rename(src, dst)


class AsyncManager(object):
    """
    A simple object to track async checkpointing.

    This class is rewritten from the Flax APIs (https://github.com/google/flax).

    Can be used as a context manager for automatic resource cleanup:

    Example:
        with AsyncManager() as manager:
            msgpack_save(filename, target, async_manager=manager)
    """

    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.save_future = None
        self._closed = False
        self._lock = threading.Lock()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        self.close()
        return False

    def __del__(self):
        """Destructor to clean up resources if close() was not called."""
        if not self._closed:
            try:
                self.close()
            except Exception as e:
                # __del__ must never raise; surface a failing background save as
                # a warning instead of silently dropping it.
                try:
                    warnings.warn(
                        f'AsyncManager encountered an error during cleanup: {e!r}',
                        RuntimeWarning,
                    )
                except Exception:
                    pass

    def close(self):
        """Explicitly close the AsyncManager and release resources.

        Waits for any pending save to complete, then shuts down the executor.
        """
        if self._closed:
            return
        self._closed = True
        self.wait_previous_save()
        self.executor.shutdown(wait=True)

    def wait_previous_save(self):
        """Block until the previous save finishes, to keep files' consistency.

        Re-raises any exception raised by the background save task.
        """
        if self.save_future is not None:
            self.save_future.result()

    def save_async(self, task: Callable[[], Any]):
        """Run a task async. The future will be tracked as self.save_future.

        Args:
          task: The callable to be executed asynchronously.

        Raises:
            RuntimeError: If the AsyncManager has been closed.
        """
        if self._closed:
            raise RuntimeError("Cannot save with a closed AsyncManager")

        with self._lock:
            # Only warn when a *new* save overlaps an unfinished one; waiting on
            # a completed (or soon-to-complete) save is normal and silent.
            if self.save_future is not None and not self.save_future.done():
                warnings.warn(
                    'A previous async braintools.file.msgpack_save has not finished '
                    'yet; waiting for it to complete before starting the next save.',
                    UserWarning,
                )
            self.wait_previous_save()
            self.save_future = self.executor.submit(task)  # type: ignore


def _save_main_ckpt_file(
    target: bytes,
    filename: str,
    overwrite: bool,
):
    """Save the main checkpoint file via file system.

    This function implements pre-emption safe saving by:
    1. Writing to a temporary file first
    2. Atomically renaming to the final destination

    Args:
        target: The serialized checkpoint bytes
        filename: The final checkpoint file path
        overwrite: Whether to overwrite existing files
    """
    # Use a unique temporary file in the same directory for atomic rename. A
    # unique suffix prevents concurrent saves (multiple managers/processes, or a
    # retried save) targeting the same final path from clobbering each other's
    # in-flight temp file.
    tmp_filename = f'{filename}.tmp-{uuid.uuid4().hex}'

    try:
        # Write to temporary file
        with open(tmp_filename, 'wb') as fp:
            fp.write(target)

        # Atomically rename to final destination
        _rename_fn(tmp_filename, filename, overwrite=overwrite)
    except Exception:
        # Clean up temporary file on failure
        if os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except OSError:
                pass
        raise


def msgpack_save(
    filename: Union[str, os.PathLike],
    target: brainstate.typing.PyTree,
    overwrite: bool = True,
    async_manager: Optional[AsyncManager] = None,
    verbose: bool = False,
) -> str:
    """
    Save a checkpoint of the model. Suitable for single-host using the ``msgpack`` library.

    This function is rewritten from the Flax APIs (https://github.com/google/flax).

    In this method, every JAX process saves the checkpoint on its own. Do not
    use it if you have multiple processes and you intend for them to save data
    to a common directory (e.g., a GCloud bucket). To save multi-process
    checkpoints to a shared storage or to save `GlobalDeviceArray`s, use
    `multiprocess_save()` instead.

    Pre-emption safe by writing to a unique temporary file before a final rename.
    When ``async_manager`` is given, the *file write* runs in the background; the
    target is serialized synchronously before returning so the on-disk snapshot
    is consistent even if the model keeps training. Wait for the write with
    ``async_manager.wait_previous_save()``.

    Parameters
    ----------
    filename: str or os.PathLike
      path to store the checkpoint file in.
    target: Any
      serializable object.
    overwrite: bool
      overwrite an existing checkpoint at ``filename`` (default: True). When
      False, an existing file raises :class:`InvalidCheckpointPath`.
    async_manager: optional, AsyncManager
      if defined, the file write will run without blocking the main thread. Only
      works for single host. Note that an ongoing save will still block
      subsequent saves, to make sure overwrite/keep logic works correctly.
    verbose: bool
      Whether to print progress information (default: False).

    Returns
    -------
    out: str
      Filename of the saved checkpoint.
    """
    check_msgpack()
    filename = os.fspath(filename)
    if verbose:
        print(f'Saving checkpoint into {filename}')

    # Make sure all saves are finished before the logic
    # of checking and removing outdated checkpoints happens.
    if async_manager:
        async_manager.wait_previous_save()

    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory {dirname}: {e}") from e
    if not overwrite and os.path.exists(filename):
        raise InvalidCheckpointPath(filename)

    # Warn on Windows if path exceeds MAX_PATH limitation
    if sys.platform == 'win32':
        abs_path = os.path.abspath(filename)
        if len(abs_path) > 260:
            warnings.warn(
                f"Path length {len(abs_path)} exceeds Windows MAX_PATH (260). "
                "Consider using shorter paths or enabling long path support.",
                UserWarning
            )

    if isinstance(target, brainstate.util.FlattedDict):
        target = target.to_nest()
    target = _to_bytes(target)

    # Save the files via I/O sync or async.
    def save_main_ckpt_task():
        return _save_main_ckpt_file(target, filename, overwrite)

    if async_manager:
        async_manager.save_async(save_main_ckpt_task)
    else:
        save_main_ckpt_task()

    return filename


def msgpack_load(
    filename: Union[str, os.PathLike],
    target: Optional[Any] = None,
    parallel: bool = True,
    mismatch: MismatchMode = 'error',
    max_size: Optional[int] = None,
    verbose: bool = False,
) -> brainstate.typing.PyTree:
    """
    Load the checkpoint from the given path using the ``msgpack`` library.

    This function is rewritten from the Flax APIs (https://github.com/google/flax).

    Parameters
    ----------
    filename: str or os.PathLike
        checkpoint file to restore from.
    target: Any
        the object to restore the state into. If None, the state is returned as a
        nested dict (with all dict keys as strings; see Notes).
    parallel: bool
        whether to read large seekable checkpoints with multiple threads.
    mismatch: MismatchMode
        How to handle mismatches between target and state dict (dict keys,
        list/tuple length, namedtuple fields, unit, and array shape).
        'error' (default): raise ValueError on mismatch
        'warn': issue warning and keep the target's value for the mismatch
        'ignore': silently keep the target's value for the mismatch
    max_size: optional int
        Maximum decoded checkpoint size in bytes (``None`` = no limit).
    verbose: bool
        Whether to print progress information (default: False).

    Returns
    -------
    out: Any
        If ``target`` is given, a structure isomorphic to ``target`` with the
        restored leaf data (``State`` leaves are restored in place). If
        ``target`` is None, the raw nested state dict read from the file.

    Raises
    ------
    ValueError
        If ``filename`` does not exist, or on a mismatch when ``mismatch='error'``.
    InvalidCheckpointPath
        If the checkpoint data is corrupt or cannot be decoded.

    Notes
    -----
    Without a ``target``, dict keys come back as strings and dict subclasses
    (e.g. ``OrderedDict``) become plain dicts, because keys are stringified on
    save. Provide a ``target`` to recover the original key types and structure.
    """
    check_msgpack()
    _validate_mismatch(mismatch)
    filename = os.fspath(filename)

    if not os.path.exists(filename):
        raise ValueError(f'Checkpoint not found: {filename}')
    if verbose:
        print(f'Loading checkpoint from {filename}')
    file_size = os.path.getsize(filename)

    buf_size = PARALLEL_READ_BUF_SIZE
    with open(filename, 'rb') as fp:
        # Only use the parallel path for files large enough to benefit; for small
        # files the extra per-chunk file opens are pure overhead.
        if parallel and fp.seekable() and file_size > buf_size:
            num_chunks = (file_size + buf_size - 1) // buf_size  # Ceiling division
            try:
                checkpoint_contents = bytearray(file_size)
            except MemoryError:
                # Fallback to sequential read for very large files
                if verbose:
                    warnings.warn(
                        f"Insufficient memory for parallel load of {file_size} bytes. "
                        "Falling back to sequential read.",
                        UserWarning
                    )
                fp.seek(0)
                checkpoint_contents = fp.read()
            else:
                def read_chunk(i):
                    # Re-open the file per chunk so the reads truly run in parallel.
                    with open(filename, 'rb') as f:
                        f.seek(i * buf_size)
                        buf = f.read(buf_size)
                        if buf:
                            checkpoint_contents[i * buf_size:i * buf_size + len(buf)] = buf
                        return len(buf)

                pool_size = min(32, max(1, os.cpu_count() or 1))
                with ThreadPoolExecutor(pool_size) as pool:
                    list(pool.map(read_chunk, range(num_chunks)))
        else:
            checkpoint_contents = fp.read()

    state_dict = _msgpack_restore(checkpoint_contents, max_size=max_size)
    if target is not None:
        state_dict = msgpack_from_state_dict(target, state_dict, mismatch=mismatch)

    return state_dict
