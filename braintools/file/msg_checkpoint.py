# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
import logging
import os
import re
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import thread
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import brainstate as bst
import brainunit as u
import jax
import numpy as np
from jax import monitoring

try:
    import msgpack
except (ModuleNotFoundError, ImportError):
    msgpack = None

__all__ = [
    'msgpack_save', 'msgpack_load', 'AsyncManager',
]


class AlreadyExistsError(Exception):
    """Attempting to overwrite a file via copy.

    You can pass ``overwrite=True`` to disable this behavior and overwrite
    existing files in.
    """

    def __init__(self, path):
        super().__init__(f'Trying overwrite an existing file: "{path}".')


class MPACheckpointingRequiredError(Exception):
    """To optimally save and restore a multiprocess array (GDA or jax Array outputted from pjit), use GlobalAsyncCheckpointManager.

    You can create an GlobalAsyncCheckpointManager at top-level and pass it as
    argument::

      from jax.experimental.gda_serialization import serialization as gdas
      gda_manager = gdas.GlobalAsyncCheckpointManager()
      brainpy.checkpoints.save(..., gda_manager=gda_manager)
    """

    def __init__(self, path, step):
        super().__init__(
            f'Checkpoint failed at step: "{step}" and path: "{path}": Target '
            'contains a multiprocess array should be saved/restored with a '
            'GlobalAsyncCheckpointManager.')


class InvalidCheckpointPath(Exception):
    """A checkpoint cannot be stored in a directory that already has

    a checkpoint at the current or a later step.

    You can pass ``overwrite=True`` to disable this behavior and
    overwrite existing checkpoints in the target directory.
    """

    def __init__(self, path):
        super().__init__(f'Invalid checkpoint at "{path}".')


class InvalidCheckpointError(Exception):
    """A checkpoint cannot be stored in a directory that already has

    a checkpoint at the current or a later step.

    You can pass ``overwrite=True`` to disable this behavior and
    overwrite existing checkpoints in the target directory.
    """

    def __init__(self, path, step):
        super().__init__(
            f'Trying to save an outdated checkpoint at step: "{step}" and path: "{path}".'
        )


_LAST_CHECKPOINT_WRITE_TIME = time.time()
_READ_CHECKPOINT_EVENT: str = '/jax/checkpoint/read/durations_sec'
_WRITE_CHECKPOINT_EVENT: str = '/jax/checkpoint/write/durations_sec'

# Single-group reg-exps for int or float numerical substrings.
# captures sign:
SIGNED_FLOAT_RE = re.compile(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# does not capture sign:
UNSIGNED_FLOAT_RE = re.compile(r'[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
# Module name followed by number.
MODULE_NUM_RE = re.compile(r'(.*)_\d+$')
# Alternative schemes handled by `gfile`, e.g. on Google Cloud Storage (GCS).
SCHEME_RE = re.compile('^(?P<scheme>[a-z][a-z0-9.+-]+://)?(?P<path>.*)', re.I)

# Multiprocess arrays (GlobalDeviceArray, or JAX array with multiprocess
# sharding) is across processes and will be stored in directories with this
# postfix, seperated from the non-distributed data (e.g. the larger pytree)
MP_ARRAY_POSTFIX = '_gda'
# Occurrences of multiprocess arrays in the target pytree will be
# replaced by this string placeholder.
MP_ARRAY_PH = '//GDAPlaceholder:'

# Add a copy-success file to a distributed array directory to indicate the
# array save is complete.
# We need this for GCS because GCS's directory move is not atomic.
COMMIT_SUCCESS_FILE = 'commit_success.txt'

# Orbax main checkpoint file name.
ORBAX_CKPT_FILENAME = 'checkpoint'

# Chunking array leaves

# msgpack has a hard limit of 2**31 - 1 bytes per object leaf.  To circumvent
# this limit for giant arrays (e.g. embedding tables), we traverse the tree
# and break up arrays near the limit into flattened array chunks.
# True limit is 2**31 - 1, but leave a margin for encoding padding.
MAX_CHUNK_SIZE = 2 ** 30

# containing jax.Array attribute.
MultiprocessArrayType = Any

_STATE_DICT_REGISTRY: Dict[Any, Any] = {}


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


def from_state_dict(target, state: Dict[str, Any], name: str = '.'):
    """Restores the state of the given target using a state dict.

    This function takes the current target as an argument. This
    lets us know the exact structure of the target,
    as well as lets us add assertions that shapes and dtypes don't change.

    In practice, none of the leaf values in `target` are actually
    used. Only the tree structure, shapes and dtypes.

    Args:
      target: the object of which the state should be restored.
      state: a dictionary generated by `to_state_dict` with the desired new
             state for `target`.
      name: name of branch taken, used to improve deserialization error messages.
    Returns:
      A copy of the object with the restored state.
    """
    ty = _NamedTuple if _is_namedtuple(target) else type(target)
    for t in _STATE_DICT_REGISTRY.keys():
        if issubclass(ty, t):
            ty = t
            break
    else:
        return state
    ty_from_state_dict = _STATE_DICT_REGISTRY[ty][1]
    with _record_path(name):
        return ty_from_state_dict(target, state)


def to_state_dict(target) -> Dict[str, Any]:
    """Returns a dictionary with the state of the given target."""
    ty = _NamedTuple if _is_namedtuple(target) else type(target)

    for t in _STATE_DICT_REGISTRY.keys():
        if issubclass(ty, t):
            ty = t
            break
    else:
        return target

    ty_to_state_dict = _STATE_DICT_REGISTRY[ty][0]
    state_dict = ty_to_state_dict(target)
    if isinstance(state_dict, dict):
        for key in state_dict.keys():
            assert isinstance(key, str), 'A state dict must only have string keys.'
        return state_dict
    elif isinstance(state_dict, jax.Array):
        return state_dict
    else:
        raise TypeError


def register_serialization_state(ty,
                                 ty_to_state_dict,
                                 ty_from_state_dict,
                                 override=False):
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
    return {str(i): to_state_dict(x) for i, x in enumerate(xs)}


def _restore_list(xs, state_dict: Dict[str, Any]) -> List[Any]:
    if len(state_dict) != len(xs):
        raise ValueError('The size of the list and the state dict do not match,'
                         f' got {len(xs)} and {len(state_dict)} '
                         f'at path {current_path()}')
    ys = []
    for i in range(len(state_dict)):
        y = from_state_dict(xs[i], state_dict[str(i)], name=str(i))
        ys.append(y)
    return ys


register_serialization_state(list, _list_state_dict, _restore_list)
register_serialization_state(tuple,
                             _list_state_dict,
                             lambda xs, state_dict: tuple(_restore_list(list(xs), state_dict)))


def _dict_state_dict(xs: Dict[str, Any]) -> Dict[str, Any]:
    str_keys = set(str(k) for k in xs.keys())
    if len(str_keys) != len(xs):
        raise ValueError('Dict keys do not have a unique string representation: '
                         f'{str_keys} vs given: {xs}')
    return {str(key): to_state_dict(value) for key, value in xs.items()}


def _restore_dict(xs, states: Dict[str, Any]) -> Dict[str, Any]:
    diff = set(map(str, xs.keys())).difference(states.keys())
    if diff:
        raise ValueError('The target dict keys and state dict keys do not match,'
                         f' target dict contains keys {diff} which are not present in state dict '
                         f'at path {current_path()}')

    return {key: from_state_dict(value, states[str(key)], name=str(key))
            for key, value in xs.items()}


register_serialization_state(dict, _dict_state_dict, _restore_dict)


def _namedtuple_state_dict(nt) -> Dict[str, Any]:
    return {key: to_state_dict(getattr(nt, key)) for key in nt._fields}


def _restore_namedtuple(xs, state_dict: Dict[str, Any]):
    """Rebuild namedtuple from serialized dict."""
    if set(state_dict.keys()) == {'name', 'fields', 'values'}:
        state_dict = {state_dict['fields'][str(i)]: state_dict['values'][str(i)]
                      for i in range(len(state_dict['fields']))}

    sd_keys = set(state_dict.keys())
    nt_keys = set(xs._fields)

    if sd_keys != nt_keys:
        raise ValueError('The field names of the state dict and the named tuple do not match,'
                         f' got {sd_keys} and {nt_keys} at path {current_path()}')
    fields = {
        k: from_state_dict(getattr(xs, k), v, name=k)
        for k, v in state_dict.items()
    }
    return type(xs)(**fields)


register_serialization_state(_NamedTuple,
                             _namedtuple_state_dict,
                             _restore_namedtuple)


def _quantity_dict_state(x: u.Quantity) -> Dict[str, jax.Array]:
    return {'mantissa': x.mantissa, 'scale': x.unit.scale, 'base': x.unit.base, 'dim': x.unit.dim._dims}


def _restore_quantity(x: u.Quantity, state_dict: Dict) -> u.Quantity:
    unit = u.Unit(dim=u.Dimension(state_dict['dim']), scale=state_dict['scale'], base=state_dict['base'])
    assert x.unit == unit
    return u.Quantity(state_dict['mantissa'], unit=unit)


register_serialization_state(u.Quantity, _quantity_dict_state, _restore_quantity)


def _brainstate_dict_state(x: bst.State) -> Dict[str, Any]:
    return to_state_dict(x.value)


def _restore_brainstate(x: bst.State, state_dict: Dict) -> bst.State:
    x.value = from_state_dict(x.value, state_dict)
    return x


register_serialization_state(bst.State, _brainstate_dict_state, _restore_brainstate)


register_serialization_state(
    jax.tree_util.Partial,
    lambda x: {"args": to_state_dict(x.args),
               "keywords": to_state_dict(x.keywords), },
    lambda x, sd: jax.tree_util.Partial(x.func,
                                        *from_state_dict(x.args, sd["args"]),
                                        **from_state_dict(x.keywords, sd["keywords"]))
)


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
    # If they are not fully addressable, use the GDA path for checkpointing.
    if isinstance(x, (np.ndarray, jax.Array)):
        return msgpack.ExtType(_MsgpackExtType.ndarray, _ndarray_to_bytes(x))
    if np.issctype(type(x)):
        # pack scalar as ndarray
        return msgpack.ExtType(_MsgpackExtType.npscalar,
                               _ndarray_to_bytes(np.asarray(x)))
    elif isinstance(x, complex):
        return msgpack.ExtType(_MsgpackExtType.native_complex,
                               msgpack.packb((x.real, x.imag)))
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
    """Convert any jax devicearray leaves to numpy arrays in place."""
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, jax.Array):
                d[k] = np.array(v)
            elif isinstance(v, dict):
                _np_convert_in_place(v)
    elif isinstance(d, jax.Array):
        return np.array(d)
    return d


_tuple_to_dict = lambda tpl: {str(x): y for x, y in enumerate(tpl)}
_dict_to_tuple = lambda dct: tuple(dct[str(i)] for i in range(len(dct)))


def _chunk(arr) -> Dict[str, Any]:
    """Convert array to a canonical dictionary of chunked arrays."""
    chunksize = max(1, int(MAX_CHUNK_SIZE / arr.dtype.itemsize))
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
    """Convert oversized array leaves to safe chunked form in place."""
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


def msgpack_serialize(pytree, in_place: bool = False) -> bytes:
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


def msgpack_restore(encoded_pytree: bytes):
    """Restore data structure from bytes in msgpack format.

    Low-level function that only supports python trees with array leaves,
    for custom objects use `from_bytes`.

    Args:
      encoded_pytree: msgpack-encoded bytes of python tree.

    Returns:
      Python tree of dict, list, tuple with python primitive
      and array leaves.
    """
    state_dict = msgpack.unpackb(
        encoded_pytree, ext_hook=_msgpack_ext_unpack, raw=False)
    return _unchunk_array_leaves_in_place(state_dict)


def from_bytes(target, encoded_bytes: bytes):
    """Restore optimizer or other object from msgpack-serialized state-dict.

    Args:
      target: template object with state-dict registrations that matches
        the structure being deserialized from `encoded_bytes`.
      encoded_bytes: msgpack serialized object structurally isomorphic to
        `target`.  Typically a flax model or optimizer.

    Returns:
      A new object structurally isomorphic to `target` containing the updated
      leaf data from saved data.
    """
    state_dict = msgpack_restore(encoded_bytes)
    return from_state_dict(target, state_dict)


def to_bytes(target) -> bytes:
    """Save optimizer or other object as msgpack-serialized state-dict.

    Args:
      target: template object with state-dict registrations to be
        serialized to msgpack format.  Typically a flax model or optimizer.

    Returns:
      Bytes of msgpack-encoded state-dict of `target` object.
    """
    state_dict = to_state_dict(target)
    return msgpack_serialize(state_dict, in_place=True)


# the empty node is a struct.dataclass to be compatible with JAX.
class _EmptyNode:
    pass


def flatten_dict(xs, keep_empty_nodes=False, is_leaf=None, sep=None):
    """Flatten a nested dictionary.

    The nested keys are flattened to a tuple.
    See `unflatten_dict` on how to restore the
    nested dictionary structure.

    Example::

      xs = {'foo': 1, 'bar': {'a': 2, 'b': {}}}
      flat_xs = flatten_dict(xs)
      print(flat_xs)
      # {
      #   ('foo',): 1,
      #   ('bar', 'a'): 2,
      # }

    Note that empty dictionaries are ignored and
    will not be restored by `unflatten_dict`.

    Args:
      xs: a nested dictionary
      keep_empty_nodes: replaces empty dictionaries
        with `traverse_util.empty_node`.
      is_leaf: an optional function that takes the
        next nested dictionary and nested keys and
        returns True if the nested dictionary is a
        leaf (i.e., should not be flattened further).
      sep: if specified, then the keys of the returned
        dictionary will be `sep`-joined strings (if
        `None`, then keys will be tuples).
    Returns:
      The flattened dictionary.
    """
    assert isinstance(xs, dict), f'expected (frozen)dict; got {type(xs)}'

    def _key(path):
        if sep is None:
            return path
        return sep.join(path)

    def _flatten(xs, prefix):
        if not isinstance(xs, dict) or (is_leaf and is_leaf(prefix, xs)):
            return {_key(prefix): xs}
        result = {}
        is_empty = True
        for key, value in xs.items():
            is_empty = False
            path = prefix + (key,)
            result.update(_flatten(value, path))
        if keep_empty_nodes and is_empty:
            if prefix == ():  # when the whole input is empty
                return {}
            return {_key(prefix): _EmptyNode()}
        return result

    return _flatten(xs, ())


def unflatten_dict(xs, sep=None):
    """Unflatten a dictionary.

    See `flatten_dict`

    Example::

      flat_xs = {
        ('foo',): 1,
        ('bar', 'a'): 2,
      }
      xs = unflatten_dict(flat_xs)
      print(xs)
      # {
      #   'foo': 1
      #   'bar': {'a': 2}
      # }

    Args:
      xs: a flattened dictionary
      sep: separator (same as used with `flatten_dict()`).
    Returns:
      The nested dictionary.
    """
    assert isinstance(xs, dict), f'input is not a dict; it is a {type(xs)}'
    result = {}
    for path, value in xs.items():
        if sep is not None:
            path = path.split(sep)
        if isinstance(value, _EmptyNode):
            value = {}
        cursor = result
        for key in path[:-1]:
            if key not in cursor:
                cursor[key] = {}
            cursor = cursor[key]
        cursor[path[-1]] = value
    return result


def _rename_fn(src, dst, overwrite=False):
    if os.path.exists(src):
        if os.path.exists(dst) and not overwrite:
            raise AlreadyExistsError(dst)
        return os.rename(src, dst)


class AsyncManager(object):
    """A simple object to track async checkpointing.

    How to use: create an instance and pass to `brainpy.checkpoints.save()` calls:
      am = AsyncManager()
      brainpy.checkpoints.save(..., async_manager=am)
    """

    def __init__(self, max_workers: int = 1):
        self.executor = thread.ThreadPoolExecutor(max_workers=max_workers)
        self.save_future = None

    def wait_previous_save(self):
        """Block until the previous save finishes, to keep files' consistency."""
        if self.save_future and not self.save_future.done():
            warnings.warn(
                'The previous async brainpy.checkpoints.save has not finished yet. Waiting '
                'for it to complete before the next save.',
                UserWarning
            )
            self.save_future.result()

    def save_async(self, task: Callable[[], Any]):
        """Run a task async. The future will be tracked as self.save_future.

        Args:
          task: The callable to be executed asynchrously.
        """
        self.wait_previous_save()
        self.save_future = self.executor.submit(task)  # type: ignore


def _record_saved_duration(checkpoint_start_time: float):
    """Record program duration that is accounted for by this checkpoint.

    For the very first checkpoint, this is the interval between program init and
    current checkpoint start time.

    Note that we use the checkpoint start time instead of end time. The saved
    duration should not include prallel training duration while the async
    checkpoint is being written in the background.

    Args:
      checkpoint_start_time: Start time of current checkpoint.
    """
    global _LAST_CHECKPOINT_WRITE_TIME
    # Note: for the very first checkpoint, this is the interval between program
    # init and the current checkpoint start time.
    duration_since_last_checkpoint = checkpoint_start_time - _LAST_CHECKPOINT_WRITE_TIME
    if monitoring is not None:
        monitoring.record_event_duration_secs(
            '/jax/checkpoint/write/duration_since_last_checkpoint_secs',
            duration_since_last_checkpoint)
    _LAST_CHECKPOINT_WRITE_TIME = checkpoint_start_time


def _save_commit2(filename: str,
                  overwrite: bool,
                  ckpt_start_time: float,
                  has_mpa: bool,
                  write_commit_success: bool,
                  async_manager: Optional[AsyncManager] = None) -> None:
    """Commit changes after saving checkpoints to disk.

    This function does the following, sequentially:
      1. Make sure all ckpt writing finishes, and rename them from temp path to
      the final path.
      2. Remove newer checkpoints (files that ordered larger than this save) if
      `overwrite=True`.
      3. Remove old checkpoint files based on `keep` and `keep_every_n_steps`.
      4. Record program duration saved by this checkpoint.
    """
    ckpt_path = os.path.dirname(filename)
    ckpt_tmp_path = os.path.join(ckpt_path, 'tmp')
    mpa_ckpt_tmp_path, mpa_ckpt_path = ckpt_tmp_path + MP_ARRAY_POSTFIX, ckpt_path + MP_ARRAY_POSTFIX
    # Rename the multiprocess array path once serialization and writing finished.
    if has_mpa:
        if write_commit_success:
            commit_success_path = os.path.join(mpa_ckpt_path, COMMIT_SUCCESS_FILE)
            with open(commit_success_path, 'w', encoding='utf-8') as f:
                f.write(f'Checkpoint commit was successful to {mpa_ckpt_path}')
        else:
            # Commits are a two stage process (renaming the array folder and renaming
            # the main ckpt file in sequential order). We always try to overwrite
            # here because the array ckpt might be already renamed in a previously
            # interrupted commit. NOTE: io.rename does not support overwriting
            # directories via `rename` so we manually overwrite it.
            if os.path.exists(mpa_ckpt_path):
                logging.info('Removing outdated checkpoint at %s', mpa_ckpt_path)
                shutil.rmtree(mpa_ckpt_path)
            _rename_fn(mpa_ckpt_tmp_path, mpa_ckpt_path)
    # Commit the main checkpoint file after arrays (if any) are committed
    if async_manager:
        async_manager.wait_previous_save()
    _rename_fn(ckpt_tmp_path, ckpt_path, overwrite=overwrite)
    logging.info('Saved checkpoint at %s', ckpt_path)

    # Remove newer and older invalid checkpoints.
    _record_saved_duration(ckpt_start_time)


def _save_main_ckpt_file2(
    target: bytes,
    has_mpa: bool,
    filename: str,
    overwrite: bool,
    ckpt_start_time: float
):
    """Save the main checkpoint file via file system."""
    with open(filename, 'wb') as fp:
        fp.write(target)
    # Postpone the commitment of checkpoint to after MPA writes are done.
    if not has_mpa:
        _save_commit2(filename, overwrite, ckpt_start_time, has_mpa=False, write_commit_success=False)


def msgpack_save(
    filename: str,
    target: bst.typing.PyTree,
    overwrite: bool = True,
    async_manager: Optional[AsyncManager] = None,
    verbose: bool = True,
) -> None:
    """
    Save a checkpoint of the model. Suitable for single-host using the ``msgpack`` library.

    In this method, every JAX process saves the checkpoint on its own. Do not
    use it if you have multiple processes and you intend for them to save data
    to a common directory (e.g., a GCloud bucket). To save multi-process
    checkpoints to a shared storage or to save `GlobalDeviceArray`s, use
    `multiprocess_save()` instead.

    Pre-emption safe by writing to temporary before a final rename and cleanup
    of past files. However, if async_manager is used, the final
    commit will happen inside an async callback, which can be explicitly waited
    by calling `async_manager.wait_previous_save()`.

    Parameters
    ----------
    filename: str
      str or pathlib-like path to store checkpoint files in.
    target: Any
      serializable object.
    overwrite: bool
      overwrite existing checkpoint files if a checkpoint at the
      current or a later step already exits (default: False).
    async_manager: optional, AsyncManager
      if defined, the save will run without blocking the main
      thread. Only works for single host. Note that an ongoing save will still
      block subsequent saves, to make sure overwrite/keep logic works correctly.
    verbose: bool
      Whether output the print information.

    Returns
    -------
    out: str
      Filename of saved checkpoint.
    """
    check_msgpack()
    if verbose:
        print(f'Saving checkpoint into {filename}')
    start_time = time.time()
    # Make sure all saves are finished before the logic of checking and removing
    # outdated checkpoints happens.
    if async_manager:
        async_manager.wait_previous_save()

    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not overwrite and os.path.exists(filename):
        raise InvalidCheckpointPath(filename)

    if isinstance(target, bst.util.FlattedDict):
        target = target.to_nest()
    target = to_bytes(target)

    # Save the files via I/O sync or async.
    def save_main_ckpt_task():
        return _save_main_ckpt_file2(target, False, filename, overwrite, start_time)

    if async_manager:
        async_manager.save_async(save_main_ckpt_task)
    else:
        save_main_ckpt_task()
    end_time = time.time()
    if monitoring is not None:
        monitoring.record_event_duration_secs(_WRITE_CHECKPOINT_EVENT,
                                              end_time - start_time)


def _use_multiprocess_serialization(value: Any) -> bool:
    """Use GlobalAsyncCheckpointManager to save the array if it's only partially available on this host."""
    if isinstance(value, jax.Array):
        return not value.is_fully_addressable
    return False


def _split_mp_arrays(
    target: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Tuple[MultiprocessArrayType, str]]]:
    """Split out the multiprocess arrays from the target pytree to save."""
    # When target is a single leaf instead of a pytree dict.
    if not isinstance(target, dict):
        if _use_multiprocess_serialization(target):
            return MP_ARRAY_PH, [(target, '')]
        return target, []
    # Traverse the target and handle distributed arrays.
    flattened = flatten_dict(target, keep_empty_nodes=True)
    mpa_targets = []
    for key, value in flattened.items():
        if _use_multiprocess_serialization(value):
            subpath = '/'.join(key)
            mpa_targets.append((value, subpath))
            flattened[key] = MP_ARRAY_PH + subpath
    target = unflatten_dict(flattened)
    return target, mpa_targets


def msgpack_load(
    filename: str,
    target: Optional[Any] = None,
    parallel: bool = True,
) -> bst.typing.PyTree:
    """
    Load the checkpoint from the given checkpoint path using the ``msgpack`` library.

    Parameters
    ----------
    filename: str
        checkpoint file or directory of checkpoints to restore from.
    target: Any
        the object to restore the state into. If None, the state is returned as a dict.
    parallel: bool
        whether to load seekable checkpoints in parallel, for speed.

    Returns
    -------
    out: Any
      Restored `target` updated from checkpoint file, or if no step specified and
      no checkpoint files present, returns the passed-in `target` unchanged.
      If a file path is specified and is not found, the passed-in `target` will be
      returned. This is to match the behavior of the case where a directory path
      is specified but the directory has not yet been created.
    """
    check_msgpack()

    start_time = time.time()
    if not os.path.exists(filename):
        raise ValueError(f'Checkpoint not found: {filename}')
    sys.stdout.write(f'Loading checkpoint from {filename}\n')
    sys.stdout.flush()
    file_size = os.path.getsize(filename)

    if isinstance(target, bst.util.FlattedDict):
        target = target.to_nest()

    with open(filename, 'rb') as fp:
        if parallel and fp.seekable():
            buf_size = 128 << 20  # 128M buffer.
            num_bufs = file_size / buf_size
            logging.debug('num_bufs: %d', num_bufs)
            checkpoint_contents = bytearray(file_size)

            def read_chunk(i):
                # NOTE: We have to re-open the file to read each chunk, otherwise the
                # parallelism has no effect. But we could reuse the file pointers
                # within each thread.
                with open(filename, 'rb') as f:
                    f.seek(i * buf_size)
                    buf = f.read(buf_size)
                    if buf:
                        checkpoint_contents[i * buf_size:i * buf_size + len(buf)] = buf
                    return len(buf) / buf_size

            pool_size = 32
            pool = thread.ThreadPoolExecutor(pool_size)
            results = pool.map(read_chunk, range(int(num_bufs) + 1))
            pool.shutdown(wait=False)
            logging.debug(f'results: {list(results)}')
        else:
            checkpoint_contents = fp.read()

    state_dict = msgpack_restore(checkpoint_contents)
    if target is not None:
        state_dict = from_state_dict(target, state_dict)

    end_time = time.time()
    if monitoring is not None:
        monitoring.record_event_duration_secs(_READ_CHECKPOINT_EVENT, end_time - start_time)

    return state_dict
