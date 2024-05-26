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

# -*- coding: utf-8 -*-

import numbers
from typing import Union, Callable, Optional, Sequence

import braincore as bc
import jax
import jax.numpy as jnp
import numpy as np

from ._base import to_size

__all__ = [
  'parameter',
  'state',
  'noise',
]


def _is_scalar(x):
  return isinstance(x, numbers.Number)


def parameter(
    param: Union[Callable, np.ndarray, jax.Array, float, int, bool],
    sizes: Union[int, Sequence[int]],
    batch_or_mode: Optional[Union[int, bool, bc.mixin.Mode]] = None,
    allow_none: bool = True,
    allow_scalar: bool = True,
):
  """Initialize parameters.

  Parameters
  ----------
  param: callable, Initializer, bm.ndarray, jnp.ndarray, onp.ndarray, float, int, bool
    The initialization of the parameter.
    - If it is None, the created parameter will be None.
    - If it is a callable function :math:`f`, the ``f(size)`` will be returned.
    - If it is an instance of :py:class:`init.Initializer``, the ``f(size)`` will be returned.
    - If it is a tensor, then this function check whether ``tensor.shape`` is equal to the given ``size``.
  sizes: int, sequence of int
    The shape of the parameter.
  batch_or_mode: int, bool, braincore.mixin.Mode
    The batch size or the mode.
  allow_none: bool
    Whether allow the parameter is None.
  allow_scalar: bool
    Whether allow the parameter is a scalar value.

  Returns
  -------
  param: ArrayType, float, int, bool, None
    The initialized parameter.

  See Also
  --------
  variable_, noise, delay
  """
  if param is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'Expect a parameter with type of float, ArrayType, Initializer, or '
                       f'Callable function, but we got None. ')
  sizes = list(to_size(sizes))
  if allow_scalar and _is_scalar(param):
    return param

  batch_axis = None
  batch_size = None
  if isinstance(batch_or_mode, bc.mixin.Mode):
    if batch_or_mode.has(bc.mixin.Batching):
      batch_axis = batch_or_mode.batch_axis
      batch_size = batch_or_mode.batch_size
  elif batch_or_mode in (None, False):
    pass
  elif isinstance(batch_or_mode, int):
    batch_axis = 0
    batch_size = batch_or_mode
  else:
    raise ValueError('Unknown batch_or_mode.')
  if batch_size is not None:
    sizes.insert(batch_axis, batch_size)

  if callable(param):
    return param(sizes)
  elif isinstance(param, (np.ndarray, jax.Array)):
    param = jnp.asarray(param)
    if batch_size is not None:
      param = jnp.repeat(jnp.expand_dims(param, axis=batch_axis), batch_size, axis=batch_axis)
  elif isinstance(param, bc.State):
    param = param
    if batch_size is not None:
      param = type(param)(jnp.repeat(jnp.expand_dims(param.value, axis=batch_axis), batch_size, axis=batch_axis))
  else:
    raise ValueError(f'Unknown parameter type: {type(param)}')

  if allow_scalar:
    if param.shape == () or param.shape == (1,):
      return param
  if param.shape != tuple(sizes):
    raise ValueError(f'The shape of the parameters should be {sizes}, but we got {param.shape}')
  return param


def state(
    init: Union[Callable, np.ndarray, jax.Array],
    sizes: Union[int, Sequence[int]] = None,
    batch_or_mode: Optional[Union[int, bool, bc.mixin.Mode]] = None,
):
  """
  Initialize a :math:`~.State` from a callable function or a data.
  """
  sizes = to_size(sizes)

  if callable(init):
    if sizes is None:
      raise ValueError('"varshape" cannot be None when data is a callable function.')
    sizes = list(sizes)
    if isinstance(batch_or_mode, bc.mixin.Mode):
      if batch_or_mode.has(bc.mixin.Batching):
        sizes.insert(batch_or_mode.batch_axis, batch_or_mode.batch_size)
    elif batch_or_mode in (None, False):
      pass
    elif isinstance(batch_or_mode, int):
      sizes.insert(0, batch_or_mode)
    else:
      raise ValueError(f'Unknown batch_size_or_mode: {batch_or_mode}')
    return bc.State(init(sizes))

  else:
    if sizes is not None:
      if jnp.shape(init) != sizes:
        raise ValueError(f'The shape of "data" {jnp.shape(init)} does not match with "var_shape" {sizes}')
    batch_axis = None
    batch_size = None
    if isinstance(batch_or_mode, bc.mixin.Mode):
      if batch_or_mode.has(bc.mixin.Batching):
        batch_axis = batch_or_mode.batch_axis
        batch_size = batch_or_mode.batch_size
    elif batch_or_mode in (None, False):
      pass
    elif isinstance(batch_or_mode, int):
      batch_axis = 0
      batch_size = batch_or_mode
    else:
      raise ValueError('Unknown batch_size_or_mode.')
    data = bc.State(jnp.repeat(jnp.expand_dims(init, axis=batch_axis), batch_size, axis=batch_axis))
  return data


def noise(
    noises: Optional[Union[int, float, np.ndarray, jax.Array, Callable]],
    size: Union[int, Sequence[int]],
    num_vars: int = 1,
    noise_idx: int = 0,
) -> Optional[Callable]:
  """Initialize a noise function.

  Parameters
  ----------
  noises: Any
  size: Shape
    The size of the noise.
  num_vars: int
    The number of variables.
  noise_idx: int
    The index of the current noise among all noise variables.

  Returns
  -------
  noise_func: function, None
    The noise function.

  See Also
  --------
  variable_, parameter, delay

  """
  if callable(noises):
    return noises
  elif noises is None:
    return None
  else:
    noises = parameter(noises, size, allow_none=False)
    if num_vars > 1:
      noises_ = [None] * num_vars
      noises_[noise_idx] = noises
      noises = tuple(noises_)
    return lambda *args, **kwargs: noises
