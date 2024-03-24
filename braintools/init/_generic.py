# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional, Sequence

import braincore as bc
import jax
import jax.numpy as jnp
import numpy as np
from brainpy.tools import to_size

__all__ = [
  'parameter',
  'state',
  'noise',
]


def _is_scalar(x):
  return isinstance(x, (float, int, bool, complex))


def parameter(
    param: Union[Callable, np.ndarray, jax.Array, float, int, bool],
    sizes: Union[int, Sequence[int]],
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
    - If it is an instance of :py:class:`brainonline.init.Initializer``, the ``f(size)`` will be returned.
    - If it is a tensor, then this function check whether ``tensor.shape`` is equal to the given ``size``.
  sizes: int, sequence of int
    The shape of the parameter.
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
  sizes = to_size(sizes)
  if allow_scalar and _is_scalar(param):
    return param

  if callable(param):
    return param(*sizes)
  elif isinstance(param, np.ndarray):
    param = jnp.asarray(param)
  elif isinstance(param, bc.State):
    param = param
  else:
    param = param

  if allow_scalar:
    if param.shape == () or param.shape == (1,):
      return param
  if param.shape != sizes:
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
  sizes = list(to_size(sizes))

  if callable(init):
    if sizes is None:
      raise ValueError('"varshape" cannot be None when data is a callable function.')
    if isinstance(batch_or_mode, bc.mixin.Mode):
      if batch_or_mode.has(bc.mixin.Batching):
        sizes.insert(batch_or_mode.batch_axis, batch_or_mode.batch_size)
    elif batch_or_mode in (None, False):
      pass
    elif isinstance(batch_or_mode, int):
      sizes.insert(0, batch_or_mode)
    else:
      raise ValueError(f'Unknown batch_size_or_mode: {batch_or_mode}')
    data = bc.State(init(*sizes))

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
