# -*- coding: utf-8 -*-

import braincore as bc
import jax.numpy as jnp

from ._base import Initializer

__all__ = [
  'ZeroInit',
  'Constant',
  'Identity',
]


class ZeroInit(Initializer):
  """Zero initializer.

  Initialize the weights with zeros.
  """

  def __init__(self, dtype=None):
    super(ZeroInit, self).__init__()
    self.dtype = dtype or bc.environ.dftype()

  def __call__(self, *shape):
    return jnp.zeros(shape, dtype=self.dtype)

  def __repr__(self):
    return f"{self.__class__.__name__}(dtype={self.dtype})"


class Constant(Initializer):
  """Constant initializer.

  Initialize the weights with the given values.

  Parameters
  ----------
  value : float, int, bm.ndarray
    The value to specify.
  """

  def __init__(self, value=1., dtype=None):
    super(Constant, self).__init__()
    self.dtype = dtype or bc.environ.dftype()
    self.value = jnp.asarray(value, dtype=self.dtype)

  def __call__(self, *shape):
    return jnp.full(shape, self.value, dtype=self.dtype)

  def __repr__(self):
    return f'{self.__class__.__name__}(value={self.value}, dtype={self.dtype})'


class Identity(Initializer):
  """Returns the identity matrix.

  This initializer was proposed in (Le, et al., 2015) [1]_.

  Parameters
  ----------
  value : float
    The optional scaling factor.

  Returns
  -------
  shape: tuple of int
    The weight shape/size.

  References
  ----------
  .. [1] Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton. "A simple way to
         initialize recurrent networks of rectified linear units." arXiv preprint
         arXiv:1504.00941 (2015).
  """

  def __init__(self, value=1., dtype=None):
    super(Identity, self).__init__()
    self.dtype = dtype or bc.environ.dftype()
    self.value = jnp.asarray(value, dtype=self.dtype)

  def __call__(self, *shape):
    if isinstance(shape, (tuple, list)):
      if len(shape) > 2:
        raise ValueError(f'Only support initialize 2D weights for {self.__class__.__name__}.')
    r = jnp.eye(*shape, dtype=self.dtype)
    r = jnp.fill_diagonal(r, self.value)
    return r

  def __repr__(self):
    return f'{self.__class__.__name__}(value={self.value}, dtype={self.dtype})'
