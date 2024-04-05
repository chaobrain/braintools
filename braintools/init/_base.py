from typing import Optional, Tuple

import numpy as np

__all__ = ['Initializer', 'to_size']


class Initializer(object):
  def __call__(self, *args, **kwargs):
    raise NotImplementedError


def to_size(x) -> Optional[Tuple[int]]:
  if isinstance(x, (tuple, list)):
    return tuple(x)
  if isinstance(x, (int, np.integer)):
    return (x,)
  if x is None:
    return x
  raise ValueError(f'Cannot make a size for {x}')
