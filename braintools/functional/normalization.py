from typing import Sequence, Optional

import jax
import jax.numpy as jnp
import numpy as np


__all__ = [
  'weight_standardization',
]


def weight_standardization(
    w: jax.typing.ArrayLike,
    axes: Sequence[int],
    eps: float,
    gain: Optional[jax.Array] = None
):
  """
  Scaled Weight Standardization.

  Parameters
  ----------
  w : jax.typing.ArrayLike
      The weight tensor.
  axes : Sequence[int]
      The axes to calculate the mean and variance.
  eps : float
      A small value to avoid division by zero.
  gain : Array
      The gain function, by default None.

  Returns
  -------
  jax.typing.ArrayLike
      The scaled weight tensor.

  """
  # Get Scaled WS weight HWIO;
  fan_in = np.prod(w.shape[:-1])
  mean = jnp.mean(w, axis=axes[:-1], keepdims=True)
  var = jnp.var(w, axis=axes[:-1], keepdims=True)
  weight = (w - mean) / (var * fan_in + eps) ** 0.5
  if gain is not None:
    weight = gain * weight
  return weight
