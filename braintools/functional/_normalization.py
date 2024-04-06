# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
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
