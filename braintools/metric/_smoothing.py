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

"""Smoothing functions."""

import brainstate
import brainunit as u
import jax.numpy as jnp

__all__ = ['smooth_labels']


def smooth_labels(
    labels: brainstate.typing.ArrayLike,
    alpha: float,
) -> jnp.ndarray:
    """Apply label smoothing.

    Label smoothing is often used in combination with a cross-entropy loss.
    Smoothed labels favour small logit gaps, and it has been shown that this can
    provide better model calibration by preventing overconfident predictions.

    References:
      [Müller et al, 2019](https://arxiv.org/pdf/1906.02629.pdf)

    Args:
      labels: One hot labels to be smoothed.
      alpha: The smoothing factor.

    Returns:
      a smoothed version of the one hot input labels.
    """
    assert u.math.is_float(labels), f'labels should be a float.'
    num_categories = labels.shape[-1]
    return (1.0 - alpha) * labels + alpha / num_categories
