# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Optional

import brainstate
import brainunit as u
import jax

from braintools._misc import set_module_as

__all__ = [
    'cosine_similarity',
]


@set_module_as('braintools.metric')
def cosine_similarity(
    X: brainstate.typing.ArrayLike,
    Y: Optional[brainstate.typing.ArrayLike] = None,
    eps: float = 1e-8
) -> jax.Array:
    """
    Compute cosine similarity between samples in X and Y.

    Cosine similarity is defined as the dot product of two vectors divided by
    the product of their magnitudes (L2 norms).

    Args:
        X: Array of shape (n_samples_X, n_features)
        Y: Array of shape (n_samples_Y, n_features), optional.
           If None, compute cosine similarity between samples in X.

    Returns:
        Cosine similarity matrix of shape (n_samples_X, n_samples_Y).
        If Y is None, returns shape (n_samples_X, n_samples_X).

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        >>> cosine_sim = cosine_similarity(X)
        >>> print(cosine_sim)
        [[1.         0.         0.70710677]
         [0.         1.         0.70710677]
         [0.70710677 0.70710677 1.        ]]
    """
    # If Y is not provided, compute similarity within X
    if Y is None:
        Y = X

    # Compute dot products between all pairs
    dot_products = u.math.dot(X, Y.T)

    # Compute L2 norms for each sample
    X_norms = u.linalg.norm(X, axis=1, keepdims=True)
    Y_norms = u.linalg.norm(Y, axis=1, keepdims=True)

    # Compute the product of norms for each pair
    norm_products = u.math.dot(X_norms, Y_norms.T)

    # Compute cosine similarity
    # Add small epsilon to avoid division by zero
    cosine_sim = dot_products / (norm_products + eps)

    return u.math.nan_to_num(cosine_sim)
