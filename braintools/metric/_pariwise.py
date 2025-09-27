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
    r"""Compute cosine similarity between samples in X and Y.

    Cosine similarity measures the cosine of the angle between two vectors,
    providing a metric of similarity that is independent of vector magnitude.
    It ranges from -1 (opposite directions) to 1 (same direction).

    The cosine similarity is defined as:

    .. math::

        \text{similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| ||\mathbf{B}||}

    where :math:`\mathbf{A}` and :math:`\mathbf{B}` are vectors, and :math:`||\cdot||`
    denotes the L2 norm.

    Parameters
    ----------
    X : brainstate.typing.ArrayLike
        Input array with shape ``(n_samples_X, n_features)``. Each row
        represents a sample/vector.
    Y : brainstate.typing.ArrayLike, optional
        Input array with shape ``(n_samples_Y, n_features)``. If None,
        computes pairwise similarities within X.
    eps : float, default=1e-8
        Small epsilon value to avoid division by zero when computing norms.

    Returns
    -------
    jax.Array
        Cosine similarity matrix:
        
        - If Y is provided: shape ``(n_samples_X, n_samples_Y)``
        - If Y is None: shape ``(n_samples_X, n_samples_X)``
        
        Element (i,j) represents the cosine similarity between sample i
        from X and sample j from Y (or X if Y is None).

    Examples
    --------
    Basic cosine similarity computation:

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> # Create sample vectors
        >>> X = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        >>> # Compute pairwise similarities
        >>> sim_matrix = braintools.metric.cosine_similarity(X)
        >>> print(sim_matrix)
        [[1.         0.         0.70710677]
         [0.         1.         0.70710677]
         [0.70710677 0.70710677 1.        ]]

    Cross-similarity between different sets:

    .. code-block:: python

        >>> # Compare with different set
        >>> Y = jnp.array([[1, 1, 1], [0, 0, 1]])
        >>> cross_sim = braintools.metric.cosine_similarity(X, Y)
        >>> print(cross_sim.shape)
        (3, 2)
        >>> print(cross_sim)
        [[0.57735026 0.        ]
         [0.57735026 0.        ]
         [0.8164966  0.        ]]

    Analyzing vector relationships:

    .. code-block:: python

        >>> # Identical vectors have similarity 1
        >>> identical = jnp.array([[1, 2, 3], [1, 2, 3]])
        >>> sim_identical = braintools.metric.cosine_similarity(identical)
        >>> print(f"Identical vectors similarity: {sim_identical[0, 1]:.6f}")

        >>> # Orthogonal vectors have similarity 0
        >>> orthogonal = jnp.array([[1, 0], [0, 1]])
        >>> sim_orthogonal = braintools.metric.cosine_similarity(orthogonal)
        >>> print(f"Orthogonal vectors similarity: {sim_orthogonal[0, 1]:.6f}")

        >>> # Opposite vectors have similarity -1
        >>> opposite = jnp.array([[1, 0], [-1, 0]])
        >>> sim_opposite = braintools.metric.cosine_similarity(opposite)
        >>> print(f"Opposite vectors similarity: {sim_opposite[0, 1]:.6f}")

    Working with high-dimensional data:

    .. code-block:: python

        >>> # Document embeddings example
        >>> doc_embeddings = jnp.random.normal(size=(100, 300))  # 100 docs, 300-dim
        >>> doc_similarities = braintools.metric.cosine_similarity(doc_embeddings)
        >>> print(f"Document similarity matrix shape: {doc_similarities.shape}")
        >>> # Find most similar document to the first one
        >>> most_similar_idx = jnp.argmax(doc_similarities[0, 1:]) + 1
        >>> print(f"Most similar to doc 0: doc {most_similar_idx}")
        >>> print(f"Similarity score: {doc_similarities[0, most_similar_idx]:.3f}")

    Notes
    -----
    The function handles zero vectors by adding a small epsilon to avoid
    division by zero. NaN values in the result are replaced with 0.

    Cosine similarity properties:

    - **Range**: Values are bounded between -1 and 1
    - **Interpretation**:
      - 1 = identical direction (perfect positive correlation)
      - 0 = orthogonal vectors (no correlation)
      - -1 = opposite direction (perfect negative correlation)
    - **Scale invariant**: Only depends on vector direction, not magnitude
    - **Symmetric**: cosine_similarity(A, B) = cosine_similarity(B, A)

    Common applications:

    - Document similarity in text mining
    - Recommendation systems (user/item similarity)
    - Image feature comparison
    - Neural network activation analysis
    - Embedding space analysis

    See Also
    --------
    braintools.metric.weighted_correlation : Weighted Pearson correlation
    jax.numpy.dot : Dot product computation
    jax.numpy.linalg.norm : Vector norm computation

    References
    ----------
    .. [1] Salton, Gerard, and Michael J. McGill. "Introduction to modern
           information retrieval." (1986).
    .. [2] Manning, Christopher D., Prabhakar Raghavan, and Hinrich Schütze.
           "Introduction to information retrieval." Cambridge university press, 2008.
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
