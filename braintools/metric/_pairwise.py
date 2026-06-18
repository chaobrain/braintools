# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    'pairwise_cosine_similarity',
    'pairwise_cosine_distance',
]


def _safe_row_norm(a: jax.Array) -> jax.Array:
    """L2 norm over the last axis that is gradient-safe at the zero vector.

    ``jnp.linalg.norm`` (and ``jnp.sqrt``) produce a ``NaN`` gradient at the
    origin because ``d/dx sqrt(sum x**2) = x / norm`` is ``0/0`` there. The
    double-``where`` construction below returns a norm of ``0`` for zero rows
    while keeping the gradient finite.
    """
    sq = jnp.sum(a * a, axis=-1, keepdims=True)
    is_zero = sq == 0.0
    safe_sq = jnp.where(is_zero, 1.0, sq)
    return jnp.where(is_zero, 0.0, jnp.sqrt(safe_sq))


@set_module_as('braintools.metric')
def pairwise_cosine_similarity(
    X: brainstate.typing.ArrayLike,
    Y: Optional[brainstate.typing.ArrayLike] = None,
    eps: float = 1e-8
) -> jax.Array:
    r"""Compute the pairwise cosine similarity matrix between samples in ``X`` and ``Y``.

    Cosine similarity measures the cosine of the angle between two vectors,
    providing a similarity metric that is independent of vector magnitude.
    It ranges from ``-1`` (opposite directions) to ``1`` (same direction).

    The cosine similarity between two vectors :math:`\mathbf{a}` and
    :math:`\mathbf{b}` is defined as:

    .. math::

        \text{similarity}(\mathbf{a}, \mathbf{b}) =
        \frac{\mathbf{a} \cdot \mathbf{b}}{\lVert \mathbf{a} \rVert\, \lVert \mathbf{b} \rVert}

    where :math:`\lVert \cdot \rVert` denotes the L2 norm.

    This function returns the **full pairwise matrix** of similarities (every row
    of ``X`` against every row of ``Y``). For an element-wise similarity between
    paired samples, see :func:`braintools.metric.cosine_similarity`.

    Parameters
    ----------
    X : brainstate.typing.ArrayLike
        Input array with shape ``(n_samples_X, n_features)``. Each row is a
        sample/vector. ``brainunit.Quantity`` inputs are accepted; because cosine
        similarity is scale invariant, the result is always dimensionless.
    Y : brainstate.typing.ArrayLike, optional
        Input array with shape ``(n_samples_Y, n_features)``. If ``None``,
        computes pairwise similarities within ``X``.
    eps : float, default=1e-8
        Lower bound applied to **each row norm** (not their product) to avoid
        division by zero. Only norms below ``eps`` are affected, so similarities
        between small but non-zero vectors are computed exactly; pairs that
        involve a zero vector still yield a similarity of ``0``.

    Returns
    -------
    jax.Array
        Cosine similarity matrix:

        - If ``Y`` is provided: shape ``(n_samples_X, n_samples_Y)``.
        - If ``Y`` is ``None``: shape ``(n_samples_X, n_samples_X)``.

        Element ``(i, j)`` is the cosine similarity between sample ``i`` of ``X``
        and sample ``j`` of ``Y`` (or ``X`` if ``Y`` is ``None``).

    See Also
    --------
    cosine_similarity : Element-wise cosine similarity between paired samples.
    cosine_distance : Element-wise cosine distance (``1 - similarity``).

    Notes
    -----
    Each row norm is floored at ``eps`` (rather than adding ``eps`` to the norm
    product), so that the value **and** the gradient remain finite for zero
    vectors while non-zero vectors -- including very small ones -- are
    unaffected. This avoids both the NaN-gradient hazard of the naive
    ``dot / (norm_product + eps)`` formulation and the magnitude-coupling bug of
    flooring the *product* of the two norms.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> X = jnp.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
        >>> sim_matrix = braintools.metric.pairwise_cosine_similarity(X)
        >>> print(sim_matrix)
        [[1.         0.         0.70710677]
         [0.         1.         0.70710677]
         [0.70710677 0.70710677 1.0000001 ]]

        >>> Y = jnp.array([[1., 1., 1.], [0., 0., 1.]])
        >>> braintools.metric.pairwise_cosine_similarity(X, Y).shape
        (3, 2)
    """
    # Cosine similarity is scale/unit invariant, so we operate on raw magnitudes.
    # This also makes the function robust to ``brainunit.Quantity`` inputs (the
    # unit-squared terms in numerator and denominator would otherwise cancel but
    # break the ``eps`` floor, which is dimensionless).
    X = jnp.asarray(u.get_magnitude(X))
    Y = X if Y is None else jnp.asarray(u.get_magnitude(Y))

    # Pairwise dot products: shape (n_samples_X, n_samples_Y)
    dot_products = X @ Y.T

    # L2 norms for each sample (gradient-safe at zero vectors). Floor *each row
    # norm* at ``eps`` rather than their product: this engages only for genuine
    # near-zero vectors, so the (scale-invariant) cosine stays correct for small
    # but non-zero vectors, while the value and gradient remain finite at the
    # origin (a zero row contributes a zero numerator, hence a similarity of 0).
    X_norms = jnp.maximum(_safe_row_norm(X), eps)
    Y_norms = jnp.maximum(_safe_row_norm(Y), eps)
    norm_products = X_norms @ Y_norms.T

    return dot_products / norm_products


@set_module_as('braintools.metric')
def pairwise_cosine_distance(
    X: brainstate.typing.ArrayLike,
    Y: Optional[brainstate.typing.ArrayLike] = None,
    eps: float = 1e-8
) -> jax.Array:
    r"""Compute the pairwise cosine distance matrix between samples in ``X`` and ``Y``.

    The cosine distance is defined as ``1 - cosine_similarity`` and therefore
    ranges from ``0`` (identical direction) to ``2`` (opposite direction).

    Parameters
    ----------
    X : brainstate.typing.ArrayLike
        Input array with shape ``(n_samples_X, n_features)``. ``brainunit.Quantity``
        inputs are accepted; the result is always dimensionless.
    Y : brainstate.typing.ArrayLike, optional
        Input array with shape ``(n_samples_Y, n_features)``. If ``None``,
        computes pairwise distances within ``X``.
    eps : float, default=1e-8
        Lower bound applied to each row norm to avoid division by zero. Pairs
        involving a zero vector therefore yield a distance of ``1``.

    Returns
    -------
    jax.Array
        Cosine distance matrix:

        - If ``Y`` is provided: shape ``(n_samples_X, n_samples_Y)``.
        - If ``Y`` is ``None``: shape ``(n_samples_X, n_samples_X)``.

    See Also
    --------
    pairwise_cosine_similarity : The underlying pairwise similarity matrix.
    cosine_distance : Element-wise cosine distance between paired samples.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> X = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
        >>> braintools.metric.pairwise_cosine_distance(X).shape
        (3, 3)
    """
    return 1.0 - pairwise_cosine_similarity(X, Y, eps=eps)
