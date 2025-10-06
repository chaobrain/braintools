# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Sequence, Any, Callable, Tuple

import brainstate
import brainunit as u
import jax
import numpy as np
from brainstate.typing import PyTree, ArrayLike

__all__ = [
    'scale',
    'mul',
    'shift',
    'add',
    'sub',
    'dot',
    'sum',
    'squared_norm',
    'concat',
    'split',
    'idx',
    'expand',
    'take',
    'as_numpy',
]


def scale(
    tree: PyTree[ArrayLike],
    x: ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Multiply every leaf array in a PyTree by a value.

    Parameters
    ----------
    tree : PyTree of array_like
        Input PyTree whose leaves are multiplied.
    x : array_like
        Value broadcast and multiplied with each leaf of `tree`.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity` so unit-quantities are considered leaves.

    Returns
    -------
    PyTree
        PyTree with each leaf multiplied by `x` (with broadcasting).

    See Also
    --------
    mul : Elementwise multiplication with a PyTree or value.
    add : Elementwise addition with a PyTree or value.

    Notes
    -----
    - Broadcasting follows the semantics of the underlying array type
      (e.g., JAX/NumPy). The structure of `tree` is preserved.
    """
    return jax.tree.map(lambda a: a * x, tree, is_leaf=is_leaf)


def mul(
    tree: PyTree[ArrayLike],
    x: PyTree[ArrayLike] | ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Elementwise multiplication of a PyTree with a value or another PyTree.

    Parameters
    ----------
    tree : PyTree of array_like
        Left operand; PyTree whose leaves are multiplied.
    x : PyTree of array_like or array_like
        Right operand. If a PyTree, its structure must match `tree` and
        corresponding leaves are multiplied elementwise. If a value
        (scalar/array_like), it is broadcast to each leaf of `tree`.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree with elementwise products.

    See Also
    --------
    scale : Multiply every leaf by a single value.
    add : Elementwise addition with a PyTree or value.
    sub : Elementwise subtraction of PyTrees.

    Notes
    -----
    - When `x` is a PyTree, both `tree` and `x` must have identical PyTree
      structures. Broadcasting applies per-leaf when `x` is array_like.
    """
    if isinstance(x, brainstate.typing.ArrayLike):
        return scale(tree, x)
    return jax.tree.map(lambda a, b: a * b, tree, x, is_leaf=is_leaf)


def shift(
    tree1: PyTree[ArrayLike],
    x: ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Add a value to every leaf array in a PyTree.

    Parameters
    ----------
    tree1 : PyTree of array_like
        Input PyTree.
    x : array_like
        Value broadcast and added to each leaf of `tree1`.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree with each leaf shifted by `x`.

    See Also
    --------
    add : Elementwise addition with a PyTree or value.
    sub : Elementwise subtraction of PyTrees.
    """
    return jax.tree.map(lambda a: a + x, tree1, is_leaf=is_leaf)


def add(
    tree1: PyTree[ArrayLike],
    tree2: PyTree[ArrayLike] | ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Elementwise addition of PyTrees or a PyTree and a value.

    Parameters
    ----------
    tree1 : PyTree of array_like
        Left operand.
    tree2 : PyTree of array_like or array_like
        Right operand. If a PyTree, its structure must match `tree1` and
        corresponding leaves are added elementwise. If a value
        (scalar/array_like), it is broadcast to each leaf of `tree1`.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree with elementwise sums.

    See Also
    --------
    sub : Elementwise subtraction of PyTrees.
    mul : Elementwise multiplication.
    shift : Add a single value to every leaf.
    """
    if isinstance(tree2, brainstate.typing.ArrayLike):
        return shift(tree1, tree2)
    return jax.tree.map(lambda a, b: a + b, tree1, tree2, is_leaf=is_leaf)


def sub(
    tree1: PyTree[ArrayLike],
    tree2: PyTree[ArrayLike],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Elementwise subtraction of two PyTrees.

    Parameters
    ----------
    tree1 : PyTree of array_like
        Minuend PyTree.
    tree2 : PyTree of array_like
        Subtrahend PyTree. Must share the same PyTree structure as `tree1`.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree where each leaf is `tree1 - tree2` elementwise.

    See Also
    --------
    add : Elementwise addition of PyTrees.
    mul : Elementwise multiplication.
    """
    return jax.tree.map(lambda a, b: a - b, tree1, tree2, is_leaf=is_leaf)


def dot(
    a: PyTree,
    b: PyTree,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> jax.Array:
    """
    Inner product over all leaves of two PyTrees.

    For each pair of corresponding leaves, compute the elementwise product,
    reduce with a full sum over all axes, and then sum across leaves.

    Parameters
    ----------
    a : PyTree of array_like
        First operand.
    b : PyTree of array_like
        Second operand with the same PyTree structure as `a`.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    jax.Array
        Scalar representing the inner product across the entire PyTree.

    See Also
    --------
    sum : Sum of all elements in a PyTree.
    squared_norm : Sum of squares of all elements (i.e., `dot(x, x)`).

    Notes
    -----
    - Leaf arrays must be broadcast-compatible for multiplication.
    - When leaves carry units (brainunit quantities), units propagate through
      the multiplications and sums according to quantity rules.
    """
    return jax.tree.reduce(
        u.math.add,
        jax.tree.map(u.math.sum, jax.tree.map(jax.lax.mul, a, b, is_leaf=is_leaf), is_leaf=is_leaf),
        is_leaf=is_leaf
    )


def sum(
    tree: PyTree[ArrayLike],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> jax.Array:
    """
    Sum all elements across every leaf in a PyTree.

    Parameters
    ----------
    tree : PyTree of array_like
        Input PyTree whose elements are reduced.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    jax.Array
        Scalar sum across all leaves and their elements.

    See Also
    --------
    dot : Inner product across leaves.
    squared_norm : Sum of squares across leaves.
    """
    return jax.tree.reduce(u.math.add, jax.tree.map(u.math.sum, tree, is_leaf=is_leaf), is_leaf=is_leaf)


def squared_norm(
    tree: PyTree[ArrayLike],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> jax.Array:
    """
    Sum of squares of all elements in a PyTree.

    Parameters
    ----------
    tree : PyTree of array_like
        Input PyTree.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    jax.Array
        Scalar representing ``sum_i sum(x_i**2)`` across all leaves `x_i`.

    See Also
    --------
    dot : Inner product across leaves.

    Notes
    -----
    Equivalent to ``dot(tree, tree)``.
    """
    return jax.tree.reduce(
        u.math.add,
        jax.tree.map(lambda x: u.math.einsum('...,...->', x, x), tree, is_leaf=is_leaf),
        is_leaf=is_leaf
    )


def concat(
    trees: Sequence[PyTree[ArrayLike]],
    axis: int = 0,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Concatenate corresponding leaves from a sequence of PyTrees.

    Parameters
    ----------
    trees : sequence of PyTree of array_like
        PyTrees with identical structure whose leaves are concatenated.
    axis : int, default: 0
        Axis along which to concatenate leaves.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree with each leaf given by ``concatenate([t[i] for t in trees], axis=axis)``.

    Notes
    -----
    All PyTrees in `trees` must share the same PyTree structure, and their
    corresponding leaves must be compatible for concatenation along `axis`.
    """
    return jax.tree.map(
        lambda *args: u.math.concatenate(args, axis=axis),
        *trees,
        is_leaf=is_leaf
    )


def split(
    tree: PyTree[jax.Array],
    sizes: Tuple[int],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> Tuple[PyTree[jax.Array], ...]:
    """
    Split each leaf of a PyTree along axis 0 according to sizes.

    Parameters
    ----------
    tree : PyTree of jax.Array
        Input PyTree. Each leaf is sliced along its first axis.
    sizes : tuple of int
        Sizes for consecutive splits along axis 0. The remainder (if any)
        after ``sum(sizes)`` is returned as a final chunk.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    tuple of PyTree
        Tuple of length ``len(sizes) + 1`` where each element is a PyTree
        containing the corresponding slice. The last PyTree may be empty if
        ``sum(sizes)`` equals the size along axis 0.

    Notes
    -----
    This function operates on axis 0 only.
    """
    idx = 0
    result: list[PyTree[jax.Array]] = []
    for s in sizes:
        result.append(jax.tree.map(lambda x: x[idx: idx + s], tree, is_leaf=is_leaf))
        idx += s
    result.append(jax.tree.map(lambda x: x[idx:], tree, is_leaf=is_leaf))
    return tuple(result)


def idx(
    tree: PyTree[ArrayLike],
    idx,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Index every leaf of a PyTree.

    Parameters
    ----------
    tree : PyTree of array_like
        Input PyTree.
    idx : int, slice or array_like
        Indices used to index each leaf ``x`` as ``x[idx]``.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree with leaves indexed by `idx`.
    """
    return jax.tree.map(lambda x: x[idx], tree, is_leaf=is_leaf)


def expand(
    tree: PyTree[ArrayLike],
    axis,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Insert a length-1 axis into every leaf of a PyTree.

    Parameters
    ----------
    tree : PyTree of array_like
        Input PyTree.
    axis : int
        Position in the expanded shape where the new axis is placed.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree with ``expand_dims(x, axis)`` applied to each leaf ``x``.
    """
    return jax.tree.map(lambda x: u.math.expand_dims(x, axis), tree, is_leaf=is_leaf)


def take(
    tree: PyTree[ArrayLike],
    idx,
    axis: int,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree:
    """
    Take elements from every leaf of a PyTree along a given axis.

    Parameters
    ----------
    tree : PyTree of array_like
        Input PyTree.
    idx : int, slice or array_like
        Indices used for selection. If a slice, it is applied by standard
        indexing; otherwise ``u.math.take`` is used per leaf.
    axis : int
        Axis along which to take values.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree
        PyTree with elements selected along `axis`.
    """

    def take_(x):
        indices = idx
        if isinstance(indices, slice):
            slices = [slice(None)] * x.ndim
            slices[axis] = idx
            return x[tuple(slices)]
        return u.math.take(x, indices, axis)

    return jax.tree.map(take_, tree, is_leaf=is_leaf)


def as_numpy(
    tree: PyTree[ArrayLike],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> PyTree[np.ndarray]:
    """
    Convert all leaves of a PyTree to NumPy arrays.

    Parameters
    ----------
    tree : PyTree of array_like
        Input PyTree whose leaves will be converted.
    is_leaf : callable, optional
        Predicate to treat a node as a leaf during traversal. Defaults to
        `u.math.is_quantity`.

    Returns
    -------
    PyTree of numpy.ndarray
        PyTree with ``np.asarray`` applied to each leaf.

    Notes
    -----
    This performs a best-effort conversion using ``np.asarray``; for JAX
    arrays this typically results in host NumPy arrays.
    """
    return jax.tree.map(lambda x: np.asarray(x), tree, is_leaf=is_leaf)
