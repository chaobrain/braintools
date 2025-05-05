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

from __future__ import annotations

from typing import Sequence, Any, Callable, Tuple

import brainunit as u
import jax
import jax.tree_util as jtu
import numpy as np
from brainstate.typing import PyTree

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
    tree: PyTree[jax.typing.ArrayLike],
    x: jax.typing.ArrayLike,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Scales each element in a PyTree by a given scalar value.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree containing elements to be scaled.
        x (jax.typing.ArrayLike): The scalar value by which each element in the PyTree is multiplied.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with each element scaled by the given scalar value.
    """
    return jtu.tree.map(lambda a: a * x, tree, is_leaf=is_leaf)


def mul(
    tree: PyTree[jax.typing.ArrayLike],
    x: PyTree[jax.typing.ArrayLike] | jax.typing.ArrayLike,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Multiplies each element in a PyTree by a corresponding element in another PyTree or a scalar.

    If `x` is a scalar, each element in `tree` is multiplied by `x`.
    If `x` is a PyTree, each element in `tree` is multiplied by the corresponding element in `x`.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree containing elements to be multiplied.
        x (PyTree[jax.typing.ArrayLike] | jax.typing.ArrayLike): The PyTree or scalar value by which each element in the PyTree is multiplied.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with each element multiplied by the corresponding element in `x` or by the scalar `x`.
    """
    if isinstance(x, jax.typing.ArrayLike):
        return scale(tree, x)
    return jtu.tree.map(lambda a, b: a * b, tree, x, is_leaf=is_leaf)


def shift(
    tree1: PyTree[jax.typing.ArrayLike],
    x: jax.typing.ArrayLike,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Shifts each element in a PyTree by a given scalar value.

    Args:
        tree1 (PyTree[jax.typing.ArrayLike]): The input PyTree containing elements to be shifted.
        x (jax.typing.ArrayLike): The scalar value to add to each element in the PyTree.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with each element shifted by the given scalar value.
    """
    return jtu.tree.map(lambda a: a + x, tree1, is_leaf=is_leaf)


def add(
    tree1: PyTree[jax.typing.ArrayLike],
    tree2: PyTree[jax.typing.ArrayLike] | jax.typing.ArrayLike,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Adds corresponding elements of two PyTrees or a PyTree and a scalar.

    If `tree2` is a scalar, each element in `tree1` is incremented by `tree2`.
    If `tree2` is a PyTree, each element in `tree1` is added to the corresponding element in `tree2`.

    Args:
        tree1 (PyTree[jax.typing.ArrayLike]): The first input PyTree containing elements to be added.
        tree2 (PyTree[jax.typing.ArrayLike] | jax.typing.ArrayLike): The second PyTree or scalar value to add to each element in `tree1`.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with each element being the sum of the corresponding elements in `tree1` and `tree2`, or `tree1` and the scalar `tree2`.
    """
    if isinstance(tree2, jax.Array):
        return shift(tree1, tree2)
    return jtu.tree.map(lambda a, b: a + b, tree1, tree2, is_leaf=is_leaf)


def sub(
    tree1: PyTree[jax.typing.ArrayLike],
    tree2: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Subtracts corresponding elements of two PyTrees.

    Args:
        tree1 (PyTree[jax.typing.ArrayLike]): The first input PyTree containing elements to be subtracted from.
        tree2 (PyTree[jax.typing.ArrayLike]): The second PyTree containing elements to subtract from the corresponding elements in `tree1`.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with each element being the difference of the corresponding elements in `tree1` and `tree2`.
    """
    return jtu.tree.map(lambda a, b: a - b, tree1, tree2, is_leaf=is_leaf)


def dot(
    a: PyTree,
    b: PyTree,
    is_leaf: Callable[[Any], bool] | None = None
) -> jax.Array:
    """
    Computes the dot product of two PyTrees.

    This function multiplies corresponding elements of two PyTrees and sums the results to produce a single scalar value.

    Args:
        a (PyTree): The first input PyTree containing elements to be multiplied.
        b (PyTree): The second input PyTree containing elements to be multiplied with the corresponding elements in `a`.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        jax.Array: A scalar value representing the dot product of the two input PyTrees.
    """
    return jtu.tree.reduce(
        u.math.add,
        jtu.tree.map(u.math.sum, jax.tree.map(jax.lax.mul, a, b, is_leaf=is_leaf), is_leaf=is_leaf),
        is_leaf=is_leaf
    )


def sum(
    tree: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> jax.Array:
    """
    Computes the sum of all elements in a PyTree.

    This function traverses the input PyTree, summing all elements to produce a single scalar value.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree containing elements to be summed.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        jax.Array: A scalar value representing the sum of all elements in the input PyTree.
    """
    return jtu.tree.reduce(u.math.add, jtu.tree.map(u.math.sum, tree, is_leaf=is_leaf), is_leaf=is_leaf)


def squared_norm(
    tree: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> jax.Array:
    """
    Computes the squared norm of all elements in a PyTree.

    This function traverses the input PyTree, computing the squared norm of each element
    and summing the results to produce a single scalar value.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree containing elements for which the squared norm is computed.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        jax.Array: A scalar value representing the squared norm of all elements in the input PyTree.
    """
    return jtu.tree.reduce(
        u.math.add,
        jtu.tree.map(lambda x: u.math.einsum('...,...->', x, x), tree, is_leaf=is_leaf),
        is_leaf=is_leaf
    )


def concat(
    trees: Sequence[PyTree[jax.typing.ArrayLike]],
    axis: int = 0,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Concatenates a sequence of PyTrees along a specified axis.

    This function takes multiple PyTrees and concatenates their corresponding elements
    along the specified axis, resulting in a single PyTree.

    Args:
        trees (Sequence[PyTree[jax.typing.ArrayLike]]): A sequence of PyTrees to be concatenated.
        axis (int, optional): The axis along which the PyTrees will be concatenated. Defaults to 0.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with elements concatenated along the specified axis.
    """
    return jtu.tree.map(lambda *args: u.math.concatenate(args, axis=axis), *trees, is_leaf=is_leaf)


def split(
    tree: PyTree[jax.Array],
    sizes: Tuple[int],
    is_leaf: Callable[[Any], bool] | None = None
) -> Tuple[PyTree[jax.Array], ...]:
    """
    Splits a PyTree into multiple sub-PyTrees based on specified sizes.

    This function divides the input PyTree into a sequence of sub-PyTrees,
    where each sub-PyTree corresponds to a specified size in the `sizes` tuple.

    Args:
        tree (PyTree[jax.Array]): The input PyTree to be split.
        sizes (Tuple[int]): A tuple of integers specifying the sizes of each split.
            The sum of the sizes should be less than or equal to the size of the PyTree.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        Tuple[PyTree[jax.Array], ...]: A tuple of sub-PyTrees, each corresponding to a specified size in `sizes`.
    """
    idx = 0
    result: list[PyTree[jax.Array]] = []
    for s in sizes:
        result.append(jtu.tree.map(lambda x: x[idx: idx + s], tree, is_leaf=is_leaf))
        idx += s
    result.append(jtu.tree.map(lambda x: x[idx:], tree, is_leaf=is_leaf))
    return tuple(result)


def idx(
    tree: PyTree[jax.typing.ArrayLike],
    idx,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Extracts elements from a PyTree at specified indices.

    This function traverses the input PyTree and extracts elements at the specified
    indices from each leaf node.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree from which elements are to be extracted.
        idx: The indices at which elements are to be extracted. This can be an integer, slice, or array-like object.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with elements extracted at the specified indices from each leaf node.
    """
    return jtu.tree.map(lambda x: x[idx], tree, is_leaf=is_leaf)


def expand(
    tree: PyTree[jax.typing.ArrayLike],
    axis,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Expands the dimensions of each element in a PyTree along a specified axis.

    This function traverses the input PyTree and applies a dimension expansion
    to each leaf node along the specified axis.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree containing elements to be expanded.
        axis: The axis along which to expand the dimensions of each element.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with each element's dimensions expanded along the specified axis.
    """
    return jtu.tree.map(lambda x: u.math.expand_dims(x, axis), tree, is_leaf=is_leaf)


def take(
    tree: PyTree[jax.typing.ArrayLike],
    idx,
    axis: int,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    """
    Extracts elements from a PyTree along a specified axis using given indices.

    This function traverses the input PyTree and extracts elements from each leaf node
    along the specified axis using the provided indices.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree from which elements are to be extracted.
        idx: The indices or slice at which elements are to be extracted. This can be an integer, slice, or array-like object.
        axis (int): The axis along which to extract elements.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree: A new PyTree with elements extracted along the specified axis from each leaf node.
    """
    def take_(x):
        indices = idx
        if isinstance(indices, slice):
            slices = [slice(None)] * x.ndim
            slices[axis] = idx
            return x[tuple(slices)]
        return u.math.take(x, indices, axis)

    return jtu.tree.map(take_, tree, is_leaf=is_leaf)


def as_numpy(
    tree: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree[np.ndarray]:
    """
    Converts all elements in a PyTree to NumPy arrays.

    This function traverses the input PyTree and converts each leaf node
    to a NumPy array using `np.asarray`.

    Args:
        tree (PyTree[jax.typing.ArrayLike]): The input PyTree containing elements to be converted.
        is_leaf (Callable[[Any], bool] | None, optional): A function that determines whether a node is a leaf.
            If None, all non-sequence nodes are considered leaves.

    Returns:
        PyTree[np.ndarray]: A new PyTree with each element converted to a NumPy array.
    """
    return jtu.tree.map(lambda x: np.asarray(x), tree, is_leaf=is_leaf)
