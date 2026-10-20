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
from brainstate.typing import ArrayLike

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
    tree: brainstate.typing.PyTree[ArrayLike],
    x: ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> out = braintools.tree.scale(tree, 2.0)
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [2.0, 4.0], 'b': [6.0]}
    """
    return jax.tree.map(lambda a: a * x, tree, is_leaf=is_leaf)


def mul(
    tree: brainstate.typing.PyTree[ArrayLike],
    x: brainstate.typing.PyTree[ArrayLike] | ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> other = {'a': jnp.array([10.0, 20.0]), 'b': jnp.array([30.0])}
        >>> out = braintools.tree.mul(tree, other)
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [10.0, 40.0], 'b': [90.0]}
        >>>
        >>> out = braintools.tree.mul(tree, 3.0)
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [3.0, 6.0], 'b': [9.0]}
    """
    if isinstance(x, brainstate.typing.ArrayLike):
        return scale(tree, x)
    return jax.tree.map(lambda a, b: a * b, tree, x, is_leaf=is_leaf)


def shift(
    tree1: brainstate.typing.PyTree[ArrayLike],
    x: ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> out = braintools.tree.shift(tree, 1.0)
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [2.0, 3.0], 'b': [4.0]}
    """
    return jax.tree.map(lambda a: a + x, tree1, is_leaf=is_leaf)


def add(
    tree1: brainstate.typing.PyTree[ArrayLike],
    tree2: brainstate.typing.PyTree[ArrayLike] | ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> other = {'a': jnp.array([10.0, 20.0]), 'b': jnp.array([30.0])}
        >>> out = braintools.tree.add(tree, other)
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [11.0, 22.0], 'b': [33.0]}
        >>>
        >>> out = braintools.tree.add(tree, 1.0)
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [2.0, 3.0], 'b': [4.0]}
    """
    if isinstance(tree2, brainstate.typing.ArrayLike):
        return shift(tree1, tree2)
    return jax.tree.map(lambda a, b: a + b, tree1, tree2, is_leaf=is_leaf)


def sub(
    tree1: brainstate.typing.PyTree[ArrayLike],
    tree2: brainstate.typing.PyTree[ArrayLike],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> other = {'a': jnp.array([10.0, 20.0]), 'b': jnp.array([30.0])}
        >>> out = braintools.tree.sub(other, tree)
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [9.0, 18.0], 'b': [27.0]}
    """
    return jax.tree.map(lambda a, b: a - b, tree1, tree2, is_leaf=is_leaf)


def dot(
    a: brainstate.typing.PyTree,
    b: brainstate.typing.PyTree,
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> other = {'a': jnp.array([10.0, 20.0]), 'b': jnp.array([30.0])}
        >>> float(braintools.tree.dot(tree, other))
        140.0
    """
    return jax.tree.reduce(
        u.math.add,
        jax.tree.map(u.math.sum, jax.tree.map(jax.lax.mul, a, b, is_leaf=is_leaf), is_leaf=is_leaf),
        is_leaf=is_leaf
    )


def sum(
    tree: brainstate.typing.PyTree[ArrayLike],
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> float(braintools.tree.sum(tree))
        6.0
    """
    return jax.tree.reduce(u.math.add, jax.tree.map(u.math.sum, tree, is_leaf=is_leaf), is_leaf=is_leaf)


def squared_norm(
    tree: brainstate.typing.PyTree[ArrayLike],
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> float(braintools.tree.squared_norm(tree))
        14.0
    """
    return jax.tree.reduce(
        u.math.add,
        jax.tree.map(lambda x: u.math.einsum('...,...->', x, x), tree, is_leaf=is_leaf),
        is_leaf=is_leaf
    )


def concat(
    trees: Sequence[brainstate.typing.PyTree[ArrayLike]],
    axis: int = 0,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> other = {'a': jnp.array([10.0, 20.0]), 'b': jnp.array([30.0])}
        >>> out = braintools.tree.concat([tree, other])
        >>> {k: v.tolist() for k, v in out.items()}
        {'a': [1.0, 2.0, 10.0, 20.0], 'b': [3.0, 30.0]}
    """
    return jax.tree.map(
        lambda *args: u.math.concatenate(args, axis=axis),
        *trees,
        is_leaf=is_leaf
    )


def split(
    tree: brainstate.typing.PyTree[jax.Array],
    sizes: Tuple[int],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> Tuple[brainstate.typing.PyTree[jax.Array], ...]:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'x': jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        >>> parts = braintools.tree.split(tree, (2, 2))
        >>> [p['x'].tolist() for p in parts]
        [[0.0, 1.0], [2.0, 3.0], [4.0]]
    """
    idx = 0
    result: list[brainstate.typing.PyTree[jax.Array]] = []
    for s in sizes:
        result.append(jax.tree.map(lambda x: x[idx: idx + s], tree, is_leaf=is_leaf))
        idx += s
    result.append(jax.tree.map(lambda x: x[idx:], tree, is_leaf=is_leaf))
    return tuple(result)


def idx(
    tree: brainstate.typing.PyTree[ArrayLike],
    idx: int | slice | ArrayLike,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'x': jnp.array([0.0, 1.0, 2.0, 3.0]), 'y': jnp.array([0.0, 10.0, 20.0, 30.0])}
        >>> out = braintools.tree.idx(tree, 1)
        >>> {k: v.tolist() for k, v in out.items()}
        {'x': 1.0, 'y': 10.0}
        >>>
        >>> out = braintools.tree.idx(tree, [0, 2])
        >>> {k: v.tolist() for k, v in out.items()}
        {'x': [0.0, 2.0], 'y': [0.0, 20.0]}
    """
    # ``idx`` is documented as array_like, which includes Python lists; indexing a
    # jax array with a list raises a confusing TypeError, so normalise it to an
    # array first. Tuples are deliberately left alone: ``x[(i, j)]`` is
    # multi-axis indexing (and may contain slices/``None``/``Ellipsis``), so
    # turning a tuple into an index array would change its meaning.
    if isinstance(idx, list):
        idx = np.asarray(idx)
    return jax.tree.map(lambda x: x[idx], tree, is_leaf=is_leaf)


def expand(
    tree: brainstate.typing.PyTree[ArrayLike],
    axis: int,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> out = braintools.tree.expand(tree, 0)
        >>> {k: v.shape for k, v in out.items()}
        {'a': (1, 2), 'b': (1, 1)}
    """
    return jax.tree.map(lambda x: u.math.expand_dims(x, axis), tree, is_leaf=is_leaf)


def take(
    tree: brainstate.typing.PyTree[ArrayLike],
    idx: int | slice | ArrayLike,
    axis: int,
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'m': jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])}
        >>> braintools.tree.take(tree, jnp.array([0, 2]), axis=1)['m'].tolist()
        [[0.0, 2.0], [3.0, 5.0]]
        >>>
        >>> braintools.tree.take(tree, slice(1, None), axis=1)['m'].tolist()
        [[1.0, 2.0], [4.0, 5.0]]
    """

    def take_(x):
        indices = idx
        if isinstance(indices, slice):
            slices = [slice(None)] * x.ndim
            slices[axis] = idx
            return x[tuple(slices)]
        # ``idx`` is documented as array_like, which includes Python lists and
        # tuples; ``u.math.take`` only accepts arrays or scalars, so normalise first.
        if isinstance(indices, (list, tuple)):
            indices = np.asarray(indices)
        return u.math.take(x, indices, axis)

    return jax.tree.map(take_, tree, is_leaf=is_leaf)


def as_numpy(
    tree: brainstate.typing.PyTree[ArrayLike],
    is_leaf: Callable[[Any], bool] | None = u.math.is_quantity
) -> brainstate.typing.PyTree[np.ndarray]:
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

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> tree = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        >>> out = braintools.tree.as_numpy(tree)
        >>> type(out['a']).__name__, out['b'].tolist()
        ('ndarray', [3.0])
    """
    return jax.tree.map(lambda x: np.asarray(x), tree, is_leaf=is_leaf)
