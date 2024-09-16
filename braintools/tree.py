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
  return jtu.tree_map(lambda a: a * x, tree, is_leaf=is_leaf)


def mul(
    tree: PyTree[jax.typing.ArrayLike],
    x: PyTree[jax.typing.ArrayLike] | jax.typing.ArrayLike,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  if isinstance(x, jax.typing.ArrayLike):
    return scale(tree, x)
  return jtu.tree_map(lambda a, b: a * b, tree, x, is_leaf=is_leaf)


def shift(
    tree1: PyTree[jax.typing.ArrayLike],
    x: jax.typing.ArrayLike,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  return jtu.tree_map(lambda a: a + x, tree1, is_leaf=is_leaf)


def add(
    tree1: PyTree[jax.typing.ArrayLike],
    tree2: PyTree[jax.typing.ArrayLike] | jax.typing.ArrayLike,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  if isinstance(tree2, jax.Array):
    return shift(tree1, tree2)
  return jtu.tree_map(lambda a, b: a + b, tree1, tree2, is_leaf=is_leaf)


def sub(
    tree1: PyTree[jax.typing.ArrayLike],
    tree2: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  return jtu.tree_map(lambda a, b: a - b, tree1, tree2, is_leaf=is_leaf)


def dot(
    a: PyTree,
    b: PyTree,
    is_leaf: Callable[[Any], bool] | None = None
) -> jax.Array:
  return jtu.tree_reduce(
    u.math.add,
    jtu.tree_map(u.math.sum, jax.tree_map(jax.lax.mul, a, b, is_leaf=is_leaf), is_leaf=is_leaf),
    is_leaf=is_leaf
  )


def sum(
    tree: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> jax.Array:
  return jtu.tree_reduce(u.math.add, jtu.tree_map(u.math.sum, tree, is_leaf=is_leaf), is_leaf=is_leaf)


def squared_norm(
    tree: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> jax.Array:
  return jtu.tree_reduce(
    u.math.add,
    jtu.tree_map(lambda x: u.math.einsum('...,...->', x, x), tree, is_leaf=is_leaf),
    is_leaf=is_leaf
  )


def concat(
    trees: Sequence[PyTree[jax.typing.ArrayLike]],
    axis: int = 0,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  return jtu.tree_map(lambda *args: u.math.concatenate(args, axis=axis), *trees, is_leaf=is_leaf)


def split(
    tree: PyTree[jax.Array],
    sizes: Tuple[int],
    is_leaf: Callable[[Any], bool] | None = None
) -> Tuple[PyTree[jax.Array], ...]:
  idx = 0
  result: list[PyTree[jax.Array]] = []
  for s in sizes:
    result.append(jtu.tree_map(lambda x: x[idx: idx + s], tree, is_leaf=is_leaf))
    idx += s
  result.append(jtu.tree_map(lambda x: x[idx:], tree, is_leaf=is_leaf))
  return tuple(result)


def idx(
    tree: PyTree[jax.typing.ArrayLike],
    idx,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  return jtu.tree_map(lambda x: x[idx], tree, is_leaf=is_leaf)


def expand(
    tree: PyTree[jax.typing.ArrayLike],
    axis,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  return jtu.tree_map(lambda x: u.math.expand_dims(x, axis), tree, is_leaf=is_leaf)


def take(
    tree: PyTree[jax.typing.ArrayLike],
    idx,
    axis: int,
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
  def take_(x):
    indices = idx
    if isinstance(indices, slice):
      slices = [slice(None)] * x.ndim
      slices[axis] = idx
      return x[tuple(slices)]
    return u.math.take(x, indices, axis)

  return jtu.tree_map(take_, tree, is_leaf=is_leaf)


def as_numpy(
    tree: PyTree[jax.typing.ArrayLike],
    is_leaf: Callable[[Any], bool] | None = None
) -> PyTree[np.ndarray]:
  return jtu.tree_map(lambda x: np.asarray(x), tree, is_leaf=is_leaf)
