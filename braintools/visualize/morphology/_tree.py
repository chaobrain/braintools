# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Navigable view of a :class:`MorphologyData` used by layouts and scenes.

:class:`MorphologyData` stores a flat, frozen table of branches. The layout
and scene code walks the tree (``node.parent``, ``node.children``), so
:func:`as_tree` wraps the data in a :class:`MorphologyTree` whose branches
are :class:`BranchNode` objects. The tree is built once per data instance and
cached on it.
"""

from __future__ import annotations

from typing import Any

import brainunit as u

from braintools.visualize.morphology._data import (
    BranchData,
    MorphologyData,
    as_morphology_data,
)

__all__ = ["BranchNode", "MorphologyTree", "as_tree"]


class BranchNode:
    """One branch of a :class:`MorphologyTree`, linked to its parent and children."""

    __slots__ = ("_tree", "_data")

    def __init__(self, tree: "MorphologyTree", data: BranchData) -> None:
        self._tree = tree
        self._data = data

    @property
    def branch(self) -> BranchData:
        """The branch geometry."""
        return self._data

    @property
    def index(self) -> int:
        return self._data.index

    @property
    def name(self) -> str:
        return self._data.name

    @property
    def type(self) -> str:
        return self._data.type

    @property
    def parent_x(self) -> float | None:
        return self._data.parent_x

    @property
    def child_x(self) -> float | None:
        return self._data.child_x

    @property
    def lengths(self) -> u.Quantity:
        return self._data.lengths

    @property
    def radii_proximal(self) -> u.Quantity:
        return self._data.radii_proximal

    @property
    def radii_distal(self) -> u.Quantity:
        return self._data.radii_distal

    @property
    def points_proximal(self) -> u.Quantity | None:
        return self._data.points_proximal

    @property
    def points_distal(self) -> u.Quantity | None:
        return self._data.points_distal

    @property
    def length(self) -> u.Quantity:
        return self._data.length

    @property
    def n_segments(self) -> int:
        return self._data.n_segments

    @property
    def parent(self) -> "BranchNode | None":
        parent = self._data.parent_index
        return None if parent is None else self._tree.branches[parent]

    @property
    def children(self) -> tuple["BranchNode", ...]:
        branches = self._tree.branches
        return tuple(branches[i] for i in self._tree.data.children[self._data.index])

    @property
    def n_children(self) -> int:
        return len(self._tree.data.children[self._data.index])

    def __repr__(self) -> str:
        return f"BranchNode(index={self.index}, name={self.name!r}, type={self.type!r})"


class MorphologyTree:
    """Tree view of a :class:`MorphologyData`; build it with :func:`as_tree`."""

    __slots__ = ("data", "branches")

    def __init__(self, data: MorphologyData) -> None:
        self.data = data
        self.branches: tuple[BranchNode, ...] = tuple(
            BranchNode(self, branch) for branch in data.branches
        )

    @property
    def root(self) -> BranchNode:
        return self.branches[self.data.root_index]

    @property
    def name(self) -> str | None:
        return self.data.name

    @property
    def has_full_point_geometry(self) -> bool:
        return self.data.has_full_point_geometry

    def __morphology_data__(self) -> MorphologyData:
        return self.data


_TREE_ATTR = "_braintools_tree"


def as_tree(obj: Any, *, caller: str | None = None) -> MorphologyTree:
    """Return the cached :class:`MorphologyTree` of ``obj``.

    ``obj`` may be a :class:`MorphologyTree`, a :class:`MorphologyData`, or any
    object implementing ``__morphology_data__()``.

    Raises
    ------
    TypeError
        If ``obj`` is none of these; ``caller`` prefixes the message.
    """
    if isinstance(obj, MorphologyTree):
        return obj
    try:
        data = as_morphology_data(obj)
    except TypeError as exc:
        if caller is None:
            raise
        raise TypeError(f"{caller} {exc}") from None
    tree = data.__dict__.get(_TREE_ATTR)
    if tree is None:
        tree = MorphologyTree(data)
        # MorphologyData is frozen; the cache lives in the instance dict like a cached_property.
        data.__dict__[_TREE_ATTR] = tree
    return tree
