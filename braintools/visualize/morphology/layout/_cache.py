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

"""Layout-result cache keyed on the morphology structure and layout arguments.

The stem layout family runs a non-trivial collision-avoidance search,
which makes it expensive to recompute for every ``plot2d`` call in a
notebook. :class:`LayoutCache` memoizes results by a key that captures:

* the structure of the morphology,
  :attr:`MorphologyData.structure_key <braintools.visualize.morphology.MorphologyData.structure_key>`:
  branch order, names, types, parents, child order, attachment
  coordinates, and the segment lengths and radii rounded to 1e-6 um;
* the :class:`LayoutConfig` fields;
* the dispatcher arguments: ``mode``, ``layout_family``,
  ``min_branch_angle_deg``, ``root_layout``.

A structural key is chosen over object identity deliberately: two
morphologies with identical structure deserve to share a cache entry, and
any edit produces new data with a different key. The key is computed once
per :class:`MorphologyData` instance (the data is immutable). A bounded LRU
keeps memory under control in long-running notebooks.
"""

import dataclasses
from collections import OrderedDict
from typing import Callable, Hashable


from braintools.visualize.morphology._tree import MorphologyTree, as_tree
from braintools.visualize.morphology.layout._common import LayoutBranch2D
from braintools.visualize.morphology.layout._config import (
    DEFAULT_LAYOUT_CONFIG,
    LayoutConfig,
)


class LayoutCache:
    """Size-bounded LRU cache for layout-branch builder results.

    Parameters
    ----------
    maxsize : int
        Maximum number of cached layouts. When exceeded, the oldest
        entry (by access order) is evicted.
    """

    __slots__ = ("_maxsize", "_entries", "hits", "misses")

    def __init__(self, maxsize: int = 64) -> None:
        if maxsize <= 0:
            raise ValueError(f"LayoutCache maxsize must be > 0, got {maxsize!r}.")
        self._maxsize = int(maxsize)
        self._entries: "OrderedDict[Hashable, tuple[LayoutBranch2D, ...]]" = (
            OrderedDict()
        )
        self.hits = 0
        self.misses = 0

    def get_or_build(
        self,
        morpho: MorphologyTree,
        *,
        mode: str,
        layout_family: str,
        root_layout: str,
        min_branch_angle_deg: float | None,
        layout_config: LayoutConfig | None,
        build: Callable[[], tuple[LayoutBranch2D, ...]],
    ) -> tuple[LayoutBranch2D, ...]:
        key = _make_cache_key(
            morpho,
            mode=mode,
            layout_family=layout_family,
            root_layout=root_layout,
            min_branch_angle_deg=min_branch_angle_deg,
            layout_config=layout_config or DEFAULT_LAYOUT_CONFIG,
        )
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return cached
        self.misses += 1
        result = tuple(build())
        self._entries[key] = result
        if len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
        return result

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)


def _make_cache_key(
    morpho: MorphologyTree,
    *,
    mode: str,
    layout_family: str,
    root_layout: str,
    min_branch_angle_deg: float | None,
    layout_config: LayoutConfig,
) -> Hashable:
    metric_key = _metric_key(morpho)
    config_key = _layout_config_key(layout_config)
    angle_key = None if min_branch_angle_deg is None else float(min_branch_angle_deg)
    return (metric_key, config_key, mode, layout_family, root_layout, angle_key)


def _metric_key(morpho: MorphologyTree) -> Hashable:
    """Return the cached structural key of the morphology (see the module docstring)."""
    return as_tree(morpho).data.structure_key


def _layout_config_key(layout_config: LayoutConfig) -> Hashable:
    """Return a hashable key for a :class:`LayoutConfig`.

    :class:`LayoutConfig` is a frozen dataclass with primitive fields
    only, so ``(field_name, value)`` tuples produce a stable hashable
    key without needing :func:`dataclasses.astuple` (which would break
    if anyone added a field containing a mutable default).
    """
    return tuple(
        (field.name, getattr(layout_config, field.name))
        for field in dataclasses.fields(layout_config)
    )


# Module-level default cache used by the dispatcher. Tests can call
# ``get_default_layout_cache().clear()`` to reset state between cases.
_default_cache = LayoutCache(maxsize=64)


def get_default_layout_cache() -> LayoutCache:
    return _default_cache
