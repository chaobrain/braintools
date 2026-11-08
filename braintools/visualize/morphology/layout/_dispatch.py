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

"""Public 2D layout dispatcher.

``build_layout_branches_2d`` is the single entry point that the scene
builders in :mod:`braintools.visualize.morphology.scene2d` call. It:

1. Validates the ``mode``, ``root_layout``, and ``layout_family``
   arguments against the supported sets. The family names and their
   aliases come from the registry in :mod:`braintools.visualize.morphology.config`, which
   is also what ``configure(layout_2d_default=...)`` validates against —
   the two entry points cannot drift apart on which families exist.
2. Normalizes aliases (``trunk_first`` → ``stem``).
3. Derives the per-branch length/radius spec once.
4. Routes to the family-specific builder.
"""

from braintools.visualize.morphology._tree import MorphologyTree, as_tree
from braintools.visualize.morphology.config import (
    DISPATCHED_2D_LAYOUTS,
    LAYOUT_2D_ALIASES,
)

from braintools.visualize.morphology.layout._balloon import (
    _build_layout_branches_balloon,
)
from braintools.visualize.morphology.layout._cache import (
    LayoutCache,
    get_default_layout_cache,
)
from braintools.visualize.morphology.layout._common import (
    LayoutBranch2D,
    _build_layout_specs,
)
from braintools.visualize.morphology.layout._config import (
    DEFAULT_LAYOUT_CONFIG,
    LayoutConfig,
)
from braintools.visualize.morphology.layout._fan import _build_layout_branches_fan
from braintools.visualize.morphology.layout._radial import (
    _build_layout_branches_radial_360,
)
from braintools.visualize.morphology.layout._stem import (
    _build_layout_branches_stem,
    _build_layout_branches_stem_linear,
)

_VALID_ROOT_LAYOUTS = ("type_split",)


def build_layout_branches_2d(
    morpho: MorphologyTree,
    *,
    mode: str,
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout_family: str = "stem",
    layout_config: LayoutConfig | None = None,
    cache: LayoutCache | None = None,
    use_cache: bool = True,
) -> tuple[LayoutBranch2D, ...]:
    morpho = as_tree(morpho, caller="build_layout_branches_2d(...)")
    if mode not in {"tree", "frustum"}:
        raise ValueError(f"Unsupported layout mode {mode!r}.")
    if root_layout not in _VALID_ROOT_LAYOUTS:
        raise ValueError(
            f"Unsupported root layout {root_layout!r}; expected one of {_VALID_ROOT_LAYOUTS}."
        )
    if layout_family not in DISPATCHED_2D_LAYOUTS:
        raise ValueError(f"Unsupported 2D layout family {layout_family!r}.")
    layout_family = LAYOUT_2D_ALIASES.get(layout_family, layout_family)

    config = layout_config or DEFAULT_LAYOUT_CONFIG

    def _build() -> tuple[LayoutBranch2D, ...]:
        layout_specs = _build_layout_specs(morpho)
        if layout_family == "fan":
            return _build_layout_branches_fan(
                morpho,
                layout_specs=layout_specs,
                min_branch_angle_deg=min_branch_angle_deg,
                layout_config=config,
            )
        if layout_family == "balloon":
            return _build_layout_branches_balloon(
                morpho,
                layout_specs=layout_specs,
                min_branch_angle_deg=min_branch_angle_deg,
                root_layout=root_layout,
                layout_config=config,
            )
        if layout_family == "radial_360":
            return _build_layout_branches_radial_360(
                morpho,
                layout_specs=layout_specs,
                min_branch_angle_deg=min_branch_angle_deg,
                layout_config=config,
            )
        if mode == "frustum":
            return _build_layout_branches_stem_linear(
                morpho,
                layout_specs=layout_specs,
                min_branch_angle_deg=min_branch_angle_deg,
                root_layout=root_layout,
                layout_config=config,
            )
        return _build_layout_branches_stem(
            morpho,
            layout_specs=layout_specs,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            layout_config=config,
        )

    if not use_cache:
        return _build()
    active_cache = cache if cache is not None else get_default_layout_cache()
    return active_cache.get_or_build(
        morpho,
        mode=mode,
        layout_family=layout_family,
        root_layout=root_layout,
        min_branch_angle_deg=min_branch_angle_deg,
        layout_config=config,
        build=_build,
    )
