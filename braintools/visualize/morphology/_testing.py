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

"""Shared fixture builders for ``braintools.visualize.morphology`` tests.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. Helpers here are consumed by the co-located
``*_test.py`` modules; nothing in this file is part of the public API.

The fixtures are built with :class:`Branch` and :class:`Morphology`, two
small test-only builders that mirror the construction calls of a simulator
morphology (``Branch.from_lengths``, ``Branch.from_points``,
``Morphology.from_root``, ``Morphology.attach``) and implement
``__morphology_data__()``. Inputs are rounded through float32 the way such a
simulator stores them, so the fixtures reproduce its geometry bit for bit
(and the pixel baselines drawn from it). The tail of the module holds
selection helpers and render-backend doubles.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

import brainunit as u
import numpy as np
import pytest

from braintools.visualize.morphology._data import (
    BranchData,
    LocsetData,
    MorphologyData,
    RegionData,
    TopologyGraph,
)  # noqa: F401
from braintools.visualize.morphology._image_compare import (
    BASELINE_DPI,
    assert_figure_matches,
    baseline_style,
)
from braintools.visualize.morphology._tree import BranchNode, MorphologyTree, as_tree


def _f32(values) -> np.ndarray:
    return np.asarray(np.asarray(values, dtype=np.float32), dtype=float)


def _um(value) -> np.ndarray:
    return (
        _f32(value.to_decimal(u.um)) if isinstance(value, u.Quantity) else _f32(value)
    )


class Branch:
    """Test-only unattached branch geometry (see the module docstring)."""

    def __init__(
        self,
        *,
        lengths,
        radii_proximal,
        radii_distal,
        points_proximal=None,
        points_distal=None,
        type="custom",
    ):
        self.lengths = lengths
        self.radii_proximal = radii_proximal
        self.radii_distal = radii_distal
        self.points_proximal = points_proximal
        self.points_distal = points_distal
        self.type = type

    @classmethod
    def from_lengths(cls, *, lengths, radii, type="custom") -> "Branch":
        """``n`` segment lengths and ``n + 1`` node radii."""
        lengths, radii = _um(lengths), _um(radii)
        if radii.shape != (lengths.shape[0] + 1,):
            raise ValueError("from_lengths() needs len(radii) == len(lengths) + 1.")
        return cls(
            lengths=lengths,
            radii_proximal=radii[:-1],
            radii_distal=radii[1:],
            type=type,
        )

    @classmethod
    def from_points(cls, *, points, radii, type="custom") -> "Branch":
        """``n + 1`` points ``(x, y, z)`` and ``n + 1`` node radii."""
        points, radii = _um(points), _um(radii)
        lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
        return cls(
            lengths=lengths,
            radii_proximal=radii[:-1],
            radii_distal=radii[1:],
            points_proximal=points[:-1],
            points_distal=points[1:],
            type=type,
        )


class _BranchHandle:
    """``tree.<name>`` access: reads like a :class:`BranchNode`, ``.<child> = Branch`` attaches."""

    def __init__(self, tree: "Morphology", name: str) -> None:
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(self, "_name", name)

    def _node(self) -> BranchNode:
        tree = object.__getattribute__(self, "_tree")
        name = object.__getattribute__(self, "_name")
        return tree.branch(name=name)

    def __getattr__(self, attr: str):
        tree = object.__getattribute__(self, "_tree")
        name = object.__getattribute__(self, "_name")
        if attr in tree._names and tree._parents[
            tree._names.index(attr)
        ] == tree._names.index(name):
            return _BranchHandle(tree, attr)
        return getattr(self._node(), attr)

    def __setattr__(self, attr: str, value) -> None:
        tree = object.__getattribute__(self, "_tree")
        tree.attach(
            parent=object.__getattribute__(self, "_name"),
            child_branch=value,
            child_name=attr,
        )


class Morphology:
    """Test-only mutable morphology builder implementing ``__morphology_data__()``."""

    def __init__(self) -> None:
        self._names: list[str] = []
        self._branches: list[Branch] = []
        self._parents: list[int | None] = []
        self._xs: list[tuple[float | None, float | None]] = []
        self._data: MorphologyData | None = None

    @classmethod
    def from_root(cls, branch: Branch, *, name: str = "soma") -> "Morphology":
        tree = cls()
        tree._add(name, branch, None, None, None)
        return tree

    def _add(self, name, branch, parent, parent_x, child_x) -> None:
        if name in self._names:
            raise ValueError(f"duplicate branch name {name!r}")
        self._names.append(name)
        self._branches.append(branch)
        self._parents.append(parent)
        self._xs.append((parent_x, child_x))
        self._data = None

    def attach(
        self,
        *,
        parent: str,
        child_branch: Branch,
        child_name: str,
        parent_x: float = 1.0,
        child_x: float = 0.0,
    ) -> None:
        self._add(
            child_name,
            child_branch,
            self._names.index(parent),
            float(parent_x),
            float(child_x),
        )

    def __morphology_data__(self) -> MorphologyData:
        if self._data is None:
            rows = []
            for index, (name, branch, parent, (parent_x, child_x)) in enumerate(
                zip(self._names, self._branches, self._parents, self._xs)
            ):
                points = {}
                if branch.points_proximal is not None:
                    points = dict(
                        points_proximal=branch.points_proximal * u.um,
                        points_distal=branch.points_distal * u.um,
                    )
                rows.append(
                    BranchData(
                        index=index,
                        name=name,
                        type=branch.type,
                        parent_index=parent,
                        parent_x=parent_x,
                        child_x=child_x,
                        lengths=branch.lengths * u.um,
                        radii_proximal=branch.radii_proximal * u.um,
                        radii_distal=branch.radii_distal * u.um,
                        **points,
                    )
                )
            self._data = MorphologyData(branches=tuple(rows))
        return self._data

    # Read access mirrors the tree view the plots use.
    @property
    def tree(self) -> MorphologyTree:
        return as_tree(self)

    @property
    def branches(self) -> tuple[BranchNode, ...]:
        return self.tree.branches

    @property
    def root(self) -> BranchNode:
        return self.tree.root

    @property
    def has_full_point_geometry(self) -> bool:
        return self.tree.has_full_point_geometry

    @property
    def total_length(self) -> u.Quantity:
        return u.Quantity(
            float(sum(np.sum(b.lengths) for b in self._branches)), unit=u.um
        )

    def branch(self, *, name: str) -> BranchNode:
        return self.tree.branches[self._names.index(name)]

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            raise AttributeError(attr)
        if attr in self._names:
            return _BranchHandle(self, attr)
        raise AttributeError(f"Morphology has no branch named {attr!r}")


# Selections, as the evaluated masks of a simulator would produce them.


def all_region(tree) -> RegionData:
    """Every branch, whole."""
    return RegionData(tuple((b.index, 0.0, 1.0) for b in as_tree(tree).branches))


def terminals(tree) -> LocsetData:
    """The distal end of every leaf branch."""
    return LocsetData(
        tuple((b.index, 1.0) for b in as_tree(tree).branches if b.n_children == 0)
    )


def branch_points(tree) -> LocsetData:
    """The distal end of every branch that has children."""
    return LocsetData(
        tuple((b.index, 1.0) for b in as_tree(tree).branches if b.n_children > 0)
    )


def branch_slice(branch_index, prox: float, dist: float) -> RegionData:
    """``[prox, dist]`` of one branch or of each branch in a list."""
    indices = [branch_index] if isinstance(branch_index, int) else list(branch_index)
    return RegionData(tuple((int(i), float(prox), float(dist)) for i in indices))


def branches_of_type(tree, branch_type: str) -> RegionData:
    """Every branch of one type, whole."""
    return RegionData(
        tuple(
            (b.index, 0.0, 1.0) for b in as_tree(tree).branches if b.type == branch_type
        )
    )


def make_soma(*, length: float = 20.0, radius: float = 10.0) -> Branch:
    return Branch.from_lengths(
        lengths=[length] * u.um, radii=[radius, radius] * u.um, type="soma"
    )


def make_deep_chain_tree(n_branches: int = 1200) -> Morphology:
    """An unbranched chain of ``n_branches`` branches hanging off a soma (pins iterative walks)."""
    if n_branches < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches!r}.")
    tree = Morphology.from_root(make_soma(), name="soma")
    parent = "soma"
    for index in range(n_branches - 1):
        child = Branch.from_lengths(
            lengths=[15.0, 10.0] * u.um,
            radii=[2.0, 1.5, 1.0] * u.um,
            type="apical_dendrite",
        )
        name = f"seg_{index}"
        tree.attach(parent=parent, child_branch=child, child_name=name, parent_x=1.0)
        parent = name
    return tree


def _make_soma_dend_tree(
    *,
    soma_length: float = 20.0,
    dend_type: str = "basal_dendrite",
    dend_length: float = 30.0,
):
    tree = Morphology.from_root(make_soma(length=soma_length), name="soma")
    child = Branch.from_lengths(
        lengths=[dend_length] * u.um, radii=[2.0, 1.0] * u.um, type=dend_type
    )
    tree.attach(parent="soma", child_branch=child, child_name="dend", parent_x=1.0)
    return tree


def make_node_tree() -> Morphology:
    """A single-branch soma with explicit 3D points.

    Useful whenever a test needs a morphology that satisfies
    ``has_full_point_geometry`` but is otherwise trivial.
    """
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [10.0, 0.0, 1.0]] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


def make_length_only_tree(*, child_name: str = "dend") -> Morphology:
    """A soma plus a two-segment apical dendrite, length-only (no 3D points).

    The resulting tree has 2 branches:
      * ``soma`` — one segment, length 20 µm, radius 10 µm.
      * ``<child_name>`` — two segments (lengths 8 µm and 12 µm; radii
        2 / 1.5 / 1 µm).

    This layout is the reference fixture for testing stem/balloon layouts
    and frustum polygon generation because its geometry is small enough
    to verify by hand.
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[8.0, 12.0] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name=child_name, parent_x=1.0)
    return tree


def make_soma_dend_tree() -> Morphology:
    """A soma plus one 100 um tapering *basal* dendrite, length-only.

    Deliberately
    distinct from :func:`make_length_only_tree`: one dendrite segment
    instead of two, and ``basal_dendrite`` instead of ``apical_dendrite``.
    The single long segment is what keeps the CV count small enough to
    assert coverage fractions by hand, and the tapering radii give the
    frustum renderer something to narrow. The zero-argument name is kept
    because that is how every call site reads.
    """
    return _make_soma_dend_tree(dend_length=100.0)


def make_four_type_tree() -> Morphology:
    """Soma with an apical dendrite, a basal dendrite, and an axon.

    Length-only geometry. The four distinct branch types exercise every
    entry of the default palette at once, so a colour-mapping regression
    shows up on a single render. Used by the matplotlib layout/colorbar
    tests and by ``compare2d_test.py``.
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"
    )
    apical = Branch.from_lengths(
        lengths=[50.0, 40.0] * u.um,
        radii=[3.0, 2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    basal = Branch.from_lengths(
        lengths=[30.0] * u.um, radii=[2.0, 1.5] * u.um, type="basal_dendrite"
    )
    axon = Branch.from_lengths(
        lengths=[40.0] * u.um, radii=[1.0, 0.6] * u.um, type="axon"
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=apical, child_name="apical", parent_x=1.0)
    tree.attach(parent="soma", child_branch=basal, child_name="basal", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


def make_root_split_tree() -> Morphology:
    """Soma with one apical dendrite and one axon, both length-only.

    Used by the ``root_layout="type_split"`` tests that check that the
    axon and dendrite end up on opposite half-planes.
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[25.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    axon = Branch.from_lengths(
        lengths=[18.0] * u.um,
        radii=[1.0, 0.8] * u.um,
        type="axon",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


def make_two_dendrite_tree() -> Morphology:
    """Soma with two dendrite children of equal length.

    Shared between legacy-angle, balloon, and radial_360 layout tests
    that need a symmetric two-child tree.
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend_a = Branch.from_lengths(
        lengths=[15.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="apical_dendrite",
    )
    dend_b = Branch.from_lengths(
        lengths=[15.0] * u.um,
        radii=[2.0, 1.5] * u.um,
        type="basal_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend_a, child_name="dend_a", parent_x=1.0)
    tree.attach(parent="soma", child_branch=dend_b, child_name="dend_b", parent_x=1.0)
    return tree


def make_projected_node_tree() -> Morphology:
    """Soma + apical dendrite with explicit 3D points.

    The dendrite has three points (so the projected 2D centerline has
    two segments) and a known 2D footprint for overlay resolution
    assertions under ``projection_plane='xy'``.
    """
    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = Branch.from_points(
        points=[[20.0, 0.0, 0.0], [20.0, 40.0, 0.0], [20.0, 80.0, 0.0]] * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.0)
    return tree


def make_baseline_tree(*, apical_scale: float = 1.0) -> Morphology:
    """The branching 3D tree every pixel baseline draws.

    A soma, an apical dendrite that forks once, a basal dendrite and an axon,
    all with explicit 3D points so the projected layout and the Sholl profile
    are defined. ``apical_scale`` stretches the trunk (and so moves its
    children) away from the soma: ``1.01`` is the one-percent length change
    the baselines must detect. The geometry is the same as braincell's
    baseline tree, so both packages share the baseline images.
    """
    s = float(apical_scale)

    def pts(rows, origin=(0.0, 0.0, 0.0)):
        o = np.asarray(origin, dtype=float)
        return (o + s * np.asarray(rows, dtype=float)) * u.um

    soma = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    trunk_rows = [[0.0, 0.0, 0.0], [10.0, 30.0, 0.0], [5.0, 70.0, 5.0]]
    trunk_end = np.asarray([20.0, 0.0, 0.0]) + s * np.asarray(trunk_rows[-1])
    trunk = Branch.from_points(
        points=pts(trunk_rows, (20.0, 0.0, 0.0)),
        radii=[3.0, 2.5, 2.0] * u.um,
        type="apical_dendrite",
    )
    fork_a = Branch.from_points(
        points=(
            trunk_end
            + np.asarray([[0.0, 0.0, 0.0], [-25.0, 30.0, 0.0], [-30.0, 60.0, 0.0]])
        )
        * u.um,
        radii=[2.0, 1.5, 1.0] * u.um,
        type="apical_dendrite",
    )
    fork_b = Branch.from_points(
        points=(
            trunk_end
            + np.asarray([[0.0, 0.0, 0.0], [25.0, 35.0, 0.0], [40.0, 55.0, 0.0]])
        )
        * u.um,
        radii=[2.0, 1.4, 1.0] * u.um,
        type="apical_dendrite",
    )
    basal = Branch.from_points(
        points=[[0.0, 0.0, 0.0], [-30.0, -20.0, 0.0], [-60.0, -25.0, 0.0]] * u.um,
        radii=[2.5, 2.0, 1.5] * u.um,
        type="basal_dendrite",
    )
    axon = Branch.from_points(
        points=[[20.0, 0.0, 0.0], [50.0, -20.0, 0.0], [90.0, -30.0, 0.0]] * u.um,
        radii=[1.2, 1.0, 0.8] * u.um,
        type="axon",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(parent="soma", child_branch=trunk, child_name="trunk", parent_x=1.0)
    tree.attach(parent="trunk", child_branch=fork_a, child_name="fork_a", parent_x=1.0)
    tree.attach(parent="trunk", child_branch=fork_b, child_name="fork_b", parent_x=1.0)
    tree.attach(parent="soma", child_branch=basal, child_name="basal", parent_x=0.0)
    tree.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=1.0)
    return tree


def make_baseline_cv_graph() -> TopologyGraph:
    """Control-volume graph of :func:`make_baseline_tree`, one CV per branch.

    These are the CVs braincell builds for that tree, with the first half of
    the trunk highlighted (``BranchSlice(1, 0.0, 0.5)``).
    """
    return TopologyGraph(
        node_ids=(0, 1, 2, 3, 4, 5),
        edges=((0, 1), (1, 2), (1, 3), (0, 4), (0, 5)),
        root_id=0,
        level="cv",
        highlight_fractions={1: 0.5499430137741315},
    )


def make_baseline_node_graph() -> TopologyGraph:
    """Solver-node graph braincell builds for :func:`make_baseline_tree` (one CV per branch)."""
    return TopologyGraph(
        node_ids=tuple(range(13)),
        edges=(
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (4, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (2, 11),
            (11, 12),
        ),  # fmt: skip
        root_id=0,
        level="node",
    )


def make_fan_root_partition_tree() -> Morphology:
    """Soma with children attached at left / middle / right root positions.

    The chosen ``parent_x`` values exercise the intended default fan
    root binning with the values the current morphology API allows:
      * ``0.0`` -> left sector
      * ``0.5`` -> middle sector
      * ``1.0`` -> right sector
    """
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(
            lengths=[12.0] * u.um, radii=[1.8, 1.1] * u.um, type="apical_dendrite"
        ),
        child_name="left_dend",
        parent_x=0.0,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(
            lengths=[12.0] * u.um, radii=[1.6, 1.0] * u.um, type="apical_dendrite"
        ),
        child_name="mid_dend",
        parent_x=0.5,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(
            lengths=[12.0] * u.um, radii=[1.0, 0.7] * u.um, type="axon"
        ),
        child_name="mid_axon",
        parent_x=0.5,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(
            lengths=[12.0] * u.um, radii=[1.5, 0.9] * u.um, type="basal_dendrite"
        ),
        child_name="right_near",
        parent_x=1.0,
    )
    tree.attach(
        parent="soma",
        child_branch=Branch.from_lengths(
            lengths=[12.0] * u.um, radii=[1.5, 0.9] * u.um, type="apical_dendrite"
        ),
        child_name="right_far",
        parent_x=1.0,
    )
    return tree


# =============================================================================
# Pixel baselines
# =============================================================================

BASELINE_DIR = Path(__file__).resolve().parent / "baseline_images"
"""Committed baseline PNGs, one per figure; not packaged in the wheel."""

UPDATE_BASELINES_ENV = "BRAINTOOLS_UPDATE_BASELINES"
"""Set to ``1`` to rewrite the baselines instead of comparing against them."""

BASELINE_FIGSIZE = (4.0, 3.0)
"""Figure size in inches; at ``BASELINE_DPI`` every baseline is 400 x 300 px."""

BASELINE_TOL = 0.0
"""RMS tolerance (0-255) for every baseline comparison."""


# =============================================================================
# Backend doubles
# =============================================================================


class FakeBackend:
    """Scene-agnostic test double that records the last :class:`RenderRequest`.

    Advertises ``supported_scene_kinds = frozenset({"2d", "3d"})`` so
    :func:`validate_backend_for_scene` accepts it for either dispatch
    direction, and returns the request unchanged so tests can inspect
    the scene that would be rendered.
    """

    name = "fake"
    supported_scene_kinds = frozenset({"2d", "3d"})

    def __init__(self) -> None:
        self.last_request = None

    def available(self) -> bool:
        return True

    def render(self, request):
        self.last_request = request
        return request


# =============================================================================
# TestCase mixins
# =============================================================================


if TYPE_CHECKING:
    _MixinBase = unittest.TestCase
else:
    _MixinBase = object


class VisDefaultsResetMixin(_MixinBase):
    """Restore ``braintools.visualize.morphology`` global defaults around every test.

    ``plot2d``/``plot3d`` read module-level defaults that several tests
    mutate. Mixing this in ahead of :class:`unittest.TestCase` gives a clean
    slate on entry and on exit, including when the test raises.
    """

    def setUp(self) -> None:  # noqa: N802 - unittest naming
        super().setUp()
        from braintools.visualize.morphology import reset_defaults

        reset_defaults()
        self.addCleanup(reset_defaults)


class BaselineMixin(VisDefaultsResetMixin):
    """Render in the pinned baseline style and close every figure afterwards.

    Mix in ahead of :class:`unittest.TestCase`; :meth:`new_axes` then returns
    axes of the fixed baseline size, and :meth:`assert_baseline` compares the
    figure with ``baseline_images/<name>.png``.
    """

    def setUp(self) -> None:  # noqa: N802 - unittest naming
        super().setUp()
        import matplotlib.pyplot as plt

        self.enterContext(baseline_style())
        self.addCleanup(plt.close, "all")

    @staticmethod
    def new_axes(figsize=BASELINE_FIGSIZE):
        import matplotlib.pyplot as plt

        return plt.subplots(figsize=figsize, dpi=BASELINE_DPI)

    @staticmethod
    def assert_baseline(fig, name: str, *, tol: float = BASELINE_TOL) -> None:
        assert_figure_matches(
            fig,
            BASELINE_DIR / f"{name}.png",
            tol=tol,
            update_env=UPDATE_BASELINES_ENV,
        )

    def assert_baseline_differs(self, fig, name: str) -> None:
        """Assert that ``fig`` does *not* match baseline ``name``, never rewriting it.

        Skipped while ``BRAINTOOLS_UPDATE_BASELINES=1`` is set: a regeneration
        run must never overwrite a baseline with the deliberately wrong figure.
        """
        if os.environ.get(UPDATE_BASELINES_ENV):
            self.skipTest("baselines are being regenerated")
        with self.assertRaisesRegex(
            AssertionError, rf"{name}\.png differs from its baseline"
        ):
            self.assert_baseline(fig, name)


# =============================================================================
# Optional plugin probes
# =============================================================================

PYTEST_BENCHMARK_AVAILABLE = importlib.util.find_spec("pytest_benchmark") is not None
"""True when the optional ``pytest-benchmark`` plugin is importable."""

needs_benchmark = pytest.mark.skipif(
    not PYTEST_BENCHMARK_AVAILABLE,
    reason="pytest-benchmark is not installed",
)
"""Skip mark for the performance baselines in ``braintools.visualize.morphology``.

Apply it per function rather than as a module-level ``pytestmark``: every file
carrying benchmarks also carries ordinary tests that must keep running when the
plugin is absent.

Benchmarks must be plain pytest functions, never :class:`unittest.TestCase`
methods — pytest cannot inject the function-scoped ``benchmark`` fixture into a
``TestCase``. They pair with the ``clean_layout_cache`` fixture from
``visualize/morphology/conftest.py``, which pytest supplies by name.
"""
