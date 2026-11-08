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

"""Neutral data consumed by :mod:`braintools.visualize.morphology`.

Every plot in this package takes plain, frozen data rather than the objects
of a particular simulator, so the package never imports one. A simulator (for
example braincell) converts its own objects into these types, or implements
the conversion protocol:

* ``obj.__morphology_data__() -> MorphologyData`` for morphology plots;
* ``obj.__topology_data__() -> TopologyGraph`` for discrete topology graphs.

:func:`as_morphology_data` and :func:`as_topology_graph` normalise any
accepted input and are what the plot functions call first.

Geometry is stored as read-only ``float64`` numpy arrays in micrometres,
wrapped in :class:`brainunit.Quantity`. The arrays are host data: drawing
needs no accelerator.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from functools import cached_property
from typing import Any, Hashable

import brainunit as u
import numpy as np

__all__ = [
    "BranchData",
    "LocsetData",
    "MorphologyData",
    "RegionData",
    "TopologyGraph",
    "as_morphology_data",
    "as_topology_graph",
]


def _as_um(name: str, owner: str, value: Any, *, ndim: int) -> u.Quantity:
    """Return ``value`` as a read-only float64 Quantity in micrometres."""
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"{owner}.{name} must be a brainunit Quantity with a length unit, got {type(value).__name__}."
        )
    try:
        mantissa = np.array(value.to_decimal(u.um), dtype=float)
    except Exception as exc:  # unit mismatch raises a brainunit error
        raise TypeError(
            f"{owner}.{name} must have a length unit, got {u.get_unit(value)}."
        ) from exc
    if mantissa.ndim != ndim:
        raise ValueError(
            f"{owner}.{name} must be {ndim}-D, got shape {mantissa.shape}."
        )
    mantissa.setflags(write=False)
    return u.Quantity(mantissa, unit=u.um)


@dataclasses.dataclass(frozen=True)
class BranchData:
    """Geometry and attachment of one unbranched section of a morphology.

    A branch is a chain of ``n_seg`` frusta (segments). Segment ``k`` runs from
    radius ``radii_proximal[k]`` to ``radii_distal[k]`` over ``lengths[k]``.

    Parameters
    ----------
    index : int
        Position of this branch in :attr:`MorphologyData.branches`.
    name : str
        Branch name, unique within the morphology. Unlike ``index``, it is
        the identifier that survives edits of the source morphology.
    type : str
        Branch type, such as ``"soma"``, ``"axon"``, ``"basal_dendrite"``,
        ``"apical_dendrite"``. Selects the colour of the branch.
    parent_index : int or None
        Index of the parent branch, ``None`` for the root.
    parent_x : float or None
        Normalised position ``[0, 1]`` on the parent where this branch is
        attached; ``None`` for the root.
    child_x : float or None
        Normalised position ``[0, 1]`` on this branch that touches the
        parent; ``None`` for the root.
    lengths : brainunit.Quantity
        Segment lengths, shape ``(n_seg,)``, a length unit.
    radii_proximal, radii_distal : brainunit.Quantity
        Segment end radii, shape ``(n_seg,)``, a length unit.
    points_proximal, points_distal : brainunit.Quantity or None, optional
        Segment end points, shape ``(n_seg, 3)``, a length unit. Both or
        neither must be given; without them only the automatic 2-D layouts
        are available.

    Raises
    ------
    TypeError
        If a geometry field is not a Quantity with a length unit.
    ValueError
        If the geometry arrays disagree in shape or are empty.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.visualize.morphology import BranchData
        >>> soma = BranchData(
        ...     index=0, name="soma", type="soma", parent_index=None,
        ...     parent_x=None, child_x=None, lengths=[10.0] * u.um,
        ...     radii_proximal=[5.0] * u.um, radii_distal=[5.0] * u.um,
        ... )
        >>> soma.n_segments
        1
        >>> soma.length
        Quantity(10., "um")
    """

    index: int
    name: str
    type: str
    parent_index: int | None
    parent_x: float | None
    child_x: float | None
    lengths: u.Quantity
    radii_proximal: u.Quantity
    radii_distal: u.Quantity
    points_proximal: u.Quantity | None = None
    points_distal: u.Quantity | None = None

    def __post_init__(self) -> None:
        owner = f"BranchData({self.name!r})"
        set_ = object.__setattr__
        set_(self, "index", int(self.index))
        set_(self, "name", str(self.name))
        set_(self, "type", str(self.type))
        if self.parent_index is not None:
            set_(self, "parent_index", int(self.parent_index))
        for name in ("parent_x", "child_x"):
            value = getattr(self, name)
            if value is not None:
                set_(self, name, float(value))
        lengths = _as_um("lengths", owner, self.lengths, ndim=1)
        n_seg = lengths.shape[0]
        if n_seg == 0:
            raise ValueError(f"{owner} must have at least one segment.")
        set_(self, "lengths", lengths)
        for name in ("radii_proximal", "radii_distal"):
            radii = _as_um(name, owner, getattr(self, name), ndim=1)
            if radii.shape[0] != n_seg:
                raise ValueError(
                    f"{owner}.{name} has {radii.shape[0]} entries, but lengths has {n_seg}."
                )
            set_(self, name, radii)
        if (self.points_proximal is None) != (self.points_distal is None):
            raise ValueError(
                f"{owner} needs both points_proximal and points_distal, or neither."
            )
        if self.points_proximal is not None:
            for name in ("points_proximal", "points_distal"):
                points = _as_um(name, owner, getattr(self, name), ndim=2)
                if points.shape != (n_seg, 3):
                    raise ValueError(
                        f"{owner}.{name} must have shape ({n_seg}, 3), got {points.shape}."
                    )
                set_(self, name, points)

    @property
    def n_segments(self) -> int:
        """Number of segments."""
        return int(self.lengths.shape[0])

    @property
    def length(self) -> u.Quantity:
        """Total path length of the branch."""
        return u.Quantity(float(np.sum(self.lengths.mantissa)), unit=u.um)

    @property
    def has_points(self) -> bool:
        """Whether the branch carries 3-D point geometry."""
        return self.points_proximal is not None


@dataclasses.dataclass(frozen=True)
class MorphologyData:
    """A tree of :class:`BranchData`, the input of every morphology plot.

    Parameters
    ----------
    branches : tuple of BranchData
        All branches; ``branches[i].index`` must equal ``i``.
    root_index : int, optional
        Index of the root branch (default ``0``).
    children : tuple of tuple of int, optional
        Child indices of each branch, in drawing order. Layouts place children
        in this order. ``None`` (default) derives it from ``parent_index`` in
        index order.
    name : str, optional
        Display name, used as a default panel title.

    Raises
    ------
    ValueError
        If the indices are not ``0..n-1``, a parent does not exist, the
        branches do not form one tree rooted at ``root_index``, or
        ``children`` disagrees with ``parent_index``.

    See Also
    --------
    as_morphology_data : Accepts this type or an object implementing
        ``__morphology_data__()``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.visualize.morphology import BranchData, MorphologyData
        >>> soma = BranchData(
        ...     index=0, name="soma", type="soma", parent_index=None,
        ...     parent_x=None, child_x=None, lengths=[10.0] * u.um,
        ...     radii_proximal=[5.0] * u.um, radii_distal=[5.0] * u.um,
        ... )
        >>> dend = BranchData(
        ...     index=1, name="dend", type="basal_dendrite", parent_index=0,
        ...     parent_x=1.0, child_x=0.0, lengths=[20.0, 20.0] * u.um,
        ...     radii_proximal=[2.0, 1.5] * u.um, radii_distal=[1.5, 1.0] * u.um,
        ... )
        >>> data = MorphologyData(branches=(soma, dend))
        >>> data.children
        ((1,), ())
        >>> data.has_full_point_geometry
        False
    """

    branches: tuple[BranchData, ...]
    root_index: int = 0
    children: tuple[tuple[int, ...], ...] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        branches = tuple(self.branches)
        object.__setattr__(self, "branches", branches)
        n = len(branches)
        if n == 0:
            raise ValueError("MorphologyData needs at least one branch.")
        for i, branch in enumerate(branches):
            if not isinstance(branch, BranchData):
                raise TypeError(
                    f"MorphologyData.branches[{i}] must be BranchData, got {type(branch).__name__}."
                )
            if branch.index != i:
                raise ValueError(
                    f"Branch {branch.name!r} has index {branch.index}, but it is at position {i}."
                )
        root_index = int(self.root_index)
        object.__setattr__(self, "root_index", root_index)
        if not 0 <= root_index < n:
            raise ValueError(
                f"MorphologyData.root_index {root_index} is out of range for {n} branches."
            )
        if branches[root_index].parent_index is not None:
            raise ValueError(
                f"Root branch {branches[root_index].name!r} must have parent_index=None."
            )
        derived: list[list[int]] = [[] for _ in branches]
        for branch in branches:
            parent = branch.parent_index
            if parent is None:
                if branch.index != root_index:
                    raise ValueError(
                        f"Branch {branch.name!r} has no parent but is not the root."
                    )
                continue
            if not 0 <= parent < n:
                raise ValueError(
                    f"Branch {branch.name!r} has parent_index {parent}, out of range for {n} branches."
                )
            derived[parent].append(branch.index)
        if self.children is None:
            children = tuple(tuple(kids) for kids in derived)
        else:
            children = tuple(tuple(int(c) for c in kids) for kids in self.children)
            if len(children) != n or any(
                sorted(c) != d for c, d in zip(children, derived)
            ):
                raise ValueError(
                    "MorphologyData.children does not describe the same tree as parent_index."
                )
        object.__setattr__(self, "children", children)
        # Every branch must be reachable from the root (rules out cycles).
        seen = 0
        stack = [root_index]
        while stack:
            seen += 1
            stack.extend(children[stack.pop()])
        if seen != n:
            raise ValueError(
                "MorphologyData.branches do not form a single tree under the root (cycle or detached branch)."
            )

    @property
    def n_branches(self) -> int:
        """Number of branches."""
        return len(self.branches)

    @property
    def has_full_point_geometry(self) -> bool:
        """Whether every branch carries 3-D points (needed by 3-D and projected plots)."""
        return all(branch.has_points for branch in self.branches)

    @cached_property
    def structure_key(self) -> Hashable:
        """Hashable snapshot of the geometry and topology, computed once.

        Two instances with equal geometry (rounded to 1e-6 um) and equal child
        order share a key; :class:`LayoutCache` uses it.
        """
        rows: list[tuple] = []
        for branch in self.branches:
            rows.append(
                (
                    branch.index,
                    branch.name,
                    branch.type,
                    branch.parent_index,
                    None if branch.parent_x is None else round(branch.parent_x, 6),
                    None if branch.child_x is None else round(branch.child_x, 6),
                    _rounded_bytes(branch.lengths.mantissa),
                    _rounded_bytes(branch.radii_proximal.mantissa),
                    _rounded_bytes(branch.radii_distal.mantissa),
                )
            )
        return ("morpho", self.root_index, self.children, tuple(rows))

    def __morphology_data__(self) -> "MorphologyData":
        return self


def _rounded_bytes(values) -> bytes:
    return np.round(np.asarray(values, dtype=float), 6).tobytes()


@dataclasses.dataclass(frozen=True)
class RegionData:
    """A continuous selection of a morphology, as branch intervals.

    Parameters
    ----------
    intervals : tuple of (int, float, float)
        ``(branch_index, prox, dist)`` with ``0 <= prox <= dist <= 1``.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.visualize.morphology import RegionData
        >>> RegionData([(1, 0.0, 0.5)]).intervals
        ((1, 0.0, 0.5),)
    """

    intervals: tuple[tuple[int, float, float], ...]

    def __post_init__(self) -> None:
        intervals = tuple((int(b), float(p), float(d)) for b, p, d in self.intervals)
        for b, p, d in intervals:
            if not (0.0 <= p <= d <= 1.0):
                raise ValueError(
                    f"RegionData interval ({b}, {p}, {d}) needs 0 <= prox <= dist <= 1."
                )
        object.__setattr__(self, "intervals", intervals)


@dataclasses.dataclass(frozen=True)
class LocsetData:
    """Discrete locations on a morphology.

    Parameters
    ----------
    points : tuple of (int, float)
        ``(branch_index, x)`` with ``0 <= x <= 1``.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.visualize.morphology import LocsetData
        >>> LocsetData([(0, 0.5)]).points
        ((0, 0.5),)
    """

    points: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        points = tuple((int(b), float(x)) for b, x in self.points)
        for b, x in points:
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"LocsetData point ({b}, {x}) needs 0 <= x <= 1.")
        object.__setattr__(self, "points", points)


def _as_region_data(region: Any) -> RegionData | None:
    """Return ``region`` as :class:`RegionData`.

    Accepts ``None``, :class:`RegionData`, or any object with an ``intervals``
    iterable of ``(branch_index, prox, dist)`` (such as an evaluated region
    mask of a simulator).

    Raises
    ------
    TypeError
        If ``region`` has no ``intervals``.
    """
    if region is None or isinstance(region, RegionData):
        return region
    intervals = getattr(region, "intervals", None)
    if intervals is None:
        raise TypeError(
            f"region must be RegionData or expose .intervals (an evaluated mask); got {type(region).__name__}."
        )
    return RegionData(tuple(intervals))


def _as_locset_data(locset: Any) -> LocsetData | None:
    """Return ``locset`` as :class:`LocsetData`.

    Accepts ``None``, :class:`LocsetData`, or any object with a ``points``
    iterable of ``(branch_index, x)``.

    Raises
    ------
    TypeError
        If ``locset`` has no ``points``.
    """
    if locset is None or isinstance(locset, LocsetData):
        return locset
    points = getattr(locset, "points", None)
    if points is None:
        raise TypeError(
            f"locset must be LocsetData or expose .points (an evaluated mask); got {type(locset).__name__}."
        )
    return LocsetData(tuple((p[0], p[1]) for p in points))


@dataclasses.dataclass(frozen=True)
class TopologyGraph:
    """A rooted graph with optional per-node values or highlight fractions.

    The input of :func:`plot_topology_graph`. Nodes are integer identifiers
    (for example branch indices, control-volume IDs or solver-node IDs).

    Parameters
    ----------
    node_ids : tuple of int
        Node identifiers; ``values`` is aligned with this order.
    edges : tuple of (int, int)
        ``(parent_id, child_id)`` pairs.
    root_id : int
        Identifier of the root node.
    level : str, optional
        What a node represents, ``"node"`` (default), ``"cv"`` or
        ``"branch"``. ``"branch"`` graphs are topology-only:
        :func:`plot_topology_graph` rejects colour-map arguments for them.
    values : array_like or brainunit.Quantity, optional
        Per-node values, shape ``(N,)`` or ``(T, N)`` for ``T`` frames.
        ``NaN`` marks nodes without a value.
    value_label : str, optional
        Default colour-bar label.
    highlight_fractions : mapping of int to float, optional
        Node identifier to highlight intensity in ``[0, 1]``. Mutually
        exclusive with ``values``.
    times : brainunit.Quantity, optional
        Frame times, shape ``(T,)``; only with two-dimensional ``values``.

    Raises
    ------
    ValueError
        If node identifiers repeat, an edge or the root names an unknown
        node, ``values`` does not match ``len(node_ids)``, ``values`` and
        ``highlight_fractions`` are both given, or ``times`` does not match
        the frame count.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.visualize.morphology import TopologyGraph
        >>> graph = TopologyGraph(
        ...     node_ids=(0, 1, 2), edges=((0, 1), (0, 2)), root_id=0,
        ...     values=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        ... )
        >>> graph.n_frames
        2
        >>> graph.frame(1)
        array([2., 3., 4.])
    """

    node_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    root_id: int
    level: str = "node"
    values: Any = None
    value_label: str | None = None
    highlight_fractions: Mapping[int, float] | None = None
    times: Any = None

    def __post_init__(self) -> None:
        node_ids = tuple(int(i) for i in self.node_ids)
        if len(node_ids) == 0:
            raise ValueError("TopologyGraph needs at least one node.")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("TopologyGraph.node_ids must be unique.")
        known = set(node_ids)
        edges = tuple((int(p), int(c)) for p, c in self.edges)
        for p, c in edges:
            if p not in known or c not in known:
                raise ValueError(
                    f"TopologyGraph edge ({p}, {c}) names a node that is not in node_ids."
                )
        root_id = int(self.root_id)
        if root_id not in known:
            raise ValueError(f"TopologyGraph.root_id {root_id} is not in node_ids.")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "root_id", root_id)
        if self.values is not None:
            if self.highlight_fractions is not None:
                raise ValueError(
                    "TopologyGraph accepts values or highlight_fractions, not both."
                )
            shape = np.shape(u.get_mantissa(self.values))
            if len(shape) not in (1, 2) or shape[-1] != len(node_ids):
                raise ValueError(
                    f"TopologyGraph.values must have shape ({len(node_ids)},) or (T, {len(node_ids)}), got {shape}."
                )
            if self.times is not None:
                if len(shape) != 2 or np.shape(u.get_mantissa(self.times)) != (
                    shape[0],
                ):
                    raise ValueError(
                        "TopologyGraph.times must have one entry per frame of two-dimensional values."
                    )
        elif self.times is not None:
            raise ValueError("TopologyGraph.times needs two-dimensional values.")
        if self.highlight_fractions is not None:
            fractions = {
                int(k): float(v) for k, v in dict(self.highlight_fractions).items()
            }
            for key, value in fractions.items():
                if key not in known:
                    raise ValueError(
                        f"TopologyGraph.highlight_fractions names unknown node {key}."
                    )
                if not (0.0 <= value <= 1.0) or math.isnan(value):
                    raise ValueError(
                        f"TopologyGraph.highlight_fractions[{key}] must be in [0, 1], got {value}."
                    )
            object.__setattr__(self, "highlight_fractions", fractions)

    @property
    def n_frames(self) -> int:
        """Number of value frames: 0 without values, 1 for ``(N,)`` values."""
        if self.values is None:
            return 0
        return (
            1
            if np.ndim(u.get_mantissa(self.values)) == 1
            else int(np.shape(u.get_mantissa(self.values))[0])
        )

    def frame(self, index: int | None = None) -> Any:
        """Return the ``(N,)`` values of one frame.

        Parameters
        ----------
        index : int or None
            Frame index; must be given when ``values`` is two-dimensional.

        Raises
        ------
        ValueError
            If ``values`` is two-dimensional and ``index`` is ``None``.
        """
        values = self.values
        if values is None or np.ndim(u.get_mantissa(values)) == 1:
            return values
        if index is None:
            raise ValueError(
                f"TopologyGraph has {self.n_frames} value frames; pass frame=<index>."
            )
        return values[index]

    def __topology_data__(self) -> "TopologyGraph":
        return self


def as_morphology_data(obj: Any) -> MorphologyData:
    """Return ``obj`` as :class:`MorphologyData`.

    Parameters
    ----------
    obj : MorphologyData or object
        The data itself, or an object whose ``__morphology_data__()`` method
        returns it (for example a simulator's morphology).

    Raises
    ------
    TypeError
        If ``obj`` is neither, or its ``__morphology_data__()`` does not
        return :class:`MorphologyData`.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.visualize.morphology import as_morphology_data
        >>> as_morphology_data(42)
        Traceback (most recent call last):
        ...
        TypeError: expected MorphologyData or an object with __morphology_data__(), got int.
    """
    if isinstance(obj, MorphologyData):
        return obj
    convert = getattr(type(obj), "__morphology_data__", None)
    if convert is None:
        raise TypeError(
            f"expected MorphologyData or an object with __morphology_data__(), got {type(obj).__name__}."
        )
    data = convert(obj)
    if not isinstance(data, MorphologyData):
        raise TypeError(
            f"{type(obj).__name__}.__morphology_data__() must return MorphologyData, got {type(data).__name__}."
        )
    return data


def as_topology_graph(obj: Any) -> TopologyGraph:
    """Return ``obj`` as :class:`TopologyGraph`.

    Parameters
    ----------
    obj : TopologyGraph or object
        The graph itself, or an object whose ``__topology_data__()`` method
        returns it (for example a simulator's solver-node tree).

    Raises
    ------
    TypeError
        If ``obj`` is neither, or its ``__topology_data__()`` does not
        return :class:`TopologyGraph`.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.visualize.morphology import TopologyGraph, as_topology_graph
        >>> graph = TopologyGraph(node_ids=(0, 1), edges=((0, 1),), root_id=0)
        >>> as_topology_graph(graph) is graph
        True
    """
    if isinstance(obj, TopologyGraph):
        return obj
    convert = getattr(type(obj), "__topology_data__", None)
    if convert is None:
        raise TypeError(
            f"expected TopologyGraph or an object with __topology_data__(), got {type(obj).__name__}."
        )
    graph = convert(obj)
    if not isinstance(graph, TopologyGraph):
        raise TypeError(
            f"{type(obj).__name__}.__topology_data__() must return TopologyGraph, got {type(graph).__name__}."
        )
    return graph
