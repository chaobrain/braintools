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


from contextlib import nullcontext

from braintools.visualize.morphology.backend import (
    BackendChooser,
    validate_backend_for_scene,
)
from braintools.visualize.morphology.config import (
    resolve_default_2d_layout,
    resolve_default_2d_shape,
    theme as vis_theme,
)
from braintools.visualize.morphology.hooks import VisHooks
from braintools.visualize.morphology.layout import LayoutConfig
from braintools.visualize.morphology.scene import OverlaySpec, RenderRequest, ValueSpec
from braintools.visualize.morphology.scene2d import build_render_scene_2d
from braintools.visualize.morphology._tree import as_tree


def plot2d(
    morpho,
    *,
    region=None,
    locset=None,
    values=None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
    value_label: str | None = None,
    show_colorbar: bool = True,
    layout: str | None = None,
    shape: str | None = None,
    branch_type_colors: dict[str, object] | None = None,
    branch_type_edge_colors_2d: dict[str, object] | None = None,
    frustum_edge_linewidth_2d: float | None = None,
    backend: str | None = None,
    chooser: BackendChooser | None = None,
    ax=None,
    notebook: bool | None = None,
    jupyter_backend: str | None = None,
    return_plotter: bool = False,
    projection_plane: str = "xy",
    min_branch_angle_deg: float | None = 25.0,
    root_layout: str = "type_split",
    layout_config: LayoutConfig | None = None,
    hooks: VisHooks | None = None,
) -> object:
    """Draw a morphology in 2-D with Matplotlib.

    ``layout`` decides where the branches go and ``shape`` how they are drawn.
    ``"projected"`` drops one axis of the 3-D points; ``"fan"``, ``"stem"``,
    ``"balloon"`` and ``"radial_360"`` are schematic layouts computed from
    branch lengths and topology, so they also work for morphologies without
    point geometry.

    Parameters
    ----------
    morpho : MorphologyData-like
        A :class:`MorphologyData` or any object implementing
        ``__morphology_data__()``.
    region : RegionData-like, optional
        Intervals ``(branch_index, prox, dist)`` to highlight.
    locset : LocsetData-like, optional
        Points ``(branch_index, x)`` to mark.
    values : array-like, optional
        Scalars to colour-map, one per branch, per segment or per centerline
        point (inferred from the length; pass a :class:`ValueSpec` with
        ``space`` to state it).
    cmap, vmin, vmax, norm, value_label, show_colorbar
        Colour-mapping settings for ``values``; see :class:`ValueSpec`.
    layout : {"fan", "stem", "balloon", "radial_360", "projected"}, optional
        Defaults to :attr:`VisDefaults.layout_2d_default` (``"fan"``).
    shape : {"frustum", "line"}, optional
        Defaults to :attr:`VisDefaults.shape_2d_default` (``"frustum"``).
        ``"projected"`` supports ``"line"`` only.
    branch_type_colors, branch_type_edge_colors_2d, frustum_edge_linewidth_2d
        Style overrides for this call only.
    backend : str, optional
        Backend name; ``"matplotlib"`` is the only 2-D backend.
    chooser : BackendChooser, optional
        Explicit backend chooser.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created otherwise.
    notebook, jupyter_backend, return_plotter
        Display options forwarded to the backend.
    projection_plane : {"xy", "xz", "yz"}
        Axis pair for ``layout="projected"``.
    min_branch_angle_deg : float or None
        Minimum angle between sibling branches in schematic layouts.
    root_layout : {"type_split"}
        How the root's children are arranged in schematic layouts.
    layout_config : LayoutConfig, optional
        Fine-grained layout parameters.
    hooks : VisHooks, optional
        Pick callbacks.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.

    Raises
    ------
    TypeError
        If ``morpho`` is not morphology data.
    ValueError
        If ``layout="projected"`` is used without full point geometry or with
        ``shape="frustum"``, or if an option is not recognised.

    See Also
    --------
    plot3d : 3-D rendering with PyVista or Plotly.
    compare_morphologies : Several morphologies side by side.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braintools.visualize.morphology import BranchData, MorphologyData, plot2d
        >>> soma = BranchData(
        ...     index=0, name="soma", type="soma", parent_index=None,
        ...     parent_x=None, child_x=None, lengths=[20.0] * u.um,
        ...     radii_proximal=[10.0] * u.um, radii_distal=[10.0] * u.um,
        ... )
        >>> dend = BranchData(
        ...     index=1, name="dend", type="basal_dendrite", parent_index=0,
        ...     parent_x=1.0, child_x=0.0, lengths=[50.0, 50.0] * u.um,
        ...     radii_proximal=[1.0, 0.8] * u.um, radii_distal=[0.8, 0.5] * u.um,
        ... )
        >>> ax = plot2d(MorphologyData(branches=(soma, dend)), layout="stem")
    """
    morpho = as_tree(morpho, caller="plot2d(...)")

    resolved_layout = resolve_default_2d_layout(layout)
    resolved_shape = resolve_default_2d_shape(shape)
    values_spec = _build_value_spec(
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        value_label=value_label,
        show_colorbar=show_colorbar,
    )
    overlay = OverlaySpec(region=region, locset=locset, values=values_spec)
    style_overrides = _build_style_overrides(
        branch_type_colors=branch_type_colors,
        branch_type_edge_colors_2d=branch_type_edge_colors_2d,
        frustum_edge_linewidth_2d=frustum_edge_linewidth_2d,
    )
    with vis_theme(**style_overrides) if style_overrides else nullcontext():
        scene = build_render_scene_2d(
            morpho,
            layout=resolved_layout,
            shape=resolved_shape,
            projection_plane=projection_plane,
            min_branch_angle_deg=min_branch_angle_deg,
            root_layout=root_layout,
            overlay=overlay,
            layout_config=layout_config,
        )
    chooser = chooser or BackendChooser.default()
    backend_options: dict = {}
    if ax is not None:
        backend_options["ax"] = ax
    if notebook is not None:
        backend_options["notebook"] = notebook
    if jupyter_backend is not None:
        backend_options["jupyter_backend"] = jupyter_backend
    if return_plotter:
        backend_options["return_plotter"] = True
    if hooks is not None:
        backend_options["hooks"] = hooks
    request = RenderRequest(
        morpho=morpho,
        scene=scene,
        overlay=overlay,
        dimensionality="2d",
        layout=resolved_layout,
        shape=resolved_shape,
        backend_options=backend_options,
    )
    backend_impl = chooser.pick(requested=backend, scene_kind="2d")
    validate_backend_for_scene(backend_impl, scene)
    return backend_impl.render(request)


def _build_style_overrides(
    *,
    branch_type_colors: dict[str, object] | None,
    branch_type_edge_colors_2d: dict[str, object] | None,
    frustum_edge_linewidth_2d: float | None,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if branch_type_colors is not None:
        overrides["branch_type_colors"] = dict(branch_type_colors)
    if branch_type_edge_colors_2d is not None:
        overrides["branch_type_edge_colors_2d"] = dict(branch_type_edge_colors_2d)
    if frustum_edge_linewidth_2d is not None:
        overrides["frustum_edge_linewidth_2d"] = frustum_edge_linewidth_2d
    return overrides


def _build_value_spec(
    values,
    *,
    cmap: str | None,
    vmin: float | None,
    vmax: float | None,
    norm,
    value_label: str | None,
    show_colorbar: bool,
) -> ValueSpec | None:
    """Normalize ``plot2d``/``plot3d`` value keywords into a :class:`ValueSpec`."""
    if values is None:
        if any(k is not None for k in (cmap, vmin, vmax, norm, value_label)):
            raise ValueError(
                "values=... is required when passing cmap/vmin/vmax/norm/value_label."
            )
        return None
    if isinstance(values, ValueSpec):
        # Fold caller-supplied style kwargs on top of the spec. A kwarg
        # that's None leaves the spec field untouched.
        return ValueSpec(
            values=values.values,
            cmap=cmap if cmap is not None else values.cmap,
            vmin=vmin if vmin is not None else values.vmin,
            vmax=vmax if vmax is not None else values.vmax,
            norm=norm if norm is not None else values.norm,
            label=value_label if value_label is not None else values.label,
            unit_label=values.unit_label,
            show_colorbar=show_colorbar,
        )
    return ValueSpec(
        values=values,
        cmap=cmap if cmap is not None else "viridis",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        label=value_label,
        show_colorbar=show_colorbar,
    )
