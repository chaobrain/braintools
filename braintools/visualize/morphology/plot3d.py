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


from braintools.visualize.morphology.backend import (
    BackendChooser,
    validate_backend_for_scene,
)
from braintools.visualize.morphology.config import (
    SUPPORTED_3D_MODES,
    resolve_default_3d_mode,
)
from braintools.visualize.morphology.hooks import VisHooks
from braintools.visualize.morphology.plot2d import _build_value_spec
from braintools.visualize.morphology.scene import OverlaySpec, RenderRequest
from braintools.visualize.morphology.scene3d import build_render_scene_3d
from braintools.visualize.morphology._tree import as_tree


def plot3d(
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
    mode: str | None = None,
    backend: str | None = None,
    chooser: BackendChooser | None = None,
    notebook: bool | None = None,
    jupyter_backend: str | None = None,
    return_plotter: bool = False,
    hooks: VisHooks | None = None,
) -> object:
    """Render a morphology in 3-D with PyVista or Plotly.

    Every branch must carry point geometry
    (:attr:`MorphologyData.has_full_point_geometry`).

    Parameters
    ----------
    morpho : MorphologyData-like
        A :class:`MorphologyData` or any object implementing
        ``__morphology_data__()``.
    region, locset, values, cmap, vmin, vmax, norm, value_label, show_colorbar
        Overlays and colour mapping, as in :func:`plot2d`.
    mode : {"geometry", "skeleton"}, optional
        Tubes with the branch radii, or thin centerlines. Defaults to
        :attr:`VisDefaults.mode_3d_default`.
    backend : {"pyvista", "plotly"}, optional
        Defaults to the first one installed.
    chooser : BackendChooser, optional
        Explicit backend chooser.
    notebook : bool, optional
        Embed the scene in the notebook cell output.
    jupyter_backend : str, optional
        PyVista Jupyter backend, such as ``"html"`` for static documents.
    return_plotter : bool
        Return the PyVista plotter or Plotly figure instead of showing it.
    hooks : VisHooks, optional
        Pick callbacks.

    Returns
    -------
    object
        The backend's plotter or figure.

    Raises
    ------
    ValueError
        If some branch lacks point geometry or ``mode`` is not recognised.
    RuntimeError
        If neither PyVista nor Plotly is installed
        (``pip install braintools[morphology]``).

    See Also
    --------
    plot2d : 2-D drawing with Matplotlib.
    braintools.visualize.dendrite_tree_3d : Plotly tree of explicit coordinates.

    Examples
    --------

    .. code-block:: python

        >>> from braintools.visualize.morphology import plot3d
        >>> plotter = plot3d(data, mode="skeleton", return_plotter=True)  # doctest: +SKIP
    """
    morpho = as_tree(morpho, caller="plot3d(...)")
    resolved_mode = resolve_default_3d_mode(mode)
    if resolved_mode not in SUPPORTED_3D_MODES:
        expected = ", ".join(sorted(repr(item) for item in SUPPORTED_3D_MODES))
        raise ValueError(
            f"Unsupported 3D mode {resolved_mode!r}. Expected one of {expected}."
        )

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
    scene = build_render_scene_3d(morpho, mode=resolved_mode, overlay=overlay)
    chooser = chooser or BackendChooser.default()
    backend_options: dict = {}
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
        dimensionality="3d",
        mode=resolved_mode,
        backend_options=backend_options,
    )
    backend_impl = chooser.pick(requested=backend, scene_kind="3d")
    validate_backend_for_scene(backend_impl, scene)
    return backend_impl.render(request)
