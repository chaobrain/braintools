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

"""Morphology, topology and value plots driven by neutral data.

The plots take :class:`MorphologyData` (a tree of :class:`BranchData`) or
:class:`TopologyGraph`, or any object implementing ``__morphology_data__()``
/ ``__topology_data__()``; a simulator converts its own objects, so this
package imports none. It is not imported by ``import braintools``:

.. code-block:: python

    import braintools.visualize.morphology as bvm

Matplotlib renders the 2-D plots. networkx (topology-graph layouts),
PyVista and Plotly (3-D) are optional: ``pip install braintools[morphology]``.
"""

from braintools.visualize.morphology._data import (
    BranchData,
    LocsetData,
    MorphologyData,
    RegionData,
    TopologyGraph,
    as_morphology_data,
    as_topology_graph,
)
from braintools.visualize.morphology.compare import compare_morphologies, compare_values
from braintools.visualize.morphology.config import (
    PUBLICATION_BRANCH_TYPE_COLORS,
    PUBLICATION_RC_PARAMS,
    PublicationTheme,
    VisDefaults,
    configure as configure_defaults,
    get_defaults,
    publication_theme,
    reset_defaults,
    theme,
)
from braintools.visualize.morphology.export import save_figure
from braintools.visualize.morphology.hooks import PickInfo, VisHooks
from braintools.visualize.morphology.layout import LayoutCache, LayoutConfig
from braintools.visualize.morphology.morphometry import (
    plot_branch_order_histogram,
    plot_dendrogram,
    plot_sholl,
    plot_topology,
)
from braintools.visualize.morphology.movie import MovieResult, plot_movie
from braintools.visualize.morphology.plot2d import plot2d
from braintools.visualize.morphology.plot3d import plot3d
from braintools.visualize.morphology.point_topology import plot_topology_graph
from braintools.visualize.morphology.scene import OverlaySpec, ValueSpec
from braintools.visualize.morphology.traces import TracesResult, plot_traces

__all__ = [
    "BranchData",
    "LayoutCache",
    "LayoutConfig",
    "LocsetData",
    "MorphologyData",
    "MovieResult",
    "OverlaySpec",
    "PUBLICATION_BRANCH_TYPE_COLORS",
    "PUBLICATION_RC_PARAMS",
    "PickInfo",
    "PublicationTheme",
    "RegionData",
    "TopologyGraph",
    "TracesResult",
    "ValueSpec",
    "VisDefaults",
    "VisHooks",
    "as_morphology_data",
    "as_topology_graph",
    "compare_morphologies",
    "compare_values",
    "configure_defaults",
    "get_defaults",
    "plot2d",
    "plot3d",
    "plot_branch_order_histogram",
    "plot_dendrogram",
    "plot_movie",
    "plot_sholl",
    "plot_topology",
    "plot_topology_graph",
    "plot_traces",
    "publication_theme",
    "reset_defaults",
    "save_figure",
    "theme",
]
