``braintools.visualize.morphology`` module
==========================================

.. currentmodule:: braintools.visualize.morphology
.. automodule:: braintools.visualize.morphology

Plots of neuron morphologies, discrete topology graphs (branches, control
volumes, solver nodes) and values over them. The plots never read a
simulator's objects: they take the neutral, frozen data types below, or any
object implementing ``__morphology_data__()`` / ``__topology_data__()``, so a
simulator only has to convert its own objects. This module is not imported
by ``import braintools``; import it explicitly:

.. code-block:: python

    import braintools.visualize.morphology as bvm

Matplotlib draws every 2-D plot. Topology-graph layouts need networkx, and
3-D rendering needs PyVista or Plotly; install them with
``pip install braintools[morphology]``.


Data
----

Geometry is stored in micrometres as read-only host arrays. Branches are
identified by ``index`` (their position) and ``name`` (unique, and stable
across edits of the source morphology); graph nodes by integer identifiers.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BranchData
   MorphologyData
   RegionData
   LocsetData
   TopologyGraph
   as_morphology_data
   as_topology_graph


Morphology plots
----------------

2-D layouts (``fan``, ``stem``, ``balloon``, ``radial_360``, ``projected``),
3-D rendering, comparisons, time series and animations. Values are given per
branch, per segment or per centerline point; :class:`ValueSpec` carries the
colour settings and, optionally, the value space.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   plot2d
   plot3d
   compare_morphologies
   compare_values
   plot_traces
   TracesResult
   plot_movie
   MovieResult
   ValueSpec
   OverlaySpec


Structure
---------

Tree schematics and morphometry of a morphology, and the discrete topology
graph drawn with a graph layout (unlike :func:`braintools.visualize.network_topology`,
which draws an adjacency matrix).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   plot_dendrogram
   plot_topology
   plot_sholl
   plot_branch_order_histogram
   plot_topology_graph


Layout, style and interaction
-----------------------------

Layout parameters and the layout cache, global plot defaults and themes
(``publication_theme`` sets both the morphology colours and Matplotlib
rcParams; :func:`braintools.visualize.publication_style` sets rcParams
only), pick callbacks, and figure export for every backend.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LayoutConfig
   LayoutCache
   VisDefaults
   configure_defaults
   get_defaults
   reset_defaults
   theme
   publication_theme
   PublicationTheme
   VisHooks
   PickInfo
   save_figure


Constants
---------

.. data:: PUBLICATION_BRANCH_TYPE_COLORS

   Branch-type colours (RGB, 0-255) of :func:`publication_theme`, from Paul
   Tol's colour-blind-safe palette; the keys are the branch types (``"soma"``,
   ``"axon"``, ``"basal_dendrite"``, ``"apical_dendrite"``, ``"dendrite"``,
   ``"custom"``).

.. data:: PUBLICATION_RC_PARAMS

   Matplotlib ``rcParams`` applied by :func:`publication_theme`: serif font,
   thicker lines, no grid and tight margins, tuned for export at 300 dpi.
