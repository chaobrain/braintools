``braintools.visualize`` module
===============================

.. currentmodule:: braintools.visualize
.. automodule:: braintools.visualize

The visualization toolkit spans quick exploratory plots, rich publication
figures, and interactive dashboards tailored to neural data analysis. The
sections below outline the main families of helpers and chart builders.

Neural Visualization
--------------------

High-level helpers that focus on spike trains, trajectories, and other neural
recordings.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   spike_raster
   population_activity
   connectivity_matrix
   neural_trajectory
   spike_histogram
   isi_distribution
   firing_rate_map
   phase_portrait
   network_topology
   tuning_curve


Statistical Visualization
-------------------------

Utility plots to inspect distributions, correlations, and model evaluation
metrics derived from experiments.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   correlation_matrix
   distribution_plot
   qq_plot
   box_plot
   violin_plot
   scatter_matrix
   regression_plot
   residual_plot
   confusion_matrix
   roc_curve
   precision_recall_curve
   learning_curve


Interactive Visualization
-------------------------

Widget-backed tools for exploratory analysis and dashboards that support live
updates or user interaction.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   interactive_spike_raster
   interactive_line_plot
   interactive_heatmap
   interactive_3d_scatter
   interactive_network
   dashboard_neural_activity


3D Visualization
----------------

Renderers and figure factories optimized for volumetric or spatial datasets.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   neural_network_3d
   brain_surface_3d
   connectivity_3d
   trajectory_3d
   volume_rendering
   electrode_array_3d
   dendrite_tree_3d
   phase_space_3d


Basic Plotting
--------------

Lightweight wrappers around matplotlib primitives for quick inspection.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   line_plot
   raster_plot
   animate_1D
   animate_2D


Styling and Colormaps
---------------------

Functions for consistent theming, color palettes, and style presets across
figures.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   neural_style
   publication_style
   dark_style
   colorblind_friendly_style
   create_neural_colormap
   brain_colormaps
   apply_style
   get_color_palette
   set_default_colors


Figure Utilities
----------------

Low-level helpers to obtain figure handles or tweak aesthetics pre/post
rendering.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   get_figure
   remove_axis
