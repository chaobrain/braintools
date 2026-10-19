# Visualization

`braintools.visualize` provides plotting functions for neural data and models: static Matplotlib figures for spiking data, networks, statistics and model evaluation, 3-D views, interactive Plotly figures, animations, and styles, palettes and colormaps.

## What you'll find here

- **Spiking data**: rasters from spike times or simulation matrices, population rates and PSTHs, inter-spike intervals and their CV, spatial rate maps, and membrane traces.
- **Networks, tuning and dynamics**: signed connectivity matrices and graphs, correlation matrices, tuning curves with Gaussian fits, phase portraits with vector fields, and population trajectories in principal-component space.
- **Statistical plots**: distributions and Q-Q plots, box versus violin plots, scatter matrices, Pearson versus rank correlations, and regression with residual diagnostics.
- **Evaluating models and decoders**: confusion matrices, ROC and precision–recall curves under class imbalance, learning curves, and heteroscedastic residuals.
- **Interactive figures**: Plotly versions of rasters, traces, heatmaps, histograms, networks, 3-D views, and a one-call dashboard.
- **Three-dimensional plots**: state-space trajectories, layered networks, projections between populations, electrode arrays, dendritic morphology, surfaces and volumes.
- **Styles, palettes and colormaps**: applying styles without side effects, color-blind-safe palettes checked by simulation, and choosing colormaps by their lightness profile.
- **Animations**: moving bumps, waves and spike images with `animate_1D`, `animate_2D` and `animator`, displayed in notebooks and saved as GIFs.

For the full list of functions and parameters, see the {doc}`API reference <../apis/visualize>`.

```{toctree}
:maxdepth: 1

01_neural_data_visualization.ipynb
02_advanced_neural_plots.ipynb
03_statistical_visualization.ipynb
04_model_evaluation_plots.ipynb
05_interactive_visualization.ipynb
06_3d_visualization.ipynb
07_styling_and_themes.ipynb
08_animation_and_dynamics.ipynb
```
