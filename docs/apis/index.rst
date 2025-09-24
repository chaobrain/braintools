API Documentation
=================

BrainTools is a collection of computational neuroscience tools and utilities designed for brain simulation and analysis. 
This documentation provides comprehensive reference for all modules and functions.

Core Modules
------------

**braintools.quad** - Lightweight numerical integrators for differential equations
   Provides JAX-friendly stepping functions for ordinary differential equations (ODEs), 
   stochastic differential equations (SDEs), and implicit-explicit (IMEX) methods.
   Includes Euler, Runge-Kutta, and specialized neuroscience-oriented integrators.

**braintools.metric** - Analysis and evaluation metrics for neural data
   Comprehensive collection of metrics for analyzing neural activity including 
   classification metrics, correlation analysis, firing rate metrics, LFP analysis,
   ranking methods, regression metrics, and data smoothing techniques.

**braintools.optim** - Optimization algorithms and utilities
   Optimization tools built on SciPy and Nevergrad backends for parameter tuning,
   model fitting, and numerical optimization tasks in computational neuroscience.

**braintools.input** - Current injection and stimulus generation
   Methods for generating various types of current inputs and stimuli for neural
   simulations, including step currents, ramps, noise, and periodic signals.

**braintools.tree** - PyTree manipulation utilities
   Utilities for working with JAX PyTrees including scaling, arithmetic operations,
   concatenation, splitting, and conversion functions for nested data structures.

**braintools.visualize** - Data visualization and plotting utilities
   Comprehensive visualization toolkit for neural data including spike rasters,
   population activity plots, connectivity visualizations, statistical plots,
   interactive visualizations, 3D rendering, and publication-ready styling.

.. toctree::
   :maxdepth: 2

   changelog.md
   braintools.rst
   quad.rst
   metric.rst
   optim.rst
   input.rst
   tree.rst
   visualize.rst

