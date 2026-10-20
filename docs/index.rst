``braintools`` documentation
============================

`braintools <https://github.com/chaobrain/braintools>`_ implements a modern toolbox for brain simulation and analysis.


Overview
^^^^^^^^

``braintools`` is a lightweight, JAX-friendly collection of utilities used across computational neuroscience workflows:

- Declarative, unit-aware synaptic connectivity builders (:mod:`braintools.conn`) for point neurons, multi-compartment morphologies, and convolution kernels, population networks built from populations and projections, memory-bounded generation for large networks, and export to ``brainevent`` sparse formats; parameter initializers and distance profiles are in :mod:`braintools.init`
- Spike data utilities (:mod:`braintools.spike`): the ``SpikeTrains`` container (a padded, ``jit``/``vmap``-friendly spike-time layout with dense/event conversions), spike encoders, and bitwise spike operations
- Neural data analysis (:mod:`braintools.analysis`): spike-train statistics (ISI/CV/Fano, PSTHs, correlograms, spike-train distances, synchrony), spectral and LFP analysis (multitaper and wavelet spectra, filtering, phase synchrony, cross-frequency coupling, current source density), population analysis (PCA, decoding, Granger causality, avalanches, functional connectivity), structural graph measures, and resampling-based statistics
- A composable **cognitive task framework** (:mod:`braintools.cogtask`) for constructing trial-structured tasks — decision making, working memory, reasoning, motor — with JIT/``vmap``-friendly batch sampling and a registry of pre-built paradigms
- Numerical integration utilities for ODE/SDE/DDE/IMEX one-step steppers (:mod:`braintools.quad`) designed for PyTrees and functional APIs, with a trajectory driver (``odeint``) and a differentiable adaptive solver (``solve_ivp``)
- Model fitting (:mod:`braintools.fit`): composable objectives, summary-statistic losses, vmapped multi-start fitting of functions or ``brainstate`` modules, parameter sweeps, and confidence intervals
- Losses and metrics for model training (:mod:`braintools.metric`): classification, regression, ranking, and pairwise similarity
- Visualization helpers (:mod:`braintools.visualize`), input generators (:mod:`braintools.input`), surrogate gradients (:mod:`braintools.surrogate`), and PyTree utilities (:mod:`braintools.tree`)
- Optimizer wrappers (:mod:`braintools.optim`) including a multi-start minimizer, online readout learning (RLS, LMS, FORCE) and closed-form regressions, and a lightweight training loop (:mod:`braintools.trainer`)
- Data files (:mod:`braintools.file`): checkpoints, MATLAB files, and optional HDF5, NWB, and NEO formats

The project favors a simple, well-typed functional style that works seamlessly with `brainstate <https://brainx.chaobrain.com/brainstate/>`_,
`brainunit <https://brainx.chaobrain.com/brainunit/>`_, and just-in-time compilation (``jit``/``vmap``).


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U braintools[cpu]


    .. tab-item:: GPU

       .. code-block:: bash

          pip install -U braintools[cuda12]
          pip install -U braintools[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U braintools[tpu]


Upgrading to 1.0
^^^^^^^^^^^^^^^^

Names that moved since 0.4 still work but emit a ``BraintoolsDeprecationWarning``
(a ``FutureWarning``, shown by default); version 1.0 removes them. The
:doc:`migration guide <migration>` lists every deprecated name with its replacement.


----

See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^

BrainTools is one part of our `brain simulation ecosystem <https://brainx.chaobrain.com/>`_.


----

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Tutorials

    analysis/index.md
    cogtask/index.md
    conn/index.md
    file/index.md
    fit/index.md
    init/index.md
    input/index.md
    metric/index.md
    optim/index.md
    quad/index.md
    spike/index.md
    surrogate/index.md
    trainer/index.md
    visualize/index.md
    misc/index.md
    migration.md


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: API Reference

    apis/changelog.md
    apis/braintools.rst
    apis/analysis.rst
    apis/cogtask.rst
    apis/conn.rst
    apis/file.rst
    apis/fit.rst
    apis/init.rst
    apis/input.rst
    apis/metric.rst
    apis/optim.rst
    apis/quad.rst
    apis/spike.rst
    apis/surrogate.rst
    apis/trainer.rst
    apis/tree.rst
    apis/visualize.rst
