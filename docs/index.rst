``braintools`` documentation
============================

`braintools <https://github.com/chaobrain/braintools>`_ implements a modern toolbox for brain simulation and analysis.


Overview
^^^^^^^^

``braintools`` is a lightweight, JAX-friendly collection of utilities used across computational neuroscience workflows:

- Declarative, unit-aware synaptic connectivity builders (:mod:`braintools.conn`) for point neurons, multi-compartment morphologies, and convolution kernels, with parameter initializers and distance profiles in :mod:`braintools.init`
- Spike data utilities (:mod:`braintools.spike`): the ``SpikeTrains`` container (a padded, ``jit``/``vmap``-friendly spike-time layout with dense/event conversions), spike encoders, and bitwise spike operations
- Neural data analysis (:mod:`braintools.analysis`): spike-train statistics (firing rates, distances, synchrony), spectral and LFP analysis (power spectra, coherence, cross-frequency coupling, current source density), and population correlations and functional connectivity
- A composable **cognitive task framework** (:mod:`braintools.cogtask`) for constructing trial-structured tasks — decision making, working memory, reasoning, motor — with JIT/``vmap``-friendly batch sampling and a registry of pre-built paradigms
- Numerical integration utilities for ODE/SDE/DDE/IMEX one-step steppers (:mod:`braintools.quad`) designed for PyTrees and functional APIs
- Losses and metrics for model training (:mod:`braintools.metric`): classification, regression, ranking, and pairwise similarity
- Visualization helpers (:mod:`braintools.visualize`), input generators (:mod:`braintools.input`), surrogate gradients (:mod:`braintools.surrogate`), and PyTree utilities (:mod:`braintools.tree`)
- Optimizer wrappers (:mod:`braintools.optim`) and a lightweight training loop (:mod:`braintools.trainer`)

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
    init/index.md
    input/index.md
    metric/index.md
    optim/index.md
    quad/index.md
    surrogate/index.md
    visualize/index.md
    misc/spike_encoding.md


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
