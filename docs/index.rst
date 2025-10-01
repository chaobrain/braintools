``braintools`` documentation
============================

`braintools <https://github.com/chaobrain/braintools>`_ implements a modern toolbox for brain simulation and analysis.


Overview
^^^^^^^^

``braintools`` is a lightweight, JAX-friendly collection of utilities used across computational neuroscience workflows:

- Composable synaptic connectivity builders (:mod:`braintools.conn`) with point, compartment, and population abstractions
- Rich visualization helpers (:mod:`braintools.visualize`) covering static plots, interactive dashboards, and animation utilities
- Metric functions for model training and evaluation (classification, regression, ranking, correlation, LFP helpers)
- Numerical integration utilities for ODE/SDE/DDE one-step steppers (:mod:`braintools.quad`) designed for PyTrees and functional APIs
- Input generators, optimization helpers, and reusable data-structure utilities
- And more ...

The project favors a simple, well-typed functional style that works seamlessly with `brainstate <https://brainstate.readthedocs.io/>`_,
`brainunit <https://brainunit.readthedocs.io/>`_, and just-in-time compilation (``jit``/``vmap``).


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

BrainTools is one part of our `brain simulation ecosystem <https://brainmodeling.readthedocs.io/>`_.


----

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Tutorials

    toc_conn.md
    toc_file.md
    toc_input.md
    toc_metric.md
    toc_optim.md
    toc_quad.md
    toc_visualize.md
    toc_spike_encoding.md


.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: API Reference

    apis/changelog.md
    apis/braintools.rst
    apis/conn.rst
    apis/file.rst
    apis/input.rst
    apis/metric.rst
    apis/optim.rst
    apis/quad.rst
    apis/visualize.rst
