``braintools`` documentation
============================

`braintools <https://github.com/brainpy/braintools>`_ implements the common toolboxes for brain simulation.

----


Overview
^^^^^^^^

``braintools`` is a lightweight, JAX‑friendly toolbox that collects practical
utilities used throughout computational neuroscience workflows:

- Metric functions for model training and evaluation (classification,
  regression, ranking, correlation/connectivity, LFP helpers)
- Numerical integration utilities for ODE/SDE/DDE one‑step steppers (under
  :mod:`braintools.quad`), designed for PyTrees and functional APIs
- Input generators and small optimization helpers to quickly prototype and test
  models
- And more to come...

The project favors a simple, well‑typed functional style that plays nicely with
JIT/VMAP and works out‑of‑the‑box with `brainstate <https://brainstate.readthedocs.io/>`_
and `brainunit <https://brainunit.readthedocs.io/>`_.


Quickstart
^^^^^^^^^^

Classification loss with integer labels::

  import jax.numpy as jnp
  import braintools

  logits = jnp.array([[2.0, 1.0, 0.1]])
  labels = jnp.array([0])
  loss = braintools.metric.softmax_cross_entropy_with_integer_labels(logits, labels)

Compute functional connectivity and its dynamics::

  # activities: (time, channels)
  activities = jnp.random.normal(size=(200, 10))
  fc  = braintools.metric.functional_connectivity(activities)
  fcd = braintools.metric.functional_connectivity_dynamics(activities, window_size=30, step_size=5)

Advance an ODE with RK4::

  from braintools.quad import ode_rk4_step

  def f(y, t):
      return -y  # dy/dt = -y

  y, t = jnp.array(1.0), 0.0
  y_next = ode_rk4_step(f, y, t)


Design Principles
^^^^^^^^^^^^^^^^^

- JAX‑friendly: pure functions, PyTree support, compatible with jit/vmap
- Minimal surface area: small, focused helpers you can compose
- Practical defaults: sensible numerical behavior with clear docstrings


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U braintools[cpu]


    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U braintools[cuda12]

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
    :caption: Numerical Integrations

    ode_integration.ipynb
    sde_integration.ipynb



.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Metric Functions


    classification_losses.ipynb
    regression_losses.ipynb
    ranking_learning_to_rank.ipynb
    pairwise_embedding_similarity.ipynb
    spiking_metrics.ipynb
    advanced_spiking_metrics.ipynb
    lfp_analysis.ipynb



.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Spike Encodings

    spike_encoding.ipynb



.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: File Processing



.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Visualization




.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: API Reference

    apis/changelog.md
    apis/braintools.rst
    apis/quad.rst
    apis/metric.rst
    apis/optim.rst
    apis/input.rst
    apis/tree.rst
