``braintools`` documentation
============================

`braintools <https://github.com/brainpy/braintools>`_ implements the common toolboxes for brain modeling.

----


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


BrainTools is one part of our `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.

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




.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: File Processing




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
