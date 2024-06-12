``braintools`` documentation
============================

`braintools <https://github.com/brainpy/braintools>`_ implements the common toolboxes for brain dynamics programming (BDP).

----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U braintools[cpu]

    .. tab-item:: GPU (CUDA 11.0)

       .. code-block:: bash

          pip install -U braintools[cuda11]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U braintools[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U braintools[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^



- `brainstate <https://github.com/brainpy/brainstate>`_: A ``State``-based transformation system for brain dynamics programming.

- `brainunit <https://github.com/brainpy/brainunit>`_: The unit system for brain dynamics programming.

- `braintaichi <https://github.com/brainpy/braintaichi>`_: Leveraging Taichi Lang to customize brain dynamics operators.

- `brainscale <https://github.com/brainpy/brainscale>`_: The scalable online learning framework for biological neural networks.

- `braintools <https://github.com/brainpy/braintools>`_: The toolbox for the brain dynamics simulation, training and analysis.



.. toctree::
   :hidden:
   :maxdepth: 2

   api.rst

