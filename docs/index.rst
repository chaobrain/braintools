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


We are building the `BDP ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_:

----

.. toctree::
   :hidden:
   :maxdepth: 2

   api.rst

