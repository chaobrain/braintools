``braintools.tree`` module
==========================

.. currentmodule:: braintools.tree
.. automodule:: braintools.tree

Operations for manipulating JAX PyTrees, including arithmetic, reshaping, and
conversion utilities used across the toolkit. The functions map over the
leaves of their PyTree arguments; the arithmetic and reduction functions treat
:class:`brainunit.Quantity` values as leaves by default, so physical units are
preserved.


Arithmetic
----------

Leaf-wise arithmetic between PyTrees, or between a PyTree and a scalar.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   scale
   mul
   shift
   add
   sub


Reductions
----------

Reduce one or two PyTrees to a single value.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dot
   sum
   squared_norm


Structure and Indexing
----------------------

Combine, split, index, and reshape the leaves of a PyTree.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   concat
   split
   idx
   expand
   take


Conversion
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   as_numpy
