``braintools.param`` module
===========================

.. currentmodule:: braintools.param
.. automodule:: braintools.param

Comprehensive parameter management module for constrained optimization and probabilistic modeling.
Provides bijective transforms, regularization priors, and parameter wrappers for neural network training.

Overview
--------

The ``braintools.param`` module provides:

- **Parameter wrappers** (``Param``, ``Const``) for managing trainable and fixed parameters
- **State containers** (``ArrayHidden``, ``ArrayParam``) for transformed array storage
- **Bijective transformations** for mapping parameters between constrained and unconstrained domains
- **Regularization priors** for Bayesian inference and weight decay
- **Jacobian computation** via ``log_abs_det_jacobian`` for probabilistic modeling
- **Data containers** for hierarchical state management

Data Container
--------------

The ``Data`` class provides a base class for creating dataclass-like containers
that support hierarchical state management and JAX pytree operations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Data

State Containers
----------------

These classes extend brainstate's ``HiddenState`` and ``ParamState`` to support
automatic bijective transformations. They store values in unconstrained space
internally while exposing constrained values via the ``.data`` property.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ArrayHidden
   ArrayParam


Parameter Wrappers
------------------

The ``Param`` class wraps parameter values and automatically applies bijective
transformations and regularization. It provides a high-level interface for
managing trainable parameters in neural networks.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Param
   Const


Regularization Classes
----------------------

These classes implement regularization priors for Bayesian inference and weight
decay. They provide ``loss()``, ``sample_init()``, and ``reset_value()`` methods
for integration with parameter optimization workflows.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Regularization
   GaussianReg
   L1Reg
   L2Reg


Base Transform Classes
----------------------

These classes provide the foundational architecture for all transforms in the module.
The ``Transform`` class defines the common interface with ``forward``, ``inverse``,
and ``log_abs_det_jacobian`` methods.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Transform
   IdentityT

Bounded Transforms
------------------

These transforms map unbounded values to bounded intervals. They are useful for
parameters that must lie within specific ranges, such as probabilities, correlation
coefficients, or physical quantities with known bounds.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SigmoidT
   ScaledSigmoidT
   TanhT
   SoftsignT
   ClipT


Positive/Negative Transforms
----------------------------

These transforms constrain parameters to be strictly positive or negative.
They are commonly used for variance parameters, rate constants, and other
quantities that must maintain a specific sign.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SoftplusT
   NegSoftplusT
   LogT
   ExpT
   PositiveT
   NegativeT
   ReluT


Advanced Transforms
-------------------

These transforms implement sophisticated constraints for specialized applications
in probabilistic modeling, ordinal regression, and directional statistics.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   PowerT
   OrderedT
   SimplexT
   UnitVectorT


Composition Transforms
----------------------

These transforms allow building complex transformations from simpler ones.
They support chaining, masking, and affine transformations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AffineT
   ChainT
   MaskedT


Utility Functions
-----------------

Helper functions for working with parameters and state objects.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   get_param
   get_size

