``braintools.param`` module
===========================

.. currentmodule:: braintools.param
.. automodule:: braintools.param

Parameter transformation module for constrained optimization and probabilistic modeling.
Provides bijective transforms that map between unconstrained and constrained parameter spaces.

Overview
--------

The ``braintools.param`` module provides:

- **Bijective transformations** for mapping parameters between constrained and unconstrained domains
- **Parameter wrapper** (``Param``) that automatically applies transforms during optimization
- **Jacobian computation** via ``log_abs_det_jacobian`` for probabilistic modeling
- **Composition utilities** for building complex transforms from simple ones

Base Classes
------------

These classes provide the foundational architecture for all transforms in the module.
The ``Transform`` class defines the common interface with ``forward``, ``inverse``,
and ``log_abs_det_jacobian`` methods.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Transform
   Identity

Parameter Wrapper
-----------------

The ``Param`` class wraps parameter values and automatically applies bijective
transformations. It stores values in unconstrained space internally while exposing
constrained values via the ``.data`` property.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Param

Bounded Transforms
------------------

These transforms map unbounded values to bounded intervals. They are useful for
parameters that must lie within specific ranges, such as probabilities, correlation
coefficients, or physical quantities with known bounds.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Sigmoid
   ScaledSigmoid
   Tanh
   Softsign
   Clip

- **Sigmoid**: Maps ℝ → [lower, upper] using the logistic sigmoid function
- **ScaledSigmoid**: Sigmoid with adjustable sharpness/temperature parameter
- **Tanh**: Maps ℝ → (lower, upper) using hyperbolic tangent
- **Softsign**: Maps ℝ → (lower, upper) using softsign function
- **Clip**: Hard clipping to bounds (non-bijective)

Positive/Negative Transforms
----------------------------

These transforms constrain parameters to be strictly positive or negative.
They are commonly used for variance parameters, rate constants, and other
quantities that must maintain a specific sign.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Softplus
   NegSoftplus
   Log
   Exp
   Positive
   Negative

- **Softplus**: Maps ℝ → [lower, ∞) using log(1 + exp(x))
- **NegSoftplus**: Maps ℝ → (-∞, upper] using negative softplus
- **Log/Exp**: Maps ℝ → (lower, ∞) using exponential transformation
- **Positive**: Convenience class for (0, ∞) constraint
- **Negative**: Convenience class for (-∞, 0) constraint

Advanced Transforms
-------------------

These transforms implement sophisticated constraints for specialized applications
in probabilistic modeling, ordinal regression, and directional statistics.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Power
   Ordered
   Simplex
   UnitVector

- **Power**: Box-Cox family power transformation for variance stabilization
- **Ordered**: Ensures monotonically increasing output vectors (useful for cutpoints)
- **Simplex**: Stick-breaking transformation for probability vectors summing to 1
- **UnitVector**: Projects vectors onto the unit sphere (L2 norm = 1)

Composition Transforms
----------------------

These transforms allow building complex transformations from simpler ones.
They support chaining, masking, and custom user-defined transformations.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Affine
   Chain
   Masked
   Custom

- **Affine**: Linear transformation y = ax + b
- **Chain**: Composes multiple transforms sequentially
- **Masked**: Applies transform selectively based on boolean mask
- **Custom**: User-defined transformation with custom forward/inverse functions
