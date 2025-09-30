``braintools.optim`` module
===========================

.. currentmodule:: braintools.optim
.. automodule:: braintools.optim

Comprehensive optimization toolkit for brain modeling, featuring PyTorch-like optimizers,
learning rate schedulers, and advanced optimization algorithms from SciPy and Nevergrad.

Overview
--------

The ``braintools.optim`` module provides:

- **Modern gradient-based optimizers** with PyTorch-compatible APIs
- **Learning rate schedulers** for dynamic learning rate adjustment
- **Black-box optimization** via SciPy and Nevergrad wrappers
- **State management utilities** for optimization workflows

Base Classes
------------

These classes provide the foundational architecture for all optimizers in the module.
The ``Optimizer`` class defines the common interface, while ``OptaxOptimizer`` serves
as the base for all gradient-based optimizers built on top of the Optax library.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Optimizer
   OptaxOptimizer

Gradient-Based Optimizers
-------------------------

These optimizers use gradient information to update model parameters. They follow
a PyTorch-like API, making them familiar to users coming from the PyTorch ecosystem.
All gradient-based optimizers support features like weight decay, gradient clipping,
and integration with learning rate schedulers.

Standard Optimizers
~~~~~~~~~~~~~~~~~~~

These are the most commonly used optimizers in deep learning and neural network training.
They provide a good balance between convergence speed and stability for most applications.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SGD
   Adam
   AdamW
   Adagrad
   Adadelta
   RMSprop
   Adamax
   Nadam

Advanced Optimizers
~~~~~~~~~~~~~~~~~~~

These optimizers implement state-of-the-art optimization algorithms designed for
specific use cases or improved performance. They often provide better convergence
properties for large-scale models, handle sparse gradients more effectively, or
offer improved stability in challenging optimization landscapes.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   RAdam
   Lamb
   Lars
   Lookahead
   Yogi
   LBFGS
   Rprop
   Adafactor
   AdaBelief
   Lion
   SM3
   Novograd
   Fromage

Learning Rate Schedulers
------------------------

Learning rate schedulers dynamically adjust the learning rate during training to improve
convergence and final model performance. They can help escape local minima, fine-tune
models more effectively, and achieve better generalization. All schedulers are compatible
with any gradient-based optimizer.

Base Scheduler
~~~~~~~~~~~~~~

The abstract base class that defines the interface for all learning rate schedulers.
Custom schedulers should inherit from this class.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LRScheduler

Step-based Schedulers
~~~~~~~~~~~~~~~~~~~~~

These schedulers adjust the learning rate at fixed intervals or following predetermined
patterns. They are simple to configure and work well for many standard training scenarios.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   StepLR
   MultiStepLR
   ConstantLR
   LinearLR
   ExponentialLR
   PolynomialLR

Annealing Schedulers
~~~~~~~~~~~~~~~~~~~~

These schedulers smoothly decrease the learning rate following mathematical functions
like cosine curves. They often provide better convergence than step-based approaches
and are particularly effective for fine-tuning and achieving optimal final performance.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CosineAnnealingLR
   CosineAnnealingWarmRestarts
   WarmupCosineSchedule

Cyclic Schedulers
~~~~~~~~~~~~~~~~~

These schedulers vary the learning rate in cycles, allowing the model to escape
sharp minima and explore the loss landscape more effectively. They can lead to
better generalization and faster convergence.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CyclicLR
   OneCycleLR

Adaptive Schedulers
~~~~~~~~~~~~~~~~~~~

These schedulers adjust the learning rate based on training dynamics or combine
multiple scheduling strategies. They can automatically adapt to the training progress
or provide complex scheduling patterns for specialized training regimes.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ReduceLROnPlateau
   WarmupScheduler
   PiecewiseConstantSchedule

Composite Schedulers
~~~~~~~~~~~~~~~~~~~~

These schedulers combine multiple scheduling strategies, allowing you to chain
different schedulers together or switch between them at different training phases.
They provide maximum flexibility for complex training scenarios.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ChainedScheduler
   SequentialLR

Black-Box Optimizers
--------------------

These optimizers are designed for derivative-free optimization problems where gradients
are not available or are expensive to compute. They are particularly useful for
hyperparameter optimization, neural architecture search, and optimizing non-differentiable
objectives. These wrappers provide a unified interface to powerful optimization libraries.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ScipyOptimizer
   NevergradOptimizer

- **ScipyOptimizer**: Wraps SciPy's optimization algorithms including BFGS, L-BFGS-B,
  Nelder-Mead, Powell, and other classical optimization methods. Best for low to
  medium dimensional problems with smooth objectives.

- **NevergradOptimizer**: Integrates Facebook's Nevergrad library, providing access to
  evolutionary algorithms, particle swarm optimization, differential evolution, and
  other population-based methods. Excellent for high-dimensional, noisy, or discrete
  optimization problems.

Utilities
---------

Helper classes and functions that support optimization workflows, including state
management for complex optimization scenarios.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   UniqueStateManager

The ``UniqueStateManager`` helps manage unique state objects in PyTree structures,
ensuring proper state isolation and preventing unintended state sharing during
optimization of complex models with nested components.
