``braintools.init`` module
==========================

.. currentmodule:: braintools.init
.. automodule:: braintools.init

Comprehensive parameter initialization toolkit for brain modeling, featuring statistical distributions,
variance scaling strategies, orthogonal initializations, and distance-dependent connectivity patterns.

Overview
--------

The ``braintools.init`` module provides:

- **Statistical distributions** for basic weight and parameter initialization
- **Variance scaling methods** (Kaiming/He, Xavier/Glorot, LeCun) for deep neural networks
- **Orthogonal initializations** for recurrent networks and deep architectures
- **Distance-dependent profiles** for spatially-structured connectivity
- **Composite distributions** for creating complex initialization patterns
- **Functional composition** via arithmetic operations and transformations

Base Classes
------------

These classes provide the foundational architecture for all initialization strategies in the module.
The ``Initialization`` class defines the common interface for weight and parameter initializations,
while ``DistanceProfile`` serves as the base for distance-dependent connectivity patterns.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Initialization
   DistanceProfile

Utility Functions
~~~~~~~~~~~~~~~~~

Helper functions and type aliases that support initialization workflows.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   init_call
   Initializer
   Compose

Basic Distributions
-------------------

Fundamental statistical distributions for parameter initialization. These provide standard
probability distributions commonly used in neural network initialization and brain modeling.
All distributions support physical units via brainunit and accept an optional random number
generator for reproducibility.

Constant and Uniform
~~~~~~~~~~~~~~~~~~~~

Simple initialization strategies for constant values and uniform random sampling.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Constant
   Uniform

Gaussian-like Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normal and log-normal distributions for generating bell-curved parameter values.
These are the most commonly used distributions for weight initialization.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Normal
   LogNormal
   TruncatedNormal

Heavy-tailed and Skewed Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Distributions with non-Gaussian shapes, useful for modeling biological variability
and creating diverse parameter patterns. These include exponential decay, gamma,
beta, and Weibull distributions.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Gamma
   Exponential
   Beta
   Weibull

Variance Scaling Initializations
---------------------------------

Variance scaling methods automatically adjust the initialization scale based on layer dimensions
to maintain stable gradient flow during training. These are essential for training deep neural
networks and are widely used in modern deep learning frameworks.

All variance scaling methods support three modes:

- **fan_in**: Scale by number of input units (default for most methods)
- **fan_out**: Scale by number of output units
- **fan_avg**: Scale by average of input and output units

Kaiming/He Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~

Recommended for ReLU and leaky ReLU activations. Maintains variance through layers with
rectified linear units by accounting for the fact that ReLU zeros out half the activations.

Reference: He et al., "Delving Deep into Rectifiers: Surpassing Human-Level Performance
on ImageNet Classification", ICCV 2015.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   KaimingUniform
   KaimingNormal

Xavier/Glorot Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recommended for tanh and sigmoid activations. Uses fan_avg mode to balance gradient
flow in both forward and backward passes. Particularly effective for symmetric activation
functions.

Reference: Glorot & Bengio, "Understanding the difficulty of training deep feedforward
neural networks", AISTATS 2010.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   XavierUniform
   XavierNormal

LeCun Initialization
~~~~~~~~~~~~~~~~~~~~

Similar to Xavier but uses fan_in only. Recommended for SELU (scaled exponential linear
unit) activations, which have self-normalizing properties.

Reference: LeCun et al., "Efficient BackProp", 1998.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LecunUniform
   LecunNormal

Orthogonal Initializations
---------------------------

Orthogonal initialization methods create weight matrices with orthonormal rows or columns,
which helps preserve gradient norms during backpropagation. These are particularly useful
for recurrent neural networks and very deep architectures where gradient flow is critical.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Orthogonal
   DeltaOrthogonal
   Identity

- **Orthogonal**: Standard orthogonal matrix initialization using QR decomposition. Preserves
  gradient norms and is ideal for recurrent networks. Often scaled by sqrt(2) for ReLU networks.

- **DeltaOrthogonal**: Specialized initialization for deep convolutional networks. Creates
  delta functions in spatial dimensions combined with orthogonal initialization in channel
  dimensions, enabling training of extremely deep CNNs (10,000+ layers).

- **Identity**: Identity matrix initialization, optionally scaled. Creates matrices that
  initially perform identity transformations, useful for residual connections and gated
  recurrent architectures.

Composite Distributions
-----------------------

Composite distributions combine or modify other distributions to create complex initialization
patterns. These enable sophisticated initialization strategies by composing simple distributions
in flexible ways.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Mixture
   Conditional
   Scaled
   Clipped

- **Mixture**: Randomly selects from multiple distributions for each parameter according to
  specified probability weights. Useful for creating heterogeneous parameter populations.

- **Conditional**: Uses different distributions based on neuron properties or indices.
  Enables excitatory/inhibitory distinction or layer-specific initialization strategies.

- **Scaled**: Multiplies another distribution by a constant factor. Simple wrapper for
  scaling existing distributions.

- **Clipped**: Clips distribution output to specified minimum and maximum values. Ensures
  parameters stay within desired bounds without changing the underlying distribution.



Distance-Modulated Initialization
---------------------------------

Distance-modulated initialization adjusts weights based on spatial distance between neurons,
using a specified distance profile. This is essential for spatially-structured brain models
where connection strength depends on neuronal proximity.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DistanceModulated


Distance Profiles
-----------------

Distance profiles define how connection probability and weight strength vary with spatial
distance between neurons. These are essential for modeling spatially-structured neural
populations where connectivity follows anatomical principles.

All distance profiles implement two methods:

- ``probability(distances)``: Connection probability as a function of distance
- ``weight_scaling(distances)``: Weight strength scaling factor as a function of distance

Base Class
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DistanceProfile

Distance Profile Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concrete distance profile implementations for various spatial connectivity patterns.
Import these from ``braintools.init``:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   GaussianProfile
   ExponentialProfile
   LinearProfile
   StepProfile
   PowerLawProfile
   MexicanHatProfile
   DoGProfile
   SigmoidProfile
   BimodalProfile
   LogisticProfile



Composition Profiles
~~~~~~~~~~~~~~~~~~~~

These profiles enable functional composition and transformation of distance profiles,
allowing complex spatial patterns to be built from simpler ones.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ComposedProfile
   ClipProfile
   ApplyProfile
   PipeProfile

Functional Composition
----------------------

Both ``Initialization`` and ``DistanceProfile`` classes support rich functional composition
through operator overloading and transformation methods:

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

- **Addition** (``+``): Sum two initializations or add a constant
- **Subtraction** (``-``): Subtract initializations or subtract a constant
- **Multiplication** (``*``): Multiply initializations or scale by a constant
- **Division** (``/``): Divide initializations or scale by a reciprocal

Transformation Methods
~~~~~~~~~~~~~~~~~~~~~~

- ``.clip(min_val, max_val)``: Clip output to specified range
- ``.add(value)``: Add a constant value
- ``.multiply(value)``: Multiply by a constant value
- ``.apply(func)``: Apply arbitrary function to output

Composition Operators
~~~~~~~~~~~~~~~~~~~~~

- **Pipe** (``|``): Chain transformations functionally
- **Compose**: Explicitly compose multiple transformations

Examples
--------

Basic Weight Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import brainunit as u
    from braintools.init import Normal, Uniform, Constant

    # Gaussian weights with physical units
    weight_init = Normal(0.5 * u.nS, 0.1 * u.nS)
    rng = np.random.default_rng(0)
    weights = weight_init(1000, rng=rng)

    # Uniform delay distribution
    delay_init = Uniform(1.0 * u.ms, 5.0 * u.ms)
    delays = delay_init(1000, rng=rng)

    # Constant bias
    bias_init = Constant(-70.0 * u.mV)
    biases = bias_init(100)

Variance Scaling for Deep Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from braintools.init import KaimingNormal, XavierUniform

    # Kaiming initialization for ReLU network
    init_relu = KaimingNormal(mode='fan_in')
    layer1_weights = init_relu((256, 784), rng=rng)
    layer2_weights = init_relu((128, 256), rng=rng)

    # Xavier initialization for tanh network
    init_tanh = XavierUniform()
    layer1_weights = init_tanh((256, 784), rng=rng)
    layer2_weights = init_tanh((128, 256), rng=rng)

Orthogonal Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from braintools.init import Orthogonal, Identity

    # Orthogonal recurrent weights
    recurrent_init = Orthogonal(scale=np.sqrt(2))
    recurrent_weights = recurrent_init((128, 128), rng=rng)

    # Identity initialization for residual connections
    residual_init = Identity(scale=1.0)
    residual_weights = residual_init((256, 256))

Functional Composition
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from braintools.init import Normal, Uniform

    # Compose with arithmetic operations
    combined = Normal(0.5 * u.nS, 0.1 * u.nS) * 2.0 + 0.1 * u.nS
    weights = combined(1000, rng=rng)

    # Chain transformations with pipe operator
    init = (Normal(1.0 * u.nS, 0.3 * u.nS) |
            (lambda x: u.math.maximum(x, 0 * u.nS)) |
            (lambda x: x * 0.5))
    weights = init(1000, rng=rng)

    # Clip to valid range
    clipped_init = Normal(0.5 * u.nS, 0.2 * u.nS).clip(0.0 * u.nS, 1.0 * u.nS)
    weights = clipped_init(1000, rng=rng)

Composite Distributions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from braintools.init import Mixture, Conditional, Scaled

    # Mixture of distributions (70% normal, 30% uniform)
    mix_init = Mixture(
        distributions=[
            Normal(0.5 * u.nS, 0.1 * u.nS),
            Uniform(0.8 * u.nS, 1.2 * u.nS)
        ],
        weights=[0.7, 0.3]
    )
    weights = mix_init(1000, rng=rng)

    # Conditional initialization (excitatory vs inhibitory)
    def is_excitatory(indices):
        return indices < 800

    cond_init = Conditional(
        condition_fn=is_excitatory,
        true_dist=Normal(0.5 * u.nS, 0.1 * u.nS),  # Excitatory
        false_dist=Normal(-0.3 * u.nS, 0.05 * u.nS)  # Inhibitory
    )
    weights = cond_init(1000, neuron_indices=np.arange(1000), rng=rng)

Distance-Dependent Connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from braintools.init import (GaussianProfile, ExponentialProfile,
                                  DistanceModulated, Normal)

    # Gaussian distance profile
    gaussian_profile = GaussianProfile(sigma=100.0 * u.um)

    # Exponential decay profile
    exp_profile = ExponentialProfile(length_scale=200.0 * u.um)

    # Distance-modulated weights
    init = DistanceModulated(
        base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
        distance_profile=gaussian_profile,
        min_weight=0.01 * u.nS
    )

    # Assuming distances is a distance matrix between neurons
    distances = np.random.uniform(0, 500, size=1000) * u.um
    weights = init(1000, distances=distances, rng=rng)

    # Compose distance profiles
    combined_profile = gaussian_profile * 0.7 + exp_profile * 0.3

    # Clip profile values
    clipped_profile = gaussian_profile.clip(0.1, 0.9)

    # Apply custom transformation
    transformed_profile = exp_profile.apply(lambda x: x ** 2)

Helper Function Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from braintools.init import init_call, Normal, Compose

    # init_call provides a unified interface
    init1 = Normal(0.5 * u.nS, 0.1 * u.nS)
    weights1 = init_call(init1, 100, rng=rng)

    # Works with scalars
    weights2 = init_call(0.5, 100)

    # Works with arrays
    weights3 = init_call(np.ones(100) * 0.5 * u.nS, 100)

    # Compose multiple initializations
    composed = Compose(
        Normal(1.0 * u.nS, 0.2 * u.nS),
        lambda x: u.math.maximum(x, 0 * u.nS),
        lambda x: x * 0.5
    )
    weights = composed(1000, rng=rng)

Best Practices
--------------

Choosing Initialization Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **For feedforward networks with ReLU**: Use ``KaimingNormal`` or ``KaimingUniform``
2. **For networks with tanh/sigmoid**: Use ``XavierNormal`` or ``XavierUniform``
3. **For recurrent networks**: Use ``Orthogonal`` with appropriate scale (often sqrt(2))
4. **For very deep CNNs**: Consider ``DeltaOrthogonal``
5. **For biological models**: Use ``Normal``, ``LogNormal``, or distance-dependent profiles

Random Number Generators
~~~~~~~~~~~~~~~~~~~~~~~~~

Always pass an RNG for reproducibility:

.. code-block:: python

    rng = np.random.default_rng(seed=42)
    weights = init(1000, rng=rng)

Physical Units
~~~~~~~~~~~~~~

Use brainunit for physical quantities:

.. code-block:: python

    # Good: explicit units
    weight_init = Normal(0.5 * u.nS, 0.1 * u.nS)

    # Also valid: dimensionless
    weight_init = Normal(0.5, 0.1)

Distance-Dependent Models
~~~~~~~~~~~~~~~~~~~~~~~~~

For spatially-structured models, combine distance profiles with base distributions:

.. code-block:: python

    profile = GaussianProfile(sigma=100 * u.um)
    init = DistanceModulated(
        base_dist=Normal(1.0 * u.nS, 0.2 * u.nS),
        distance_profile=profile
    )

Composition Strategies
~~~~~~~~~~~~~~~~~~~~~~

Build complex patterns incrementally:

.. code-block:: python

    # Start simple
    base = Normal(0.5 * u.nS, 0.1 * u.nS)

    # Add constraints
    base = base.clip(0.0 * u.nS, 1.0 * u.nS)

    # Apply scaling
    base = base * 2.0

    # Add offset
    final_init = base + 0.1 * u.nS
