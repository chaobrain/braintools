``braintools.conn`` module
==========================

.. currentmodule:: braintools.conn
.. automodule:: braintools.conn

Modular connectivity system for building neural network connection patterns
across different types of neural models. The system provides specialized
implementations for point neurons, population rate models, and multi-compartment
models with a unified API.


Base Classes and Results
------------------------

Core infrastructure for connectivity patterns and results.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ConnectionResult
   Connectivity
   CompositeConnectivity
   ScaledConnectivity


Point Neuron Connectivity
--------------------------

Connectivity patterns for single-compartment point neuron models.

Basic patterns
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Random
   AllToAll
   OneToOne

Spatial patterns
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DistanceDependent
   Grid
   RadialPatches

Network patterns
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ExcitatoryInhibitory
   Custom


Kernel patterns
~~~~~~~~~~~~~~~

Connectivity patterns using convolution kernels for spatial processing.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ConvKernel
   GaussianKernel
   GaborKernel
   DoGKernel
   MexicanHat
   SobelKernel
   LaplacianKernel
   CustomKernel


Multi-Compartment Connectivity
-------------------------------

Connectivity patterns for detailed multi-compartment neuron models with
compartment-specific targeting.

Compartment-specific patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CompartmentSpecific
   SomaToDendrite
   AxonToSoma
   AxonToDendrite
   DendriteToSoma

Morphology-aware patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   MorphologyDistance
   DendriticTree
   AxonalProjection
   CustomCompartment

Compartment constants
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SOMA
   BASAL_DENDRITE
   APICAL_DENDRITE
   AXON




Population Rate Connectivity
-----------------------------

Connectivity patterns for population rate models and mean-field dynamics.

Population coupling
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   PopulationCoupling
   MeanField
   ExcitatoryInhibitoryPopulation

Network architectures
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HierarchicalPopulations
   WilsonCowanNetwork


Initialization Classes
----------------------

Classes for initializing connection weights, delays, and distance-dependent profiles.

Weight initialization
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   WeightInit
   ConstantWeight
   UniformWeight
   NormalWeight
   LogNormalWeight
   ExponentialWeight
   GammaWeight
   BetaWeight
   OrthogonalWeight
   XavierUniformWeight
   XavierNormalWeight
   HeUniformWeight
   HeNormalWeight
   LeCunUniformWeight
   LeCunNormalWeight

Delay initialization
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DelayInit
   ConstantDelay
   UniformDelay
   NormalDelay
   GammaDelay

Distance profiles
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DistanceProfile
   GaussianProfile
   ExponentialProfile
   PowerLawProfile
   LinearProfile
   StepProfile

