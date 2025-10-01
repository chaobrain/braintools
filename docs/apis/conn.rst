``braintools.conn`` module
==========================

.. currentmodule:: braintools.conn
.. automodule:: braintools.conn

Modular connectivity system for building neural network connection patterns
across different types of neural models. The system provides specialized
implementations for point neurons and multi-compartment models with direct
class access.


Base Classes and Results
------------------------

Core infrastructure for connectivity patterns and results.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ConnectionResult
   Connectivity
   PointNeuronConnectivity
   MultiCompartmentConnectivity


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
   FixedProbability
   Custom

Spatial patterns
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DistanceDependent
   Gaussian
   Exponential
   Ring
   Grid
   RadialPatches
   Regular

Topological patterns
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SmallWorld
   ScaleFree
   Modular
   ClusteredRandom

Biological patterns
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ExcitatoryInhibitory
   SynapticPlasticity
   ActivityDependent


Convolutional Kernels
---------------------

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

Basic compartment patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CompartmentSpecific
   RandomCompartment
   AllToAllCompartments
   CustomCompartment

Anatomical targeting patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SomaToDendrite
   AxonToSoma
   DendriteToSoma
   AxonToDendrite
   DendriteToDendrite

Morphology-aware patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ProximalTargeting
   DistalTargeting
   BranchSpecific
   MorphologyDistance

Dendritic patterns
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DendriticTree
   BasalDendriteTargeting
   ApicalDendriteTargeting
   DendriticIntegration

Axonal patterns
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AxonalProjection
   AxonalBranching
   AxonalArborization
   TopographicProjection

Synaptic patterns
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SynapticPlacement
   SynapticClustering
   ActivityDependentSynapses

