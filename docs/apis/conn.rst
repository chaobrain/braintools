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
   PointConnectivity
   MultiCompartmentConnectivity
   ScaledConnectivity
   CompositeConnectivity


Point Neuron Connectivity
--------------------------

Connectivity patterns for single-compartment point neuron models.

Basic Patterns
~~~~~~~~~~~~~~

Simple connectivity patterns including random and deterministic connections.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Random
   FixedProb
   AllToAll
   OneToOne

Spatial Patterns
~~~~~~~~~~~~~~~~

Distance-dependent and spatially-structured connectivity patterns.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DistanceDependent
   Gaussian
   Exponential
   Ring
   Grid2d
   RadialPatches
   ClusteredRandom

Topological Patterns
~~~~~~~~~~~~~~~~~~~~

Complex network topology patterns based on graph theory.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SmallWorld
   ScaleFree
   Regular
   ModularRandom
   ModularGeneral
   HierarchicalRandom
   CorePeripheryRandom

Biological Patterns
~~~~~~~~~~~~~~~~~~~

Biologically-inspired connectivity patterns following neural principles.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ExcitatoryInhibitory


Kernel-Based Connectivity
--------------------------

Connectivity patterns using convolution kernels for spatial receptive fields.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Conv2dKernel
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

Compartment Constants
~~~~~~~~~~~~~~~~~~~~~

Predefined constants for identifying neural compartments.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SOMA
   BASAL_DENDRITE
   APICAL_DENDRITE
   AXON

Basic Compartment Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fundamental patterns for compartment-specific connectivity.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CompartmentSpecific
   AllToAllCompartments
   CustomCompartment

Anatomical Targeting Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Connectivity patterns based on anatomical organization.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SomaToDendrite
   AxonToSoma
   DendriteToSoma
   AxonToDendrite
   DendriteToDendrite

Morphology-Aware Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~

Patterns that utilize detailed morphological information.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ProximalTargeting
   DistalTargeting
   BranchSpecific
   MorphologyDistance

Dendritic Patterns
~~~~~~~~~~~~~~~~~~

Specialized patterns for dendritic targeting and integration.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DendriticTree
   BasalDendriteTargeting
   ApicalDendriteTargeting
   DendriticIntegration

Axonal Patterns
~~~~~~~~~~~~~~~

Patterns for axonal projection and arborization.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AxonalProjection
   AxonalBranching
   AxonalArborization
   TopographicProjection

Synaptic Patterns
~~~~~~~~~~~~~~~~~

Patterns for synaptic placement and organization.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SynapticPlacement
   SynapticClustering
