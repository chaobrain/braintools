``braintools.spike`` module
===========================

.. currentmodule:: braintools.spike
.. automodule:: braintools.spike

Spike data in one place: a fixed-shape container for spike times, encoders
that turn numbers or time series into spike rasters, and element-wise bitwise
operations on spike tensors.


Spike Trains
------------

:class:`SpikeTrains` stores the spike times of a population as a padded
``(n_neurons, capacity)`` array. It is a JAX pytree, so it can be built and
used inside ``jax.jit`` and ``jax.vmap``, and it converts to and from dense
rasters (``from_dense`` / ``to_dense``), flat event lists (``from_events`` /
``to_events``) and ragged per-neuron lists (``from_list`` / ``to_list``).
The ``braintools.visualize`` spike plots accept it directly.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   SpikeTrains


Spike Encoders
--------------

Encoders that convert continuous-valued signals into spike trains (or, for
:class:`StepCurrentEncoder`, into input currents) for spiking network
simulations. Before 0.4.0 they lived at the top level (``braintools.RateEncoder``
and so on); those names are deprecated aliases that will be removed in 1.0.0.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LatencyEncoder
   RateEncoder
   PoissonEncoder
   PopulationEncoder
   BernoulliEncoder
   DeltaEncoder
   StepCurrentEncoder
   SpikeCountEncoder
   TemporalEncoder
   RankOrderEncoder


Bitwise Operations
------------------

Element-wise logical operations on spike tensors. Boolean inputs give boolean
results; numeric inputs use arithmetic identities that keep the dtype and stay
differentiable for surrogate-gradient spikes. The former top-level names
(``braintools.spike_bitwise_or`` and so on) remain available as aliases.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bitwise_or
   bitwise_and
   bitwise_iand
   bitwise_not
   bitwise_xor
   bitwise_ixor
   bitwise
