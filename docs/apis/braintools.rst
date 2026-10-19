``braintools`` module
=====================

.. currentmodule:: braintools
.. automodule:: braintools

The top-level ``braintools`` namespace exposes the subpackages (``braintools.conn``,
``braintools.spike``, ``braintools.analysis``, ...) together with the
spike-operation aliases and the deprecated encoder aliases listed below.

Spike Encoders (moved)
----------------------

Since 0.4.0 the spike encoders live in :mod:`braintools.spike`:
``braintools.spike.LatencyEncoder``, ``braintools.spike.RateEncoder``,
``braintools.spike.PoissonEncoder``, ``braintools.spike.PopulationEncoder``,
``braintools.spike.BernoulliEncoder``, ``braintools.spike.DeltaEncoder``,
``braintools.spike.StepCurrentEncoder``, ``braintools.spike.SpikeCountEncoder``,
``braintools.spike.TemporalEncoder`` and ``braintools.spike.RankOrderEncoder``
(see :doc:`spike`).

The top-level names (``braintools.LatencyEncoder`` and so on) are deprecated
aliases: they still work, but instantiating one emits a
``BraintoolsDeprecationWarning``. They will be removed in 1.0.0.


Spike Operations
----------------

Vectorized boolean and arithmetic helpers for manipulating spike trains and
combining encoder outputs. They are implemented in :mod:`braintools.spike`
under shorter names (``spike_bitwise_or`` is ``braintools.spike.bitwise_or``,
``spike_bitwise`` is ``braintools.spike.bitwise``, and so on); the top-level
names below are kept as aliases of those functions and do not warn.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    spike_bitwise_or
    spike_bitwise_and
    spike_bitwise_iand
    spike_bitwise_not
    spike_bitwise_xor
    spike_bitwise_ixor
    spike_bitwise
