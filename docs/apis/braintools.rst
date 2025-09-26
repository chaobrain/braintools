``braintools`` module
=====================

.. currentmodule:: braintools
.. automodule:: braintools

BrainTools aggregates reusable building blocks for spiking neural computations,
including encoders, spike arithmetic utilities, and parameter transforms.

Spike Encoders
--------------

Encoders that convert continuous-valued signals or event streams into spike
trains suitable for spiking network simulations.

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



Spike Operations
----------------

Vectorized boolean and arithmetic helpers for manipulating spike trains and
combining encoder outputs.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spike_bitwise_or
    spike_bitwise_and
    spike_bitwise_iand
    spike_bitwise_not
    spike_bitwise_xor
    spike_bitwise_ixor
    spike_bitwise



Parameter Transformations
-------------------------

Smooth and invertible transforms used to keep optimization parameters within
valid domains during training.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    Transform
    IdentityTransform
    SigmoidTransform
    SoftplusTransform
    NegSoftplusTransform
    LogTransform
    ExpTransform
    TanhTransform
    SoftsignTransform
    AffineTransform
    ChainTransform
    MaskedTransform
    CustomTransform
