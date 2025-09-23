``braintools`` module
=====================

.. currentmodule:: braintools
.. automodule:: braintools



Spike Encoders
--------------

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
