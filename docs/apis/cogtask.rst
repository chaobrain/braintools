``braintools.cogtask`` module
=============================

.. currentmodule:: braintools.cogtask
.. automodule:: braintools.cogtask

A modular, composable framework for constructing cognitive tasks for neural
network training and computational neuroscience simulations.

.. seealso::

   For runnable, narrative walkthroughs, see the tutorials:

   - :doc:`../cogtask/01_quickstart`
   - :doc:`../cogtask/02_building_custom_tasks`
   - :doc:`../cogtask/03_variable_length_trials`

Overview
--------

The ``braintools.cogtask`` module provides:

- **A phase-based task model** that decomposes trials into fixation, stimulus,
  delay, response, and other epochs with explicit duration, input encoding,
  and output (target) encoding
- **Composition operators** (``>>``, ``*``, ``|``) and compound phases
  (:class:`Sequence`, :class:`Repeat`, :class:`Parallel`) for building rich
  trial structures from simple parts
- **Conditional control flow** with :class:`If`, :class:`Switch`, and
  :class:`While` for trial-by-trial branching and variable-iteration tasks
- **A feature-encoding system** that maps trial state into input/output
  channels via :class:`Feature`/:class:`FeatureSet` and value-spec encoders
  (``one_hot``, ``circular``, ``von_mises``, ``gaussian``, ``cos_sin``, ...)
- **A library of pre-built tasks** spanning decision making, working memory,
  reasoning, and motor control, drawn from systems-neuroscience literature
- **JIT/``vmap``-friendly trial generation** through :meth:`Task.sample` and
  :meth:`Task.batch_sample`, designed to integrate cleanly with
  `brainstate <https://brainstate.readthedocs.io/>`_ and JAX training loops


Core Task Framework
-------------------

The :class:`Task` class orchestrates phase execution, owns the per-trial
random key, and exposes the dataset-style ``sample``/``batch_sample`` API.
:class:`Context` is the mutable trial-level state container shared across
phases; it carries the RNG, input/output buffers, timing information, and
trial-level user data.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Task
   Context

Two equivalent ways to define a task are supported:

- **Instance-based**: pass ``phases=``, ``input_features=``,
  ``output_features=``, and ``trial_init=`` directly to :class:`Task`. Best
  for one-off tasks or interactive exploration.
- **Class-based**: subclass :class:`Task` and override
  :meth:`Task.define_features`, :meth:`Task.define_phases`, and
  :meth:`Task.trial_init`. Best for reusable, parameterized tasks — all
  pre-built tasks follow this pattern. See
  :doc:`../cogtask/02_building_custom_tasks` for worked examples of both.


Phases and Composition
----------------------

Phases are the atomic units of a trial. :class:`Phase` is the abstract base
class; concrete *declarative* phases (:class:`Fixation`, :class:`Stimulus`,
:class:`Delay`, :class:`Response`, ...) inherit from :class:`DeclarativePhase`
and describe their inputs/outputs/noise via dictionaries instead of code.

Phases compose with operators:

- ``a >> b`` — sequential composition (yields :class:`Sequence`)
- ``a * n`` — repeat ``n`` times (yields :class:`Repeat`)
- ``a | b`` — parallel composition (yields :class:`Parallel`)
- :func:`concat` — sequence from a list

Base Class
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Phase
   DeclarativePhase

Compound Phases
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Sequence
   Repeat
   Parallel

Declarative Phase Types
~~~~~~~~~~~~~~~~~~~~~~~

These are convenience subclasses of :class:`DeclarativePhase` that share its
interface but provide semantic names so trial structures read naturally.
They differ only in identity — use whichever name best describes the epoch.

Basic epochs:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Fixation
   Stimulus
   Delay
   Response
   Cue
   Blank

Working-memory epochs:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Sample
   Test
   Recall
   Match
   Comparison

Composition Helpers
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   concat
   execute_phase


Conditional Phases
------------------

Phases can branch on trial state at runtime. :class:`If` selects between
``then`` / ``else_``; :class:`Switch` dispatches over many cases; :class:`While`
loops until a condition fails (bounded by ``max_iterations``).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   If
   Switch
   While

Because these phases inspect trial state during a Python-level pass over the
tree, the branch they take must be derivable from values set in
``trial_init`` (or in earlier phases' ``on_exit`` hooks). Their total
duration, summed across iterations or branches, contributes to the per-trial
buffer size — see :doc:`../cogtask/03_variable_length_trials` for the
implications.


Features
--------

A :class:`Feature` declares one logical input or output channel of the task,
with a fixed dimensionality and a name. :class:`FeatureSet` collects features
into a flat vector and tracks per-feature index slices automatically.
:class:`CircleFeature` adds a value range for angular / directional outputs.

Compose features with ``+`` (concatenate, immutable), ``-`` (remove by name),
``|`` (alias for ``+``), and ``*n`` (named repetition).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Feature
   FeatureSet
   CircleFeature

Feature predicates:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   is_feature
   as_feature


Encoders
--------

Encoders are *value specifications* — callables of the form
``f(ctx, feature) -> jnp.ndarray`` that :class:`DeclarativePhase` uses to fill
its input slice for one feature. They translate trial-level state (e.g. a
direction angle, a discrete class index) into per-timestep input activations.

Discrete / class encoders:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   one_hot
   identity

Directional / population encoders:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   circular
   von_mises
   cos_sin

Scalar / shape encoders:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   scalar
   gaussian
   ctx_value


Output Labels
-------------

Label helpers build the ``outputs={'label': ...}`` spec used by phases in
categorical mode. They convert per-trial state into integer labels, time-
varying label arrays, or match/comparison codes.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   label
   match_label
   comparison_label


Pre-built Tasks
---------------

The ``cogtask`` package ships a library of standard cognitive paradigms,
each implemented as a subclass of :class:`Task` that defines its own
features, phase structure, and trial-init logic. Construct them like any
other :class:`Task`, optionally passing ``seed=`` for reproducibility — see
:doc:`../cogtask/01_quickstart` for a runnable example.

Decision Making
~~~~~~~~~~~~~~~

Two-alternative and multi-modal perceptual decision tasks with motion
coherence, context cues, or discrete evidence pulses.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   PerceptualDecisionMaking
   PerceptualDecisionMakingDelayResponse
   ContextDecisionMaking
   SingleContextDecisionMaking
   PulseDecisionMaking

Working Memory
~~~~~~~~~~~~~~

Delay-bridging tasks that require holding stimulus identity, magnitude,
category, direction, or interval information across a memory period.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DelayMatchSample
   DualDelayMatchSample
   DelayComparison
   DelayMatchCategory
   DelayPairedAssociation
   GoNoGo
   IntervalDiscrimination
   PostDecisionWager
   ReadySetGo
   DelayDirectionReproduction
   ImmediateDirectionReproduction
   DelayDirectionClassification
   ImmediateDirectionClassification

Reasoning
~~~~~~~~~

Tasks that require integrating multiple cues or rules to arrive at a
decision under uncertainty.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HierarchicalReasoning
   ProbabilisticReasoning

Motor
~~~~~

Reaching, anti-reaching, and evidence-accumulation tasks that produce
continuous motor outputs.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AntiReach
   Reaching1D
   EvidenceAccumulation


Utilities
---------

Duration distributions for sampling variable-length phases:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   TruncExp
   UniformDuration

Dataset transforms applied around :meth:`Task.batch_sample`:

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Transform
   TransformIT

Helper functions for periods, label arrays, and rate conversion:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   initialize
   initialize2
   interval_of
   period_to_arr
   firing_rate


Concepts
--------

Trial generation
~~~~~~~~~~~~~~~~

A :class:`Task` produces one trial as follows:

1. Construct a :class:`Context`, seeded by ``jax.random.fold_in(seed, index)``
   when ``Task`` was given a ``seed``.
2. Call ``trial_init(ctx)`` (or :meth:`Task.trial_init` for subclasses) to
   populate trial-level state — ground truth, stimulus identity, coherence,
   etc.
3. Compute total duration with a dry-run pass over the phase tree (so
   variable-duration phases can read state set by ``trial_init``).
4. Allocate ``ctx.inputs`` of shape ``(T, num_inputs)`` and ``ctx.outputs``
   either ``(T,)`` (categorical mode) or ``(T, num_outputs)`` (vector mode).
5. Walk the phase tree a second time, with each phase calling
   :meth:`Phase.encode_inputs` and :meth:`Phase.encode_outputs` to fill its
   slice of the buffers.

:meth:`Task.batch_sample` ``vmap`` s this process, producing batches whose
keys differ by ``fold_in`` of the trial index so batches are reproducible.

Sampling APIs and tensor shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A configured :class:`Task` exposes three sampling entry points. The shapes
below assume ``num_inputs == task.num_inputs`` and ``num_outputs ==
task.num_outputs``; ``T`` is the per-trial timestep count.

================================================  =====================================  ============================================
Method                                            Returns                                 Shapes
================================================  =====================================  ============================================
:meth:`Task.sample_trial(index)`                  ``(X, Y, info)``                       ``X: (T, num_inputs)``, ``Y: (T,)`` or ``(T, num_outputs)``
:meth:`Task.sample(index)` / ``task[index]``      ``(X, Y)``                             same as above (JIT-compiled)
:meth:`Task.batch_sample(B)`                      ``(X, Y)``                             ``X: (T, B, num_inputs)``, ``Y: (T, B)`` or ``(T, B, num_outputs)``
:meth:`Task.batch_sample(B, time_first=False)`    ``(X, Y)``                             ``X: (B, T, num_inputs)``, ``Y: (B, T)`` or ``(B, T, num_outputs)``
:meth:`Task.batch_sample(B, return_meta=True)`    ``(X, Y, meta)``                        as above; ``meta`` is task-defined
================================================  =====================================  ============================================

The third value returned from :meth:`Task.sample_trial` is a dictionary
with the following keys:

- ``phase_history`` — list of ``(name, start, end)`` tuples logging each
  phase's contribution to the timeline
- ``trial_state`` — copy of the user state set via ``trial_init`` (e.g.
  ``ground_truth``, ``coherence``)
- ``dt`` — the resolved time step (from ``brainstate.environ.get_dt()``)
- ``index`` — the trial index requested

To customize the metadata returned by ``batch_sample(..., return_meta=True)``,
override :meth:`Task.get_trial_meta` in your subclass.

Output modes
~~~~~~~~~~~~

- ``'categorical'`` (default): ``ctx.outputs`` has shape ``(T,)`` and holds
  integer labels. Phases set the ``'label'`` key in their ``outputs=`` dict.
- ``'vector'``: ``ctx.outputs`` has shape ``(T, num_outputs)``. Phases set
  each output feature by name (e.g. ``'direction'``, ``'fixation_out'``).
  Use this for continuous-report tasks such as
  :class:`DelayDirectionReproduction`.

Declarative phase shape conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A value spec for ``inputs=`` can be a constant or a callable
``f(ctx, feature) -> array``. The encoded value is broadcast into the
phase's slice of ``ctx.inputs`` according to its shape:

- scalar → constant for every timestep and feature unit
- 1-D, shape ``(feature.num,)`` → broadcast along the time axis
- 2-D, shape ``(duration, feature.num)`` → written directly

For ``outputs=`` the conventions depend on the output mode:

- Categorical (``ctx.outputs.ndim == 1``): use the ``'label'`` key.
  Accepts a scalar (constant label for the phase) or a 1-D array of shape
  ``(duration,)`` (time-varying labels). Features other than ``'label'`` are
  ignored.
- Vector (``ctx.outputs.ndim == 2``): write per-output-feature; accept
  ``(feature.num,)`` (broadcast along time) or ``(duration, feature.num)``.

The ``noise=`` field maps a feature name to a ``sigma`` Quantity in units of
``ms**0.5``. Noise is sampled fresh per phase and scaled by
``1 / sqrt(dt)`` so the resulting signal variance is invariant under changes
of ``dt``.

Feature index management
~~~~~~~~~~~~~~~~~~~~~~~~

Features expose ``.i``, a Python ``slice`` into the flat input/output vector.
When a feature is composed via ``a + b``, its ``_start``/``_end`` are
shifted automatically, so phase encoders can write ``ctx.inputs[..., feat.i]``
without bookkeeping. Composition is immutable: both operands are copied
before being shifted.

Reproducibility
~~~~~~~~~~~~~~~

A :class:`Task` constructed with ``seed=N`` derives each trial's key as
``jax.random.fold_in(jax.random.PRNGKey(N), trial_index)``. This makes
``task.sample(i)`` deterministic and ``task.batch_sample(B, start_index=k)``
reproducible and non-overlapping across calls. If ``seed`` is omitted, trials
draw fresh randomness from ``brainstate``'s default RNG.

Time step
~~~~~~~~~

All durations are resolved against the *currently active* time step,
``brainstate.environ.get_dt()``. The same task can be re-sampled at a finer
or coarser ``dt`` simply by wrapping it in a ``brainstate.environ.context``;
see :doc:`../cogtask/01_quickstart` for a worked example.


Variable-length trial sequences
-------------------------------

.. note::
   **Status — partially supported / under active development.** Per-phase
   variable durations, ``If``/``Switch``/``While``, and single-trial JIT
   work today; uniform-length ``batch_sample`` is a hard requirement and
   first-class mask-based batching is *planned*. The API for automatic
   padding/masking is **not yet stable** and may change.

What works today:

- :class:`TruncExp` and :class:`UniformDuration` as callables sampled in
  ``trial_init`` and stored in :class:`Context`.
- :class:`If`, :class:`Switch`, and :class:`While` for data-dependent
  control flow within a single trial. Their *upper-bound* duration sets the
  trial's tensor size.
- ``task.sample(index)`` — JIT-compiled per trial.

Current limitation: :meth:`Task.batch_sample` uses ``vmap`` over the trial
index, so every trial in a batch must produce buffers of *identical* shape.
Variable lengths *across the batch axis* are not yet expressible in the
JIT/``vmap`` path.

Planned design: a fixed ``T_max`` per task, padded buffers, and a returned
``mask`` of shape ``(T_max, B)`` marking the live region. A
``masked_loss`` helper is part of the same roadmap.

See :doc:`../cogtask/03_variable_length_trials` for worked workarounds
available today (gated encoders, bucketed batching, explicit mask channels)
and extension points if you need to experiment before the official API
lands.
