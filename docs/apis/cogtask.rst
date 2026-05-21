``braintools.cogtask`` module
=============================

.. currentmodule:: braintools.cogtask
.. automodule:: braintools.cogtask

A modular, composable framework for constructing cognitive tasks for neural
network training and computational neuroscience simulations.

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
  :class:`While` for trial-by-trial branching and variable-length tasks
- **A feature-encoding system** that maps trial state into input/output
  channels via :class:`Feature`/:class:`FeatureSet` and value-spec encoders
  (``one_hot``, ``circular``, ``von_mises``, ``gaussian``, ``cos_sin``, ...)
- **A library of pre-built tasks** spanning decision making, working memory,
  reasoning, and motor control, drawn from systems-neuroscience literature
- **JIT/``vmap``-friendly trial generation** through :meth:`Task.sample` and
  :meth:`Task.batch_sample`, designed to integrate cleanly with
  `brainstate <https://brainstate.readthedocs.io/>`_ and JAX training loops

Quick Start
-----------

Using a pre-built task:

.. code-block:: python

   import brainunit as u
   from braintools.cogtask import PerceptualDecisionMaking

   task = PerceptualDecisionMaking(t_stimulus=1500 * u.ms, num_choices=2, seed=0)
   X, Y = task.batch_sample(32)
   # X: (T, B, num_inputs)  Y: (T, B) categorical labels

Building a custom task from phases:

.. code-block:: python

   import brainunit as u
   from braintools.cogtask import (
       Task, Feature, concat,
       Fixation, Stimulus, Delay, Response,
       circular,
   )

   fix = Feature(1, 'fixation')
   stim = Feature(8, 'stimulus')
   choice = Feature(2, 'choice')

   task = Task(
       phases=concat([
           Fixation(100 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
           Stimulus(500 * u.ms,
                    inputs={'fixation': 1.0,
                            'stimulus': circular('direction', 'coherence')},
                    outputs={'label': 0}),
           Delay(500 * u.ms, inputs={'fixation': 1.0}, outputs={'label': 0}),
           Response(100 * u.ms,
                    inputs={'fixation': 0.0},
                    outputs={'label': lambda ctx, f: ctx['ground_truth'] + 1}),
       ]),
       input_features=fix + stim,
       output_features=fix + choice,
       trial_init=lambda ctx: ctx.update(
           ground_truth=ctx.rng.choice(2),
           coherence=51.2,
           direction=ctx.rng.uniform(0, 6.2832),
       ),
       seed=0,
   )

   X, Y, info = task.sample_trial(0)


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
other :class:`Task`, optionally passing ``seed=`` for reproducibility:

.. code-block:: python

   from braintools.cogtask import DelayMatchSample

   task = DelayMatchSample(t_delay=2000 * u.ms, num_stimuli=16, seed=0)
   X, Y = task.batch_sample(64)

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

Output modes
~~~~~~~~~~~~

- ``'categorical'`` (default): ``ctx.outputs`` has shape ``(T,)`` and holds
  integer labels. Phases set the ``'label'`` key in their ``outputs=`` dict.
- ``'vector'``: ``ctx.outputs`` has shape ``(T, num_outputs)``. Phases set
  each output feature by name (e.g. ``'direction'``, ``'fixation_out'``).
  Use this for continuous-report tasks such as
  :class:`DelayDirectionReproduction`.

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
reproducible and non-overlapping across calls.
