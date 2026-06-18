# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Task class for composable cognitive task construction."""

import contextlib

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import jax.random
import numpy as np

from typing import Optional, Callable, Tuple, Dict, Any, Literal
from .context import Context
from .feature import Feature, FeatureSet
from .phase import (
    Phase,
    Sequence,
    Repeat,
    Parallel,
    execute_phase,
    execute_phase_packed,
    phase_tree_is_variable,
)

__all__ = ['Task', 'create_task']

# categorical for classification task like dms, vector for regression like MemoryPro
OutputMode = Literal['categorical', 'vector']

# Sentinel for "no explicit seed". A Task without a seed will draw fresh
# randomness from brainstate's default RNG on every call.
_NO_SEED = object()


def _drop_string_leaves(meta):
    """Recursively drop string/bytes leaves from a meta structure.

    ``batch_sample(return_meta=True)`` stacks per-trial meta with ``vmap2``, which
    requires every leaf to be a JAX type. Tasks that fall back to the default
    ``get_trial_meta`` return the whole ``trial_state`` dict, which can include
    descriptive strings (e.g. ``output_mode``). Such constant strings cannot be
    batched, so they are dropped from the batched meta; numeric fields are kept.
    """
    if isinstance(meta, dict):
        return {k: _drop_string_leaves(v) for k, v in meta.items()
                if not isinstance(v, (str, bytes))}
    if isinstance(meta, (list, tuple)):
        kept = [_drop_string_leaves(v) for v in meta if not isinstance(v, (str, bytes))]
        return type(meta)(kept)
    return meta


class Task:
    """
    A cognitive task composed of phases.

    The Task class orchestrates phase execution and provides a dataset interface
    for integration with DataLoaders. Supports both instance-based and class-based
    definition patterns.

    Class-Based Definition
    ----------------------
    Subclasses can define tasks by overriding class attributes and methods:

    - Class attributes: t_fixation, t_sample, t_delay, num_stimuli, etc.
    - define_features(): Return (input_features, output_features)
    - define_phases(): Return the phase structure
    - trial_init(ctx): Initialize trial-level state

    Examples
    --------
    Instance-based (traditional):

    >>> import brainunit as u
    >>> from braintools.cogtask import Feature, Fixation, Stimulus, Response, circular, label
    >>> fix = Feature(1, 'fixation')
    >>> stim = Feature(8, 'stimulus')
    >>> choice = Feature(2, 'choice')
    >>> task = Task(
    ...     phases=(
    ...         Fixation(100 * u.ms, inputs={'fixation': 1.0})
    ...         >> Stimulus(2000 * u.ms, inputs={'stimulus': circular('direction')})
    ...         >> Response(100 * u.ms, outputs={'label': label('ground_truth')})
    ...     ),
    ...     input_features=fix + stim,
    ...     output_features=fix + choice,
    ...     trial_init=lambda ctx: ctx.update(
    ...         ground_truth=ctx.rng.choice(2),
    ...         direction=ctx.rng.uniform(0, 2 * 3.14159)
    ...     )
    ... )

    Class-based (new):

    >>> class MyTask(Task):
    ...     t_fixation = 300 * u.ms
    ...     num_stimuli = 8
    ...
    ...     def define_features(self):
    ...         fix = Feature(1, 'fixation')
    ...         stim = Feature(self.num_stimuli, 'stimulus')
    ...         return fix + stim, fix + Feature(2, 'response')
    ...
    ...     def define_phases(self):
    ...         return (Fixation(self.t_fixation, inputs={'fixation': 1.0})
    ...                 >> Response(100 * u.ms, outputs={'label': label('ground_truth')}))
    ...
    ...     def trial_init(self, ctx):
    ...         ctx['ground_truth'] = ctx.rng.choice(2)
    ...
    >>> task = MyTask(num_stimuli=16, seed=42)

    Parameters
    ----------
    phases : Phase, optional
        The phase structure. If None, uses define_phases() method.
    input_features : Feature or FeatureSet, optional
        Input feature definitions. If None, uses define_features() method.
    output_features : Feature or FeatureSet, optional
        Output feature definitions. If None, uses define_features() method.
    trial_init : Callable[[Context], None], optional
        Function called at the start of each trial. If None and phases is None,
        uses the trial_init() method.
    name : str, optional
        Task name (defaults to class name).
    dt : float or Quantity, optional
        Time step used to resolve phase durations into a timestep count. If
        ``None`` (default), the ambient ``brainstate.environ.get_dt()`` is used,
        preserving the previous behaviour. When set, the value is pinned for the
        whole of trial generation so buffer sizes and the reported ``dt`` stay
        consistent regardless of the ambient environment.
    **kwargs
        Override class attributes (e.g., t_fixation=500*u.ms, num_stimuli=16).
    """

    def __init__(
        self,
        phases: Phase = None,
        input_features=None,
        output_features=None,
        trial_init: Optional[Callable[[Context], None]] = None,
        name: Optional[str] = None,
        output_mode: OutputMode = 'categorical',
        seed: Optional[int] = None,
        num_classes: Optional[int] = None,
        dt=None,
        **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if phases is None:
            input_features, output_features = self.define_features()
            phases = self.define_phases()
            trial_init = self._class_trial_init

        self.phases = phases
        # Accept a lone Feature in place of a FeatureSet: phases look features
        # up by name (``name in features``) and slice them (``features[name]``),
        # which only FeatureSet supports. Wrap so single-channel tasks work.
        self.input_features = FeatureSet(input_features) if isinstance(input_features, Feature) else input_features
        self.output_features = FeatureSet(output_features) if isinstance(output_features, Feature) else output_features

        # When phases are supplied directly (instance-based path), features are
        # mandatory: the phases bind feature slices by name and fail with a
        # cryptic ``NoneType`` error otherwise. Surface a clear message instead.
        if self.input_features is None or self.output_features is None:
            raise ValueError(
                "When `phases` is provided, both `input_features` and "
                "`output_features` must be given (got "
                f"input_features={self.input_features!r}, "
                f"output_features={self.output_features!r})."
            )

        self._trial_init_func = trial_init
        self.name = name or self.__class__.__name__

        if output_mode not in ('categorical', 'vector'):
            raise ValueError(f"Unknown output_mode: {output_mode}")
        self.output_mode: OutputMode = output_mode

        # Number of categorical classes for the 1-D label target. Independent
        # of the output-feature dimensionality (see the ``num_classes``
        # property). Set explicitly via the ``num_classes=`` argument or by a
        # subclass assigning ``self._num_classes`` before ``super().__init__``;
        # falls back to ``num_outputs`` when left unset.
        self._num_classes: Optional[int] = (
            num_classes if num_classes is not None else getattr(self, '_num_classes', None)
        )

        self.seed = seed
        self._root_key: Optional[jax.Array] = (
            jax.random.PRNGKey(seed) if seed is not None else None
        )

        # Task-level time step. ``None`` means "defer to the ambient
        # brainstate.environ dt". When set, it is pinned around trial
        # generation (see ``_dt_environ``) so phase durations and buffer sizes
        # are computed against this dt rather than the global one.
        self._dt = dt

        self._bind_features_to_phases(phases)

        # Variable-length detection: any phase in the tree that declares
        # ``is_variable = True`` (VariableDuration, If/Switch/While, …)
        # promotes the task to packed mode. Fixed tasks keep the original
        # fast path with no buffer-size overhead.
        self._is_variable_length: bool = phase_tree_is_variable(phases)

    def _trial_key(self, index: int, key: Optional[jax.Array] = None) -> Optional[jax.Array]:
        """Derive a per-trial PRNG key from ``seed`` and ``index``.

        Returns ``None`` if neither a seed nor an explicit key is available
        (in which case ``Context`` will fall back to brainstate's default).
        """
        if key is not None:
            return key
        if self._root_key is not None:
            return jax.random.fold_in(self._root_key, jnp.asarray(index, dtype=jnp.uint32))
        return None

    def _class_trial_init(self, ctx: Context) -> None:
        self.trial_init(ctx)

    def define_features(self) -> Tuple[Any, Any]:
        """
        Define input and output features.

        Override in subclass for class-based task definition.

        Returns
        -------
        input_features : Feature or FeatureSet
            Input feature definitions.
        output_features : Feature or FeatureSet
            Output feature definitions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement define_features() "
            "for class-based task definition, or pass phases/features to __init__."
        )

    def define_phases(self) -> Phase:
        """
        Define the phase structure.

        Override in subclass for class-based task definition.

        Returns
        -------
        Phase
            The task phase structure (single phase or composition).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement define_phases() "
            "for class-based task definition, or pass phases to __init__."
        )

    def trial_init(self, ctx: Context) -> None:
        """
        Initialize trial-level state.

        Override in subclass to set up trial parameters like ground_truth,
        stimulus indices, etc.

        Parameters
        ----------
        ctx : Context
            Trial context to populate with state.
        """
        pass  # Default: no initialization

    @property
    def num_inputs(self) -> int:
        return self.input_features.num

    @property
    def num_outputs(self) -> int:
        return self.output_features.num

    @property
    def num_classes(self) -> Optional[int]:
        """Number of classes for the categorical 1-D label target.

        In ``output_mode='categorical'`` the target ``Y`` is a 1-D integer
        label array, and the size of a classifier head is the number of
        distinct label values — which is **independent of**
        :attr:`num_outputs` (the summed output-feature dimensionality).
        ``num_outputs`` only coincidentally equals the class count in the
        default configurations and drifts apart when e.g. ``cue_dim`` changes,
        so prefer ``num_classes`` when wiring a categorical model head.

        Returns the explicit value passed as ``num_classes=`` when set,
        otherwise falls back to :attr:`num_outputs` (preserving historical
        behaviour). Returns ``None`` in ``output_mode='vector'``, where the
        target is a continuous vector of width :attr:`num_outputs`.
        """
        if self.output_mode != 'categorical':
            return None
        return self._num_classes if self._num_classes is not None else self.num_outputs

    @property
    def dt(self):
        """Task time step.

        Returns the value passed as ``dt=`` at construction, or falls back to
        the ambient ``brainstate.environ.get_dt()`` when none was given.
        """
        return self._dt if self._dt is not None else brainstate.environ.get_dt()

    def _dt_environ(self):
        """Pin ``brainstate.environ`` to the task's ``dt`` during sampling.

        Returns a no-op context manager when ``dt`` was not set, so the ambient
        environment dt is used (the historical behaviour). When set, phases read
        this dt via ``ctx.dt`` so durations/buffer sizes match the reported dt.
        """
        if self._dt is None:
            return contextlib.nullcontext()
        return brainstate.environ.context(dt=self._dt)

    @property
    def is_variable_length(self) -> bool:
        """True if any phase in the tree has ``is_variable = True``.

        Variable-length tasks allocate trial buffers of size
        ``max_trial_duration`` and return a per-timestep mask alongside
        ``X``/``Y`` from :meth:`batch_sample` (use ``return_mask=True``).
        """
        return self._is_variable_length

    def max_trial_duration(self, ctx: Optional[Context] = None) -> int:
        """Static upper bound on the trial's timestep count.

        For fixed tasks this equals the sum of per-phase ``get_duration``
        outputs. For variable-length tasks each phase contributes its
        ``max_steps`` (e.g. ``ceil(max_duration / dt)`` for
        :class:`VariableDuration`). The result is a Python ``int`` and is
        safe to use as a static buffer dimension under JIT/vmap.
        """
        with self._dt_environ():
            if ctx is None:
                ctx = Context(key=self._root_key)
            return int(self.phases.max_steps(ctx))

    def sample_trial(
        self,
        index: int = 0,
        key: Optional[jax.Array] = None
    ):
        """Generate one trial.

        Parameters
        ----------
        index : int
            Trial index, made available to ``trial_init`` as ``ctx['trial_index']``.
            If the task was constructed with ``seed=...``, the per-trial RNG key
            is ``jax.random.fold_in(PRNGKey(seed), index)``, so reproducibility is
            keyed on ``(seed, index)``.
        key : jax.Array, optional
            Explicit PRNG key. Overrides the (seed, index) derivation when given.
        """
        with self._dt_environ():
            trial_key = self._trial_key(index, key)
            ctx = Context(key=trial_key)
            ctx['trial_index'] = index

            if self._trial_init_func:
                self._trial_init_func(ctx)

            if self._is_variable_length:
                return self._sample_trial_packed(ctx, index)

            # Dry-run pass: compute total duration without mutating ctx state. Note
            # that ``get_duration`` must be pure with respect to ctx.rng — predicate
            # functions and variable-duration phases should read state set by
            # ``trial_init``, not sample new randomness.
            total_duration = self._compute_duration(ctx)

            # allocate buffers
            ctx.inputs = jnp.zeros((total_duration, self.num_inputs), dtype=jnp.float32)

            if self.output_mode == 'categorical':
                ctx.outputs = jnp.zeros((total_duration,), dtype=jnp.int32)
            else:
                ctx.outputs = jnp.zeros((total_duration, self.num_outputs), dtype=jnp.float32)

            # expose mode to phases
            ctx['output_mode'] = self.output_mode

            # second pass
            ctx.current_step = 0
            ctx.phase_history.clear()
            execute_phase(self.phases, ctx)

            info = {
                'phase_history': ctx.phase_history.copy(),
                'trial_state': ctx.state,
                'dt': self.dt,
                'index': index,
                'mask': None,
            }
            return ctx.inputs, ctx.outputs, info

    def _sample_trial_packed(self, ctx: Context, index: int):
        """Variable-length trial generation.

        Buffers are sized by ``max_trial_duration`` (a Python int derived
        from each phase's ``max_steps``) so shapes stay static under JIT
        and ``vmap``. A boolean ``mask`` of length ``max_trial_duration``
        marks the valid timesteps written by the phases; positions past
        the actual trial length are zeroed in ``X``/``Y`` and ``False`` in
        the mask.
        """
        max_dur = self.max_trial_duration(ctx)

        ctx.inputs = jnp.zeros((max_dur, self.num_inputs), dtype=jnp.float32)
        if self.output_mode == 'categorical':
            ctx.outputs = jnp.zeros((max_dur,), dtype=jnp.int32)
        else:
            ctx.outputs = jnp.zeros((max_dur, self.num_outputs), dtype=jnp.float32)
        ctx.mask = jnp.zeros((max_dur,), dtype=jnp.bool_)
        ctx.t_cursor = jnp.int32(0)

        ctx['output_mode'] = self.output_mode

        ctx.current_step = 0
        ctx.phase_history.clear()
        execute_phase_packed(self.phases, ctx)

        info = {
            'phase_history': ctx.phase_history.copy(),
            'trial_state': ctx.state,
            'dt': self.dt,
            'index': index,
            'mask': ctx.mask,
            't_cursor': ctx.t_cursor,
        }
        return ctx.inputs, ctx.outputs, info

    def _compute_duration(self, ctx: Context) -> int:
        """Compute total trial duration without leaking state into ctx.

        We snapshot ``ctx._state`` and ``ctx.phase_history`` around the call so
        any incidental writes are undone. ``ctx.rng`` is *not* snapshotted; see
        the contract in ``sample_trial``.
        """
        state_snapshot = dict(ctx._state)
        history_snapshot = list(ctx.phase_history)
        original_step = ctx.current_step
        try:
            duration = self.phases.get_duration(ctx)
        finally:
            ctx._state = state_snapshot
            ctx.phase_history = history_snapshot
            ctx.current_step = original_step
        return int(duration)

    def _bind_features_to_phases(self, phase: Phase) -> None:
        """Recursively bind the task's feature sets to every DeclarativePhase
        in the tree, regardless of compound type (Sequence, Repeat, Parallel,
        If, Switch, While, …).
        """
        from .phase import DeclarativePhase

        if isinstance(phase, DeclarativePhase):
            # Pass the *explicit* class count (None unless the user declared
            # it) so labels are validated only against a sound upper bound.
            phase.bind_features(
                self.input_features, self.output_features,
                num_classes=self._num_classes if self.output_mode == 'categorical' else None,
            )
        for child in phase.children():
            self._bind_features_to_phases(child)

    @brainstate.transform.jit(static_argnums=0)
    def sample(self, index: int):
        return self[index]

    @brainstate.transform.jit(static_argnums=(0, 1), static_argnames=['time_first', 'return_meta', 'return_mask'])
    def batch_sample(
        self,
        size: int,
        /,
        time_first: bool = True,
        return_meta: bool = False,
        start_index: int = 0,
        return_mask: bool = False,
    ):
        """Sample a batch of ``size`` trials with indices ``start_index..start_index+size-1``.

        When the task was constructed with ``seed=...``, each trial in the batch
        uses ``jax.random.fold_in(PRNGKey(seed), start_index + i)`` so calling
        ``batch_sample`` with the same ``start_index`` is reproducible, and
        successive calls with different ``start_index`` produce non-overlapping
        batches.

        Parameters
        ----------
        return_mask : bool, optional
            If True, also return a ``(T, B)`` (or ``(B, T)``) boolean mask
            of valid timesteps. Required for variable-length tasks if you
            want to know which trailing positions are padding. The mask is
            always-True for fixed-length tasks.
        """
        indices = jnp.arange(size, dtype=jnp.int32) + jnp.asarray(start_index, dtype=jnp.int32)
        out_axes_xy = 1 if time_first else 0
        mask_axis = 1 if time_first else 0
        if return_mask:
            if return_meta:
                def _one_mask_meta(i):
                    X, Y, mask, meta = self.__getitem_with_mask_and_meta__(i)
                    return X, Y, mask, _drop_string_leaves(meta)
                X, Y, mask, meta = brainstate.transform.vmap2(
                    _one_mask_meta,
                    out_axes=(out_axes_xy, out_axes_xy, mask_axis, 0)
                )(indices)
                return X, Y, mask, meta
            else:
                X, Y, mask = brainstate.transform.vmap2(
                    lambda i: self.__getitem_with_mask__(i),
                    out_axes=(out_axes_xy, out_axes_xy, mask_axis)
                )(indices)
                return X, Y, mask
        if return_meta:
            def _one_meta(i):
                X, Y, meta = self.__getitem_with_meta__(i)
                return X, Y, _drop_string_leaves(meta)
            X, Y, meta = brainstate.transform.vmap2(
                _one_meta,
                out_axes=(out_axes_xy, out_axes_xy, 0)
            )(indices)
            return X, Y, meta
        else:
            X, Y = brainstate.transform.vmap2(
                lambda i: self[i], out_axes=(out_axes_xy, out_axes_xy)
            )(indices)
            return X, Y

    def __getitem__(self, index: int):
        X, Y, _ = self.sample_trial(index)
        return X, Y

    def __getitem_with_mask__(self, index: int):
        X, Y, info = self.sample_trial(index)
        mask = info.get('mask')
        if mask is None:
            # Fixed-length task: mask is all True for the trial's length.
            mask = jnp.ones((X.shape[0],), dtype=jnp.bool_)
        return X, Y, mask

    def __getitem_with_mask_and_meta__(self, index: int):
        X, Y, info = self.sample_trial(index)
        mask = info.get('mask')
        if mask is None:
            mask = jnp.ones((X.shape[0],), dtype=jnp.bool_)
        meta = self.get_trial_meta(info["trial_state"])
        return X, Y, mask, meta

    def __repr__(self) -> str:
        return (
            f"Task(name={self.name}, inputs={self.num_inputs}, "
            f"outputs={self.num_outputs}, output_mode={self.output_mode})"
        )

    # a version that returns X, Y and sample/test directions
    def get_trial_meta(self, trial_state: Dict[str, Any]) -> Any:
        return trial_state

    def __getitem_with_meta__(self, index: int):
        X, Y, info = self.sample_trial(index)
        meta = self.get_trial_meta(info["trial_state"])
        return X, Y, meta


def create_task(
    phases: Phase,
    input_features,
    output_features,
    trial_init: Optional[Callable[[Context], None]] = None,
    name: Optional[str] = None,
    seed: Optional[int] = None,
    output_mode: OutputMode = 'categorical',
    num_classes: Optional[int] = None,
    dt=None,
) -> Task:  # noqa: D401
    """
    Convenience factory for creating tasks.

    Parameters
    ----------
    phases : Phase
        The phase structure.
    input_features : Feature or FeatureSet
        Input feature definitions.
    output_features : Feature or FeatureSet
        Output feature definitions.
    trial_init : Callable, optional
        Trial initialization function.
    name : str, optional
        Task name.
    seed : int, optional
        Random seed.
    output_mode: OutputMode, optional
        Output mode.
    num_classes : int, optional
        Number of categorical classes for the 1-D label target. Used to size a
        classifier head in ``output_mode='categorical'``; falls back to
        ``num_outputs`` when unset. Ignored in ``output_mode='vector'``.
    dt : float or Quantity, optional
        Time step used to resolve phase durations. Defaults to
        ``brainstate.environ.get_dt()`` when not given.

    Returns
    -------
    Task
        Configured task instance.

    Note
    ----
    The time step (dt) defaults to brainstate.environ.get_dt() when not set.

    Examples
    --------
    >>> import brainunit as u
    >>> from braintools.cogtask import Feature, Fixation, Stimulus, Response, circular, label
    >>> task = create_task(
    ...     phases=(
    ...         Fixation(100 * u.ms, inputs={'fixation': 1.0})
    ...         >> Stimulus(500 * u.ms, inputs={'stimulus': circular('direction')})
    ...         >> Response(100 * u.ms, outputs={'label': label('ground_truth')})
    ...     ),
    ...     input_features=Feature(1, 'fixation') + Feature(2, 'stimulus'),
    ...     output_features=Feature(1, 'fixation') + Feature(2, 'choice'),
    ...     trial_init=lambda ctx: ctx.update(
    ...         ground_truth=ctx.rng.choice(2),
    ...         direction=ctx.rng.uniform(0, 2 * 3.14159)
    ...     ),
    ... )
    """
    return Task(phases, input_features, output_features, trial_init, name,
                output_mode=output_mode, seed=seed, num_classes=num_classes, dt=dt)
