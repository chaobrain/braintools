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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import jax.random
import numpy as np

from typing import Optional, Callable, Tuple, Dict, Any, Literal
from .context import Context
from .phase import Phase, Sequence, Repeat, Parallel, execute_phase

__all__ = ['Task', 'create_task']

# categorical for classification task like dms, vector for regression like MemoryPro
OutputMode = Literal['categorical', 'vector']

# Sentinel for "no explicit seed". A Task without a seed will draw fresh
# randomness from brainstate's default RNG on every call.
_NO_SEED = object()


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

    >>> task = Task(
    ...     phases=(
    ...         Fixation(100 * u.ms)
    ...         >> Stimulus(2000 * u.ms, feature=stim, encoder=circular_encoder())
    ...         >> Response(100 * u.ms, output_feature=choice)
    ...     ),
    ...     input_features=fix + stim,
    ...     output_features=fix + choice,
    ...     trial_init=lambda ctx: ctx.update(
    ...         ground_truth=ctx.rng.choice(2),
    ...         direction=ctx.rng.uniform(0, 2*np.pi)
    ...     )
    ... )

    Class-based (new):

    >>> class MyTask(Task):
    ...     t_fixation = 300 * u.ms
    ...     num_stimuli = 8
    ...
    ...     def define_features(self):
    ...         fix = Feature(1, 40*u.Hz, 'fixation')
    ...         stim = Feature(self.num_stimuli, 40*u.Hz, 'stimulus')
    ...         return fix + stim, fix + Feature(2, 40*u.Hz, 'response')
    ...
    ...     def define_phases(self):
    ...         return FixationPhase(self.t_fixation) >> ResponsePhase()
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
        **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if phases is None:
            input_features, output_features = self.define_features()
            phases = self.define_phases()
            trial_init = self._class_trial_init

        self.phases = phases
        self.input_features = input_features
        self.output_features = output_features
        self._trial_init_func = trial_init
        self.name = name or self.__class__.__name__

        if output_mode not in ('categorical', 'vector'):
            raise ValueError(f"Unknown output_mode: {output_mode}")
        self.output_mode: OutputMode = output_mode

        self.seed = seed
        self._root_key: Optional[jax.Array] = (
            jax.random.PRNGKey(seed) if seed is not None else None
        )

        self._bind_features_to_phases(phases)

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
    def dt(self):
        return brainstate.environ.get_dt()

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
        trial_key = self._trial_key(index, key)
        ctx = Context(key=trial_key)
        ctx['trial_index'] = index

        if self._trial_init_func:
            self._trial_init_func(ctx)

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
            phase.bind_features(self.input_features, self.output_features)
        for child in phase.children():
            self._bind_features_to_phases(child)

    @brainstate.transform.jit(static_argnums=0)
    def sample(self, index: int):
        return self[index]

    @brainstate.transform.jit(static_argnums=(0, 1), static_argnames=['time_first', 'return_meta'])
    def batch_sample(self, size: int, /, time_first: bool = True, return_meta: bool = False, start_index: int = 0):
        """Sample a batch of ``size`` trials with indices ``start_index..start_index+size-1``.

        When the task was constructed with ``seed=...``, each trial in the batch
        uses ``jax.random.fold_in(PRNGKey(seed), start_index + i)`` so calling
        ``batch_sample`` with the same ``start_index`` is reproducible, and
        successive calls with different ``start_index`` produce non-overlapping
        batches.
        """
        indices = jnp.arange(size, dtype=jnp.int32) + jnp.asarray(start_index, dtype=jnp.int32)
        out_axes_xy = 1 if time_first else 0
        if return_meta:
            X, Y, meta = brainstate.transform.vmap2(
                lambda i: self.__getitem_with_meta__(i),
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

    Returns
    -------
    Task
        Configured task instance.

    Note
    ----
    The time step (dt) is obtained from brainstate.environ.get_dt().

    Examples
    --------
    >>> task = create_task(
    ...     phases=Fixation(100*u.ms) >> Stimulus(500*u.ms) >> Response(100*u.ms),
    ...     input_features=Feature('fix', 1) + Feature('stim', 2),
    ...     output_features=Feature('fix', 1) + Feature('choice', 2),
    ...     num_trial=1000
    ... )
    """
    return Task(phases, input_features, output_features, trial_init, name, output_mode=output_mode, seed=seed)
