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

"""Phase base class and composition operators for cognitive task construction."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Callable, Any

import brainunit as u
import jax
import jax.numpy as jnp

from .context import Context
from .feature import Feature, FeatureSet
from ._typing import Data, Duration

__all__ = [
    'Phase',
    'Sequence',
    'Repeat',
    'Parallel',
    'concat',
    'execute_phase',
    'execute_phase_packed',
    'phase_tree_is_variable',

    # Declarative phases
    'DeclarativePhase',
    'Fixation',
    'Delay',
    'Stimulus',
    'Response',
    'Cue',
    'Sample',
    'Test',
    'Recall',
    'Match',
    'Comparison',
    'Blank',
    'VariableDuration',
]


class Phase(ABC):
    """
    Base class for task phases (epochs/periods).

    A phase represents a time interval with specific:
    - Input encoding rules (how to fill input features)
    - Output/target encoding rules (what the expected output should be)
    - Duration (fixed or sampled)

    Phases are composable via:
    - ``>>`` operator: sequential concatenation
    - ``*`` operator: repetition
    - ``|`` operator: parallel composition

    Examples
    --------
    >>> # Sequential composition
    >>> phases = Fixation(100 * u.ms) >> Stimulus(500 * u.ms) >> Response(100 * u.ms)

    >>> # Using concat function
    >>> phases = concat([Fixation(100 * u.ms), Stimulus(500 * u.ms)])

    >>> # Repetition
    >>> repeated = Stimulus(100 * u.ms) * 5  # 5 repetitions

    Parameters
    ----------
    duration : Duration
        Phase duration as Quantity (e.g., 100 * u.ms, 1 * u.second).
    name : str, optional
        Phase name. Defaults to class name.
    """

    # Subclasses set this True when they manage their own children inside
    # ``execute()`` (Sequence, Repeat, Parallel, If, Switch, While). Leaf
    # phases keep it False and rely on encode_inputs/encode_outputs.
    IS_COMPOUND: bool = False

    # Subclasses set this True when their actual step count is only known at
    # trial time (e.g. drawn from ``ctx`` state). Compound phases propagate
    # via ``children()``; the ``phase_tree_is_variable`` walker uses both
    # signals together. Leaf phases keep ``False`` to stay on the static
    # buffer path.
    is_variable: bool = False

    def __init__(
        self,
        duration: Duration,
        name: Optional[str] = None,
    ):
        self._duration = duration
        self.name = name or self.__class__.__name__

    def children(self) -> List['Phase']:
        """Return the immediate child phases of a compound phase.

        Leaf phases return ``[]``. Subclasses like ``Sequence``, ``Repeat``,
        ``Parallel``, ``If``, ``Switch``, ``While`` override this so that the
        ``Task`` can traverse the whole tree to bind features.
        """
        return []

    def get_duration(self, ctx: Context) -> int:
        """
        Resolve duration to integer timesteps.

        Parameters
        ----------
        ctx : Context
            Context with dt and rng for duration sampling.

        Returns
        -------
        int
            Number of timesteps for this phase.
        """
        return max(1, int(self._duration / ctx.dt))

    def max_steps(self, ctx: Context) -> int:
        """Static upper bound on this phase's length in timesteps.

        Must return a **Python int** with no dependence on traced values.
        Used by ``Task`` in variable-length mode to size shape-stable
        buffers. The default delegates to ``get_duration`` which is correct
        for fixed-duration phases. Variable-duration phases (e.g. those
        wrapping ``TruncExp``/``UniformDuration``) override this to return
        the truncation upper bound divided by ``ctx.dt``.

        Parameters
        ----------
        ctx : Context
            A stub or trial context providing ``ctx.dt``. The default
            implementation does not read ``ctx.rng`` or trial state.

        Returns
        -------
        int
            Upper bound on number of timesteps for this phase.
        """
        return int(self.get_duration(ctx))

    def step_count(self, ctx: Context) -> jax.Array:
        """Traced actual length of this phase in timesteps.

        Returns a ``jax.Array`` ``int32`` scalar. May depend on
        ``ctx[...]`` values populated by ``trial_init``. Must satisfy
        ``0 <= step_count(ctx) <= max_steps(ctx)`` for every trial.

        The default returns a static value equal to ``get_duration``; that
        is correct for any phase whose actual length matches its upper
        bound. Variable-duration phases override this to compute the
        traced length from ``ctx`` state without any ``int(...)`` cast.
        """
        return jnp.asarray(self.get_duration(ctx), dtype=jnp.int32)

    @abstractmethod
    def encode_inputs(self, ctx: Context) -> None:
        """
        Fill ctx.inputs[phase_start:phase_end] with input encoding.

        Called once per phase after duration is determined.
        Must modify ctx.inputs in-place.

        Parameters
        ----------
        ctx : Context
            Context with input buffer and trial state.
        """
        pass

    @abstractmethod
    def encode_outputs(self, ctx: Context) -> None:
        """
        Fill ctx.outputs[phase_start:phase_end] with target encoding.

        Called once per phase after duration is determined.
        Must modify ctx.outputs in-place.

        Parameters
        ----------
        ctx : Context
            Context with output buffer and trial state.
        """
        pass

    def on_enter(self, ctx: Context) -> None:
        """Hook called when phase begins. Override for setup logic."""
        pass

    def on_exit(self, ctx: Context) -> None:
        """Hook called when phase ends. Override for cleanup/state updates."""
        pass

    # ========== Composition Operators ==========

    def __rshift__(self, other: 'Phase') -> 'Sequence':
        """
        Sequential concatenation: phase1 >> phase2
        """
        if isinstance(other, Sequence):
            return Sequence(self, *other.phases)
        return Sequence(self, other)

    def __rrshift__(self, other: 'Phase') -> 'Sequence':
        """Support for right-side shift when left is Phase."""
        if isinstance(other, Sequence):
            return Sequence(*other.phases, self)
        return Sequence(other, self)

    def __mul__(self, n: int) -> 'Repeat':
        """
        Repetition: phase * 3 → Repeat(phase, 3)

        Note: this differs from ``Feature.__mul__`` which creates named copies
        of a feature. The two operators share syntax but produce different
        types (``Repeat`` vs ``FeatureSet``).
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("Repeat count must be positive integer")
        return Repeat(self, n)

    def __or__(self, other: 'Phase') -> 'Parallel':
        """
        Parallel composition: phase1 | phase2
        Both phases run simultaneously (for multi-modality).
        """
        return Parallel(self, other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class Sequence(Phase):
    """
    Sequential composition of phases.

    Created via:
    - ``Sequence(phase1, phase2, phase3)``
    - ``phase1 >> phase2 >> phase3``
    - ``concat([phase1, phase2, phase3])``

    Examples
    --------
    >>> seq = Sequence(Fixation(100*u.ms), Stimulus(500*u.ms))
    >>> # Equivalent to:
    >>> seq = Fixation(100*u.ms) >> Stimulus(500*u.ms)
    """

    IS_COMPOUND = True

    def __init__(self, *phases: Phase, name: str = 'Sequence'):
        self.phases = list(phases)
        # Duration is sum of children - computed dynamically
        super().__init__(duration=0 * u.ms, name=name)

    def get_duration(self, ctx: Context) -> int:
        """Get total duration by summing child phases."""
        return sum(p.get_duration(ctx) for p in self.phases)

    def max_steps(self, ctx: Context) -> int:
        return sum(p.max_steps(ctx) for p in self.phases)

    def step_count(self, ctx: Context) -> jax.Array:
        total = jnp.asarray(0, dtype=jnp.int32)
        for p in self.phases:
            total = total + p.step_count(ctx)
        return total

    def encode_inputs(self, ctx: Context) -> None:
        # Delegated to children during execute
        pass

    def encode_outputs(self, ctx: Context) -> None:
        # Delegated to children during execute
        pass

    def execute(self, ctx: Context) -> None:
        """Execute all child phases sequentially."""
        for phase in self.phases:
            execute_phase(phase, ctx)

    def execute_packed(self, ctx: Context) -> None:
        for phase in self.phases:
            execute_phase_packed(phase, ctx)

    def children(self) -> List[Phase]:
        return list(self.phases)

    def __rshift__(self, other: Phase) -> 'Sequence':
        if isinstance(other, Sequence):
            return Sequence(*self.phases, *other.phases)
        return Sequence(*self.phases, other)

    def __iter__(self):
        return iter(self.phases)

    def __len__(self):
        return len(self.phases)

    def __repr__(self) -> str:
        phase_names = [p.name for p in self.phases]
        return f"Sequence({' >> '.join(phase_names)})"


class Repeat(Phase):
    """
    Repeat a phase N times.

    Created via: ``phase * N``

    During execution, the loop iteration is exposed under both
    ``ctx['repeat_index']`` and ``ctx['_repeat_stack']`` (a list, supporting
    nested ``Repeat``). Restoration on exit keeps the outer index intact.

    Examples
    --------
    >>> # Repeat stimulus 5 times
    >>> repeated = Stimulus(100*u.ms) * 5

    >>> # In a sequence
    >>> phases = Fixation(100*u.ms) >> (Stimulus(50*u.ms) * 10) >> Response(100*u.ms)
    """

    IS_COMPOUND = True

    def __init__(self, phase: Phase, count: int, name: str = None):
        self.phase = phase
        self.count = count
        super().__init__(duration=0 * u.ms, name=name or f'Repeat({phase.name}, {count})')

    def get_duration(self, ctx: Context) -> int:
        """Get total duration by multiplying single phase duration."""
        return self.phase.get_duration(ctx) * self.count

    def max_steps(self, ctx: Context) -> int:
        return self.phase.max_steps(ctx) * self.count

    def step_count(self, ctx: Context) -> jax.Array:
        # Conservative: assume every iteration uses the same step_count
        # (Repeat does not currently carry per-iteration variable durations).
        # The wrapped phase's step_count is queried once; under traced mode
        # any randomness inside it is folded into the trial-level RNG.
        return self.phase.step_count(ctx) * jnp.asarray(self.count, dtype=jnp.int32)

    def encode_inputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def encode_outputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def execute(self, ctx: Context) -> None:
        """Execute the phase ``count`` times with a scoped repeat index."""
        prev_index = ctx.get('repeat_index')
        stack = ctx.get('_repeat_stack', [])
        for i in range(self.count):
            ctx['repeat_index'] = i
            stack.append(i)
            ctx['_repeat_stack'] = stack
            execute_phase(self.phase, ctx)
            stack.pop()
        ctx['_repeat_stack'] = stack
        if prev_index is None:
            # Was not set before; remove our scratch entry.
            ctx._state.pop('repeat_index', None)
        else:
            ctx['repeat_index'] = prev_index

    def execute_packed(self, ctx: Context) -> None:
        prev_index = ctx.get('repeat_index')
        stack = ctx.get('_repeat_stack', [])
        for i in range(self.count):
            ctx['repeat_index'] = i
            stack.append(i)
            ctx['_repeat_stack'] = stack
            execute_phase_packed(self.phase, ctx)
            stack.pop()
        ctx['_repeat_stack'] = stack
        if prev_index is None:
            ctx._state.pop('repeat_index', None)
        else:
            ctx['repeat_index'] = prev_index

    def children(self) -> List[Phase]:
        return [self.phase]

    def __repr__(self) -> str:
        return f"Repeat({self.phase.name}, {self.count})"


class Parallel(Phase):
    """
    Parallel composition - phases execute simultaneously.

    Useful for multi-modality tasks where different input channels have
    different encoding patterns during the same time period.

    Created via: ``phase1 | phase2``

    Semantics
    ---------
    - The parent's duration is the maximum of its children's durations.
    - Each child runs through its own ``on_enter`` / ``on_exit`` lifecycle
      bound to *its own* duration (not the parent's max), so noise and
      time-varying labels respect the child's declared length.
    - Inputs from all children are written into the parent slice.
    - Outputs come from the *first* child by convention. To combine outputs
      from multiple children, declare disjoint output features per child and
      compose explicitly.

    Examples
    --------
    >>> # Two stimuli presented simultaneously
    >>> parallel = StimulusA(500*u.ms) | StimulusB(500*u.ms)
    """

    IS_COMPOUND = True

    def __init__(self, *phases: Phase, name: str = 'Parallel'):
        self.phases = list(phases)
        super().__init__(duration=0 * u.ms, name=name)

    def get_duration(self, ctx: Context) -> int:
        """Duration is max of children."""
        return max(p.get_duration(ctx) for p in self.phases)

    def max_steps(self, ctx: Context) -> int:
        return max(p.max_steps(ctx) for p in self.phases)

    def step_count(self, ctx: Context) -> jax.Array:
        counts = [p.step_count(ctx) for p in self.phases]
        out = counts[0]
        for c in counts[1:]:
            out = jnp.maximum(out, c)
        return out

    def encode_inputs(self, ctx: Context) -> None:
        # Delegated through execute
        pass

    def encode_outputs(self, ctx: Context) -> None:
        # Delegated through execute
        pass

    def execute(self, ctx: Context) -> None:
        """Encode each child within its own ``[parent_start, parent_start+dur)``.

        Every child — leaf *or* compound — is dispatched through
        ``execute_phase`` so that nested ``Sequence``/``Repeat``/conditional
        children execute correctly. Routing compound children through
        ``encode_inputs``/``encode_outputs`` directly would be a no-op (those
        classes drive their sub-phases via ``execute``), silently dropping the
        branch. This mirrors ``execute_packed`` and keeps the two paths
        consistent. All children start at ``parent_start``; only the first child
        contributes to the output buffer by convention.
        """
        parent_start = ctx.phase_start
        parent_end = ctx.phase_end
        parent_name = ctx.current_phase

        for i, child in enumerate(self.phases):
            ctx.current_step = parent_start
            if i == 0:
                execute_phase(child, ctx)
            else:
                # Non-first children must not contribute to the output buffer.
                # Stash, encode (inputs + any state), then restore outputs.
                saved_out = ctx.outputs
                execute_phase(child, ctx)
                ctx.outputs = saved_out

        # Restore parent scope and advance past the (longest) child.
        ctx.phase_start = parent_start
        ctx.phase_end = parent_end
        ctx.current_phase = parent_name
        ctx.current_step = parent_end

    def execute_packed(self, ctx: Context) -> None:
        """Packed-mode parallel execution.

        Every child starts at the same ``parent_cursor`` slot; the parent
        cursor advances by the maximum of the children's actual step
        counts. Child input/output writes compose via the leaf merge logic
        in ``_execute_leaf_packed`` (later children inherit earlier slot
        contents outside their active range). Only the first child's
        output writes are propagated (matching the static semantic).
        """
        parent_cursor = ctx.t_cursor
        max_child_actual = jnp.asarray(0, dtype=jnp.int32)

        for i, child in enumerate(self.phases):
            ctx.t_cursor = parent_cursor
            if i == 0:
                execute_phase_packed(child, ctx)
            else:
                # Other children must not advance the output buffer.
                # Stash, encode, then restore output buffer.
                saved_out = ctx.outputs
                execute_phase_packed(child, ctx)
                ctx.outputs = saved_out
            child_actual = ctx.t_cursor - parent_cursor
            max_child_actual = jnp.maximum(max_child_actual, child_actual)

        ctx.t_cursor = parent_cursor + max_child_actual
        ctx.phase_start = parent_cursor
        ctx.phase_end = parent_cursor + max_child_actual

    def children(self) -> List[Phase]:
        return list(self.phases)

    def __or__(self, other: Phase) -> 'Parallel':
        if isinstance(other, Parallel):
            return Parallel(*self.phases, *other.phases)
        return Parallel(*self.phases, other)

    def __repr__(self) -> str:
        phase_names = [p.name for p in self.phases]
        return f"Parallel({' | '.join(phase_names)})"


def execute_phase(phase: Phase, ctx: Context) -> None:
    """
    Execute a single phase, updating context appropriately.

    Compound phases (``IS_COMPOUND=True``) dispatch to their own ``execute``
    method, which is responsible for advancing ``ctx.current_step``. Leaf
    phases use ``encode_inputs``/``encode_outputs`` and we advance time here.

    Parameters
    ----------
    phase : Phase
        The phase to execute.
    ctx : Context
        The trial context.
    """
    duration = phase.get_duration(ctx)
    ctx.phase_start = ctx.current_step
    ctx.phase_end = ctx.current_step + duration
    ctx.current_phase = phase.name

    phase.on_enter(ctx)

    if phase.IS_COMPOUND:
        phase.execute(ctx)
    else:
        phase.encode_inputs(ctx)
        phase.encode_outputs(ctx)
        ctx.current_step = ctx.phase_end

    phase.on_exit(ctx)

    ctx.phase_history.append((phase.name, ctx.phase_start, ctx.phase_end))


def phase_tree_is_variable(phase: Phase) -> bool:
    """Walk a phase tree and return True if any node declares
    ``is_variable = True``.

    Used by ``Task`` to decide whether to allocate variable-length buffers
    and dispatch through ``execute_phase_packed``. Pure structural walk;
    runs at construction time, no ``ctx`` needed.
    """
    if getattr(phase, 'is_variable', False):
        return True
    for child in phase.children():
        if phase_tree_is_variable(child):
            return True
    return False


def execute_phase_packed(phase: Phase, ctx: Context) -> None:
    """Variable-length / packed-mode dispatch for a single phase.

    Unlike :func:`execute_phase` the phase's actual length is a **traced**
    ``int32`` scalar; the buffer slot reserved for it is sized by the
    Python-int ``max_steps``. Leaf phases are run against a phase-local
    block of shape ``(max_dur, …)``, the block is gated to zero for
    positions beyond the traced actual length, and the block is then
    written into the trial-level buffers at ``ctx.t_cursor`` via
    ``jax.lax.dynamic_update_slice``. Compound phases dispatch to their
    own ``execute_packed`` implementation; the runtime swap/gate/write
    happens at each leaf.
    """
    if ctx.t_cursor is None or ctx.mask is None:
        raise RuntimeError(
            "execute_phase_packed called without t_cursor/mask. Did the "
            "Task forget to enter variable-length mode?"
        )

    max_dur = int(phase.max_steps(ctx))
    actual = jnp.asarray(phase.step_count(ctx), dtype=jnp.int32)
    actual = jnp.clip(actual, 0, max_dur)
    slot_start = ctx.t_cursor

    ctx.phase_max_steps = max_dur
    ctx.phase_step_count = actual
    ctx.current_phase = phase.name
    # Expose the phase's trial-level position before ``on_enter`` so hooks see
    # the same (phase_start, phase_end) contract as the fixed-mode path in
    # ``execute_phase``. Leaf encoding temporarily repoints these at a local
    # block and restores them to the slot afterwards.
    ctx.phase_start = slot_start
    ctx.phase_end = slot_start + actual

    phase.on_enter(ctx)

    if phase.IS_COMPOUND:
        execute = getattr(phase, 'execute_packed', None)
        if execute is None:
            raise NotImplementedError(
                f"Compound phase '{phase.name}' ({type(phase).__name__}) does "
                "not implement execute_packed; variable-length mode for this "
                "compound is not supported yet. See the design spec section "
                "on conditional compounds."
            )
        execute(ctx)
    else:
        _execute_leaf_packed(phase, ctx, max_dur, actual, slot_start)

    phase.on_exit(ctx)

    ctx.phase_history.append((phase.name, slot_start, slot_start + actual))


def _execute_leaf_packed(
    phase: Phase,
    ctx: Context,
    max_dur: int,
    actual: jax.Array,
    slot_start: jax.Array,
) -> None:
    """Encode a leaf phase into the trial buffer at ``slot_start``.

    The strategy is to make the phase's ordinary ``encode_inputs`` /
    ``encode_outputs`` operate against a freshly allocated phase-local
    block (``[0, max_dur)``). We swap the main buffers out, let the phase
    write into the local block, then gate by ``actual`` and write the
    block back into the trial buffers at the traced ``slot_start``.
    """
    real_inputs = ctx.inputs
    real_outputs = ctx.outputs
    real_mask = ctx.mask

    num_inputs = real_inputs.shape[1]

    # Initialise the phase-local block from the existing slot so that
    # Parallel children can compose without erasing one another. For
    # Sequence the slot is freshly zeroed and this is a no-op.
    slot_in = jax.lax.dynamic_slice(real_inputs, (slot_start, jnp.int32(0)),
                                    (max_dur, num_inputs))
    if real_outputs.ndim == 1:
        slot_out = jax.lax.dynamic_slice_in_dim(real_outputs, slot_start, max_dur, axis=0)
    else:
        slot_out = jax.lax.dynamic_slice(real_outputs, (slot_start, jnp.int32(0)),
                                         (max_dur, real_outputs.shape[1]))
    slot_mask = jax.lax.dynamic_slice_in_dim(real_mask, slot_start, max_dur, axis=0)

    ctx.inputs = slot_in
    ctx.outputs = slot_out
    ctx.phase_start = 0
    ctx.phase_end = max_dur

    phase.encode_inputs(ctx)
    phase.encode_outputs(ctx)

    written_in = ctx.inputs
    written_out = ctx.outputs

    t = jnp.arange(max_dur, dtype=jnp.int32)
    gate1d = (t < actual)
    gate2d = gate1d[:, None]
    # Inside the active region use the phase's writes; outside fall back to
    # whatever was already in the slot (the slot is zeros under Sequence, but
    # may carry sibling Parallel contributions).
    merged_in = jnp.where(gate2d, written_in, slot_in)
    if written_out.ndim == 1:
        merged_out = jnp.where(gate1d, written_out, slot_out)
    else:
        merged_out = jnp.where(gate2d, written_out, slot_out)

    ctx.inputs = jax.lax.dynamic_update_slice(
        real_inputs, merged_in.astype(real_inputs.dtype), (slot_start, jnp.int32(0))
    )
    if real_outputs.ndim == 1:
        ctx.outputs = jax.lax.dynamic_update_slice_in_dim(
            real_outputs, merged_out.astype(real_outputs.dtype), slot_start, axis=0
        )
    else:
        ctx.outputs = jax.lax.dynamic_update_slice(
            real_outputs, merged_out.astype(real_outputs.dtype), (slot_start, jnp.int32(0))
        )
    new_mask = slot_mask | gate1d
    ctx.mask = jax.lax.dynamic_update_slice_in_dim(real_mask, new_mask, slot_start, axis=0)

    ctx.t_cursor = slot_start + actual
    ctx.phase_start = slot_start
    ctx.phase_end = slot_start + actual


def concat(phases: List[Phase]) -> Sequence:
    """
    Concatenate phases into a sequence.

    Equivalent to: ``phases[0] >> phases[1] >> ... >> phases[n]``

    Parameters
    ----------
    phases : List[Phase]
        List of phases to concatenate.

    Returns
    -------
    Sequence
        A sequence containing all phases.

    Examples
    --------
    >>> task = concat([
    ...     Fixation(100 * u.ms),
    ...     Stimulus(200 * u.ms),
    ...     Response(100 * u.ms)
    ... ])
    """
    if not phases:
        raise ValueError("Cannot concatenate empty list of phases")
    return Sequence(*phases)


# Type alias for value specifications
ValueSpec = Union[float, int, jnp.ndarray, Callable[[Context, Optional[Feature]], Any]]


class DeclarativePhase(Phase):
    """
    Declarative phase definition with explicit input/output specifications.

    Inputs and outputs are described by dictionaries mapping feature names to
    value specifications. A value spec is either a constant or a callable
    ``f(ctx, feature) -> value``.

    Value shape conventions
    -----------------------
    Inputs (per spec):
        - scalar → broadcast to all (duration, feature.num)
        - 1-D, shape ``(feature.num,)`` → broadcast along time
        - 2-D, shape ``(duration, feature.num)`` → written directly

    Outputs (categorical mode, ``ctx.outputs.ndim == 1``):
        - scalar (int) → constant label over the whole phase
        - 1-D, shape ``(duration,)`` → time-varying labels

    Outputs (vector mode, ``ctx.outputs.ndim == 2``):
        - 1-D, shape ``(feature.num,)`` → broadcast along time
        - 2-D, shape ``(duration, feature.num)`` → written directly

    Parameters
    ----------
    duration : Duration
        Phase duration.
    name : str
        Phase name.
    inputs : dict, optional
        Mapping of feature name → value spec.
    outputs : dict, optional
        Output specification (see shape conventions above).
    noise : dict, optional
        Mapping of feature name → noise sigma (Quantity with unit ms**0.5).
        Noise is scaled as ``sigma / sqrt(dt)``. The implementation strips
        units and uses the bare mantissas of ``sigma`` and ``dt``, so it
        assumes ``dt`` is expressed in milliseconds (the framework default).
        If you change ``brainstate.environ`` to a non-ms ``dt`` (e.g.
        seconds), the noise magnitude will be scaled incorrectly — pass
        ``sigma`` already matched to that unit, or keep ``dt`` in ms.
    on_enter : callable, optional
        Hook called when phase begins.
    on_exit : callable, optional
        Hook called when phase ends.
    """

    def __init__(
        self,
        duration: Duration,
        inputs: Optional[Dict[str, ValueSpec]] = None,
        outputs: Optional[Dict[str, ValueSpec]] = None,
        noise: Optional[Dict[str, Data]] = None,
        on_enter: Optional[Callable[[Context], None]] = None,
        on_exit: Optional[Callable[[Context], None]] = None,
        name: str = None,
    ):
        name = name if name is not None else self.__class__.__name__
        if name == 'DeclarativePhase':
            raise ValueError(
                "DeclarativePhase must be subclassed (e.g. use Fixation, "
                "Stimulus, …) or constructed with an explicit name= argument."
            )
        super().__init__(duration, name)
        self._input_specs = inputs or {}
        self._output_specs = outputs or {}
        self._noise_specs = noise or {}
        self._on_enter_hook = on_enter
        self._on_exit_hook = on_exit

        # Will be set by Task during binding
        self._task_input_features: Optional[FeatureSet] = None
        self._task_output_features: Optional[FeatureSet] = None

    def bind_features(self, input_features: FeatureSet, output_features: FeatureSet,
                      num_classes: Optional[int] = None):
        self._task_input_features = input_features
        self._task_output_features = output_features

        # Validate input feature names
        for name in self._input_specs:
            if name not in input_features:
                raise ValueError(
                    f"Unknown input feature '{name}' in phase '{self.name}'. "
                    f"Available features: {[f.name for f in input_features]}"
                )

        # Validate noise feature names
        for name in self._noise_specs:
            if name not in input_features:
                raise ValueError(
                    f"Unknown noise feature '{name}' in phase '{self.name}'. "
                    f"Available features: {[f.name for f in input_features]}"
                )

        # Validate output feature names. 'label' is reserved for categorical
        # mode and is allowed unconditionally.
        for name in self._output_specs:
            if name == 'label':
                continue
            if name not in output_features:
                raise ValueError(
                    f"Unknown output feature '{name}' in phase '{self.name}'. "
                    f"Available output features: {[f.name for f in output_features]}"
                )

        # Catch a *statically* out-of-range categorical label early (e.g. a
        # typo like ``outputs={'label': 99}``). The only sound upper bound is
        # ``num_classes`` — the output-feature dimensionality is unrelated to
        # the label space (that is exactly what ``Task.num_classes`` exists to
        # decouple). So this only fires when the task declared ``num_classes``
        # explicitly. Callable / array label specs are data-dependent and
        # skipped here.
        label_spec = self._output_specs.get('label')
        if (num_classes is not None
                and isinstance(label_spec, int) and not isinstance(label_spec, bool)):
            if not (0 <= label_spec < num_classes):
                raise ValueError(
                    f"Phase '{self.name}': categorical label {label_spec} is out "
                    f"of range; expected 0 <= label < num_classes "
                    f"({num_classes})."
                )

    def on_enter(self, ctx: Context) -> None:
        """Hook called when phase begins."""
        if self._on_enter_hook:
            self._on_enter_hook(ctx)

    def on_exit(self, ctx: Context) -> None:
        """Hook called when phase ends."""
        if self._on_exit_hook:
            self._on_exit_hook(ctx)

    def _resolve_value(
        self,
        spec: ValueSpec,
        ctx: Context,
        feature: Optional[Feature]
    ) -> Any:
        if callable(spec):
            return spec(ctx, feature)
        else:
            return spec

    @staticmethod
    def _sigma_magnitude(sigma: Any) -> float:
        if hasattr(sigma, 'mantissa'):
            return float(sigma.mantissa)
        return float(sigma)

    def encode_inputs(self, ctx: Context) -> None:
        """
        Fill input buffer based on declarative specifications.

        Supports scalar, ``(feature.num,)``, and ``(duration, feature.num)``
        value shapes. Features not specified remain at their default value.
        """
        if self._task_input_features is None:
            raise RuntimeError(
                f"Phase '{self.name}' has not been bound to task features. "
                "Call bind_features() first or ensure Task binds features automatically."
            )

        start, end = ctx.phase_start, ctx.phase_end
        duration = end - start

        for feat_name, value_spec in self._input_specs.items():
            feature = self._task_input_features[feat_name]
            value = self._resolve_value(value_spec, ctx, feature)
            value = jnp.asarray(value)
            ctx.inputs = ctx.inputs.at[start:end, feature.i].set(value)

            if feat_name in self._noise_specs:
                sigma_val = self._sigma_magnitude(self._noise_specs[feat_name])
                if sigma_val > 0:
                    dt_val = self._sigma_magnitude(ctx.dt)
                    noise = ctx.rng.randn(duration, feature.num)
                    noise = noise * (sigma_val / jnp.sqrt(dt_val))
                    ctx.inputs = ctx.inputs.at[start:end, feature.i].add(noise)

    def encode_outputs(self, ctx: Context) -> None:
        """
        Fill output buffer based on declarative specifications.

        - Categorical mode (``ctx.outputs.ndim == 1``): use the ``label`` key.
          Accepts a scalar (constant label) or 1-D array of shape
          ``(duration,)`` (time-varying labels).
        - Vector mode (``ctx.outputs.ndim == 2``): write each specified output
          feature into its slice. Accepts shape ``(feature.num,)`` (broadcast
          along time) or ``(duration, feature.num)``.
        """
        start, end = ctx.phase_start, ctx.phase_end
        duration = end - start

        # ----- categorical mode -----
        if ctx.outputs.ndim == 1:
            label_spec = self._output_specs.get('label', 0)
            label_value = self._resolve_value(label_spec, ctx, None)
            arr = jnp.asarray(label_value, dtype=jnp.int32)
            if arr.ndim == 0:
                ctx.outputs = ctx.outputs.at[start:end].set(arr)
            elif arr.ndim == 1:
                # time-varying labels
                ctx.outputs = ctx.outputs.at[start:end].set(arr)
            else:
                raise ValueError(
                    f"Phase '{self.name}': categorical label must be a scalar "
                    f"or 1-D array of shape (duration,), got shape {arr.shape}"
                )
            return

        # ----- vector mode -----
        if self._task_output_features is None:
            raise RuntimeError(
                f"Phase '{self.name}' has not been bound to task output features. "
                "Call bind_features() first or ensure Task binds features automatically."
            )

        for feat_name, value_spec in self._output_specs.items():
            if feat_name == 'label':
                continue  # ignore label in vector mode
            feature = self._task_output_features[feat_name]
            value = self._resolve_value(value_spec, ctx, feature)
            value = jnp.asarray(value, dtype=jnp.float32)
            ctx.outputs = ctx.outputs.at[start:end, feature.i].set(value)

    def describe(self) -> str:
        """Return a human-readable description of this phase's behavior."""
        lines = [f"Phase '{self.name}' (duration={self._duration}):"]
        lines.append("  Inputs:")
        for name, spec in self._input_specs.items():
            if callable(spec):
                spec_str = f"<callable: {getattr(spec, '__name__', 'anonymous')}>"
            else:
                spec_str = str(spec)
            lines.append(f"    {name}: {spec_str}")
        if not self._input_specs:
            lines.append("    (none - all features default to 0)")

        lines.append("  Outputs:")
        for name, spec in self._output_specs.items():
            if callable(spec):
                spec_str = f"<callable: {getattr(spec, '__name__', 'anonymous')}>"
            else:
                spec_str = str(spec)
            lines.append(f"    {name}: {spec_str}")

        if self._noise_specs:
            lines.append("  Noise:")
            for name, sigma in self._noise_specs.items():
                lines.append(f"    {name}: sigma={sigma}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        input_names = list(self._input_specs.keys())
        return f"DeclarativePhase('{self.name}', inputs={input_names})"


class Fixation(DeclarativePhase):
    pass


class Sample(DeclarativePhase):
    pass


class Delay(DeclarativePhase):
    pass


class Test(DeclarativePhase):
    pass


class Response(DeclarativePhase):
    pass


class Stimulus(DeclarativePhase):
    pass


class Cue(DeclarativePhase):
    pass


class Recall(DeclarativePhase):
    pass


class Match(DeclarativePhase):
    pass


class Comparison(DeclarativePhase):
    pass


class Blank(DeclarativePhase):
    pass


class VariableDuration(DeclarativePhase):
    """Declarative phase whose actual length is decided per trial.

    The phase reserves a buffer slot of ``ceil(max_duration / dt)``
    timesteps and writes its content into the first
    ``ceil(ctx[ctx_key] / dt)`` of them. Anything past the trial's actual
    length is automatically zeroed by the packed runtime and masked out in
    the trial mask. Compatible with ``brainstate.transform.jit`` and
    ``brainstate.transform.vmap2``.

    ``ctx_key`` names the trial state entry holding the sampled duration.
    The value can be a scalar number of milliseconds or a Quantity; in
    either case the runtime divides by ``ctx.dt`` to obtain a timestep
    count. ``min_duration`` and ``max_duration`` must be Quantities with
    matching units; ``min_duration`` is the static lower bound used to
    floor the step count (always >= 1 timestep).

    Examples
    --------
    >>> import brainunit as u
    >>> from braintools.cogtask import VariableDuration
    >>> phase = VariableDuration(
    ...     min_duration=300 * u.ms,
    ...     max_duration=1500 * u.ms,
    ...     ctx_key='delay_duration',
    ...     inputs={'fixation': 1.0},
    ...     outputs={'label': 0},
    ...     name='variable_delay',
    ... )
    """

    is_variable = True

    def __init__(
        self,
        min_duration,
        max_duration,
        ctx_key: str,
        inputs: Optional[Dict[str, 'ValueSpec']] = None,
        outputs: Optional[Dict[str, 'ValueSpec']] = None,
        noise: Optional[Dict[str, Any]] = None,
        on_enter: Optional[Callable[[Context], None]] = None,
        on_exit: Optional[Callable[[Context], None]] = None,
        name: Optional[str] = None,
    ):
        if name is None:
            name = 'VariableDuration'
        # Use max_duration as the nominal "duration" so get_duration falls
        # back to the upper bound when called outside variable-length mode.
        super().__init__(
            duration=max_duration,
            inputs=inputs,
            outputs=outputs,
            noise=noise,
            on_enter=on_enter,
            on_exit=on_exit,
            name=name,
        )
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._ctx_key = ctx_key

    def _dt_mantissa(self, ctx: Context) -> float:
        dt = ctx.dt
        return float(dt.mantissa) if hasattr(dt, 'mantissa') else float(dt)

    def _duration_unit_mantissa(self, value: Any) -> Any:
        """Return the duration in the same unit-base as dt.

        Trial-init code often stores ``ctx[ctx_key]`` as a Python float
        already expressed in the same unit as dt (the existing tasks do
        this). When a Quantity is provided we strip the unit; otherwise we
        pass it through as a JAX scalar.
        """
        if hasattr(value, 'to') and hasattr(value, 'mantissa'):
            # brainunit Quantity — convert to the dt unit before stripping.
            return value.mantissa
        return value

    def max_steps(self, ctx: Context) -> int:
        return max(1, int(self._max_duration / ctx.dt))

    def step_count(self, ctx: Context) -> jax.Array:
        if self._ctx_key not in ctx:
            # Fall back to the max bound when the trial init didn't populate
            # the key; this keeps shape contracts well-defined.
            return jnp.asarray(self.max_steps(ctx), dtype=jnp.int32)
        raw = self._duration_unit_mantissa(ctx[self._ctx_key])
        dt_val = self._dt_mantissa(ctx)
        steps = jnp.asarray(raw, dtype=jnp.float32) / jnp.asarray(dt_val, dtype=jnp.float32)
        steps = jnp.maximum(jnp.int32(1), steps.astype(jnp.int32))
        max_steps_static = self.max_steps(ctx)
        return jnp.minimum(steps, jnp.int32(max_steps_static))

    def get_duration(self, ctx: Context) -> int:
        """Eager-mode duration. Reads ``ctx[self._ctx_key]`` if available
        (and converts to a Python int via ``int(jnp.asarray(...))``); falls
        back to ``max_steps`` otherwise. Not used on the packed JIT path.
        """
        if self._ctx_key not in ctx:
            return self.max_steps(ctx)
        try:
            raw = self._duration_unit_mantissa(ctx[self._ctx_key])
            dt_val = self._dt_mantissa(ctx)
            v = int(jnp.asarray(raw) / jnp.asarray(dt_val))
            return max(1, min(v, self.max_steps(ctx)))
        except (TypeError, jax.errors.TracerArrayConversionError):
            return self.max_steps(ctx)
