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

    def encode_inputs(self, ctx: Context) -> None:
        # Delegated through execute
        pass

    def encode_outputs(self, ctx: Context) -> None:
        # Delegated through execute
        pass

    def execute(self, ctx: Context) -> None:
        """Encode each child within its own [phase_start, phase_start+dur)."""
        parent_start = ctx.phase_start
        parent_end = ctx.phase_end
        parent_name = ctx.current_phase

        for i, child in enumerate(self.phases):
            child_dur = child.get_duration(ctx)
            ctx.phase_start = parent_start
            ctx.phase_end = parent_start + child_dur
            ctx.current_phase = child.name
            child.on_enter(ctx)
            child.encode_inputs(ctx)
            # Only the first child contributes outputs by default.
            if i == 0:
                child.encode_outputs(ctx)
            child.on_exit(ctx)
            ctx.phase_history.append((child.name, ctx.phase_start, ctx.phase_end))

        # Restore parent scope.
        ctx.phase_start = parent_start
        ctx.phase_end = parent_end
        ctx.current_phase = parent_name
        ctx.current_step = parent_end

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

    def bind_features(self, input_features: FeatureSet, output_features: FeatureSet):
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
