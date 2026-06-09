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

"""Conditional phase classes for dynamic task flow."""

from typing import Callable, Dict, Any, Optional

import brainunit as u
import jax
import jax.numpy as jnp

from .context import Context
from .phase import Phase, execute_phase, execute_phase_packed

__all__ = [
    'If', 'Switch', 'While'
]


def _coerce_concrete_key(key: Any) -> Any:
    """Normalize a Switch selector result to a hashable key.

    ``ctx.rng.choice(...)`` (the natural way to pick a random rule in
    ``trial_init``) returns a 0-d JAX array. In eager ``sample_trial`` that
    value is *concrete but not a* :class:`jax.core.Tracer`, and it is
    unhashable — so ``key in cases`` raises ``TypeError``. Coerce such
    concrete 0-d arrays to a Python scalar so dict lookup works, while
    leaving tracers (the ``jit``/``vmap`` path) and already-hashable keys
    (``int``, ``str``) untouched.
    """
    if isinstance(key, jax.core.Tracer):
        return key
    if hasattr(key, 'ndim') and getattr(key, 'ndim', None) == 0:
        return key.item()
    return key


class _CompoundCond(Phase):
    """Marker base — these phases manage their own children via ``execute``."""
    IS_COMPOUND = True


class If(_CompoundCond):
    """
    Conditional phase selection based on a boolean condition.

    Evaluates the condition at runtime and executes either the `then` phase
    or the `else_` phase (if provided).

    Note
    ----
    Under ``batch_sample`` (jit/vmap) the branch is selected with
    ``jax.lax.cond``, so both branches are traced. The output buffers
    (``X``/``Y``/``mask``) are correct, but Python-side ``ctx`` state written
    inside a branch (and ``phase_history``) reflects whichever branch traced
    last, not the one actually taken. Do not read branch-written ``ctx[...]``
    values in ``get_trial_meta`` or downstream phases when batching; gate such
    state on the same condition at the top level instead.

    Examples
    --------
    >>> # Match vs non-match response
    >>> phases = (
    ...     Sample(500 * u.ms)
    ...     >> Delay(1000 * u.ms)
    ...     >> Test(500 * u.ms)
    ...     >> If(
    ...         lambda ctx: ctx['match'],
    ...         then=MatchResponse(500 * u.ms),
    ...         else_=NonMatchResponse(500 * u.ms)
    ...     )
    ... )

    >>> # Go/NoGo with no else branch
    >>> phases = (
    ...     Stimulus(500 * u.ms)
    ...     >> If(
    ...         lambda ctx: ctx['is_go'],
    ...         then=Response(500 * u.ms)
    ...     )
    ... )

    Parameters
    ----------
    condition : Callable[[Context], bool]
        Function that takes context and returns True/False.
    then : Phase
        Phase to execute if condition is True.
    else_ : Phase, optional
        Phase to execute if condition is False.
    name : str
        Name for this conditional phase.
    """

    def __init__(
        self,
        condition: Callable[[Context], bool],
        then: Phase,
        else_: Optional[Phase] = None,
        name: str = 'If'
    ):
        self.condition = condition
        self.then_phase = then
        self.else_phase = else_
        super().__init__(duration=0 * u.ms, name=name)

    # Branch selection is decided at trial time, so the actual length is
    # data-dependent. Compound phases that contain an ``If`` propagate
    # ``is_variable = True`` via ``children()``.
    is_variable = True

    def get_duration(self, ctx: Context) -> int:
        """Duration depends on which branch is taken."""
        if self.condition(ctx):
            return self.then_phase.get_duration(ctx)
        elif self.else_phase:
            return self.else_phase.get_duration(ctx)
        return 0

    def max_steps(self, ctx: Context) -> int:
        a = self.then_phase.max_steps(ctx)
        b = self.else_phase.max_steps(ctx) if self.else_phase else 0
        return max(a, b)

    def step_count(self, ctx: Context) -> jax.Array:
        pred = jnp.asarray(self.condition(ctx), dtype=jnp.bool_)
        a = self.then_phase.step_count(ctx)
        b = (self.else_phase.step_count(ctx)
             if self.else_phase
             else jnp.asarray(0, dtype=jnp.int32))
        return jnp.where(pred, a, b)

    def encode_inputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def encode_outputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def execute(self, ctx: Context) -> None:
        """Execute the appropriate branch based on condition."""
        if self.condition(ctx):
            execute_phase(self.then_phase, ctx)
        elif self.else_phase:
            execute_phase(self.else_phase, ctx)

    def execute_packed(self, ctx: Context) -> None:
        """Packed-mode branch using ``jax.lax.cond``.

        Both branches mutate ``ctx.inputs/outputs/mask/t_cursor``. To make
        the cond functional we (a) snapshot ``ctx._state`` so per-branch
        scratch writes don't leak across branches, and (b) thread the
        buffer state through ``lax.cond`` as a pytree. Anything not in
        that pytree (e.g. ``ctx.phase_history``) is best-effort metadata
        and may contain entries from both branches during tracing — it is
        not part of the trial's tensor output.
        """
        pred = jnp.asarray(self.condition(ctx), dtype=jnp.bool_)
        state_snapshot = dict(ctx._state)

        def _run(branch: Phase, state):
            ctx._state = dict(state_snapshot)
            ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor = state
            execute_phase_packed(branch, ctx)
            return (ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor)

        def _then(state):
            return _run(self.then_phase, state)

        def _else(state):
            if self.else_phase is None:
                return state
            return _run(self.else_phase, state)

        state = (ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor)
        new_state = jax.lax.cond(pred, _then, _else, state)
        ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor = new_state

    def children(self):
        out = [self.then_phase]
        if self.else_phase is not None:
            out.append(self.else_phase)
        return out

    def __repr__(self) -> str:
        else_str = f", else={self.else_phase.name}" if self.else_phase else ""
        return f"If(then={self.then_phase.name}{else_str})"


class Switch(_CompoundCond):
    """
    Multi-way conditional phase selection.

    Evaluates the selector function to get a key, then executes the
    corresponding phase from the cases dictionary.

    Examples
    --------
    >>> # Rule-dependent response
    >>> phases = (
    ...     Stimulus(500 * u.ms)
    ...     >> Delay(1000 * u.ms)
    ...     >> Switch(
    ...         lambda ctx: ctx['rule'],
    ...         cases={
    ...             'pro': ProResponse(500 * u.ms),
    ...             'anti': AntiResponse(500 * u.ms),
    ...         },
    ...         default=DefaultResponse(500 * u.ms)
    ...     )
    ... )

    >>> # Multiple choice selection
    >>> phases = Switch(
    ...     lambda ctx: ctx['choice'],
    ...     cases={
    ...         0: Choice0Response(100 * u.ms),
    ...         1: Choice1Response(100 * u.ms),
    ...         2: Choice2Response(100 * u.ms),
    ...     }
    ... )

    Parameters
    ----------
    selector : Callable[[Context], Any]
        Function that takes context and returns a key.
    cases : Dict[Any, Phase]
        Mapping from keys to phases.
    default : Phase, optional
        Phase to execute if key not found in cases.
    name : str
        Name for this conditional phase.
    """

    def __init__(
        self,
        selector: Callable[[Context], Any],
        cases: Dict[Any, Phase],
        default: Optional[Phase] = None,
        name: str = 'Switch'
    ):
        self.selector = selector
        self.cases = cases
        self.default = default
        super().__init__(duration=0 * u.ms, name=name)

    is_variable = True

    def get_duration(self, ctx: Context) -> int:
        """Duration depends on which case is selected."""
        key = _coerce_concrete_key(self.selector(ctx))
        if key in self.cases:
            return self.cases[key].get_duration(ctx)
        elif self.default:
            return self.default.get_duration(ctx)
        return 0

    def max_steps(self, ctx: Context) -> int:
        candidates = [p.max_steps(ctx) for p in self.cases.values()]
        if self.default is not None:
            candidates.append(self.default.max_steps(ctx))
        return max(candidates) if candidates else 0

    def step_count(self, ctx: Context) -> jax.Array:
        # Best-effort traced length: when the selector returns a hashable
        # python key, evaluate that case's step_count directly. When it
        # returns a tracer, callers should use the packed compound dispatch
        # which builds a lax.switch over ordered branches; until that path
        # lands we fall back to the default branch (or zero) so the value is
        # still well-typed.
        key = _coerce_concrete_key(self.selector(ctx))
        try:
            phase = self.cases.get(key)
        except TypeError:
            phase = None
        if phase is None:
            phase = self.default
        if phase is None:
            return jnp.asarray(0, dtype=jnp.int32)
        return phase.step_count(ctx)

    def encode_inputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def encode_outputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def execute(self, ctx: Context) -> None:
        """Execute the phase corresponding to the selector's key."""
        key = _coerce_concrete_key(self.selector(ctx))
        ctx['switch_key'] = key  # Store for debugging/analysis
        if key in self.cases:
            execute_phase(self.cases[key], ctx)
        elif self.default:
            execute_phase(self.default, ctx)

    def execute_packed(self, ctx: Context) -> None:
        """Packed-mode dispatch.

        When the selector returns a concrete Python value (eager
        ``sample_trial``, or a selector that ignores trial state) we dispatch
        with an ordinary ``key in cases`` lookup, which supports arbitrary
        hashable keys (e.g. strings).

        When the selector returns a *tracer* (under ``jit``/``vmap``, i.e.
        ``batch_sample``) we dispatch with :func:`jax.lax.switch` over an
        ordered list of branches. In this path the case keys must be integers
        — a traced value cannot be a string. Keys not present in ``cases`` (or
        any value when there is no matching integer key) route to ``default``,
        or to a no-op when there is no default.

        Note
        ----
        Like :meth:`If.execute_packed`, the tensor buffers are threaded through
        the dispatch and are correct, but Python-side ``ctx`` state written
        inside a branch is not reliable under tracing (every branch is traced).
        Do not rely on branch-written ``ctx[...]`` values in
        ``get_trial_meta``/downstream phases when using ``batch_sample``.
        """
        key = _coerce_concrete_key(self.selector(ctx))
        ctx['switch_key'] = key

        if not isinstance(key, jax.core.Tracer):
            # Concrete dispatch: supports arbitrary hashable keys.
            if key in self.cases:
                execute_phase_packed(self.cases[key], ctx)
            elif self.default is not None:
                execute_phase_packed(self.default, ctx)
            # Otherwise: no-op. The parent reserved ``max_steps`` worth of
            # buffer space; we simply don't advance ``t_cursor``.
            return

        # Traced dispatch via lax.switch over ordered branches.
        keys = list(self.cases.keys())
        phases = list(self.cases.values())
        state_snapshot = dict(ctx._state)

        def make_branch(phase: Phase):
            def run(state):
                ctx._state = dict(state_snapshot)
                ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor = state
                execute_phase_packed(phase, ctx)
                return (ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor)
            return run

        branches = [make_branch(p) for p in phases]
        # Fallback branch (last position): default phase, or an identity no-op.
        if self.default is not None:
            branches.append(make_branch(self.default))
        else:
            branches.append(lambda state: state)
        fallback = len(branches) - 1

        # Map the (possibly non-contiguous) integer key to a branch position.
        index = jnp.asarray(fallback, dtype=jnp.int32)
        for i, k in enumerate(keys):
            if isinstance(k, int):  # non-int keys can never match a traced int
                index = jnp.where(jnp.asarray(key) == k, jnp.int32(i), index)

        state = (ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor)
        new_state = jax.lax.switch(index, branches, state)
        ctx.inputs, ctx.outputs, ctx.mask, ctx.t_cursor = new_state

    def children(self):
        out = list(self.cases.values())
        if self.default is not None:
            out.append(self.default)
        return out

    def __repr__(self) -> str:
        case_names = list(self.cases.keys())
        default_str = f", default={self.default.name}" if self.default else ""
        return f"Switch(cases={case_names}{default_str})"


class While(_CompoundCond):
    """
    Loop phase while condition is true.

    Useful for tasks with variable numbers of repetitions, such as
    evidence accumulation until a threshold is reached.

    Note: The duration computed by get_duration uses max_iterations
    as an upper bound, but actual execution may be shorter.

    Examples
    --------
    >>> # Evidence accumulation until threshold
    >>> phases = (
    ...     Fixation(500 * u.ms)
    ...     >> While(
    ...         lambda ctx: ctx.get('accumulated_evidence', 0) < threshold,
    ...         body=EvidenceSample(50 * u.ms),
    ...         max_iterations=50
    ...     )
    ...     >> Response(500 * u.ms)
    ... )

    >>> # Repeated sampling with early termination
    >>> phases = While(
    ...     lambda ctx: ctx.get('sample_count', 0) < ctx['required_samples'],
    ...     body=Sample(100 * u.ms),
    ...     max_iterations=20
    ... )

    Parameters
    ----------
    condition : Callable[[Context], bool]
        Function that returns True to continue looping.
    body : Phase
        Phase to execute each iteration.
    max_iterations : int
        Maximum number of iterations (safety limit).
    name : str
        Name for this loop phase.
    """

    def __init__(
        self,
        condition: Callable[[Context], bool],
        body: Phase,
        max_iterations: int = 100,
        name: str = 'While'
    ):
        self.condition = condition
        self.body = body
        self.max_iterations = max_iterations
        super().__init__(duration=0 * u.ms, name=name)

    is_variable = True

    def get_duration(self, ctx: Context) -> int:
        """
        Estimate duration using max_iterations.

        Note: Actual duration may be less if condition becomes False.
        """
        return self.body.get_duration(ctx) * self.max_iterations

    def max_steps(self, ctx: Context) -> int:
        return self.body.max_steps(ctx) * self.max_iterations

    def step_count(self, ctx: Context) -> jax.Array:
        # The actual iteration count depends on a runtime condition; the
        # packed compound dispatch (lax.fori_loop) handles this. For the
        # static/eager path we mirror execute() and return the bound.
        return self.body.step_count(ctx) * jnp.asarray(self.max_iterations, dtype=jnp.int32)

    def encode_inputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def encode_outputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def execute(self, ctx: Context) -> None:
        """Execute body phase while condition is true."""
        iteration = 0
        while self.condition(ctx) and iteration < self.max_iterations:
            ctx['while_iteration'] = iteration
            execute_phase(self.body, ctx)
            iteration += 1
        ctx['while_total_iterations'] = iteration

    def execute_packed(self, ctx: Context) -> None:
        """Packed-mode loop (eager-only).

        The loop is unrolled in Python, so the condition must evaluate to a
        concrete ``bool``. This works for eager ``sample_trial`` but NOT under
        ``jit``/``vmap`` (``batch_sample``) when the condition depends on a
        traced value: we detect that case and raise a clear
        :class:`NotImplementedError` instead of a cryptic
        ``TracerBoolConversionError``. Full traced support would require a
        ``jax.lax.while_loop`` with the buffers + trial state threaded as a
        pytree, which is not yet implemented.
        """
        iteration = 0
        while iteration < self.max_iterations:
            pred = self.condition(ctx)
            if isinstance(pred, jax.core.Tracer):
                raise NotImplementedError(
                    "While with a data-dependent (traced) condition is not "
                    "supported under jit/vmap (e.g. batch_sample). Evaluate it "
                    "eagerly via sample_trial, or replace the While with a "
                    "bounded Repeat / VariableDuration. (Full lax.while_loop "
                    "support is not yet implemented.)"
                )
            if not pred:
                break
            ctx['while_iteration'] = iteration
            execute_phase_packed(self.body, ctx)
            iteration += 1
        ctx['while_total_iterations'] = iteration

    def children(self):
        return [self.body]

    def __repr__(self) -> str:
        return f"While(body={self.body.name}, max={self.max_iterations})"
