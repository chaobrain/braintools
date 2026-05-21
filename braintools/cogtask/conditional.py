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

from .context import Context
from .phase import Phase, execute_phase

__all__ = [
    'If', 'Switch', 'While'
]


class _CompoundCond(Phase):
    """Marker base — these phases manage their own children via ``execute``."""
    IS_COMPOUND = True


class If(_CompoundCond):
    """
    Conditional phase selection based on a boolean condition.

    Evaluates the condition at runtime and executes either the `then` phase
    or the `else_` phase (if provided).

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

    def get_duration(self, ctx: Context) -> int:
        """Duration depends on which branch is taken."""
        if self.condition(ctx):
            return self.then_phase.get_duration(ctx)
        elif self.else_phase:
            return self.else_phase.get_duration(ctx)
        return 0

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

    def get_duration(self, ctx: Context) -> int:
        """Duration depends on which case is selected."""
        key = self.selector(ctx)
        if key in self.cases:
            return self.cases[key].get_duration(ctx)
        elif self.default:
            return self.default.get_duration(ctx)
        return 0

    def encode_inputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def encode_outputs(self, ctx: Context) -> None:
        pass  # Delegated during execute

    def execute(self, ctx: Context) -> None:
        """Execute the phase corresponding to the selector's key."""
        key = self.selector(ctx)
        ctx['switch_key'] = key  # Store for debugging/analysis
        if key in self.cases:
            execute_phase(self.cases[key], ctx)
        elif self.default:
            execute_phase(self.default, ctx)

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

    def get_duration(self, ctx: Context) -> int:
        """
        Estimate duration using max_iterations.

        Note: Actual duration may be less if condition becomes False.
        """
        return self.body.get_duration(ctx) * self.max_iterations

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

    def children(self):
        return [self.body]

    def __repr__(self) -> str:
        return f"While(body={self.body.name}, max={self.max_iterations})"
