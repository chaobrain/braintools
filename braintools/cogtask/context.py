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

"""Context class for inter-phase communication during trial generation."""

from typing import Any, Dict, Optional, List, Tuple, Union

import brainstate
import brainunit as u
import jax

__all__ = ['Context']


class Context:
    """
    Mutable trial-level state container shared across phases.

    The Context object is the backbone of inter-phase communication. It holds:
    - Trial-level state that persists across phases (e.g., ground_truth, stimulus_value)
    - Time tracking (current timestep, phase boundaries)
    - Input/output buffers for trial data
    - RNG for reproducible randomness

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> ctx = Context(key=jax.random.PRNGKey(42))
        >>> ctx['ground_truth'] = 1
        >>> ctx['stimulus_direction'] = 0.5 * jnp.pi
        >>> print(ctx['ground_truth'])
        1

        >>> # In a phase, write into the buffer with a functional update
        >>> # (JAX arrays are immutable; ``.at[...].set(...)`` returns a copy):
        >>> # ctx.inputs = ctx.inputs.at[ctx.phase_start:ctx.phase_end, :].set(stimulus_encoding)

    Parameters
    ----------
    key : jax.Array, optional
        JAX PRNGKey for random number generation. If None, draws fresh
        randomness from brainstate's default RNG.

    Note
    ----
    The time step (dt) is obtained from brainstate.environ.get_dt().
    """

    def __init__(
        self,
        key: Optional[jax.Array] = None,
    ):
        self.rng = brainstate.random.default_rng(key)

        # Trial state dictionary - phases read/write here
        self._state: Dict[str, Any] = {}

        # Time tracking
        self.current_step: int = 0
        self.phase_start: int = 0
        self.phase_end: int = 0

        # Phase metadata
        self.current_phase: Optional[str] = None
        self.phase_history: List[Tuple[str, int, int]] = []  # (name, start, end)

        # I/O buffers (set by Task before trial generation)
        self.inputs: Optional[jax.Array] = None
        self.outputs: Optional[jax.Array] = None

        # Variable-length / packed-mode bookkeeping (None in fixed mode).
        # ``mask`` is a ``(T_max,)`` bool array; True for "real" timesteps.
        # ``t_cursor`` is a traced int32 scalar counting valid timesteps
        # written so far. ``phase_max_steps`` / ``phase_step_count`` expose
        # the current phase's static upper bound and traced actual length
        # to encode callbacks that need to size per-phase blocks.
        self.mask: Optional[jax.Array] = None
        self.t_cursor: Optional[jax.Array] = None
        self.phase_max_steps: Optional[int] = None
        self.phase_step_count: Optional[jax.Array] = None

    def __getitem__(self, key: str) -> Any:
        """Get a value from the trial state."""
        return self._state[key]

    def __setitem__(self, key: str, value: Any):
        """Set a value in the trial state."""
        self._state[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the trial state."""
        return key in self._state

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the trial state with a default."""
        return self._state.get(key, default)

    def update(self, **kwargs):
        """Update multiple values in the trial state."""
        self._state.update(kwargs)

    def clear(self):
        """Reset state for a new trial."""
        self._state.clear()
        self.current_step = 0
        self.phase_start = 0
        self.phase_end = 0
        self.current_phase = None
        self.phase_history.clear()
        self.mask = None
        self.t_cursor = None
        self.phase_max_steps = None
        self.phase_step_count = None

    @property
    def dt(self) -> Union[float, u.Quantity]:
        """Time step in milliseconds (from brainstate.environ.get_dt())."""
        return brainstate.environ.get_dt()

    @property
    def phase_duration(self) -> int:
        """Duration of current phase in timesteps."""
        return self.phase_end - self.phase_start

    @property
    def phase_time(self) -> int:
        """Current timestep relative to phase start."""
        return self.current_step - self.phase_start

    @property
    def total_steps(self) -> int:
        """Total number of timesteps in the trial (based on input buffer)."""
        if self.inputs is not None:
            return self.inputs.shape[0]
        return 0

    @property
    def state(self) -> Dict[str, Any]:
        """Get a copy of the trial state dictionary."""
        return dict(self._state)

    def phase_slice(self) -> slice:
        """Get a slice object for the current phase's time range."""
        return slice(self.phase_start, self.phase_end)

    def __repr__(self) -> str:
        return (
            f"Context(dt={self.dt}, "
            f"step={self.current_step}, "
            f"phase={self.current_phase})"
        )
