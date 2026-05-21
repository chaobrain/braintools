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

"""Utility functions and classes for cognitive tasks."""

from typing import Union, Callable, Optional, Any, Dict, Sequence, Tuple, List

import brainstate.random
import brainunit as u
import jax
import jax.numpy as jnp

__all__ = [
    'TruncExp',
    'UniformDuration',
    'Transform',
    'TransformIT',
    'initialize',
    'initialize2',
    'interval_of',
    'period_to_arr',
    'firing_rate',
]


def choice(rng, n_total, i_exclude):
    other = jnp.arange(n_total - 1)
    other = jnp.where(other >= i_exclude, other + 1, other)
    return rng.choice(other)


class TruncExp:
    """
    Truncated exponential distribution for sampling time durations.

    Useful for generating variable-duration phases with exponential
    distribution truncated to a specified range.

    Examples
    --------
    >>> # Duration sampled from truncated exponential
    >>> duration = TruncExp(600*u.ms, 300*u.ms, 1500*u.ms)
    >>> Delay(duration)  # Variable delay

    >>> # Use in trial_init
    >>> def trial_init(ctx):
    ...     t_delay = TruncExp(600*u.ms, 300*u.ms, 1500*u.ms)
    ...     ctx['delay_duration'] = t_delay()
    """

    def __init__(
        self,
        mean,
        min_val=0,
        max_val=jnp.inf,
        key: Optional[jax.Array] = None,
    ):
        """
        Initialize a truncated exponential distribution.

        Parameters
        ----------
        mean : Quantity or float
            Mean of the exponential distribution (before truncation).
        min_val : Quantity or float
            Minimum value (inclusive).
        max_val : Quantity or float
            Maximum value (exclusive).
        key : jax.Array, optional
            JAX PRNGKey for random number generation.
        seed : int, optional
            Random seed (used if key is None).
        """
        mean = u.Quantity(mean)
        self._time_unit = mean.unit

        self._mean = mean.mantissa
        self._min = u.Quantity(min_val).to(self._time_unit).mantissa
        self._max = u.Quantity(max_val).to(self._time_unit).mantissa

        self.rng = brainstate.random.default_rng(key)

    def __call__(self, ctx=None):
        """
        Sample a value from the truncated exponential distribution.

        Parameters
        ----------
        ctx : Context, optional
            If provided, use context's RNG key. Otherwise use internal key.

        Returns
        -------
        Quantity
            Sampled duration value.
        """
        # Get the random key
        rng = ctx.rng if ctx is not None else self.rng

        # Handle edge case
        if self._min >= self._max:
            return u.maybe_decimal(self._max * self._time_unit)

        # Inverse CDF method (JIT/vmap compatible — keeps v as a JAX scalar).
        F_min = 1 - jnp.exp(-self._min / self._mean)
        F_max = 1 - jnp.exp(-self._max / self._mean)

        u_sample = rng.random()
        u_scaled = F_min + u_sample * (F_max - F_min)
        v = -self._mean * jnp.log(1 - u_scaled)

        return u.maybe_decimal(v * self._time_unit)

    def __repr__(self):
        return (
            f"TruncExp(mean={self._mean}{self._time_unit}, "
            f"min={self._min}{self._time_unit}, "
            f"max={self._max}{self._time_unit})"
        )


class UniformDuration:
    """
    Uniform distribution for sampling time durations.

    Examples
    --------
    >>> duration = UniformDuration(200*u.ms, 400*u.ms)
    >>> Delay(duration)  # Variable delay sampled uniformly
    """

    def __init__(
        self,
        min_val,
        max_val,
        key: Optional[jax.Array] = None,
    ):
        """
        Initialize a uniform duration distribution.

        Parameters
        ----------
        min_val : Quantity or float
            Minimum duration.
        max_val : Quantity or float
            Maximum duration.
        key : jax.Array, optional
            JAX PRNGKey for random number generation.
        """
        min_val = u.Quantity(min_val)
        self._time_unit = min_val.unit

        self._min = min_val.mantissa
        self._max = u.Quantity(max_val).to(self._time_unit).mantissa

        self.rng = brainstate.random.default_rng(key)

    def __call__(self, ctx=None):
        """
        Sample a value from the uniform distribution.

        Parameters
        ----------
        ctx : Context, optional
            If provided, use context's RNG key. Otherwise use internal key.

        Returns
        -------
        Quantity
            Sampled duration value.
        """
        rng = ctx.rng if ctx is not None else self.rng
        v = rng.uniform(self._min, self._max)
        return u.maybe_decimal(v * self._time_unit)

    def __repr__(self):
        return f"UniformDuration({self._min}{self._time_unit}, {self._max}{self._time_unit})"


class Transform:
    """Base class for dataset transformations."""
    pass


class TransformIT(Transform):
    """
    Transformation which transforms input and target separately.

    Parameters
    ----------
    input_transform : Callable, optional
        Transform applied to inputs.
    target_transform : Callable, optional
        Transform applied to targets.
    """

    def __init__(
        self,
        input_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __call__(self, input: Any, output: Any) -> Tuple[Any, Any]:
        if self.input_transform is not None:
            input = self.input_transform(input)
        if self.target_transform is not None:
            output = self.target_transform(output)
        return input, output

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = repr(transform).splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.input_transform is not None:
            body += self._format_transform_repr(self.input_transform, "Input transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")
        return "\n".join(body)


def initialize(
    data: Union[int, float, u.Quantity, Callable],
    allow_none: bool = False
):
    """
    Initialize/resolve a parameter value.

    Parameters
    ----------
    data : int, float, Quantity, or Callable
        Value to resolve. If callable, calls it to get value.
    allow_none : bool
        Whether to allow None values.

    Returns
    -------
    Resolved value.
    """
    if data is None:
        if allow_none:
            return None
        else:
            raise TypeError('Not allow None value.')
    if isinstance(data, (int, float, u.Quantity)):
        return data
    elif callable(data):
        return data()
    else:
        raise TypeError(f'Not support type {type(data)}')


def initialize2(
    data: Union[int, float, u.Quantity, Callable],
    dt: Union[int, float, u.Quantity],
    allow_none: bool = False
) -> Optional[int]:
    """
    Initialize/resolve a parameter and convert to timesteps.

    Parameters
    ----------
    data : int, float, Quantity, or Callable
        Duration value to resolve.
    dt : int, float, or Quantity
        Time step for conversion.
    allow_none : bool
        Whether to allow None values.

    Returns
    -------
    int or None
        Number of timesteps.
    """
    if data is None:
        if allow_none:
            return None
        else:
            raise TypeError('Not allow None value.')
    if dt == 0:
        raise ValueError("dt cannot be zero")
    if isinstance(data, (int, float, u.Quantity)):
        return int(data / dt)
    elif callable(data):
        return int(data() / dt)
    else:
        raise TypeError(f'Not support type {type(data)}')


def interval_of(
    elem: str,
    total: Union[Dict[str, int], Sequence[Tuple[str, int]]]
) -> slice:
    """
    Get slice for a named period in a sequence of periods.

    Parameters
    ----------
    elem : str
        Name of the period to find.
    total : dict or sequence
        Period definitions as {name: duration} or [(name, duration), ...].

    Returns
    -------
    slice
        Slice object for accessing the period in time-indexed arrays.

    Examples
    --------
    >>> periods = {'fixation': 10, 'stimulus': 20, 'delay': 15}
    >>> interval_of('stimulus', periods)
    slice(10, 30, None)
    """
    if isinstance(total, dict):
        total = tuple(total.items())
    s = 0
    for k, v in total:
        if k == elem:
            return slice(s, s + v, None)
        else:
            s += v
    raise ValueError(f'Period "{elem}" not found')


def period_to_arr(periods: Dict[str, int]) -> jax.Array:
    """
    Convert period dictionary to label array.

    Parameters
    ----------
    periods : dict
        Period definitions as {name: duration}.

    Returns
    -------
    jax.Array
        Array where each element is the period index for that timestep.

    Examples
    --------
    >>> periods = {'fixation': 3, 'stimulus': 2}
    >>> period_to_arr(periods)
    array([0, 0, 0, 1, 1])
    """
    res = [jnp.ones(length, dtype=jnp.int32) * i for i, length in enumerate(periods.values())]
    return jnp.concatenate(res)


def firing_rate(base, dt):
    """
    Convert firing rate to spike probability per timestep.

    Parameters
    ----------
    base : float
        Base firing rate in Hz.
    dt : float
        Time step in ms.

    Returns
    -------
    float
        Probability of spike per timestep or scaled rate.
    """
    if base * dt > 1e3:
        raise ValueError(f'dt is too big, so that dt * fr > 1e3.')
    return base * dt / 1e3
