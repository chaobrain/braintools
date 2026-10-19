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

"""
Fixed-step trajectory driver for ODEs.

:func:`odeint` repeats any registered one-step ODE integrator inside a single
compiled ``scan`` and returns the saved trajectory.
"""

import math
from typing import Callable, Optional, Union

import brainstate
import brainunit as u
import jax
import numpy as np

from braintools._misc import set_module_as
from ._registry import get_method

__all__ = [
    'odeint',
]


def _stack_with_initial(y0, ys):
    """Prepend ``y0`` to the leaves of ``ys`` (stacked on axis 0)."""
    return jax.tree.map(
        lambda a, b: u.math.concatenate([u.math.expand_dims(a, 0), b], axis=0),
        y0, ys,
        is_leaf=u.math.is_quantity,
    )


def _as_leading_axis(y0):
    return jax.tree.map(lambda a: u.math.expand_dims(a, 0), y0, is_leaf=u.math.is_quantity)


def _step_ratio(t0, t1, dt) -> float:
    """Return the concrete float ``(t1 - t0) / dt``."""
    try:
        ratio = u.maybe_decimal((t1 - t0) / dt)
    except ZeroDivisionError as e:
        raise ValueError('dt must be non-zero.') from e
    if isinstance(ratio, u.Quantity):
        raise u.UnitMismatchError(
            f'(t1 - t0) / dt must be dimensionless, but has unit {ratio.unit}.'
        )
    try:
        return float(np.asarray(ratio))
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError) as e:
        raise ValueError(
            'odeint computes the static step count from t0, t1 and dt, so they must be '
            'concrete. Under jax.jit, pass `n_steps=` instead of `t1`.'
        ) from e


@set_module_as('braintools.quad')
def odeint(
    f: Callable,
    y0,
    t0,
    t1=None,
    dt=None,
    *args,
    method: Union[str, Callable] = 'rk4',
    save_every: int = 1,
    checkpoint: bool = False,
    n_steps: Optional[int] = None,
    **kwargs,
):
    r"""Integrate an ODE with a fixed step and return the trajectory.

    Runs ``n`` steps of a one-step integrator (see
    :func:`braintools.quad.get_method`) as one compiled ``scan`` inside
    ``brainstate.environ.context(dt=dt)``, saving every ``save_every``-th
    state. Use it instead of a Python ``for`` loop over
    ``ode_*_step`` calls: the step function is traced once and the whole
    loop compiles into a single XLA program.

    Parameters
    ----------
    f : callable
        Right-hand side ``f(y, t, *args, **kwargs) -> dy/dt`` with the same
        PyTree structure (and units ``[y]/[t]``) as ``y``. Note the
        ``(y, t)`` argument order of the :mod:`braintools.quad` steppers.
    y0 : PyTree
        Initial state at ``t0``. Leaves may be arrays or brainunit
        Quantities.
    t0 : float or Quantity
        Initial time.
    t1 : float or Quantity, optional
        Final time. The number of steps is ``n = round((t1 - t0) / dt)``,
        computed on the host, so ``t0``, ``t1`` and ``dt`` must be concrete
        (not traced). Mutually exclusive with ``n_steps``.
    dt : float or Quantity
        Step size (required). May be negative to integrate backwards
        (then ``t1 < t0``).
    *args
        Extra positional arguments forwarded to ``f``.
    method : str or callable, optional
        Name of a registered ``'ode'`` stepper (``'euler'``, ``'rk2'``,
        ``'rk4'``, ``'dopri5'``, ...; see
        :func:`braintools.quad.list_methods`) or a stepper callable with the
        ``step(f, y, t, *args, **kwargs)`` convention. Default ``'rk4'``.
    save_every : int, optional
        Save one state every ``save_every`` steps. Default ``1``.
    checkpoint : bool, optional
        If ``True``, use ``brainstate.transform.checkpointed_scan`` so that
        reverse-mode gradients through long rollouts rematerialise the
        intermediate states instead of storing them all. The values and
        gradients are identical to ``checkpoint=False``. Default ``False``.
    n_steps : int, optional
        Number of steps, given instead of ``t1``. Use this under ``jax.jit``
        when ``t0`` or ``dt`` are traced. Must be a Python ``int >= 0``.
    **kwargs
        Extra keyword arguments forwarded to ``f``.

    Returns
    -------
    ts : array or Quantity
        Saved times, shape ``(n // save_every + 1,)``, starting at ``t0``:
        ``ts[k] = t0 + k * save_every * dt``.
    ys : PyTree
        Saved states with the structure of ``y0``; every leaf gains a leading
        axis of length ``n // save_every + 1`` and ``ys[0] == y0``.

    Raises
    ------
    ValueError
        If ``dt`` is missing or zero, if neither or both of ``t1`` and
        ``n_steps`` are given, if the step count is negative (``t1`` lies
        behind ``t0`` in the direction of ``dt``), if ``n_steps`` is not a
        non-negative ``int``, if ``save_every < 1``, or if ``t1`` is given
        but ``t0``, ``t1`` or ``dt`` is traced (use ``n_steps`` under jit).
    brainunit.UnitMismatchError
        If ``(t1 - t0) / dt`` is not dimensionless.
    KeyError
        If ``method`` is an unknown name.

    See Also
    --------
    solve_ivp : Adaptive-step solver with error control.
    get_method : Resolve a stepper by name.

    Notes
    -----
    The ``i``-th step is taken at time ``t0 + i * dt`` (computed from the
    step index, so round-off does not accumulate in ``t``).

    If ``save_every`` does not divide ``n``, the trajectory stops at the last
    saved state: only ``(n // save_every) * save_every`` steps are run, and the
    last saved time is ``t0 + (n // save_every) * save_every * dt``, which is
    earlier than ``t1``. Choose ``save_every`` dividing ``n`` to reach ``t1``.
    ``t1 == t0`` (or ``n < save_every``) returns ``ts == [t0]`` and ``ys``
    holding only ``y0``.

    Since ``n`` is rounded, the last time ``t0 + n * dt`` can differ from
    ``t1`` by up to ``dt / 2``.

    Transformations: ``odeint`` is ``jax.jit``-compatible (with ``n_steps``
    instead of ``t1`` when times are traced; ``method``, ``save_every``,
    ``checkpoint`` and ``n_steps`` are static), ``jax.vmap``-compatible over
    ``y0``, ``args`` and ``kwargs``, and differentiable in forward and reverse
    mode with respect to ``y0``, ``args``, ``kwargs``, ``t0`` and ``dt``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.quad import odeint
        >>> ts, ys = odeint(lambda y, t: -y, jnp.array(1.0), 0.0, 1.0, 0.1)
        >>> ts.shape, ys.shape
        ((11,), (11,))
        >>> print(f'{float(ys[-1]):.5f}  exact {float(jnp.exp(-1.0)):.5f}')
        0.36788  exact 0.36788
        >>>
        >>> ts, ys = odeint(lambda y, t: -y, jnp.array(1.0), 0.0, 1.0, 0.1,
        ...                 method='euler', save_every=5)
        >>> ts.shape
        (3,)

    Pytree and unit-aware states:

    .. code-block:: python

        >>> import brainunit as u
        >>> def neuron(s, t, I):
        ...     return {'V': (-65. * u.mV - s['V'] + 10. * u.Mohm * I) / (10. * u.ms)}
        >>> ts, ys = odeint(neuron, {'V': -65. * u.mV}, 0. * u.ms, 50. * u.ms,
        ...                 0.1 * u.ms, 0.5 * u.nA)
        >>> ys['V'].shape
        (501,)
        >>> print(u.math.round(ys['V'][-1], 2))
        -60.03 mV
    """
    if dt is None:
        raise ValueError('odeint requires the step size `dt`.')
    if not isinstance(save_every, (int, np.integer)) or isinstance(save_every, bool) or save_every < 1:
        raise ValueError(f'save_every must be a positive int, but got {save_every!r}.')
    save_every = int(save_every)
    step = method if callable(method) else get_method(method, 'ode')

    if n_steps is None:
        if t1 is None:
            raise ValueError('odeint requires either `t1` or `n_steps`.')
        ratio = _step_ratio(t0, t1, dt)
        if not math.isfinite(ratio):
            raise ValueError('dt must be non-zero and finite.')
        n = int(round(ratio))
        if n < 0:
            raise ValueError(
                f'The step count round((t1 - t0) / dt) = {n} is negative: t1 lies behind t0 '
                f'in the direction of dt. Use a dt of the opposite sign.'
            )
    else:
        if t1 is not None:
            raise ValueError('Pass either `t1` or `n_steps`, not both.')
        if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError(f'n_steps must be a non-negative int, but got {n_steps!r}.')
        n = int(n_steps)

    n_saved = n // save_every
    ts = t0 + dt * (save_every * u.math.arange(n_saved + 1))
    if n_saved == 0:
        return ts, _as_leading_axis(y0)

    def one_step(y, i):
        with brainstate.environ.context(dt=dt):
            return step(f, y, t0 + i * dt, *args, **kwargs)

    def chunk(carry, _):
        y, i = carry
        if save_every == 1:
            y = one_step(y, i)
        else:
            def inner(y_, j):
                return one_step(y_, i + j), None

            y, _ = brainstate.transform.scan(inner, y, u.math.arange(save_every))
        return (y, i + save_every), y

    scan = brainstate.transform.checkpointed_scan if checkpoint else brainstate.transform.scan
    _, ys = scan(chunk, (y0, 0), None, length=n_saved)
    return ts, _stack_with_initial(y0, ys)
