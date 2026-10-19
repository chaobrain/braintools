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
Name-based registry of one-step integrators.

Every built-in stepper of :mod:`braintools.quad` is registered under a short
name (``'euler'``, ``'rk4'``, ``'dopri5'``, ...) for each equation kind
(``'ode'``, ``'sde'``, ``'imex'``, ``'dde'``). Drivers such as
:func:`braintools.quad.odeint` resolve ``method=`` strings through this
registry, and downstream packages can add their own solvers with
:func:`register_method`.
"""

import difflib
from typing import Callable, Dict, List, Optional, Union

from braintools._misc import set_module_as
from ._dde_integrator import (
    dde_euler_step,
    dde_heun_step,
    dde_rk4_step,
    dde_euler_pc_step,
    dde_heun_pc_step,
)
from ._imex_integrator import (
    imex_euler_step,
    imex_ars222_step,
    imex_cnab_step,
)
from ._ode_integrator import (
    ode_euler_step,
    ode_rk2_step,
    ode_rk3_step,
    ode_rk4_step,
    ode_expeuler_step,
    ode_midpoint_step,
    ode_heun_step,
    ode_rk4_38_step,
    ode_rk45_step,
    ode_rk23_step,
    ode_dopri5_step,
    ode_rk45_dopri_step,
    ode_rkf45_step,
    ode_ssprk33_step,
    ode_dopri8_step,
    ode_rk87_dopri_step,
    ode_bs32_step,
    ode_ralston2_step,
    ode_ralston3_step,
)
from ._sde_integrator import (
    sde_euler_step,
    sde_milstein_step,
    sde_expeuler_step,
    sde_heun_step,
    sde_tamed_euler_step,
    sde_implicit_euler_step,
    sde_srk2_step,
    sde_srk3_step,
    sde_srk4_step,
)

__all__ = [
    'register_method',
    'get_method',
    'list_methods',
]

#: The equation kinds understood by the registry.
KINDS = ('ode', 'sde', 'imex', 'dde')

_REGISTRY: Dict[str, Dict[str, Callable]] = {
    'ode': {
        'euler': ode_euler_step,
        'rk2': ode_rk2_step,
        'rk3': ode_rk3_step,
        'rk4': ode_rk4_step,
        'expeuler': ode_expeuler_step,
        'midpoint': ode_midpoint_step,
        'heun': ode_heun_step,
        'rk4_38': ode_rk4_38_step,
        'rk45': ode_rk45_step,
        'rk23': ode_rk23_step,
        'dopri5': ode_dopri5_step,
        'rk45_dopri': ode_rk45_dopri_step,
        'rkf45': ode_rkf45_step,
        'ssprk33': ode_ssprk33_step,
        'dopri8': ode_dopri8_step,
        'rk87_dopri': ode_rk87_dopri_step,
        'bs32': ode_bs32_step,
        'ralston2': ode_ralston2_step,
        'ralston3': ode_ralston3_step,
    },
    'sde': {
        'euler': sde_euler_step,
        'milstein': sde_milstein_step,
        'expeuler': sde_expeuler_step,
        'heun': sde_heun_step,
        'tamed_euler': sde_tamed_euler_step,
        'implicit_euler': sde_implicit_euler_step,
        'srk2': sde_srk2_step,
        'srk3': sde_srk3_step,
        'srk4': sde_srk4_step,
    },
    'imex': {
        'euler': imex_euler_step,
        'ars222': imex_ars222_step,
        'cnab': imex_cnab_step,
    },
    'dde': {
        'euler': dde_euler_step,
        'heun': dde_heun_step,
        'rk4': dde_rk4_step,
        'euler_pc': dde_euler_pc_step,
        'heun_pc': dde_heun_pc_step,
    },
}


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f'Unknown integrator kind {kind!r}. Expected one of {KINDS}.')
    return kind


def _normalize(name: str, kind: str) -> str:
    # Accept the full public function name, e.g. ``'ode_rk4_step'`` -> ``'rk4'``.
    prefix, suffix = f'{kind}_', '_step'
    if name.startswith(prefix) and name.endswith(suffix) and len(name) > len(prefix) + len(suffix):
        return name[len(prefix):-len(suffix)]
    return name


@set_module_as('braintools.quad')
def register_method(
    name: str,
    fn: Callable,
    kind: str = 'ode',
    *,
    overwrite: bool = False,
) -> Callable:
    """Register a one-step integrator under a short name.

    After registration the stepper can be selected by name in the drivers,
    e.g. ``odeint(f, y0, t0, t1, dt, method=name)``, and looked up with
    :func:`get_method`. The stepper must follow the calling convention of the
    built-in steppers of its kind (for ``'ode'``: ``fn(f, y, t, *args,
    **kwargs) -> y_next``, reading the step size from
    ``brainstate.environ.get_dt()``).

    Parameters
    ----------
    name : str
        Short name of the method, e.g. ``'my_rk'``. Must be a non-empty string.
    fn : callable
        The one-step integrator.
    kind : {'ode', 'sde', 'imex', 'dde'}, optional
        Equation kind the stepper solves. Default ``'ode'``.
    overwrite : bool, optional
        Replace an existing entry with the same name. Default ``False``.

    Returns
    -------
    callable
        ``fn`` unchanged, so the call can be used inline.

    Raises
    ------
    TypeError
        If ``name`` is not a non-empty string or ``fn`` is not callable.
    ValueError
        If ``kind`` is unknown, or ``name`` is already registered for ``kind``
        and ``overwrite`` is ``False``.

    See Also
    --------
    get_method : Look up a registered stepper.
    list_methods : List the registered names.

    Notes
    -----
    The registry is a process-wide dictionary; registration is a host-side
    Python operation and has nothing to do with ``jit``/``vmap``/``grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> from braintools.quad import register_method, get_method
        >>> def forward_euler(f, y, t, *args, **kwargs):
        ...     return y + brainstate.environ.get_dt() * f(y, t, *args, **kwargs)
        >>> _ = register_method('my_euler', forward_euler)
        >>> get_method('my_euler') is forward_euler
        True
    """
    _check_kind(kind)
    if not isinstance(name, str) or not name:
        raise TypeError(f'The method name must be a non-empty string, but got {name!r}.')
    if not callable(fn):
        raise TypeError(f'The method {name!r} must be callable, but got {type(fn).__name__}.')
    table = _REGISTRY[kind]
    if name in table and not overwrite and table[name] is not fn:
        raise ValueError(
            f'A {kind!r} method named {name!r} is already registered. '
            f'Pass overwrite=True to replace it.'
        )
    table[name] = fn
    return fn


@set_module_as('braintools.quad')
def get_method(name: str, kind: str = 'ode') -> Callable:
    """Look up a registered one-step integrator by name.

    Parameters
    ----------
    name : str
        Short name of the method (e.g. ``'rk4'``, ``'dopri5'``,
        ``'milstein'``). The full public function name (e.g.
        ``'ode_rk4_step'``) is accepted too.
    kind : {'ode', 'sde', 'imex', 'dde'}, optional
        Equation kind. Default ``'ode'``.

    Returns
    -------
    callable
        The registered stepper.

    Raises
    ------
    KeyError
        If no method of that name is registered for ``kind``. The message
        lists the closest registered names (``difflib``).
    ValueError
        If ``kind`` is unknown.
    TypeError
        If ``name`` is not a string.

    See Also
    --------
    register_method : Add a stepper to the registry.
    list_methods : List the registered names.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.quad import get_method, ode_rk4_step, sde_milstein_step
        >>> get_method('rk4') is ode_rk4_step
        True
        >>> get_method('milstein', kind='sde') is sde_milstein_step
        True
        >>> get_method('ode_rk4_step') is ode_rk4_step
        True
    """
    _check_kind(kind)
    if not isinstance(name, str):
        raise TypeError(f'The method name must be a string, but got {type(name).__name__}.')
    table = _REGISTRY[kind]
    key = name if name in table else _normalize(name, kind)
    if key in table:
        return table[key]
    close = difflib.get_close_matches(name, list(table), n=5, cutoff=0.5)
    hint = f' Did you mean one of {close}?' if close else ''
    raise KeyError(
        f'Unknown {kind!r} method {name!r}.{hint} '
        f'Available methods: {sorted(table)}.'
    )


@set_module_as('braintools.quad')
def list_methods(kind: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
    """List the registered method names.

    Parameters
    ----------
    kind : {'ode', 'sde', 'imex', 'dde'} or None, optional
        Equation kind. If ``None`` (default), list every kind.

    Returns
    -------
    list of str or dict
        The sorted names registered for ``kind``; if ``kind`` is ``None``, a
        dict mapping each kind to its sorted list of names.

    Raises
    ------
    ValueError
        If ``kind`` is not ``None`` and unknown.

    See Also
    --------
    get_method : Look up a registered stepper.
    register_method : Add a stepper to the registry.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.quad import list_methods
        >>> list_methods('imex')
        ['ars222', 'cnab', 'euler']
        >>> sorted(list_methods())
        ['dde', 'imex', 'ode', 'sde']
    """
    if kind is None:
        return {k: sorted(v) for k, v in _REGISTRY.items()}
    _check_kind(kind)
    return sorted(_REGISTRY[kind])
