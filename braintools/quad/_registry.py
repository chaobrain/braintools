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
(``'ode'``, ``'sde'``, ``'imex'``, ``'dde'``), together with its metadata
(convergence order, category, aliases and a description, see
:class:`MethodInfo`). Drivers such as :func:`braintools.quad.odeint` resolve
``method=`` strings through this registry, and downstream packages can add
their own solvers with :func:`register_method`.

The ``'module'`` kind holds no built-in steppers. It is a namespace for
stateful hosts (such as braincell) whose steps advance a model in place,
``step(target, *args) -> None``; the drivers of this package do not accept it.
"""

import difflib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

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
    ode_ralston4_step,
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
    'MethodInfo',
    'register_method',
    'get_method',
    'method_info',
    'list_methods',
]

#: The equation kinds understood by the registry. ``'module'`` holds steps of
#: stateful hosts, ``step(target, *args) -> None``; it has no built-in entries.
KINDS = ('ode', 'sde', 'imex', 'dde', 'module')

Order = Union[int, float]


@set_module_as('braintools.quad')
@dataclass(frozen=True)
class MethodInfo:
    """Metadata of a registered one-step integrator.

    Returned by :func:`method_info`. Instances are immutable snapshots: later
    registrations do not change an instance already returned.

    Attributes
    ----------
    name : str
        Canonical short name of the method (an alias resolves to it).
    kind : str
        Equation kind, one of :data:`KINDS`.
    fn : callable
        The registered stepper.
    order : int or float or None
        Convergence order (the strong order for SDE steppers; for an embedded
        pair, the order of the solution it returns). ``None`` if not recorded.
    category : str or None
        Family of the method, e.g. ``'explicit'``, ``'embedded'`` or
        ``'exponential'`` for the built-in ODE steppers. ``None`` if not
        recorded.
    aliases : tuple of str
        Alternative names that resolve to this method.
    description : str
        One-line description; empty if not recorded.

    See Also
    --------
    method_info : Look up the metadata of a method.
    register_method : Register a method with its metadata.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.quad import method_info
        >>> info = method_info('heun3')
        >>> info.name, info.order, info.category
        ('heun', 3, 'explicit')
    """

    name: str
    kind: str
    fn: Callable
    order: Optional[Order] = None
    category: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    description: str = ''

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
        'ralston4': ode_ralston4_step,
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
    'module': {},
}

# Metadata per kind and canonical name: ``order``, ``category``, ``description``.
_META: Dict[str, Dict[str, dict]] = {kind: {} for kind in KINDS}
# Aliases per kind: alias -> canonical name.
_ALIASES: Dict[str, Dict[str, str]] = {kind: {} for kind in KINDS}


def _builtin(kind: str, name: str, order, category=None, description='', aliases=()):
    _META[kind][name] = dict(order=order, category=category, description=description)
    for alias in aliases:
        _ALIASES[kind][alias] = name


for _name, _order, _desc, _aliases in (
    ('euler', 1, 'Forward (explicit) Euler method.', ()),
    ('midpoint', 2, 'Explicit midpoint method.', ()),
    ('rk2', 2, "Heun's second-order method (explicit trapezoid).", ('heun2',)),
    ('ralston2', 2, "Ralston's second-order method.", ()),
    ('rk3', 3, "Kutta's third-order method.", ()),
    ('heun', 3, "Heun's third-order method.", ('heun3',)),
    ('ssprk33', 3, 'Strong-stability-preserving RK(3,3) (Shu-Osher).', ('ssprk3',)),
    ('ralston3', 3, "Ralston's third-order method.", ()),
    ('rk4', 4, 'Classical fourth-order Runge-Kutta.', ()),
    ('rk4_38', 4, 'Fourth-order Runge-Kutta, 3/8 rule.', ()),
    ('ralston4', 4, "Ralston's fourth-order method (minimum error bound).", ()),
):
    _builtin('ode', _name, _order, 'explicit', _desc, _aliases)
for _name, _order, _desc in (
    ('rk23', 3, 'Bogacki-Shampine 3(2) embedded pair.'),
    ('bs32', 3, 'Bogacki-Shampine 3(2) embedded pair.'),
    ('rk45', 5, 'Cash-Karp 4(5) embedded pair.'),
    ('rkf45', 5, 'Runge-Kutta-Fehlberg 4(5) embedded pair.'),
    ('dopri5', 5, 'Dormand-Prince 5(4) embedded pair.'),
    ('rk45_dopri', 5, 'Dormand-Prince 5(4) embedded pair.'),
    ('dopri8', 8, 'Dormand-Prince 8(7) embedded pair (DOP853).'),
    ('rk87_dopri', 8, 'Dormand-Prince 8(7) embedded pair (DOP853).'),
):
    _builtin('ode', _name, _order, 'embedded', _desc)
_builtin('ode', 'expeuler', 1, 'exponential', 'Exponential Euler with the diagonal Jacobian.')
_builtin('sde', 'euler', 0.5, description='Euler-Maruyama scheme.')
_builtin('sde', 'milstein', 1.0, description='Milstein scheme.')
_builtin('imex', 'euler', 1, description='First-order IMEX Euler.')
_builtin('imex', 'ars222', 2, description='Second-order L-stable IMEX Runge-Kutta ARS(2,2,2).')
_builtin('imex', 'cnab', 2, description='Crank-Nicolson / Adams-Bashforth IMEX step.')
_builtin('dde', 'euler', 1, description='Euler method for delay equations.')
_builtin('dde', 'heun', 2, description='Heun method for delay equations.')
_builtin('dde', 'rk4', 4, description='Fourth-order Runge-Kutta for delay equations.')


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


def _resolve(name: str, kind: str) -> Optional[str]:
    """Return the canonical name ``name`` refers to in ``kind``, or ``None``."""
    table, aliases = _REGISTRY[kind], _ALIASES[kind]
    for key in (name, _normalize(name, kind)):
        if key in table:
            return key
        target = aliases.get(key)
        if target is not None and target in table:
            return target
    return None


def _unknown(name: str, kind: str) -> KeyError:
    table = _REGISTRY[kind]
    known = list(table) + [a for a, target in _ALIASES[kind].items() if target in table]
    close = difflib.get_close_matches(name, known, n=5, cutoff=0.5)
    hint = f' Did you mean one of {close}?' if close else ''
    return KeyError(
        f'Unknown {kind!r} method {name!r}.{hint} '
        f'Available methods: {sorted(table)}.'
    )


def _check_metadata(name, order, category, aliases, description) -> Tuple[str, ...]:
    if order is not None and (isinstance(order, bool) or not isinstance(order, (int, float)) or order <= 0):
        raise TypeError(f'The order of {name!r} must be a positive number or None, but got {order!r}.')
    if category is not None and not isinstance(category, str):
        raise TypeError(f'The category of {name!r} must be a string or None, but got {category!r}.')
    if not isinstance(description, str):
        raise TypeError(f'The description of {name!r} must be a string, but got {description!r}.')
    if isinstance(aliases, str):
        aliases = (aliases,)
    aliases = tuple(aliases)
    for alias in aliases:
        if not isinstance(alias, str) or not alias:
            raise TypeError(f'Aliases of {name!r} must be non-empty strings, but got {alias!r}.')
    if name in aliases or len(set(aliases)) != len(aliases):
        raise ValueError(f'The aliases of {name!r} must be unique and differ from the name, but got {aliases!r}.')
    return aliases


@set_module_as('braintools.quad')
def register_method(
    name: str,
    fn: Callable,
    kind: str = 'ode',
    *,
    overwrite: bool = False,
    order: Optional[Order] = None,
    category: Optional[str] = None,
    aliases: Tuple[str, ...] = (),
    description: str = '',
) -> Callable:
    """Register a one-step integrator under a short name.

    After registration the stepper can be selected by name in the drivers,
    e.g. ``odeint(f, y0, t0, t1, dt, method=name)``, and looked up with
    :func:`get_method` (by its name or any alias) and :func:`method_info`.
    The stepper must follow the calling convention of the built-in steppers of
    its kind (for ``'ode'``: ``fn(f, y, t, *args, **kwargs) -> y_next``,
    reading the step size from ``brainstate.environ.get_dt()``). Steppers of
    the ``'module'`` kind advance a stateful model in place,
    ``fn(target, *args) -> None``, reading ``t`` and ``dt`` from
    ``brainstate.environ``; the drivers of this package do not accept them.

    Parameters
    ----------
    name : str
        Short name of the method, e.g. ``'my_rk'``. Must be a non-empty string.
    fn : callable
        The one-step integrator.
    kind : {'ode', 'sde', 'imex', 'dde', 'module'}, optional
        Equation kind the stepper solves. Default ``'ode'``.
    overwrite : bool, optional
        Replace an existing entry with the same name, and take over aliases
        that already point to another method. Default ``False``.
    order : int or float, optional
        Convergence order (strong order for SDE steppers). Default ``None``.
    category : str, optional
        Family of the method, e.g. ``'explicit'``. Default ``None``.
    aliases : tuple of str, optional
        Alternative names that resolve to this method. Default ``()``.
    description : str, optional
        One-line description. Default ``''``.

    Returns
    -------
    callable
        ``fn`` unchanged, so the call can be used inline.

    Raises
    ------
    TypeError
        If ``name`` is not a non-empty string, ``fn`` is not callable, or a
        metadata argument has the wrong type (``order`` must be a positive
        number).
    ValueError
        If ``kind`` is unknown; if ``name`` is already registered for ``kind``
        with another function, or is an alias of another method, and
        ``overwrite`` is ``False``; if an alias is the name of another
        registered method (never allowed), or already an alias of another
        method and ``overwrite`` is ``False``; or if the aliases repeat or
        contain ``name``.

    See Also
    --------
    get_method : Look up a registered stepper.
    method_info : Look up a method's metadata.
    list_methods : List the registered names.

    Notes
    -----
    The registry is a process-wide dictionary; registration is a host-side
    Python operation and has nothing to do with ``jit``/``vmap``/``grad``.

    Registering the same function under the same name again is allowed
    without ``overwrite``. Without any metadata argument it is a no-op;
    otherwise the given metadata and aliases replace the recorded ones.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> from braintools.quad import register_method, get_method, method_info
        >>> def forward_euler(f, y, t, *args, **kwargs):
        ...     return y + brainstate.environ.get_dt() * f(y, t, *args, **kwargs)
        >>> # ``overwrite=True`` lets the example be re-run in the same process.
        >>> _ = register_method('my_euler', forward_euler, overwrite=True,
        ...                     order=1, category='explicit', aliases=('my_fe',))
        >>> get_method('my_fe') is forward_euler
        True
        >>> method_info('my_euler').order
        1
    """
    _check_kind(kind)
    if not isinstance(name, str) or not name:
        raise TypeError(f'The method name must be a non-empty string, but got {name!r}.')
    if not callable(fn):
        raise TypeError(f'The method {name!r} must be callable, but got {type(fn).__name__}.')
    aliases = _check_metadata(name, order, category, aliases, description)
    table, alias_map = _REGISTRY[kind], _ALIASES[kind]
    if name in table and not overwrite and table[name] is not fn:
        raise ValueError(
            f'A {kind!r} method named {name!r} is already registered. '
            f'Pass overwrite=True to replace it.'
        )
    owner = alias_map.get(name)
    if owner is not None and owner != name and owner in table and not overwrite:
        raise ValueError(
            f'{name!r} is already an alias of the {kind!r} method {owner!r}. '
            f'Pass overwrite=True to replace it.'
        )
    for alias in aliases:
        if alias in table and alias != name:
            raise ValueError(f'The alias {alias!r} of {name!r} is already the name of a {kind!r} method.')
        target = alias_map.get(alias)
        if target is not None and target != name and target in table and not overwrite:
            raise ValueError(
                f'The alias {alias!r} of {name!r} already refers to the {kind!r} method {target!r}. '
                f'Pass overwrite=True to replace it.'
            )
    no_metadata = order is None and category is None and not aliases and description == ''
    if table.get(name) is fn and not overwrite and no_metadata:
        return fn
    alias_map.pop(name, None)
    for alias in [a for a, target in alias_map.items() if target == name]:
        del alias_map[alias]
    for alias in aliases:
        alias_map[alias] = name
    table[name] = fn
    _META[kind][name] = dict(order=order, category=category, description=description)
    return fn


@set_module_as('braintools.quad')
def get_method(name: str, kind: str = 'ode') -> Callable:
    """Look up a registered one-step integrator by name.

    Parameters
    ----------
    name : str
        Short name of the method (e.g. ``'rk4'``, ``'dopri5'``,
        ``'milstein'``) or one of its aliases (e.g. ``'heun3'``). The full
        public function name (e.g. ``'ode_rk4_step'``) is accepted too.
    kind : {'ode', 'sde', 'imex', 'dde', 'module'}, optional
        Equation kind. Default ``'ode'``.

    Returns
    -------
    callable
        The registered stepper.

    Raises
    ------
    KeyError
        If no method of that name or alias is registered for ``kind``. The message
        lists the closest registered names (``difflib``).
    ValueError
        If ``kind`` is unknown.
    TypeError
        If ``name`` is not a string.

    See Also
    --------
    register_method : Add a stepper to the registry.
    method_info : Look up a method's metadata.
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
        >>> get_method('heun2') is get_method('rk2')
        True
    """
    _check_kind(kind)
    if not isinstance(name, str):
        raise TypeError(f'The method name must be a string, but got {type(name).__name__}.')
    key = _resolve(name, kind)
    if key is None:
        raise _unknown(name, kind)
    return _REGISTRY[kind][key]


@set_module_as('braintools.quad')
def method_info(name: str, kind: str = 'ode') -> MethodInfo:
    """Look up the metadata of a registered one-step integrator.

    Parameters
    ----------
    name : str
        Short name, alias (e.g. ``'heun3'``) or full public function name
        (e.g. ``'ode_rk4_step'``) of the method.
    kind : {'ode', 'sde', 'imex', 'dde', 'module'}, optional
        Equation kind. Default ``'ode'``.

    Returns
    -------
    MethodInfo
        A snapshot of the method's canonical name, stepper and metadata.

    Raises
    ------
    KeyError
        If no method of that name or alias is registered for ``kind``. The
        message lists the closest registered names (``difflib``).
    ValueError
        If ``kind`` is unknown.
    TypeError
        If ``name`` is not a string.

    See Also
    --------
    get_method : Look up the stepper only.
    register_method : Register a method with its metadata.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.quad import method_info, ode_ssprk33_step
        >>> info = method_info('ssprk3')
        >>> info.name, info.order, info.fn is ode_ssprk33_step
        ('ssprk33', 3, True)
        >>> method_info('ode_rk4_step').aliases
        ()
    """
    _check_kind(kind)
    if not isinstance(name, str):
        raise TypeError(f'The method name must be a string, but got {type(name).__name__}.')
    key = _resolve(name, kind)
    if key is None:
        raise _unknown(name, kind)
    meta = _META[kind].get(key, {})
    return MethodInfo(
        name=key,
        kind=kind,
        fn=_REGISTRY[kind][key],
        order=meta.get('order'),
        category=meta.get('category'),
        aliases=tuple(a for a, target in _ALIASES[kind].items() if target == key),
        description=meta.get('description', ''),
    )


@set_module_as('braintools.quad')
def list_methods(kind: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
    """List the registered method names.

    Parameters
    ----------
    kind : {'ode', 'sde', 'imex', 'dde', 'module'} or None, optional
        Equation kind. If ``None`` (default), list every kind.

    Returns
    -------
    list of str or dict
        The sorted canonical names registered for ``kind`` (aliases are not
        listed; see :func:`method_info`); if ``kind`` is ``None``, a dict
        mapping each kind to its sorted list of names.

    Raises
    ------
    ValueError
        If ``kind`` is not ``None`` and unknown.

    See Also
    --------
    get_method : Look up a registered stepper.
    method_info : Look up a method's metadata.
    register_method : Add a stepper to the registry.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.quad import list_methods
        >>> list_methods('imex')
        ['ars222', 'cnab', 'euler']
        >>> sorted(list_methods())
        ['dde', 'imex', 'module', 'ode', 'sde']
    """
    if kind is None:
        return {k: sorted(v) for k, v in _REGISTRY.items()}
    _check_kind(kind)
    return sorted(_REGISTRY[kind])
