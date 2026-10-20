# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Library-wide deprecation machinery.

Every deprecated public name in ``braintools`` is created through this module,
so that

- the warning is emitted when the deprecated object is *called* (or a class is
  instantiated), never when it is imported or accessed -- downstream packages
  bind ``braintools`` names at import time;
- all warnings share one category, :class:`BraintoolsDeprecationWarning`, a
  :class:`FutureWarning` subclass so that Python shows it by default, which
  the test suite turns into an error outside ``pytest.warns``;
- every deprecation, including deprecated arguments and argument values, is
  recorded in a registry (:func:`register_deprecation`) and emitted through
  :func:`warn_deprecated`; tests check the registry for wiring, for the
  scheduled removal version, and against the migration guide
  (``docs/migration.md``).

To turn every ``braintools`` deprecation into an error, run Python with
``-W error::FutureWarning`` or install
``warnings.simplefilter('error', BraintoolsDeprecationWarning)``.
"""

import dataclasses
import functools
import inspect
import warnings
from typing import Any, Callable, Tuple, Type

__all__ = [
    'BraintoolsDeprecationWarning',
    'DeprecationRecord',
    'register_deprecation',
    'warn_deprecated',
    'deprecated_function',
    'deprecated_class',
    'registered_deprecations',
]

_KINDS = ('function', 'class', 'argument', 'feature')


class BraintoolsDeprecationWarning(FutureWarning):
    """Warning category for deprecated ``braintools`` APIs.

    The class derives from :class:`FutureWarning`, not
    :class:`DeprecationWarning`. Python shows ``FutureWarning`` under its
    default filters wherever it is raised, which is the convention for
    deprecations aimed at the end users of a library; a ``DeprecationWarning``
    raised from library code (outside ``__main__``) would stay hidden. As a
    dedicated subclass it also lets users and the test suite filter
    ``braintools`` deprecations independently of warnings from other
    libraries.

    Notes
    -----
    Up to braintools 0.3 this class derived from :class:`DeprecationWarning`.
    Filters written against ``DeprecationWarning`` no longer match it; filter
    on this class (or on ``FutureWarning``) instead.

    Examples
    --------
    .. code-block:: python

        >>> import warnings
        >>> from braintools._deprecation import BraintoolsDeprecationWarning
        >>> issubclass(BraintoolsDeprecationWarning, FutureWarning)
        True
        >>> issubclass(BraintoolsDeprecationWarning, DeprecationWarning)
        False
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter('always')
        ...     warnings.filterwarnings('ignore', category=BraintoolsDeprecationWarning)
        ...     warnings.warn('old API', BraintoolsDeprecationWarning)
        >>> caught
        []
    """


@dataclasses.dataclass(frozen=True)
class DeprecationRecord:
    """Description of one deprecated API.

    Parameters
    ----------
    old_name : str
        Fully qualified deprecated name, e.g. ``'braintools.metric.firing_rate'``.
    new_name : str
        Replacement shown to users: a dotted path or a short phrase.
    since : str
        Version in which the deprecation was introduced.
    removed_in : str
        Version in which the deprecated API will be removed.
    kind : str
        One of ``'function'``, ``'class'``, ``'argument'`` or ``'feature'``.
    alias : object, optional
        The deprecated alias object, for ``'function'`` and ``'class'`` records.
    target : object, optional
        The object the alias forwards to.
    """

    old_name: str
    new_name: str
    since: str
    removed_in: str
    kind: str
    alias: Any = dataclasses.field(default=None, compare=False, repr=False)
    target: Any = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def message(self) -> str:
        """str: The warning message emitted for this deprecation."""
        return (
            f"`{self.old_name}` is deprecated since {self.since} and will be "
            f"removed in {self.removed_in}. Use `{self.new_name}` instead."
        )


_REGISTRY: list = []


def _version_tuple(version: str) -> Tuple[int, ...]:
    """Parse ``'X.Y.Z'`` (ignoring any non-numeric suffix) into a tuple."""
    parts = []
    for piece in version.split('.'):
        digits = ''
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def register_deprecation(
    old_name: str,
    new_name: str,
    *,
    since: str,
    removed_in: str,
    kind: str,
    alias: Any = None,
    target: Any = None,
) -> DeprecationRecord:
    """Record a deprecation in the library-wide registry.

    Parameters
    ----------
    old_name : str
        Fully qualified deprecated name.
    new_name : str
        Replacement shown to users.
    since : str
        Version in which the deprecation was introduced.
    removed_in : str
        Version in which the deprecated API will be removed.
    kind : str
        One of ``'function'``, ``'class'``, ``'argument'`` or ``'feature'``.
    alias : object, optional
        The deprecated alias object.
    target : object, optional
        The object the alias forwards to.

    Returns
    -------
    DeprecationRecord
        The registered record; pass it to :func:`warn_deprecated`.

    Raises
    ------
    ValueError
        If ``kind`` is unknown or ``removed_in`` is not later than ``since``.
    """
    if kind not in _KINDS:
        raise ValueError(f'Unknown deprecation kind {kind!r}; expected one of {_KINDS}.')
    if _version_tuple(removed_in) <= _version_tuple(since):
        raise ValueError(
            f'removed_in ({removed_in}) must be later than since ({since}) for `{old_name}`.'
        )
    record = DeprecationRecord(
        old_name=old_name,
        new_name=new_name,
        since=since,
        removed_in=removed_in,
        kind=kind,
        alias=alias,
        target=target,
    )
    _REGISTRY.append(record)
    return record


def warn_deprecated(record: DeprecationRecord, stacklevel: int = 2) -> None:
    """Emit the :class:`BraintoolsDeprecationWarning` for ``record``.

    Parameters
    ----------
    record : DeprecationRecord
        The deprecation to warn about.
    stacklevel : int, default 2
        Passed to :func:`warnings.warn`; ``2`` attributes the warning to the
        caller of the function that calls ``warn_deprecated``.
    """
    warnings.warn(record.message, BraintoolsDeprecationWarning, stacklevel=stacklevel + 1)


def _split_name(old_name: str) -> Tuple[str, str]:
    module, _, short = old_name.rpartition('.')
    return module, short


def deprecated_function(
    new_func: Callable,
    old_name: str,
    new_name: str,
    *,
    since: str = '0.4.0',
    removed_in: str = '1.0.0',
) -> Callable:
    """Create a deprecated alias of a function that warns when called.

    Parameters
    ----------
    new_func : callable
        The function the alias forwards to.
    old_name : str
        Fully qualified deprecated name, e.g. ``'braintools.metric.firing_rate'``.
    new_name : str
        Fully qualified replacement name.
    since : str, default '0.4.0'
        Version in which the deprecation was introduced.
    removed_in : str, default '1.0.0'
        Version in which the alias will be removed.

    Returns
    -------
    callable
        A wrapper that emits :class:`BraintoolsDeprecationWarning` on every
        call and then forwards all arguments to ``new_func``.

    Examples
    --------
    .. code-block:: python

        >>> import warnings
        >>> from braintools._deprecation import deprecated_function
        >>> def add(a, b):
        ...     return a + b
        >>> plus = deprecated_function(add, 'mypkg.plus', 'mypkg.add')
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter('always')
        ...     result = plus(1, 2)
        >>> result
        3
        >>> caught[0].category.__name__
        'BraintoolsDeprecationWarning'
    """
    module, short = _split_name(old_name)

    @functools.wraps(new_func)
    def alias(*args, **kwargs):
        warn_deprecated(record)
        return new_func(*args, **kwargs)

    alias.__name__ = short
    alias.__qualname__ = short
    if module:
        alias.__module__ = module
    record = register_deprecation(
        old_name, new_name, since=since, removed_in=removed_in,
        kind='function', alias=alias, target=new_func,
    )
    setattr(alias, '__deprecated__', record.message)
    return alias


def deprecated_class(
    new_class: Type,
    old_name: str,
    new_name: str,
    *,
    since: str = '0.4.0',
    removed_in: str = '1.0.0',
) -> Type:
    """Create a deprecated alias of a class that warns when instantiated.

    The alias is a subclass of ``new_class``, so instances pass
    ``isinstance(obj, new_class)`` checks.

    Parameters
    ----------
    new_class : type
        The class the alias forwards to.
    old_name : str
        Fully qualified deprecated name, e.g. ``'braintools.PoissonEncoder'``.
    new_name : str
        Fully qualified replacement name.
    since : str, default '0.4.0'
        Version in which the deprecation was introduced.
    removed_in : str, default '1.0.0'
        Version in which the alias will be removed.

    Returns
    -------
    type
        A subclass of ``new_class`` whose ``__init__`` emits
        :class:`BraintoolsDeprecationWarning` and then defers to
        ``new_class.__init__``.

    Examples
    --------
    .. code-block:: python

        >>> import warnings
        >>> from braintools._deprecation import deprecated_class
        >>> class Point:
        ...     def __init__(self, x):
        ...         self.x = x
        >>> OldPoint = deprecated_class(Point, 'mypkg.OldPoint', 'mypkg.Point')
        >>> with warnings.catch_warnings(record=True):
        ...     warnings.simplefilter('always')
        ...     p = OldPoint(3)
        >>> isinstance(p, Point), p.x
        (True, 3)
    """
    module, short = _split_name(old_name)

    def __init__(self, *args, **kwargs):
        warn_deprecated(record)
        super(alias, self).__init__(*args, **kwargs)

    try:
        setattr(__init__, '__signature__', inspect.signature(new_class.__init__))
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        pass

    namespace = {
        '__init__': __init__,
        '__doc__': new_class.__doc__,
        '__module__': module or new_class.__module__,
        '__qualname__': short,
    }

    # Before the move the old and new names were the same class, so
    # ``isinstance(obj, OldName)`` must also hold for objects built from the
    # new name. The alias's metaclass derives from the target's metaclass (so
    # classes with a custom metaclass can be aliased too) and delegates the
    # checks to the target.
    def __instancecheck__(cls, instance):
        return isinstance(instance, new_class)

    def __subclasscheck__(cls, subclass):
        return issubclass(subclass, new_class)

    target_meta: Any = type(new_class)
    metaclass: Any = type(
        f'_Deprecated{target_meta.__name__}', (target_meta,),
        {'__instancecheck__': __instancecheck__, '__subclasscheck__': __subclasscheck__},
    )
    alias = metaclass(short, (new_class,), namespace)
    record = register_deprecation(
        old_name, new_name, since=since, removed_in=removed_in,
        kind='class', alias=alias, target=new_class,
    )
    setattr(alias, '__deprecated__', record.message)
    return alias


def registered_deprecations() -> Tuple[DeprecationRecord, ...]:
    """Return all registered deprecations.

    Returns
    -------
    tuple of DeprecationRecord
        The records registered so far, in registration order.
    """
    return tuple(_REGISTRY)
