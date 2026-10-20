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

"""Composable fitting objectives: weighted loss terms.

A :class:`Term` wraps one loss ``fn(prediction, target) -> scalar`` with a
weight and an optional key into dict-valued predictions; terms add up into an
:class:`Objective`::

    objective = 0.5 * Term(rmse) + Term(fc_correlation_loss, key='bold')
"""

import numbers
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple, Union

import brainunit as u
import jax

__all__ = [
    'Term',
    'Objective',
]


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Number):
        return True
    if isinstance(value, (jax.Array, u.Quantity)) or hasattr(value, 'ndim'):
        return u.math.ndim(value) == 0
    return False


def _select(value: Any, key: Optional[Hashable]) -> Any:
    return value if key is None else value[key]


class Term:
    """A weighted loss term of an :class:`Objective`.

    Parameters
    ----------
    fn : callable
        ``fn(prediction, target) -> scalar``, the loss.
    weight : float or scalar array, default 1.0
        Multiplier of the loss.
    key : hashable, optional
        When given, the term evaluates ``fn(prediction[key], target[key])``,
        e.g. to pick one output of a dict-valued prediction.
    name : str, optional
        Label used by :meth:`Objective.breakdown`. Defaults to ``key`` when
        given, otherwise to the name of ``fn``.

    Raises
    ------
    TypeError
        If ``fn`` is not callable or ``weight`` is not a scalar.

    See Also
    --------
    Objective : The sum of several terms.
    Fitter : Fits a model to data by minimising an objective.

    Notes
    -----
    A term is a plain callable: it holds no JAX state, so it can be used
    inside ``jax.jit``, ``jax.vmap`` and ``jax.grad`` as long as ``fn`` can.
    ``2.0 * term`` returns a new term with the weight doubled; ``term + other``
    returns an :class:`Objective`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import Term
        >>> mse = lambda p, t: jnp.mean((p - t) ** 2)
        >>> term = 2.0 * Term(mse, name='mse')
        >>> term
        Term(name='mse', weight=2.0, key=None)
        >>> print(term(jnp.array([1., 2.]), jnp.array([1., 4.])))
        4.0
    """
    __module__ = 'braintools.fit'

    def __init__(
        self,
        fn: Callable[[Any, Any], Any],
        weight: Any = 1.0,
        key: Optional[Hashable] = None,
        name: Optional[str] = None,
    ):
        if not callable(fn):
            raise TypeError(f'`fn` must be callable, got {type(fn).__name__}.')
        if not _is_scalar(weight):
            raise TypeError(f'`weight` must be a scalar, got {weight!r}.')
        self.fn = fn
        self.weight = weight
        self.key = key
        if name is None:
            name = str(key) if key is not None else getattr(fn, '__name__', type(fn).__name__)
        self.name = str(name)

    def __call__(self, prediction: Any, target: Any) -> Any:
        """Return ``weight * fn(prediction[key], target[key])``."""
        return self.weight * self.fn(_select(prediction, self.key), _select(target, self.key))

    def __repr__(self) -> str:
        return f'Term(name={self.name!r}, weight={self.weight!r}, key={self.key!r})'

    def _scaled(self, factor: Any) -> 'Term':
        return Term(self.fn, weight=self.weight * factor, key=self.key, name=self.name)

    def __mul__(self, other: Any) -> 'Term':
        if not _is_scalar(other):
            return NotImplemented
        return self._scaled(other)

    __rmul__ = __mul__

    def __add__(self, other: Any) -> 'Objective':
        if isinstance(other, (Term, Objective)):
            return Objective(self, other)
        return NotImplemented

    def __radd__(self, other: Any) -> 'Objective':
        # ``sum([t1, t2])`` starts from the integer 0
        if isinstance(other, numbers.Number) and not isinstance(other, bool) and other == 0:
            return Objective(self)
        return NotImplemented


class Objective:
    """A weighted sum of loss terms.

    Parameters
    ----------
    *terms : Term, Objective or callable
        The terms. Objectives are flattened into their terms; a bare callable
        ``fn(prediction, target)`` is wrapped as ``Term(fn)``.

    Raises
    ------
    ValueError
        If no term is given.
    TypeError
        If a term is neither a :class:`Term`, an :class:`Objective` nor a
        callable.

    See Also
    --------
    Term : One weighted loss.
    Fitter : Fits a model to data by minimising an objective.

    Notes
    -----
    ``objective(prediction, target)`` returns the sum of the weighted terms.
    The objective is a plain callable that works under ``jax.jit``,
    ``jax.vmap`` and ``jax.grad`` whenever its terms do. Objectives support
    ``+`` with terms and objectives and multiplication by a scalar, which
    scales every weight.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import Term
        >>> mse = lambda p, t: jnp.mean((p - t) ** 2)
        >>> mae = lambda p, t: jnp.mean(jnp.abs(p - t))
        >>> obj = 0.5 * Term(mse, key='v', name='mse_v') + Term(mae, key='r')
        >>> pred = {'v': jnp.array([0., 2.]), 'r': jnp.array([1., 1.])}
        >>> target = {'v': jnp.array([0., 0.]), 'r': jnp.array([0., 2.])}
        >>> print(obj(pred, target))
        2.0
        >>> {k: float(v) for k, v in obj.breakdown(pred, target).items()}
        {'mse_v': 1.0, 'r': 1.0}
    """
    __module__ = 'braintools.fit'

    def __init__(self, *terms: Union[Term, 'Objective', Callable]):
        flat: List[Term] = []
        for t in terms:
            if isinstance(t, Objective):
                flat.extend(t.terms)
            elif isinstance(t, Term):
                flat.append(t)
            elif callable(t):
                flat.append(Term(t))
            else:
                raise TypeError(f'Objective terms must be Term, Objective or callable, got {type(t).__name__}.')
        if not flat:
            raise ValueError('An Objective needs at least one term.')
        self.terms: Tuple[Term, ...] = tuple(flat)

    @property
    def names(self) -> Tuple[str, ...]:
        """Unique labels of the terms, as used by :meth:`breakdown`.

        The first term with a given name keeps it; every later term with the
        same name gets the first suffix ``_1``, ``_2``, ... that no other
        label uses, so the labels never collide.
        """
        used = set()
        first = []
        for t in self.terms:
            first.append(t.name not in used)
            used.add(t.name)
        out = []
        for t, is_first in zip(self.terms, first):
            if is_first:
                out.append(t.name)
                continue
            k = 1
            while f'{t.name}_{k}' in used:
                k += 1
            label = f'{t.name}_{k}'
            used.add(label)
            out.append(label)
        return tuple(out)

    def __call__(self, prediction: Any, target: Any) -> Any:
        """Return the sum of the weighted terms."""
        total = self.terms[0](prediction, target)
        for t in self.terms[1:]:
            total = total + t(prediction, target)
        return total

    def breakdown(self, prediction: Any, target: Any) -> Dict[str, Any]:
        """Return the weighted value of every term, for logging.

        Parameters
        ----------
        prediction, target : Any
            As for :meth:`__call__`.

        Returns
        -------
        dict
            ``{name: weight * fn(...)}`` in term order, keyed by the unique
            labels of :attr:`names` (repeated names get the first unused
            suffix ``_1``, ``_2``, ...); the values sum to the objective.
        """
        return {n: t(prediction, target) for n, t in zip(self.names, self.terms)}

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterable[Term]:
        return iter(self.terms)

    def __repr__(self) -> str:
        return 'Objective(' + ' + '.join(f'{t.weight!r}*{t.name}' for t in self.terms) + ')'

    def __add__(self, other: Any) -> 'Objective':
        if isinstance(other, (Term, Objective)):
            return Objective(self, other)
        return NotImplemented

    def __radd__(self, other: Any) -> 'Objective':
        if isinstance(other, numbers.Number) and not isinstance(other, bool) and other == 0:
            return self
        return NotImplemented

    def __mul__(self, other: Any) -> 'Objective':
        if not _is_scalar(other):
            return NotImplemented
        return Objective(*[t._scaled(other) for t in self.terms])

    __rmul__ = __mul__
