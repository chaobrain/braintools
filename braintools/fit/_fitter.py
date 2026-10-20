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

"""Model-agnostic parameter fitting with three optimiser backends.

:class:`Fitter` fits either a pure function ``simulate(params) -> prediction``
(functional mode) or the trainable :class:`brainstate.nn.Param` parameters of
a module (module mode, :meth:`Fitter.from_module`) to data, by minimising an
objective ``objective(prediction, target)`` with

- ``'grad'``: :func:`braintools.optim.multistart_minimize`, all starts
  ``vmap``-ped in one compiled program;
- ``'scipy'``: :func:`scipy.optimize.minimize` through the JAX wrapper of
  :class:`braintools.optim.ScipyOptimizer`, once per start;
- ``'nevergrad'``: :class:`braintools.optim.NevergradOptimizer`, every
  generation evaluated as one ``vmap``-ped batch.
"""

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import optax

from braintools.optim._multistart import (
    _is_pair, _make_bijector, _pull_inside, multistart_minimize,
)
from braintools.optim._optax_optimizer import OptaxOptimizer

__all__ = [
    'Fitter',
    'FitResult',
]

_BACKENDS = ('grad', 'scipy', 'nevergrad')
# Constrained values are pulled this fraction of the box width inside the box
# before a bounded inverse transform: _EDGE only keeps candidate values finite,
# _START_EDGE also keeps the transform's slope large enough for a start on a
# bound to move (cf. braintools.optim._multistart._EDGE).
_EDGE = 1e-6
_START_EDGE = 1e-3


@dataclasses.dataclass
class FitResult:
    """Outcome of :meth:`Fitter.fit`.

    Attributes
    ----------
    best_params : dict
        ``{name: value}`` at the best start, in the constrained (physical)
        space, with units restored.
    best_loss : float
        Loss of ``best_params``.
    params : dict
        ``{name: values}`` of every start (leading axis ``S``), constrained,
        with units. For ``'nevergrad'`` these are the candidates of the last
        generation.
    losses : jax.Array
        Final loss of every start, shape ``(S,)``.
    history : jax.Array
        Losses per optimisation step, shape ``(S, n_steps)``. For ``'grad'``
        the loss after every iteration (entries after a start stopped repeat
        its final loss); for ``'scipy'`` the loss after every SciPy iteration,
        padded the same way; for ``'nevergrad'`` the loss of candidate ``s``
        of every generation.
    converged : jax.Array
        Boolean ``(S,)``. ``'grad'``: the start met the convergence test;
        ``'scipy'``: ``OptimizeResult.success``; ``'nevergrad'``: the
        candidate's loss is finite.
    backend : str
        ``'grad'``, ``'scipy'`` or ``'nevergrad'``.
    raw : Any
        Backend output: the :class:`~braintools.optim.MultiStartResult`, the
        list of SciPy ``OptimizeResult`` objects, or the
        :class:`~braintools.optim.NevergradOptimizer`.
    """
    __module__ = 'braintools.fit'

    best_params: Dict[str, Any]
    best_loss: float
    params: Dict[str, Any]
    losses: Any
    history: Any
    converged: Any
    backend: str
    raw: Any = None

    def __repr__(self) -> str:
        names = ', '.join(self.best_params)
        return (
            f'FitResult(backend={self.backend!r}, best_loss={self.best_loss:.6g}, '
            f'n_starts={int(np.shape(self.losses)[0])}, params=[{names}])'
        )


# ---------------------------------------------------------------------------
# parameter specifications
# ---------------------------------------------------------------------------

def _wrap(value: Any, unit: Optional[u.Unit]) -> Any:
    return value if unit is None else u.Quantity(value, unit)


def _in_unit(value: Any, unit: Optional[u.Unit], what: str) -> jax.Array:
    """``value`` as a plain array in ``unit`` (``None``: no unit allowed)."""
    if unit is None:
        if isinstance(value, u.Quantity) and not value.dim.is_dimensionless:
            raise TypeError(f'{what} carries the unit {value.unit}, but the parameter has no unit.')
        return jnp.asarray(u.get_mantissa(value))
    try:
        return jnp.asarray(u.Quantity(value).to_decimal(unit))
    except u.UnitMismatchError as e:
        raise TypeError(f'{what} must be expressed in a unit compatible with {unit}: {e}') from None


def _unit_of(value: Any) -> Optional[u.Unit]:
    if isinstance(value, u.Quantity) and not value.unit.is_unitless:
        return value.unit
    return None


@dataclasses.dataclass
class _Spec:
    """One fitted parameter: shape, units, bijector and search box."""
    name: str
    shape: Tuple[int, ...]
    dtype: Any
    x_unit: Optional[u.Unit]
    forward: Callable[[jax.Array], jax.Array]  # unconstrained -> constrained mantissa
    inverse: Callable[[jax.Array], jax.Array]  # constrained -> unconstrained mantissa
    box: Optional[Tuple[jax.Array, jax.Array]]  # constrained mantissas, broadcast to shape
    init: Optional[jax.Array]  # constrained mantissa
    z_unit: Optional[u.Unit] = None
    state: Any = None  # the ParamState in module mode
    param: Any = None

    def inside(self, x: jax.Array, edge: float = _EDGE) -> jax.Array:
        """Pull ``x`` ``edge`` of the box width inside the box (keeps a bounded inverse finite)."""
        if self.box is None:
            return x
        low, high = self.box
        pad = edge * (high - low)
        return jnp.clip(x, low + pad, high - pad)


def _functional_specs(space: Dict[str, Any], init: Dict[str, Any]) -> List[_Spec]:
    if not isinstance(space, dict) or not space:
        raise ValueError('`space` must be a non-empty dict {name: (low, high) or None}.')
    unknown = set(init) - set(space)
    if unknown:
        raise ValueError(f'`init` names unknown parameters {sorted(unknown)}; `space` has {list(space)}.')
    dtype = brainstate.environ.dftype()
    specs = []
    for name, bound in space.items():
        given = init.get(name)
        if bound is None:
            if given is None:
                raise ValueError(
                    f'Parameter {name!r} is unbounded (space entry None), so it needs an '
                    f'initial value: pass init={{{name!r}: value}}.'
                )
            unit = _unit_of(given)
            x_init = _in_unit(given, unit, f'init[{name!r}]').astype(dtype)
            specs.append(_Spec(
                name=name, shape=tuple(x_init.shape), dtype=dtype, x_unit=unit,
                forward=lambda z: z, inverse=lambda x: x, box=None, init=x_init,
            ))
            continue
        if not _is_pair(bound):
            raise ValueError(f'space[{name!r}] must be a (low, high) pair or None, got {bound!r}.')
        ref = bound[0] if bound[0] is not None else bound[1]
        unit = _unit_of(ref)
        t = _make_bijector(bound, unit, name)  # type: ignore[arg-type]
        low = None if bound[0] is None else _in_unit(bound[0], unit, f'space[{name!r}] low').astype(dtype)
        high = None if bound[1] is None else _in_unit(bound[1], unit, f'space[{name!r}] high').astype(dtype)
        shapes = [jnp.shape(v) for v in (low, high) if v is not None]
        x_init = None if given is None else _in_unit(given, unit, f'init[{name!r}]').astype(dtype)
        if x_init is not None:
            shapes.append(x_init.shape)
        shape = tuple(jnp.broadcast_shapes(*shapes))
        box = None
        if low is not None and high is not None:
            box = (jnp.broadcast_to(low, shape), jnp.broadcast_to(high, shape))
            if x_init is None:
                x_init = 0.5 * (box[0] + box[1])
        elif x_init is None:
            raise ValueError(
                f'Parameter {name!r} has a one-sided bound, so it needs an initial value: '
                f'pass init={{{name!r}: value}}.'
            )
        if t is None:  # pragma: no cover - a pair with two None sides
            forward, inverse = (lambda z: z), (lambda x: x)
        else:
            forward = (lambda tt: (lambda z: tt.forward(z)))(t)
            inverse = (lambda tt: (lambda x: tt.inverse(_pull_inside(x, tt))))(t)
        specs.append(_Spec(
            name=name, shape=shape, dtype=dtype, x_unit=unit, forward=forward, inverse=inverse,
            box=box, init=jnp.broadcast_to(x_init, shape),
        ))
    return specs


def _transform_box(t: Any) -> Optional[Tuple[Any, Any]]:
    """The finite constrained interval of a brainstate transform, or None."""
    lower = getattr(t, 'lower', None)
    if lower is None or isinstance(t, (brainstate.nn.SoftplusT, brainstate.nn.NegSoftplusT)):
        return None
    width = getattr(t, 'width', None)
    if width is not None:
        return lower, lower + width
    upper = getattr(t, 'upper', None)
    if upper is not None:
        return lower, upper
    return None


def _module_specs(model: Any, search_space: Dict[str, Any]) -> List[_Spec]:
    params = {}
    for path, node in model.nodes(brainstate.nn.Param).items():
        if node.fit and isinstance(node.val, brainstate.State):
            params['.'.join(str(p) for p in path)] = node
    if not params:
        raise ValueError(
            'The model has no trainable parameters: wrap at least one value in '
            'brainstate.nn.Param(..., fit=True).'
        )
    unknown = set(search_space) - set(params)
    if unknown:
        raise ValueError(
            f'search_space names unknown parameters {sorted(unknown)}; trainable '
            f'parameters are {sorted(params)}.'
        )
    specs = []
    for name, p in params.items():
        p.clear_cache()
        z_val = p.val.value
        z_unit = _unit_of(z_val)
        z_m = jnp.asarray(u.get_mantissa(z_val))
        x_val = p.t.forward(z_val)
        x_unit = _unit_of(x_val)
        x_m = _in_unit(x_val, x_unit, name)
        shape = tuple(x_m.shape)

        def forward(z, _t=p.t, _zu=z_unit, _xu=x_unit, _n=name):
            return _in_unit(_t.forward(_wrap(z, _zu)), _xu, _n)

        def inverse(x, _t=p.t, _zu=z_unit, _xu=x_unit, _n=name):
            return _in_unit(_t.inverse(_wrap(x, _xu)), _zu, _n)

        if name in search_space:
            bound = search_space[name]
            if not _is_pair(bound):
                raise ValueError(f'search_space[{name!r}] must be a (low, high) pair, got {bound!r}.')
        else:
            bound = _transform_box(p.t)
        box = None
        if bound is not None:
            low = _in_unit(bound[0], x_unit, f'Lower bound of {name!r}').astype(x_m.dtype)
            high = _in_unit(bound[1], x_unit, f'Upper bound of {name!r}').astype(x_m.dtype)
            box = (jnp.broadcast_to(low, shape), jnp.broadcast_to(high, shape))
        specs.append(_Spec(
            name=name, shape=shape, dtype=z_m.dtype, x_unit=x_unit, forward=forward,
            inverse=inverse, box=box, init=x_m, z_unit=z_unit, state=p.val, param=p,
        ))
    return specs


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

class Fitter:
    r"""Fit model parameters to data by minimising an objective.

    Functional mode (this constructor) fits a pure function; module mode
    (:meth:`from_module`) fits the trainable parameters of a
    :class:`brainstate.nn.Module`.

    Parameters
    ----------
    simulate : callable
        ``simulate(params) -> prediction``, a pure JAX function of a dict
        ``{name: value}`` (values carry the units of ``space``). It must be
        traceable by ``jax.jit``/``jax.vmap``/``jax.grad`` (``'grad'`` needs
        gradients; the other backends only need ``vmap``).
    objective : callable
        ``objective(prediction, target) -> scalar``, e.g. an
        :class:`~braintools.fit.Objective` or a loss from
        :mod:`braintools.fit.losses`.
    space : dict
        ``{name: (low, high)}`` box bounds (scalars, arrays or Quantities;
        ``low`` or ``high`` may be ``None`` for a one-sided bound) or
        ``{name: None}`` for an unbounded parameter whose value comes from
        ``init``. The shape of a parameter is the broadcast shape of its
        bounds and initial value.
    init : dict, optional
        Initial values ``{name: value}``, used for unbounded parameters
        (required there), for one-sided bounds (required there), and as the
        single start when ``n_starts=1`` and no ``key`` is given (box
        parameters default to the midpoint of their box).
    optimizer : str, dict, optax.GradientTransformation or OptaxOptimizer, optional
        Backend options. A string names the method (``'lbfgs'``/``'adam'``/
        ``'gd'`` for ``'grad'``, a :func:`scipy.optimize.minimize` method
        for ``'scipy'``, a Nevergrad optimiser such as ``'DE'`` or ``'CMA'``
        for ``'nevergrad'``). A dict holds ``'method'`` plus options:
        ``'tol'`` and ``'learning_rate'`` for ``'grad'``; ``'tol'`` plus
        :func:`scipy.optimize.minimize` options for ``'scipy'``;
        ``'n_sample'`` (population per generation, default ``n_starts``)
        plus :class:`~braintools.optim.NevergradOptimizer` keyword arguments
        for ``'nevergrad'``. For ``'grad'`` an ``optax`` transformation or a
        :class:`braintools.optim` optax optimiser (its ``tx``) is used as the
        update rule. Defaults: L-BFGS, ``'L-BFGS-B'``, ``'DE'``.
    backend : {'grad', 'scipy', 'nevergrad'}, default 'grad'
        Optimiser backend.
    transient : int, optional
        Number of leading samples (axis 0 of every prediction leaf) dropped
        before the objective is evaluated. :meth:`fit` may override it.

    Raises
    ------
    ValueError
        If ``backend`` is unknown, or ``space``/``init`` are inconsistent.
    TypeError
        If ``simulate`` or ``objective`` is not callable, or ``optimizer``
        has the wrong type for the backend.

    See Also
    --------
    FitResult : The result of :meth:`fit`.
    braintools.optim.multistart_minimize : The ``'grad'`` backend.
    sweep : Scan parameters on a grid instead.
    hessian_ci : Confidence intervals at the optimum.

    Notes
    -----
    **Parameter spaces.** The ``'grad'`` and ``'scipy'`` backends optimise in
    an unconstrained space: a bounded parameter is mapped through
    :class:`brainstate.nn.SigmoidT` (``SoftplusT`` for one-sided bounds), a
    module parameter through its own transform. ``'nevergrad'`` searches the
    constrained box directly, so every parameter needs finite bounds.

    **Starts.** With ``n_starts=1`` and no ``key``, the single start is the
    initial value (``init``, the box midpoint, or the module's current
    value). Otherwise ``key`` is required and every bounded parameter is
    drawn uniformly from its box (unbounded ones use their initial value).
    ``x0`` in :meth:`fit` gives the starts explicitly. For ``'grad'`` and
    ``'scipy'``, a start within ``1e-3`` of the box width of a bound is
    moved that far inside, so a start on a bound (including a module
    parameter sitting at its transform's bound) maps to a finite
    unconstrained value where the transform is not flat. The ``'grad'``
    convergence test then acts on the unconstrained gradient (see
    :func:`~braintools.optim.multistart_minimize` for the test itself).

    **Randomness.** All random draws derive from ``key``; ``'nevergrad'`` is
    seeded from it and does not touch NumPy's global random generator.

    **Purity.** Functional mode is pure. In module mode, the model is not
    touched while fitting: the parameter values are fed into a
    functionalised copy of ``predict`` (see :meth:`from_module`), and the
    best values are written into the parameters only after :meth:`fit`
    finishes.

    **Transformations.** :meth:`fit` drives the search from the host (it
    compiles its own programs: one ``jit``-ed, ``vmap``-ped multi-start
    loop for ``'grad'``, a ``jit``-ed loss and gradient for ``'scipy'``, a
    ``jit``-ed, ``vmap``-ped batch loss for ``'nevergrad'``), so it cannot
    itself be called inside ``jax.jit`` or ``jax.vmap``.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> from braintools.fit import Fitter
        >>> t = jnp.linspace(0., 1., 50)
        >>> simulate = lambda p: p['a'] * jnp.exp(-t / p['tau'])
        >>> target = simulate({'a': 2.0, 'tau': 0.3})
        >>> mse = lambda pred, y: jnp.mean((pred - y) ** 2)
        >>> fitter = Fitter(simulate, mse, {'a': (0., 5.), 'tau': (0.05, 1.)})
        >>> res = fitter.fit(target, n_starts=4, key=jax.random.PRNGKey(0))
        >>> print(round(float(res.best_params['a']), 3), round(float(res.best_params['tau']), 3))
        2.0 0.3
    """
    __module__ = 'braintools.fit'

    def __init__(
        self,
        simulate: Callable[[Dict[str, Any]], Any],
        objective: Callable[[Any, Any], Any],
        space: Dict[str, Any],
        *,
        init: Optional[Dict[str, Any]] = None,
        optimizer: Any = None,
        backend: str = 'grad',
        transient: Optional[int] = None,
    ):
        if not callable(simulate):
            raise TypeError(f'`simulate` must be callable, got {type(simulate).__name__}.')
        self._setup(objective, optimizer, backend, transient)
        self.simulate: Optional[Callable[[Dict[str, Any]], Any]] = simulate
        self.space: Optional[Dict[str, Any]] = dict(space) if isinstance(space, dict) else space
        self.init = dict(init or {})
        self.model: Any = None
        self.predict: Optional[Callable[[Any], Any]] = None
        self.search_space: Dict[str, Any] = {}
        self._specs = _functional_specs(self.space, self.init)

    def _setup(self, objective: Any, optimizer: Any, backend: str, transient: Optional[int]) -> None:
        if not callable(objective):
            raise TypeError(f'`objective` must be callable, got {type(objective).__name__}.')
        if backend not in _BACKENDS:
            raise ValueError(f'`backend` must be one of {_BACKENDS}, got {backend!r}.')
        self.objective = objective
        self.backend = backend
        self.optimizer = optimizer
        self.transient = _check_transient(transient)
        self._options()  # validate early

    @classmethod
    def from_module(
        cls,
        model: Any,
        predict: Callable[[Any], Any],
        objective: Callable[[Any, Any], Any],
        *,
        optimizer: Any = None,
        backend: str = 'grad',
        transient: Optional[int] = None,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> 'Fitter':
        r"""Fit the trainable :class:`brainstate.nn.Param` parameters of a module.

        Parameters
        ----------
        model : brainstate.nn.Module
            The model. Every :class:`brainstate.nn.Param` with ``fit=True``
            is fitted, named by its dotted attribute path (e.g.
            ``'w_ee'``, ``'layer.tau'``).
        predict : callable
            ``predict(model) -> prediction``; runs the model and reads its
            output. It is traced once per :meth:`fit` call.
        objective : callable
            ``objective(prediction, target) -> scalar``.
        optimizer : optional
            As for the :class:`Fitter` constructor.
        backend : {'grad', 'scipy', 'nevergrad'}, default 'grad'
            Optimiser backend.
        transient : int, optional
            Leading prediction samples dropped before the objective.
        search_space : dict, optional
            ``{name: (low, high)}`` constrained bounds overriding the box
            implied by a parameter's transform (``SigmoidT``, ``TanhT``,
            ``SoftsignT``, ``ScaledSigmoidT``, ``ClipT``). Parameters with
            a box have their starts drawn from it; ``'nevergrad'`` needs a
            box for every parameter.

        Returns
        -------
        Fitter
            A fitter in module mode.

        Raises
        ------
        ValueError
            If the model has no ``fit=True`` parameter, or ``search_space``
            names an unknown parameter.

        Notes
        -----
        ``predict(model)`` (plus ``model.reg_loss()`` when the model defines
        it, as brainmass does) is turned into a pure function of the
        parameter values with :class:`brainstate.transform.StatefulFunction`:
        the unconstrained values are substituted for the parameters'
        ``ParamState`` values in the traced program, so starts and
        candidates are ``vmap``-ped and the model's states are never
        written during the search. The parameter's own transform maps the
        unconstrained value to the constrained one. After :meth:`fit`
        returns, the model holds the best parameters.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> import jax
            >>> import jax.numpy as jnp
            >>> from braintools.fit import Fitter
            >>> class Decay(brainstate.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.tau = brainstate.nn.Param(0.5, t=brainstate.nn.SigmoidT(0.05, 2.))
            ...     def __call__(self, t):
            ...         return jnp.exp(-t / self.tau.value())
            >>> t = jnp.linspace(0., 1., 40)
            >>> model = Decay()
            >>> mse = lambda p, y: jnp.mean((p - y) ** 2)
            >>> fitter = Fitter.from_module(model, lambda m: m(t), mse)
            >>> res = fitter.fit(jnp.exp(-t / 0.2))
            >>> print(round(float(res.best_params['tau']), 3), round(float(model.tau.value()), 3))
            0.2 0.2
        """
        self = cls.__new__(cls)
        if not callable(predict):
            raise TypeError(f'`predict` must be callable, got {type(predict).__name__}.')
        self._setup(objective, optimizer, backend, transient)
        self.simulate = None
        self.space = None
        self.init = {}
        self.model = model
        self.predict = predict
        self.search_space = dict(search_space or {})
        self._specs = _module_specs(model, self.search_space)
        return self

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @property
    def names(self) -> Tuple[str, ...]:
        """Names of the fitted parameters."""
        return tuple(s.name for s in self._specs)

    def _options(self) -> Dict[str, Any]:
        opt = self.optimizer
        if opt is None:
            out: Dict[str, Any] = {}
        elif isinstance(opt, str):
            out = {'method': opt}
        elif isinstance(opt, dict):
            out = dict(opt)
        elif self.backend == 'grad' and isinstance(opt, optax.GradientTransformation):
            out = {'method': opt}
        elif self.backend == 'grad' and isinstance(opt, OptaxOptimizer):
            out = {'method': opt.tx}
        else:
            raise TypeError(
                f'For backend={self.backend!r}, `optimizer` must be None, a method name, '
                f'an options dict' + (', an optax.GradientTransformation or a braintools.optim '
                                      'optax optimizer' if self.backend == 'grad' else '')
                + f'; got {type(opt).__name__}.'
            )
        default = {'grad': 'lbfgs', 'scipy': 'L-BFGS-B', 'nevergrad': 'DE'}[self.backend]
        out.setdefault('method', default)
        return out

    def _trim(self, prediction: Any, transient: Optional[int]) -> Any:
        if not transient:
            return prediction
        return jax.tree.map(lambda leaf: leaf[transient:], prediction, is_leaf=u.math.is_quantity)

    def _loss_functions(self, target: Any, transient: Optional[int]):
        """Return ``(loss_z, loss_x)`` over dicts of unconstrained / constrained mantissas."""
        specs = self._specs
        objective = self.objective

        def scalar(value):
            if isinstance(value, u.Quantity):
                if not value.dim.is_dimensionless:
                    raise TypeError(f'The objective must be dimensionless, got unit {value.unit}.')
                value = value.mantissa
            return jnp.asarray(value)

        if self.model is None:
            simulate = self.simulate
            assert simulate is not None

            def f_loss_x(xs):
                params = {s.name: _wrap(xs[s.name], s.x_unit) for s in specs}
                pred = self._trim(simulate(params), transient)
                return scalar(objective(pred, target))

            def f_loss_z(zs):
                return f_loss_x({s.name: s.forward(zs[s.name]) for s in specs})

            return f_loss_z, f_loss_x

        model, predict = self.model, self.predict
        assert predict is not None

        def run():
            pred = predict(model)
            reg = model.reg_loss() if hasattr(model, 'reg_loss') else 0.
            return pred, jnp.asarray(u.get_mantissa(reg))

        for s in specs:
            s.param.clear_cache()
        sf = brainstate.transform.StatefulFunction(run)
        sf.make_jaxpr()
        states = sf.get_states()
        by_state = {id(s.state): s for s in specs}
        current = [st.value for st in states]

        def loss_z(zs):
            vals = []
            for st, val in zip(states, current):
                s = by_state.get(id(st))
                if s is None:
                    vals.append(val)
                else:
                    vals.append(_wrap(jnp.asarray(zs[s.name], s.dtype), s.z_unit))
            _, (pred, reg) = sf.jaxpr_call(vals)
            return scalar(objective(self._trim(pred, transient), target)) + reg

        def loss_x(xs):
            return loss_z({s.name: s.inverse(s.inside(xs[s.name])) for s in specs})

        return loss_z, loss_x

    def _starts(self, n_starts: int, key: Any, x0: Optional[Dict[str, Any]]) -> Dict[str, jax.Array]:
        """Starting points as constrained mantissas with a leading axis ``S``."""
        specs = self._specs
        if x0 is not None:
            if not isinstance(x0, dict) or set(x0) != set(self.names):
                raise ValueError(f'`x0` must be a dict with exactly the keys {list(self.names)}.')
            out = {}
            sizes = set()
            for s in specs:
                v = _in_unit(x0[s.name], s.x_unit, f'x0[{s.name!r}]')
                if v.ndim != len(s.shape) + 1:
                    raise ValueError(
                        f'x0[{s.name!r}] must have shape (S, *{s.shape}), got {v.shape}.'
                    )
                out[s.name] = jnp.broadcast_to(v, (v.shape[0],) + s.shape).astype(s.dtype)
                sizes.add(v.shape[0])
            if len(sizes) != 1:
                raise ValueError(f'All x0 entries must have the same number of starts, got {sorted(sizes)}.')
            return out
        if n_starts == 1 and key is None:
            return {s.name: s.init[None].astype(s.dtype) for s in specs}  # type: ignore[index]
        if key is None:
            raise ValueError('A `key` is required to draw several starting points (n_starts > 1).')
        keys = jax.random.split(key, len(specs))
        out = {}
        for k, s in zip(keys, specs):
            shape = (n_starts,) + s.shape
            if s.box is None:
                out[s.name] = jnp.broadcast_to(s.init, shape).astype(s.dtype)  # type: ignore[arg-type]
            else:
                low, high = s.box
                out[s.name] = jax.random.uniform(k, shape, dtype=s.dtype, minval=low, maxval=high)
        return out

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        target: Any,
        *,
        n_starts: int = 1,
        key: Any = None,
        n_steps: int = 100,
        transient: Optional[int] = None,
        x0: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        """Run the optimisation.

        Parameters
        ----------
        target : Any
            Data compared with the prediction by the objective.
        n_starts : int, default 1
            Number of starts ``S``: parallel local searches for ``'grad'``,
            sequential SciPy runs for ``'scipy'``, the population per
            generation for ``'nevergrad'`` (unless ``'n_sample'`` is set in
            the options).
        key : jax.Array, optional
            Random key for drawing starts (required when ``n_starts > 1``
            and ``x0`` is not given, and always for ``'nevergrad'``).
        n_steps : int, default 100
            Maximum iterations per start (``'grad'``, ``'scipy'``) or number
            of generations (``'nevergrad'``).
        transient : int, optional
            Overrides the constructor's ``transient``.
        x0 : dict, optional
            Explicit starts ``{name: values}`` in the constrained space, each
            with a leading axis ``S`` (which then sets ``n_starts``). Not used
            by ``'nevergrad'``.

        Returns
        -------
        FitResult
            Best and per-start parameters and losses.

        Raises
        ------
        ValueError
            If ``n_starts`` or ``n_steps`` is not a positive integer, a key
            is needed but missing, ``x0`` is malformed, or ``'nevergrad'``
            is used with a parameter that has no finite box.
        """
        for name, v in (('n_starts', n_starts), ('n_steps', n_steps)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f'`{name}` must be a positive integer, got {v!r}.')
        transient = self.transient if transient is None else _check_transient(transient)
        loss_z, loss_x = self._loss_functions(target, transient)
        options = self._options()
        if self.backend == 'nevergrad':
            result, z_best = self._fit_nevergrad(loss_x, options, int(n_starts), key, int(n_steps))
        else:
            xs0 = self._starts(int(n_starts), key, x0)
            zs0 = {s.name: jax.vmap(lambda x, s=s: s.inverse(s.inside(x, _START_EDGE)))(xs0[s.name])
                   for s in self._specs}
            if self.backend == 'grad':
                result, z_best = self._fit_grad(loss_z, options, zs0, int(n_steps))
            else:
                result, z_best = self._fit_scipy(loss_z, options, zs0, int(n_steps))
        if self.model is not None:
            self._write_back(z_best)
        return result

    def _constrained(self, zs: Dict[str, jax.Array]) -> Dict[str, Any]:
        return {s.name: _wrap(jax.vmap(s.forward)(zs[s.name]), s.x_unit) for s in self._specs}

    def _write_back(self, z_best: Dict[str, jax.Array]) -> None:
        for s in self._specs:
            s.state.value = _wrap(jnp.asarray(z_best[s.name], s.dtype), s.z_unit)
            s.param.clear_cache()

    def _fit_grad(self, loss_z, options, zs0, n_steps):
        options = dict(options)
        method = options.pop('method')
        tol = options.pop('tol', 1e-6)
        lr = options.pop('learning_rate', 1e-2)
        if options:
            raise ValueError(f"Unknown options for backend 'grad': {sorted(options)}.")
        res = multistart_minimize(loss_z, zs0, method=method, max_iter=n_steps, tol=tol, learning_rate=lr)
        params = self._constrained(res.x)
        best = int(res.best_index)
        result = FitResult(
            best_params=_take(params, best),
            best_loss=float(res.best_fun),
            params=params,
            losses=res.fun,
            history=res.history,
            converged=res.converged,
            backend='grad',
            raw=res,
        )
        return result, {k: v[best] for k, v in res.x.items()}

    def _fit_scipy(self, loss_z, options, zs0, n_steps):
        from braintools.optim._scipy_optimizer import scipy_minimize_with_jax

        options = dict(options)
        method = options.pop('method')
        tol = options.pop('tol', None)
        options.setdefault('maxiter', n_steps)
        fun = jax.jit(loss_z)
        jac = jax.jit(jax.grad(loss_z))
        n = next(iter(zs0.values())).shape[0]
        raws, finals, hists = [], [], []
        for i in range(n):
            trace: List[float] = []

            def record(z, *_, _trace=trace):
                _trace.append(float(fun(z)))

            z0 = {k: v[i] for k, v in zs0.items()}
            res = scipy_minimize_with_jax(
                z0, fun, jac=jac, method=method, tol=tol, callback=record, options=options,
            )
            raws.append(res)
            final = float(res.fun)
            finals.append(final)
            trace = trace[:n_steps]
            hists.append(trace + [final] * (n_steps - len(trace)))
        zs = {k: jnp.stack([jnp.asarray(r.x[k]) for r in raws]).astype(zs0[k].dtype) for k in zs0}
        losses = jnp.asarray(finals)
        best = int(jnp.argmin(jnp.where(jnp.isfinite(losses), losses, jnp.inf)))
        params = self._constrained(zs)
        result = FitResult(
            best_params=_take(params, best),
            best_loss=finals[best],
            params=params,
            losses=losses,
            history=jnp.asarray(hists),
            converged=jnp.asarray([bool(r.success) for r in raws]),
            backend='scipy',
            raw=raws,
        )
        return result, {k: v[best] for k, v in zs.items()}

    def _fit_nevergrad(self, loss_x, options, n_starts, key, n_steps):
        from braintools.optim._nevergrad_optimizer import NevergradOptimizer

        missing = [s.name for s in self._specs if s.box is None]
        if missing:
            raise ValueError(
                f"backend='nevergrad' searches a finite box, but {missing} have no finite bounds; "
                f"give (low, high) bounds" + (' via search_space.' if self.model is not None else '.')
            )
        if key is None:
            raise ValueError("backend='nevergrad' needs a `key` to seed its random search.")
        options = dict(options)
        method = options.pop('method')
        n_sample = int(options.pop('n_sample', n_starts))
        seed = int(jax.random.randint(key, (), 0, np.iinfo(np.int32).max))
        bounds = {}
        for s in self._specs:
            low, high = s.box  # type: ignore[misc]
            if s.shape == ():
                bounds[s.name] = (float(low), float(high))
            else:
                bounds[s.name] = (np.asarray(low), np.asarray(high))
        names = list(self.names)
        batched_loss = jax.jit(jax.vmap(loss_x))

        def batched(**candidates):
            return batched_loss({k: jnp.asarray(candidates[k]) for k in names})

        # Nevergrad builds a default random state for every parameter it creates
        # by drawing a seed from NumPy's global generator; those states are then
        # replaced by the one seeded from ``key``. Save and restore the global
        # generator so the fit neither depends on nor advances it.
        global_state = np.random.get_state()
        try:
            opt = NevergradOptimizer(
                batched, bounds, n_sample=n_sample, method=method,
                budget=n_sample * n_steps, seed=seed, **options,
            )
            opt.parametrization.random_state = np.random.RandomState(seed)
            opt.minimize(n_iter=n_steps, verbose=False)
        finally:
            np.random.set_state(global_state)
        errors = np.asarray(opt.errors, dtype=float)
        finite = np.where(np.isfinite(errors), errors, np.inf)
        best = int(np.argmin(finite))
        last = opt.candidates[-n_sample:]
        params = {
            s.name: _wrap(jnp.asarray(np.stack([np.asarray(c[s.name]) for c in last])).astype(s.dtype), s.x_unit)
            for s in self._specs
        }
        best_x = {s.name: jnp.asarray(opt.candidates[best][s.name], dtype=s.dtype) for s in self._specs}
        losses = jnp.asarray(errors[-n_sample:])
        result = FitResult(
            best_params={s.name: _wrap(best_x[s.name], s.x_unit) for s in self._specs},
            best_loss=float(errors[best]),
            params=params,
            losses=losses,
            history=jnp.asarray(errors.reshape(n_steps, n_sample).T),
            converged=jnp.isfinite(losses),
            backend='nevergrad',
            raw=opt,
        )
        z_best = {s.name: s.inverse(s.inside(best_x[s.name])) for s in self._specs}
        return result, z_best


def _take(params: Dict[str, Any], i: int) -> Dict[str, Any]:
    return {k: v[i] for k, v in params.items()}


def _check_transient(transient: Any) -> Optional[int]:
    if transient is None:
        return None
    if isinstance(transient, bool) or not isinstance(transient, (int, np.integer)) or transient < 0:
        raise ValueError(f'`transient` must be a non-negative integer or None, got {transient!r}.')
    return int(transient)

