# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


from typing import Callable, Optional, Union, Sequence, Dict, List

import brainunit as u
import jax
import numpy as np
from brainstate._compatible_import import safe_zip

from ._base import Optimizer

try:
    import nevergrad as ng
except (ImportError, ModuleNotFoundError):
    ng = None

__all__ = [
    'NevergradOptimizer',
]


def concat_parameters(*parameters):
    """
    Stack candidate parameters across the first dimension per-leaf.

    Parameters
    ----------
    parameters : sequence of PyTree
        Candidate parameter values returned by Nevergrad (e.g., tuples or
        dicts of scalars/arrays). All items must share the same PyTree
        structure and compatible leaf shapes.

    Returns
    -------
    PyTree
        A PyTree with the same structure where each leaf is a JAX array of
        shape ``(n_candidates, ...)``, created by stacking corresponding
        leaves from ``parameters``.

    Notes
    -----
    This utility prepares a batched set of parameters to pass to a batched
    loss function. Broadcasting and dtype follow JAX semantics.
    """
    final_parameters = jax.tree.map(lambda *ps: jax.numpy.asarray(ps), *parameters)
    return final_parameters


class NevergradOptimizer(Optimizer):
    """
    Ask/tell optimizer wrapper around Nevergrad with batched evaluation support.

    This optimizer draws ``n_sample`` candidate parameter sets per iteration
    (via ``ask``), evaluates them in batch using a user-provided loss function,
    and reports the losses back to Nevergrad (via ``tell``). It then returns
    the current best parameters according to either the lowest observed loss
    or Nevergrad's recommendation.

    Parameters
    ----------
    batched_loss_fun : callable
        Callable evaluating a batch of candidate parameters and returning one
        scalar loss per candidate. Its signature depends on ``bounds``:
          - If ``bounds`` is a sequence/tuple, the callable is invoked as
            ``batched_loss_fun(*params)`` where each element of ``params`` is a
            JAX array stacked over the candidate dimension, e.g., shape
            ``(n_sample, ...)`` per argument.
          - If ``bounds`` is a dict, the callable is invoked as
            ``batched_loss_fun(**params)`` where each value is a stacked JAX
            array of shape ``(n_sample, ...)``.

        The return value must be a 1D array-like of length ``n_sample`` with
        the loss per candidate.
    bounds : dict or sequence of tuple
        Search space bounds. Each bound is a pair ``(min, max)``. Values can be
        scalars or arrays (broadcasting not applied), optionally wrapped as
        ``brainunit.Quantity`` to specify units. All leaves within a pair must
        have identical shapes. Two forms are supported:
          - dict: ``{"name": (min, max), ...}`` producing named parameters;
          - sequence/tuple: ``[(min, max), ...]`` producing positional
            parameters passed to ``batched_loss_fun`` in the given order.
    n_sample : int
        Number of candidates to evaluate per iteration.
    method : str, default: 'DE'
        Nevergrad optimizer name, e.g. ``'DE'``, ``'TwoPointsDE'``, ``'CMA'``,
        ``'PSO'``, ``'OnePlusOne'``, or any valid key from
        ``nevergrad.optimizers.registry``.
    use_nevergrad_recommendation : bool, default: False
        If True, return Nevergrad's recommendation (based on its internal
        sampling history) instead of the parameters with the lowest observed
        loss so far. For very close losses under noise, recommendations can
        sometimes be preferable.
    budget : int or None, default: None
        Maximum number of evaluations given to Nevergrad. ``None`` lets the
        optimizer run without an explicit budget limit.
    num_workers : int, default: 1
        Degree of parallelism hinted to Nevergrad.
    method_params : dict, optional
        Extra keyword arguments forwarded to the Nevergrad optimizer
        constructor.

    Attributes
    ----------
    candidates : list
        History of all parameter sets evaluated (one entry per candidate).
    errors : numpy.ndarray
        Aggregated losses corresponding to ``candidates``.

    Examples
    --------
    Optimize two scalars with tuple bounds and a simple quadratic loss:

    >>> import jax.numpy as jnp
    >>> def batched_loss_fun(x, y):
    ...     # x, y have shape (n_sample,)
    ...     return (x**2 + y**2)
    >>> bounds = [(-5.0, 5.0), (-3.0, 3.0)]
    >>> opt = NevergradOptimizer(batched_loss_fun, bounds, n_sample=8, method='OnePlusOne')
    >>> best = opt.minimize(n_iter=5, verbose=False)
    >>> len(best) == 2
    True

    Optimize named parameters using dict bounds:

    >>> def batched_loss_fun(**p):
    ...     # p['a'], p['b'] have shape (n_sample,)
    ...     return p['a']**2 + (p['b']-1.0)**2
    >>> bounds = {"a": (-5.0, 5.0), "b": (-3.0, 3.0)}
    >>> opt = NevergradOptimizer(batched_loss_fun, bounds, n_sample=8, method='DE')
    >>> best = opt.minimize(n_iter=3, verbose=False)
    >>> set(best.keys()) == {"a", "b"}
    True
    """
    __module__ = 'braintools.optim'

    candidates: List
    errors: np.ndarray

    def __init__(
        self,
        batched_loss_fun: Callable,
        bounds: Optional[Union[Sequence, Dict]],
        n_sample: int,
        method: str = 'DE',
        use_nevergrad_recommendation: bool = False,
        budget: Optional[int] = None,
        num_workers: int = 1,
        method_params: Optional[Dict] = None,
    ):
        if ng is None:
            raise ImportError("Nevergrad is not installed. Please install it using 'pip install nevergrad'.")

        # loss function to evaluate
        assert callable(batched_loss_fun), "'batched_loss_fun' must be a callable function."
        self.vmap_loss_fun = batched_loss_fun

        # population size
        assert n_sample > 0, "'n_sample' must be a positive integer."
        self.n_sample = n_sample

        # optimization method
        self.method = method
        self.optimizer: ng.optimizers.base.ConfiguredOptimizer | ng.optimizers.base.Optimizer

        # bounds
        if bounds is None:
            raise ValueError("'bounds' must be provided as a dict or a sequence of (min, max) pairs.")
        bounds = bounds
        self.bounds = bounds
        if isinstance(self.bounds, dict):
            bound_units = dict()
            parameters = dict()
            for key, bound in self.bounds.items():
                assert len(bound) == 2, f'Each bound must be a tuple of two elements (min, max), got {bound}.'
                bound = (u.Quantity(bound[0]), u.Quantity(bound[1]))
                u.fail_for_unit_mismatch(bound[0], bound[1])
                bound = (bound[0], bound[1].in_unit(bound[0].unit))
                bound_units[key] = bound[0].unit
                # treat only true 0-d arrays as scalars; (1,) remains an Array
                if np.ndim(np.asarray(bound[0].mantissa)) == 0 and np.ndim(np.asarray(bound[1].mantissa)) == 0:
                    parameters[key] = ng.p.Scalar(
                        lower=float(np.asarray(bound[0].mantissa)),
                        upper=float(np.asarray(bound[1].mantissa))
                    )
                else:
                    assert bound[0].shape == bound[1].shape, (f"Shape of the bounds must be the same, "
                                                              f"got {bound[0].shape} and {bound[1].shape}.")
                    parameters[key] = ng.p.Array(
                        shape=bound[0].shape,
                        lower=np.asarray(bound[0].mantissa),
                        upper=np.asarray(bound[1].mantissa)
                    )
            parametrization = ng.p.Dict(**parameters)
        elif isinstance(self.bounds, (list, tuple)):
            parameters = list()
            bound_units = list()
            for i, bound in enumerate(self.bounds):
                assert len(bound) == 2, f'Each bound must be a tuple of two elements (min, max), got {bound}.'
                bound = (u.Quantity(bound[0]), u.Quantity(bound[1]))
                u.fail_for_unit_mismatch(bound[0], bound[1])
                bound = (bound[0], bound[1].in_unit(bound[0].unit))
                bound_units.append(bound[0].unit)
                # treat only true 0-d arrays as scalars; (1,) remains an Array
                if np.ndim(np.asarray(bound[0].mantissa)) == 0 and np.ndim(np.asarray(bound[1].mantissa)) == 0:
                    parameters.append(
                        ng.p.Scalar(lower=float(np.asarray(bound[0].mantissa)),
                                    upper=float(np.asarray(bound[1].mantissa)))
                    )
                else:
                    assert bound[0].shape == bound[1].shape, (f"Shape of the bounds must be the same, "
                                                              f"got {bound[0].shape} and {bound[1].shape}.")
                    parameters.append(
                        ng.p.Array(shape=bound[0].shape,
                                   lower=np.asarray(bound[0].mantissa),
                                   upper=np.asarray(bound[1].mantissa))
                    )
            parametrization = ng.p.Tuple(*parameters)
        else:
            raise ValueError(f"Unknown type of 'bounds': {type(self.bounds)}")
        self.parametrization = parametrization
        self._bound_units = bound_units

        # others
        self.budget = budget
        self.num_workers = num_workers
        self.use_nevergrad_recommendation = use_nevergrad_recommendation
        self.method_params = method_params if method_params is not None else dict()

    def initialize(self):
        # initialize optimizer
        parameters = dict(
            budget=self.budget,
            num_workers=self.num_workers,
            parametrization=self.parametrization,
            **self.method_params
        )
        if self.method == 'DE':
            self.optimizer = ng.optimizers.DE(**parameters)
        elif self.method == 'TwoPointsDE':
            self.optimizer = ng.optimizers.TwoPointsDE(**parameters)
        elif self.method == 'CMA':
            self.optimizer = ng.optimizers.CMA(**parameters)
        elif self.method == 'PSO':
            self.optimizer = ng.optimizers.PSO(**parameters)
        elif self.method == 'OnePlusOne':
            self.optimizer = ng.optimizers.OnePlusOne(**parameters)
        else:
            self.optimizer = ng.optimizers.registry[self.method](**parameters)
        # Some optimizers expose internal population size as a private attribute.
        # Avoid relying on private API unless it exists and is writable.
        for attr in ("_llambda", "llambda"):
            try:
                if hasattr(self.optimizer, attr):
                    setattr(self.optimizer, attr, self.n_sample)
                    break
            except Exception:
                pass

        # initialize the candidates and errors
        self.candidates = []
        self.errors: np.ndarray = None

    def _add_unit(self, parameters):
        if isinstance(self.parametrization, ng.p.Tuple):
            parameters = [(param if unit.dim.is_dimensionless else u.Quantity(param, unit))
                          for unit, param in zip(self._bound_units, parameters)]
        elif isinstance(self.parametrization, ng.p.Dict):
            parameters = {
                key: (
                    param if self._bound_units[key].dim.is_dimensionless else u.Quantity(param, self._bound_units[key]))
                for key, param in parameters.items()
            }
        else:
            raise ValueError(f"Unknown type of 'parametrization': {type(self.parametrization)}")
        return parameters

    def _one_trial(self, choice_best: bool = False):
        # draw parameters
        candidates = [self.optimizer.ask() for _ in range(self.n_sample)]
        parameters = [c.value for c in candidates]
        mapped_parameters = concat_parameters(*parameters)

        # evaluate parameters
        if isinstance(self.parametrization, ng.p.Tuple):
            mapped_parameters = self._add_unit(mapped_parameters)
            errors = self.vmap_loss_fun(*mapped_parameters)
        elif isinstance(self.parametrization, ng.p.Dict):
            mapped_parameters = self._add_unit(mapped_parameters)
            errors = self.vmap_loss_fun(**mapped_parameters)
        else:
            raise ValueError(f"Unknown type of 'parametrization': {type(self.parametrization)}")
        errors = np.asarray(errors)

        # tell the optimizer
        assert len(candidates) == len(errors), "Number of parameters and errors must be the same"
        for candidate, error in safe_zip(candidates, errors):
            self.optimizer.tell(candidate, error)

        # record the tested parameters and errors
        self.candidates.extend(parameters)
        self.errors = errors if self.errors is None else np.concatenate([self.errors, errors])

        # return the best parameter
        if choice_best:
            if self.use_nevergrad_recommendation:
                res = self.optimizer.provide_recommendation()
                # use value which matches the parametrization structure
                return self._add_unit(res.value)
            else:
                best = np.nanargmin(self.errors)
                return self._add_unit(self.candidates[best])

    def minimize(self, n_iter: int = 1, verbose: bool = True):
        # check the number of iterations
        assert isinstance(n_iter, int), "'n_iter' must be an integer."
        assert n_iter > 0, "'n_iter' must be a positive integer."

        # initialize the optimizer
        self.initialize()

        # run the optimization
        best_result = None
        for i in range(n_iter):
            r = self._one_trial(choice_best=True)
            best_result = r
            if verbose:
                print(f'Iteration {i}, best error: {np.nanmin(self.errors):.5f}, best parameters: {r}')
        return best_result
