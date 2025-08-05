# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import warnings
from typing import Callable, Optional, Union, Sequence, Dict, List, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate._compatible_import import safe_zip, unzip2

try:
    from scipy.optimize import minimize
except (ImportError, ModuleNotFoundError):
    minimize = None

try:
    import nevergrad as ng
except (ImportError, ModuleNotFoundError):
    ng = None

__all__ = [
    'NevergradOptimizer',
    'ScipyOptimizer',
]


class Optimizer:
    def minimize(self, *args):
        raise NotImplementedError("minimize method is not implemented.")


def concat_parameters(*parameters):
    """
    Concatenate parameters from a list of dictionaries into a single dictionary.

    Parameters
    ----------
    parameters: list of pytree
        A list of dictionaries containing parameters.

    Returns
    -------
    dict
        A dictionary containing all the parameters.
    """
    final_parameters = jax.tree.map(lambda *ps: jax.numpy.asarray(ps), *parameters)
    return final_parameters


class NevergradOptimizer(Optimizer):
    """
    ``NevergradOptimizer`` instance creates all the tools necessary for the user
    to use it with ``Nevergrad`` library.

    Parameters
    ----------
    batched_loss_fun: callable
        The loss function to be minimized. It should be a JAX function that
        takes as input the parameters to optimize and returns the loss value.
    bounds: dict or list
        The bounds for the parameters to optimize. If a dictionary, the keys
        are the parameter names and the values are tuples of the lower and upper
        bounds. If a list, it should be a list of tuples of the lower and upper
        bounds. The order of the list must be the same as the order of the
        parameters in the loss function.
    n_sample: int
        The number of samples to evaluate at each iteration.
    method: `str`, optional
        The optimization method. By default, ``DE``: differential evolution. But
        it can be chosen from any method in Nevergrad registry.
    use_nevergrad_recommendation: bool, optional
        Whether to use Nevergrad's recommendation as the "best result". This
        recommendation takes several evaluations of the same parameters (for
        stochastic simulations) into account. The alternative is to simply
        return the parameters with the lowest error so far (the default). The
        problem with Nevergrad's recommendation is that it can give wrong result
        for errors that are very close in magnitude due.
    budget: int or None
        The number of allowed evaluations.
    num_workers: int
        The number of parallel workers.
    method_params: dict, optional
        Additional parameters for the optimization method.
    """

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
        bounds = () if bounds is None else bounds
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
                if np.size(bound[0].mantissa) == 1 and np.size(bound[1].mantissa) == 1:
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
                if np.size(bound[0]) == 1 and np.size(bound[1]) == 1:
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
        self.optimizer._llambda = self.n_sample

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
                return self._add_unit(res.args)
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


class HashablePartial:
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, other):
        return (type(other) is HashablePartial and
                self.f.__code__ == other.f.__code__ and
                self.args == other.args and self.kwargs == other.kwargs)

    def __hash__(self):
        return hash(
            (
                self.f.__code__,
                self.args,
                tuple(sorted(self.kwargs.items(), key=lambda kv: kv[0])),
            ),
        )

    def __call__(self, *args, **kwargs):
        return self.f(*self.args, *args, **self.kwargs, **kwargs)


def ravel_pytree(pytree):
    """Ravel (flatten) a pytree of arrays down to a 1D array.

    Args:
      pytree: a pytree of arrays and scalars to ravel.

    Returns:
      A pair where the first element is a 1D array representing the flattened and
      concatenated leaf values, with dtype determined by promoting the dtypes of
      leaf values, and the second element is a callable for unflattening a 1D
      vector of the same length back to a pytree of the same structure as the
      input ``pytree``. If the input pytree is empty (i.e. has no leaves) then as
      a convention a 1D empty array of dtype float32 is returned in the first
      component of the output.

    """
    leaves, treedef = jax.tree.flatten(pytree)
    flat, unravel_list = _ravel_list(leaves)
    return flat, HashablePartial(unravel_pytree, treedef, unravel_list)


def unravel_pytree(treedef, unravel_list, flat):
    return jax.tree.unflatten(treedef, unravel_list(flat))


def _ravel_list(lst):
    if not lst:
        return jnp.array([], jnp.float32), lambda _: []
    from_dtypes = tuple(u.math.get_dtype(l) for l in lst)
    to_dtype = jax.dtypes.result_type(*from_dtypes)
    sizes, shapes = unzip2((jnp.size(x), jnp.shape(x)) for x in lst)
    indices = tuple(np.cumsum(sizes))

    if all(dt == to_dtype for dt in from_dtypes):
        # Skip any dtype conversion, resulting in a dtype-polymorphic `unravel`.
        # See https://github.com/google/jax/issues/7809.
        del from_dtypes, to_dtype
        raveled = jnp.concatenate([jnp.ravel(e) for e in lst])
        return raveled, HashablePartial(_unravel_list_single_dtype, indices, shapes)

    # When there is more than one distinct input dtype, we perform type
    # conversions and produce a dtype-specific unravel function.
    ravel = lambda e: jnp.ravel(jax.lax.convert_element_type(e, to_dtype))
    raveled = jnp.concatenate([ravel(e) for e in lst])
    unrav = HashablePartial(_unravel_list, indices, shapes, from_dtypes, to_dtype)
    return raveled, unrav


def _unravel_list_single_dtype(indices, shapes, arr):
    chunks = jnp.split(arr, indices[:-1])
    return [chunk.reshape(shape) for chunk, shape in safe_zip(chunks, shapes)]


def _unravel_list(indices, shapes, from_dtypes, to_dtype, arr):
    arr_dtype = u.math.get_dtype(arr)
    if arr_dtype != to_dtype:
        raise TypeError(
            f"unravel function given array of dtype {arr_dtype}, "
            f"but expected dtype {to_dtype}"
        )
    chunks = jnp.split(arr, indices[:-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore complex-to-real cast warning
        return [
            jax.lax.convert_element_type(chunk.reshape(shape), dtype)
            for chunk, shape, dtype in safe_zip(chunks, shapes, from_dtypes)
        ]


def scipy_minimize_with_jax(
    x0,
    loss_fun: Callable,
    jac: Optional[Callable] = None,
    method: Optional[str] = None,
    args: Tuple = (),
    bounds=None,
    constraints=(),
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[Dict] = None
):
    """
    A simple wrapper for scipy.optimize.minimize using JAX.

    Parameters
    ----------
    loss_fun: Callable
      The objective function to be minimized, written in JAX code
      so that it is automatically differentiable.  It is of type,
      ```fun: x, *args -> float``` where `x` is a PyTree and args
      is a tuple of the fixed parameters needed to completely specify the function.

    jac: Callable
      The gradient of the objective function, written in JAX code
      so that it is automatically differentiable.  It is of type,
      ```jac: x, *args -> float``` where `x` is a PyTree and args
      is a tuple of the fixed parameters needed to completely specify the function.

    x0: ArrayLike
      Initial guess represented as a JAX PyTree.

    args: tuple, optional.
      Extra arguments passed to the objective function
      and its derivative.  Must consist of valid JAX types; e.g. the leaves
      of the PyTree must be floats.

    method : str or callable, optional
      Type of solver.  Should be one of
          - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
          - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
          - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
          - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
          - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
          - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
          - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
          - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
          - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
          - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
          - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
          - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
          - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
          - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
          - custom - a callable object (added in version 0.14.0),
            see below for description.
      If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
      depending on if the problem has constraints or bounds.

    bounds : sequence or `Bounds`, optional
      Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
      trust-constr methods. There are two ways to specify the bounds:
          1. Instance of `Bounds` class.
          2. Sequence of ``(min, max)`` pairs for each element in `x`. None
          is used to specify no bound.
      Note that in order to use `bounds` you will need to manually flatten
      them in the same order as your inputs `x0`.

    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
      Constraints definition (only for COBYLA, SLSQP and trust-constr).
      Constraints for 'trust-constr' are defined as a single object or a
      list of objects specifying constraints to the optimization problem.
      Available constraints are:
          - `LinearConstraint`
          - `NonlinearConstraint`
      Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
      Each dictionary with fields:
          type : str
              Constraint type: 'eq' for equality, 'ineq' for inequality.
          fun : callable
              The function defining the constraint.
          jac : callable, optional
              The Jacobian of `fun` (only for SLSQP).
          args : sequence, optional
              Extra arguments to be passed to the function and Jacobian.
      Equality constraint means that the constraint function result is to
      be zero whereas inequality means that it is to be non-negative.
      Note that COBYLA only supports inequality constraints.

      Note that in order to use `constraints` you will need to manually flatten
      them in the same order as your inputs `x0`.

    tol : float, optional
      Tolerance for termination. For detailed control, use solver-specific
      options.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:
            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see :func:`show_options()`.

    callback : callable, optional
        Called after each iteration. For 'trust-constr' it is a callable with
        the signature:
            ``callback(xk, OptimizeResult state) -> bool``
        where ``xk`` is the current parameter vector represented as a PyTree,
         and ``state`` is an `OptimizeResult` object, with the same fields
        as the ones from the return. If callback returns True the algorithm
        execution is terminated.

        For all the other methods, the signature is:
            ```callback(xk)```
        where `xk` is the current parameter vector, represented as a PyTree.

    Returns
    -------
    res : The optimization result represented as a ``OptimizeResult`` object.
      Important attributes are:
          ``x``: the solution array, represented as a JAX PyTree
          ``success``: a Boolean flag indicating if the optimizer exited successfully
          ``message``: describes the cause of the termination.
      See `scipy.optimize.OptimizeResult` for a description of other attributes.

    """

    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    x0_flat, unravel = ravel_pytree(x0)

    # Wrap the objective function to consume flat _original_
    # numpy arrays and produce scalar outputs.
    def fun_wrapper(x_flat, *fun_args):
        x = unravel(x_flat)
        r = loss_fun(x, *fun_args)
        return float(r)

    # Wrap the gradient in a similar manner
    jac = jax.jit(jax.grad(loss_fun)) if jac is None else jac

    def jac_wrapper(x_flat, *fun_args):
        x = unravel(x_flat)
        g_flat, _ = ravel_pytree(jac(x, *fun_args))
        return np.array(g_flat)

    # Wrap the callback to consume a pytree
    def callback_wrapper(x_flat, *fun_args):
        if callback is not None:
            x = unravel(x_flat)
            return callback(x, *fun_args)

    # Minimize with scipy
    results = minimize(
        fun_wrapper,
        x0_flat,
        args=args,
        method=method,
        jac=jac_wrapper,
        callback=callback_wrapper,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options=options
    )

    # pack the output back into a PyTree
    results["x"] = unravel(results["x"])
    return results


class ScipyOptimizer(Optimizer):
    """
    A simple wrapper for scipy.optimize.minimize using JAX.

    Parameters
    ----------
    loss_fun: Callable
      The objective function to be minimized, written in JAX code
      so that it is automatically differentiable.  It is of type,
      ```fun: x, *args -> float``` where `x` is a PyTree and args
      is a tuple of the fixed parameters needed to completely specify the function.

    method : str or callable, optional
      Type of solver.  Should be one of
          - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
          - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
          - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
          - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
          - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
          - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
          - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
          - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
          - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
          - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
          - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
          - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
          - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
          - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
          - custom - a callable object (added in version 0.14.0),
            see below for description.
      If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
      depending on if the problem has constraints or bounds.

    bounds : sequence or `Bounds`, optional
      Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and
      trust-constr methods. There are two ways to specify the bounds:
          1. Instance of `Bounds` class.
          2. Sequence of ``(min, max)`` pairs for each element in `x`. None
          is used to specify no bound.
      Note that in order to use `bounds` you will need to manually flatten
      them in the same order as your inputs `x0`.

    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
      Constraints definition (only for COBYLA, SLSQP and trust-constr).
      Constraints for 'trust-constr' are defined as a single object or a
      list of objects specifying constraints to the optimization problem.
      Available constraints are:
          - `LinearConstraint`
          - `NonlinearConstraint`
      Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
      Each dictionary with fields:
          type : str
              Constraint type: 'eq' for equality, 'ineq' for inequality.
          fun : callable
              The function defining the constraint.
          jac : callable, optional
              The Jacobian of `fun` (only for SLSQP).
          args : sequence, optional
              Extra arguments to be passed to the function and Jacobian.
      Equality constraint means that the constraint function result is to
      be zero whereas inequality means that it is to be non-negative.
      Note that COBYLA only supports inequality constraints.

      Note that in order to use `constraints` you will need to manually flatten
      them in the same order as your inputs `x0`.

    tol : float, optional
      Tolerance for termination. For detailed control, use solver-specific
      options.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:
            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.
        For method-specific options, see :func:`show_options()`.

    callback : callable, optional
        Called after each iteration. For 'trust-constr' it is a callable with
        the signature:
            ``callback(xk, OptimizeResult state) -> bool``
        where ``xk`` is the current parameter vector represented as a PyTree,
         and ``state`` is an `OptimizeResult` object, with the same fields
        as the ones from the return. If callback returns True the algorithm
        execution is terminated.

        For all the other methods, the signature is:
            ```callback(xk)```
        where `xk` is the current parameter vector, represented as a PyTree.

    Returns
    -------
    res : The optimization result represented as a ``OptimizeResult`` object.
      Important attributes are:
          ``x``: the solution array, represented as a JAX PyTree
          ``success``: a Boolean flag indicating if the optimizer exited successfully
          ``message``: describes the cause of the termination.
      See `scipy.optimize.OptimizeResult` for a description of other attributes.

    """

    def __init__(
        self,
        loss_fun: Callable,
        bounds: np.ndarray | Sequence,
        method: Optional[str] = None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
    ):
        if minimize is None:
            raise ImportError("Scipy is not installed. Please install it using 'pip install scipy'.")

        # The loss function
        self.loss_fun = loss_fun
        # Wrap the gradient in a similar manner
        self.jac = jax.jit(jax.grad(loss_fun))
        # The optimization method
        assert method in ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr', 'COBYLA']
        self.method = method
        # The bounds
        self.bounds = bounds
        assert len(bounds) == 2, "Bounds must be a tuple of two elements: (min, max)"
        # other parameters
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

    def minimize(self, n_iter: int = 1):
        bounds = np.asarray(self.bounds).T
        xs = np.random.uniform(self.bounds[0], self.bounds[1], size=(n_iter, len(self.bounds[0])))
        best_l = np.inf
        best_r = None

        for x0 in xs:
            results = scipy_minimize_with_jax(
                x0,
                self.loss_fun,
                jac=self.jac,
                method=self.method,
                callback=self.callback,
                bounds=bounds,
                constraints=self.constraints,
                tol=self.tol,
                options=self.options
            )
            if results.fun < best_l:
                best_l = results.fun
                best_r = results
        return best_r
