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


import warnings
from typing import Callable, Optional, Sequence, Dict, Tuple, Union, Any

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainstate._compatible_import import safe_zip, unzip2
from scipy.optimize import minimize

from ._base import Optimizer

__all__ = [
    'ScipyOptimizer',
]


class HashablePartial:
    __module__ = 'braintools.optim'

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
    jac = brainstate.transform.jit(brainstate.transform.grad(loss_fun)) if jac is None else jac

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
    SciPy-based optimizer with dict/sequence bounds compatible with Nevergrad.

    This optimizer accepts the same bounds structure as
    ``braintools.optim.NevergradOptimizer`` and a loss function with matching
    signature:
      - If ``bounds`` is a sequence of ``(min, max)`` pairs, ``loss_fun`` is
        called positionally as ``loss_fun(*params)``.
      - If ``bounds`` is a dict mapping names to ``(min, max)`` pairs,
        ``loss_fun`` is called by keyword as ``loss_fun(**params)``.

    Bounds can be scalars or arrays, and may optionally be provided as
    ``brainunit.Quantity``. Internally, SciPy works on the numeric mantissas
    while your ``loss_fun`` receives values in the same structure as the
    bounds (without units).

    Parameters
    ----------
    loss_fun : callable
        Objective function returning a scalar (0-D array-like). Its signature
        must match the structure of ``bounds`` as described above.
    bounds : dict or sequence of tuple
        Search space bounds, as in Nevergrad: each value is ``(min, max)``
        where elements are scalars or arrays, optionally ``u.Quantity``. All
        elements in a pair must share shape, and units (if any) must be
        compatible. For dict bounds, names define keyword arguments to
        ``loss_fun``.
    method : str, optional
        SciPy method (e.g., ``'L-BFGS-B'``, ``'TNC'``, ``'SLSQP'``, ``'Powell'``).
        If omitted, SciPy selects a default based on constraints/bounds.
    constraints, tol, callback, options
        Passed through to ``scipy.optimize.minimize``.

    Notes
    -----
    - This wrapper flattens parameters to a 1-D vector for SciPy and
      unflattens results back to the same structure found in ``bounds``.
    - Gradients are computed via JAX auto-differentiation.

    Examples
    --------
    Minimize a simple quadratic with tuple bounds:

    >>> import jax.numpy as jnp
    >>> def loss(x, y):
    ...     return (x - 1.0)**2 + (y + 2.0)**2
    >>> bounds = [(-5.0, 5.0), (-3.0, 3.0)]
    >>> opt = ScipyOptimizer(loss_fun=loss, bounds=bounds, method='L-BFGS-B')
    >>> res = opt.minimize(n_iter=3)  # doctest: +SKIP
    >>> isinstance(res.x, (list, tuple))  # same structure as bounds  # doctest: +SKIP
    True

    With named parameters using dict bounds:

    >>> def loss(**p):  # doctest: +SKIP
    ...     return (p['a'] - 1.0)**2 + (p['b'] + 2.0)**2
    >>> bounds = {"a": (-5.0, 5.0), "b": (-3.0, 3.0)}  # doctest: +SKIP
    >>> opt = ScipyOptimizer(loss_fun=loss, bounds=bounds, method='TNC')  # doctest: +SKIP
    >>> res = opt.minimize(n_iter=2)  # doctest: +SKIP
    >>> isinstance(res.x, dict)  # doctest: +SKIP
    True
    """
    __module__ = 'braintools.optim'

    def __init__(
        self,
        loss_fun: Callable,
        bounds: Union[Sequence, Dict[str, Any]],
        method: Optional[str] = None,
        constraints=(),
        tol=None,
        callback=None,
        options=None,
    ):
        if bounds is None:
            raise ValueError("'bounds' must be provided as a dict or a sequence of (min, max) pairs.")

        self.loss_fun = loss_fun
        self.method = method
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options

        # Parse bounds into per-leaf low/high arrays preserving structure
        self._is_dict = isinstance(bounds, dict)
        self._keys: Optional[Sequence[str]] = None
        if self._is_dict:
            self._keys = list(bounds.keys())
            low_struct: Dict[str, Any] = {}
            high_struct: Dict[str, Any] = {}
            for k, bnd in bounds.items():
                assert len(bnd) == 2, f"Each bound must be (min, max); got {bnd}."
                lo = u.Quantity(bnd[0])
                hi = u.Quantity(bnd[1])
                u.fail_for_unit_mismatch(lo, hi)
                hi = hi.in_unit(lo.unit)
                l_arr = np.asarray(lo.mantissa)
                h_arr = np.asarray(hi.mantissa)
                assert l_arr.shape == h_arr.shape, (
                    f"Bounds for '{k}' must share shape, got {l_arr.shape} and {h_arr.shape}.")
                low_struct[k] = jnp.asarray(l_arr)
                high_struct[k] = jnp.asarray(h_arr)
            self._low_struct = low_struct
            self._high_struct = high_struct
        elif isinstance(bounds, (list, tuple)):
            low_list = []
            high_list = []
            for i, bnd in enumerate(bounds):
                assert len(bnd) == 2, f"Each bound must be (min, max); got {bnd}."
                lo = u.Quantity(bnd[0])
                hi = u.Quantity(bnd[1])
                u.fail_for_unit_mismatch(lo, hi)
                hi = hi.in_unit(lo.unit)
                l_arr = np.asarray(lo.mantissa)
                h_arr = np.asarray(hi.mantissa)
                assert l_arr.shape == h_arr.shape, (
                    f"Bounds at index {i} must share shape, got {l_arr.shape} and {h_arr.shape}.")
                low_list.append(jnp.asarray(l_arr))
                high_list.append(jnp.asarray(h_arr))
            self._low_struct = tuple(low_list)
            self._high_struct = tuple(high_list)
        else:
            raise ValueError(f"Unknown type of 'bounds': {type(bounds)}")

        # Build flat SciPy bounds (list of (low, high) for each scalar variable)
        lows, _ = jax.tree.flatten(self._low_struct)
        highs, _ = jax.tree.flatten(self._high_struct)
        self._flat_bounds: list[tuple[float, float]] = []
        for lo_leaf, hi_leaf in safe_zip(lows, highs):
            lo_flat = np.ravel(np.asarray(lo_leaf))
            hi_flat = np.ravel(np.asarray(hi_leaf))
            self._flat_bounds.extend(list(zip(lo_flat, hi_flat)))

    def _sample_x0(self) -> Any:
        def sample(lo, hi):
            lo_np = np.asarray(lo)
            hi_np = np.asarray(hi)
            x = np.random.uniform(lo_np, hi_np, size=lo_np.shape)
            return jnp.asarray(x, dtype=brainstate.environ.dftype())

        return jax.tree.map(sample, self._low_struct, self._high_struct)

    def _struct_to_args(self, x_struct: Any):
        if self._is_dict:
            return {k: x_struct[k] for k in self._keys}
        else:
            return tuple(x_struct)

    def _objective(self, x_struct: Any):
        params = self._struct_to_args(x_struct)
        if self._is_dict:
            r = self.loss_fun(**params)
        else:
            r = self.loss_fun(*params)
        return jnp.asarray(r)

    def minimize(self, n_iter: int = 1):
        assert isinstance(n_iter, int) and n_iter > 0, "'n_iter' must be a positive integer."

        best_res = None
        best_fun = np.inf

        for _ in range(n_iter):
            x0_struct = self._sample_x0()
            results = scipy_minimize_with_jax(
                x0_struct,
                self._objective,
                jac=None,
                method=self.method,
                callback=self.callback,
                bounds=self._flat_bounds,
                constraints=self.constraints,
                tol=self.tol,
                options=self.options
            )
            if results.fun < best_fun:
                best_fun = results.fun
                best_res = results
        return best_res
