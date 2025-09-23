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
