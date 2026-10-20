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

"""Fit brain models to data.

``braintools.fit`` is model-agnostic: it fits any pure ``simulate(params)``
function, or any ``brainstate`` module whose parameters are
``brainstate.nn.Param(..., fit=True)``, to a target.

- **Objectives**: :class:`Term` wraps a loss ``fn(prediction, target)``;
  terms compose with ``+`` and scalar ``*`` into an :class:`Objective`.
  Ready-made losses live in :mod:`braintools.fit.losses`.
- **Fitting**: :class:`Fitter` (functional mode, or :meth:`Fitter.from_module`)
  with the ``'grad'`` (vmapped multi-start, see
  :func:`braintools.optim.multistart_minimize`), ``'scipy'`` and
  ``'nevergrad'`` backends; results come back as a :class:`FitResult`.
- **Exploration**: :func:`sweep` evaluates a function over a parameter grid
  in memory-bounded vmapped chunks.
- **Uncertainty**: :func:`hessian_ci` gives curvature-based confidence
  intervals at the optimum (:class:`HessianCI`).

Examples
--------
.. code-block:: python

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from braintools.fit import Fitter, Term, losses
    >>> t = jnp.linspace(0., 1., 50)
    >>> simulate = lambda p: p['a'] * jnp.exp(-t / p['tau'])
    >>> target = simulate({'a': 2.0, 'tau': 0.3})
    >>> objective = Term(losses.mse) + 0.1 * Term(losses.correlation_loss)
    >>> fitter = Fitter(simulate, objective, {'a': (0., 5.), 'tau': (0.05, 1.)})
    >>> res = fitter.fit(target, n_starts=4, key=jax.random.PRNGKey(0))
    >>> print(round(float(res.best_params['a']), 3), round(float(res.best_params['tau']), 3))
    2.0 0.3
"""

from . import losses
from ._fitter import (
    Fitter,
    FitResult,
)
from ._objective import (
    Term,
    Objective,
)
from ._sweep import sweep
from ._uncertainty import (
    HessianCI,
    hessian_ci,
)

__all__ = [
    'losses',
    'Term',
    'Objective',
    'Fitter',
    'FitResult',
    'sweep',
    'hessian_ci',
    'HessianCI',
]
