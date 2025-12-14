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

import jax
import jax.numpy as jnp
from brainstate._compatible_import import Primitive
from brainstate.util import PrettyObject
from jax.interpreters import batching, ad, mlir

__all__ = ['Surrogate']


def _heaviside_abstract(x, dx):
    return [x]


def _heaviside_imp(x, dx):
    z = jnp.asarray(x >= 0, dtype=x.dtype)
    return [z]


def _heaviside_batching(args, axes):
    x, dx = args
    x_axis, dx_axis = axes

    # Handle case where both are batched but on different axes
    if x_axis is not None and dx_axis is not None and x_axis != dx_axis:
        dx = jnp.moveaxis(dx, dx_axis, x_axis)
        out_axis = x_axis
    elif x_axis is not None:
        out_axis = x_axis
    elif dx_axis is not None:
        out_axis = dx_axis
        x = jnp.repeat(jnp.expand_dims(x, axis=dx_axis), axis=dx_axis, repeats=dx.shape[dx_axis])
    else:
        out_axis = None

    # Since heaviside_p.multiple_results = True, bind returns a tuple
    # and we need to return (result_tuple, axes_tuple)
    result = heaviside_p.bind(x, dx)
    return result, (out_axis,)


def _heaviside_jvp(primals, tangents):
    x, dx = primals
    tx, tdx = tangents
    # Call the implementation directly instead of bind to avoid recursion
    primal_outs = _heaviside_imp(x, dx)
    # Handle gradients w.r.t. both x and dx
    # ∂output/∂x via surrogate gradient dx, plus ∂output/∂dx contribution
    # Need to handle JAX's Zero type for optimization
    if type(tx) is ad.Zero:
        tangent_x = tx  # Keep as Zero
    else:
        tangent_x = dx * tx

    if type(tdx) is ad.Zero:
        tangent_dx = tdx  # Keep as Zero
    else:
        tangent_dx = tdx

    # Combine tangents using add_tangents which handles Zero properly
    tangent_outs = [ad.add_tangents(tangent_x, tangent_dx)]
    return primal_outs, tangent_outs


def _heaviside_transpose(ct, x, dx):
    """
    Transpose rule for reverse-mode autodiff.

    This computes cotangents for the tangents (tx, tdx) given the output cotangent.

    From JVP: output_tangent = dx * tx + tdx
    Transpose: cotangent_tx = dx * ct_out, cotangent_tdx = ct_out
    """
    # ct is a tuple/list containing the cotangent for each output
    ct_out = ct[0]

    # Cotangent for tx (from dx * tx term)
    # In JAX transpose, dx is a residual and might be UndefinedPrimal if it's symbolic
    if type(dx) is ad.UndefinedPrimal:
        # Can't use dx if it's undefined - return zero
        cotangent_tx = ad.Zero(dx.aval)
    else:
        cotangent_tx = dx * ct_out

    # Cotangent for tdx (from tdx term)
    cotangent_tdx = ct_out

    return (cotangent_tx, cotangent_tdx)


heaviside_p = Primitive('heaviside_surrogate_gradient')
heaviside_p.multiple_results = True
heaviside_p.def_abstract_eval(_heaviside_abstract)
heaviside_p.def_impl(_heaviside_imp)
batching.primitive_batchers[heaviside_p] = _heaviside_batching
ad.primitive_jvps[heaviside_p] = _heaviside_jvp
# Let JAX automatically derive transpose from JVP
# ad.primitive_transposes[heaviside_p] = _heaviside_transpose
mlir.register_lowering(heaviside_p, mlir.lower_fun(_heaviside_imp, multiple_results=True))


class Surrogate(PrettyObject):
    r"""The base surrogate gradient function.

    This abstract base class defines the interface for surrogate gradient functions
    used in training spiking neural networks. Surrogate gradients replace the
    non-differentiable spike function with smooth approximations during backpropagation.

    To customize a surrogate gradient function, inherit this class and implement
    the `surrogate_fun` and `surrogate_grad` methods.

    Methods
    -------
    surrogate_fun(x)
        Defines the smooth surrogate function used for visualization and analysis.
    surrogate_grad(x)
        Defines the gradient of the surrogate function used during backpropagation.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a custom surrogate gradient function
        >>> class MySurrogate(braintools.surrogate.Surrogate):
        ...     def __init__(self, alpha=1.):
        ...         super().__init__()
        ...         self.alpha = alpha
        ...
        ...     def surrogate_fun(self, x):
        ...         # Define the smooth approximation function
        ...         return jnp.tanh(x * self.alpha) * 0.5 + 0.5
        ...
        ...     def surrogate_grad(self, x):
        ...         # Define its gradient for backpropagation
        ...         return self.alpha * 0.5 * (1 - jnp.tanh(x * self.alpha) ** 2)
        >>>
        >>> # Use the custom surrogate
        >>> my_surrogate = MySurrogate(alpha=2.0)
        >>> x = jnp.array([-1.0, 0.0, 1.0])
        >>> spikes = my_surrogate(x)  # Forward: step function
        >>> print(spikes)  # [0., 1., 1.]

    Notes
    -----
    The forward pass always returns a Heaviside step function (0 or 1),
    while the backward pass uses the custom surrogate gradient defined in
    `surrogate_grad` method. This straight-through estimator approach enables
    gradient-based training of spiking neural networks.

    """
    __module__ = 'braintools.surrogate'

    def __call__(self, x):
        dx = self.surrogate_grad(jax.lax.stop_gradient(x))
        return heaviside_p.bind(x, dx)[0]

    def surrogate_fun(self, x) -> jax.Array:
        """The surrogate function."""
        raise NotImplementedError

    def surrogate_grad(self, x) -> jax.Array:
        """The gradient function of the surrogate function."""
        raise NotImplementedError
