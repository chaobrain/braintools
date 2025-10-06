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


from typing import Any, Protocol

import jax.numpy as jnp

from braintools._misc import set_module_as

__all__ = [
    "make_fenchel_young_loss",
]


class MaxFun(Protocol):

    def __call__(self, scores, *args, **kwargs: Any):
        ...


@set_module_as('braintools.metric')
def make_fenchel_young_loss(
    max_fun: MaxFun
):
    r"""Create a Fenchel-Young loss function from a max function.

    Fenchel-Young losses provide a framework for building differentiable loss
    functions from convex regularizers. They are particularly useful in machine
    learning for structured prediction tasks and provide a principled way to
    construct losses that encourage sparsity or specific structure in predictions.

    The Fenchel-Young loss is defined as:

    .. math::

        \ell_{FY}(y, \theta) = \Omega(\theta) - \langle y, \theta \rangle

    where :math:`\Omega` is a convex regularizer (the max function), 
    :math:`\theta` are the scores, and :math:`y` are the targets.

    Parameters
    ----------
    max_fun : MaxFun
        The max function (convex regularizer) on which the Fenchel-Young loss
        is built. Common choices include ``jax.scipy.special.logsumexp`` for
        softmax-based losses or custom max functions for structured outputs.

    Returns
    -------
    callable
        A Fenchel-Young loss function with signature 
        ``fenchel_young_loss(scores, targets, *args, **kwargs)`` that computes
        the loss between scores and targets.

    Notes
    -----
    .. warning::
        The resulting loss operates over the last dimension of the input arrays
        and accepts arbitrary leading dimensions. This differs from some other
        implementations that flatten inputs into 1D vectors.

    The choice of max function determines the properties of the resulting loss:
    
    - ``logsumexp``: Creates a softmax-based cross-entropy loss
    - ``max``: Creates a max-margin loss
    - Custom functions: Can create structured losses for specific applications

    Examples
    --------
    Create a softmax-based Fenchel-Young loss:

    >>> import jax.numpy as jnp
    >>> from jax.scipy.special import logsumexp
    >>> import braintools as braintools
    >>> # Create the loss function
    >>> fy_loss = braintools.metric.make_fenchel_young_loss(max_fun=logsumexp)
    >>> # Example usage
    >>> scores = jnp.array([[2.0, 1.0, 0.5], [1.5, 2.5, 1.0]])
    >>> targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    >>> loss = fy_loss(scores, targets)
    >>> print(f"Fenchel-Young loss: {loss}")

    Create a custom max function for structured prediction:

    >>> def custom_max(x):
    ...     return jnp.max(x) + 0.1 * jnp.sum(x**2)  # L2 regularized max
    >>> structured_loss = braintools.metric.make_fenchel_young_loss(max_fun=custom_max)

    See Also
    --------
    jax.scipy.special.logsumexp : Common choice for softmax-based losses
    braintools.metric.sigmoid_binary_cross_entropy : Alternative binary loss

    References
    ----------
    .. [1] Blondel, Mathieu, André FT Martins, and Vlad Niculae. 
           "Learning with Fenchel-Young losses." Journal of Machine Learning 
           Research 21.35 (2020): 1-69.
           https://arxiv.org/pdf/1901.02324.pdf
    """

    vdot_last_dim = jnp.vectorize(jnp.vdot, signature="(n),(n)->()")
    max_fun_last_dim = jnp.vectorize(max_fun, signature="(n)->()")

    def fenchel_young_loss(scores, targets, *args, **kwargs):
        max_value = max_fun_last_dim(scores, *args, **kwargs)
        return max_value - vdot_last_dim(targets, scores)

    return fenchel_young_loss
