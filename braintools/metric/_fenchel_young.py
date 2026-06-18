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

    Given a strictly convex regularizer :math:`\Omega`, its convex conjugate
    (a.k.a. the *max function* or log-partition / soft-max function) is

    .. math::

        \Omega^*(\theta) = \max_{\mu \in \mathcal{C}}
            \; \langle \theta, \mu \rangle - \Omega(\mu),

    and the associated Fenchel-Young loss is

    .. math::

        \ell_{FY}(\theta, y) = \Omega^*(\theta) - \langle \theta, y \rangle,

    where :math:`\theta` are the scores and :math:`y` is the target. ``max_fun``
    is exactly this conjugate :math:`\Omega^*` (NOT the regularizer
    :math:`\Omega` itself). The loss is convex in :math:`\theta`, non-negative,
    and minimized when the prediction matches the target. Its gradient w.r.t.
    the scores is

    .. math::

        \nabla_\theta \ell_{FY}(\theta, y) = \hat{y}(\theta) - y,
        \qquad \hat{y}(\theta) = \nabla \Omega^*(\theta),

    i.e. the prediction :math:`\hat{y}(\theta) = \nabla \Omega^*(\theta)` minus
    the target. For ``max_fun = logsumexp`` we have
    :math:`\nabla \Omega^*(\theta) = \mathrm{softmax}(\theta)`, recovering the
    softmax cross-entropy loss.

    Parameters
    ----------
    max_fun : MaxFun
        The max function :math:`\Omega^*` (the convex conjugate of the
        regularizer) on which the Fenchel-Young loss is built. It must map a
        score vector over the last dimension to a scalar, consistent with the
        ``vectorize`` signature ``"(n)->()"``. Common choices include
        ``jax.scipy.special.logsumexp`` for softmax-based losses or custom max
        functions for structured outputs.

    Returns
    -------
    callable
        A Fenchel-Young loss function with signature
        ``fenchel_young_loss(scores, targets, *args, **kwargs)`` that computes
        the loss between scores and targets. Any extra ``*args``/``**kwargs``
        are forwarded to ``max_fun``.

    Notes
    -----
    .. warning::
        The resulting loss operates over the last dimension of the input arrays
        and accepts arbitrary leading dimensions. This differs from some other
        implementations that flatten inputs into 1D vectors.

    .. warning::
        The gradient :math:`\hat{y}(\theta) - y` is obtained by *autodiff* of
        ``max_fun``. This is only correct when :math:`\Omega^*` is smooth (i.e.
        differentiable), as it is for ``logsumexp``. Sparse / piecewise-linear
        conjugates such as ``sparsemax`` or ``entmax`` are non-smooth: their
        argmax is set-valued at kink points and plain autodiff of ``max_fun``
        gives a wrong or undefined gradient. Supporting those correctly
        requires registering a ``custom_vjp`` whose backward pass returns the
        sparse prediction oracle :math:`\hat{y}(\theta) - y`; this is *not*
        implemented here (future work). Only pass a smooth, differentiable
        ``max_fun``.

    The choice of max function determines the properties of the resulting loss:

    - ``logsumexp``: Creates a softmax-based cross-entropy loss
    - ``max``: Creates a (non-smooth) max-margin loss; use only for the forward
      value, not for gradients (see warning above)
    - Custom smooth functions: Can create structured losses for specific
      applications

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
    >>> print(loss.shape)
    (2,)

    The gradient is the softmax prediction minus the target:

    >>> import jax
    >>> grad = jax.grad(lambda s, t: fy_loss(s, t).sum())(scores, targets)
    >>> print(jnp.allclose(grad, jax.nn.softmax(scores, axis=-1) - targets))
    True

    Create a custom smooth max function for structured prediction. The function
    must return a SCALAR per core call (consistent with ``"(n)->()"``):

    >>> def custom_max(x):
    ...     return logsumexp(x) + 0.1 * jnp.sum(x ** 2)  # L2-regularized soft-max
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

    def fenchel_young_loss(scores, targets, *args, **kwargs):
        # Bind the extra arguments BEFORE vectorizing. ``jnp.vectorize`` treats
        # every positional passed to the vectorized callable as an additional
        # core input, so forwarding ``*args`` through it would mis-broadcast or
        # raise. Closing over them here keeps a single core input ``s``.
        mf = lambda s: max_fun(s, *args, **kwargs)
        max_fun_last_dim = jnp.vectorize(mf, signature="(n)->()")
        max_value = max_fun_last_dim(scores)
        # Use an explicit (non-conjugating) inner product over the last
        # dimension. ``jnp.vdot`` conjugates its first argument, which would be
        # wrong for complex inputs.
        inner = jnp.sum(targets * scores, axis=-1)
        return max_value - inner

    return fenchel_young_loss
