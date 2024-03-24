"""Shared neural network activations and other functions."""

import operator
from functools import partial
from typing import Any, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_jvp
from jax import lax
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

__all__ = [
  "relu",
  "squareplus",
  "softplus",
  "soft_sign",
  "sigmoid",
  "silu",
  "swish",
  "log_sigmoid",
  "elu",
  "leaky_relu",
  "hard_tanh",
  "celu",
  "selu",
  "gelu",
  "glu",
  "logsumexp",
  "log_softmax",
  "softmax",
  "standardize",
  "one_hot",
  "relu6",
  "hard_sigmoid",
  "hard_silu",
  "hard_swish",
  'hard_shrink',
  'rrelu',
  'mish',
  'soft_shrink',
  'prelu',
  'tanh_shrink',
  'softmin',
]


def softmin(x, axis=-1):
  r"""Applies the Softmin function to an n-dimensional input Tensor
  rescaling them so that the elements of the n-dimensional output Tensor
  lie in the range `[0, 1]` and sum to 1.

  Softmin is defined as:

  .. math::
      \text{Softmin}(x_{i}) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}

  Shape:
      - Input: :math:`(*)` where `*` means, any number of additional
        dimensions
      - Output: :math:`(*)`, same shape as the input

  Args:
      axis (int): A dimension along which Softmin will be computed (so every slice
          along dim will sum to 1).
  """
  unnormalized = jnp.exp(-x)
  return unnormalized / unnormalized.sum(axis, keepdims=True)


def tanh_shrink(x):
  r"""Applies the element-wise function:

  .. math::
      \text{Tanhshrink}(x) = x - \tanh(x)
  """
  return x - jnp.tanh(x)


def prelu(x, a=0.25):
  r"""Applies the element-wise function:

  .. math::
      \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

  or

  .. math::
      \text{PReLU}(x) =
      \begin{cases}
      x, & \text{ if } x \geq 0 \\
      ax, & \text{ otherwise }
      \end{cases}

  Here :math:`a` is a learnable parameter. When called without arguments, `nn.PReLU()` uses a single
  parameter :math:`a` across all input channels. If called with `nn.PReLU(nChannels)`,
  a separate :math:`a` is used for each input channel.
  """
  return jnp.where(x >= 0., x, a * x)


def soft_shrink(x, lambd=0.5):
  r"""Applies the soft shrinkage function elementwise:

  .. math::
      \text{SoftShrinkage}(x) =
      \begin{cases}
      x - \lambda, & \text{ if } x > \lambda \\
      x + \lambda, & \text{ if } x < -\lambda \\
      0, & \text{ otherwise }
      \end{cases}

  Args:
      lambd: the :math:`\lambda` (must be no less than zero) value for the Softshrink formulation. Default: 0.5

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.
  """
  return jnp.where(x > lambd, x - lambd, jnp.where(x < -lambd, x + lambd, 0.))


def mish(x):
  r"""Applies the Mish function, element-wise.

  Mish: A Self Regularized Non-Monotonic Neural Activation Function.

  .. math::
      \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

  .. note::
      See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.
  """
  return x * jnp.tanh(softplus(x))


def rrelu(key, x, lower=0.125, upper=0.3333333333333333):
  r"""Applies the randomized leaky rectified liner unit function, element-wise,
  as described in the paper:

  `Empirical Evaluation of Rectified Activations in Convolutional Network`_.

  The function is defined as:

  .. math::
      \text{RReLU}(x) =
      \begin{cases}
          x & \text{if } x \geq 0 \\
          ax & \text{ otherwise }
      \end{cases}

  where :math:`a` is randomly sampled from uniform distribution
  :math:`\mathcal{U}(\text{lower}, \text{upper})`.

   See: https://arxiv.org/pdf/1505.00853.pdf

  Args:
      lower: lower bound of the uniform distribution. Default: :math:`\frac{1}{8}`
      upper: upper bound of the uniform distribution. Default: :math:`\frac{1}{3}`

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  .. _`Empirical Evaluation of Rectified Activations in Convolutional Network`:
      https://arxiv.org/abs/1505.00853
  """
  x = jnp.asarray(x)
  a = jax.random.uniform(key, x.shape, x.dtype, lower, upper)
  return jnp.where(x >= 0., x, a * x)


def hard_shrink(x, lambd=0.5):
  r"""Applies the Hard Shrinkage (Hardshrink) function element-wise.

  Hardshrink is defined as:

  .. math::
      \text{HardShrink}(x) =
      \begin{cases}
      x, & \text{ if } x > \lambda \\
      x, & \text{ if } x < -\lambda \\
      0, & \text{ otherwise }
      \end{cases}

  Args:
      lambd: the :math:`\lambda` value for the Hardshrink formulation. Default: 0.5

  Shape:
      - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
      - Output: :math:`(*)`, same shape as the input.

  """
  return jnp.where(x > lambd, x, jnp.where(x < -lambd, x, 0.))


def canonicalize_axis(axis, num_dims) -> int:
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(f"axis {axis} is out of bounds for array of dimension {num_dims}")
  if axis < 0:
    axis = axis + num_dims
  return axis


# activations

@custom_jvp
@jax.jit
def relu(x: ArrayLike) -> jax.Array:
  r"""Rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{relu}(x) = \max(x, 0)

  except under differentiation, we take:

  .. math::
    \nabla \mathrm{relu}(0) = 0

  For more information see
  `Numerical influence of ReLU’(0) on backpropagation
  <https://openreview.net/forum?id=urrcVI-_jRm>`_.

  Args:
    x : input array

  Returns:
    An array.

  Example:
    >>> jax.nn.relu(jax.numpy.array([-2., -1., -0.5, 0, 0.5, 1., 2.]))
    Array([0. , 0. , 0. , 0. , 0.5, 1. , 2. ], dtype=float32)

  See also:
    :func:`relu6`

  """
  return jnp.maximum(x, 0)


# For behavior at 0, see https://openreview.net/forum?id=urrcVI-_jRm
relu.defjvps(lambda g, ans, x: lax.select(x > 0, g, lax.full_like(g, 0)))


@jax.jit
def squareplus(x: ArrayLike, b: ArrayLike = 4) -> jax.Array:
  r"""Squareplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}

  as described in https://arxiv.org/abs/2112.11687.

  Args:
    x : input array
    b : smoothness parameter
  """
  x = jnp.asarray(x)
  b = jnp.asarray(b)
  y = x + jnp.sqrt(jnp.square(x) + b)
  return y / 2


@jax.jit
def softplus(x: ArrayLike) -> jax.Array:
  r"""Softplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{softplus}(x) = \log(1 + e^x)

  Args:
    x : input array
  """
  return jnp.logaddexp(x, 0)


@jax.jit
def soft_sign(x: ArrayLike) -> jax.Array:
  r"""Soft-sign activation function.

  Computes the element-wise function

  .. math::
    \mathrm{soft\_sign}(x) = \frac{x}{|x| + 1}

  Args:
    x : input array
  """
  x_arr = jnp.asarray(x)
  return x_arr / (jnp.abs(x_arr) + 1)


@partial(jax.jit, inline=True)
def sigmoid(x: ArrayLike) -> jax.Array:
  r"""Sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`log_sigmoid`

  """
  return lax.logistic(x)


@jax.jit
def silu(x: ArrayLike) -> jax.Array:
  r"""SiLU (a.k.a. swish) activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}

  :func:`swish` and :func:`silu` are both aliases for the same function.

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`sigmoid`
  """
  x_arr = jnp.asarray(x)
  return x_arr * sigmoid(x_arr)


swish = silu


@jax.jit
def log_sigmoid(x: ArrayLike) -> jax.Array:
  r"""Log-sigmoid activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{log\_sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`sigmoid`
  """
  x_arr = jnp.asarray(x)
  return -softplus(-x_arr)


@jax.jit
def elu(x: ArrayLike, alpha: ArrayLike = 1.0) -> jax.Array:
  r"""Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(x) - 1\right), & x \le 0
    \end{cases}

  Args:
    x : input array
    alpha : scalar or array of alpha values (default: 1.0)

  Returns:
    An array.

  See also:
    :func:`selu`
  """
  x_arr = jnp.asarray(x)
  return jnp.where(x_arr > 0,
                   x_arr,
                   alpha * jnp.expm1(jnp.where(x_arr > 0, 0., x_arr)))


@jax.jit
def leaky_relu(x: ArrayLike, negative_slope: ArrayLike = 1e-2) -> jax.Array:
  r"""Leaky rectified linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{leaky\_relu}(x) = \begin{cases}
      x, & x \ge 0\\
      \alpha x, & x < 0
    \end{cases}

  where :math:`\alpha` = :code:`negative_slope`.

  Args:
    x : input array
    negative_slope : array or scalar specifying the negative slope (default: 0.01)

  Returns:
    An array.

  See also:
    :func:`relu`
  """
  x_arr = jnp.asarray(x)
  return jnp.where(x_arr >= 0, x_arr, negative_slope * x_arr)


@jax.jit
def hard_tanh(x: ArrayLike) -> jax.Array:
  r"""Hard :math:`\mathrm{tanh}` activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{hard\_tanh}(x) = \begin{cases}
      -1, & x < -1\\
      x, & -1 \le x \le 1\\
      1, & 1 < x
    \end{cases}

  Args:
    x : input array

  Returns:
    An array.
  """
  x_arr = jnp.asarray(x)
  return jnp.where(x_arr > 1, 1, jnp.where(x_arr < -1, -1, x_arr))


@jax.jit
def celu(x: ArrayLike, alpha: ArrayLike = 1.0) -> jax.Array:
  r"""Continuously-differentiable exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{celu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0
    \end{cases}

  For more information, see
  `Continuously Differentiable Exponential Linear Units
  <https://arxiv.org/pdf/1704.07483.pdf>`_.

  Args:
    x : input array
    alpha : array or scalar (default: 1.0)

  Returns:
    An array.
  """
  return jnp.maximum(x, 0.0) + alpha * jnp.expm1(jnp.minimum(x, 0.0) / alpha)


@jax.jit
def selu(x: ArrayLike) -> jax.Array:
  r"""Scaled exponential linear unit activation.

  Computes the element-wise function:

  .. math::
    \mathrm{selu}(x) = \lambda \begin{cases}
      x, & x > 0\\
      \alpha e^x - \alpha, & x \le 0
    \end{cases}

  where :math:`\lambda = 1.0507009873554804934193349852946` and
  :math:`\alpha = 1.6732632423543772848170429916717`.

  For more information, see
  `Self-Normalizing Neural Networks
  <https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf>`_.

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`elu`
  """
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * elu(x, alpha)


def gelu(x: ArrayLike, approximate: bool = True) -> jax.Array:
  r"""Gaussian error linear unit activation function.

  If ``approximate=False``, computes the element-wise function:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(
      \frac{x}{\sqrt{2}} \right) \right)

  If ``approximate=True``, uses the approximate formulation of GELU:

  .. math::
    \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
      \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

  For more information, see `Gaussian Error Linear Units (GELUs)
  <https://arxiv.org/abs/1606.08415>`_, section 2.

  Args:
    x : input array
    approximate: whether to use the approximate or exact formulation.
  """
  from jax._src.numpy import util as numpy_util
  [x_arr] = numpy_util.promote_args_inexact("gelu", x)

  if approximate:
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x_arr.dtype)
    cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x_arr + 0.044715 * (x_arr ** 3))))
    return x_arr * cdf
  else:
    sqrt_2 = np.sqrt(2).astype(x_arr.dtype)
    return jnp.array(x_arr * (lax.erf(x_arr / sqrt_2) + 1) / 2, dtype=x_arr.dtype)


@partial(jax.jit, static_argnames=("axis",))
def glu(x: ArrayLike, axis: int = -1) -> jax.Array:
  r"""Gated linear unit activation function.

  Computes the function:

  .. math::
    \mathrm{glu}(x) =  x\left[\ldots, 0:\frac{n}{2}, \ldots\right] \cdot
      \mathrm{sigmoid} \left( x\left[\ldots, \frac{n}{2}:n, \ldots\right]
        \right)

  where the array is split into two along ``axis``. The size of the ``axis``
  dimension must be divisible by two.

  Args:
    x : input array
    axis: the axis along which the split should be computed (default: -1)

  Returns:
    An array.

  See also:
    :func:`sigmoid`
  """
  x_arr = jnp.asarray(x)
  size = x_arr.shape[axis]
  assert size % 2 == 0, "axis size must be divisible by 2"
  x1, x2 = jnp.split(x_arr, 2, axis)
  return x1 * sigmoid(x2)


@partial(jax.jit, static_argnames=("axis",))
def log_softmax(x: ArrayLike,
                axis: int | tuple[int, ...] | None = -1,
                where: ArrayLike | None = None,
                initial: ArrayLike | None = None) -> jax.Array:
  r"""Log-Softmax function.

  Computes the logarithm of the :code:`softmax` function, which rescales
  elements to the range :math:`[-\infty, 0)`.

  .. math ::
    \mathrm{log\_softmax}(x)_i = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    \right)

  Args:
    x : input array
    axis: the axis or axes along which the :code:`log_softmax` should be
      computed. Either an integer or a tuple of integers.
    where: Elements to include in the :code:`log_softmax`.
    initial: The minimum value used to shift the input array. Must be present
      when :code:`where` is not None.

  Returns:
    An array.

  See also:
    :func:`softmax`
  """
  x_arr = jnp.asarray(x)
  x_max = jnp.max(x_arr, axis, where=where, initial=initial, keepdims=True)
  x_safe = x_arr if where is None else jnp.where(where, x_arr, initial)
  shifted = x_safe - lax.stop_gradient(x_max)
  shifted_logsumexp = jnp.log(
    jnp.sum(jnp.exp(shifted), axis, where=where, keepdims=True))
  result = shifted - shifted_logsumexp
  if where is not None:
    return jnp.where(where, result, -jnp.inf)
  return result


def softmax(x: ArrayLike,
            axis: int | tuple[int, ...] | None = -1,
            where: ArrayLike | None = None,
            initial: ArrayLike | None = None) -> jax.Array:
  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
    where: Elements to include in the :code:`softmax`.
    initial: The minimum value used to shift the input array. Must be present
      when :code:`where` is not None.

  Returns:
    An array.

  See also:
    :func:`log_softmax`
  """
  return _softmax(x, axis, where, initial)  # type: ignore[return-value]


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _softmax(
    x: ArrayLike,
    axis: int | tuple[int, ...] | None = -1,
    where: ArrayLike | None = None,
    initial: ArrayLike | None = None
) -> jax.Array:
  x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
  x_safe = x if where is None else jnp.where(where, x, initial)
  unnormalized = jnp.exp(x_safe - x_max)
  result = unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)
  if where is not None:
    result = jnp.where(where, result, 0)
  return result


@_softmax.defjvp
def _softmax_jvp(axis, primals, tangents):
  (x, where, initial), (x_dot, _, _) = primals, tangents
  y = _softmax(x, axis, where, initial)
  return y, y * (x_dot - (y * x_dot).sum(axis, where=where, keepdims=True))


@partial(jax.jit, static_argnames=("axis",))
def standardize(x: ArrayLike,
                axis: int | tuple[int, ...] | None = -1,
                variance: ArrayLike | None = None,
                epsilon: ArrayLike = 1e-5,
                where: ArrayLike | None = None) -> jax.Array:
  r"""Normalizes an array by subtracting ``mean`` and dividing by :math:`\sqrt{\mathrm{variance}}`."""
  mean = jnp.mean(x, axis, keepdims=True, where=where)
  if variance is None:
    # this definition is traditionally seen as less accurate than jnp.var's
    # mean((x - mean(x))**2) but may be faster and even, given typical
    # activation distributions and low-precision arithmetic, more accurate
    # when used in neural network normalization layers
    variance = jnp.mean(
      jnp.square(x), axis, keepdims=True, where=where) - jnp.square(mean)
  return jnp.subtract(x, jnp.asarray(mean)) * lax.rsqrt(jnp.asarray(variance) + epsilon)


@partial(jax.jit, static_argnames=("num_classes", "dtype", "axis"))
def _one_hot(x: Any,
             num_classes: int, *,
             dtype: Any,
             axis: Union[int, Sequence[int]]) -> jax.Array:
  num_classes = jax.core.concrete_dim_or_error(
    num_classes,
    "The error arose in jax.nn.one_hot argument `num_classes`.")
  dtype = jax.dtypes.canonicalize_dtype(dtype)
  x_arr = jnp.asarray(x)
  try:
    output_pos_axis = canonicalize_axis(axis, x_arr.ndim + 1)
  except TypeError:
    axis_size = lax.psum(1, axis)
    if num_classes != axis_size:
      raise ValueError(f"Expected num_classes to match the size of axis {axis}, "
                       f"but {num_classes} != {axis_size}") from None
    axis_idx = lax.axis_index(axis)
    return jnp.asarray(x_arr == axis_idx, dtype=dtype)
  axis = operator.index(axis)  # type: ignore[arg-type]
  lhs = lax.expand_dims(x_arr, (axis,))
  rhs_shape = [1] * x_arr.ndim
  rhs_shape.insert(output_pos_axis, num_classes)
  rhs = lax.broadcasted_iota(x_arr.dtype, rhs_shape, output_pos_axis)
  return jnp.asarray(lhs == rhs, dtype=dtype)


def one_hot(x: Any,
            num_classes: int, *,
            dtype: Any = jnp.float_,
            axis: Union[int, Sequence[int]] = -1) -> jax.Array:
  """One-hot encodes the given indices.

  Each index in the input ``x`` is encoded as a vector of zeros of length
  ``num_classes`` with the element at ``index`` set to one::

    >>> jax.nn.one_hot(jnp.array([0, 1, 2]), 3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

  Indices outside the range [0, num_classes) will be encoded as zeros::

    >>> jax.nn.one_hot(jnp.array([-1, 3]), 3)
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

  Args:
    x: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: optional, a float dtype for the returned values (default :obj:`jnp.float_`).
    axis: the axis or axes along which the function should be
      computed.
  """
  num_classes = jax.core.concrete_dim_or_error(
    num_classes,
    "The error arose in jax.nn.one_hot argument `num_classes`.")
  return _one_hot(x, num_classes, dtype=dtype, axis=axis)


@jax.custom_jvp
@jax.jit
def relu6(x: ArrayLike) -> jax.Array:
  r"""Rectified Linear Unit 6 activation function.

  Computes the element-wise function

  .. math::
    \mathrm{relu6}(x) = \min(\max(x, 0), 6)

  except under differentiation, we take:

  .. math::
    \nabla \mathrm{relu}(0) = 0

  and

  .. math::
    \nabla \mathrm{relu}(6) = 0

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`relu`
  """
  return jnp.minimum(jnp.maximum(x, 0), 6.)


relu6.defjvps(lambda g, ans, x:
              lax.select((x > 0) & (x < 6), g, lax.full_like(g, 0)))


@jax.jit
def hard_sigmoid(x: ArrayLike) -> jax.Array:
  r"""Hard Sigmoid activation function.

  Computes the element-wise function

  .. math::
    \mathrm{hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`relu6`
  """
  return relu6(x + 3.) / 6.


@jax.jit
def hard_silu(x: ArrayLike) -> jax.Array:
  r"""Hard SiLU (swish) activation function

  Computes the element-wise function

  .. math::
    \mathrm{hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)

  Both :func:`hard_silu` and :func:`hard_swish` are aliases for the same
  function.

  Args:
    x : input array

  Returns:
    An array.

  See also:
    :func:`hard_sigmoid`
  """
  x_arr = jnp.asarray(x)
  return x_arr * hard_sigmoid(x_arr)


hard_swish = hard_silu
