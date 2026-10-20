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


import jax.numpy as jnp
from brainstate.typing import ArrayLike

from braintools._misc import set_module_as

__all__ = [
    'bitwise_or',
    'bitwise_and',
    'bitwise_iand',
    'bitwise_not',
    'bitwise_xor',
    'bitwise_ixor',
    'bitwise',
]


def _is_bool(a: ArrayLike) -> bool:
    return jnp.asarray(a).dtype == jnp.bool_


def _as_numeric(a: ArrayLike) -> ArrayLike:
    """Promote boolean spike masks to integers for the arithmetic identities.

    ``bool + bool`` is *logical OR* in numpy/jax (and ``bool - bool`` is rejected
    outright), so the polynomial formulas used below are wrong or raise for
    boolean inputs. When *every* input is boolean the ops use the logical
    functions instead and return a boolean mask; this promotion only covers a
    boolean mask mixed with a numeric spike tensor, so the result takes the
    numeric dtype. Non-boolean inputs are returned untouched.
    """
    if _is_bool(a):
        return jnp.asarray(a, dtype=jnp.int32)
    return a


@set_module_as('braintools.spike')
def bitwise_or(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform a bitwise OR operation on spike tensors.

    This function computes the OR operation between two spike tensors.
    The OR operation is implemented using the formula ``x + y - x * y``,
    which is equivalent to the OR operation for binary values.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The element-wise OR of the inputs, with their broadcast shape.

    See Also
    --------
    bitwise : Dispatch to a bitwise operation by name.
    bitwise_and, bitwise_xor, bitwise_not

    Notes
    -----
    This operation assumes that the input tensors contain binary (0 or 1)
    values. For non-binary inputs, the behavior may not correspond to a true
    bitwise OR. Boolean inputs give a boolean result (computed with the
    logical functions); numeric inputs use the arithmetic formula, which keeps
    the dtype and stays differentiable for surrogate-gradient spikes.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> x = jnp.array([0, 0, 1, 1])
        >>> y = jnp.array([0, 1, 0, 1])
        >>> braintools.spike.bitwise_or(x, y)
        Array([0, 1, 1, 1], dtype=int32)
        >>>
        >>> braintools.spike.bitwise_or(x.astype(bool), y.astype(bool))
        Array([False,  True,  True,  True], dtype=bool)
    """
    if _is_bool(x) and _is_bool(y):
        return jnp.logical_or(x, y)
    x, y = _as_numeric(x), _as_numeric(y)
    return x + y - x * y


@set_module_as('braintools.spike')
def bitwise_and(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform a bitwise AND operation on spike tensors.

    This function computes the AND operation between two spike tensors.
    The AND operation is equivalent to element-wise multiplication for binary
    values.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The element-wise AND of the inputs, with their broadcast shape.

    See Also
    --------
    bitwise : Dispatch to a bitwise operation by name.
    bitwise_or, bitwise_iand

    Notes
    -----
    This operation is implemented using element-wise multiplication
    (``x * y``), which is equivalent to the AND operation for binary values.
    Boolean inputs give a boolean result (computed with the logical
    functions); numeric inputs use the arithmetic formula, which keeps the
    dtype and stays differentiable for surrogate-gradient spikes.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> x = jnp.array([0, 0, 1, 1])
        >>> y = jnp.array([0, 1, 0, 1])
        >>> braintools.spike.bitwise_and(x, y)
        Array([0, 0, 0, 1], dtype=int32)
        >>>
        >>> # A boolean mask mixed with a float tensor takes the float dtype.
        >>> braintools.spike.bitwise_and(x.astype(bool), y.astype(jnp.float32))
        Array([0., 0., 0., 1.], dtype=float32)
    """
    if _is_bool(x) and _is_bool(y):
        return jnp.logical_and(x, y)
    x, y = _as_numeric(x), _as_numeric(y)
    return x * y


@set_module_as('braintools.spike')
def bitwise_iand(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform a bitwise IAND (Inverse AND) operation on spike tensors.

    This function computes the Inverse AND (IAND) operation between two spike
    tensors. IAND is defined as ``(NOT x) AND y``.

    Parameters
    ----------
    x : array_like
        The first input spike tensor (the negated operand).
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The element-wise IAND of the inputs, with their broadcast shape.

    See Also
    --------
    bitwise : Dispatch to a bitwise operation by name.
    bitwise_and, bitwise_not

    Notes
    -----
    This operation is implemented using the formula ``(1 - x) * y``, which is
    equivalent to the IAND operation for binary values. Boolean inputs give a
    boolean result (computed with the logical functions); numeric inputs use
    the arithmetic formula, which keeps the dtype and stays differentiable for
    surrogate-gradient spikes.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> x = jnp.array([0, 0, 1, 1])
        >>> y = jnp.array([0, 1, 0, 1])
        >>> braintools.spike.bitwise_iand(x, y)
        Array([0, 1, 0, 0], dtype=int32)
    """
    if _is_bool(x) and _is_bool(y):
        return jnp.logical_and(jnp.logical_not(x), y)
    x, y = _as_numeric(x), _as_numeric(y)
    return (1 - x) * y


@set_module_as('braintools.spike')
def bitwise_not(x: ArrayLike) -> ArrayLike:
    """
    Perform a bitwise NOT operation on spike tensors.

    This function computes the NOT operation on a spike tensor.
    The NOT operation inverts the binary values in the tensor.

    Parameters
    ----------
    x : array_like
        The input spike tensor.

    Returns
    -------
    array_like
        The element-wise NOT of the input, with the same shape.

    See Also
    --------
    bitwise_iand : ``(NOT x) AND y``.

    Notes
    -----
    This operation is implemented using the formula ``1 - x``, which is
    equivalent to the NOT operation for binary values. Boolean inputs give a
    boolean result (computed with the logical functions); numeric inputs use
    the arithmetic formula, which keeps the dtype and stays differentiable for
    surrogate-gradient spikes.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> braintools.spike.bitwise_not(jnp.array([0, 0, 1, 1]))
        Array([1, 1, 0, 0], dtype=int32)
        >>>
        >>> braintools.spike.bitwise_not(jnp.array([False, False, True, True]))
        Array([ True,  True, False, False], dtype=bool)
    """
    if _is_bool(x):
        return jnp.logical_not(x)
    return 1 - x


@set_module_as('braintools.spike')
def bitwise_xor(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform a bitwise XOR operation on spike tensors.

    This function computes the XOR operation between two spike tensors.
    XOR is defined as ``(x OR y) AND NOT (x AND y)``.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The element-wise XOR of the inputs, with their broadcast shape.

    See Also
    --------
    bitwise : Dispatch to a bitwise operation by name.
    bitwise_ixor, bitwise_or

    Notes
    -----
    This operation is implemented using the formula ``x + y - 2 * x * y``,
    which is equivalent to the XOR operation for binary values. Boolean inputs
    give a boolean result (computed with the logical functions); numeric
    inputs use the arithmetic formula, which keeps the dtype and stays
    differentiable for surrogate-gradient spikes.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> x = jnp.array([0, 0, 1, 1])
        >>> y = jnp.array([0, 1, 0, 1])
        >>> braintools.spike.bitwise_xor(x, y)
        Array([0, 1, 1, 0], dtype=int32)
    """
    if _is_bool(x) and _is_bool(y):
        return jnp.logical_xor(x, y)
    x, y = _as_numeric(x), _as_numeric(y)
    return x + y - 2 * x * y


@set_module_as('braintools.spike')
def bitwise_ixor(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """
    Perform a bitwise IXOR (Inverse XOR) operation on spike tensors.

    This function computes the Inverse XOR (IXOR) operation between two spike
    tensors. IXOR is defined as ``(x AND NOT y) OR (NOT x AND y)``, which
    equals XOR on binary values.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The element-wise IXOR of the inputs, with their broadcast shape.

    See Also
    --------
    bitwise : Dispatch to a bitwise operation by name.
    bitwise_xor

    Notes
    -----
    This operation is implemented using the formula
    ``x * (1 - y) + (1 - x) * y``, which is equivalent to the IXOR operation
    for binary values. Boolean inputs give a boolean result (computed with the
    logical functions); numeric inputs use the arithmetic formula, which keeps
    the dtype and stays differentiable for surrogate-gradient spikes.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> x = jnp.array([0, 0, 1, 1])
        >>> y = jnp.array([0, 1, 0, 1])
        >>> braintools.spike.bitwise_ixor(x, y)
        Array([0, 1, 1, 0], dtype=int32)
    """
    if _is_bool(x) and _is_bool(y):
        return jnp.logical_xor(x, y)
    x, y = _as_numeric(x), _as_numeric(y)
    return x * (1 - y) + (1 - x) * y


@set_module_as('braintools.spike')
def bitwise(x: ArrayLike, y: ArrayLike, op: str) -> ArrayLike:
    r"""
    Perform bitwise operations on spike tensors.

    This function applies various bitwise operations on spike tensors based on
    the specified operation. It supports the ``'or'``, ``'and'``, ``'iand'``,
    ``'xor'`` and ``'ixor'`` operations.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.
    op : {'or', 'and', 'iand', 'xor', 'ixor'}
        The bitwise operation to perform.

    Returns
    -------
    array_like
        The result of the bitwise operation applied to the input tensors.

    Raises
    ------
    NotImplementedError
        If an unsupported bitwise operation is specified.

    See Also
    --------
    bitwise_or, bitwise_and, bitwise_iand, bitwise_xor, bitwise_ixor

    Notes
    -----
    For numeric inputs the function uses the following expressions; boolean
    inputs use the equivalent logical functions and return a boolean mask:

    .. math::

       \begin{array}{ccc}
        \hline \text { Mode } & \text { Expression for } \mathrm{g}(\mathrm{x}, \mathrm{y}) & \text { Code for } \mathrm{g}(\mathrm{x}, \mathrm{y}) \\
        \hline \text { AND } & x \cap y & x \cdot y \\
        \text { IAND } & (\neg x) \cap y & (1-x) \cdot y \\
        \text { OR } & x \cup y & (x+y)-(x \cdot y) \\
        \text { XOR } & (x \cup y) \cap \neg (x \cap y) & x+y-2 x \cdot y \\
        \text { IXOR } & (x \cap \neg y) \cup (\neg x \cap y) & x \cdot (1-y)+(1-x) \cdot y \\
        \hline
        \end{array}

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import braintools
        >>> x = jnp.array([0, 0, 1, 1])
        >>> y = jnp.array([0, 1, 0, 1])
        >>> braintools.spike.bitwise(x, y, 'xor')
        Array([0, 1, 1, 0], dtype=int32)
        >>>
        >>> braintools.spike.bitwise(x.astype(bool), y.astype(bool), 'and')
        Array([False, False, False,  True], dtype=bool)
    """
    if op == 'or':
        return bitwise_or(x, y)
    elif op == 'and':
        return bitwise_and(x, y)
    elif op == 'iand':
        return bitwise_iand(x, y)
    elif op == 'xor':
        return bitwise_xor(x, y)
    elif op == 'ixor':
        return bitwise_ixor(x, y)
    else:
        raise NotImplementedError(f"Unsupported bitwise operation: {op}.")
