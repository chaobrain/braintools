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


from braintools._misc import set_module_as

__all__ = [
    'spike_bitwise_or',
    'spike_bitwise_and',
    'spike_bitwise_iand',
    'spike_bitwise_not',
    'spike_bitwise_xor',
    'spike_bitwise_ixor',
    'spike_bitwise',
]


@set_module_as('braintools')
def spike_bitwise_or(x, y):
    """
    Perform a bitwise OR operation on spike tensors.

    This function computes the OR operation between two spike tensors.
    The OR operation is implemented using the formula: x + y - x * y,
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
        The result of the bitwise OR operation applied to the input tensors.
        The output tensor has the same shape as the input tensors.

    Notes
    -----
    This operation assumes that the input tensors contain binary (0 or 1) values.
    For non-binary inputs, the behavior may not correspond to a true bitwise OR.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import braintools as bt

        # Create binary spike tensors
        x = jnp.array([0, 1, 0, 1])
        y = jnp.array([0, 0, 1, 1])

        # Perform OR operation
        result = bt.spike_bitwise_or(x, y)
        # result: [0, 1, 1, 1]
    """
    return x + y - x * y


@set_module_as('braintools')
def spike_bitwise_and(x, y):
    """
    Perform a bitwise AND operation on spike tensors.

    This function computes the AND operation between two spike tensors.
    The AND operation is equivalent to element-wise multiplication for binary values.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The result of the bitwise AND operation applied to the input tensors.
        The output tensor has the same shape as the input tensors.

    Notes
    -----
    This operation is implemented using element-wise multiplication (x * y),
    which is equivalent to the AND operation for binary values.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import braintools as bt

        # Create binary spike tensors
        x = jnp.array([0, 1, 0, 1])
        y = jnp.array([0, 0, 1, 1])

        # Perform AND operation
        result = bt.spike_bitwise_and(x, y)
        # result: [0, 0, 0, 1]
    """
    return x * y


@set_module_as('braintools')
def spike_bitwise_iand(x, y):
    """
    Perform a bitwise IAND (Inverse AND) operation on spike tensors.

    This function computes the Inverse AND (IAND) operation between two spike tensors.
    IAND is defined as (NOT x) AND y.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The result of the bitwise IAND operation applied to the input tensors.
        The output tensor has the same shape as the input tensors.

    Notes
    -----
    This operation is implemented using the formula: (1 - x) * y,
    which is equivalent to the IAND operation for binary values.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import braintools as bt

        # Create binary spike tensors
        x = jnp.array([0, 1, 0, 1])
        y = jnp.array([0, 0, 1, 1])

        # Perform IAND operation: (NOT x) AND y
        result = bt.spike_bitwise_iand(x, y)
        # result: [0, 0, 1, 0]
    """
    return (1 - x) * y


@set_module_as('braintools')
def spike_bitwise_not(x):
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
        The result of the bitwise NOT operation applied to the input tensor.
        The output tensor has the same shape as the input tensor.

    Notes
    -----
    This operation is implemented using the formula: 1 - x,
    which is equivalent to the NOT operation for binary values.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import braintools as bt

        # Create binary spike tensor
        x = jnp.array([0, 1, 0, 1])

        # Perform NOT operation
        result = bt.spike_bitwise_not(x)
        # result: [1, 0, 1, 0]
    """
    return 1 - x


@set_module_as('braintools')
def spike_bitwise_xor(x, y):
    """
    Perform a bitwise XOR operation on spike tensors.

    This function computes the XOR operation between two spike tensors.
    XOR is defined as (x OR y) AND NOT (x AND y).

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The result of the bitwise XOR operation applied to the input tensors.
        The output tensor has the same shape as the input tensors.

    Notes
    -----
    This operation is implemented using the formula: x + y - 2 * x * y,
    which is equivalent to the XOR operation for binary values.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import braintools as bt

        # Create binary spike tensors
        x = jnp.array([0, 1, 0, 1])
        y = jnp.array([0, 0, 1, 1])

        # Perform XOR operation
        result = bt.spike_bitwise_xor(x, y)
        # result: [0, 1, 1, 0]
    """
    return x + y - 2 * x * y


@set_module_as('braintools')
def spike_bitwise_ixor(x, y):
    """
    Perform a bitwise IXOR (Inverse XOR) operation on spike tensors.

    This function computes the Inverse XOR (IXOR) operation between two spike tensors.
    IXOR is defined as (x AND NOT y) OR (NOT x AND y).

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.

    Returns
    -------
    array_like
        The result of the bitwise IXOR operation applied to the input tensors.
        The output tensor has the same shape as the input tensors.

    Notes
    -----
    This operation is implemented using the formula: x * (1 - y) + (1 - x) * y,
    which is equivalent to the IXOR operation for binary values.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import braintools as bt

        # Create binary spike tensors
        x = jnp.array([0, 1, 0, 1])
        y = jnp.array([0, 0, 1, 1])

        # Perform IXOR operation
        result = bt.spike_bitwise_ixor(x, y)
        # result: [0, 1, 1, 0] (same as XOR in this case)
    """
    return x * (1 - y) + (1 - x) * y


@set_module_as('braintools')
def spike_bitwise(x, y, op: str):
    r"""
    Perform bitwise operations on spike tensors.

    This function applies various bitwise operations on spike tensors based on the
    specified operation. It supports 'or', 'and', 'iand', 'xor', and 'ixor' operations.

    Parameters
    ----------
    x : array_like
        The first input spike tensor.
    y : array_like
        The second input spike tensor.
    op : {'or', 'and', 'iand', 'xor', 'ixor'}
        A string indicating the bitwise operation to perform.

    Returns
    -------
    array_like
        The result of the bitwise operation applied to the input tensors.

    Raises
    ------
    NotImplementedError
        If an unsupported bitwise operation is specified.

    Notes
    -----
    The function uses the following mathematical expressions for different operations:

    .. math::

       \begin{array}{ccc}
        \hline \text { Mode } & \text { Expression for } \mathrm{g}(\mathrm{x}, \mathrm{y}) & \text { Code for } \mathrm{g}(\mathrm{x}, \mathrm{y}) \\
        \hline \text { ADD } & x+y & x+y \\
        \text { AND } & x \cap y & x \cdot y \\
        \text { IAND } & (\neg x) \cap y & (1-x) \cdot y \\
        \text { OR } & x \cup y & (x+y)-(x \cdot y) \\
        \hline
        \end{array}

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import braintools as bt

        # Create binary spike tensors
        x = jnp.array([0, 1, 0, 1])
        y = jnp.array([0, 0, 1, 1])

        # Perform various operations
        or_result = bt.spike_bitwise(x, y, 'or')    # [0, 1, 1, 1]
        and_result = bt.spike_bitwise(x, y, 'and')  # [0, 0, 0, 1]
        xor_result = bt.spike_bitwise(x, y, 'xor')  # [0, 1, 1, 0]
    """
    if op == 'or':
        return spike_bitwise_or(x, y)
    elif op == 'and':
        return spike_bitwise_and(x, y)
    elif op == 'iand':
        return spike_bitwise_iand(x, y)
    elif op == 'xor':
        return spike_bitwise_xor(x, y)
    elif op == 'ixor':
        return spike_bitwise_ixor(x, y)
    else:
        raise NotImplementedError(f"Unsupported bitwise operation: {op}.")
