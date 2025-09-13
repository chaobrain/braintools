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


from typing import Optional

import brainstate
import brainunit as u
import jax
import numpy as np

__all__ = [
    'LatencyEncoder',
    'spike_bitwise_or',
    'spike_bitwise_and',
    'spike_bitwise_iand',
    'spike_bitwise_not',
    'spike_bitwise_xor',
    'spike_bitwise_ixor',
    'spike_bitwise',
]


class LatencyEncoder:
    r"""Encode the rate input as the spike train using the latency encoding.

    Use input features to determine time-to-first spike.

    Expected inputs should be between 0 and 1. If not, the latency encoder will encode ``x``
    (normalized into ``[0, 1]`` according to
    :math:`x_{\text{normalize}} = \frac{x-\text{min_val}}{\text{max_val} - \text{min_val}}`)
    to spikes whose firing time is :math:`0 \le t_f \le \text{num_period}-1`.
    A larger ``x`` will cause the earlier firing time.


    Example::

      >>> import jax
      >>> a = jax.numpy.array([0.02, 0.5, 1])
      >>> encoder = LatencyEncoder(method='linear', normalize=True)
      >>> encoder(a, n_time=5)
      Array([[0., 0., 1.],
             [0., 0., 0.],
             [0., 1., 0.],
             [0., 0., 0.],
             [1., 0., 0.]])


    Args:
      min_val: float. The minimal value in the given data `x`, used to the data normalization.
      max_val: float. The maximum value in the given data `x`, used to the data normalization.
      method: str. How to convert intensity to firing time. Currently, we support `linear` or `log`.
        - If ``method='linear'``, the firing rate is calculated as
          :math:`t_f(x) = (\text{num_period} - 1)(1 - x)`.
        - If ``method='log'``, the firing rate is calculated as
          :math:`t_f(x) = (\text{num_period} - 1) - ln(\alpha * x + 1)`,
          where :math:`\alpha` satisfies :math:`t_f(1) = \text{num_period} - 1`.
      threshold: float. Input features below the threhold will fire at the
        final time step unless ``clip=True`` in which case they will not
        fire at all, defaults to ``0.01``.
      clip: bool. Option to remove spikes from features that fall
          below the threshold, defaults to ``False``.
      tau: float. RC Time constant for LIF model used to calculate
        firing time, defaults to ``1``.
      normalize: bool. Option to normalize the latency code such that
        the final spike(s) occur within num_steps, defaults to ``False``.
      epsilon: float. A tiny positive value to avoid rounding errors when
        using torch.arange, defaults to ``1e-7``.
    """

    def __init__(
        self,
        min_val: float = None,
        max_val: float = None,
        method: str = 'log',
        threshold: float = 0.01,
        clip: bool = False,
        tau: float = 1. * u.ms,
        normalize: bool = False,
        first_spk_time: float = 0. * u.ms,
        epsilon: float = 1e-7,
    ):
        super().__init__()

        if method not in ['linear', 'log']:
            raise ValueError('The conversion method can only be "linear" and "log".')
        self.method = method
        self.min_val = min_val
        self.max_val = max_val
        if threshold < 0 or threshold > 1:
            raise ValueError(f"``threshold`` [{threshold}] must be between [0, 1]")
        self.threshold = threshold
        self.clip = clip
        self.tau = tau
        self.normalize = normalize
        self.first_spk_time = first_spk_time
        self.first_spk_step = int(first_spk_time / brainstate.environ.get_dt())
        self.epsilon = epsilon

    def __call__(self, data, n_time: Optional[brainstate.typing.ArrayLike] = None):
        """Generate latency spikes according to the given input data.

        Ensuring x in [0., 1.].

        Args:
          data: The rate-based input.
          n_time: float. The total time to generate data. If None, use ``tau`` instead.

        Returns:
          out: array. The output spiking trains.
        """
        with jax.ensure_compile_time_eval():
            if n_time is None:
                n_time = self.tau
            tau = n_time if self.normalize else self.tau
            x = data
            if self.min_val is not None and self.max_val is not None:
                x = (x - self.min_val) / (self.max_val - self.min_val)

            # Calculate the spike time
            dt = brainstate.environ.get_dt()
            if self.method == 'linear':
                spike_time = (tau - self.first_spk_time - dt) * (1 - x) + self.first_spk_time

            elif self.method == 'log':
                x = u.math.maximum(x, self.threshold + self.epsilon)  # saturates all values below threshold.
                spike_time = (tau - self.first_spk_time - dt) * u.math.log(
                    x / (x - self.threshold)) + self.first_spk_time

            else:
                raise ValueError(f'Unsupported method: {self.method}. Only support "log" and "linear".')

            # Clip the spike time
            if self.clip:
                spike_time = u.math.where(data < self.threshold, np.inf, spike_time)
            spike_steps = u.math.round(spike_time / dt).astype(int)
            return brainstate.functional.one_hot(spike_steps, num_classes=int(n_time / dt), axis=0, dtype=x.dtype)


def spike_bitwise_or(x, y):
    """
    Perform a bitwise OR operation on spike tensors.

    This function computes the OR operation between two spike tensors.
    The OR operation is implemented using the formula: x + y - x * y,
    which is equivalent to the OR operation for binary values.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise OR operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation assumes that the input tensors contain binary (0 or 1) values.
        For non-binary inputs, the behavior may not correspond to a true bitwise OR.
    """
    return x + y - x * y


def spike_bitwise_and(x, y):
    """
    Perform a bitwise AND operation on spike tensors.

    This function computes the AND operation between two spike tensors.
    The AND operation is equivalent to element-wise multiplication for binary values.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise AND operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using element-wise multiplication (x * y),
        which is equivalent to the AND operation for binary values.
    """
    return x * y


def spike_bitwise_iand(x, y):
    """
    Perform a bitwise IAND (Inverse AND) operation on spike tensors.

    This function computes the Inverse AND (IAND) operation between two spike tensors.
    IAND is defined as (NOT x) AND y.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise IAND operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using the formula: (1 - x) * y,
        which is equivalent to the IAND operation for binary values.
    """
    return (1 - x) * y


def spike_bitwise_not(x):
    """
    Perform a bitwise NOT operation on spike tensors.

    This function computes the NOT operation on a spike tensor.
    The NOT operation inverts the binary values in the tensor.

    Args:
        x (Tensor): The input spike tensor.

    Returns:
        Tensor: The result of the bitwise NOT operation applied to the input tensor.
               The output tensor has the same shape as the input tensor.

    Note:
        This operation is implemented using the formula: 1 - x,
        which is equivalent to the NOT operation for binary values.
    """
    return 1 - x


def spike_bitwise_xor(x, y):
    """
    Perform a bitwise XOR operation on spike tensors.

    This function computes the XOR operation between two spike tensors.
    XOR is defined as (x OR y) AND NOT (x AND y).

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise XOR operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using the formula: x + y - 2 * x * y,
        which is equivalent to the XOR operation for binary values.
    """
    return x + y - 2 * x * y


def spike_bitwise_ixor(x, y):
    """
    Perform a bitwise IXOR (Inverse XOR) operation on spike tensors.

    This function computes the Inverse XOR (IXOR) operation between two spike tensors.
    IXOR is defined as (x AND NOT y) OR (NOT x AND y).

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.

    Returns:
        Tensor: The result of the bitwise IXOR operation applied to the input tensors.
               The output tensor has the same shape as the input tensors.

    Note:
        This operation is implemented using the formula: x * (1 - y) + (1 - x) * y,
        which is equivalent to the IXOR operation for binary values.
    """
    return x * (1 - y) + (1 - x) * y


def spike_bitwise(x, y, op: str):
    r"""
    Perform bitwise operations on spike tensors.

    This function applies various bitwise operations on spike tensors based on the specified operation.
    It supports 'or', 'and', 'iand', 'xor', and 'ixor' operations.

    Args:
        x (Tensor): The first input spike tensor.
        y (Tensor): The second input spike tensor.
        op (str): A string indicating the bitwise operation to perform.
            Supported operations are 'or', 'and', 'iand', 'xor', and 'ixor'.

    Returns:
        Tensor: The result of the bitwise operation applied to the input tensors.

    Raises:
        NotImplementedError: If an unsupported bitwise operation is specified.

    Note:
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
