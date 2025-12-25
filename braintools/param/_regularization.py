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

"""
Parameter regularization classes.

This module provides regularization classes that can be applied to parameters
during training to encourage certain properties (sparsity, smoothness, etc.)
and prevent overfitting.
"""

from abc import ABC, abstractmethod

import brainstate
import brainunit as u

from ._util import get_size, get_param

Data = brainstate.typing.ArrayLike
Size = brainstate.typing.Size

__all__ = [
    'Regularization',
    'GaussianReg',
    'L1Reg',
    'L2Reg',
]


class Regularization(brainstate.nn.Module, ABC):
    """
    Abstract base class for parameter regularization.

    Provides the interface for implementing regularization terms that can be
    added to the training loss. Subclasses must implement ``loss``, ``sample_init``,
    and ``reset_value`` methods.

    Parameters
    ----------
    fit_hyper : bool, optional
        Whether to optimize the hyperparameters of the regularization
        as trainable parameters. Default is ``False``.

    Attributes
    ----------
    fit_hyper : bool
        Whether hyperparameters are trainable.

    Notes
    -----
    Regularization can be used with the ``Param`` class to add regularization
    terms to the training loss.
    """

    def __init__(self, fit_hyper: bool = False):
        super().__init__()
        self.fit_hyper = fit_hyper

    @abstractmethod
    def loss(self, value: Data) -> Data:
        """
        Calculate regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values to compute regularization for.

        Returns
        -------
        array_like
            Scalar regularization loss.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_init(self, shape: Size) -> Data:
        """
        Sample initial value from the regularization's implied prior distribution.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the parameter to initialize.

        Returns
        -------
        array_like
            Sampled initial value.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_value(self) -> Data:
        """
        Return the reset value (e.g., prior mean).

        Returns
        -------
        array_like
            Value to reset the parameter to.
        """
        raise NotImplementedError


class GaussianReg(Regularization):
    r"""
    Gaussian prior regularization.

    Implements regularization based on the negative log-likelihood of a
    Gaussian distribution:

    .. math::
        L = \sum_i \text{precision}_i \cdot (x_i - \mu_i)^2 - \sum_i \log(\text{precision}_i)

    where precision = 1/std^2.

    Parameters
    ----------
    mean : array_like
        Prior mean value.
    std : array_like
        Prior standard deviation.
    fit_hyper : bool, optional
        Whether to optimize mean and precision as trainable parameters.
        Default is ``False``.

    Attributes
    ----------
    mean : array_like or ParamState
        Prior mean (trainable if ``fit_hyper=True``).
    precision : array_like or ParamState
        Prior precision (trainable if ``fit_hyper=True``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from braintools.param import GaussianReg
    >>> reg = GaussianReg(mean=0.0, std=1.0)
    >>> value = jnp.array([0.5, -0.5])
    >>> loss = reg.loss(value)
    """

    def __init__(
        self,
        mean: Data,
        std: Data,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        mean_t = u.math.asarray(mean, dtype=brainstate.environ.dftype())
        std_t = u.math.asarray(std, dtype=brainstate.environ.dftype())
        precision = 1.0 / (std_t ** 2 + 1e-8)

        if fit_hyper:
            mean_t = brainstate.ParamState(mean_t)
            precision = brainstate.ParamState(precision)
        self.mean = mean_t
        self.precision = precision

    def loss(self, value: Data) -> Data:
        """
        Calculate Gaussian regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            Gaussian negative log-likelihood loss.
        """
        # Add lower bound for numerical stability
        prec = u.math.relu(get_param(self.precision)) + 1e-6
        loss = u.math.sum(prec * (value - get_param(self.mean)) ** 2) - u.math.sum(u.math.log(prec))
        return loss

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Gaussian prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from N(mean, std^2).
        """
        noise = brainstate.random.randn(*get_size(shape))
        std = 1.0 / u.math.sqrt(u.math.relu(get_param(self.precision)) + 1e-8)
        return get_param(self.mean) + std * noise

    def reset_value(self) -> Data:
        """
        Return the prior mean.

        Returns
        -------
        array_like
            The mean value.
        """
        return get_param(self.mean)


class L1Reg(Regularization):
    r"""
    L1 (Lasso) regularization.

    Implements L1 regularization:

    .. math::
        L = \lambda \sum_i |x_i|

    The corresponding prior is the Laplace distribution.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Attributes
    ----------
    weight : array_like or ParamState
        Regularization weight (trainable if ``fit_hyper=True``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from braintools.param import L1Reg
    >>> reg = L1Reg(weight=0.01)
    >>> value = jnp.array([1.0, -2.0, 0.5])
    >>> loss = reg.loss(value)  # Returns 0.01 * (1.0 + 2.0 + 0.5)

    Notes
    -----
    L1 regularization encourages sparsity in the parameter values.
    """

    def __init__(
        self,
        weight: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())

        if fit_hyper:
            weight_t = brainstate.ParamState(weight_t)
        self.weight = weight_t

    def loss(self, value: Data) -> Data:
        """
        Calculate L1 regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            L1 loss: weight * sum(|value|).
        """
        return u.math.relu(get_param(self.weight)) * u.math.sum(u.math.abs(value))

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Laplace prior.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from Laplace(0, 1/weight).
        """
        # L1 prior corresponds to Laplace distribution
        # Sample from Laplace(0, 1/weight)
        scale = 1.0 / (u.math.relu(get_param(self.weight)) + 1e-8)
        u_ = brainstate.random.rand(*get_size(shape)) - 0.5
        return scale * u.math.sign(u_) * u.math.log(1 - 2 * u.math.abs(u_) + 1e-8)

    def reset_value(self) -> Data:
        """
        Return zero (the mode of Laplace(0, b)).

        Returns
        -------
        float
            Zero.
        """
        return 0.0


class L2Reg(Regularization):
    r"""
    L2 (Ridge) regularization.

    Implements L2 regularization:

    .. math::
        L = \lambda \sum_i x_i^2

    The corresponding prior is the Gaussian distribution with zero mean.

    Parameters
    ----------
    weight : float, optional
        Regularization weight (lambda). Default is 1.0.
    fit_hyper : bool, optional
        Whether to optimize weight as a trainable parameter.
        Default is ``False``.

    Attributes
    ----------
    weight : array_like or ParamState
        Regularization weight (trainable if ``fit_hyper=True``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from braintools.param import L2Reg
    >>> reg = L2Reg(weight=0.01)
    >>> value = jnp.array([1.0, -2.0, 0.5])
    >>> loss = reg.loss(value)  # Returns 0.01 * (1.0 + 4.0 + 0.25)

    Notes
    -----
    L2 regularization encourages small parameter values and is more
    numerically stable than L1 regularization.
    """

    def __init__(
        self,
        weight: float = 1.0,
        fit_hyper: bool = False,
    ):
        super().__init__(fit_hyper)

        weight_t = u.math.asarray(weight, dtype=brainstate.environ.dftype())
        if fit_hyper:
            weight_t = brainstate.ParamState(weight_t)
        self.weight = weight_t

    def loss(self, value: Data) -> Data:
        """
        Calculate L2 regularization loss.

        Parameters
        ----------
        value : array_like
            Parameter values.

        Returns
        -------
        array_like
            L2 loss: weight * sum(value^2).
        """
        return brainstate.nn.relu(get_param(self.weight)) * u.math.sum(value ** 2)

    def sample_init(self, shape: Size) -> Data:
        """
        Sample from the Gaussian prior with zero mean.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the sample.

        Returns
        -------
        array_like
            Sample from N(0, 1/weight).
        """
        # L2 prior corresponds to Gaussian with zero mean
        std = 1.0 / u.math.sqrt(u.math.relu(get_param(self.weight)) + 1e-8)
        return std * brainstate.random.randn(*get_size(shape))

    def reset_value(self) -> Data:
        """
        Return zero (the mean of N(0, sigma^2)).

        Returns
        -------
        float
            Zero.
        """
        return 0.0
