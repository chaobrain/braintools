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

"""
Comprehensive tests for surrogate gradient functions.
"""

import brainstate.transform
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.surrogate as surrogate


def _surrogate_grad_param_grad(fn, argnums):
    """Derivative w.r.t. a surrogate parameter of the surrogate gradient d spike / d x.

    The spike itself does not depend on surrogate parameters (its parameter
    gradient is 0); the parameters act through the gradient with respect to x.
    """
    return brainstate.transform.vector_grad(brainstate.transform.vector_grad(fn, argnums=0), argnums=argnums)


class TestSigmoid:
    """Test sigmoid-based surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [1.0, 2.0, 4.])
    def test_sigmoid_forward_backward(self, x, alpha):
        """Test Sigmoid backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.Sigmoid(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert (jnp.allclose(y, forward_expected))

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [1.0, 2.0, 4.])
    def test_sigmoid_functional_api(self, x, alpha):
        """Test sigmoid functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.Sigmoid(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.sigmoid)(x, alpha=alpha)
        assert (jnp.allclose(y_class, y_func))

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [1.0, 2.0, 4.])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.sigmoid, 1)(x, alpha)

        def f(alpha):
            return surrogate.Sigmoid(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestPiecewiseQuadratic:
    """Test piecewise quadratic surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_piecewise_quadratic_forward_backward(self, x, alpha):
        """Test PiecewiseQuadratic backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.PiecewiseQuadratic(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_piecewise_quadratic_functional_api(self, x, alpha):
        """Test piecewise quadratic functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.PiecewiseQuadratic(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.piecewise_quadratic)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.piecewise_quadratic, 1)(x, alpha)

        def f(alpha):
            return surrogate.PiecewiseQuadratic(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestPiecewiseExp:
    """Test piecewise exponential surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_piecewise_exp_forward_backward(self, x, alpha):
        """Test PiecewiseExp backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.PiecewiseExp(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_piecewise_exp_functional_api(self, x, alpha):
        """Test piecewise exp functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.PiecewiseExp(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.piecewise_exp)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.piecewise_exp, 1)(x, alpha)

        def f(alpha):
            return surrogate.PiecewiseExp(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestSoftSign:
    """Test soft sign surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_soft_sign_forward_backward(self, x, alpha):
        """Test SoftSign backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.SoftSign(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_soft_sign_functional_api(self, x, alpha):
        """Test soft sign functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.SoftSign(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.soft_sign)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.soft_sign, 1)(x, alpha)

        def f(alpha):
            return surrogate.SoftSign(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestArctan:
    """Test arctan surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_arctan_forward_backward(self, x, alpha):
        """Test Arctan backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.Arctan(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_arctan_functional_api(self, x, alpha):
        """Test arctan functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.Arctan(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.arctan)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.arctan, 1)(x, alpha)

        def f(alpha):
            return surrogate.Arctan(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestNonzeroSignLog:
    """Test nonzero sign log surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_nonzero_sign_log_forward_backward(self, x, alpha):
        """Test NonzeroSignLog backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.NonzeroSignLog(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_nonzero_sign_log_functional_api(self, x, alpha):
        """Test nonzero sign log functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.NonzeroSignLog(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.nonzero_sign_log)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.nonzero_sign_log, 1)(x, alpha)

        def f(alpha):
            return surrogate.NonzeroSignLog(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestERF:
    """Test ERF surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_erf_forward_backward(self, x, alpha):
        """Test ERF backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.ERF(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_erf_functional_api(self, x, alpha):
        """Test ERF functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.ERF(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.erf)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.erf, 1)(x, alpha)

        def f(alpha):
            return surrogate.ERF(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestQPseudoSpike:
    """Test QPseudoSpike surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [1.0, 2.0, 4.0])
    def test_q_pseudo_spike_forward_backward(self, x, alpha):
        """Test QPseudoSpike backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.QPseudoSpike(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0])
    def test_q_pseudo_spike_functional_api(self, x, alpha):
        """Test QPseudoSpike functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.QPseudoSpike(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.q_pseudo_spike)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.q_pseudo_spike, 1)(x, alpha)

        def f(alpha):
            return surrogate.QPseudoSpike(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestLogTailedRelu:
    """Test LogTailedRelu surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
    def test_log_tailed_relu_forward_backward(self, x, alpha):
        """Test LogTailedRelu backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.LogTailedRelu(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_log_tailed_relu_functional_api(self, x, alpha):
        """Test LogTailedRelu functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.LogTailedRelu(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.log_tailed_relu)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.log_tailed_relu, 1)(x, alpha)

        def f(alpha):
            return surrogate.LogTailedRelu(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestInvSquareGrad:
    """Test InvSquareGrad surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [50.0, 100.0, 200.0])
    def test_inv_square_grad_forward_backward(self, x, alpha):
        """Test InvSquareGrad backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.InvSquareGrad(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [50.0, 100.0, 200.0])
    def test_inv_square_grad_functional_api(self, x, alpha):
        """Test InvSquareGrad functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.InvSquareGrad(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.inv_square_grad)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [50.0, 100.0, 200.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.inv_square_grad, 1)(x, alpha)

        def f(alpha):
            return surrogate.InvSquareGrad(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestSlayerGrad:
    """Test SlayerGrad surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.5, 1.0, 2.0])
    def test_slayer_grad_forward_backward(self, x, alpha):
        """Test SlayerGrad backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.SlayerGrad(alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_slayer_grad_functional_api(self, x, alpha):
        """Test SlayerGrad functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.SlayerGrad(alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.slayer_grad)(x, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_grad_of_parameters(self, x, alpha):
        """Test gradient of parameters."""
        x = jnp.array(x)
        grad = _surrogate_grad_param_grad(surrogate.slayer_grad, 1)(x, alpha)

        def f(alpha):
            return surrogate.SlayerGrad(alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestPiecewiseLeakyRelu:
    """Test PiecewiseLeakyRelu surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('c', [0.01, 0.05, 0.1])
    @pytest.mark.parametrize('w', [0.5, 1.0, 2.0])
    def test_piecewise_leaky_relu_forward_backward(self, x, c, w):
        """Test PiecewiseLeakyRelu backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.PiecewiseLeakyRelu(c=c, w=w)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("c", [0.01, 0.05])
    @pytest.mark.parametrize("w", [0.5, 1.0])
    def test_piecewise_leaky_relu_functional_api(self, x, c, w):
        """Test PiecewiseLeakyRelu functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.PiecewiseLeakyRelu(c=c, w=w))(x)
        y_func = brainstate.transform.vector_grad(surrogate.piecewise_leaky_relu)(x, c=c, w=w)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_c(self, x):
        """Test gradient of c parameter."""
        x = jnp.array(x)
        c, w = 0.01, 1.0
        grad = _surrogate_grad_param_grad(surrogate.piecewise_leaky_relu, 1)(x, c, w)

        def f(c):
            return surrogate.PiecewiseLeakyRelu(c=c, w=w).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(c)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_w(self, x):
        """Test gradient of w parameter."""
        x = jnp.array(x)
        c, w = 0.01, 1.0
        grad = _surrogate_grad_param_grad(surrogate.piecewise_leaky_relu, 2)(x, c, w)

        def f(w):
            return surrogate.PiecewiseLeakyRelu(c=c, w=w).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(w)
        assert jnp.allclose(grad, expected)


class TestLeakyRelu:
    """Test LeakyRelu surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.05, 0.1, 0.2])
    @pytest.mark.parametrize('beta', [0.5, 1.0, 2.0])
    def test_leaky_relu_forward_backward(self, x, alpha, beta):
        """Test LeakyRelu backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.LeakyRelu(alpha=alpha, beta=beta)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.05, 0.1])
    @pytest.mark.parametrize("beta", [0.5, 1.0])
    def test_leaky_relu_functional_api(self, x, alpha, beta):
        """Test LeakyRelu functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.LeakyRelu(alpha=alpha, beta=beta))(x)
        y_func = brainstate.transform.vector_grad(surrogate.leaky_relu)(x, alpha=alpha, beta=beta)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_alpha(self, x):
        """Test gradient of alpha parameter."""
        x = jnp.array(x)
        alpha, beta = 0.1, 1.0
        grad = _surrogate_grad_param_grad(surrogate.leaky_relu, 1)(x, alpha, beta)

        def f(alpha):
            return surrogate.LeakyRelu(alpha=alpha, beta=beta).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_beta(self, x):
        """Test gradient of beta parameter."""
        x = jnp.array(x)
        alpha, beta = 0.1, 1.0
        grad = _surrogate_grad_param_grad(surrogate.leaky_relu, 2)(x, alpha, beta)

        def f(beta):
            return surrogate.LeakyRelu(alpha=alpha, beta=beta).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(beta)
        assert jnp.allclose(grad, expected)


class TestReluGrad:
    """Test ReluGrad surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [0.2, 0.3, 0.5])
    @pytest.mark.parametrize('width', [0.5, 1.0, 2.0])
    def test_relu_grad_forward_backward(self, x, alpha, width):
        """Test ReluGrad backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.ReluGrad(alpha=alpha, width=width)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [0.2, 0.3])
    @pytest.mark.parametrize("width", [0.5, 1.0])
    def test_relu_grad_functional_api(self, x, alpha, width):
        """Test ReluGrad functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.ReluGrad(alpha=alpha, width=width))(x)
        y_func = brainstate.transform.vector_grad(surrogate.relu_grad)(x, alpha=alpha, width=width)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_alpha(self, x):
        """Test gradient of alpha parameter."""
        x = jnp.array(x)
        alpha, width = 0.3, 1.0
        grad = _surrogate_grad_param_grad(surrogate.relu_grad, 1)(x, alpha, width)

        def f(alpha):
            return surrogate.ReluGrad(alpha=alpha, width=width).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_width(self, x):
        """Test gradient of width parameter."""
        x = jnp.array(x)
        alpha, width = 0.3, 1.0
        grad = _surrogate_grad_param_grad(surrogate.relu_grad, 2)(x, alpha, width)

        def f(width):
            return surrogate.ReluGrad(alpha=alpha, width=width).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(width)
        assert jnp.allclose(grad, expected)


class TestGaussianGrad:
    """Test GaussianGrad surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('sigma', [0.3, 0.5, 0.7])
    @pytest.mark.parametrize('alpha', [0.3, 0.5, 0.7])
    def test_gaussian_grad_forward_backward(self, x, sigma, alpha):
        """Test GaussianGrad backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.GaussianGrad(sigma=sigma, alpha=alpha)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("sigma", [0.3, 0.5])
    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    def test_gaussian_grad_functional_api(self, x, sigma, alpha):
        """Test GaussianGrad functional API matches class API."""
        x = jnp.array(x)
        y_class = brainstate.transform.vector_grad(surrogate.GaussianGrad(sigma=sigma, alpha=alpha))(x)
        y_func = brainstate.transform.vector_grad(surrogate.gaussian_grad)(x, sigma=sigma, alpha=alpha)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_sigma(self, x):
        """Test gradient of sigma parameter."""
        x = jnp.array(x)
        sigma, alpha = 0.5, 0.5
        grad = _surrogate_grad_param_grad(surrogate.gaussian_grad, 1)(x, sigma, alpha)

        def f(sigma):
            return surrogate.GaussianGrad(sigma=sigma, alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(sigma)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_alpha(self, x):
        """Test gradient of alpha parameter."""
        x = jnp.array(x)
        sigma, alpha = 0.5, 0.5
        grad = _surrogate_grad_param_grad(surrogate.gaussian_grad, 2)(x, sigma, alpha)

        def f(alpha):
            return surrogate.GaussianGrad(sigma=sigma, alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)


class TestS2NN:
    """Test S2NN surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('alpha', [2.0, 4.0])
    @pytest.mark.parametrize('beta', [0.5, 1.0])
    @pytest.mark.parametrize('epsilon', [1e-8, 1e-7])
    def test_s2nn_forward_backward(self, x, alpha, beta, epsilon):
        """Test S2NN backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.S2NN(alpha=alpha, beta=beta, epsilon=epsilon)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("alpha", [2.0, 4.0])
    @pytest.mark.parametrize("beta", [0.5, 1.0])
    def test_s2nn_functional_api(self, x, alpha, beta):
        """Test S2NN functional API matches class API."""
        x = jnp.array(x)
        epsilon = 1e-8
        y_class = brainstate.transform.vector_grad(surrogate.S2NN(alpha=alpha, beta=beta, epsilon=epsilon))(x)
        y_func = brainstate.transform.vector_grad(surrogate.s2nn)(x, alpha=alpha, beta=beta, epsilon=epsilon)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_alpha(self, x):
        """Test gradient of alpha parameter."""
        x = jnp.array(x)
        alpha, beta, epsilon = 4.0, 1.0, 1e-8
        grad = _surrogate_grad_param_grad(surrogate.s2nn, 1)(x, alpha, beta, epsilon)

        def f(alpha):
            return surrogate.S2NN(alpha=alpha, beta=beta, epsilon=epsilon).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [0.0, 1.0])
    def test_grad_of_parameters_beta(self, x):
        """Test gradient of beta parameter."""
        x = jnp.array(x)
        alpha, beta, epsilon = 4.0, 1.0, 1e-8
        grad = _surrogate_grad_param_grad(surrogate.s2nn, 2)(x, alpha, beta, epsilon)

        def f(beta):
            return surrogate.S2NN(alpha=alpha, beta=beta, epsilon=epsilon).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(beta)
        assert jnp.allclose(grad, expected)


class TestMultiGaussianGrad:
    """Test MultiGaussianGrad surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('h', [0.1, 0.15])
    @pytest.mark.parametrize('s', [5.0, 6.0])
    def test_multi_gaussian_grad_forward_backward(self, x, h, s):
        """Test MultiGaussianGrad backward pass computes gradients."""
        x = jnp.array(x)
        sigma, scale = 0.5, 0.5

        sg = surrogate.MultiGaussianGrad(h=h, s=s, sigma=sigma, scale=scale)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_multi_gaussian_grad_functional_api(self, x):
        """Test MultiGaussianGrad functional API matches class API."""
        x = jnp.array(x)
        h, s, sigma, scale = 0.15, 6.0, 0.5, 0.5
        y_class = brainstate.transform.vector_grad(surrogate.MultiGaussianGrad(h=h, s=s, sigma=sigma, scale=scale))(x)
        y_func = brainstate.transform.vector_grad(surrogate.multi_gaussian_grad)(x, h=h, s=s, sigma=sigma, scale=scale)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_h(self, x):
        """Test gradient of h parameter."""
        x = jnp.array(x)
        h, s, sigma, scale = 0.15, 6.0, 0.5, 0.5
        grad = _surrogate_grad_param_grad(surrogate.multi_gaussian_grad, 1)(x, h, s, sigma, scale)

        def f(h):
            return surrogate.MultiGaussianGrad(h=h, s=s, sigma=sigma, scale=scale).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(h)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_s(self, x):
        """Test gradient of s parameter."""
        x = jnp.array(x)
        h, s, sigma, scale = 0.15, 6.0, 0.5, 0.5
        grad = _surrogate_grad_param_grad(surrogate.multi_gaussian_grad, 2)(x, h, s, sigma, scale)

        def f(s):
            return surrogate.MultiGaussianGrad(h=h, s=s, sigma=sigma, scale=scale).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(s)
        assert jnp.allclose(grad, expected)


class TestSquarewaveFourierSeries:
    """Test SquarewaveFourierSeries surrogate gradients."""

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize('n', [1, 2, 3])
    @pytest.mark.parametrize('t_period', [6.0, 8.0, 10.0])
    def test_squarewave_fourier_series_forward_backward(self, x, n, t_period):
        """Test SquarewaveFourierSeries backward pass computes gradients."""
        x = jnp.array(x)

        sg = surrogate.SquarewaveFourierSeries(n=n, t_period=t_period)
        y = sg(x)
        # Forward should be Heaviside step function
        forward_expected = jnp.array(x >= 0.0, dtype=float)
        assert jnp.allclose(y, forward_expected)

        grad = brainstate.transform.vector_grad(sg)(x)
        expected = sg.surrogate_grad(x)
        assert jnp.allclose(grad, expected)


# ==============================================================================
# Correctness tests. Unlike the plumbing tests above (backprop reproduces
# ``surrogate_grad``), these check the surrogate *formulas* themselves against
# their documented closed-form references and the function/derivative
# consistency ``d/dx surrogate_fun == surrogate_grad``.
# ==============================================================================


# ---------------------------------------------------------------------------
# 1. Function / derivative consistency: d/dx surrogate_fun == surrogate_grad.
#    Catches the Arctan, ERF, PiecewiseQuadratic, PiecewiseLeakyRelu and
#    QPseudoSpike fun/grad mismatches.
# ---------------------------------------------------------------------------

# (label, surrogate-with-surrogate_fun). Only surrogates that implement
# ``surrogate_fun`` are included.
_FUN_SURROGATES = [
    ('Sigmoid', surrogate.Sigmoid(alpha=4.0)),
    ('Sigmoid(a=1)', surrogate.Sigmoid(alpha=1.0)),
    ('PiecewiseQuadratic(a=0.5)', surrogate.PiecewiseQuadratic(alpha=0.5)),
    ('PiecewiseQuadratic(a=1)', surrogate.PiecewiseQuadratic(alpha=1.0)),
    ('PiecewiseQuadratic(a=2)', surrogate.PiecewiseQuadratic(alpha=2.0)),
    ('PiecewiseExp', surrogate.PiecewiseExp(alpha=1.0)),
    ('PiecewiseExp(a=2)', surrogate.PiecewiseExp(alpha=2.0)),
    ('SoftSign', surrogate.SoftSign(alpha=2.0)),
    ('Arctan(a=0.5)', surrogate.Arctan(alpha=0.5)),
    ('Arctan(a=1)', surrogate.Arctan(alpha=1.0)),
    ('Arctan(a=2)', surrogate.Arctan(alpha=2.0)),
    ('NonzeroSignLog', surrogate.NonzeroSignLog(alpha=1.0)),
    ('ERF(a=0.5)', surrogate.ERF(alpha=0.5)),
    ('ERF(a=1)', surrogate.ERF(alpha=1.0)),
    ('ERF(a=2)', surrogate.ERF(alpha=2.0)),
    ('PiecewiseLeakyRelu', surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0)),
    ('PiecewiseLeakyRelu(w=2)', surrogate.PiecewiseLeakyRelu(c=0.05, w=2.0)),
    ('SquarewaveFourierSeries', surrogate.SquarewaveFourierSeries(n=2, t_period=8.0)),
    ('SquarewaveFourierSeries(n=3)', surrogate.SquarewaveFourierSeries(n=3, t_period=6.0)),
    ('S2NN', surrogate.S2NN(alpha=4.0, beta=1.0)),
    ('QPseudoSpike(a=0.5)', surrogate.QPseudoSpike(alpha=0.5)),
    ('QPseudoSpike(a=1)', surrogate.QPseudoSpike(alpha=1.0)),
    ('QPseudoSpike(a=2)', surrogate.QPseudoSpike(alpha=2.0)),
    ('QPseudoSpike(a=3)', surrogate.QPseudoSpike(alpha=3.0)),
    ('LeakyRelu', surrogate.LeakyRelu(alpha=0.1, beta=1.0)),
    ('LogTailedRelu', surrogate.LogTailedRelu(alpha=0.1)),
]


@pytest.mark.parametrize("label, sg", _FUN_SURROGATES, ids=[s[0] for s in _FUN_SURROGATES])
def test_surrogate_fun_derivative_matches_grad(label, sg):
    # Avoid the exact piecewise break-points where the autodiff of a kink is
    # ambiguous (e.g. |x| = 1/alpha, |x| = w, x in {0, 1}).
    xs = jnp.linspace(-1.7, 1.7, 69) + 1e-3
    auto = jax.vmap(jax.grad(sg.surrogate_fun))(xs)
    manual = sg.surrogate_grad(xs)
    assert jnp.all(jnp.isfinite(auto))
    assert jnp.allclose(auto, manual, atol=1e-4, rtol=1e-4), label


# ---------------------------------------------------------------------------
# 2. Shape of the smooth surrogate function: a valid soft-Heaviside passes
#    through 0.5 at the threshold and is monotonically increasing.  Catches the
#    ERF (decreasing) and Arctan (out-of-range) bugs.
# ---------------------------------------------------------------------------

# CDF-like surrogates: smooth Heaviside passing through (0, 0.5).
_HALF_AT_THRESHOLD = [
    surrogate.Sigmoid(alpha=4.0),
    surrogate.PiecewiseExp(alpha=1.0),
    surrogate.SoftSign(alpha=2.0),
    surrogate.Arctan(alpha=1.0),
    surrogate.ERF(alpha=1.0),
    surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0),
]


# Monotonically increasing surrogate functions (CDF-like plus the relu-like
# LeakyRelu, which passes through the origin rather than (0, 0.5)).
_MONOTONE = _HALF_AT_THRESHOLD + [surrogate.LeakyRelu(alpha=0.1, beta=1.0)]


@pytest.mark.parametrize("sg", _HALF_AT_THRESHOLD, ids=lambda s: type(s).__name__)
def test_surrogate_fun_half_at_threshold(sg):
    assert np.allclose(float(sg.surrogate_fun(jnp.array(0.0))), 0.5, atol=1e-6)


@pytest.mark.parametrize("sg", _MONOTONE, ids=lambda s: type(s).__name__)
def test_surrogate_fun_monotone_increasing(sg):
    xs = jnp.linspace(-3, 3, 121)
    z = sg.surrogate_fun(xs)
    assert bool(jnp.all(jnp.diff(z) >= -1e-6))


@pytest.mark.parametrize(
    "sg",
    [surrogate.Sigmoid(alpha=4.0), surrogate.Arctan(alpha=1.0), surrogate.ERF(alpha=1.0)],
    ids=lambda s: type(s).__name__,
)
def test_bounded_surrogate_fun_in_unit_interval(sg):
    # Sigmoid/Arctan/ERF smooth functions are CDFs: they must stay within [0, 1].
    xs = jnp.linspace(-50, 50, 401)
    z = sg.surrogate_fun(xs)
    assert float(jnp.min(z)) >= -1e-6
    assert float(jnp.max(z)) <= 1.0 + 1e-6


def test_arctan_fun_not_arctan2_regression():
    # Regression for the arctan2 misuse: the value at x=2, alpha=1 must be
    # arctan(pi)/pi + 0.5, NOT arctan(1.0) + 0.5.
    sg = surrogate.Arctan(alpha=1.0)
    got = float(sg.surrogate_fun(jnp.array(2.0)))
    expected = float(jnp.arctan(jnp.pi) / jnp.pi + 0.5)
    assert np.allclose(got, expected, atol=1e-6)
    assert got <= 1.0  # the buggy version returned ~1.6


# ---------------------------------------------------------------------------
# 3. Reference gradients: surrogate_grad against independent closed forms.
# ---------------------------------------------------------------------------

XS = np.linspace(-2.5, 2.5, 51)


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_sigmoid_grad_reference():
    a = 4.0
    ref = a * _expit(a * XS) * (1 - _expit(a * XS))
    got = surrogate.Sigmoid(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_piecewise_exp_grad_reference():
    a = 1.5
    ref = (a / 2) * np.exp(-a * np.abs(XS))
    got = surrogate.PiecewiseExp(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_soft_sign_grad_reference():
    a = 2.0
    ref = a * 0.5 / (1 + np.abs(a * XS)) ** 2
    got = surrogate.SoftSign(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_arctan_grad_reference():
    a = 2.0
    ref = a * 0.5 / (1 + (np.pi / 2 * a * XS) ** 2)
    got = surrogate.Arctan(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_nonzero_sign_log_grad_reference():
    a = 1.0
    ref = 1.0 / (1 / a + np.abs(XS))
    got = surrogate.NonzeroSignLog(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_erf_grad_reference():
    a = 1.5
    ref = (a / np.sqrt(np.pi)) * np.exp(-(a ** 2) * XS ** 2)
    got = surrogate.ERF(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


@pytest.mark.parametrize("sigma", [0.3, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_gaussian_grad_reference(sigma, alpha):
    # Standard Gaussian PDF scaled by alpha (NOT the buggy exp(-x^2 * sigma^2 / 2)).
    ref = alpha / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(XS ** 2) / (2 * sigma ** 2))
    got = surrogate.GaussianGrad(sigma=sigma, alpha=alpha).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_gaussian_grad_peak_decreases_with_sigma():
    # Regression for the inverted-sigma bug: wider sigma => lower, broader peak.
    peaks = [float(surrogate.GaussianGrad(sigma=s, alpha=1.0).surrogate_grad(jnp.array(0.0)))
             for s in (0.3, 0.5, 1.0, 2.0)]
    assert peaks == sorted(peaks, reverse=True)


def test_gaussian_grad_matches_multi_gaussian_central_component():
    # MultiGaussianGrad with h=0 reduces to scale * central Gaussian; with
    # scale == alpha it must equal GaussianGrad exactly.
    sigma = 0.5
    gg = surrogate.GaussianGrad(sigma=sigma, alpha=0.7).surrogate_grad(jnp.asarray(XS))
    mg = surrogate.MultiGaussianGrad(h=0.0, s=6.0, sigma=sigma, scale=0.7).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(gg), np.asarray(mg), atol=1e-6)


def test_inv_square_grad_reference():
    a = 100.0
    ref = 1.0 / (a * np.abs(XS) + 1.0) ** 2
    got = surrogate.InvSquareGrad(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_slayer_grad_reference():
    a = 1.0
    ref = np.exp(-a * np.abs(XS))
    got = surrogate.SlayerGrad(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_leaky_relu_grad_reference():
    a, b = 0.1, 1.0
    ref = np.where(XS < 0, a, b)
    got = surrogate.LeakyRelu(alpha=a, beta=b).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-6)


def test_relu_grad_is_triangle():
    a, w = 0.3, 1.0
    ref = np.maximum(a * w - np.abs(XS) * a, 0.0)
    got = surrogate.ReluGrad(alpha=a, width=w).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-6)
    # Peak alpha*width at 0, zero beyond width.
    assert np.allclose(float(surrogate.ReluGrad(alpha=a, width=w).surrogate_grad(jnp.array(0.0))), a * w)
    assert float(surrogate.ReluGrad(alpha=a, width=w).surrogate_grad(jnp.array(2.0))) == 0.0


def test_q_pseudo_spike_grad_reference():
    a = 2.0
    ref = (1 + 2 / (a + 1) * np.abs(XS)) ** (-a)
    got = surrogate.QPseudoSpike(alpha=a).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


def test_s2nn_grad_reference():
    a, b = 4.0, 1.0
    sg = _expit(a * XS)
    x_safe = np.where(XS < 0, 1.0, XS)  # mirror the implementation's guarded denominator
    ref = np.where(XS < 0, a * sg * (1 - sg), b / (x_safe + 1.0))
    got = surrogate.S2NN(alpha=a, beta=b).surrogate_grad(jnp.asarray(XS))
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. PiecewiseQuadratic gradient is a *continuous triangle* (not a parabola).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 4.0])
def test_piecewise_quadratic_grad_triangle_and_continuous(alpha):
    xs = np.linspace(-3, 3, 601)
    ref = np.where(np.abs(xs) > 1 / alpha, 0.0, alpha - alpha ** 2 * np.abs(xs))
    got = np.asarray(surrogate.PiecewiseQuadratic(alpha=alpha).surrogate_grad(jnp.asarray(xs)))
    assert np.allclose(got, ref, atol=1e-5)
    # Continuity: the gradient should approach 0 at the window edge |x| = 1/alpha,
    # i.e. no jump (the old parabola jumped to alpha-1 there for alpha != 1).
    edge = 1.0 / alpha
    inside = float(surrogate.PiecewiseQuadratic(alpha=alpha).surrogate_grad(jnp.array(edge - 1e-4)))
    assert abs(inside) < 1e-2


# ---------------------------------------------------------------------------
# 5. PiecewiseLeakyRelu central slope equals 1/(2w) (matches surrogate_fun).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("c", [0.01, 0.05])
def test_piecewise_leaky_relu_central_slope(w, c):
    sg = surrogate.PiecewiseLeakyRelu(c=c, w=w)
    # Inside the window: 1/(2w).
    inside = float(sg.surrogate_grad(jnp.array(0.3 * w)))
    assert np.allclose(inside, 1.0 / (2 * w), atol=1e-6)
    # Outside the window: the leak c.
    outside = float(sg.surrogate_grad(jnp.array(2.0 * w + 1.0)))
    assert np.allclose(outside, c, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. SquarewaveFourierSeries actually sums ``n`` harmonics (off-by-one fix).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("t_period", [6.0, 8.0])
def test_squarewave_term_count(n, t_period):
    xs = np.linspace(-3, 3, 121)
    w = 2 * np.pi / t_period
    # Reference: sum_{i=1}^{n} cos((2i-1) w x), scaled by 4/T.
    ref = np.zeros_like(xs)
    for i in range(1, n + 1):
        ref += np.cos((2 * i - 1.0) * w * xs)
    ref *= 4.0 / t_period
    got = np.asarray(surrogate.SquarewaveFourierSeries(n=n, t_period=t_period).surrogate_grad(jnp.asarray(xs)))
    assert np.allclose(got, ref, atol=1e-5)


def test_squarewave_n2_differs_from_n1():
    # The default n=2 must include a second harmonic the n=1 series lacks.
    xs = jnp.linspace(-3, 3, 101)
    g1 = surrogate.SquarewaveFourierSeries(n=1, t_period=8.0).surrogate_grad(xs)
    g2 = surrogate.SquarewaveFourierSeries(n=2, t_period=8.0).surrogate_grad(xs)
    assert not bool(jnp.allclose(g1, g2))


# ---------------------------------------------------------------------------
# 7. Numerical robustness: finite gradients at the dangerous break-points,
#    including parameter gradients (which exercise the dead where-branch).
# ---------------------------------------------------------------------------

def test_s2nn_grad_finite_at_minus_one():
    # x = -1 makes the dead beta/(x+1) branch infinite without the guard.
    sg = surrogate.S2NN(alpha=4.0, beta=1.0)
    g = sg.surrogate_grad(jnp.array([-2.0, -1.0, -0.5, 0.0, 1.0]))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_s2nn_param_grad_finite_at_minus_one():
    x = jnp.array(-1.0)
    import brainstate.transform
    g_beta = _surrogate_grad_param_grad(surrogate.s2nn, 2)(x, 4.0, 1.0, 1e-8)
    g_alpha = _surrogate_grad_param_grad(surrogate.s2nn, 1)(x, 4.0, 1.0, 1e-8)
    assert jnp.isfinite(g_beta)
    assert jnp.isfinite(g_alpha)


def test_log_tailed_relu_grad_finite_at_zero():
    sg = surrogate.LogTailedRelu(alpha=0.1)
    g = sg.surrogate_grad(jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_log_tailed_relu_param_grad_finite_at_zero():
    x = jnp.array(0.0)
    import brainstate.transform
    g = _surrogate_grad_param_grad(surrogate.log_tailed_relu, 1)(x, 0.1)
    assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# 7b. LogTailedRelu.surrogate_fun is a *continuous, monotone* original function.
#     Regression for the x>1 branch returning log(x) (a jump down to 0 at x=1)
#     instead of the continuous 1 + log(x) (Cai et al. 2017).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.3])
def test_log_tailed_relu_fun_continuous_at_one(alpha):
    sg = surrogate.LogTailedRelu(alpha=alpha)
    # Approach the x=1 break-point from both sides: no jump.
    left = float(sg.surrogate_fun(jnp.array(1.0 - 1e-5)))
    at = float(sg.surrogate_fun(jnp.array(1.0)))
    right = float(sg.surrogate_fun(jnp.array(1.0 + 1e-5)))
    assert np.allclose(left, 1.0, atol=1e-4)
    assert np.allclose(at, 1.0, atol=1e-6)
    assert np.allclose(right, 1.0, atol=1e-4)
    # The old buggy log(x) branch jumped to ~0 just past x=1.
    assert right > 0.5


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.3])
def test_log_tailed_relu_fun_monotone_and_finite(alpha):
    sg = surrogate.LogTailedRelu(alpha=alpha)
    xs = jnp.linspace(-2.0, 4.0, 241)
    z = sg.surrogate_fun(xs)
    assert bool(jnp.all(jnp.isfinite(z)))
    assert bool(jnp.all(jnp.diff(z) >= -1e-6))  # monotone non-decreasing


def test_log_tailed_relu_fun_derivative_matches_grad_in_tail():
    # d/dx (1 + log(x)) == 1/x == surrogate_grad for x > 1.
    sg = surrogate.LogTailedRelu(alpha=0.1)
    xs = jnp.linspace(1.05, 3.0, 40)
    auto = jax.vmap(jax.grad(sg.surrogate_fun))(xs)
    assert jnp.allclose(auto, sg.surrogate_grad(xs), atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Forward pass is the exact Heaviside step for every public surrogate.
# ---------------------------------------------------------------------------

_ALL_SURROGATES = [
    surrogate.Sigmoid(), surrogate.PiecewiseQuadratic(), surrogate.PiecewiseExp(),
    surrogate.SoftSign(), surrogate.Arctan(), surrogate.NonzeroSignLog(),
    surrogate.ERF(), surrogate.PiecewiseLeakyRelu(), surrogate.SquarewaveFourierSeries(),
    surrogate.S2NN(), surrogate.QPseudoSpike(), surrogate.LeakyRelu(),
    surrogate.LogTailedRelu(), surrogate.ReluGrad(), surrogate.GaussianGrad(),
    surrogate.InvSquareGrad(), surrogate.MultiGaussianGrad(), surrogate.SlayerGrad(),
]


@pytest.mark.parametrize("sg", _ALL_SURROGATES, ids=lambda s: type(s).__name__)
def test_forward_is_exact_heaviside(sg):
    xs = jnp.array([-3.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 3.0])
    y = sg(xs)
    expected = jnp.asarray(xs >= 0, dtype=xs.dtype)
    assert jnp.allclose(y, expected)


@pytest.mark.parametrize("sg", _ALL_SURROGATES, ids=lambda s: type(s).__name__)
def test_all_grads_finite(sg):
    import brainstate.transform
    xs = jnp.linspace(-2, 2, 41)
    g = brainstate.transform.vector_grad(sg)(xs)
    assert bool(jnp.all(jnp.isfinite(g)))


# ==============================================================================
# Extra coverage: every ``__repr__``/``__hash__``, the smooth ``surrogate_fun``
# methods, the ``NotImplementedError`` of the gradient-only surrogates, and the
# remaining functional-API / parameter-gradient paths of SquarewaveFourierSeries.
# ==============================================================================


# ---------------------------------------------------------------------------
# __repr__ for every class.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sg, expected",
    [
        (surrogate.Sigmoid(alpha=4.0), 'Sigmoid(alpha=4.0)'),
        (surrogate.PiecewiseQuadratic(alpha=1.0), 'PiecewiseQuadratic(alpha=1.0)'),
        (surrogate.PiecewiseExp(alpha=1.0), 'PiecewiseExp(alpha=1.0)'),
        (surrogate.SoftSign(alpha=1.0), 'SoftSign(alpha=1.0)'),
        (surrogate.Arctan(alpha=1.0), 'Arctan(alpha=1.0)'),
        (surrogate.NonzeroSignLog(alpha=1.0), 'NonzeroSignLog(alpha=1.0)'),
        (surrogate.ERF(alpha=1.0), 'ERF(alpha=1.0)'),
        (surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0), 'PiecewiseLeakyRelu(c=0.01, w=1.0)'),
        (surrogate.SquarewaveFourierSeries(n=2, t_period=8.0),
         'SquarewaveFourierSeries(n=2, t_period=8.0)'),
        (surrogate.S2NN(alpha=4.0, beta=1.0, epsilon=1e-8),
         'S2NN(alpha=4.0, beta=1.0, epsilon=1e-08)'),
        (surrogate.QPseudoSpike(alpha=2.0), 'QPseudoSpike(alpha=2.0)'),
        (surrogate.LeakyRelu(alpha=0.1, beta=1.0), 'LeakyRelu(alpha=0.1, beta=1.0)'),
        (surrogate.LogTailedRelu(alpha=0.0), 'LogTailedRelu(alpha=0.0)'),
        (surrogate.ReluGrad(alpha=0.3, width=1.0), 'ReluGrad(alpha=0.3, width=1.0)'),
        (surrogate.GaussianGrad(sigma=0.5, alpha=0.5), 'GaussianGrad(alpha=0.5, sigma=0.5)'),
        (surrogate.MultiGaussianGrad(h=0.15, s=6.0, sigma=0.5, scale=0.5),
         'MultiGaussianGrad(h=0.15, s=6.0, sigma=0.5, scale=0.5)'),
        (surrogate.InvSquareGrad(alpha=100.0), 'InvSquareGrad(alpha=100.0)'),
        (surrogate.SlayerGrad(alpha=1.0), 'SlayerGrad(alpha=1.0)'),
    ],
)
def test_repr(sg, expected):
    assert repr(sg) == expected


# ---------------------------------------------------------------------------
# __hash__ for every class (stable, and sensitive to parameters).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "make",
    [
        lambda a: surrogate.Sigmoid(alpha=a),
        lambda a: surrogate.PiecewiseQuadratic(alpha=a),
        lambda a: surrogate.PiecewiseExp(alpha=a),
        lambda a: surrogate.SoftSign(alpha=a),
        lambda a: surrogate.Arctan(alpha=a),
        lambda a: surrogate.NonzeroSignLog(alpha=a),
        lambda a: surrogate.ERF(alpha=a),
        lambda a: surrogate.PiecewiseLeakyRelu(c=a),
        lambda a: surrogate.SquarewaveFourierSeries(n=int(a) + 1),
        lambda a: surrogate.S2NN(alpha=a),
        lambda a: surrogate.QPseudoSpike(alpha=a),
        lambda a: surrogate.LeakyRelu(alpha=a),
        lambda a: surrogate.LogTailedRelu(alpha=a),
        lambda a: surrogate.ReluGrad(alpha=a),
        lambda a: surrogate.GaussianGrad(alpha=a),
        lambda a: surrogate.MultiGaussianGrad(h=a),
        lambda a: surrogate.InvSquareGrad(alpha=a),
        lambda a: surrogate.SlayerGrad(alpha=a),
    ],
)
def test_hash_stable_and_param_sensitive(make):
    assert hash(make(1.0)) == hash(make(1.0))
    assert hash(make(1.0)) != hash(make(2.0))


# ---------------------------------------------------------------------------
# surrogate_fun for classes that define it.  The smooth function equals 0.5 at
# the threshold for the symmetric surrogates, and must be finite over a range.
# ---------------------------------------------------------------------------

class TestSurrogateFunThresholdHalf:
    """Surrogates whose smooth function passes through 0.5 at x=0."""

    @pytest.mark.parametrize(
        "sg",
        [
            surrogate.Sigmoid(alpha=4.0),
            surrogate.PiecewiseQuadratic(alpha=1.0),
            surrogate.PiecewiseExp(alpha=1.0),
            surrogate.SoftSign(alpha=2.0),
            surrogate.Arctan(alpha=1.0),
            surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0),
            surrogate.SquarewaveFourierSeries(n=2, t_period=8.0),
            surrogate.S2NN(alpha=4.0, beta=1.0),
            surrogate.QPseudoSpike(alpha=2.0),
        ],
    )
    def test_fun_at_zero_is_half(self, sg):
        assert np.allclose(float(sg.surrogate_fun(jnp.array(0.0))), 0.5, atol=1e-6)

    @pytest.mark.parametrize(
        "sg",
        [
            surrogate.Sigmoid(alpha=4.0),
            surrogate.PiecewiseQuadratic(alpha=1.0),
            surrogate.PiecewiseExp(alpha=1.0),
            surrogate.SoftSign(alpha=2.0),
            surrogate.Arctan(alpha=1.0),
            surrogate.PiecewiseLeakyRelu(c=0.01, w=1.0),
            surrogate.SquarewaveFourierSeries(n=3, t_period=8.0),
            surrogate.S2NN(alpha=4.0, beta=1.0),
            surrogate.LeakyRelu(alpha=0.1, beta=1.0),
        ],
    )
    def test_fun_is_finite_over_range(self, sg):
        xs = jnp.linspace(-2.5, 2.5, 51)
        z = sg.surrogate_fun(xs)
        assert z.shape == xs.shape
        assert bool(jnp.all(jnp.isfinite(z)))


class TestSurrogateFunSpecificValues:
    """surrogate_fun anchor values for the asymmetric surrogates."""

    def test_nonzero_sign_log_at_zero(self):
        # log(|alpha*0| + 1) == 0
        sg = surrogate.NonzeroSignLog(alpha=1.0)
        assert np.allclose(float(sg.surrogate_fun(jnp.array(0.0))), 0.0)

    def test_nonzero_sign_log_sign(self):
        sg = surrogate.NonzeroSignLog(alpha=1.0)
        z = sg.surrogate_fun(jnp.array([-1.0, 1.0]))
        # symmetric magnitude, opposite sign
        assert float(z[0]) < 0.0 < float(z[1])
        assert np.allclose(float(z[0]), -float(z[1]))

    def test_erf_at_zero(self):
        # ERF.surrogate_fun is a proper [0, 1] smooth Heaviside: 0.5 * (1 - erf(-a*x)),
        # which equals 0.5 at the threshold (NOT 0.0).
        sg = surrogate.ERF(alpha=1.0)
        assert np.allclose(float(sg.surrogate_fun(jnp.array(0.0))), 0.5, atol=1e-7)

    def test_erf_finite(self):
        sg = surrogate.ERF(alpha=1.5)
        z = sg.surrogate_fun(jnp.linspace(-2, 2, 21))
        assert bool(jnp.all(jnp.isfinite(z)))

    def test_erf_is_increasing_in_unit_range(self):
        # The fixed ERF surrogate function increases monotonically within [0, 1].
        sg = surrogate.ERF(alpha=1.0)
        xs = jnp.linspace(-3, 3, 51)
        z = sg.surrogate_fun(xs)
        assert bool(jnp.all(jnp.diff(z) > 0))
        assert float(jnp.min(z)) >= 0.0 and float(jnp.max(z)) <= 1.0

    def test_leaky_relu_fun(self):
        sg = surrogate.LeakyRelu(alpha=0.1, beta=1.0)
        z = sg.surrogate_fun(jnp.array([-2.0, 2.0]))
        assert np.allclose(float(z[0]), -0.2)  # alpha * x
        assert np.allclose(float(z[1]), 2.0)  # beta * x

    def test_log_tailed_relu_fun_regimes(self):
        sg = surrogate.LogTailedRelu(alpha=0.1)
        # x <= 0 -> alpha*x ; 0<x<=1 -> x ; x>1 -> 1 + log(x)
        z = sg.surrogate_fun(jnp.array([-2.0, 0.5, jnp.e]))
        assert np.allclose(float(z[0]), -0.2)
        assert np.allclose(float(z[1]), 0.5)
        assert np.allclose(float(z[2]), 2.0, atol=1e-6)  # 1 + log(e) == 2

    def test_s2nn_fun_two_branches(self):
        sg = surrogate.S2NN(alpha=4.0, beta=1.0)
        z = sg.surrogate_fun(jnp.array([-1.0, 1.0]))
        assert bool(jnp.all(jnp.isfinite(z)))
        # x>=0 branch: beta*log(|x+1|+eps)+0.5 ; at x=1 -> log(2)+0.5
        assert np.allclose(float(z[1]), float(jnp.log(2.0)) + 0.5, atol=1e-6)

    def test_q_pseudo_spike_fun_positive(self):
        # The x<0 branch with alpha=2 diverges (power of 0); use positive x.
        sg = surrogate.QPseudoSpike(alpha=2.0)
        z = sg.surrogate_fun(jnp.array([0.5, 1.0, 2.0]))
        assert bool(jnp.all(jnp.isfinite(z)))
        assert bool(jnp.all(z > 0.5))

    def test_q_pseudo_spike_fun_both_branches_finite(self):
        # alpha=3 keeps both branches finite over a small symmetric range.
        sg = surrogate.QPseudoSpike(alpha=3.0)
        z = sg.surrogate_fun(jnp.array([-0.3, 0.3]))
        assert bool(jnp.all(jnp.isfinite(z)))


# ---------------------------------------------------------------------------
# The five gradient-only surrogates do not implement surrogate_fun.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sg",
    [
        surrogate.ReluGrad(),
        surrogate.GaussianGrad(),
        surrogate.MultiGaussianGrad(),
        surrogate.InvSquareGrad(),
        surrogate.SlayerGrad(),
    ],
)
def test_surrogate_fun_not_implemented(sg):
    with pytest.raises(NotImplementedError):
        sg.surrogate_fun(jnp.array(0.5))


# ---------------------------------------------------------------------------
# SquarewaveFourierSeries: functional API, parameter gradient, and the n=1
# branch where the ``for`` loop body never executes.
# ---------------------------------------------------------------------------

class TestSquarewaveFourierSeriesExtra:

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_functional_api(self, x, n):
        # ``n`` indexes a Python ``range`` inside the surrogate, so it must stay
        # a static constant: bind it via a closure and differentiate w.r.t. x.
        x = jnp.array(x)
        t_period = 8.0
        y_class = brainstate.transform.vector_grad(
            surrogate.SquarewaveFourierSeries(n=n, t_period=t_period))(x)
        y_func = brainstate.transform.vector_grad(
            lambda xi: surrogate.squarewave_fourier_series(xi, n=n, t_period=t_period))(x)
        assert jnp.allclose(y_class, y_func)

    @pytest.mark.parametrize("x", [-0.7, 0.0, 0.7])
    @pytest.mark.parametrize("n", [2, 3])
    def test_backward_matches_surrogate_grad(self, x, n):
        x = jnp.array(x)
        t_period = 8.0
        sg = surrogate.SquarewaveFourierSeries(n=n, t_period=t_period)
        grad = brainstate.transform.vector_grad(sg)(x)
        assert jnp.allclose(grad, sg.surrogate_grad(x))

    def test_n_equals_one_no_loop(self):
        # With n=1 the range(2, 1) loop is empty: grad is a single cosine term.
        sg = surrogate.SquarewaveFourierSeries(n=1, t_period=8.0)
        xs = jnp.linspace(-2, 2, 11)
        w = jnp.pi * 2.0 / 8.0
        assert jnp.allclose(sg.surrogate_grad(xs), jnp.cos(w * xs) * 4.0 / 8.0)

    def test_forward_is_step(self):
        sg = surrogate.SquarewaveFourierSeries(n=4, t_period=8.0)
        xs = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = sg(xs)
        assert jnp.allclose(y, jnp.asarray(xs >= 0, dtype=xs.dtype))


# ---------------------------------------------------------------------------
# Functional-API forward equality for the gradient-only surrogates whose
# functional wrappers are otherwise lightly covered.
# ---------------------------------------------------------------------------

class TestGradOnlyFunctionalForward:

    def test_relu_grad_forward(self):
        xs = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert jnp.allclose(surrogate.relu_grad(xs, alpha=0.3, width=1.0),
                            jnp.asarray(xs >= 0, dtype=xs.dtype))

    def test_gaussian_grad_forward(self):
        xs = jnp.array([-1.0, 0.0, 1.0])
        assert jnp.allclose(surrogate.gaussian_grad(xs, sigma=0.5, alpha=0.5),
                            jnp.asarray(xs >= 0, dtype=xs.dtype))

    def test_multi_gaussian_grad_forward(self):
        xs = jnp.array([-1.0, 0.0, 1.0])
        assert jnp.allclose(surrogate.multi_gaussian_grad(xs),
                            jnp.asarray(xs >= 0, dtype=xs.dtype))

    def test_inv_square_grad_forward(self):
        xs = jnp.array([-0.1, 0.0, 0.1])
        assert jnp.allclose(surrogate.inv_square_grad(xs, alpha=100.0),
                            jnp.asarray(xs >= 0, dtype=xs.dtype))

    def test_slayer_grad_forward(self):
        xs = jnp.array([-2.0, 0.0, 2.0])
        assert jnp.allclose(surrogate.slayer_grad(xs, alpha=1.0),
                            jnp.asarray(xs >= 0, dtype=xs.dtype))
