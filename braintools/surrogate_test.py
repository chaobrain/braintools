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
import jax.numpy as jnp
import pytest

import braintools.surrogate as surrogate


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
        grad = brainstate.transform.vector_grad(surrogate.sigmoid, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.piecewise_quadratic, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.piecewise_exp, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.soft_sign, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.arctan, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.nonzero_sign_log, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.erf, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.q_pseudo_spike, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.log_tailed_relu, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.inv_square_grad, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.slayer_grad, argnums=1)(x, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.piecewise_leaky_relu, argnums=1)(x, c, w)

        def f(c):
            return surrogate.PiecewiseLeakyRelu(c=c, w=w).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(c)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_w(self, x):
        """Test gradient of w parameter."""
        x = jnp.array(x)
        c, w = 0.01, 1.0
        grad = brainstate.transform.vector_grad(surrogate.piecewise_leaky_relu, argnums=2)(x, c, w)

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
        grad = brainstate.transform.vector_grad(surrogate.leaky_relu, argnums=1)(x, alpha, beta)

        def f(alpha):
            return surrogate.LeakyRelu(alpha=alpha, beta=beta).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_beta(self, x):
        """Test gradient of beta parameter."""
        x = jnp.array(x)
        alpha, beta = 0.1, 1.0
        grad = brainstate.transform.vector_grad(surrogate.leaky_relu, argnums=2)(x, alpha, beta)

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
        grad = brainstate.transform.vector_grad(surrogate.relu_grad, argnums=1)(x, alpha, width)

        def f(alpha):
            return surrogate.ReluGrad(alpha=alpha, width=width).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_width(self, x):
        """Test gradient of width parameter."""
        x = jnp.array(x)
        alpha, width = 0.3, 1.0
        grad = brainstate.transform.vector_grad(surrogate.relu_grad, argnums=2)(x, alpha, width)

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
        grad = brainstate.transform.vector_grad(surrogate.gaussian_grad, argnums=1)(x, sigma, alpha)

        def f(sigma):
            return surrogate.GaussianGrad(sigma=sigma, alpha=alpha).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(sigma)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_alpha(self, x):
        """Test gradient of alpha parameter."""
        x = jnp.array(x)
        sigma, alpha = 0.5, 0.5
        grad = brainstate.transform.vector_grad(surrogate.gaussian_grad, argnums=2)(x, sigma, alpha)

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
        grad = brainstate.transform.vector_grad(surrogate.s2nn, argnums=1)(x, alpha, beta, epsilon)

        def f(alpha):
            return surrogate.S2NN(alpha=alpha, beta=beta, epsilon=epsilon).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(alpha)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [0.0, 1.0])
    def test_grad_of_parameters_beta(self, x):
        """Test gradient of beta parameter."""
        x = jnp.array(x)
        alpha, beta, epsilon = 4.0, 1.0, 1e-8
        grad = brainstate.transform.vector_grad(surrogate.s2nn, argnums=2)(x, alpha, beta, epsilon)

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
        grad = brainstate.transform.vector_grad(surrogate.multi_gaussian_grad, argnums=1)(x, h, s, sigma, scale)

        def f(h):
            return surrogate.MultiGaussianGrad(h=h, s=s, sigma=sigma, scale=scale).surrogate_grad(x)

        expected = brainstate.transform.vector_grad(f)(h)
        assert jnp.allclose(grad, expected)

    @pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
    def test_grad_of_parameters_s(self, x):
        """Test gradient of s parameter."""
        x = jnp.array(x)
        h, s, sigma, scale = 0.15, 6.0, 0.5, 0.5
        grad = brainstate.transform.vector_grad(surrogate.multi_gaussian_grad, argnums=2)(x, h, s, sigma, scale)

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
