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
Extra coverage for ``braintools/surrogate/_impl.py``.

These tests target the code paths that the existing ``_impl_test.py`` does not
exercise: every ``__repr__``/``__hash__``, the smooth ``surrogate_fun`` methods,
the ``NotImplementedError`` of the gradient-only surrogates, and the remaining
functional-API / parameter-gradient paths of ``SquarewaveFourierSeries``.
"""

import brainstate.transform
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.surrogate as surrogate


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
        # x <= 0 -> alpha*x ; 0<x<=1 -> x ; x>1 -> log(x)
        z = sg.surrogate_fun(jnp.array([-2.0, 0.5, jnp.e]))
        assert np.allclose(float(z[0]), -0.2)
        assert np.allclose(float(z[1]), 0.5)
        assert np.allclose(float(z[2]), 1.0, atol=1e-6)  # log(e) == 1

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
