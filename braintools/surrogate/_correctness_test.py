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
Correctness tests for ``braintools.surrogate``.

Unlike ``_impl_test.py`` (which checks that backprop reproduces
``surrogate_grad`` -- i.e. the *plumbing*), this module checks the surrogate
*formulas* themselves against:

* their documented mathematical definitions / closed-form references, and
* the function/derivative consistency ``d/dx surrogate_fun == surrogate_grad``.

These are the checks that catch a wrong ``surrogate_grad``/``surrogate_fun``
formula -- the failure mode the self-referential suite cannot detect.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.surrogate as surrogate


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
    g_beta = brainstate.transform.vector_grad(surrogate.s2nn, argnums=2)(x, 4.0, 1.0, 1e-8)
    g_alpha = brainstate.transform.vector_grad(surrogate.s2nn, argnums=1)(x, 4.0, 1.0, 1e-8)
    assert jnp.isfinite(g_beta)
    assert jnp.isfinite(g_alpha)


def test_log_tailed_relu_grad_finite_at_zero():
    sg = surrogate.LogTailedRelu(alpha=0.1)
    g = sg.surrogate_grad(jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_log_tailed_relu_param_grad_finite_at_zero():
    x = jnp.array(0.0)
    import brainstate.transform
    g = brainstate.transform.vector_grad(surrogate.log_tailed_relu, argnums=1)(x, 0.1)
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
