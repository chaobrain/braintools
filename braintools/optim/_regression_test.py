# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.optim._regression import (
    RegressionResult,
    elastic_net_regression,
    lasso_regression,
    linear_regression,
    ridge_regression,
)

sklm = pytest.importorskip('sklearn.linear_model')


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def make_data(n=40, p=6, k=None, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    w = rng.normal(size=(p,) if k is None else (p, k))
    w[::3] = 0.  # some exact zeros for the sparse problems
    Y = X @ w + 0.7 + noise * rng.normal(size=(n,) if k is None else (n, k))
    return X, Y


def make_ill_conditioned(n=60, p=8, cond=1e4, seed=1):
    """n > p design whose centred singular values span ``cond`` decades."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, p)))
    V, _ = np.linalg.qr(rng.normal(size=(p, p)))
    s = np.logspace(0, -np.log10(cond), p) * np.sqrt(n)
    X = (U * s) @ V.T + 0.3
    w = rng.normal(size=(p, 2))
    w[::3] = 0.
    Y = X @ w + 0.5 + 0.1 * rng.normal(size=(n, 2))
    return X, Y


def sk_coef(model, multi):
    """sklearn's coef_ in braintools' (n_features, n_targets) layout."""
    return model.coef_.T if multi else model.coef_


def assert_matches(res, model, multi, atol=1e-5):
    np.testing.assert_allclose(np.asarray(res.coef), sk_coef(model, multi), atol=atol)
    np.testing.assert_allclose(np.asarray(res.intercept), model.intercept_, atol=atol)


# ---------------------------------------------------------------------------
# reference values (sklearn)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('k', [None, 3])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_linear_matches_sklearn(x64, k, fit_intercept):
    X, Y = make_data(k=k)
    res = linear_regression(X, Y, fit_intercept=fit_intercept)
    ref = sklm.LinearRegression(fit_intercept=fit_intercept).fit(X, Y)
    assert isinstance(res, RegressionResult)
    assert res.coef.shape == ((6,) if k is None else (6, k))
    assert_matches(res, ref, k is not None, atol=1e-8)


@pytest.mark.parametrize('shape', [(40, 6), (10, 25)])  # n > p and p > n
@pytest.mark.parametrize('k', [None, 2])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('alpha', [0.1, 3.0])
def test_ridge_matches_sklearn(x64, shape, k, fit_intercept, alpha):
    X, Y = make_data(*shape, k=k)
    res = ridge_regression(X, Y, alpha, fit_intercept=fit_intercept)
    ref = sklm.Ridge(alpha=alpha, fit_intercept=fit_intercept, solver='svd').fit(X, Y)
    assert_matches(res, ref, k is not None, atol=1e-8)


@pytest.mark.parametrize('shape', [(40, 6), (15, 30)])
@pytest.mark.parametrize('k', [None, 2])
@pytest.mark.parametrize('alpha', [0.01, 0.2])
def test_lasso_matches_sklearn(x64, shape, k, alpha):
    X, Y = make_data(*shape, k=k)
    res = lasso_regression(X, Y, alpha, tol=1e-12, max_iter=200000)
    ref = sklm.Lasso(alpha=alpha, tol=1e-14, max_iter=1000000).fit(X, Y)
    assert_matches(res, ref, k is not None)


def test_lasso_no_intercept_matches_sklearn(x64):
    X, Y = make_data()
    res = lasso_regression(X, Y, 0.05, fit_intercept=False, tol=1e-12, max_iter=200000)
    ref = sklm.Lasso(alpha=0.05, fit_intercept=False, tol=1e-14, max_iter=1000000).fit(X, Y)
    assert_matches(res, ref, False)
    assert float(res.intercept) == 0.


@pytest.mark.parametrize('shape', [(40, 6), (15, 30)])
@pytest.mark.parametrize('k', [None, 2])
@pytest.mark.parametrize('l1_ratio', [0., 0.3, 1.])
def test_elastic_net_matches_sklearn(x64, shape, k, l1_ratio):
    X, Y = make_data(*shape, k=k)
    alpha = 0.1
    res = elastic_net_regression(X, Y, alpha, l1_ratio, tol=1e-12, max_iter=200000)
    if l1_ratio == 0.:
        # sklearn's coordinate descent warns for l1_ratio=0; the objective
        # is then (1/2n)||y - Xw||^2 + alpha/2 ||w||^2, i.e. Ridge(n * alpha).
        ref = sklm.Ridge(alpha=shape[0] * alpha, solver='svd').fit(X, Y)
    else:
        ref = sklm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-14, max_iter=1000000).fit(X, Y)
    assert_matches(res, ref, k is not None)


DEFAULT_TOL_PROBLEMS = {
    # p > n: the least-squares term is not strongly convex.
    'p>n': (lambda: make_data(15, 30, k=2), 0.01),
    # n > p with a condition number of 1e4: tiny curvature along some directions.
    'ill-conditioned': (make_ill_conditioned, 0.001),
}


@pytest.mark.parametrize('problem', sorted(DEFAULT_TOL_PROBLEMS))
@pytest.mark.parametrize('l1_ratio', [1., 0.5])
def test_default_tol_matches_sklearn(x64, problem, l1_ratio):
    # Regression: the old rule stopped on a small step |W - Z|, which happens
    # long before convergence when the curvature is small (rel. error ~1e-2).
    make, alpha = DEFAULT_TOL_PROBLEMS[problem]
    X, Y = make()
    if l1_ratio == 1.:
        res = lasso_regression(X, Y, alpha)
        ref = sklm.Lasso(alpha=alpha, tol=1e-14, max_iter=1000000).fit(X, Y)
    else:
        res = elastic_net_regression(X, Y, alpha, l1_ratio)
        ref = sklm.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-14, max_iter=1000000).fit(X, Y)
    scale = np.max(np.abs(ref.coef_))
    rel_err = np.max(np.abs(np.asarray(res.coef) - ref.coef_.T)) / scale
    assert rel_err <= 1e-4, rel_err
    # The intercept inherits the coefficient error through x_mean @ coef.
    np.testing.assert_allclose(res.intercept, ref.intercept_,
                               atol=1e-4 * scale * (1. + np.abs(X.mean(0)).sum()))


def test_default_tol_is_scale_invariant(x64):
    # The stopping rule is relative to max|X^T Y| / n, so rescaling the data
    # (and the penalty accordingly) rescales the solution exactly.
    X, Y = make_data(15, 30, k=2)
    base = lasso_regression(X, Y, 0.01)
    cx, cy = 1e-3, 1e4
    scaled = lasso_regression(cx * X, cy * Y, 0.01 * cx * cy)
    np.testing.assert_allclose(np.asarray(scaled.coef) * cx / cy, base.coef,
                               atol=1e-6 * np.max(np.abs(base.coef)))


def test_default_tol_float32_matches_sklearn():
    X, Y = make_data(15, 30, k=2)
    ref = sklm.Lasso(alpha=0.01, tol=1e-14, max_iter=1000000).fit(X, Y)
    res = lasso_regression(X.astype(np.float32), Y.astype(np.float32), 0.01)
    assert res.coef.dtype == jnp.float32
    scale = np.max(np.abs(ref.coef_))
    assert np.max(np.abs(np.asarray(res.coef, np.float64) - ref.coef_.T)) <= 1e-3 * scale


def test_lasso_equals_elastic_net_l1_ratio_one(x64):
    X, Y = make_data()
    a = lasso_regression(X, Y, 0.1, tol=1e-12)
    b = elastic_net_regression(X, Y, 0.1, 1.0, tol=1e-12)
    np.testing.assert_allclose(a.coef, b.coef, atol=1e-12)
    np.testing.assert_allclose(a.intercept, b.intercept, atol=1e-12)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_ridge_alpha_zero_is_least_squares(x64):
    X, Y = make_data(k=2)
    res = ridge_regression(X, Y, 0.)
    lin = linear_regression(X, Y)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    np.testing.assert_allclose(res.coef, np.linalg.lstsq(Xc, Yc, rcond=None)[0], atol=1e-10)
    np.testing.assert_allclose(res.coef, lin.coef, atol=1e-10)
    np.testing.assert_allclose(res.intercept, lin.intercept, atol=1e-10)


def test_ridge_alpha_zero_more_features_is_min_norm(x64):
    X, Y = make_data(8, 20)
    res = ridge_regression(X, Y, 0., fit_intercept=False)
    np.testing.assert_allclose(res.coef, np.linalg.pinv(X) @ Y, atol=1e-10)
    np.testing.assert_allclose(X @ np.asarray(res.coef), Y, atol=1e-10)  # interpolates


def test_more_features_than_samples(x64):
    X, Y = make_data(8, 20)
    lin = linear_regression(X, Y)
    ref = sklm.LinearRegression().fit(X, Y)
    assert_matches(lin, ref, False, atol=1e-8)
    assert np.all(np.isfinite(ridge_regression(X, Y, 1.).coef))


def test_constant_feature_with_intercept(x64):
    X, Y = make_data()
    X[:, 2] = 4.2
    for res in (linear_regression(X, Y),
                ridge_regression(X, Y, 0.),
                ridge_regression(X, Y, 1.),
                lasso_regression(X, Y, 0.05, tol=1e-12),
                elastic_net_regression(X, Y, 0.05, 0.5, tol=1e-12)):
        assert abs(float(res.coef[2])) < 1e-10
        assert np.all(np.isfinite(res.coef)) and np.isfinite(res.intercept)
    ref = sklm.LinearRegression().fit(X, Y)
    assert_matches(linear_regression(X, Y), ref, False, atol=1e-8)
    ref = sklm.Ridge(alpha=1.0, solver='svd').fit(X, Y)
    assert_matches(ridge_regression(X, Y, 1.), ref, False, atol=1e-8)


def test_lasso_large_alpha_gives_zero(x64):
    X, Y = make_data()
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    alpha_max = np.max(np.abs(Xc.T @ Yc)) / X.shape[0]
    res = lasso_regression(X, Y, 1.01 * alpha_max)
    np.testing.assert_array_equal(res.coef, 0.)
    np.testing.assert_allclose(res.intercept, Y.mean(), atol=1e-12)
    res = lasso_regression(X, Y, 0.99 * alpha_max, tol=1e-12)
    assert np.count_nonzero(np.asarray(res.coef)) >= 1


def test_all_zero_design_matrix(x64):
    X = np.zeros((10, 3))
    Y = np.arange(10.)
    for res in (linear_regression(X, Y), ridge_regression(X, Y, 0.), ridge_regression(X, Y),
                lasso_regression(X, Y, 0.1), elastic_net_regression(X, Y, 0.1, 0.)):
        np.testing.assert_array_equal(res.coef, 0.)
        np.testing.assert_allclose(res.intercept, 4.5)


def test_lasso_max_iter_caps_iterations(x64):
    X, Y = make_data()
    one = lasso_regression(X, Y, 0.01, max_iter=1, tol=0.)
    full = lasso_regression(X, Y, 0.01, tol=1e-12)
    assert not np.allclose(one.coef, full.coef, atol=1e-3)


def test_float32_inputs():
    X, Y = make_data()
    X32, Y32 = X.astype(np.float32), Y.astype(np.float32)
    for fn, kw in ((linear_regression, {}), (ridge_regression, {'alpha': 0.5}),
                   (lasso_regression, {'alpha': 0.05}), (elastic_net_regression, {'alpha': 0.05})):
        res = fn(X32, Y32, **kw)
        assert res.coef.dtype == jnp.float32
        assert np.all(np.isfinite(res.coef))
    ref = sklm.Ridge(alpha=0.5).fit(X, Y)
    np.testing.assert_allclose(ridge_regression(X32, Y32, 0.5).coef, ref.coef_, atol=1e-4)


def test_integer_inputs_are_promoted():
    X = np.arange(12).reshape(6, 2) % 5
    Y = np.arange(6)
    res = ridge_regression(X, Y, 1.)
    assert jnp.issubdtype(res.coef.dtype, jnp.floating)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fn', [linear_regression, ridge_regression, lasso_regression,
                                elastic_net_regression])
def test_shape_validation(fn):
    with pytest.raises(ValueError, match='2-D'):
        fn(np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match='1-D or 2-D'):
        fn(np.ones((5, 2)), np.ones((5, 1, 1)))
    with pytest.raises(ValueError, match='same number of samples'):
        fn(np.ones((5, 2)), np.ones(4))
    with pytest.raises(ValueError, match='at least one sample'):
        fn(np.ones((0, 2)), np.ones(0))


def test_parameter_validation():
    X, Y = make_data()
    with pytest.raises(ValueError, match='alpha'):
        ridge_regression(X, Y, -1.)
    with pytest.raises(ValueError, match='alpha'):
        lasso_regression(X, Y, -1.)
    with pytest.raises(ValueError, match='l1_ratio'):
        elastic_net_regression(X, Y, 1., 1.5)
    with pytest.raises(ValueError, match='max_iter'):
        lasso_regression(X, Y, 1., max_iter=0)
    with pytest.raises(ValueError, match='max_iter'):
        lasso_regression(X, Y, 1., max_iter=2.5)
    with pytest.raises(ValueError, match='tol'):
        elastic_net_regression(X, Y, 1., tol=-1.)


@pytest.mark.parametrize('fn, args', [
    (ridge_regression, (-0.5,)),
    (lasso_regression, (-0.05,)),
    (elastic_net_regression, (-0.05, 0.5)),
    (elastic_net_regression, (0.05, 1.5)),   # l1_ratio > 1: negative L2 penalty
    (elastic_net_regression, (0.05, -0.5)),  # l1_ratio < 0: negative L1 penalty
    (lasso_regression, (np.nan,)),
])
def test_traced_invalid_penalty_gives_nan(x64, fn, args):
    # Traced penalties cannot be validated eagerly; they must not give a
    # silently wrong finite answer.
    X, Y = make_data(k=2)
    res = jax.jit(fn)(X, Y, *args)
    assert np.all(np.isnan(res.coef))
    assert np.all(np.isnan(res.intercept))


def test_vmap_negative_alpha_only_poisons_its_member(x64):
    X, Y = make_data()
    alphas = jnp.array([0.05, -0.05, 0.1])
    for fn in (ridge_regression, lasso_regression):
        coefs = jax.vmap(lambda a: fn(X, Y, a).coef)(alphas)
        assert np.all(np.isnan(coefs[1]))
        np.testing.assert_allclose(coefs[0], fn(X, Y, 0.05).coef, atol=1e-10)
        np.testing.assert_allclose(coefs[2], fn(X, Y, 0.1).coef, atol=1e-10)


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_linear_regression_with_units(x64):
    X, Y = make_data()
    res = linear_regression(X * u.mV, Y * u.nA)
    plain = linear_regression(X, Y)
    assert u.get_unit(res.coef) == u.nA / u.mV
    assert u.get_unit(res.intercept) == u.nA
    np.testing.assert_allclose(res.coef.to_decimal(u.nA / u.mV), plain.coef, atol=1e-10)
    np.testing.assert_allclose(res.intercept.to_decimal(u.nA), plain.intercept, atol=1e-10)


def test_ridge_with_units(x64):
    X, Y = make_data()
    res = ridge_regression(X * u.mV, Y * u.nA, 2. * u.mV ** 2)
    plain = ridge_regression(X, Y, 2.)
    np.testing.assert_allclose(res.coef.to_decimal(u.nA / u.mV), plain.coef, atol=1e-10)
    # The same penalty in other units gives the same fit.
    res_v = ridge_regression(X * u.mV, Y * u.nA, 2e-6 * u.volt ** 2)
    np.testing.assert_allclose(res_v.coef.to_decimal(u.nA / u.mV), plain.coef, atol=1e-8)
    with pytest.raises(u.UnitMismatchError):
        ridge_regression(X * u.mV, Y * u.nA, 2.)
    with pytest.raises(u.UnitMismatchError):
        ridge_regression(X * u.mV, Y * u.nA, 2. * u.mV)
    # dimensionless Quantity penalty with unitless data is fine
    np.testing.assert_allclose(ridge_regression(X, Y, u.Quantity(2.)).coef, plain.coef)


def test_lasso_with_units(x64):
    X, Y = make_data()
    res = lasso_regression(X * u.mV, Y * u.nA, 0.05 * u.mV * u.nA, tol=1e-12)
    plain = lasso_regression(X, Y, 0.05, tol=1e-12)
    np.testing.assert_allclose(res.coef.to_decimal(u.nA / u.mV), plain.coef, atol=1e-10)
    with pytest.raises(u.UnitMismatchError):
        lasso_regression(X * u.mV, Y * u.nA, 0.05 * u.mV ** 2)


def test_elastic_net_rejects_units():
    X, Y = make_data()
    with pytest.raises(u.UnitMismatchError):
        elastic_net_regression(X * u.mV, Y, 0.1)


# ---------------------------------------------------------------------------
# transformations
# ---------------------------------------------------------------------------

def test_jit_matches_eager(x64):
    X, Y = make_data(k=2)
    for fn, args in ((linear_regression, ()), (ridge_regression, (0.5,)),
                     (lasso_regression, (0.05,)), (elastic_net_regression, (0.05, 0.4))):
        eager = fn(X, Y, *args)
        jitted = jax.jit(fn)(X, Y, *args)
        np.testing.assert_allclose(jitted.coef, eager.coef, atol=1e-10)
        np.testing.assert_allclose(jitted.intercept, eager.intercept, atol=1e-10)


def test_vmap_over_alpha(x64):
    X, Y = make_data()
    alphas = jnp.array([0.01, 0.1, 1.])
    ridge = jax.vmap(lambda a: ridge_regression(X, Y, a).coef)(alphas)
    lasso = jax.vmap(lambda a: lasso_regression(X, Y, a, tol=1e-12).coef)(alphas)
    for i, a in enumerate(np.asarray(alphas)):
        np.testing.assert_allclose(ridge[i], ridge_regression(X, Y, float(a)).coef, atol=1e-10)
        np.testing.assert_allclose(lasso[i], lasso_regression(X, Y, float(a), tol=1e-12).coef,
                                   atol=1e-8)


def test_vmap_over_datasets(x64):
    data = [make_data(seed=s) for s in range(3)]
    Xs = jnp.stack([d[0] for d in data])
    Ys = jnp.stack([d[1] for d in data])
    for fn, args in ((linear_regression, ()), (ridge_regression, (0.5,)),
                     (elastic_net_regression, (0.05, 0.4))):
        batched = jax.vmap(lambda X, Y: fn(X, Y, *args))(Xs, Ys)
        for i in range(3):
            single = fn(Xs[i], Ys[i], *args)
            np.testing.assert_allclose(batched.coef[i], single.coef, atol=1e-8)
            np.testing.assert_allclose(batched.intercept[i], single.intercept, atol=1e-8)


def _fd(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_ridge_grad(x64):
    X, Y = make_data()

    def loss_alpha(a):
        return jnp.sum(ridge_regression(X, Y, a).coef ** 2)

    g = jax.grad(loss_alpha)(0.5)
    np.testing.assert_allclose(g, _fd(loss_alpha, 0.5), rtol=1e-5)

    def loss_x(X_):
        return jnp.sum(ridge_regression(X_, Y, 0.5).coef)

    gX = jax.grad(loss_x)(jnp.asarray(X))
    E = np.zeros_like(X)
    E[3, 1] = 1.
    fd = _fd(lambda s: loss_x(jnp.asarray(X) + s * E), 0.)
    np.testing.assert_allclose(gX[3, 1], fd, rtol=1e-5)

    def loss_y(Y_):
        return jnp.sum(ridge_regression(X, Y_, 0.5).coef)

    gY = jax.grad(loss_y)(jnp.asarray(Y))
    assert np.all(np.isfinite(gY))


def test_linear_regression_grad(x64):
    X, Y = make_data()

    def loss_y(Y_):
        return jnp.sum(linear_regression(X, Y_).coef ** 2)

    gY = jax.grad(loss_y)(jnp.asarray(Y))
    E = np.zeros_like(Y)
    E[5] = 1.
    np.testing.assert_allclose(gY[5], _fd(lambda s: loss_y(jnp.asarray(Y) + s * E), 0.), rtol=1e-5)


def test_lasso_forward_mode_only(x64):
    X, Y = make_data()

    def coef_sum(a):
        return jnp.sum(lasso_regression(X, Y, a, tol=1e-12).coef)

    _, tangent = jax.jvp(coef_sum, (0.05,), (1.,))
    np.testing.assert_allclose(tangent, _fd(coef_sum, 0.05, eps=1e-5), rtol=1e-3)
    with pytest.raises(ValueError, match='Reverse-mode differentiation'):
        jax.grad(coef_sum)(0.05)
