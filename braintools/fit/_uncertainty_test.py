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

from braintools.fit._uncertainty import HessianCI, hessian_ci


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def _regression(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.uniform(-1, 1, size=n)
    y = 1.5 - 0.7 * x1 + 2.0 * x2 + rng.normal(scale=0.3, size=n)
    return x1, x2, y


@pytest.mark.parametrize('confidence', [0.95, 0.9, 0.5])
def test_sse_matches_statsmodels_ols(x64, confidence):
    sm = pytest.importorskip('statsmodels.api')
    x1, x2, y = _regression()
    X = sm.add_constant(np.column_stack([x1, x2]))
    ols = sm.OLS(y, X).fit()
    best = {'b0': jnp.asarray(ols.params[0]), 'w': jnp.asarray(ols.params[1:])}
    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    def sse(p):
        return jnp.sum((yj - p['b0'] - Xj[:, 1:] @ p['w']) ** 2)

    ci = hessian_ci(sse, best, confidence=confidence, n_data=len(y))
    assert isinstance(ci, HessianCI)
    ref_ci = ols.conf_int(alpha=1. - confidence)
    np.testing.assert_allclose(float(ci.stderr['b0']), ols.bse[0], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(ci.stderr['w']), ols.bse[1:], rtol=1e-8)
    np.testing.assert_allclose([float(ci.low['b0']), float(ci.high['b0'])], ref_ci[0], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(ci.low['w']), ref_ci[1:, 0], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(ci.high['w']), ref_ci[1:, 1], rtol=1e-8)
    np.testing.assert_allclose(np.asarray(ci.estimate['w']), ols.params[1:])


def test_nll_gaussian_mean_matches_analytic(x64):
    stats = pytest.importorskip('scipy.stats')
    rng = np.random.default_rng(1)
    data = jnp.asarray(rng.normal(3., 2., size=200))
    sigma = 2.

    def nll(p):
        return 0.5 * jnp.sum((data - p['mu']) ** 2) / sigma ** 2

    ci = hessian_ci(nll, {'mu': jnp.mean(data)}, kind='nll', confidence=0.9)
    se = sigma / np.sqrt(200)
    np.testing.assert_allclose(float(ci.stderr['mu']), se, rtol=1e-10)
    z = stats.norm.ppf(0.95)
    np.testing.assert_allclose(float(ci.high['mu']), float(jnp.mean(data)) + z * se, rtol=1e-10)


def test_quantity_parameters_keep_units():
    t = jnp.linspace(0., 10., 20)
    y = 2. * t + 1.

    def sse(p):
        return jnp.sum((y - p['k'].to_decimal(u.mV) * t - p['c']) ** 2) + 1e-3

    ci = hessian_ci(sse, {'k': 2. * u.mV, 'c': jnp.asarray(1.)}, n_data=20)
    assert isinstance(ci.stderr['k'], u.Quantity) and ci.stderr['k'].unit == u.mV
    assert ci.low['k'].to_decimal(u.mV) < 2. < ci.high['k'].to_decimal(u.mV)
    assert not isinstance(ci.stderr['c'], u.Quantity)


def test_tuple_params_and_jit_vmap(x64):
    x = jnp.linspace(0., 1., 10)
    y = 3. * x - 1. + 0.05 * jnp.sin(17. * x)

    def fit_ci(y):
        A = jnp.stack([jnp.ones_like(x), x], 1)
        coef = jnp.linalg.lstsq(A, y)[0]
        sse = lambda p: jnp.sum((y - p[0] - p[1] * x) ** 2)
        return hessian_ci(sse, (coef[0], coef[1]), n_data=10).stderr[1]

    eager = fit_ci(y)
    np.testing.assert_allclose(float(jax.jit(fit_ci)(y)), float(eager), rtol=1e-10)
    batch = jax.vmap(fit_ci)(jnp.stack([y, 2. * y]))
    np.testing.assert_allclose(float(batch[0]), float(eager), rtol=1e-10)
    np.testing.assert_allclose(float(batch[1]), 2. * float(eager), rtol=1e-8)


def test_singular_hessian_gives_non_finite_or_huge_stderr():
    # parameter 'b' does not enter the loss: it is not identifiable
    ci = hessian_ci(lambda p: (p['a'] - 1.) ** 2, {'a': jnp.asarray(1.), 'b': jnp.asarray(0.)}, kind='nll')
    se_b = float(ci.stderr['b'])
    assert not np.isfinite(se_b) or se_b > 1e3


def test_validation():
    loss = lambda p: jnp.sum(p['a'] ** 2)
    best = {'a': jnp.zeros(2)}
    with pytest.raises(ValueError, match='kind'):
        hessian_ci(loss, best, kind='mse')
    with pytest.raises(ValueError, match='confidence'):
        hessian_ci(loss, best, kind='nll', confidence=1.)
    with pytest.raises(ValueError, match='n_data'):
        hessian_ci(loss, best)
    with pytest.raises(ValueError, match='exceed'):
        hessian_ci(loss, best, n_data=2)
    with pytest.raises(ValueError, match='no leaves'):
        hessian_ci(loss, {}, kind='nll')
    with pytest.raises(ValueError, match='scalar'):
        hessian_ci(lambda p: p['a'], best, kind='nll')
    with pytest.raises(TypeError, match='dimensionless'):
        hessian_ci(lambda p: jnp.sum(p['a'] ** 2) * u.mV, best, kind='nll')
    with pytest.raises(TypeError, match='concrete'):
        jax.jit(lambda c: hessian_ci(loss, best, kind='nll', confidence=c).stderr)(0.9)
