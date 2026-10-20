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

# -*- coding: utf-8 -*-

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from braintools.analysis.population._dimensionality import (
    CCAResult,
    PCAResult,
    cca,
    participation_ratio,
    pca,
    rdm,
    rsa,
)


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def _data(seed, shape, mix=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape)
    if mix:
        x = x @ rng.normal(size=(shape[1], shape[1]))
    return x


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------

class TestPCA:
    @pytest.mark.parametrize('n_components', [None, 1, 3])
    def test_matches_sklearn_float64(self, x64, n_components):
        skd = pytest.importorskip('sklearn.decomposition')
        x = _data(0, (40, 5))
        res = pca(x, n_components=n_components)
        ref = skd.PCA(n_components=n_components, svd_solver='full').fit(x)
        np.testing.assert_allclose(res.components, ref.components_, atol=1e-10)
        np.testing.assert_allclose(res.explained_variance, ref.explained_variance_, rtol=1e-10)
        np.testing.assert_allclose(res.explained_variance_ratio, ref.explained_variance_ratio_, rtol=1e-10)
        np.testing.assert_allclose(res.mean, ref.mean_, atol=1e-12)
        np.testing.assert_allclose(res.projection, ref.transform(x), atol=1e-10)
        assert res.components.dtype == jnp.float64

    def test_matches_sklearn_float32(self):
        skd = pytest.importorskip('sklearn.decomposition')
        x = _data(1, (50, 4)).astype(np.float32)
        res = pca(x)
        ref = skd.PCA(svd_solver='full').fit(x.astype(np.float64))
        assert res.components.dtype == jnp.float32
        np.testing.assert_allclose(res.components, ref.components_, atol=1e-5)
        np.testing.assert_allclose(res.explained_variance, ref.explained_variance_, rtol=1e-5)
        np.testing.assert_allclose(res.explained_variance_ratio, ref.explained_variance_ratio_, atol=1e-6)
        np.testing.assert_allclose(res.projection, ref.transform(x), atol=1e-4)

    def test_more_features_than_samples(self, x64):
        skd = pytest.importorskip('sklearn.decomposition')
        x = _data(2, (5, 8), mix=False)
        res = pca(x)
        ref = skd.PCA(svd_solver='full').fit(x)
        assert res.components.shape == (5, 8)
        # Rank is T - 1 = 4; the last component spans an arbitrary null direction.
        np.testing.assert_allclose(res.components[:4], ref.components_[:4], atol=1e-10)
        np.testing.assert_allclose(res.explained_variance, ref.explained_variance_, atol=1e-10)
        np.testing.assert_allclose(res.projection[:, :4], ref.transform(x)[:, :4], atol=1e-10)

    def test_sign_convention(self):
        res = pca(_data(3, (30, 4)))
        comps = np.asarray(res.components)
        idx = np.argmax(np.abs(comps), axis=1)
        assert np.all(comps[np.arange(comps.shape[0]), idx] > 0)
        # Flipping the data does not flip the components.
        res2 = pca(-_data(3, (30, 4)))
        np.testing.assert_allclose(res2.components, res.components, atol=1e-5)

    def test_orthonormal_and_ordered(self):
        res = pca(_data(4, (60, 5)))
        np.testing.assert_allclose(res.components @ res.components.T, np.eye(5), atol=1e-5)
        ev = np.asarray(res.explained_variance)
        assert np.all(np.diff(ev) <= 1e-6)
        np.testing.assert_allclose(np.sum(res.explained_variance_ratio), 1.0, rtol=1e-5)

    def test_one_dimensional_input(self):
        x = jnp.arange(6.0)
        res = pca(x)
        assert res.components.shape == (1, 1)
        np.testing.assert_allclose(res.explained_variance_ratio, [1.0])
        np.testing.assert_allclose(res.explained_variance, [np.var(np.arange(6.0), ddof=1)], rtol=1e-6)

    def test_integer_input(self):
        res = pca(jnp.array([[1, 2], [3, 5], [6, 4]]))
        assert jnp.issubdtype(res.components.dtype, jnp.floating)

    def test_constant_data_ratio_nan(self):
        res = pca(jnp.ones((10, 3)))
        assert np.all(np.isnan(res.explained_variance_ratio))
        np.testing.assert_allclose(res.explained_variance, 0.0)
        assert np.all(np.isfinite(res.components))

    def test_units(self):
        x = _data(5, (20, 3)).astype(np.float32)
        plain = pca(x)
        res = pca(x * u.mV)
        assert res.explained_variance.unit == u.mV ** 2
        assert res.mean.unit == u.mV
        assert res.projection.unit == u.mV
        assert not isinstance(res.components, u.Quantity)
        assert not isinstance(res.explained_variance_ratio, u.Quantity)
        np.testing.assert_allclose(res.explained_variance.mantissa, plain.explained_variance, rtol=1e-6)
        np.testing.assert_allclose(res.projection.mantissa, plain.projection, rtol=1e-5, atol=1e-6)
        # A unitless Quantity behaves like a plain array.
        res_unitless = pca(u.Quantity(x))
        assert not isinstance(res_unitless.mean, u.Quantity)

    def test_jit(self):
        x = _data(6, (30, 4))
        res = jax.jit(lambda a: pca(a, n_components=2))(x)
        assert isinstance(res, PCAResult)
        ref = pca(x, n_components=2)
        for a, b in zip(res, ref):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def test_vmap_over_trials(self):
        xs = np.stack([_data(s, (25, 3)) for s in range(3)])
        res = jax.vmap(lambda a: pca(a, n_components=2))(xs)
        assert res.components.shape == (3, 2, 3)
        assert res.projection.shape == (3, 25, 2)
        for i in range(3):
            np.testing.assert_allclose(res.components[i], pca(xs[i], n_components=2).components, atol=1e-5)

    def test_grad(self, x64):
        x = jnp.asarray(_data(7, (12, 3)))
        check_grads(lambda a: jnp.sum(pca(a).explained_variance * jnp.array([1.0, 2.0, 3.0])),
                    (x,), order=1, modes=['rev'])
        check_grads(lambda a: jnp.sum(pca(a, n_components=1).projection ** 3), (x,), order=1, modes=['rev'])

    def test_errors(self):
        with pytest.raises(ValueError, match='2 samples'):
            pca(jnp.ones((1, 3)))
        with pytest.raises(ValueError, match='shape'):
            pca(jnp.ones((2, 3, 4)))
        with pytest.raises(ValueError, match='n_components'):
            pca(jnp.ones((5, 3)), n_components=4)
        with pytest.raises(ValueError, match='n_components'):
            pca(jnp.ones((5, 3)), n_components=0)
        with pytest.raises(TypeError, match='static int'):
            pca(jnp.ones((5, 3)), n_components=1.5)
        with pytest.raises(TypeError, match='static int'):
            pca(jnp.ones((5, 3)), n_components=True)
        # numpy integers are accepted
        assert pca(jnp.ones((5, 3)) + jnp.arange(3.), n_components=np.int64(2)).components.shape == (2, 3)


# ---------------------------------------------------------------------------
# participation_ratio
# ---------------------------------------------------------------------------

def _pr_numpy(x):
    lam = np.linalg.eigvalsh(np.cov(x, rowvar=False))
    return lam.sum() ** 2 / np.sum(lam ** 2)


class TestParticipationRatio:
    @pytest.mark.parametrize('shape', [(50, 4), (6, 10)])
    def test_matches_numpy(self, x64, shape):
        x = _data(10, shape)
        np.testing.assert_allclose(participation_ratio(x), _pr_numpy(x), rtol=1e-10)

    def test_float32(self):
        x = _data(11, (40, 5)).astype(np.float32)
        out = participation_ratio(x)
        assert out.dtype == jnp.float32
        np.testing.assert_allclose(out, _pr_numpy(x.astype(np.float64)), rtol=1e-5)

    def test_analytic_cases(self):
        latent = np.random.default_rng(0).normal(size=(100, 1))
        np.testing.assert_allclose(participation_ratio(latent @ np.ones((1, 5))), 1.0, rtol=1e-5)
        # Exactly isotropic: orthonormal, zero-mean columns (QR of centred data
        # spans a subspace orthogonal to the ones vector), so PR = D.
        z = np.random.default_rng(1).normal(size=(30, 4))
        q = np.linalg.qr(z - z.mean(0))[0]
        np.testing.assert_allclose(participation_ratio(q), 4.0, rtol=1e-5)

    def test_constant_is_nan(self):
        assert np.isnan(participation_ratio(jnp.ones((10, 3))))

    def test_units_cancel(self):
        x = _data(12, (20, 3)).astype(np.float32)
        out = participation_ratio(x * u.mV)
        assert not isinstance(out, u.Quantity)
        np.testing.assert_allclose(out, participation_ratio(x), rtol=1e-6)

    def test_jit_vmap(self):
        xs = np.stack([_data(s, (20, 3)) for s in range(4)])
        out = jax.jit(jax.vmap(participation_ratio))(xs)
        np.testing.assert_allclose(out, [_pr_numpy(a) for a in xs], rtol=1e-4)

    def test_grad(self, x64):
        check_grads(participation_ratio, (jnp.asarray(_data(13, (15, 3))),), order=1, modes=['rev'])
        check_grads(participation_ratio, (jnp.asarray(_data(13, (4, 6))),), order=1, modes=['rev'])

    def test_errors(self):
        with pytest.raises(ValueError, match='shape'):
            participation_ratio(jnp.ones(5))
        with pytest.raises(ValueError, match='2 samples'):
            participation_ratio(jnp.ones((1, 5)))


# ---------------------------------------------------------------------------
# cca
# ---------------------------------------------------------------------------

def _cca_qr_numpy(x, y):
    """Björck-Golub reference: singular values of Qx^T Qy (reg = 0)."""
    qx = np.linalg.qr(x - x.mean(0))[0]
    qy = np.linalg.qr(y - y.mean(0))[0]
    return np.linalg.svd(qx.T @ qy, compute_uv=False)


def _cca_eig_numpy(x, y, reg):
    """Reference with ridge: rho^2 = eig(Cxx^-1 Cxy Cyy^-1 Cyx)."""
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    t = x.shape[0] - 1
    cxx = xc.T @ xc / t + reg * np.eye(x.shape[1])
    cyy = yc.T @ yc / t + reg * np.eye(y.shape[1])
    cxy = xc.T @ yc / t
    m = np.linalg.solve(cxx, cxy) @ np.linalg.solve(cyy, cxy.T)
    ev = np.sort(np.real(np.linalg.eigvals(m)))[::-1]
    k = min(x.shape[1], y.shape[1])
    return np.sqrt(np.clip(ev[:k], 0, None))


def _xy(seed, t=80, p=3, q=4, noise=1.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(t, 2))
    x = z @ rng.normal(size=(2, p)) + noise * rng.normal(size=(t, p))
    y = z @ rng.normal(size=(2, q)) + noise * rng.normal(size=(t, q))
    return x, y


class TestCCA:
    def test_matches_qr_reference(self, x64):
        x, y = _xy(0)
        res = cca(x, y)
        np.testing.assert_allclose(res.correlations, _cca_qr_numpy(x, y)[:3], rtol=1e-10)
        assert res.x_weights.shape == (3, 3)
        assert res.y_weights.shape == (4, 3)

    @pytest.mark.parametrize('reg', [0.0, 0.1, 2.0])
    def test_matches_eig_reference_with_reg(self, x64, reg):
        x, y = _xy(1)
        np.testing.assert_allclose(cca(x, y, reg=reg).correlations, _cca_eig_numpy(x, y, reg), rtol=1e-8)

    def test_float32(self):
        x, y = _xy(2)
        res = cca(x.astype(np.float32), y.astype(np.float32))
        assert res.correlations.dtype == jnp.float32
        np.testing.assert_allclose(res.correlations, _cca_qr_numpy(x, y)[:3], atol=1e-5)

    def test_variates_properties(self, x64):
        x, y = _xy(3)
        res = cca(x, y, n_components=2)
        uu = (x - x.mean(0)) @ np.asarray(res.x_weights)
        vv = (y - y.mean(0)) @ np.asarray(res.y_weights)
        np.testing.assert_allclose(np.cov(uu, rowvar=False), np.eye(2), atol=1e-10)
        np.testing.assert_allclose(np.cov(vv, rowvar=False), np.eye(2), atol=1e-10)
        cross = np.cov(uu.T, vv.T)[:2, 2:]
        np.testing.assert_allclose(cross, np.diag(res.correlations), atol=1e-10)

    def test_linear_transform_gives_one(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(200, 3))
        y = x @ rng.normal(size=(3, 3)) + 5.0
        np.testing.assert_allclose(cca(x, y).correlations, 1.0, atol=1e-4)

    def test_sign_convention(self):
        x, y = _xy(5)
        res = cca(x, y)
        w = np.asarray(res.x_weights)
        idx = np.argmax(np.abs(w), axis=0)
        assert np.all(w[idx, np.arange(w.shape[1])] > 0)
        assert np.all(np.asarray(res.correlations) >= 0)

    def test_sklearn_first_correlation_loose(self, x64):
        skc = pytest.importorskip('sklearn.cross_decomposition')
        x, y = _xy(6, t=300, noise=0.5)
        model = skc.CCA(n_components=1, max_iter=2000, tol=1e-10).fit(x, y)
        xs, ys = model.transform(x, y)
        r_sk = abs(np.corrcoef(xs[:, 0], ys[:, 0])[0, 1])
        np.testing.assert_allclose(cca(x, y).correlations[0], r_sk, atol=1e-3)

    def test_rank_deficient_no_nan(self):
        x, _ = _xy(7)
        y = np.concatenate([x[:, :1], x[:, :1]], axis=1)  # collinear columns
        res = cca(x, y)
        assert np.all(np.isfinite(res.correlations))
        assert np.all(np.isfinite(res.x_weights)) and np.all(np.isfinite(res.y_weights))
        np.testing.assert_allclose(res.correlations[0], 1.0, atol=1e-4)
        np.testing.assert_allclose(res.correlations[1], 0.0, atol=1e-3)

    def test_one_dimensional_inputs(self):
        x, y = _xy(8)
        res = cca(x[:, 0], y[:, 0])
        np.testing.assert_allclose(res.correlations[0], abs(np.corrcoef(x[:, 0], y[:, 0])[0, 1]), rtol=1e-4)

    def test_units(self):
        x, y = _xy(9)
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        res = cca(x * u.mV, y * u.nA)
        plain = cca(x, y)
        assert res.x_weights.unit == u.mV ** -1
        assert res.y_weights.unit == u.nA ** -1
        assert not isinstance(res.correlations, u.Quantity)
        np.testing.assert_allclose(res.correlations, plain.correlations, rtol=1e-5)
        np.testing.assert_allclose(res.x_weights.mantissa, plain.x_weights, rtol=1e-4, atol=1e-5)

    def test_jit_with_traced_reg(self):
        x, y = _xy(10)
        res = jax.jit(lambda a, b, r: cca(a, b, n_components=2, reg=r))(x, y, 0.1)
        assert isinstance(res, CCAResult)
        np.testing.assert_allclose(res.correlations, cca(x, y, n_components=2, reg=0.1).correlations, rtol=1e-5)

    def test_vmap(self):
        pairs = [_xy(s) for s in range(3)]
        xs = np.stack([p[0] for p in pairs])
        ys = np.stack([p[1] for p in pairs])
        res = jax.vmap(cca)(xs, ys)
        assert res.correlations.shape == (3, 3)
        for i in range(3):
            np.testing.assert_allclose(res.correlations[i], _cca_qr_numpy(xs[i], ys[i])[:3], atol=1e-5)

    def test_grad(self, x64):
        x, y = _xy(11, t=30)
        check_grads(lambda a: jnp.sum(cca(a, jnp.asarray(y), reg=0.05).correlations),
                    (jnp.asarray(x),), order=1, modes=['rev'])

    def test_errors(self):
        x, y = _xy(12)
        with pytest.raises(ValueError, match='same number of samples'):
            cca(x, y[:-1])
        with pytest.raises(ValueError, match='2 samples'):
            cca(x[:1], y[:1])
        with pytest.raises(ValueError, match='non-negative'):
            cca(x, y, reg=-1.0)
        with pytest.raises(ValueError, match='n_components'):
            cca(x, y, n_components=4)
        with pytest.raises(ValueError, match='shape'):
            cca(x[None], y[None])


# ---------------------------------------------------------------------------
# rdm / rsa
# ---------------------------------------------------------------------------

class TestRDM:
    @pytest.mark.parametrize('metric', ['correlation', 'euclidean', 'cosine'])
    def test_matches_scipy(self, x64, metric):
        dist = pytest.importorskip('scipy.spatial.distance')
        x = _data(20, (7, 5))
        ref = dist.squareform(dist.pdist(x, metric=metric))
        out = rdm(x, metric=metric)
        np.testing.assert_allclose(out, ref, atol=1e-12)
        assert np.all(np.diag(out) == 0)
        np.testing.assert_array_equal(out, np.asarray(out).T)

    @pytest.mark.parametrize('metric', ['correlation', 'euclidean', 'cosine'])
    def test_float32(self, metric):
        dist = pytest.importorskip('scipy.spatial.distance')
        x = _data(21, (6, 8)).astype(np.float32)
        ref = dist.squareform(dist.pdist(x.astype(np.float64), metric=metric))
        np.testing.assert_allclose(rdm(x, metric=metric), ref, atol=1e-5)

    def test_identical_rows(self):
        x = jnp.array([[1., 2., 3.], [1., 2., 3.], [0., 1., 5.]])
        for metric in ('correlation', 'euclidean', 'cosine'):
            np.testing.assert_allclose(rdm(x, metric)[0, 1], 0.0, atol=1e-6)

    def test_degenerate_rows_nan(self):
        x = jnp.array([[0., 0., 0.], [1., 2., 3.], [2., 2., 2.]])
        d = rdm(x, 'cosine')
        assert np.isnan(d[0, 1]) and np.isnan(d[1, 0])
        assert d[0, 0] == 0
        d = rdm(x, 'correlation')
        assert np.isnan(d[2, 1]) and d[2, 2] == 0

    def test_units(self):
        x = _data(22, (4, 3)).astype(np.float32)
        d = rdm(x * u.mV, 'euclidean')
        assert d.unit == u.mV
        np.testing.assert_allclose(d.mantissa, rdm(x, 'euclidean'), rtol=1e-6)
        d = rdm(x * u.mV, 'correlation')
        assert not isinstance(d, u.Quantity)

    def test_jit_vmap(self):
        xs = np.stack([_data(s, (5, 4)) for s in range(3)])
        out = jax.jit(jax.vmap(lambda a: rdm(a, 'euclidean')))(xs)
        assert out.shape == (3, 5, 5)
        np.testing.assert_allclose(out[1], rdm(xs[1], 'euclidean'), rtol=1e-5)

    @pytest.mark.parametrize('metric', ['correlation', 'euclidean', 'cosine'])
    def test_grad(self, x64, metric):
        x = jnp.asarray(_data(23, (4, 3)))
        check_grads(lambda a: jnp.sum(rdm(a, metric) ** 2), (x,), order=1, modes=['rev'])

    def test_grad_euclidean_finite_at_zero_distance(self):
        x = jnp.array([[1., 2.], [1., 2.], [0., 0.]])
        g = jax.grad(lambda a: jnp.sum(rdm(a, 'euclidean')))(x)
        assert np.all(np.isfinite(g))

    def test_errors(self):
        with pytest.raises(ValueError, match='metric'):
            rdm(jnp.ones((3, 2)), 'cityblock')
        with pytest.raises(ValueError, match='shape'):
            rdm(jnp.ones(3))


class TestRSA:
    def test_spearman_with_ties_matches_scipy(self, x64):
        stats = pytest.importorskip('scipy.stats')
        rng = np.random.default_rng(30)
        a = rng.integers(0, 4, 21).astype(float)  # many ties
        b = rng.integers(0, 3, 21).astype(float)
        np.testing.assert_allclose(rsa(a, b), stats.spearmanr(a, b).statistic, rtol=1e-12)

    def test_pearson_matches_scipy(self, x64):
        stats = pytest.importorskip('scipy.stats')
        rng = np.random.default_rng(31)
        a, b = rng.normal(size=(2, 15))
        np.testing.assert_allclose(rsa(a, b, method='pearson'), stats.pearsonr(a, b).statistic, rtol=1e-12)

    def test_square_vs_condensed(self):
        dist = pytest.importorskip('scipy.spatial.distance')
        stats = pytest.importorskip('scipy.stats')
        x = _data(32, (6, 4))
        y = _data(33, (6, 4))
        ra, rb = rdm(x), rdm(y, 'euclidean')
        ca = dist.pdist(x, 'correlation')
        cb = dist.pdist(y, 'euclidean')
        np.testing.assert_allclose(rsa(ra, rb), stats.spearmanr(ca, cb).statistic, rtol=1e-5)
        np.testing.assert_allclose(rsa(ra, rb), rsa(ca, cb), rtol=1e-5)
        np.testing.assert_allclose(rsa(ra, rb, 'pearson'), stats.pearsonr(ca, cb).statistic, rtol=1e-5)

    def test_rank_helper_matches_rankdata(self):
        stats = pytest.importorskip('scipy.stats')
        from braintools.analysis.population._dimensionality import _average_ranks
        v = np.array([3., 1., 3., 2., 3., 0., 1.])
        np.testing.assert_allclose(_average_ranks(jnp.asarray(v)), stats.rankdata(v))

    def test_monotone_transform(self):
        a = rdm(_data(34, (5, 3)), 'euclidean')
        np.testing.assert_allclose(rsa(a, jnp.exp(a)), 1.0, rtol=1e-6)
        np.testing.assert_allclose(rsa(a, -a), -1.0, rtol=1e-6)

    def test_nan_and_constant(self):
        a = jnp.array([1., 2., jnp.nan, 4.])
        b = jnp.array([1., 3., 2., 4.])
        assert np.isnan(rsa(a, b))
        assert np.isnan(rsa(a, b, 'pearson'))
        assert np.isnan(rsa(jnp.ones(4), b))

    def test_units_cancel(self):
        a = rdm(_data(35, (5, 3)).astype(np.float32) * u.mV, 'euclidean')
        b = rdm(_data(36, (5, 3)).astype(np.float32), 'euclidean')
        out = rsa(a, b)
        assert not isinstance(out, u.Quantity)
        np.testing.assert_allclose(out, rsa(a.mantissa, b), rtol=1e-6)

    def test_jit_vmap(self):
        target = rdm(_data(37, (6, 3)))
        models = jnp.stack([rdm(_data(s, (6, 3))) for s in range(4)])
        out = jax.jit(jax.vmap(lambda m: rsa(m, target)))(models)
        assert out.shape == (4,)
        np.testing.assert_allclose(out[2], rsa(models[2], target), rtol=1e-6)

    def test_grad_pearson(self, x64):
        rng = np.random.default_rng(38)
        a, b = (jnp.asarray(v) for v in rng.normal(size=(2, 10)))
        check_grads(lambda v: rsa(v, b, 'pearson'), (a,), order=1, modes=['rev'])

    def test_errors(self):
        with pytest.raises(ValueError, match='method'):
            rsa(jnp.ones(3), jnp.ones(3), 'kendall')
        with pytest.raises(ValueError, match='same shape'):
            rsa(jnp.ones(3), jnp.ones(4))
        with pytest.raises(ValueError, match='square'):
            rsa(jnp.ones((3, 4)), jnp.ones((3, 4)))
