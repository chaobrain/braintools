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

from braintools.analysis.population._decoding import decode


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def _dataset(seed, n_samples=103, n_features=6, n_classes=3, sep=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n_samples)
    centres = rng.normal(size=(n_classes, n_features)) * sep
    x = rng.normal(size=(n_samples, n_features)) + centres[y]
    return x, y


def _sklearn_folds(x, y, n_folds, make_model, perm=None):
    """Per-fold accuracy of a sklearn model on the same folds as `decode`."""
    if perm is not None:
        x, y = x[perm], y[perm]
    fold = x.shape[0] // n_folds
    n_used = fold * n_folds
    x, y = x[:n_used], y[:n_used]
    acc = []
    for f in range(n_folds):  # reference only; decode itself is vmapped
        test = np.arange(f * fold, (f + 1) * fold)
        train = np.setdiff1d(np.arange(n_used), test)
        model = make_model().fit(x[train], y[train])
        acc.append(np.mean(model.predict(x[test]) == y[test]))
    return np.asarray(acc)


def _lda(reg):
    da = pytest.importorskip('sklearn.discriminant_analysis')
    return lambda: da.LinearDiscriminantAnalysis(solver='lsqr', shrinkage=reg)


def _ridge(reg):
    lm = pytest.importorskip('sklearn.linear_model')
    return lambda: lm.RidgeClassifier(alpha=reg)


class TestAgainstSklearn:
    @pytest.mark.parametrize('reg', [0.0, 1e-3, 0.3, 1.0])
    @pytest.mark.parametrize('n_classes', [2, 4])
    def test_lda(self, x64, reg, n_classes):
        x, y = _dataset(0, n_classes=n_classes)
        acc = decode(x, y, n_folds=5, model='lda', reg=reg)
        np.testing.assert_allclose(acc, _sklearn_folds(x, y, 5, _lda(reg)), atol=1e-9)

    @pytest.mark.parametrize('reg', [0.0, 1e-3, 1.0, 50.0])
    @pytest.mark.parametrize('n_classes', [2, 4])
    def test_ridge(self, x64, reg, n_classes):
        x, y = _dataset(1, n_classes=n_classes)
        acc = decode(x, y, n_folds=5, model='ridge', reg=reg)
        np.testing.assert_allclose(acc, _sklearn_folds(x, y, 5, _ridge(reg)), atol=1e-9)

    @pytest.mark.parametrize('model,make', [('lda', _lda), ('ridge', _ridge)])
    def test_more_features_than_samples(self, x64, model, make):
        x, y = _dataset(2, n_samples=40, n_features=50, sep=0.5)
        acc = decode(x, y, n_folds=4, model=model, reg=0.5)
        np.testing.assert_allclose(acc, _sklearn_folds(x, y, 4, make(0.5)), atol=1e-9)

    @pytest.mark.parametrize('model,make', [('lda', _lda), ('ridge', _ridge)])
    def test_float32_matches(self, model, make):
        x, y = _dataset(3, sep=2.0)
        acc = decode(x.astype(np.float32), y, n_folds=5, model=model, reg=0.1)
        assert acc.dtype == jnp.float32
        np.testing.assert_allclose(acc, _sklearn_folds(x, y, 5, make(0.1)))

    @pytest.mark.parametrize('model,make', [('lda', _lda), ('ridge', _ridge)])
    def test_key_permutation(self, x64, model, make):
        x, y = _dataset(4)
        key = jax.random.PRNGKey(7)
        acc = decode(x, y, n_folds=3, model=model, reg=0.2, key=key)
        perm = np.asarray(jax.random.permutation(key, x.shape[0]))
        np.testing.assert_allclose(acc, _sklearn_folds(x, y, 3, make(0.2), perm=perm), atol=1e-9)

    @pytest.mark.parametrize('model,make', [('lda', _lda), ('ridge', _ridge)])
    def test_class_absent_from_training_fold(self, x64, model, make):
        # Class 3 occurs only in the first fold, so it is absent from the
        # training set of fold 0 (and present in the training sets of the
        # others).
        x, y = _dataset(5, n_samples=60, n_classes=3, sep=2.0)
        x = np.concatenate([x[:12] + 4.0, x])
        y = np.concatenate([np.full(12, 3), y])
        y[12:] = np.where(y[12:] == 3, 0, y[12:])
        acc = decode(x, y, n_folds=6, model=model, reg=0.1)
        assert np.all(np.isfinite(acc))
        np.testing.assert_allclose(acc, _sklearn_folds(x, y, 6, make(0.1)), atol=1e-9)
        # The absent class is never predicted, so fold 0 (all class 3) is 0 accurate.
        assert acc[0] == 0.0

    def test_extra_unused_class(self, x64):
        # n_classes larger than the observed labels: the empty classes never win.
        x, y = _dataset(6)
        acc = decode(x, y, n_classes=6, n_folds=5, reg=0.1)
        np.testing.assert_array_equal(acc, decode(x, y, n_folds=5, reg=0.1))


class TestBehaviour:
    def test_drops_trailing_samples(self):
        x, y = _dataset(10, n_samples=23)
        acc = decode(x, y, n_folds=5)
        np.testing.assert_array_equal(acc, decode(x[:20], y[:20], n_folds=5))
        # Every fold has 4 test samples, so accuracies are multiples of 1/4.
        np.testing.assert_allclose(np.asarray(acc) * 4, np.round(np.asarray(acc) * 4), atol=1e-6)

    def test_separable_is_perfect(self):
        y = jnp.arange(60) % 3
        x = jax.nn.one_hot(y, 3) * 10.0 + 0.01 * jax.random.normal(jax.random.PRNGKey(0), (60, 3))
        for model in ('lda', 'ridge'):
            np.testing.assert_allclose(decode(x, y, model=model), 1.0)

    def test_one_dimensional_x(self):
        y = jnp.arange(40) % 2
        x = y * 5.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (40,))
        np.testing.assert_allclose(decode(x, y, n_folds=4), 1.0)

    def test_integer_counts_input(self):
        # Spike counts (integer dtype) are promoted to float.
        rng = np.random.default_rng(16)
        y = np.arange(60) % 2
        x = rng.poisson(lam=np.where(y[:, None] == 1, [8.0, 2.0], [2.0, 8.0]))
        assert np.issubdtype(x.dtype, np.integer)
        acc = decode(x, y, n_folds=3)
        assert jnp.issubdtype(acc.dtype, jnp.floating)
        np.testing.assert_allclose(acc, decode(x.astype(np.float32), y, n_folds=3))

    def test_quantity_input(self):
        x, y = _dataset(11)
        x = x.astype(np.float32)
        np.testing.assert_array_equal(decode(x * u.mV, y, model='ridge'), decode(x, y, model='ridge'))

    def test_jit_requires_n_classes(self):
        x, y = _dataset(12)
        with pytest.raises(ValueError, match='n_classes'):
            jax.jit(lambda a, b: decode(a, b))(x, y)

    def test_jit(self):
        x, y = _dataset(13)
        key = jax.random.PRNGKey(3)
        f = jax.jit(lambda a, b, k, r: decode(a, b, n_classes=3, n_folds=4, model='ridge', reg=r, key=k))
        np.testing.assert_allclose(f(x, y, key, 0.5),
                                   decode(x, y, n_folds=4, model='ridge', reg=0.5, key=key), atol=1e-6)

    def test_vmap_over_sessions(self):
        data = [_dataset(s, n_samples=50) for s in range(3)]
        xs = jnp.asarray(np.stack([d[0] for d in data]), dtype=jnp.float32)
        ys = jnp.asarray(np.stack([d[1] for d in data]))
        out = jax.vmap(lambda a, b: decode(a, b, n_classes=3, n_folds=5))(xs, ys)
        assert out.shape == (3, 5)
        for i in range(3):
            np.testing.assert_allclose(out[i], decode(xs[i], ys[i], n_classes=3, n_folds=5))

    def test_vmap_over_keys(self):
        x, y = _dataset(14)
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        out = jax.vmap(lambda k: decode(x, y, n_classes=3, key=k))(keys)
        assert out.shape == (4, 5)
        np.testing.assert_allclose(out[2], decode(x, y, key=keys[2]))

    def test_errors(self):
        x, y = _dataset(15, n_samples=20)
        with pytest.raises(ValueError, match='model'):
            decode(x, y, model='svm')
        with pytest.raises(TypeError, match='integer'):
            decode(x, y.astype(float))
        with pytest.raises(ValueError, match='shape'):
            decode(x, y[:-1])
        with pytest.raises(ValueError, match='shape'):
            decode(x[None], y)
        with pytest.raises(ValueError, match='n_folds'):
            decode(x, y, n_folds=1)
        with pytest.raises(TypeError, match='static int'):
            decode(x, y, n_folds=2.5)
        with pytest.raises(TypeError, match='static int'):
            decode(x, y, n_classes=True)
        with pytest.raises(ValueError, match='at least'):
            decode(x[:3], y[:3], n_folds=5)
        with pytest.raises(ValueError, match=r'\[0, 1\]'):
            decode(x, y, model='lda', reg=1.5)
        with pytest.raises(ValueError, match='non-negative'):
            decode(x, y, model='ridge', reg=-1.0)
