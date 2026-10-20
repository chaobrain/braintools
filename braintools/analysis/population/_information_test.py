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

from braintools.analysis.population._information import (
    _equal_width_bin,
    _pair_counts,
    mutual_information,
)


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def _labels(seed, n=300):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 5, n)
    y = (x + rng.integers(0, 3, n)) % 4
    return x, y


class TestDiscrete:
    def test_matches_sklearn_float64(self, x64):
        metrics = pytest.importorskip('sklearn.metrics')
        x, y = _labels(0)
        np.testing.assert_allclose(mutual_information(x, y), metrics.mutual_info_score(x, y), rtol=1e-12)

    def test_matches_sklearn_float32(self):
        metrics = pytest.importorskip('sklearn.metrics')
        x, y = _labels(1)
        out = mutual_information(x, y)
        assert out.dtype == jnp.float32
        np.testing.assert_allclose(out, metrics.mutual_info_score(x, y), rtol=1e-5)

    def test_arbitrary_label_values(self, x64):
        metrics = pytest.importorskip('sklearn.metrics')
        x, y = _labels(2)
        xs = np.array([-7, 3, 1_000_000, 42, 5])[x]  # negative, huge, unsorted symbols
        ys = np.array([0.5, -1.25, 9.0, 3.0])[y]       # float symbols
        np.testing.assert_allclose(mutual_information(xs, ys), metrics.mutual_info_score(x, y), rtol=1e-12)

    def test_analytic(self):
        x = jnp.array([0, 0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_allclose(mutual_information(x, x), np.log(4), rtol=1e-6)
        np.testing.assert_allclose(mutual_information(x, 10 - x), np.log(4), rtol=1e-6)
        # Exactly independent (balanced product design).
        a = jnp.array([0, 0, 1, 1])
        b = jnp.array([0, 1, 0, 1])
        assert float(mutual_information(a, b)) == 0.0
        # A constant variable carries no information.
        assert float(mutual_information(x, jnp.zeros(8, dtype=int))) == 0.0
        # Single sample.
        assert float(mutual_information(jnp.array([3]), jnp.array([1]))) == 0.0

    def test_symmetric(self):
        x, y = _labels(3)
        np.testing.assert_allclose(mutual_information(x, y), mutual_information(y, x), rtol=1e-6)

    def test_pair_counts(self):
        a = jnp.array([1, 0, 1, 1, 0, 2])
        b = jnp.array([5, 5, 5, 4, 5, 5])
        np.testing.assert_array_equal(_pair_counts(a, b), [2, 2, 2, 1, 2, 1])


class TestBinned:
    @pytest.mark.parametrize('bins', [1, 4, (5, 7), [3, 2]])
    def test_matches_histogram2d(self, x64, bins):
        metrics = pytest.importorskip('sklearn.metrics')
        rng = np.random.default_rng(10)
        x = rng.normal(size=500)
        y = np.sin(2 * x) + 0.3 * rng.normal(size=500)
        contingency = np.histogram2d(x, y, bins=bins)[0]
        ref = metrics.mutual_info_score(None, None, contingency=contingency)
        np.testing.assert_allclose(mutual_information(x, y, bins=bins), ref, rtol=1e-10, atol=1e-15)

    def test_float32(self):
        metrics = pytest.importorskip('sklearn.metrics')
        rng = np.random.default_rng(11)
        x = rng.uniform(size=200).astype(np.float32)
        y = (x ** 2 + 0.05 * rng.normal(size=200)).astype(np.float32)
        ref = metrics.mutual_info_score(None, None, contingency=np.histogram2d(x, y, bins=6)[0])
        np.testing.assert_allclose(mutual_information(x, y, bins=6), ref, rtol=1e-5)

    def test_values_on_bin_edges(self, x64):
        v = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.5])
        # Half-open bins, last bin closed: 1.0 falls in the last bin.
        np.testing.assert_array_equal(_equal_width_bin(jnp.asarray(v), 4), [0, 1, 2, 3, 3, 2])
        # Same as numpy's per-sample bin assignment.
        edges = np.histogram_bin_edges(v, bins=4)
        expected = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, 3)
        np.testing.assert_array_equal(_equal_width_bin(jnp.asarray(v), 4), expected)

    def test_constant_input(self):
        x = jnp.full(10, 2.5)
        np.testing.assert_array_equal(_equal_width_bin(x, 3), np.full(10, 1))  # numpy's middle bin
        assert float(mutual_information(x, jnp.arange(10.0), bins=3)) == 0.0

    def test_integer_input_binned(self):
        x = jnp.arange(12)
        np.testing.assert_allclose(mutual_information(x, x, bins=3), np.log(3), rtol=1e-6)

    def test_quantity_input(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=100).astype(np.float32)
        y = (x + rng.normal(size=100)).astype(np.float32)
        out = mutual_information(x * u.mV, y * u.second, bins=5)
        assert not isinstance(out, u.Quantity)
        np.testing.assert_allclose(out, mutual_information(x, y, bins=5), rtol=1e-6)


class TestTransforms:
    def test_jit(self):
        x, y = _labels(20)
        np.testing.assert_allclose(jax.jit(mutual_information)(x, y), mutual_information(x, y), rtol=1e-6)
        xf = np.random.default_rng(0).normal(size=100)
        f = jax.jit(lambda a, b: mutual_information(a, b, bins=(4, 5)))
        np.testing.assert_allclose(f(xf, xf ** 2), mutual_information(xf, xf ** 2, bins=(4, 5)), rtol=1e-6)

    def test_vmap_over_trials(self):
        metrics = pytest.importorskip('sklearn.metrics')
        pairs = [_labels(s, n=80) for s in range(4)]
        xs = np.stack([p[0] for p in pairs])
        ys = np.stack([p[1] for p in pairs])
        out = jax.vmap(mutual_information)(xs, ys)
        np.testing.assert_allclose(out, [metrics.mutual_info_score(a, b) for a, b in pairs], rtol=1e-5)

    def test_errors(self):
        with pytest.raises(ValueError, match='same length'):
            mutual_information(jnp.arange(3), jnp.arange(4))
        with pytest.raises(ValueError, match='same length'):
            mutual_information(jnp.ones((2, 2)), jnp.ones((2, 2)))
        with pytest.raises(ValueError, match='empty'):
            mutual_information(jnp.zeros(0), jnp.zeros(0))
        with pytest.raises(ValueError, match='>= 1'):
            mutual_information(jnp.arange(3.), jnp.arange(3.), bins=0)
        with pytest.raises(ValueError, match='pair'):
            mutual_information(jnp.arange(3.), jnp.arange(3.), bins=(1, 2, 3))
        with pytest.raises(TypeError, match='static ints'):
            mutual_information(jnp.arange(3.), jnp.arange(3.), bins=2.5)
        with pytest.raises(TypeError, match='static ints'):
            mutual_information(jnp.arange(3.), jnp.arange(3.), bins=True)
