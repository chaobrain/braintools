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

import numpy as np
import pytest

from braintools.conn._sampling import (
    MAX_CHUNK_PAIRS,
    bernoulli_pairs,
    randint,
    row_chunks,
    sample_flat_without_replacement,
    sample_rows_without_replacement,
)

RNGS = [lambda s: np.random.RandomState(s), lambda s: np.random.default_rng(s)]


def test_row_chunks_cover_and_bound():
    chunks = list(row_chunks(10, 3, max_pairs=7))
    assert chunks == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    assert all((b - a) * 3 <= 7 for a, b in chunks)
    # at least one row per chunk even when a single row exceeds the bound
    assert list(row_chunks(3, 100, max_pairs=10)) == [(0, 1), (1, 2), (2, 3)]
    assert list(row_chunks(0, 5)) == []
    assert list(row_chunks(4, 0)) == [(0, 4)]
    assert MAX_CHUNK_PAIRS == 2 ** 22


@pytest.mark.parametrize('make', RNGS)
def test_randint_dtype_and_range(make):
    x = randint(make(0), 5, 1000)
    assert x.dtype == np.int64 and x.min() >= 0 and x.max() < 5
    big = randint(make(0), 2 ** 40, 10)
    assert big.max() < 2 ** 40


@pytest.mark.parametrize('make', RNGS)
@pytest.mark.parametrize('k', [0, 1, 3, 10, 17, 20])
def test_rows_distinct_sorted_in_range(make, k):
    out = sample_rows_without_replacement(make(1), 50, 20, k)
    assert out.shape == (50, k) and out.dtype == np.int64
    if k:
        assert out.min() >= 0 and out.max() < 20
        assert np.all(np.diff(out, axis=1) > 0)
    if k == 20:
        np.testing.assert_array_equal(out, np.tile(np.arange(20), (50, 1)))


@pytest.mark.parametrize('k', [3, 15])
def test_rows_exclude_self(k):
    n = 16
    out = sample_rows_without_replacement(np.random.RandomState(2), n, n, k, exclude=np.arange(n))
    assert not np.any(out == np.arange(n)[:, None])
    assert np.all(np.diff(out, axis=1) > 0) and out.max() < n


@pytest.mark.parametrize('k', [2, 7])
def test_rows_uniform(k):
    # every column appears with frequency k / n_cols (chi-square on the marginals)
    n_rows, n_cols = 4000, 10
    out = sample_rows_without_replacement(np.random.RandomState(3), n_rows, n_cols, k)
    counts = np.bincount(out.ravel(), minlength=n_cols)
    expected = n_rows * k / n_cols
    chi2 = np.sum((counts - expected) ** 2 / expected)
    assert chi2 < 30.0  # chi2(9) 99.9% quantile ~ 27.9


def test_rows_subsets_uniform():
    # all C(5, 2) = 10 subsets appear equally often
    out = sample_rows_without_replacement(np.random.RandomState(4), 20000, 5, 2)
    codes = out[:, 0] * 5 + out[:, 1]
    _, counts = np.unique(codes, return_counts=True)
    assert counts.size == 10
    assert np.all(np.abs(counts - 2000) < 5 * np.sqrt(2000))


def test_rows_errors():
    with pytest.raises(ValueError, match='non-negative'):
        sample_rows_without_replacement(np.random.RandomState(0), 3, 5, -1)
    with pytest.raises(ValueError, match='6 distinct'):
        sample_rows_without_replacement(np.random.RandomState(0), 3, 5, 6)
    with pytest.raises(ValueError, match='self excluded'):
        sample_rows_without_replacement(np.random.RandomState(0), 3, 5, 5, exclude=np.arange(3))
    assert sample_rows_without_replacement(np.random.RandomState(0), 0, 5, 2).shape == (0, 2)


def test_rows_chunked_complement_branch(monkeypatch):
    import braintools.conn._sampling as s
    monkeypatch.setattr(s, 'MAX_CHUNK_PAIRS', 16)

    def small_chunks(n_rows, n_cols, max_pairs=16):
        return row_chunks(n_rows, n_cols, max_pairs)

    monkeypatch.setattr(s, 'row_chunks', small_chunks)
    out = s.sample_rows_without_replacement(np.random.RandomState(5), 9, 8, 6)
    assert out.shape == (9, 6) and np.all(np.diff(out, axis=1) > 0)


@pytest.mark.parametrize('make', RNGS)
@pytest.mark.parametrize('n', [0, 1, 40, 99, 100])
def test_flat_distinct_sorted(make, n):
    out = sample_flat_without_replacement(make(6), 100, n)
    assert out.shape == (n,) and out.dtype == np.int64
    assert np.all(np.diff(out) > 0)
    if n:
        assert out.min() >= 0 and out.max() < 100


def test_flat_large_space_and_errors():
    out = sample_flat_without_replacement(np.random.RandomState(7), 10 ** 12, 1000)
    assert out.size == 1000 and np.unique(out).size == 1000 and out.max() < 10 ** 12
    with pytest.raises(ValueError):
        sample_flat_without_replacement(np.random.RandomState(0), 5, 6)
    with pytest.raises(ValueError):
        sample_flat_without_replacement(np.random.RandomState(0), 5, -1)


def test_flat_uniform():
    counts = np.zeros(20)
    rng = np.random.RandomState(8)
    for _ in range(2000):
        counts[sample_flat_without_replacement(rng, 20, 3)] += 1
    expected = 2000 * 3 / 20
    assert np.sum((counts - expected) ** 2 / expected) < 45.0  # chi2(19) 99.9% ~ 43.8


@pytest.mark.parametrize('make', RNGS)
def test_bernoulli_pairs_statistics(make):
    pre, post = bernoulli_pairs(make(9), 300, 200, 0.05)
    assert pre.dtype == np.int64 and post.dtype == np.int64
    n = pre.size
    mean, sd = 300 * 200 * 0.05, np.sqrt(300 * 200 * 0.05 * 0.95)
    assert abs(n - mean) < 5 * sd
    codes = pre * 200 + post
    assert np.all(np.diff(codes) > 0)  # sorted, unique
    assert pre.max() < 300 and post.max() < 200


def test_bernoulli_pairs_no_self_and_edges():
    pre, post = bernoulli_pairs(np.random.RandomState(10), 50, 50, 1.0, allow_self_connections=False)
    assert pre.size == 50 * 49 and not np.any(pre == post)
    pre, post = bernoulli_pairs(np.random.RandomState(10), 50, 50, 0.3, allow_self_connections=False)
    assert not np.any(pre == post)
    assert bernoulli_pairs(np.random.RandomState(0), 10, 10, 0.0)[0].size == 0
    assert bernoulli_pairs(np.random.RandomState(0), 0, 10, 0.5)[0].size == 0
    assert bernoulli_pairs(np.random.RandomState(0), 1, 1, 1.0, allow_self_connections=False)[0].size == 0
    with pytest.raises(ValueError, match='p must be'):
        bernoulli_pairs(np.random.RandomState(0), 3, 3, 1.5)
    with pytest.raises(ValueError, match='n_pre == n_post'):
        bernoulli_pairs(np.random.RandomState(0), 3, 4, 0.5, allow_self_connections=False)


def test_bernoulli_pairs_large_block_is_memory_light():
    import tracemalloc
    tracemalloc.start()
    pre, post = bernoulli_pairs(np.random.RandomState(11), 100_000, 100_000, 1e-5)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert abs(pre.size - 1e5) < 5 * np.sqrt(1e5)
    assert peak < 50 * 2 ** 20  # a dense mask would need ~10 GB


class _Long32RandomState(np.random.RandomState):
    """``RandomState`` whose ``binomial`` rejects counts beyond a 32-bit C ``long``, as on Windows."""

    def binomial(self, n, p, size=None):
        if np.any(np.asarray(n) > np.iinfo(np.int32).max):
            raise ValueError('n is out of bounds for long')
        return super().binomial(n, p, size)


def test_bernoulli_pairs_edge_count_beyond_int32_with_legacy_rng():
    # 1e5 x 1e5 = 1e10 pairs: the legacy binomial must not see n > 2**31 - 1.
    pre, post = bernoulli_pairs(_Long32RandomState(11), 100_000, 100_000, 1e-5)
    assert abs(pre.size - 1e5) < 5 * np.sqrt(1e5)
    assert pre.max() < 100_000 and post.max() < 100_000
    pre, post = bernoulli_pairs(_Long32RandomState(3), 70_000, 70_000, 1.0 / 70_000,
                                allow_self_connections=False)
    assert not np.any(pre == post)
    # p = 1 over a block larger than int32 must still count every pair exactly.
    from braintools.conn._sampling import _binomial
    assert _binomial(_Long32RandomState(0), 3 * 2 ** 31, 1.0) == 3 * 2 ** 31
    assert _binomial(_Long32RandomState(0), 3 * 2 ** 31, 0.0) == 0


def test_binomial_split_matches_distribution():
    # Summing chunked binomials is exact: mean and variance of Binomial(n, p).
    from braintools.conn._sampling import _binomial
    rng = _Long32RandomState(5)
    n, p = 5 * 2 ** 31, 2e-9
    draws = np.array([_binomial(rng, n, p) for _ in range(400)], dtype=float)
    mean, var = n * p, n * p * (1 - p)
    assert abs(draws.mean() - mean) < 5 * np.sqrt(var / draws.size)
    assert 0.6 * var < draws.var() < 1.5 * var
    assert _binomial(np.random.default_rng(0), n, 1.0) == n


def test_sorted_unique_matches_numpy():
    from braintools.conn._sampling import _sorted_unique
    x = np.random.RandomState(12).randint(0, 50, 1000).astype(np.int64)
    np.testing.assert_array_equal(_sorted_unique(x), np.unique(x))
    assert _sorted_unique(np.zeros(0, np.int64)).size == 0
    np.testing.assert_array_equal(_sorted_unique(np.array([3], np.int64)), [3])
