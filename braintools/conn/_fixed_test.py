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

"""Tests of the fixed-number connectivity generators."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.conn._fixed import (
    FixedOutDegree,
    FixedInDegree,
    FixedTotalNum,
    _check_count,
    _edge_distances,
)
from braintools.init import Constant, Normal, DistanceModulated, GaussianProfile

stats = pytest.importorskip('scipy.stats')


def _rows(primary, secondary, n_rows):
    """Group ``secondary`` by ``primary`` into an ``(n_rows, k)`` array (rows sorted)."""
    order = np.lexsort((secondary, primary))
    return secondary[order].reshape(n_rows, -1)


def _assert_distinct_rows(rows):
    rows = np.sort(rows, axis=1)
    assert not np.any(rows[:, 1:] == rows[:, :-1])


def _codes(result):
    return result.pre_indices * result.shape[1] + result.post_indices


# ----------------------------------------------------------------------------
# count validation
# ----------------------------------------------------------------------------

class TestCountValidation:
    @pytest.mark.parametrize('cls', [FixedOutDegree, FixedInDegree, FixedTotalNum])
    def test_negative_int_raises(self, cls):
        with pytest.raises(ValueError, match='non-negative'):
            cls(-1)

    @pytest.mark.parametrize('bad', [0.0, -0.5, 1.5])
    @pytest.mark.parametrize('cls', [FixedOutDegree, FixedInDegree, FixedTotalNum])
    def test_fraction_out_of_range_raises(self, cls, bad):
        with pytest.raises(ValueError, match=r'\(0, 1\]'):
            cls(bad)

    @pytest.mark.parametrize('bad', [True, '3', None, [3]])
    def test_bad_type_raises(self, bad):
        with pytest.raises(TypeError):
            FixedOutDegree(bad)

    def test_numpy_scalars_accepted(self):
        assert _check_count(np.int32(4), 'k') == 4
        assert isinstance(_check_count(np.int32(4), 'k'), int)
        assert _check_count(np.float32(0.5), 'k') == pytest.approx(0.5)

    def test_properties(self):
        assert FixedOutDegree(3).k == 3
        assert FixedInDegree(0.5).k == 0.5
        assert FixedTotalNum(7).n == 7


# ----------------------------------------------------------------------------
# FixedOutDegree
# ----------------------------------------------------------------------------

class TestFixedOutDegree:
    def test_exact_degree_distinct_sorted(self):
        res = FixedOutDegree(7, seed=0)(pre_size=30, post_size=50)
        assert res.n_connections == 210
        np.testing.assert_array_equal(np.bincount(res.pre_indices, minlength=30), 7)
        # sorted by pre then post
        np.testing.assert_array_equal(np.lexsort((res.post_indices, res.pre_indices)), np.arange(210))
        _assert_distinct_rows(_rows(res.pre_indices, res.post_indices, 30))
        assert res.post_indices.min() >= 0 and res.post_indices.max() < 50
        assert res.weights is None and res.delays is None
        assert res.metadata['pattern'] == 'fixed_out_degree'
        assert res.metadata['degree'] == 7
        assert res.metadata['with_replacement'] is False
        assert res.metadata['allow_self_connections'] is True
        assert res.model_type == 'point'

    @pytest.mark.parametrize('k', [3, 30, 39])  # small path, complement path, all but self
    def test_no_self_connections(self, k):
        res = FixedOutDegree(k, allow_self_connections=False, seed=1)(40, 40)
        assert not np.any(res.pre_indices == res.post_indices)
        np.testing.assert_array_equal(np.bincount(res.pre_indices, minlength=40), k)
        _assert_distinct_rows(_rows(res.pre_indices, res.post_indices, 40))

    def test_self_flag_ignored_for_different_sizes(self):
        res = FixedOutDegree(20, allow_self_connections=False, seed=0)(10, 20)
        # complete rows: every pre connects to all 20 posts, including index == pre
        assert np.any(res.pre_indices == res.post_indices)
        assert res.n_connections == 200

    def test_k_zero(self):
        res = FixedOutDegree(0, weight=1.0 * u.nS, seed=0)(5, 8)
        assert res.n_connections == 0
        assert res.weights is None
        assert res.metadata['degree'] == 0
        w = res.to_fixed_num_per_pre()
        assert w.indices.shape == (5, 0)

    def test_k_equals_post_num_complete_rows(self):
        res = FixedOutDegree(6, seed=0)(4, 6)
        np.testing.assert_array_equal(res.weight2dense(), np.ones((4, 6)))

    def test_k_too_large_raises(self):
        with pytest.raises(ValueError, match='exceeds the 6 available'):
            FixedOutDegree(7, seed=0)(4, 6)
        with pytest.raises(ValueError, match='self-connections excluded'):
            FixedOutDegree(6, allow_self_connections=False)(6, 6)

    def test_fraction(self):
        res = FixedOutDegree(0.25, seed=0)(pre_size=12, post_size=40)
        assert res.metadata['degree'] == 10
        np.testing.assert_array_equal(np.bincount(res.pre_indices, minlength=12), 10)
        # rounding instead of truncation: 0.29 * 100 = 28.999...
        assert FixedOutDegree(0.29, seed=0)(2, 100).metadata['degree'] == 29

    def test_fraction_of_allowed_targets_without_self(self):
        # the fraction applies to the allowed partners: 1.0 means "all other neurons"
        res = FixedOutDegree(1.0, allow_self_connections=False, seed=0)(5, 5)
        assert res.metadata['degree'] == 4 and res.n_connections == 20
        assert not np.any(res.pre_indices == res.post_indices)
        assert FixedOutDegree(0.5, allow_self_connections=False, seed=0)(11, 11).metadata['degree'] == 5
        # different sizes: nothing is excluded, so the fraction is of post_num
        assert FixedOutDegree(0.5, allow_self_connections=False, seed=0)(4, 10).metadata['degree'] == 5

    def test_multidimensional_sizes(self):
        res = FixedOutDegree(5, seed=0)(pre_size=(4, 5), post_size=(2, 3, 4))
        assert res.shape == (20, 24)
        assert res.pre_size == (4, 5) and res.post_size == (2, 3, 4)
        np.testing.assert_array_equal(np.bincount(res.pre_indices, minlength=20), 5)
        assert res.post_indices.max() < 24

    def test_sizes_from_constructor(self):
        res = FixedOutDegree(2, pre_size=6, post_size=9, seed=0)()
        assert res.shape == (6, 9) and res.n_connections == 12

    def test_seed_reproducible(self):
        a = FixedOutDegree(5, seed=3)(50, 60)
        b = FixedOutDegree(5, seed=3)(50, 60)
        c = FixedOutDegree(5, seed=4)(50, 60)
        np.testing.assert_array_equal(a.post_indices, b.post_indices)
        assert not np.array_equal(a.post_indices, c.post_indices)

    def test_uniform_targets_chi2(self):
        res = FixedOutDegree(10, seed=5)(pre_size=2000, post_size=50)
        counts = np.bincount(res.post_indices, minlength=50)
        assert stats.chisquare(counts).pvalue > 1e-3

    def test_uniform_targets_chi2_complement_path(self):
        # k > post_num / 2 uses the key-selection path
        res = FixedOutDegree(35, seed=6)(pre_size=500, post_size=50)
        counts = np.bincount(res.post_indices, minlength=50)
        assert stats.chisquare(counts).pvalue > 1e-3

    def test_uniform_without_self_chi2(self):
        n = 30
        res = FixedOutDegree(4, allow_self_connections=False, seed=7)(n, n)
        # offset (post - pre) mod n is uniform over 1..n-1 for a uniform draw
        offs = (res.post_indices - res.pre_indices) % n
        counts = np.bincount(offs, minlength=n)
        assert counts[0] == 0
        res = FixedOutDegree(4, allow_self_connections=False, seed=7)(3000, 3000)
        offs = (res.post_indices - res.pre_indices) % 3000
        hist = np.bincount(offs * 20 // 3000, minlength=20)
        # 20 bins of 150 offsets each; bin 0 lacks offset 0 (never drawn)
        expected = np.full(20, 150.0)
        expected[0] = 149.0
        expected *= res.n_connections / expected.sum()
        assert stats.chisquare(hist, expected).pvalue > 1e-3

    def test_with_replacement(self):
        res = FixedOutDegree(30, with_replacement=True, seed=0)(pre_size=50, post_size=20)
        np.testing.assert_array_equal(np.bincount(res.pre_indices, minlength=50), 30)
        rows = _rows(res.pre_indices, res.post_indices, 50)
        assert np.any(rows[:, 1:] == rows[:, :-1])  # k > post_num forces duplicates
        assert res.metadata['with_replacement'] is True

    def test_with_replacement_no_self(self):
        res = FixedOutDegree(15, with_replacement=True, allow_self_connections=False, seed=0)(10, 10)
        assert not np.any(res.pre_indices == res.post_indices)
        assert res.n_connections == 150

    def test_with_replacement_empty_pool_raises(self):
        with pytest.raises(ValueError, match='no post neurons available'):
            FixedOutDegree(1, with_replacement=True, allow_self_connections=False)(1, 1)
        # k = 0 is fine even with an empty pool
        assert FixedOutDegree(0, with_replacement=True, allow_self_connections=False)(1, 1).n_connections == 0

    def test_homogeneous_weight_stays_scalar(self):
        res = FixedOutDegree(3, weight=0.5 * u.nS, delay=1.0 * u.ms, seed=0)(6, 9)
        assert u.math.size(res.weights) == 1
        w = res.to_fixed_num_per_pre()
        assert u.math.shape(w.data) == ()
        assert u.get_unit(w.data) == u.nS
        assert u.math.allclose(w.todense(), res.weight2dense())

    def test_heterogeneous_quantity_weights_and_delays(self):
        res = FixedOutDegree(
            4, weight=Normal(1.0 * u.nS, 0.2 * u.nS), delay=Normal(2.0 * u.ms, 0.1 * u.ms), seed=0
        )(10, 12)
        assert u.math.shape(res.weights) == (40,)
        assert u.get_unit(res.weights) == u.nS
        assert u.get_unit(res.delays) == u.ms
        w = res.to_fixed_num_per_pre()
        assert w.data.shape == (10, 4)
        assert u.math.allclose(w.todense(), res.weight2dense(), rtol=1e-5)
        d = res.to_fixed_num_per_pre(data='delay')
        assert u.math.allclose(d.todense(), res.delay2matrix(), rtol=1e-5)

    def test_matvec_matches_dense(self):
        res = FixedOutDegree(5, weight=Normal(0.0, 1.0), seed=0)(20, 30)
        w = res.to_fixed_num_per_pre()
        dense = np.asarray(res.weight2dense())
        x = np.random.RandomState(0).randn(8, 30).astype(np.float32)
        y = jax.vmap(lambda v: w @ v)(jnp.asarray(x))
        np.testing.assert_allclose(np.asarray(y), x @ dense.T, rtol=1e-4, atol=1e-5)
        spikes = (np.random.RandomState(1).rand(4, 20) < 0.3).astype(np.float32)
        z = jax.vmap(lambda s: s @ w)(jnp.asarray(spikes))
        np.testing.assert_allclose(np.asarray(z), spikes @ dense, rtol=1e-4, atol=1e-5)

    def test_distance_dependent_weights_use_edge_distances(self):
        rng = np.random.RandomState(0)
        pre_pos = rng.uniform(0, 100, (15, 2)) * u.um
        post_pos = rng.uniform(0, 100, (20, 2)) * u.um
        init = DistanceModulated(Constant(1.0 * u.nS), GaussianProfile(sigma=30 * u.um))
        res = FixedOutDegree(4, weight=init, seed=0)(15, 20, pre_pos, post_pos)
        d = np.linalg.norm(
            pre_pos.mantissa[res.pre_indices] - post_pos.mantissa[res.post_indices], axis=1
        )
        expected = np.exp(-d ** 2 / (2 * 30.0 ** 2))
        np.testing.assert_allclose(u.get_mantissa(res.weights.to(u.nS)), expected, rtol=1e-6)


# ----------------------------------------------------------------------------
# FixedInDegree
# ----------------------------------------------------------------------------

class TestFixedInDegree:
    def test_exact_in_degree_grouped_by_post(self):
        res = FixedInDegree(6, seed=0)(pre_size=40, post_size=25)
        assert res.n_connections == 150
        np.testing.assert_array_equal(np.bincount(res.post_indices, minlength=25), 6)
        np.testing.assert_array_equal(res.post_indices, np.repeat(np.arange(25), 6))
        _assert_distinct_rows(_rows(res.post_indices, res.pre_indices, 25))
        assert res.metadata['pattern'] == 'fixed_in_degree'
        assert res.metadata['degree'] == 6

    def test_no_self(self):
        res = FixedInDegree(9, allow_self_connections=False, seed=2)(10, 10)
        assert not np.any(res.pre_indices == res.post_indices)
        np.testing.assert_array_equal(res.weight2dense(), 1.0 - np.eye(10))

    def test_fraction_of_allowed_sources_without_self(self):
        res = FixedInDegree(1.0, allow_self_connections=False, seed=0)(5, 5)
        assert res.metadata['degree'] == 4
        assert not np.any(res.pre_indices == res.post_indices)

    def test_fraction_of_pre(self):
        res = FixedInDegree(0.5, seed=0)(pre_size=30, post_size=7)
        assert res.metadata['degree'] == 15

    def test_too_large_raises(self):
        with pytest.raises(ValueError, match='pre neurons'):
            FixedInDegree(11)(10, 3)

    def test_uniform_sources_chi2(self):
        res = FixedInDegree(10, seed=5)(pre_size=50, post_size=2000)
        counts = np.bincount(res.pre_indices, minlength=50)
        assert stats.chisquare(counts).pvalue > 1e-3

    def test_with_replacement(self):
        res = FixedInDegree(12, with_replacement=True, allow_self_connections=False, seed=1)(8, 8)
        np.testing.assert_array_equal(np.bincount(res.post_indices, minlength=8), 12)
        assert not np.any(res.pre_indices == res.post_indices)

    def test_export_round_trip_and_matvec(self):
        res = FixedInDegree(4, weight=Normal(0.5 * u.nS, 0.1 * u.nS), seed=3)(pre_size=(3, 5), post_size=12)
        w = res.to_fixed_num_per_post()
        assert w.shape == (15, 12) and w.indices.shape == (12, 4)
        assert u.math.allclose(w.todense(), res.weight2dense(), rtol=1e-5)
        dense = u.get_mantissa(res.weight2dense().to(u.nS))
        x = np.random.RandomState(0).rand(5, 15).astype(np.float32)
        y = jax.vmap(lambda v: v @ w)(jnp.asarray(x))
        np.testing.assert_allclose(u.get_mantissa(y.to(u.nS)), x @ dense, rtol=1e-4, atol=1e-5)
        # fixed in-degree does not have uniform out-degree in general
        with pytest.raises(ValueError, match='observed range'):
            res.to_fixed_num_per_pre()

    def test_csr_and_csc_round_trip(self):
        res = FixedInDegree(3, weight=Normal(0.0, 1.0), seed=0)(9, 7)
        dense = np.asarray(res.weight2dense())
        np.testing.assert_allclose(np.asarray(res.to_csc().todense()), dense, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(res.weight2csr().todense()), dense, rtol=1e-6)
        np.testing.assert_allclose(res.to_scipy_csr().toarray(), dense)


# ----------------------------------------------------------------------------
# FixedTotalNum
# ----------------------------------------------------------------------------

class TestFixedTotalNum:
    @pytest.mark.parametrize('n', [50, 500])  # rejection path, permutation path
    def test_exact_distinct_sorted(self, n):
        res = FixedTotalNum(n, seed=0)(pre_size=20, post_size=30)
        assert res.n_connections == n
        codes = _codes(res)
        assert np.unique(codes).size == n
        assert np.all(np.diff(codes) > 0)
        assert res.metadata['pattern'] == 'fixed_total_num'
        assert res.metadata['n'] == n

    @pytest.mark.parametrize('n', [30, 380])
    def test_no_self(self, n):
        res = FixedTotalNum(n, allow_self_connections=False, seed=1)(20, 20)
        assert res.n_connections == n
        assert not np.any(res.pre_indices == res.post_indices)
        assert np.unique(_codes(res)).size == n

    def test_all_pairs_without_self(self):
        res = FixedTotalNum(20 * 19, allow_self_connections=False, seed=0)(20, 20)
        np.testing.assert_array_equal(res.weight2dense(), 1.0 - np.eye(20))

    def test_fraction(self):
        res = FixedTotalNum(0.1, seed=0)(20, 30)
        assert res.n_connections == 60

    def test_fraction_of_allowed_pairs_without_self(self):
        res = FixedTotalNum(1.0, allow_self_connections=False, seed=0)(6, 6)
        assert res.n_connections == 30 and not np.any(res.pre_indices == res.post_indices)

    def test_too_many_raises(self):
        with pytest.raises(ValueError, match='available pairs'):
            FixedTotalNum(601)(20, 30)
        with pytest.raises(ValueError, match='self-connections excluded'):
            FixedTotalNum(91, allow_self_connections=False)(10, 10)

    def test_zero(self):
        res = FixedTotalNum(0, seed=0)(5, 5)
        assert res.n_connections == 0

    def test_with_replacement(self):
        res = FixedTotalNum(100, with_replacement=True, seed=0)(5, 5)
        assert res.n_connections == 100
        assert np.unique(_codes(res)).size < 100
        assert np.all(np.diff(_codes(res)) >= 0)
        assert FixedTotalNum(0, with_replacement=True)(3, 3).n_connections == 0
        with pytest.raises(ValueError, match='no pairs available'):
            FixedTotalNum(1, with_replacement=True, allow_self_connections=False)(1, 1)

    @pytest.mark.parametrize('n', [4000, 8000])
    def test_uniform_marginals_chi2(self, n):
        res = FixedTotalNum(n, seed=11)(100, 100)
        for idx in (res.pre_indices, res.post_indices):
            assert stats.chisquare(np.bincount(idx, minlength=100)).pvalue > 1e-3

    def test_weights(self):
        res = FixedTotalNum(40, weight=Normal(1.0, 0.1), delay=0.5 * u.ms, seed=0)(10, 10)
        assert u.math.shape(res.weights) == (40,)
        assert u.math.size(res.delays) == 1
        np.testing.assert_allclose(np.asarray(res.to_csc().todense()), np.asarray(res.weight2dense()),
                                   rtol=1e-6)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

class TestEdgeDistances:
    def test_none_without_positions(self):
        assert _edge_distances(None, np.zeros((2, 2)), np.array([0]), np.array([1])) is None

    def test_one_dimensional_unitless(self):
        d = _edge_distances(np.array([0.0, 1.0]), np.array([3.0, 5.0]), np.array([0, 1]), np.array([1, 0]))
        np.testing.assert_allclose(d, [5.0, 2.0])

    def test_unit_conversion_and_mismatch(self):
        d = _edge_distances(np.array([[0.0]]) * u.mm, np.array([[1000.0]]) * u.um, np.array([0]), np.array([0]))
        assert u.math.allclose(d, 1.0 * u.mm)
        with pytest.raises(u.UnitMismatchError):
            _edge_distances(np.array([[0.0]]) * u.mm, np.array([[1.0]]) * u.ms, np.array([0]), np.array([0]))
