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

"""Tests of populations, projections and network assembly."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.conn._fixed import FixedInDegree, FixedOutDegree, FixedTotalNum
from braintools.conn._network import Population, Projection, Network
from braintools.init import Normal


def _csr_edges(csr):
    indptr = np.asarray(csr.indptr)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return rows, np.asarray(csr.indices)


def _ei_network(n_e=800, n_i=200, k_e=80, k_i=20):
    E = Population(n_e, name='E')
    I = Population(n_i, name='I')
    projs = [
        Projection(E, E, FixedInDegree(k_e, weight=0.6 * u.mS, seed=1)),
        Projection(E, I, FixedInDegree(k_e, weight=0.6 * u.mS, seed=2)),
        Projection(I, E, FixedInDegree(k_i, weight=-6.7 * u.mS, seed=3)),
        Projection(I, I, FixedInDegree(k_i, weight=-6.7 * u.mS, seed=4)),
    ]
    return Network([E, I], projs)


# ----------------------------------------------------------------------------
# Population
# ----------------------------------------------------------------------------

class TestPopulation:
    def test_int_and_tuple_sizes(self):
        assert Population(5, name='a').num == 5
        p = Population((2, 3), name='b')
        assert p.size == (2, 3) and p.num == 6
        q = Population([2, 3], name='c')
        assert q.size == (2, 3)
        assert Population(np.int64(4), name='d').size == 4
        assert repr(p) == "Population(name='b', size=(2, 3))"

    @pytest.mark.parametrize('size', [True, 2.5, 'a', (), (2, 2.0), (True, 2)])
    def test_bad_size_type(self, size):
        with pytest.raises(TypeError):
            Population(size, name='x')

    @pytest.mark.parametrize('size', [0, -1, (3, 0)])
    def test_non_positive_size(self, size):
        with pytest.raises(ValueError, match='positive'):
            Population(size, name='x')

    def test_name_validation(self):
        with pytest.raises(TypeError):
            Population(3, name=1)
        with pytest.raises(ValueError):
            Population(3, name='')

    def test_positions(self):
        pos = np.zeros((6, 2)) * u.um
        assert Population((2, 3), name='p', positions=pos).positions is pos
        Population(4, name='line', positions=np.arange(4.0))
        with pytest.raises(ValueError, match=r'\(6, d\)'):
            Population(6, name='p', positions=np.zeros((5, 2)))
        with pytest.raises(ValueError):
            Population(6, name='p', positions=np.zeros((6, 2, 1)))


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------

class TestProjection:
    def test_default_and_custom_name(self):
        a, b = Population(4, name='a'), Population(3, name='b')
        assert Projection(a, b, FixedOutDegree(1)).name == 'a->b'
        p = Projection(a, b, FixedOutDegree(1), name='ff')
        assert p.name == 'ff'
        assert repr(p) == "Projection(name='ff', pre='a', post='b', connectivity=FixedOutDegree)"
        with pytest.raises(ValueError):
            Projection(a, b, FixedOutDegree(1), name='')

    def test_type_errors(self):
        a = Population(4, name='a')
        with pytest.raises(TypeError):
            Projection(a, 4, FixedOutDegree(1))
        with pytest.raises(TypeError):
            Projection(a, a, 'conn')

    def test_size_mismatch(self):
        a, b = Population(4, name='a'), Population((2, 3), name='b')
        with pytest.raises(ValueError, match='pre_size'):
            Projection(a, b, FixedOutDegree(1, pre_size=5))
        with pytest.raises(ValueError, match='post_size'):
            Projection(a, b, FixedOutDegree(1, post_size=5))
        Projection(a, b, FixedOutDegree(1, pre_size=4, post_size=6))  # counts agree

    @pytest.mark.parametrize('make', [
        lambda: FixedOutDegree(3, allow_self_connections=False, seed=0),
        lambda: FixedInDegree(3, allow_self_connections=False, seed=0),
        lambda: FixedTotalNum(12, allow_self_connections=False, seed=0),
    ])
    def test_distinct_populations_keep_diagonal_pairs(self, make):
        # i -> i between two different populations is not an autapse: the
        # generator's self-connection exclusion must not remove those pairs.
        a, b = Population(4, name='a'), Population(4, name='b')
        seen_diagonal = False
        for seed in range(20):
            conn = make()
            conn.rng = np.random.RandomState(seed)
            res = Projection(a, b, conn).build()
            seen_diagonal |= bool(np.any(res.pre_indices == res.post_indices))
        assert seen_diagonal
        # the caller's generator is left untouched
        assert conn.allow_self_connections is False

    def test_distinct_populations_with_random_default(self):
        from braintools.conn import FixedProb
        a, b = Population(30, name='a'), Population(30, name='b')
        res = Projection(a, b, FixedProb(1.0, seed=0)).build()  # FixedProb excludes self by default
        assert res.n_connections == 900

    def test_recurrent_projection_still_excludes_self(self):
        a = Population(10, name='a')
        res = Projection(a, a, FixedOutDegree(9, allow_self_connections=False, seed=0)).build()
        assert res.n_connections == 90 and not np.any(res.pre_indices == res.post_indices)

    def test_build_passes_sizes_and_positions(self):
        pos_a = np.random.RandomState(0).rand(4, 2) * u.um
        a = Population(4, name='a', positions=pos_a)
        b = Population((2, 3), name='b')
        res = Projection(a, b, FixedOutDegree(2, seed=0)).build()
        assert res.pre_size == 4 and res.post_size == (2, 3)
        assert res.pre_positions is pos_a and res.post_positions is None
        assert res.shape == (4, 6)


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------

class TestNetwork:
    def test_to_csr_uses_default_float_dtype(self):
        import brainstate
        a = Population(5, name='a')
        net = Network([a], [Projection(a, a, FixedOutDegree(2, weight=0.5 * u.nS, seed=0))])
        W = net.to_csr()
        assert u.get_mantissa(W.data).dtype == brainstate.environ.dftype()
        net = Network([a], [Projection(a, a, FixedOutDegree(2, weight=Normal(1.0, 0.1), seed=0))])
        assert u.get_mantissa(net.to_csr().data).dtype == brainstate.environ.dftype()

    def test_ei_block_degrees(self):
        net = _ei_network()
        assert net.offsets == {'E': 0, 'I': 800}
        assert net.n_neurons == 1000
        results = net.build()
        assert list(results) == ['E->E', 'E->I', 'I->E', 'I->I']
        W = net.to_csr()
        assert W.shape == (1000, 1000)
        assert W.nse == 800 * 80 + 200 * 80 + 800 * 20 + 200 * 20
        assert u.get_unit(W.data) == u.mS
        rows, cols = _csr_edges(W)
        from_e = rows < 800
        np.testing.assert_array_equal(np.bincount(cols[from_e], minlength=1000), 80)
        np.testing.assert_array_equal(np.bincount(cols[~from_e], minlength=1000), 20)
        # rows sorted, columns ascending within each row
        assert np.all(np.diff(rows) >= 0)
        assert np.all((np.diff(cols) > 0) | (np.diff(rows) > 0))
        data = u.get_mantissa(W.data.to(u.mS))
        np.testing.assert_allclose(np.asarray(data)[from_e], 0.6, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(data)[~from_e], -6.7, rtol=1e-6)
        assert repr(net) == "Network(populations=['E', 'I'], projections=['E->E', 'E->I', 'I->E', 'I->I'], n_neurons=1000)"

    def test_block_matrix_equals_blocks(self):
        net = _ei_network(n_e=40, n_i=10, k_e=8, k_i=2)
        results = net.build()
        dense = np.zeros((50, 50))
        sl = {'E': slice(0, 40), 'I': slice(40, 50)}
        for name, proj in net.projections.items():
            dense[sl[proj.pre.name], sl[proj.post.name]] = u.get_mantissa(results[name].weight2dense().to(u.mS))
        W = net.to_csr()
        np.testing.assert_allclose(u.get_mantissa(W.todense().to(u.mS)), dense, rtol=1e-6)
        x = np.random.RandomState(0).rand(3, 50).astype(np.float32)
        y = jax.vmap(lambda v: v @ W)(jnp.asarray(x))
        np.testing.assert_allclose(u.get_mantissa(y.to(u.mS)), x @ dense, rtol=1e-4, atol=1e-4)

    def test_single_population_self_projection(self):
        P = Population((3, 4), name='P')
        conn = FixedOutDegree(3, allow_self_connections=False, weight=Normal(1.0, 0.1), seed=0)
        net = Network([P], [Projection(P, P, conn)])
        res = net.build()['P->P']
        assert not np.any(res.pre_indices == res.post_indices)
        np.testing.assert_allclose(np.asarray(net.to_csr().todense()), np.asarray(res.weight2dense()), rtol=1e-6)

    def test_duplicate_population_names(self):
        a = Population(3, name='a')
        with pytest.raises(ValueError, match='Duplicate population'):
            Network([a, Population(4, name='a')])
        with pytest.raises(ValueError, match='Duplicate population'):
            Network([a, a])

    def test_duplicate_projection_names(self):
        a = Population(3, name='a')
        with pytest.raises(ValueError, match='Duplicate projection'):
            Network([a], [Projection(a, a, FixedOutDegree(1)), Projection(a, a, FixedOutDegree(2))])
        # distinct explicit names are fine
        Network([a], [Projection(a, a, FixedOutDegree(1)), Projection(a, a, FixedOutDegree(2), name='x')])

    def test_foreign_population(self):
        a = Population(3, name='a')
        other = Population(3, name='a')
        with pytest.raises(ValueError, match='not part of this network'):
            Network([a], [Projection(a, other, FixedOutDegree(1))])

    def test_type_errors(self):
        with pytest.raises(TypeError):
            Network([3])
        a = Population(3, name='a')
        with pytest.raises(TypeError):
            Network([a], ['proj'])

    def test_mixed_units_raise(self):
        a, b = Population(5, name='a'), Population(5, name='b')
        net = Network([a, b], [
            Projection(a, b, FixedOutDegree(2, weight=1.0 * u.nS, seed=0)),
            Projection(b, a, FixedOutDegree(2, weight=1.0 * u.ms, seed=0)),
        ])
        with pytest.raises(ValueError, match='units must agree'):
            net.to_csr()
        net = Network([a, b], [
            Projection(a, b, FixedOutDegree(2, weight=1.0 * u.nS, seed=0)),
            Projection(b, a, FixedOutDegree(2, seed=0)),  # dimensionless weights (1.0)
        ])
        with pytest.raises(ValueError, match='units must agree'):
            net.to_csr()

    def test_compatible_units_converted(self):
        a, b = Population(4, name='a'), Population(4, name='b')
        net = Network([a, b], [
            Projection(a, b, FixedOutDegree(1, weight=2.0 * u.nS, seed=0)),
            Projection(b, a, FixedOutDegree(1, weight=0.003 * u.uS, seed=0)),
        ])
        W = net.to_csr()
        assert u.get_unit(W.data) == u.nS
        rows, _ = _csr_edges(W)
        vals = np.asarray(u.get_mantissa(W.data))
        np.testing.assert_allclose(vals[rows < 4], 2.0)
        np.testing.assert_allclose(vals[rows >= 4], 3.0)

    def test_homogeneous_weights_stay_scalar(self):
        a, b = Population(8, name='a'), Population(2, name='b')
        net = Network([a, b], [
            Projection(a, b, FixedInDegree(2, weight=0.5 * u.mS, seed=0)),
            Projection(b, a, FixedInDegree(1, weight=500.0 * u.uS, seed=0)),  # same value
        ])
        W = net.to_csr()
        assert u.math.shape(W.data) == ()
        assert u.math.allclose(W.data, 0.5 * u.mS)
        # different homogeneous values become per-edge data
        net = Network([a, b], [
            Projection(a, b, FixedInDegree(2, weight=0.5 * u.mS, seed=0)),
            Projection(b, a, FixedInDegree(1, weight=0.25 * u.mS, seed=0)),
        ])
        W = net.to_csr()
        assert u.math.shape(W.data) == (12,)
        rows, _ = _csr_edges(W)
        np.testing.assert_allclose(np.asarray(u.get_mantissa(W.data))[rows < 8], 0.5)
        np.testing.assert_allclose(np.asarray(u.get_mantissa(W.data))[rows >= 8], 0.25)

    def test_unweighted_network_scalar_one(self):
        a = Population(6, name='a')
        W = Network([a], [Projection(a, a, FixedOutDegree(2, seed=0))]).to_csr()
        assert np.asarray(W.data).shape == ()
        assert float(W.data) == 1.0
        assert W.nse == 12

    def test_empty_projection(self):
        a, b = Population(5, name='a'), Population(3, name='b')
        net = Network([a, b], [
            Projection(a, b, FixedInDegree(2, weight=1.0 * u.nS, seed=0)),
            Projection(b, a, FixedInDegree(0, weight=1.0 * u.ms, seed=0)),  # empty, unit ignored
            Projection(b, b, FixedTotalNum(0)),
        ])
        assert net.build()['b->a'].n_connections == 0
        W = net.to_csr()
        assert W.nse == 6
        rows, cols = _csr_edges(W)
        assert np.all(rows < 5) and np.all(cols >= 5)

    def test_all_empty_and_no_projections(self):
        a = Population(4, name='a')
        W = Network([a], [Projection(a, a, FixedTotalNum(0))]).to_csr()
        assert W.shape == (4, 4) and W.nse == 0
        W = Network([a]).to_csr()
        assert W.nse == 0
        np.testing.assert_array_equal(np.asarray(W.todense()), np.zeros((4, 4)))

    def test_delay_export(self):
        a, b = Population(4, name='a'), Population(3, name='b')
        net = Network([a, b], [
            Projection(a, b, FixedOutDegree(2, delay=Normal(2.0 * u.ms, 0.2 * u.ms), seed=0)),
            Projection(b, a, FixedOutDegree(1, delay=1.0 * u.ms, seed=0)),
        ])
        D = net.to_csr(data='delay')
        assert u.get_unit(D.data) == u.ms
        res = net.build()
        dense = np.zeros((7, 7))
        dense[:4, 4:] = u.get_mantissa(res['a->b'].delay2matrix().to(u.ms))
        dense[4:, :4] = u.get_mantissa(res['b->a'].delay2matrix().to(u.ms))
        np.testing.assert_allclose(u.get_mantissa(D.todense().to(u.ms)), dense, rtol=1e-6)
        with pytest.raises(ValueError, match="'weight' or 'delay'"):
            net.to_csr(data='bogus')

    def test_build_recompute(self):
        net = _ei_network(n_e=40, n_i=10, k_e=5, k_i=2)
        first = net.build()
        again = net.build()
        assert first['E->E'] is again['E->E']  # cached by the generator
        redrawn = net.build(recompute=True)
        assert not np.array_equal(first['E->E'].pre_indices, redrawn['E->E'].pre_indices)

    def test_positions_reach_generators(self):
        pos = np.random.RandomState(0).rand(6, 2) * u.um
        a = Population(6, name='a', positions=pos)
        res = Network([a], [Projection(a, a, FixedOutDegree(2, seed=0))]).build()['a->a']
        assert res.pre_positions is pos and res.post_positions is pos
