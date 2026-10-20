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

"""Tests of the structural graph measures against networkx."""

import math
import tracemalloc

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from braintools.analysis.graph import _graph as G
from braintools.analysis.graph._graph import (
    TRIAD_NAMES,
    characteristic_path_length,
    clustering_coefficient,
    degree_distribution,
    modularity,
    motif_counts,
    small_world_index,
    spectral_radius,
)
from braintools.conn import ConnectionResult, FixedProb

nx = pytest.importorskip('networkx')


# ----------------------------------------------------------------------------
# graph fixtures
# ----------------------------------------------------------------------------

def _array(graph, n=None, weight='weight'):
    n = graph.number_of_nodes() if n is None else n
    return nx.to_numpy_array(graph, nodelist=range(n), weight=weight)


def _orient(graph, seed, p_mutual=0.3):
    """Directed version of an undirected graph: random direction, some mutual pairs."""
    rng = np.random.default_rng(seed)
    out = nx.DiGraph()
    out.add_nodes_from(graph.nodes)
    for i, j in graph.edges:
        r = rng.random()
        if r < p_mutual:
            out.add_edge(i, j)
            out.add_edge(j, i)
        elif r < (1 + p_mutual) / 2:
            out.add_edge(i, j)
        else:
            out.add_edge(j, i)
    return out


def _undirected_graphs():
    return {
        'erdos_renyi': nx.gnp_random_graph(80, 0.08, seed=1),
        'watts_strogatz': nx.watts_strogatz_graph(100, 6, 0.1, seed=2),
        'barabasi_albert': nx.barabasi_albert_graph(100, 3, seed=3),
        'planted_partition': nx.planted_partition_graph(4, 20, 0.4, 0.02, seed=4),
    }


def _directed_graphs():
    und = _undirected_graphs()
    graphs = {f'{k}_oriented': _orient(g, seed=i) for i, (k, g) in enumerate(und.items())}
    graphs['erdos_renyi_directed'] = nx.gnp_random_graph(70, 0.1, seed=5, directed=True)
    graphs['dense_directed'] = nx.gnp_random_graph(30, 0.4, seed=6, directed=True)
    return graphs


UNDIRECTED = _undirected_graphs()
DIRECTED = _directed_graphs()


def _weighted(adj, seed, low=0.1, high=2.0):
    rng = np.random.default_rng(seed)
    return adj * rng.uniform(low, high, adj.shape)


def _symmetric_weighted(adj, seed):
    w = np.triu(_weighted(adj, seed), 1)
    return w + w.T


def _as_connection_result(adj, weights=True):
    pre, post = np.nonzero(adj)
    return ConnectionResult(
        pre, post, pre_size=adj.shape[0], post_size=adj.shape[1],
        weights=adj[pre, post] if weights else None,
    )


def _formats(adj):
    """The same graph as dense NumPy, JAX, scipy CSR/COO array and ConnectionResult."""
    return {
        'numpy': adj,
        'jax': jnp.asarray(adj),
        'csr': sp.csr_matrix(adj),
        'coo_array': sp.coo_array(adj),
        'connection_result': _as_connection_result(adj),
        'nested_list': adj.tolist(),
    }


@pytest.fixture
def small_chunks(monkeypatch):
    """Force many chunks in the chunked sparse products and BFS blocks."""
    monkeypatch.setattr(G, '_CHUNK_WORK', 64)


# ----------------------------------------------------------------------------
# input adapter
# ----------------------------------------------------------------------------

class TestToCsr:
    def test_formats_agree(self):
        adj = _weighted(_array(DIRECTED['dense_directed']), seed=0)
        for name, conn in _formats(adj).items():
            csr, unit = G._to_csr(conn, weighted=True)
            assert isinstance(csr, sp.csr_matrix), name
            assert unit is None
            np.testing.assert_allclose(csr.toarray(), adj, err_msg=name)
            binary, _ = G._to_csr(conn, weighted=False)
            np.testing.assert_array_equal(binary.toarray(), adj != 0, err_msg=name)

    def test_connection_result_weight_forms(self):
        pre = np.array([0, 1, 2, 0])
        post = np.array([1, 2, 0, 1])  # (0, 1) is repeated
        no_w = ConnectionResult(pre, post, pre_size=3, post_size=3)
        csr, unit = G._to_csr(no_w, weighted=True)
        assert csr[0, 1] == 2.0 and unit is None
        scalar = ConnectionResult(pre, post, pre_size=3, post_size=3, weights=np.array([0.5]))
        csr, _ = G._to_csr(scalar, weighted=True)
        assert csr[0, 1] == 1.0 and csr[1, 2] == 0.5
        per_edge = ConnectionResult(pre, post, pre_size=3, post_size=3,
                                    weights=np.array([1., 2., 3., 4.]) * u.nS)
        csr, unit = G._to_csr(per_edge, weighted=True)
        assert unit == u.nS
        assert csr[0, 1] == 5.0 and csr[2, 0] == 3.0
        binary, unit = G._to_csr(per_edge, weighted=False)
        assert unit is None and binary[0, 1] == 1.0 and binary.nnz == 3

    def test_single_edge_connection_result(self):
        conn = ConnectionResult(np.array([0]), np.array([1]), pre_size=2, post_size=2, weights=np.array([3.]))
        csr, _ = G._to_csr(conn, weighted=True)
        assert csr[0, 1] == 3.0 and csr.nnz == 1

    def test_multidimensional_sizes(self):
        conn = ConnectionResult(np.array([0, 5]), np.array([5, 0]), pre_size=(2, 3), post_size=(2, 3))
        csr, _ = G._to_csr(conn, weighted=False)
        assert csr.shape == (6, 6) and csr.nnz == 2

    def test_dense_quantity(self):
        w = np.array([[0., 2.], [1., 0.]]) * u.mV
        csr, unit = G._to_csr(w, weighted=True)
        assert unit == u.mV
        np.testing.assert_allclose(csr.toarray(), [[0, 2], [1, 0]])

    def test_zero_weights_are_not_edges(self):
        conn = ConnectionResult(np.array([0, 1]), np.array([1, 0]), pre_size=2, post_size=2,
                                weights=np.array([0., 1.]))
        csr, _ = G._to_csr(conn, weighted=True)
        assert csr.nnz == 1
        # explicit zeros of a sparse matrix are dropped too
        explicit = sp.csr_matrix((np.array([0., 1.]), np.array([1, 0]), np.array([0, 1, 2])), shape=(2, 2))
        assert G._to_csr(explicit, weighted=False)[0].nnz == 1

    def test_bad_inputs(self):
        with pytest.raises(TypeError, match='ConnectionResult'):
            G._to_csr('not a graph', weighted=False)
        with pytest.raises(ValueError, match='2-D'):
            G._to_csr(np.zeros((2, 2, 2)), weighted=False)
        rect = np.ones((3, 4))
        for fn in (clustering_coefficient, characteristic_path_length, motif_counts, spectral_radius,
                   small_world_index):
            with pytest.raises(ValueError, match='square'):
                fn(rect)
        with pytest.raises(ValueError, match='square'):
            modularity(rect, np.zeros(3))

    def test_symmetrise_weight_is_mean_of_present_directions(self):
        w = sp.csr_matrix(np.array([[0., 2., 3.], [4., 0., 0.], [0., 0., 0.]]))
        sym = G._symmetrise(w, weighted=True).toarray()
        np.testing.assert_allclose(sym, [[0, 3, 3], [3, 0, 0], [3, 0, 0]])
        np.testing.assert_array_equal(G._symmetrise(w, weighted=False).toarray(), sym != 0)

    def test_triangular_pairs_decoding(self):
        for n in (2, 3, 7, 50):
            i, j = G._triangular_pairs(np.arange(n * (n - 1) // 2), n)
            ref_i, ref_j = np.triu_indices(n, 1)
            np.testing.assert_array_equal(i, ref_i)
            np.testing.assert_array_equal(j, ref_j)
        n = 30_000  # large indices: float rounding guard
        idx = np.array([0, 1, n - 2, n - 1, n * (n - 1) // 2 - 1])
        i, j = G._triangular_pairs(idx, n)
        np.testing.assert_array_equal(i, [0, 0, 0, 1, n - 2])
        np.testing.assert_array_equal(j, [1, 2, n - 1, 2, n - 1])

    def test_row_chunks_cover_rows(self, small_chunks):
        adj = sp.csr_matrix(_array(DIRECTED['dense_directed']))
        chunks = list(G._row_chunks(adj, adj))
        assert len(chunks) > 3
        assert chunks[0][0] == 0 and chunks[-1][1] == adj.shape[0]
        assert all(a[1] == b[0] for a, b in zip(chunks[:-1], chunks[1:]))
        assert list(G._row_chunks(sp.csr_matrix((0, 0)), sp.csr_matrix((0, 0)))) == []


def _messy_csr(n=24, seed=0):
    """Float64 CSR with unsorted column indices, duplicate entries and explicit zeros."""
    rng = np.random.default_rng(seed)
    per_row = rng.integers(2, 6, n)
    indptr = np.concatenate([[0], np.cumsum(per_row)]).astype(np.int32)
    indices = rng.integers(0, n, indptr[-1]).astype(np.int32)  # unsorted, with repeats
    data = rng.uniform(0.5, 2.0, indptr[-1])
    data[::7] = 0.0  # explicit zeros
    indices[1], indices[0] = indices[0], indices[0]  # a guaranteed duplicate in row 0
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def _all_public_calls(conn):
    n = 24
    labels = np.arange(n) % 3
    degree_distribution(conn, direction='in')
    degree_distribution(conn, direction='out')
    degree_distribution(conn, direction='total')
    for directed in (True, False):
        clustering_coefficient(conn, directed=directed)
        clustering_coefficient(conn, directed=directed, weighted=True)
        characteristic_path_length(conn, directed=directed)
        characteristic_path_length(conn, directed=directed, n_sources=5, seed=0)
        modularity(conn, labels, directed=directed)
        modularity(conn, labels, directed=directed, weighted=True)
    motif_counts(conn)
    for method in ('auto', 'dense', 'arpack'):
        spectral_radius(conn, method=method)
        spectral_radius(conn, weighted=False, method=method)
    small_world_index(conn, n_random=1, seed=0)


def _snapshot(obj):
    if sp.issparse(obj):
        fmt = obj.format
        if fmt == 'coo':
            arrays = (obj.row, obj.col, obj.data)
        else:
            arrays = (obj.indptr, obj.indices, obj.data)
        return fmt, obj.shape, tuple(np.array(a, copy=True) for a in arrays)
    if isinstance(obj, ConnectionResult):
        return 'conn', obj.shape, tuple(np.array(a, copy=True) for a in (obj.pre_indices, obj.post_indices,
                                                                        obj.weights))
    return 'dense', np.shape(obj), (np.array(obj, copy=True),)


class TestInputNotMutated:
    """No public function may modify its input (canonicalisation must work on a copy)."""

    def test_repro_spectral_radius(self):
        s = sp.csr_matrix((np.array([1., 0., 2., 3.]), np.array([2, 0, 1, 1]), np.array([0, 2, 4, 4])),
                          shape=(3, 3))
        before = _snapshot(s)
        spectral_radius(s)
        after = _snapshot(s)
        for a, b in zip(before[2], after[2]):
            np.testing.assert_array_equal(a, b)
        assert s.nnz == 4

    @pytest.mark.parametrize('kind', ['csr_matrix', 'csr_array', 'coo_matrix', 'coo_array', 'csc_matrix',
                                      'dense', 'connection_result'])
    def test_every_function_leaves_input_unchanged(self, kind):
        messy = _messy_csr()
        assert not messy.has_canonical_format and np.any(messy.data == 0)
        if kind == 'csr_matrix':
            conn = messy
        elif kind == 'csr_array':
            conn = sp.csr_array(messy, copy=True)
        elif kind == 'coo_matrix':
            coo = messy.tocoo()
            conn = sp.coo_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)
        elif kind == 'coo_array':
            coo = messy.tocoo()
            conn = sp.coo_array((coo.data, (coo.row, coo.col)), shape=coo.shape)
        elif kind == 'csc_matrix':
            conn = sp.csc_matrix((messy.data, messy.indices, messy.indptr), shape=messy.shape)
        elif kind == 'dense':
            conn = messy.toarray()
        else:
            coo = messy.tocoo()
            conn = ConnectionResult(coo.row, coo.col, pre_size=24, post_size=24, weights=coo.data)
        before = _snapshot(conn)
        _all_public_calls(conn)
        after = _snapshot(conn)
        assert before[0] == after[0] and before[1] == after[1]
        for a, b in zip(before[2], after[2]):
            np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------
# degree distribution
# ----------------------------------------------------------------------------

class TestDegreeDistribution:
    @pytest.mark.parametrize('name', sorted(DIRECTED))
    def test_matches_networkx(self, name):
        graph = DIRECTED[name]
        adj = _array(graph)
        n = adj.shape[0]
        for direction, ref in (('in', graph.in_degree), ('out', graph.out_degree), ('total', graph.degree)):
            res = degree_distribution(sp.csr_matrix(adj), direction=direction)
            expected = np.array([ref[i] for i in range(n)])
            np.testing.assert_array_equal(res.degrees, expected)
            np.testing.assert_array_equal(res.counts, np.bincount(expected))
            np.testing.assert_array_equal(res.values, np.arange(expected.max() + 1))
            assert res.counts.sum() == n
            assert res.degrees.dtype == np.int64

    def test_rectangular(self):
        conn = FixedProb(0.2, seed=0)(pre_size=30, post_size=50)
        dense = np.asarray(conn.weight2dense())
        assert degree_distribution(conn, direction='in').degrees.shape == (50,)
        np.testing.assert_array_equal(degree_distribution(conn, direction='in').degrees, (dense != 0).sum(0))
        np.testing.assert_array_equal(degree_distribution(conn, direction='out').degrees, (dense != 0).sum(1))
        with pytest.raises(ValueError, match='square'):
            degree_distribution(conn, direction='total')

    def test_self_loops_counted_duplicates_merged(self):
        # a self-loop counts once in the in- and out-degree, twice in the total
        conn = ConnectionResult(np.array([0, 0, 0, 1]), np.array([0, 1, 1, 2]), pre_size=3, post_size=3)
        np.testing.assert_array_equal(degree_distribution(conn, direction='out').degrees, [2, 1, 0])
        np.testing.assert_array_equal(degree_distribution(conn, direction='in').degrees, [1, 1, 1])
        np.testing.assert_array_equal(degree_distribution(conn, direction='total').degrees, [3, 2, 1])

    def test_fixed_in_degree_with_self_connections(self):
        from braintools.conn import FixedInDegree
        k = 50
        res = FixedInDegree(k, seed=0)(pre_size=1000, post_size=1000)
        assert np.any(res.pre_indices == res.post_indices)  # self-connections present
        dd = degree_distribution(res, direction='in')
        np.testing.assert_array_equal(dd.degrees, np.full(1000, k))
        np.testing.assert_array_equal(dd.counts[k], 1000)
        assert degree_distribution(res, direction='out').degrees.sum() == k * 1000

    @pytest.mark.parametrize('name', ['erdos_renyi_directed', 'dense_directed'])
    def test_self_loops_match_networkx(self, name):
        graph = DIRECTED[name].copy()
        n = graph.number_of_nodes()
        graph.add_edges_from((i, i) for i in range(0, n, 3))
        adj = _array(graph)
        for direction, ref in (('in', graph.in_degree), ('out', graph.out_degree), ('total', graph.degree)):
            for conn in (adj, _as_connection_result(adj), sp.csr_matrix(adj)):
                np.testing.assert_array_equal(degree_distribution(conn, direction=direction).degrees,
                                              [ref[i] for i in range(n)], err_msg=direction)

    def test_rectangular_diagonal_entries_are_ordinary_edges(self):
        rect = np.zeros((3, 4))
        rect[0, 0] = rect[1, 1] = rect[1, 3] = 1
        np.testing.assert_array_equal(degree_distribution(rect, direction='out').degrees, [1, 2, 0])
        np.testing.assert_array_equal(degree_distribution(rect, direction='in').degrees, [1, 1, 0, 1])

    def test_empty_and_edgeless(self):
        res = degree_distribution(np.zeros((0, 0)))
        assert res.degrees.size == 0 and res.counts.size == 0 and res.values.size == 0
        res = degree_distribution(np.zeros((4, 4)), direction='total')
        np.testing.assert_array_equal(res.degrees, [0, 0, 0, 0])
        np.testing.assert_array_equal(res.counts, [4])

    def test_bad_direction(self):
        with pytest.raises(ValueError, match='direction'):
            degree_distribution(np.zeros((2, 2)), direction='both')


# ----------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------

def _node_array(values, n):
    return np.array([values[i] for i in range(n)])


class TestClustering:
    @pytest.mark.parametrize('name', sorted(UNDIRECTED))
    def test_undirected_matches_networkx(self, name):
        graph = UNDIRECTED[name]
        adj = _array(graph)
        n = adj.shape[0]
        local = clustering_coefficient(sp.csr_matrix(adj), directed=False, average=False)
        np.testing.assert_allclose(local, _node_array(nx.clustering(graph), n), atol=1e-12)
        assert clustering_coefficient(adj, directed=False) == pytest.approx(nx.average_clustering(graph), abs=1e-12)

    @pytest.mark.parametrize('name', sorted(UNDIRECTED))
    def test_undirected_weighted_matches_networkx(self, name):
        adj = _symmetric_weighted(_array(UNDIRECTED[name]), seed=1)
        graph = nx.from_numpy_array(adj)
        local = clustering_coefficient(adj, directed=False, weighted=True, average=False)
        np.testing.assert_allclose(local, _node_array(nx.clustering(graph, weight='weight'), len(adj)), atol=1e-12)

    @pytest.mark.parametrize('name', sorted(DIRECTED))
    def test_directed_matches_networkx(self, name):
        graph = DIRECTED[name]
        adj = _array(graph)
        n = adj.shape[0]
        local = clustering_coefficient(_as_connection_result(adj), average=False)
        np.testing.assert_allclose(local, _node_array(nx.clustering(graph), n), atol=1e-12)
        assert clustering_coefficient(adj) == pytest.approx(nx.average_clustering(graph), abs=1e-12)

    @pytest.mark.parametrize('name', sorted(DIRECTED))
    def test_directed_weighted_matches_networkx(self, name):
        adj = _weighted(_array(DIRECTED[name]), seed=2)
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        local = clustering_coefficient(adj, weighted=True, average=False)
        np.testing.assert_allclose(local, _node_array(nx.clustering(graph, weight='weight'), len(adj)), atol=1e-12)

    def test_signed_weights_match_networkx(self):
        adj = _weighted(_array(DIRECTED['dense_directed']), seed=3, low=-1.0, high=2.0)
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        local = clustering_coefficient(adj, weighted=True, average=False)
        np.testing.assert_allclose(local, _node_array(nx.clustering(graph, weight='weight'), len(adj)), atol=1e-12)

    def test_undirected_view_of_directed_graph(self):
        graph = DIRECTED['erdos_renyi_directed']
        adj = _array(graph)
        ref = nx.average_clustering(graph.to_undirected())
        assert clustering_coefficient(adj, directed=False) == pytest.approx(ref, abs=1e-12)

    def test_weighted_quantity_units_cancel(self):
        adj = _weighted(_array(DIRECTED['dense_directed']), seed=4)
        plain = clustering_coefficient(adj, weighted=True)
        assert clustering_coefficient(adj * u.nS, weighted=True) == pytest.approx(plain, abs=1e-14)
        assert clustering_coefficient(5.0 * adj, weighted=True) == pytest.approx(plain, abs=1e-14)

    def test_weighted_vs_unweighted_differ(self):
        adj = _weighted(_array(DIRECTED['dense_directed']), seed=5)
        assert clustering_coefficient(adj, weighted=True) != pytest.approx(clustering_coefficient(adj))
        # constant weights give the unweighted value
        assert clustering_coefficient(3.0 * (adj != 0), weighted=True) == pytest.approx(clustering_coefficient(adj))

    def test_weighted_needs_positive_max(self):
        with pytest.raises(ValueError, match='positive'):
            clustering_coefficient(-np.ones((3, 3)), weighted=True)

    def test_self_loops_ignored(self):
        graph = DIRECTED['erdos_renyi_directed']
        adj = _array(graph)
        with_loops = adj + np.eye(len(adj))
        np.testing.assert_allclose(clustering_coefficient(with_loops, average=False),
                                   clustering_coefficient(adj, average=False))
        w = _weighted(adj, seed=6)
        # a large self-loop weight must not change the weight normalisation
        np.testing.assert_allclose(clustering_coefficient(w + 100 * np.eye(len(w)), weighted=True),
                                   clustering_coefficient(w, weighted=True))

    def test_empty_isolated_and_complete(self):
        assert math.isnan(clustering_coefficient(np.zeros((0, 0))))
        assert clustering_coefficient(np.zeros((0, 0)), average=False).shape == (0,)
        assert clustering_coefficient(np.zeros((5, 5))) == 0.0
        assert clustering_coefficient(np.zeros((5, 5)), weighted=True) == 0.0
        assert clustering_coefficient(np.zeros((5, 5)), directed=False, weighted=True) == 0.0
        complete = np.ones((6, 6)) - np.eye(6)
        np.testing.assert_allclose(clustering_coefficient(complete, average=False), 1.0)
        # a triangle plus isolated nodes: isolated nodes count as 0 in the average
        tri = np.zeros((6, 6))
        tri[np.ix_([0, 1, 2], [0, 1, 2])] = 1 - np.eye(3)
        assert clustering_coefficient(tri, directed=False) == pytest.approx(0.5)

    def test_chunking_does_not_change_result(self, small_chunks):
        graph = DIRECTED['dense_directed']
        adj = _array(graph)
        local = clustering_coefficient(adj, average=False)
        np.testing.assert_allclose(local, _node_array(nx.clustering(graph), len(adj)), atol=1e-12)


# ----------------------------------------------------------------------------
# path length and efficiency
# ----------------------------------------------------------------------------

def _directed_efficiency(graph):
    n = graph.number_of_nodes()
    total = sum(1.0 / d for s, lengths in nx.all_pairs_shortest_path_length(graph)
                for t, d in lengths.items() if t != s)
    return total / (n * (n - 1))


class TestPathLength:
    @pytest.mark.parametrize('name', sorted(UNDIRECTED))
    def test_undirected_matches_networkx(self, name):
        graph = UNDIRECTED[name]
        if not nx.is_connected(graph):
            graph = graph.subgraph(max(nx.connected_components(graph), key=len))
            graph = nx.convert_node_labels_to_integers(graph)
        adj = _array(graph)
        res = characteristic_path_length(sp.csr_matrix(adj), directed=False)
        n = len(adj)
        assert res.path_length == pytest.approx(nx.average_shortest_path_length(graph), rel=1e-12)
        assert res.efficiency == pytest.approx(nx.global_efficiency(graph), rel=1e-12)
        assert res.n_reachable == n * (n - 1) and res.n_sources == n

    @pytest.mark.parametrize('name', ['erdos_renyi_directed', 'dense_directed'])
    def test_directed_matches_networkx(self, name):
        graph = DIRECTED[name]
        assert nx.is_strongly_connected(graph)
        adj = _array(graph)
        res = characteristic_path_length(_as_connection_result(adj))
        assert res.path_length == pytest.approx(nx.average_shortest_path_length(graph), rel=1e-12)
        assert res.efficiency == pytest.approx(_directed_efficiency(graph), rel=1e-12)

    def test_disconnected_semantics(self):
        # two disjoint directed chains 0->1->2 and 3->4
        adj = np.zeros((5, 5))
        adj[0, 1] = adj[1, 2] = adj[3, 4] = 1
        res = characteristic_path_length(adj)
        # reachable pairs: (0,1)=1 (1,2)=1 (0,2)=2 (3,4)=1
        assert res.n_reachable == 4
        assert res.path_length == pytest.approx(5 / 4)
        assert res.efficiency == pytest.approx((1 + 1 + 0.5 + 1) / 20)
        graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        assert res.efficiency == pytest.approx(_directed_efficiency(graph))
        und = characteristic_path_length(adj, directed=False)
        assert und.n_reachable == 8
        assert und.path_length == pytest.approx(10 / 8)
        assert und.efficiency == pytest.approx(nx.global_efficiency(graph.to_undirected()))

    def test_isolated_nodes_and_self_loops(self):
        graph = UNDIRECTED['watts_strogatz']
        adj = _array(graph)
        padded = np.zeros((len(adj) + 3, len(adj) + 3))
        padded[:len(adj), :len(adj)] = adj
        padded += np.eye(len(padded))
        res = characteristic_path_length(padded, directed=False)
        assert res.path_length == pytest.approx(nx.average_shortest_path_length(graph))
        n, m = len(adj), len(padded)
        assert res.efficiency == pytest.approx(nx.global_efficiency(graph) * n * (n - 1) / (m * (m - 1)))

    def test_empty_and_tiny(self):
        for n in (0, 1, 4):
            res = characteristic_path_length(np.zeros((n, n)))
            assert math.isnan(res.path_length)
            assert res.efficiency == 0.0 and res.n_reachable == 0 and res.n_sources == n

    def test_source_sampling(self):
        graph = UNDIRECTED['watts_strogatz']
        adj = sp.csr_matrix(_array(graph))
        exact = characteristic_path_length(adj, directed=False)
        a = characteristic_path_length(adj, directed=False, n_sources=40, seed=1)
        b = characteristic_path_length(adj, directed=False, n_sources=40, seed=1)
        assert a == b
        assert a.n_sources == 40 and a.n_reachable == 40 * 99
        assert a.path_length == pytest.approx(exact.path_length, rel=0.1)
        assert a.efficiency == pytest.approx(exact.efficiency, rel=0.1)
        rng_seed = characteristic_path_length(adj, directed=False, n_sources=40, seed=np.random.default_rng(1))
        assert rng_seed.n_sources == 40
        assert characteristic_path_length(adj, directed=False, n_sources=500) == exact
        with pytest.raises(ValueError, match='n_sources'):
            characteristic_path_length(adj, n_sources=0)

    def test_chunked_sources(self, small_chunks):
        graph = DIRECTED['dense_directed']
        res = characteristic_path_length(_array(graph))
        assert res.path_length == pytest.approx(nx.average_shortest_path_length(graph), rel=1e-12)

    def test_weights_ignored(self):
        adj = _array(DIRECTED['dense_directed'])
        assert characteristic_path_length(_weighted(adj, seed=0)) == characteristic_path_length(adj)


# ----------------------------------------------------------------------------
# modularity
# ----------------------------------------------------------------------------

def _communities(labels):
    return [set(np.flatnonzero(labels == c).tolist()) for c in np.unique(labels)]


class TestModularity:
    @pytest.mark.parametrize('resolution', [1.0, 0.5, 2.0])
    def test_planted_partition_undirected(self, resolution):
        graph = UNDIRECTED['planted_partition']
        adj = _array(graph)
        labels = np.repeat(np.arange(4), 20)
        q = modularity(adj, labels, directed=False, resolution=resolution)
        ref = nx.community.modularity(graph, _communities(labels), resolution=resolution)
        assert q == pytest.approx(ref, abs=1e-12)
        assert q > 0.4 if resolution == 1.0 else True

    @pytest.mark.parametrize('name', sorted(DIRECTED))
    def test_directed_matches_networkx(self, name):
        graph = DIRECTED[name]
        adj = _array(graph)
        labels = np.random.default_rng(0).integers(0, 3, len(adj))
        for resolution in (1.0, 0.7):
            ref = nx.community.modularity(graph, _communities(labels), weight=None, resolution=resolution)
            assert modularity(sp.csr_matrix(adj), labels, resolution=resolution) == pytest.approx(ref, abs=1e-12)

    def test_weighted(self):
        graph = DIRECTED['planted_partition_oriented']
        adj = _weighted(_array(graph), seed=1)
        wgraph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        labels = np.repeat(np.arange(4), 20)
        ref = nx.community.modularity(wgraph, _communities(labels))
        assert modularity(_as_connection_result(adj), labels, weighted=True) == pytest.approx(ref, abs=1e-12)
        assert modularity(adj * u.nS, labels, weighted=True) == pytest.approx(ref, abs=1e-12)
        sym = _symmetric_weighted(_array(UNDIRECTED['planted_partition']), seed=2)
        ref = nx.community.modularity(nx.from_numpy_array(sym), _communities(labels))
        assert modularity(sym, labels, directed=False, weighted=True) == pytest.approx(ref, abs=1e-12)
        assert modularity(adj, labels, weighted=True) != pytest.approx(modularity(adj, labels))

    def test_labels_of_any_type(self):
        adj = _array(UNDIRECTED['planted_partition'])
        ints = np.repeat(np.arange(4), 20)
        names = np.array(['a', 'b', 'c', 'd'])[ints]
        assert modularity(adj, names) == modularity(adj, ints)
        assert modularity(adj, list(ints + 10)) == modularity(adj, ints)

    def test_trivial_partitions(self):
        adj = _array(UNDIRECTED['erdos_renyi'])
        n = len(adj)
        assert modularity(adj, np.zeros(n), directed=False) == pytest.approx(0.0, abs=1e-12)
        assert modularity(adj, np.zeros(n)) == pytest.approx(0.0, abs=1e-12)

    def test_self_loops_match_networkx(self):
        labels = np.repeat(np.arange(4), 20)
        comms = _communities(labels)
        loops = np.diag((np.arange(80) % 3 == 0).astype(float))
        und = _array(UNDIRECTED['planted_partition']) + loops
        g_und = nx.from_numpy_array(und)
        assert modularity(und, labels, directed=False) == pytest.approx(
            nx.community.modularity(g_und, comms, weight=None), abs=1e-12)
        w_und = _symmetric_weighted(und, seed=3) + 2.5 * loops
        assert modularity(w_und, labels, directed=False, weighted=True) == pytest.approx(
            nx.community.modularity(nx.from_numpy_array(w_und), comms), abs=1e-12)
        dirg = _array(DIRECTED['planted_partition_oriented']) + loops
        g_dir = nx.from_numpy_array(dirg, create_using=nx.DiGraph)
        assert modularity(dirg, labels) == pytest.approx(
            nx.community.modularity(g_dir, comms, weight=None), abs=1e-12)
        w_dir = _weighted(dirg, seed=4)
        assert modularity(w_dir, labels, weighted=True) == pytest.approx(
            nx.community.modularity(nx.from_numpy_array(w_dir, create_using=nx.DiGraph), comms), abs=1e-12)
        # a graph made only of self-loops
        only = np.eye(3)
        assert modularity(only, [0, 1, 2]) == pytest.approx(
            nx.community.modularity(nx.from_numpy_array(only, create_using=nx.DiGraph), [{0}, {1}, {2}]))

    def test_edge_cases(self):
        assert math.isnan(modularity(np.zeros((3, 3)), [0, 1, 2]))
        assert math.isnan(modularity(np.zeros((0, 0)), []))
        with pytest.raises(ValueError, match='labels'):
            modularity(np.ones((3, 3)), [0, 1])
        with pytest.raises(ValueError, match='non-negative'):
            modularity(-np.ones((3, 3)), [0, 1, 1], weighted=True)


# ----------------------------------------------------------------------------
# motifs
# ----------------------------------------------------------------------------

class TestMotifCounts:
    @pytest.mark.parametrize('name', sorted(DIRECTED))
    def test_matches_networkx(self, name):
        graph = DIRECTED[name]
        adj = _array(graph)
        res = motif_counts(_as_connection_result(adj))
        assert res.triads == nx.triadic_census(graph)
        assert list(res.triads) == list(TRIAD_NAMES)
        assert res.reciprocity == pytest.approx(nx.reciprocity(graph), abs=1e-12)
        n = len(adj)
        assert sum(res.triads.values()) == math.comb(n, 3)
        assert sum(res.dyads.values()) == math.comb(n, 2)
        assert 2 * res.dyads['mutual'] + res.dyads['asymmetric'] == graph.number_of_edges()

    def test_all_classes_present(self):
        graph = DIRECTED['dense_directed']
        res = motif_counts(_array(graph))
        assert all(v > 0 for v in res.triads.values())

    @pytest.mark.parametrize('name', sorted(UNDIRECTED))
    def test_undirected_input_is_all_mutual(self, name):
        graph = UNDIRECTED[name]
        res = motif_counts(_array(graph))
        assert res.triads == nx.triadic_census(graph.to_directed())
        assert res.reciprocity == 1.0 and res.dyads['asymmetric'] == 0

    def test_small_known_graphs(self):
        complete = np.ones((5, 5)) - np.eye(5)
        res = motif_counts(complete)
        assert res.triads['300'] == 10 and sum(res.triads.values()) == 10
        empty = motif_counts(np.zeros((4, 4)))
        assert empty.triads['003'] == 4 and math.isnan(empty.reciprocity)
        assert empty.dyads == {'mutual': 0, 'asymmetric': 0, 'null': 6}
        for n in (0, 1, 2):
            res = motif_counts(np.zeros((n, n)))
            assert sum(res.triads.values()) == 0
        two = motif_counts(np.array([[0, 1], [1, 0]]))
        assert two.dyads == {'mutual': 1, 'asymmetric': 0, 'null': 0} and two.reciprocity == 1.0

    def test_self_loops(self):
        # triads ignore self-loops (as networkx); reciprocity counts them as
        # unreciprocated edges (as networkx)
        graph = DIRECTED['erdos_renyi_directed'].copy()
        adj = _array(graph)
        graph.add_edges_from((i, i) for i in range(0, len(adj), 3))
        loops = _array(graph)
        res = motif_counts(loops)
        assert res.triads == motif_counts(adj).triads == nx.triadic_census(graph)
        assert res.dyads == motif_counts(adj).dyads
        assert res.reciprocity == pytest.approx(nx.reciprocity(graph), abs=1e-12)
        assert res.reciprocity < motif_counts(adj).reciprocity
        only = motif_counts(np.eye(3))
        assert only.reciprocity == nx.reciprocity(nx.from_numpy_array(np.eye(3), create_using=nx.DiGraph)) == 0.0

    def test_chunking_does_not_change_result(self, small_chunks):
        graph = DIRECTED['dense_directed']
        assert motif_counts(_array(graph)).triads == nx.triadic_census(graph)

    def test_size_limit(self):
        adj = np.ones((30, 30)) - np.eye(30)
        with pytest.raises(ValueError, match='max_wedges'):
            motif_counts(adj, max_wedges=1000)
        assert motif_counts(adj, max_wedges=30 * 29 * 28).triads['300'] == math.comb(30, 3)


# ----------------------------------------------------------------------------
# spectral radius
# ----------------------------------------------------------------------------

def _dense_radius(m):
    with G._single_threaded_blas():
        return float(np.max(np.abs(np.linalg.eigvals(m))))


class TestSpectralRadius:
    def test_small_dense(self):
        adj = _weighted(_array(DIRECTED['dense_directed']), seed=0, low=-1.0, high=1.0)
        assert spectral_radius(adj) == pytest.approx(_dense_radius(adj), rel=1e-12)
        assert spectral_radius(adj, weighted=False) == pytest.approx(_dense_radius(adj != 0), rel=1e-12)

    @pytest.mark.parametrize('seed', [0, 1])
    def test_arpack_path_matches_numpy(self, seed):
        rng = np.random.default_rng(seed)
        n = 600
        w = sp.random(n, n, density=0.02, random_state=seed, data_rvs=rng.standard_normal, format='csr')
        assert n > G._DENSE_EIG_MAX
        assert spectral_radius(w) == pytest.approx(_dense_radius(w.toarray()), rel=1e-8)
        assert spectral_radius(w, weighted=False) == pytest.approx(_dense_radius(w.toarray() != 0), rel=1e-8)

    def test_methods(self):
        adj = _weighted(_array(DIRECTED['dense_directed']), seed=1)
        ref = _dense_radius(adj)
        assert spectral_radius(adj, method='arpack') == pytest.approx(ref, rel=1e-10)
        assert spectral_radius(adj, method='dense') == pytest.approx(ref, rel=1e-12)
        # ARPACK needs three nodes: tiny graphs fall back to the dense method
        assert spectral_radius(np.array([[0.0, 2.0], [0.5, 0.0]]), method='arpack') == pytest.approx(1.0)
        big = sp.random(600, 600, density=0.02, random_state=3, format='csr')
        assert spectral_radius(big, method='dense') == pytest.approx(_dense_radius(big.toarray()), rel=1e-12)
        with pytest.raises(ValueError, match='method'):
            spectral_radius(adj, method='power')

    def test_diagonal_kept(self):
        assert spectral_radius(np.diag([1.0, -3.0, 2.0])) == pytest.approx(3.0)

    def test_units(self):
        w = np.array([[0.0, 2.0], [0.5, 0.0]])
        res = spectral_radius(_as_connection_result(w * u.mV))
        assert isinstance(res, u.Quantity) and res.unit == u.mV
        assert res.to_decimal(u.mV) == pytest.approx(1.0)
        assert spectral_radius(w * u.mV, weighted=False) == pytest.approx(1.0)

    def test_empty(self):
        assert spectral_radius(np.zeros((0, 0))) == 0.0
        assert spectral_radius(np.zeros((3, 3))) == 0.0
        assert spectral_radius(sp.csr_matrix((1000, 1000))) == 0.0


# ----------------------------------------------------------------------------
# small-world index
# ----------------------------------------------------------------------------

class TestSmallWorld:
    def test_watts_strogatz_is_small_world(self):
        graph = UNDIRECTED['watts_strogatz']
        res = small_world_index(_array(graph), n_random=5, seed=0)
        assert res.clustering == pytest.approx(nx.average_clustering(graph), abs=1e-12)
        assert res.path_length == pytest.approx(nx.average_shortest_path_length(graph), rel=1e-12)
        assert res.sigma > 2

    def test_erdos_renyi_is_not(self):
        graph = nx.gnm_random_graph(200, 1000, seed=3)
        res = small_world_index(_array(graph), n_random=5, seed=0)
        assert 0.6 < res.sigma < 1.5
        assert res.clustering_random == pytest.approx(2 * 1000 / (200 * 199), rel=0.3)

    def test_random_reference_has_same_size(self):
        rng = np.random.default_rng(0)
        rand = G._random_undirected(50, 120, rng)
        assert (rand != rand.T).nnz == 0
        assert rand.nnz == 240 and rand.diagonal().sum() == 0

    def test_seed_and_sampling(self):
        adj = _array(UNDIRECTED['barabasi_albert'])
        a = small_world_index(adj, n_random=2, seed=1, n_sources=30)
        b = small_world_index(adj, n_random=2, seed=1, n_sources=30)
        assert a == b
        assert np.isfinite(a.sigma)

    def test_edge_cases(self):
        res = small_world_index(np.zeros((5, 5)), n_random=2, seed=0)
        assert math.isnan(res.sigma)
        res = small_world_index(np.zeros((1, 1)), n_random=1, seed=0)
        assert math.isnan(res.sigma)
        with pytest.raises(ValueError, match='n_random'):
            small_world_index(np.zeros((3, 3)), n_random=0)


# ----------------------------------------------------------------------------
# end-to-end with braintools.conn and a scale check of the sparse path
# ----------------------------------------------------------------------------

def test_fixed_prob_connection_result_end_to_end():
    conn = FixedProb(0.1, weight=0.5 * u.nS, seed=7)(pre_size=120, post_size=120)
    dense = np.asarray(u.get_mantissa(conn.weight2dense()))
    graph = nx.from_numpy_array((dense != 0).astype(float), create_using=nx.DiGraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    n = 120
    np.testing.assert_array_equal(degree_distribution(conn).degrees,
                                  [graph.in_degree(i) for i in range(n)])
    assert clustering_coefficient(conn) == pytest.approx(nx.average_clustering(graph), abs=1e-12)
    assert motif_counts(conn).triads == nx.triadic_census(graph)
    labels = np.arange(n) % 3
    assert modularity(conn, labels) == pytest.approx(
        nx.community.modularity(graph, _communities(labels), weight=None), abs=1e-12)
    res = characteristic_path_length(conn)
    if nx.is_strongly_connected(graph):
        assert res.path_length == pytest.approx(nx.average_shortest_path_length(graph), rel=1e-12)
    radius = spectral_radius(conn)
    assert radius.unit == u.nS
    assert radius.to_decimal(u.nS) == pytest.approx(_dense_radius(dense), rel=1e-10)


def test_sparse_path_memory_bound():
    """N=2e4 nodes, ~2e5 edges: no dense N x N (3.2 GB in float64) is ever built."""
    n, n_edges = 20_000, 200_000
    rng = np.random.default_rng(0)
    pre = rng.integers(0, n, n_edges)
    post = rng.integers(0, n, n_edges)
    # excitation-dominated weights: an outlier eigenvalue, so ARPACK converges fast
    conn = ConnectionResult(pre, post, pre_size=n, post_size=n, weights=rng.normal(1.0, 0.5, n_edges))
    tracemalloc.start()
    try:
        deg = degree_distribution(conn, direction='total')
        c = clustering_coefficient(conn)
        census = motif_counts(conn)
        path = characteristic_path_length(conn, n_sources=10, seed=0)
        radius = spectral_radius(conn)
        q = modularity(conn, np.arange(n) % 10)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 300 * 2 ** 20, f'peak traced memory {peak / 2 ** 20:.0f} MiB'
    assert deg.degrees.sum() == pytest.approx(2 * n_edges, rel=0.01)
    assert 0 <= c < 0.01
    assert sum(census.triads.values()) == math.comb(n, 3)
    assert 0 < path.path_length < 10 and path.n_sources == 10
    assert radius == pytest.approx(10.0, rel=0.1)  # Perron root ~ mean weight x mean degree
    assert abs(q) < 0.05
