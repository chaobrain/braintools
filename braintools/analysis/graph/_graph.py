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

"""Structural graph measures of connectivity.

Every function accepts a :class:`~braintools.conn.ConnectionResult`, a
``scipy.sparse`` matrix/array or a dense 2-D array (NumPy, JAX or a
:class:`brainunit.Quantity`). Row ``i`` / column ``j`` is the edge
``i -> j`` (pre -> post), the layout of :meth:`ConnectionResult.weight2dense`.

The measures are build-time structural statistics. They run on the host with
NumPy and ``scipy.sparse`` (the input is copied to a host CSR matrix), are not
traceable by ``jax.jit``/``jax.vmap``/``jax.grad``, and never build a dense
``N x N`` array from sparse input: sparse matrix products are evaluated in row
chunks of bounded work.

Conventions shared by all functions:

- **Edges.** For a :class:`~braintools.conn.ConnectionResult` an edge is every
  ``(pre_indices[e], post_indices[e])`` pair; for a matrix it is every non-zero
  entry. Repeated pairs are merged (their weights summed).
- **Self-loops** (diagonal entries) are handled as in networkx. They are
  counted by :func:`degree_distribution` (once in the in- and out-degree,
  twice in the total degree), :func:`modularity` and the reciprocity of
  :func:`motif_counts`, and kept by :func:`spectral_radius` (a property of the
  matrix). They are ignored by :func:`clustering_coefficient`,
  :func:`characteristic_path_length`, the triad and dyad census of
  :func:`motif_counts` and :func:`small_world_index`.
- **Undirected view.** With ``directed=False`` the graph is symmetrised: ``i``
  and ``j`` are adjacent when either ``i -> j`` or ``j -> i`` exists, and the
  weight of the pair is the mean of the weights of the directions present
  (just the weight for a symmetric input).
"""

import contextlib
import math
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import brainunit as u
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla

from braintools._misc import set_module_as

__all__ = [
    'DegreeDistribution',
    'PathLengthResult',
    'MotifCounts',
    'SmallWorldResult',
    'degree_distribution',
    'clustering_coefficient',
    'characteristic_path_length',
    'modularity',
    'motif_counts',
    'spectral_radius',
    'small_world_index',
]

# Upper bound on the number of scalar products evaluated at once in a chunked
# sparse product, and on the number of entries of a dense distance block.
_CHUNK_WORK = 2 ** 22

# Graphs with at most this many nodes use dense ``numpy.linalg.eigvals`` in
# :func:`spectral_radius`; larger graphs use ARPACK.
_DENSE_EIG_MAX = 512

# Krylov dimension of the ARPACK iteration in :func:`spectral_radius` (the
# default, 20, too often converges to an eigenvalue just inside the spectral
# circle when many eigenvalues crowd its boundary).
_ARPACK_NCV = 64

# The 16 triad classes in the order of ``networkx.triadic_census``.
TRIAD_NAMES = (
    '003', '012', '102', '021D', '021U', '021C', '111D', '111U',
    '030T', '030C', '201', '120D', '120U', '120C', '210', '300',
)

_Seed = Union[None, int, np.random.Generator]


# ----------------------------------------------------------------------------
# input adapter and sparse helpers
# ----------------------------------------------------------------------------

def _is_connection_result(conn: Any) -> bool:
    from braintools.conn._base import ConnectionResult
    return isinstance(conn, ConnectionResult)


def _plain_unit(unit: Any) -> Optional[u.Unit]:
    """``None`` for a dimensionless unit, the unit otherwise."""
    if unit is None or unit.is_unitless:
        return None
    return unit


def _to_csr(conn: Any, weighted: bool) -> Tuple[sp.csr_matrix, Optional[u.Unit]]:
    """Convert a connectivity input to a canonical host ``float64`` CSR matrix.

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        The connectivity. Rows are presynaptic, columns postsynaptic.
    weighted : bool
        If False, every edge gets weight 1 (structure only). If True, the
        weights are kept (summed over repeated pairs) and their unit is
        returned separately; edges whose weight is exactly zero are dropped.

    Returns
    -------
    csr : scipy.sparse.csr_matrix
        Canonical (sorted, duplicate-free, no explicit zeros) ``float64`` CSR
        matrix. The diagonal is kept; callers drop it where needed.
    unit : brainunit.Unit or None
        Unit of the weights (``None`` if unweighted or dimensionless).
    """
    unit = None
    if _is_connection_result(conn):
        shape = tuple(int(s) for s in conn.shape)
        rows = np.asarray(conn.pre_indices, dtype=np.int64)
        cols = np.asarray(conn.post_indices, dtype=np.int64)
        n_edges = rows.size
        if weighted and conn.weights is not None:
            mantissa, unit = u.split_mantissa_unit(conn.weights)
            data = np.asarray(mantissa, dtype=np.float64).reshape(-1)
            if data.size == 1 and n_edges != 1:
                data = np.full(n_edges, data[0], dtype=np.float64)
        else:
            data = np.ones(n_edges, dtype=np.float64)
        csr = sp.csr_matrix(sp.coo_matrix((data, (rows, cols)), shape=shape))
    elif sp.issparse(conn):
        # copy=True: a float64 CSR input would otherwise be shared, and the
        # in-place canonicalisation below would rewrite the caller's arrays
        csr = sp.csr_matrix(conn, dtype=np.float64, copy=True)
    elif isinstance(conn, (np.ndarray, u.Quantity, list, tuple)) or hasattr(conn, '__array__'):
        mantissa, unit = u.split_mantissa_unit(conn)
        dense = np.asarray(mantissa, dtype=np.float64)
        if dense.ndim != 2:
            raise ValueError(f'A dense connectivity matrix must be 2-D, got shape {dense.shape}.')
        csr = sp.csr_matrix(dense)
    else:
        raise TypeError(
            'Connectivity must be a braintools.conn.ConnectionResult, a scipy.sparse '
            f'matrix or a 2-D array, got {type(conn).__name__}.'
        )
    # ``csr`` owns its arrays here (every branch above builds or copies them),
    # so canonicalising it in place never touches the caller's input
    csr.sum_duplicates()
    csr.eliminate_zeros()
    if not weighted:
        csr.data = np.ones_like(csr.data)
        unit = None
    return csr, _plain_unit(unit)


def _square(csr: sp.csr_matrix, name: str) -> int:
    n_rows, n_cols = csr.shape
    if n_rows != n_cols:
        raise ValueError(
            f'{name} needs a square connectivity (pre and post are the same population), '
            f'got shape {csr.shape}.'
        )
    return n_rows


def _drop_diagonal(csr: sp.csr_matrix) -> sp.csr_matrix:
    """Remove self-loops."""
    coo = csr.tocoo()
    keep = coo.row != coo.col
    return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=csr.shape)


def _graph(conn: Any, name: str, weighted: bool = False) -> Tuple[sp.csr_matrix, Optional[u.Unit]]:
    """Square CSR matrix without self-loops."""
    csr, unit = _to_csr(conn, weighted)
    _square(csr, name)
    return _drop_diagonal(csr), unit


def _symmetrise(csr: sp.csr_matrix, weighted: bool) -> sp.csr_matrix:
    """Undirected view: union of both directions; weight = mean of present directions."""
    coo = csr.tocoo()
    rows = np.concatenate([coo.row, coo.col])
    cols = np.concatenate([coo.col, coo.row])
    if not weighted:
        out = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=csr.shape)
        out.data = np.ones_like(out.data)
        return out
    weights = np.concatenate([coo.data, coo.data])
    total = sp.csr_matrix((weights, (rows, cols)), shape=csr.shape)
    count = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=csr.shape)
    # Both matrices are built from the same coordinates, so after the
    # (canonicalising) COO -> CSR conversion their data arrays are aligned.
    total.data = total.data / count.data
    return total


def _binary(csr: sp.csr_matrix) -> sp.csr_matrix:
    out = csr.copy()
    out.data = np.ones_like(out.data)
    return out


def _row_chunks(x: sp.csr_matrix, y: sp.csr_matrix):
    """Yield ``(start, stop)`` row ranges of ``x`` whose product with ``y`` is bounded.

    The work of row ``i`` of ``x @ y`` is the number of scalar products
    ``sum_k [x_ik != 0] nnz(y_k)``; consecutive rows are grouped until a group
    reaches about :data:`_CHUNK_WORK` products.
    """
    n = x.shape[0]
    if n == 0:
        return
    y_row_nnz = np.diff(y.indptr).astype(np.float64)
    work = _binary(x) @ y_row_nnz
    cum = np.cumsum(work)
    cuts = np.searchsorted(cum, np.arange(_CHUNK_WORK, cum[-1] + _CHUNK_WORK, _CHUNK_WORK), side='right')
    bounds = np.unique(np.concatenate([[0], np.minimum(cuts, n), [n]]))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        yield int(start), int(stop)


def _masked_product_rowsum(x: sp.csr_matrix, y: sp.csr_matrix, z: sp.csr_matrix) -> np.ndarray:
    """Row sums of ``(x @ y) * z`` (element-wise mask), computed in row chunks."""
    out: np.ndarray = np.zeros(x.shape[0], dtype=np.float64)
    for start, stop in _row_chunks(x, y):
        block = (x[start:stop] @ y).multiply(z[start:stop])
        out[start:stop] = np.asarray(block.sum(axis=1)).reshape(-1)
    return out


def _masked_product_sum(x: sp.csr_matrix, y: sp.csr_matrix, z: sp.csr_matrix) -> float:
    return float(_masked_product_rowsum(x, y, z).sum())


def _rowsum(csr: sp.csr_matrix) -> np.ndarray:
    return np.asarray(csr.sum(axis=1), dtype=np.float64).reshape(-1)


def _colsum(csr: sp.csr_matrix) -> np.ndarray:
    return np.asarray(csr.sum(axis=0), dtype=np.float64).reshape(-1)


# ----------------------------------------------------------------------------
# degree distribution
# ----------------------------------------------------------------------------

class DegreeDistribution(NamedTuple):
    """Result of :func:`degree_distribution`.

    Attributes
    ----------
    degrees : numpy.ndarray
        Degree of every node, ``int64``. Its length is the number of post
        neurons for ``direction='in'``, of pre neurons for ``'out'``, and of
        nodes for ``'total'``.
    values : numpy.ndarray
        Degree values ``0, 1, ..., max(degrees)`` (``int64``).
    counts : numpy.ndarray
        Number of nodes with each degree in ``values`` (``int64``, sums to
        ``len(degrees)``).
    """
    degrees: np.ndarray
    values: np.ndarray
    counts: np.ndarray


@set_module_as('braintools.analysis.graph')
def degree_distribution(conn: Any, *, direction: str = 'in') -> DegreeDistribution:
    """Compute node degrees and their histogram.

    The graph is treated as directed and unweighted: the degree counts edges,
    not weights. Repeated pairs count once. A self-loop counts once in the in-
    and the out-degree and twice in the total degree, as in networkx.

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        Connectivity with rows as pre and columns as post neurons. It may be
        rectangular (pre and post populations of different sizes) for
        ``direction='in'`` or ``'out'``.
    direction : {'in', 'out', 'total'}, default 'in'
        ``'in'``: number of incoming edges of each post neuron (column
        counts); ``'out'``: outgoing edges of each pre neuron (row counts);
        ``'total'``: their sum, as ``networkx.DiGraph.degree`` (needs a square
        connectivity).

    Returns
    -------
    DegreeDistribution
        Named tuple ``(degrees, values, counts)``.

    Raises
    ------
    ValueError
        If ``direction`` is unknown, or ``'total'`` is asked for a
        rectangular connectivity.

    See Also
    --------
    clustering_coefficient : Local clustering of every node.

    Notes
    -----
    Host-only (NumPy/SciPy): not traceable by ``jax.jit``, ``jax.vmap`` or
    ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.analysis.graph import degree_distribution
        >>> adj = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        >>> result = degree_distribution(adj, direction='in')
        >>> result.degrees
        array([0, 1, 2])
        >>> result.counts
        array([1, 1, 1])
        >>>
        >>> degree_distribution(adj, direction='total').degrees
        array([2, 2, 2])
        >>>
        >>> self_loop = np.array([[1, 0], [1, 0]])  # 0 -> 0 and 1 -> 0
        >>> degree_distribution(self_loop, direction='total').degrees
        array([3, 1])
    """
    if direction not in ('in', 'out', 'total'):
        raise ValueError(f"`direction` must be 'in', 'out' or 'total', got {direction!r}.")
    csr, _ = _to_csr(conn, weighted=False)
    if direction == 'total':
        _square(csr, "degree_distribution(direction='total')")
    if direction == 'in':
        degrees = np.diff(csr.tocsc().indptr)
    elif direction == 'out':
        degrees = np.diff(csr.indptr)
    else:
        degrees = np.diff(csr.indptr) + np.diff(csr.tocsc().indptr)
    degrees = degrees.astype(np.int64)
    counts = np.bincount(degrees).astype(np.int64) if degrees.size else np.zeros(0, dtype=np.int64)
    return DegreeDistribution(degrees, np.arange(counts.size, dtype=np.int64), counts)


# ----------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------

def _normalised_cbrt(csr: sp.csr_matrix) -> sp.csr_matrix:
    """``cbrt(W / max(W))`` on the stored entries (networkx weighted convention)."""
    out = csr.copy()
    if out.nnz == 0:
        return out
    max_weight = out.data.max()
    if not max_weight > 0:
        raise ValueError(
            'Weighted clustering normalises the weights by the largest weight, which must be '
            f'positive; got a largest weight of {max_weight}.'
        )
    out.data = np.cbrt(out.data / max_weight)
    return out


@set_module_as('braintools.analysis.graph')
def clustering_coefficient(
    conn: Any,
    *,
    directed: bool = True,
    weighted: bool = False,
    average: bool = True,
) -> Union[float, np.ndarray]:
    r"""Compute the local clustering coefficient, or its average over nodes.

    With ``directed=True`` this is Fagiolo's directed clustering [1]_, which
    counts all eight kinds of directed triangles around node :math:`i`:

    .. math::

        C_i = \frac{\left[(A + A^\top)^3\right]_{ii}}
                   {2\left[d_i^{tot}(d_i^{tot} - 1) - 2 d_i^{\leftrightarrow}\right]},

    with :math:`d^{tot}` the in- plus out-degree and :math:`d^{\leftrightarrow}`
    the number of reciprocated neighbours. With ``directed=False`` the graph is
    symmetrised and :math:`C_i = [A^3]_{ii} / (k_i (k_i - 1))` [2]_. Nodes with
    fewer than two neighbours have :math:`C_i = 0`.

    With ``weighted=True`` the adjacency :math:`A` is replaced by
    :math:`\hat W^{1/3}`, with :math:`\hat W = W / \max(W)` (geometric mean of
    the triangle weights [3]_, signed cube root as networkx), while the degrees
    stay unweighted. The result matches ``networkx.clustering`` and
    ``networkx.average_clustering`` with the same directedness and
    ``weight='weight'`` / ``weight=None``.

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        Square connectivity, rows pre and columns post. Self-loops are ignored
        (also in the weight normalisation, where networkx would include them).
    directed : bool, default True
        Treat the graph as directed (Fagiolo) or undirected (symmetrised; the
        pair weight is the mean of the directions present).
    weighted : bool, default False
        Use the edge weights. They are normalised by their maximum, so their
        unit cancels (a Quantity is accepted). The largest weight must be
        positive.
    average : bool, default True
        Return the mean over all nodes (nodes with zero clustering included,
        as ``networkx.average_clustering``) instead of the per-node array.

    Returns
    -------
    float or numpy.ndarray
        The average clustering (``nan`` for a graph with no nodes), or the
        per-node clustering of shape ``(N,)``.

    Raises
    ------
    ValueError
        If the connectivity is not square, or ``weighted=True`` and the
        largest weight is not positive.

    See Also
    --------
    small_world_index : Clustering and path length relative to random graphs.
    motif_counts : Directed triad census.

    Notes
    -----
    The diagonal of the matrix cube is computed as row sums of
    :math:`(A A) \circ A^\top` with sparse products evaluated in row chunks,
    so no dense :math:`N \times N` array is formed. The cost is about the
    number of length-2 paths, :math:`\sum_k d_k^2`.

    Host-only (NumPy/SciPy): not traceable by ``jax.jit``, ``jax.vmap`` or
    ``jax.grad``.

    References
    ----------
    .. [1] G. Fagiolo, "Clustering in complex directed networks," Physical
       Review E 76, 026107 (2007).
    .. [2] D. J. Watts and S. H. Strogatz, "Collective dynamics of
       'small-world' networks," Nature 393, 440-442 (1998).
    .. [3] J.-P. Onnela, J. Saramäki, J. Kertész and K. Kaski, "Intensity and
       coherence of motifs in weighted complex networks," Physical Review E
       71, 065103 (2005).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.analysis.graph import clustering_coefficient
        >>> triangle_plus_tail = np.array([[0, 1, 1, 0],
        ...                                [1, 0, 1, 0],
        ...                                [1, 1, 0, 1],
        ...                                [0, 0, 1, 0]])
        >>> clustering_coefficient(triangle_plus_tail, directed=False, average=False)
        array([1.        , 1.        , 0.33333333, 0.        ])
        >>>
        >>> round(clustering_coefficient(triangle_plus_tail, directed=False), 4)
        0.5833
    """
    csr, _ = _graph(conn, 'clustering_coefficient', weighted=weighted)
    n = csr.shape[0]
    if directed:
        structure = _binary(csr)
        values = _normalised_cbrt(csr) if weighted else structure
        both = (values + values.T).tocsr()
        triangles = _masked_product_rowsum(both, both, both)
        d_total = _rowsum(structure) + _colsum(structure)
        d_recip = _rowsum(structure.multiply(structure.T))
        denominator = 2.0 * (d_total * (d_total - 1.0) - 2.0 * d_recip)
    else:
        sym = _symmetrise(csr, weighted)
        values = _normalised_cbrt(sym) if weighted else sym
        triangles = _masked_product_rowsum(values, values, values)
        degree = np.diff(sym.indptr).astype(np.float64)
        denominator = degree * (degree - 1.0)
    local: np.ndarray = np.zeros(n, dtype=np.float64)
    nonzero = triangles != 0
    local[nonzero] = triangles[nonzero] / denominator[nonzero]
    if not average:
        return local
    return float(local.mean()) if n else float('nan')


# ----------------------------------------------------------------------------
# path length and efficiency
# ----------------------------------------------------------------------------

class PathLengthResult(NamedTuple):
    """Result of :func:`characteristic_path_length`.

    Attributes
    ----------
    path_length : float
        Mean shortest-path length (in hops) over the ordered pairs
        ``(source, target)``, ``source != target``, where ``target`` is
        reachable from ``source``. ``nan`` when no pair is reachable.
    efficiency : float
        Global efficiency: the mean of ``1 / d(source, target)`` over all
        ordered pairs with ``source != target``, unreachable pairs counting
        as 0. ``0`` for a graph with fewer than two nodes.
    n_reachable : int
        Number of reachable ordered pairs the mean ``path_length`` is over.
    n_sources : int
        Number of source nodes used (all nodes unless sampled).
    """
    path_length: float
    efficiency: float
    n_reachable: int
    n_sources: int


@set_module_as('braintools.analysis.graph')
def characteristic_path_length(
    conn: Any,
    *,
    directed: bool = True,
    n_sources: Optional[int] = None,
    seed: _Seed = None,
) -> PathLengthResult:
    r"""Compute the characteristic path length and the global efficiency.

    Shortest paths are counted in hops (the graph is treated as unweighted)
    with breadth-first search (``scipy.sparse.csgraph.shortest_path``):

    .. math::

        L = \frac{1}{|R|} \sum_{(i, j) \in R} d_{ij}, \qquad
        E = \frac{1}{N (N - 1)} \sum_{i \neq j} \frac{1}{d_{ij}},

    with :math:`R` the reachable ordered pairs :math:`i \neq j` [1]_ [2]_. On
    a (strongly) connected graph these equal
    ``networkx.average_shortest_path_length`` and, for undirected graphs,
    ``networkx.global_efficiency``. On a disconnected graph :math:`L` is over
    reachable pairs only and :math:`E` counts unreachable pairs as 0 (for a
    directed graph :math:`E` is the directed generalisation [2]_).

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        Square connectivity, rows pre and columns post. Self-loops and weights
        are ignored.
    directed : bool, default True
        Follow edge directions, or treat the graph as undirected.
    n_sources : int, optional
        Number of source nodes, sampled uniformly without replacement, to
        estimate :math:`L` and :math:`E` on large graphs (cost
        :math:`O(n_{sources} (N + E))` instead of :math:`O(N (N + E))`). The
        sums then run over pairs whose source is in the sample. ``None`` (or a
        value ``>= N``) uses every node, giving the exact values.
    seed : int or numpy.random.Generator, optional
        Seed of the source sampling.

    Returns
    -------
    PathLengthResult
        Named tuple ``(path_length, efficiency, n_reachable, n_sources)``.

    Raises
    ------
    ValueError
        If the connectivity is not square or ``n_sources < 1``.

    See Also
    --------
    small_world_index : Path length relative to random graphs.

    Notes
    -----
    Sources are processed in chunks so that each block of distances holds at
    most about :math:`2^{22}` entries.

    Host-only (NumPy/SciPy): not traceable by ``jax.jit``, ``jax.vmap`` or
    ``jax.grad``.

    References
    ----------
    .. [1] D. J. Watts and S. H. Strogatz, "Collective dynamics of
       'small-world' networks," Nature 393, 440-442 (1998).
    .. [2] V. Latora and M. Marchiori, "Efficient behavior of small-world
       networks," Physical Review Letters 87, 198701 (2001).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.analysis.graph import characteristic_path_length
        >>> chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0 -> 1 -> 2
        >>> result = characteristic_path_length(chain)
        >>> result.path_length, result.n_reachable
        (1.3333333333333333, 3)
        >>> round(result.efficiency, 4)
        0.4167
        >>>
        >>> characteristic_path_length(chain, directed=False).path_length
        1.3333333333333333
    """
    csr, _ = _graph(conn, 'characteristic_path_length')
    n = csr.shape[0]
    graph = csr if directed else _symmetrise(csr, weighted=False)
    if n_sources is None:
        sources = np.arange(n)
    else:
        n_sources = int(n_sources)
        if n_sources < 1:
            raise ValueError(f'`n_sources` must be at least 1, got {n_sources}.')
        if n_sources >= n:
            sources = np.arange(n)
        else:
            rng = np.random.default_rng(seed)
            sources = np.sort(rng.choice(n, size=n_sources, replace=False))
    total = 0.0
    inverse = 0.0
    reachable = 0
    chunk = max(1, _CHUNK_WORK // max(n, 1))
    for start in range(0, sources.size, chunk):
        block = csgraph.shortest_path(
            graph, method='D', directed=directed, unweighted=True,
            indices=sources[start:start + chunk],
        )
        finite = np.isfinite(block) & (block > 0)
        dist = block[finite]
        total += float(dist.sum())
        inverse += float((1.0 / dist).sum())
        reachable += int(dist.size)
    path_length = total / reachable if reachable else float('nan')
    efficiency = inverse / (sources.size * (n - 1)) if n > 1 else 0.0
    return PathLengthResult(path_length, efficiency, reachable, int(sources.size))


# ----------------------------------------------------------------------------
# modularity
# ----------------------------------------------------------------------------

@set_module_as('braintools.analysis.graph')
def modularity(
    conn: Any,
    labels: Any,
    *,
    directed: bool = True,
    weighted: bool = False,
    resolution: float = 1.0,
) -> float:
    r"""Compute the modularity of a given partition.

    For a directed graph this is the Leicht–Newman modularity [1]_

    .. math::

        Q = \frac{1}{m} \sum_{ij} \left(A_{ij}
            - \gamma \frac{k_i^{out} k_j^{in}}{m}\right) \delta(c_i, c_j),
        \qquad m = \sum_{ij} A_{ij},

    and for an undirected graph Newman's modularity [2]_, the same expression
    on the symmetrised adjacency (where :math:`m` is twice the number, or
    total weight, of edges and :math:`k^{out} = k^{in}`). :math:`\gamma` is
    the resolution. Matches ``networkx.community.modularity``, including its
    treatment of self-loops.

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        Square connectivity, rows pre and columns post. Self-loops are
        counted as in networkx: a directed self-loop adds its weight to the
        node's in- and out-degree and to :math:`m`; in the undirected view it
        adds twice its weight to the node's degree (:math:`A_{ii} = 2 w_{ii}`).
    labels : array_like
        Community label of every node, shape ``(N,)``; any values that
        ``numpy.unique`` can sort (ints, strings, ...).
    directed : bool, default True
        Leicht–Newman directed modularity, or Newman's undirected modularity
        on the symmetrised graph.
    weighted : bool, default False
        Use the edge weights (which must be non-negative). The unit of
        Quantity weights cancels.
    resolution : float, default 1.0
        Resolution :math:`\gamma`; smaller values favour larger communities.

    Returns
    -------
    float
        The modularity, or ``nan`` for a graph without edges.

    Raises
    ------
    ValueError
        If the connectivity is not square, ``labels`` does not have one entry
        per node, or ``weighted=True`` with negative weights.

    See Also
    --------
    clustering_coefficient : Local clustering.

    Notes
    -----
    Host-only (NumPy/SciPy): not traceable by ``jax.jit``, ``jax.vmap`` or
    ``jax.grad``.

    References
    ----------
    .. [1] E. A. Leicht and M. E. J. Newman, "Community structure in directed
       networks," Physical Review Letters 100, 118703 (2008).
    .. [2] M. E. J. Newman, "Modularity and community structure in networks,"
       PNAS 103, 8577-8582 (2006).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.analysis.graph import modularity
        >>> two_pairs = np.array([[0, 1, 0, 0],
        ...                       [1, 0, 0, 0],
        ...                       [0, 0, 0, 1],
        ...                       [0, 0, 1, 0]])
        >>> modularity(two_pairs, [0, 0, 1, 1])
        0.5
        >>> modularity(two_pairs, [0, 1, 0, 1], directed=False)
        -0.5
    """
    csr, _ = _to_csr(conn, weighted)
    n = _square(csr, 'modularity')
    labels = np.asarray(labels)
    if labels.shape != (n,):
        raise ValueError(f'`labels` must have shape ({n},), one label per node, got {labels.shape}.')
    if weighted and csr.nnz and csr.data.min() < 0:
        raise ValueError('Weighted modularity needs non-negative weights.')
    if directed:
        graph = csr
    else:
        sym = _symmetrise(csr, weighted).tocoo()
        # an undirected self-loop adds twice its weight to its node's degree
        # (networkx convention), i.e. A_ii = 2 w_ii in the formula
        sym.data[sym.row == sym.col] *= 2.0
        graph = sym.tocsr()
    m = float(graph.data.sum())
    if graph.nnz == 0 or m == 0:
        return float('nan')
    _, community = np.unique(labels, return_inverse=True)
    community = community.reshape(-1)
    coo = graph.tocoo()
    within = float(coo.data[community[coo.row] == community[coo.col]].sum())
    k_out = np.bincount(community, weights=_rowsum(graph))
    k_in = np.bincount(community, weights=_colsum(graph))
    return within / m - resolution * float(np.dot(k_out, k_in)) / (m * m)


# ----------------------------------------------------------------------------
# motifs
# ----------------------------------------------------------------------------

class MotifCounts(NamedTuple):
    """Result of :func:`motif_counts`.

    Attributes
    ----------
    triads : dict of str to int
        Number of node triples in each of the 16 directed triad classes, keyed
        by the MAN codes in the order of ``networkx.triadic_census``:
        ``'003', '012', '102', '021D', '021U', '021C', '111D', '111U',
        '030T', '030C', '201', '120D', '120U', '120C', '210', '300'``.
        The counts sum to :math:`\\binom{N}{3}`.
    dyads : dict of str to int
        Dyad census: number of node pairs that are ``'mutual'`` (``i <-> j``),
        ``'asymmetric'`` (one direction) or ``'null'`` (unconnected). The counts
        sum to :math:`\\binom{N}{2}`.
    reciprocity : float
        Fraction of edges whose reverse edge also exists,
        ``2 * mutual / (2 * mutual + asymmetric + self_loops)`` (``nan``
        without edges). As in ``networkx.reciprocity``, a self-loop counts as
        an unreciprocated edge.
    """
    triads: Dict[str, int]
    dyads: Dict[str, int]
    reciprocity: float


def _comb(n: int, k: int) -> int:
    return math.comb(n, k) if n >= k else 0


@set_module_as('braintools.analysis.graph')
def motif_counts(conn: Any, *, max_wedges: int = 10 ** 8) -> MotifCounts:
    r"""Count the directed triad classes (triad census) and dyads.

    Every unordered triple of nodes is classified into one of the 16
    isomorphism classes of directed graphs on three nodes, named by the
    Holland–Leinhardt MAN code (numbers of mutual, asymmetric and null dyads,
    plus a letter for the orientation) [1]_. The counts match
    ``networkx.triadic_census``; the reciprocity matches
    ``networkx.reciprocity``.

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        Square connectivity, rows pre and columns post, treated as directed
        and unweighted. Self-loops are ignored by the triad and dyad census
        and counted as unreciprocated edges by the reciprocity (both as
        networkx).
    max_wedges : int, default 10**8
        Size limit. The cost is dominated by the number of wedges (paths of
        length two in the undirected view), :math:`\sum_k d_k (d_k - 1)` with
        :math:`d_k` the number of neighbours of node :math:`k`. Graphs with
        more wedges raise instead of running for a long time; raise the limit
        to proceed anyway.

    Returns
    -------
    MotifCounts
        Named tuple ``(triads, dyads, reciprocity)``.

    Raises
    ------
    ValueError
        If the connectivity is not square or has more than ``max_wedges``
        wedges.

    See Also
    --------
    clustering_coefficient : Directed clustering (Fagiolo).

    Notes
    -----
    The census is computed without enumerating triples. Triads with one
    connected dyad are counted per edge from the number of nodes adjacent to
    neither end; triads with two connected dyads per centre node from its
    in-, out- and mutual neighbour counts, minus the closed ones; and the
    seven closed classes from masked sparse products of the mutual and
    asymmetric adjacency matrices (e.g. ``030C`` is :math:`\mathrm{tr}(S^3)/3`
    with :math:`S` the asymmetric part). ``003`` is the remainder of
    :math:`\binom{N}{3}`. Counts are exact integers.

    Host-only (NumPy/SciPy): not traceable by ``jax.jit``, ``jax.vmap`` or
    ``jax.grad``.

    References
    ----------
    .. [1] V. Batagelj and A. Mrvar, "A subquadratic triad census algorithm
       for large sparse networks with small maximum degree," Social Networks
       23, 237-243 (2001).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.analysis.graph import motif_counts
        >>> cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # 0 -> 1 -> 2 -> 0
        >>> result = motif_counts(cycle)
        >>> result.triads['030C'], result.triads['003']
        (1, 0)
        >>> result.dyads
        {'mutual': 0, 'asymmetric': 3, 'null': 0}
        >>> result.reciprocity
        0.0
    """
    csr, _ = _to_csr(conn, weighted=False)
    n = _square(csr, 'motif_counts')
    n_self_loops = int(np.count_nonzero(csr.diagonal()))
    csr = _drop_diagonal(csr)
    adj = _binary(csr)
    adj_t = adj.T.tocsr()
    mutual = adj.multiply(adj_t).tocsr()
    asym = (adj - mutual).tocsr()
    asym.eliminate_zeros()
    asym_t = asym.T.tocsr()
    undirected = (adj + adj_t - mutual).tocsr()
    undirected.eliminate_zeros()

    n_out = _rowsum(asym)
    n_in = _colsum(asym)
    n_mut = _rowsum(mutual)
    degree = n_out + n_in + n_mut
    wedges = float(np.sum(degree * (degree - 1.0)))
    if wedges > max_wedges:
        raise ValueError(
            f'motif_counts: the graph has {int(wedges)} wedges, above max_wedges={max_wedges}. '
            f'Pass a larger `max_wedges` to compute the census anyway.'
        )

    def count(x):
        return int(round(x))

    n_mutual_pairs = mutual.nnz // 2
    n_asym_pairs = asym.nnz
    triads = dict.fromkeys(TRIAD_NAMES, 0)

    # one connected dyad: third node adjacent to neither end
    a_coo = asym.tocoo()
    m_coo = mutual.tocoo()
    triads['012'] = count(
        np.sum(n - degree[a_coo.row] - degree[a_coo.col])
        + _masked_product_sum(undirected, undirected, asym)
    )
    triads['102'] = count(
        (np.sum(n - degree[m_coo.row] - degree[m_coo.col])
         + _masked_product_sum(undirected, undirected, mutual)) / 2
    )

    # two connected dyads, by the types of the two dyads at the centre node
    # (out: centre -> x only, in: x -> centre only, mut: centre <-> x)
    pairs = {
        'out': (asym, n_out),
        'in': (asym_t, n_in),
        'mut': (mutual, n_mut),
    }

    def open_pairs(a, b):
        (pa, na), (pb, nb) = pairs[a], pairs[b]
        total = float(np.sum(na * (na - 1.0))) if a == b else float(np.sum(na * nb))
        closed = _masked_product_sum(pa.T.tocsr(), pb, undirected)
        return total - closed

    triads['021D'] = count(open_pairs('out', 'out') / 2)
    triads['021U'] = count(open_pairs('in', 'in') / 2)
    triads['021C'] = count(open_pairs('out', 'in'))
    triads['111D'] = count(open_pairs('mut', 'in'))
    triads['111U'] = count(open_pairs('mut', 'out'))
    triads['201'] = count(open_pairs('mut', 'mut') / 2)

    # three connected dyads
    triads['030T'] = count(_masked_product_sum(asym, asym, asym))
    triads['030C'] = count(_masked_product_sum(asym, asym, asym_t) / 3)
    triads['120D'] = count(_masked_product_sum(asym_t, asym, mutual) / 2)
    triads['120U'] = count(_masked_product_sum(asym, asym_t, mutual) / 2)
    triads['120C'] = count(_masked_product_sum(asym, asym, mutual))
    triads['210'] = count(_masked_product_sum(mutual, mutual, asym))
    triads['300'] = count(_masked_product_sum(mutual, mutual, mutual) / 6)

    triads['003'] = _comb(n, 3) - sum(v for k, v in triads.items() if k != '003')
    dyads = {
        'mutual': n_mutual_pairs,
        'asymmetric': n_asym_pairs,
        'null': _comb(n, 2) - n_mutual_pairs - n_asym_pairs,
    }
    n_edges = 2 * n_mutual_pairs + n_asym_pairs + n_self_loops
    reciprocity = 2 * n_mutual_pairs / n_edges if n_edges else float('nan')
    return MotifCounts(triads, dyads, reciprocity)


# ----------------------------------------------------------------------------
# spectral radius
# ----------------------------------------------------------------------------

def _single_threaded_blas():
    """Limit BLAS to one thread, if ``threadpoolctl`` is available.

    The dense algebra inside ARPACK works on small ``(ncv, ncv)`` blocks, for
    which a multi-threaded BLAS spends most of its time synchronising (an
    order of magnitude slower on many-core machines).
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api='blas')


@set_module_as('braintools.analysis.graph')
def spectral_radius(
    conn: Any,
    *,
    weighted: bool = True,
    method: str = 'auto',
) -> Union[float, u.Quantity]:
    r"""Compute the largest absolute eigenvalue of the connectivity matrix.

    :math:`\rho(W) = \max_k |\lambda_k(W)|`, which governs, for example, the
    linear stability of :math:`x \leftarrow W x` and the echo-state property
    of reservoirs. The matrix is treated as directed (not symmetrised).

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        Square connectivity, rows pre and columns post. The diagonal
        (self-connections) is kept, because the spectral radius is a property
        of the matrix. Repeated
        pairs of a ``ConnectionResult`` have their weights summed.
    weighted : bool, default True
        Use the weights (signed, e.g. inhibitory weights negative), or the
        binary adjacency matrix.
    method : {'auto', 'dense', 'arpack'}, default 'auto'
        ``'dense'``: all eigenvalues with ``numpy.linalg.eigvals`` (exact,
        :math:`O(N^3)` time and a dense :math:`N \times N` array);
        ``'arpack'``: the largest-magnitude eigenvalue with ARPACK (sparse);
        ``'auto'``: dense for at most 512 nodes, ARPACK above.

    Returns
    -------
    float or brainunit.Quantity
        The spectral radius, carrying the weight unit for Quantity weights
        with ``weighted=True``. ``0.0`` for a graph without edges or nodes.

    Raises
    ------
    ValueError
        If the connectivity is not square or ``method`` is unknown.
    RuntimeError
        If ARPACK does not converge.

    See Also
    --------
    degree_distribution : Node degrees.

    Notes
    -----
    ARPACK (``scipy.sparse.linalg.eigs`` for the eigenvalue of largest
    magnitude, Krylov dimension 64, fixed random start vector) only needs
    sparse matrix-vector products. When one eigenvalue stands out (e.g.
    non-negative or excitation-dominated weights, the Perron root) it
    converges in a few iterations and agrees with the dense result to about
    :math:`10^{-10}`. When many eigenvalues crowd the boundary of the spectral
    disc, as for large zero-mean random matrices (circular law), it may
    return an eigenvalue slightly inside the disc (a relative underestimate
    of up to about :math:`10^{-2}` was observed at :math:`N = 2 \times 10^4`);
    use ``method='dense'`` when exactness matters and :math:`N` allows. BLAS
    is limited to one thread during the computation (through
    ``threadpoolctl``, when installed), which is much faster for the small
    dense blocks of either method.

    Host-only (NumPy/SciPy): not traceable by ``jax.jit``, ``jax.vmap`` or
    ``jax.grad``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braintools.analysis.graph import spectral_radius
        >>> w = np.array([[0.0, 2.0], [0.5, 0.0]])  # eigenvalues +-1
        >>> spectral_radius(w)
        1.0
        >>> spectral_radius(w * u.nS)
        Quantity(1., "nS")
        >>> spectral_radius(w, weighted=False)
        1.0
    """
    if method not in ('auto', 'dense', 'arpack'):
        raise ValueError(f"`method` must be 'auto', 'dense' or 'arpack', got {method!r}.")
    csr, unit = _to_csr(conn, weighted)
    n = _square(csr, 'spectral_radius')
    if method == 'arpack' and n < 3:
        method = 'dense'  # ARPACK needs at least three nodes
    if n == 0 or csr.nnz == 0:
        radius = 0.0
    else:
        if method == 'auto':
            method = 'dense' if n <= _DENSE_EIG_MAX else 'arpack'
        with _single_threaded_blas():
            if method == 'dense':
                radius = float(np.max(np.abs(np.linalg.eigvals(csr.toarray()))))
            else:
                v0 = np.random.default_rng(0).standard_normal(n)
                try:
                    values = spla.eigs(
                        csr, k=1, ncv=min(n, _ARPACK_NCV), which='LM', v0=v0, return_eigenvectors=False,
                    )
                except spla.ArpackNoConvergence as e:  # pragma: no cover - ARPACK failure path
                    raise RuntimeError(f'spectral_radius: ARPACK did not converge ({e}).') from e
                radius = float(np.max(np.abs(values)))
    if unit is not None:
        return u.Quantity(radius, unit=unit)
    return radius


# ----------------------------------------------------------------------------
# small-world index
# ----------------------------------------------------------------------------

class SmallWorldResult(NamedTuple):
    """Result of :func:`small_world_index`.

    Attributes
    ----------
    sigma : float
        Small-world index :math:`\\sigma = (C / C_r) / (L / L_r)`.
    clustering : float
        Average undirected clustering :math:`C` of the graph.
    clustering_random : float
        Mean average clustering :math:`C_r` of the random reference graphs.
    path_length : float
        Characteristic path length :math:`L` of the (undirected) graph.
    path_length_random : float
        Mean characteristic path length :math:`L_r` of the random graphs.
    """
    sigma: float
    clustering: float
    clustering_random: float
    path_length: float
    path_length_random: float


def _triangular_pairs(index: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Decode linear indices of the strict upper triangle (row-major) to ``(i, j)``."""
    index = index.astype(np.int64)
    n_pairs = n * (n - 1) // 2
    # number of pairs in the rows after row i is (n - 1 - i)(n - 2 - i) / 2
    rest: np.ndarray = n_pairs - 1 - index
    t = np.floor((np.sqrt(8.0 * rest + 1.0) - 1.0) / 2.0).astype(np.int64)
    # guard against floating-point rounding of the square root
    t = np.where((t + 1) * (t + 2) // 2 <= rest, t + 1, t)
    t = np.where(t * (t + 1) // 2 > rest, t - 1, t)
    i = n - 2 - t
    row_start = i * (2 * n - i - 1) // 2
    j = index - row_start + i + 1
    return i, j


def _random_undirected(n: int, n_edges: int, rng: np.random.Generator) -> sp.csr_matrix:
    """Erdős–Rényi G(n, M): ``n_edges`` distinct undirected pairs, uniformly."""
    flat = rng.choice(n * (n - 1) // 2, size=n_edges, replace=False)
    i, j = _triangular_pairs(np.asarray(flat), n)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    return sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))


@set_module_as('braintools.analysis.graph')
def small_world_index(
    conn: Any,
    *,
    n_random: int = 10,
    n_sources: Optional[int] = None,
    seed: _Seed = None,
) -> SmallWorldResult:
    r"""Compute the small-world index :math:`\sigma` against random graphs.

    .. math::

        \sigma = \frac{C / C_r}{L / L_r},

    with :math:`C` the average clustering and :math:`L` the characteristic
    path length of the undirected, unweighted view of the graph, and
    :math:`C_r`, :math:`L_r` their means over ``n_random`` Erdős–Rényi
    graphs :math:`G(N, M)` with the same numbers of nodes and edges [1]_.
    :math:`\sigma > 1` indicates a small-world network.

    The reference is Erdős–Rényi with the same density, not the
    degree-preserving rewiring of ``networkx.sigma`` (whose Python edge swaps
    do not scale); for graphs with broad degree distributions the two
    references differ.

    Parameters
    ----------
    conn : ConnectionResult, scipy.sparse matrix or array_like
        Square connectivity; symmetrised, self-loops and weights ignored.
    n_random : int, default 10
        Number of random reference graphs.
    n_sources : int, optional
        Number of sampled BFS sources for the path lengths (see
        :func:`characteristic_path_length`); ``None`` uses all nodes.
    seed : int or numpy.random.Generator, optional
        Seed of the random graphs and the source sampling.

    Returns
    -------
    SmallWorldResult
        Named tuple ``(sigma, clustering, clustering_random, path_length,
        path_length_random)``. ``sigma`` is ``nan`` when a ratio is undefined
        (e.g. no triangles in the random graphs, or no edges).

    Raises
    ------
    ValueError
        If the connectivity is not square or ``n_random < 1``.

    See Also
    --------
    clustering_coefficient : The clustering :math:`C`.
    characteristic_path_length : The path length :math:`L`.

    Notes
    -----
    Path lengths of disconnected graphs (real or random) are over reachable
    pairs, as in :func:`characteristic_path_length`.

    Host-only (NumPy/SciPy): not traceable by ``jax.jit``, ``jax.vmap`` or
    ``jax.grad``.

    References
    ----------
    .. [1] M. D. Humphries and K. Gurney, "Network 'small-world-ness': a
       quantitative method for determining canonical network equivalence,"
       PLoS ONE 3, e0002051 (2008).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.analysis.graph import small_world_index
        >>> n, k = 200, 6
        >>> ring = np.zeros((n, n))
        >>> for shift in range(1, k // 2 + 1):
        ...     ring[np.arange(n), (np.arange(n) + shift) % n] = 1
        >>> result = small_world_index(ring + ring.T, n_random=3, seed=0)
        >>> round(result.clustering, 2)
        0.6
        >>> bool(result.sigma > 1)
        True
    """
    n_random = int(n_random)
    if n_random < 1:
        raise ValueError(f'`n_random` must be at least 1, got {n_random}.')
    csr, _ = _graph(conn, 'small_world_index')
    sym = _symmetrise(csr, weighted=False)
    n = sym.shape[0]
    n_edges = sym.nnz // 2
    rng = np.random.default_rng(seed)
    clustering = clustering_coefficient(sym, directed=False)
    path_length = characteristic_path_length(sym, directed=False, n_sources=n_sources, seed=rng).path_length
    c_rand = []
    l_rand = []
    for _ in range(n_random):
        rand = _random_undirected(n, n_edges, rng) if n > 1 else sym
        c_rand.append(clustering_coefficient(rand, directed=False))
        l_rand.append(
            characteristic_path_length(rand, directed=False, n_sources=n_sources, seed=rng).path_length
        )
    clustering_random = float(np.mean(c_rand))
    path_length_random = float(np.mean(l_rand))
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma = float((np.float64(clustering) / np.float64(clustering_random))
                      / (np.float64(path_length) / np.float64(path_length_random)))
    if not np.isfinite(sigma):
        sigma = float('nan')
    return SmallWorldResult(sigma, clustering, clustering_random, path_length, path_length_random)
