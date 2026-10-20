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

The functions take a :class:`braintools.conn.ConnectionResult`, a SciPy sparse
matrix, or a dense adjacency matrix (NumPy, JAX or a nested list), and work on
sparse matrices throughout, so they scale to large networks without building
an ``N x N`` array. They run on the host and are not jit/vmap/grad compatible.
Self-connections are handled as in networkx: they are counted by
:func:`degree_distribution` (once in the in- and out-degree, twice in the total
degree), :func:`modularity` and the reciprocity of :func:`motif_counts`, and
kept by :func:`spectral_radius`; they are ignored by
:func:`clustering_coefficient`, :func:`characteristic_path_length`, the triad
census of :func:`motif_counts` and :func:`small_world_index`.

- :func:`degree_distribution`: in-, out- or total degrees and their histogram.
- :func:`clustering_coefficient`: directed (Fagiolo) or undirected clustering,
  optionally weighted.
- :func:`characteristic_path_length`: mean shortest path length and global
  efficiency, exact or from sampled sources.
- :func:`modularity`: Newman (undirected) or Leicht-Newman (directed)
  modularity of a partition.
- :func:`motif_counts`: the 16-class triad census, dyad counts and reciprocity.
- :func:`spectral_radius`: largest eigenvalue magnitude of the weight matrix.
- :func:`small_world_index`: small-world coefficient sigma against
  Erdos-Renyi graphs with the same numbers of nodes and edges.

Examples
--------
.. code-block:: python

    >>> import brainunit as u
    >>> from braintools.conn import FixedProb
    >>> from braintools.analysis import graph
    >>> conn = FixedProb(0.1, weight=0.5 * u.nS, seed=0)(pre_size=200, post_size=200)
    >>> dd = graph.degree_distribution(conn, direction='in')
    >>> dd.degrees.shape
    (200,)
    >>> 0.0 < graph.clustering_coefficient(conn) < 0.2
    True
"""

from ._graph import (
    DegreeDistribution,
    PathLengthResult,
    MotifCounts,
    SmallWorldResult,
    degree_distribution,
    clustering_coefficient,
    characteristic_path_length,
    modularity,
    motif_counts,
    spectral_radius,
    small_world_index,
)

__all__ = [
    'degree_distribution',
    'clustering_coefficient',
    'characteristic_path_length',
    'modularity',
    'motif_counts',
    'spectral_radius',
    'small_world_index',
    'DegreeDistribution',
    'PathLengthResult',
    'MotifCounts',
    'SmallWorldResult',
]
