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

"""Named populations, projections between them, and whole-network assembly.

A :class:`Network` binds :class:`Connectivity` generators to named
:class:`Population` objects and assembles all projections into one block
matrix over the concatenated populations (host-side NumPy, then
``brainevent.CSR``).
"""

import copy
from typing import Dict, Optional, Sequence, Tuple, Union

import brainevent
import brainstate
import brainunit as u
import numpy as np

from ._base import Connectivity, ConnectionResult, _size_count

__all__ = [
    'Population',
    'Projection',
    'Network',
]


class Population:
    """A named group of neurons.

    Parameters
    ----------
    size : int or tuple of int
        Number of neurons, or the shape of a neuron grid (flattened in C order
        when indexing neurons).
    name : str
        Non-empty name, unique within a :class:`Network`.
    positions : array_like or Quantity, optional
        Neuron positions of shape ``(num, d)`` (or ``(num,)`` for a 1-D
        layout), passed to the connectivity generators of the projections.

    Raises
    ------
    TypeError
        If ``size`` is not an int or a tuple of ints, or ``name`` is not a str.
    ValueError
        If a size entry is not positive, ``name`` is empty, or ``positions``
        does not have ``num`` rows.

    See Also
    --------
    Projection, Network

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.conn import Population
        >>> grid = Population((10, 20), name='V1', positions=np.zeros((200, 2)) * u.um)
        >>> grid.num
        200
        >>> grid
        Population(name='V1', size=(10, 20))
    """

    __module__ = 'braintools.conn'

    def __init__(
        self,
        size: Union[int, Tuple[int, ...]],
        *,
        name: str,
        positions=None,
    ):
        self.size = _check_size(size)
        if not isinstance(name, str):
            raise TypeError(f'Population name must be a str, got {type(name).__name__}.')
        if not name:
            raise ValueError('Population name must be a non-empty string.')
        self.name = name
        if positions is not None:
            shape = u.math.shape(positions)
            if len(shape) not in (1, 2) or shape[0] != self.num:
                raise ValueError(
                    f'positions of population {name!r} must have shape ({self.num}, d), got {shape}.'
                )
        self.positions = positions

    @property
    def num(self) -> int:
        """Total number of neurons."""
        return _size_count(self.size)

    def __repr__(self) -> str:
        return f'Population(name={self.name!r}, size={self.size!r})'


class Projection:
    """A connectivity generator bound to a presynaptic and a postsynaptic population.

    Parameters
    ----------
    pre : Population
        Source population.
    post : Population
        Target population (may be ``pre`` itself for recurrent connections).
    connectivity : Connectivity
        Generator called with the populations' sizes and positions.
    name : str, optional
        Projection name; defaults to ``f"{pre.name}->{post.name}"``.

    Raises
    ------
    TypeError
        If ``pre``/``post`` are not :class:`Population` or ``connectivity`` is
        not a :class:`Connectivity`.
    ValueError
        If ``connectivity`` was constructed with a ``pre_size``/``post_size``
        whose neuron count differs from the population's, or ``name`` is empty.

    See Also
    --------
    Population, Network

    Notes
    -----
    Generators cache their result per (sizes, positions). Sharing one
    generator instance between two projections with the same sizes therefore
    gives both the same edges; use one instance per projection for
    independent draws.

    A generator's ``allow_self_connections=False`` (the default of
    :class:`FixedProb`) removes the pairs ``(i, i)`` when the two sides have
    the same size. Those pairs are autapses only when ``pre`` and ``post`` are
    the same population, so for a projection between two different
    populations the generator is built from a copy with self-connections
    allowed (sharing its random state); the passed generator is not modified.

    Examples
    --------
    .. code-block:: python

        >>> from braintools.conn import Population, Projection, FixedInDegree
        >>> E = Population(80, name='E')
        >>> I = Population(20, name='I')
        >>> proj = Projection(E, I, FixedInDegree(8, seed=0))
        >>> proj.name
        'E->I'
        >>> proj.build().shape
        (80, 20)
    """

    __module__ = 'braintools.conn'

    def __init__(
        self,
        pre: Population,
        post: Population,
        connectivity: Connectivity,
        *,
        name: Optional[str] = None,
    ):
        if not isinstance(pre, Population) or not isinstance(post, Population):
            raise TypeError('pre and post must be Population instances.')
        if not isinstance(connectivity, Connectivity):
            raise TypeError(
                f'connectivity must be a Connectivity instance, got {type(connectivity).__name__}.'
            )
        for side, pop, size in (('pre', pre, connectivity.pre_size), ('post', post, connectivity.post_size)):
            if size is not None and _size_count(size) != pop.num:
                raise ValueError(
                    f'connectivity {side}_size={size} does not match population {pop.name!r} '
                    f'of {pop.num} neurons.'
                )
        if name is None:
            name = f'{pre.name}->{post.name}'
        if not isinstance(name, str) or not name:
            raise ValueError(f'Projection name must be a non-empty string, got {name!r}.')
        self.pre = pre
        self.post = post
        self.connectivity = connectivity
        self.name = name
        self._generator = connectivity
        if pre is not post and getattr(connectivity, 'allow_self_connections', True) is False:
            # (i, i) between two different populations is not a self-connection.
            self._generator = copy.copy(connectivity)
            self._generator.allow_self_connections = True
            self._generator._cached_result = None
            self._generator._cache_key = None

    def build(self, recompute: bool = False) -> ConnectionResult:
        """Generate the projection's connections.

        Parameters
        ----------
        recompute : bool, default False
            Redraw even if the generator has a cached result.

        Returns
        -------
        ConnectionResult
            Local indices (``0 .. pre.num - 1`` and ``0 .. post.num - 1``).
        """
        return self._generator(
            pre_size=self.pre.size,
            post_size=self.post.size,
            pre_positions=self.pre.positions,
            post_positions=self.post.positions,
            recompute=recompute,
        )

    def __repr__(self) -> str:
        return (f'Projection(name={self.name!r}, pre={self.pre.name!r}, post={self.post.name!r}, '
                f'connectivity={type(self.connectivity).__name__})')


class Network:
    """A set of populations and the projections between them.

    Neurons of all populations are numbered globally by concatenating the
    populations in the given order: neuron ``i`` of population ``p`` has
    global index ``offsets[p] + i``.

    Parameters
    ----------
    populations : sequence of Population
        Populations, with unique names.
    projections : sequence of Projection, default ()
        Projections, with unique names, whose ``pre`` and ``post`` are among
        ``populations``.

    Attributes
    ----------
    populations : dict of str to Population
        Populations by name, in order.
    projections : dict of str to Projection
        Projections by name, in order.
    offsets : dict of str to int
        Global index of the first neuron of each population.
    n_neurons : int
        Total number of neurons.

    Raises
    ------
    TypeError
        If an element is not a :class:`Population`/:class:`Projection`.
    ValueError
        If population or projection names are duplicated, or a projection
        refers to a population that is not part of the network.

    See Also
    --------
    Population, Projection, FixedInDegree

    Notes
    -----
    Building is host-side NumPy (not traceable by ``jax.jit``). The block
    matrix from :meth:`to_csr` is a ``brainevent.CSR`` whose products support
    ``jit``/``vmap``/``grad``.

    Examples
    --------
    An E/I network with fixed in-degrees (each neuron receives 80 excitatory
    and 20 inhibitory inputs):

    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.conn import Population, Projection, Network, FixedInDegree
        >>> E = Population(800, name='E')
        >>> I = Population(200, name='I')
        >>> net = Network(
        ...     [E, I],
        ...     [
        ...         Projection(E, E, FixedInDegree(80, weight=0.6 * u.mS, seed=1)),
        ...         Projection(E, I, FixedInDegree(80, weight=0.6 * u.mS, seed=2)),
        ...         Projection(I, E, FixedInDegree(20, weight=-6.7 * u.mS, seed=3)),
        ...         Projection(I, I, FixedInDegree(20, weight=-6.7 * u.mS, seed=4)),
        ...     ],
        ... )
        >>> net.offsets
        {'E': 0, 'I': 800}
        >>> results = net.build()
        >>> list(results)
        ['E->E', 'E->I', 'I->E', 'I->I']
        >>> W = net.to_csr()
        >>> W.shape, W.nse
        ((1000, 1000), 100000)
        >>> dense = np.asarray(u.get_mantissa(W.todense()))
        >>> exc_inputs = (dense[:800] > 0).sum(axis=0)   # from E rows
        >>> inh_inputs = (dense[800:] < 0).sum(axis=0)   # from I rows
        >>> np.unique(exc_inputs), np.unique(inh_inputs)
        (array([80]), array([20]))
    """

    __module__ = 'braintools.conn'

    def __init__(
        self,
        populations: Sequence[Population],
        projections: Sequence[Projection] = (),
    ):
        self.populations: Dict[str, Population] = {}
        for pop in populations:
            if not isinstance(pop, Population):
                raise TypeError(f'populations must contain Population instances, got {type(pop).__name__}.')
            if pop.name in self.populations:
                raise ValueError(f'Duplicate population name {pop.name!r}.')
            self.populations[pop.name] = pop

        self.projections: Dict[str, Projection] = {}
        for proj in projections:
            if not isinstance(proj, Projection):
                raise TypeError(f'projections must contain Projection instances, got {type(proj).__name__}.')
            if proj.name in self.projections:
                raise ValueError(f'Duplicate projection name {proj.name!r}.')
            for pop in (proj.pre, proj.post):
                if self.populations.get(pop.name) is not pop:
                    raise ValueError(
                        f'Projection {proj.name!r} refers to population {pop.name!r}, '
                        f'which is not part of this network.'
                    )
            self.projections[proj.name] = proj

        self.offsets: Dict[str, int] = {}
        total = 0
        for name, pop in self.populations.items():
            self.offsets[name] = total
            total += pop.num
        self.n_neurons: int = total

    def build(self, recompute: bool = False) -> Dict[str, ConnectionResult]:
        """Generate every projection.

        Parameters
        ----------
        recompute : bool, default False
            Redraw even where a generator has a cached result.

        Returns
        -------
        dict of str to ConnectionResult
            Results with local (per-population) indices, keyed by projection
            name in order.
        """
        return {name: proj.build(recompute=recompute) for name, proj in self.projections.items()}

    def to_csr(self, data: str = 'weight') -> brainevent.CSR:
        """Assemble all projections into one block matrix over the whole network.

        Parameters
        ----------
        data : {'weight', 'delay'}, default 'weight'
            Which edge values to store. Missing weights count as ``1.0`` and
            missing delays as ``0.0`` (dimensionless).

        Returns
        -------
        brainevent.CSR
            Matrix of shape ``(n_neurons, n_neurons)`` with edge
            ``(offsets[pre] + i, offsets[post] + j)`` for every edge ``(i, j)``
            of every projection. Values of compatible units are converted to
            the unit of the first non-empty projection. If every projection
            has the same homogeneous value, the data stays a scalar.

        Raises
        ------
        ValueError
            If the projections' values have different physical dimensions
            (e.g. ``nS`` and dimensionless), or ``data`` is invalid.

        Notes
        -----
        Builds the projections (or reuses the generators' cached results).
        Memory is ``O(total edges)``; rows are sorted by global presynaptic
        index and columns ascending within a row.
        """
        results = self.build()
        pre_parts, post_parts, values, sizes = [], [], [], []
        ref_unit, ref_name = None, None
        for name, proj in self.projections.items():
            res = results[name]
            if res.n_connections == 0:
                continue
            mantissa, unit = u.split_mantissa_unit(res._export_values(data))
            if ref_unit is None:
                ref_unit, ref_name = unit, name
            elif unit.dim != ref_unit.dim:
                raise ValueError(
                    f'{data.capitalize()} units must agree across projections: '
                    f'{ref_name!r} has unit {ref_unit}, {name!r} has unit {unit}.'
                )
            elif unit != ref_unit:
                mantissa = u.Quantity(mantissa, unit=unit).to(ref_unit).mantissa
            values.append(np.asarray(mantissa).reshape(-1))
            sizes.append(res.n_connections)
            pre_parts.append(res.pre_indices + self.offsets[proj.pre.name])
            post_parts.append(res.post_indices + self.offsets[proj.post.name])

        n = self.n_neurons
        if not pre_parts:
            return brainevent.CSR(
                (np.zeros(0, brainstate.environ.dftype()), np.zeros(0, np.int64), np.zeros(n + 1, np.int64)),
                shape=(n, n),
            )
        pre = np.concatenate(pre_parts)
        post = np.concatenate(post_parts)

        order = np.lexsort((post, pre))
        if all(v.size == 1 and v[0] == values[0][0] for v in values):
            data_arr = values[0].reshape(())
        else:
            data_arr = np.concatenate([
                np.full(m, v[0], dtype=v.dtype) if v.size == 1 else v for v, m in zip(values, sizes)
            ])[order]

        indptr: np.ndarray = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
        return brainevent.CSR(
            (u.maybe_decimal(data_arr * ref_unit), post[order], indptr), shape=(n, n)
        )

    def __repr__(self) -> str:
        return (f'Network(populations={list(self.populations)}, '
                f'projections={list(self.projections)}, n_neurons={self.n_neurons})')


def _check_size(size) -> Union[int, Tuple[int, ...]]:
    """Validate a population size (positive int or tuple of positive ints)."""
    if isinstance(size, (bool, np.bool_)):
        raise TypeError('Population size must be an int or a tuple of ints, got a bool.')
    dims: Tuple[int, ...]
    out: Union[int, Tuple[int, ...]]
    if isinstance(size, (int, np.integer)):
        dims = (int(size),)
        out = int(size)
    elif isinstance(size, (tuple, list)) and size and all(
        isinstance(s, (int, np.integer)) and not isinstance(s, (bool, np.bool_)) for s in size
    ):
        dims = tuple(int(s) for s in size)
        out = dims
    else:
        raise TypeError(f'Population size must be an int or a tuple of ints, got {size!r}.')
    if any(d <= 0 for d in dims):
        raise ValueError(f'Population size entries must be positive, got {size!r}.')
    return out
