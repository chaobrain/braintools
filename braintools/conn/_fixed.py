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

"""Fixed-number connectivity: fixed out-degree, fixed in-degree and fixed total number.

All generators are host-side NumPy, vectorised over neurons and ``O(n_edges)``
in memory (no ``n_pre x n_post`` array is ever built). Their results map
directly onto brainevent's fixed-number formats through
:meth:`ConnectionResult.to_fixed_num_per_pre` and
:meth:`ConnectionResult.to_fixed_num_per_post`.
"""

from typing import Optional, Tuple, Union

import brainunit as u
import numpy as np

from braintools.init._init_base import param, Initializer
from ._base import PointConnectivity, ConnectionResult, _size_count
from ._sampling import (
    MAX_CHUNK_PAIRS,
    randint,
    sample_rows_without_replacement,
    sample_flat_without_replacement,
)

__all__ = [
    'FixedOutDegree',
    'FixedInDegree',
    'FixedTotalNum',
]

Count = Union[int, float]
Size = Union[int, Tuple[int, ...]]


def _check_count(value, name: str) -> Count:
    """Validate a count given as a non-negative int or a fraction in (0, 1]."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an int or a float fraction, got a bool.')
    if isinstance(value, (int, np.integer)):
        if value < 0:
            raise ValueError(f'{name} must be non-negative, got {value}.')
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f'A fractional {name} must lie in (0, 1], got {value}; '
                f'pass an int for an absolute count.'
            )
        return float(value)
    raise TypeError(f'{name} must be an int or a float fraction, got {type(value).__name__}.')


def _resolve_count(value: Count, n: int) -> int:
    """Turn an int count or a fraction of ``n`` into an int count."""
    if isinstance(value, int):
        return value
    return int(round(value * n))


def _sample_rows(rng, n_rows: int, n_cols: int, k: int, exclude_self: bool, with_replacement: bool):
    """Draw ``k`` column indices per row, sorted per row, excluding ``i`` from row ``i`` if asked."""
    exclude: Optional[np.ndarray] = np.arange(n_rows, dtype=np.int64) if exclude_self else None
    if not with_replacement:
        return sample_rows_without_replacement(rng, n_rows, n_cols, k, exclude=exclude)
    avail = n_cols - int(exclude_self)
    out = randint(rng, avail, (n_rows, k)) if k > 0 and n_rows > 0 else np.zeros((n_rows, k), np.int64)
    if exclude is not None:
        out += out >= exclude[:, None]
    out.sort(axis=1)
    return out


def _edge_distances(pre_positions, post_positions, pre, post):
    """Euclidean distance of every edge, computed in bounded chunks.

    Returns ``None`` when either position array is missing.
    """
    if pre_positions is None or post_positions is None:
        return None
    pre_m, unit = u.split_mantissa_unit(pre_positions)
    post_m = u.Quantity(post_positions).to(unit).mantissa
    pre_m = np.asarray(pre_m, dtype=float)
    post_m = np.asarray(post_m, dtype=float)
    if pre_m.ndim == 1:
        pre_m = pre_m[:, None]
    if post_m.ndim == 1:
        post_m = post_m[:, None]
    out = np.empty(pre.shape[0], dtype=float)
    step = max(1, MAX_CHUNK_PAIRS // max(pre_m.shape[1], 1))
    for start in range(0, pre.shape[0], step):
        diff = pre_m[pre[start:start + step]] - post_m[post[start:start + step]]
        out[start:start + step] = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    return u.maybe_decimal(out * unit)


class _FixedNumber(PointConnectivity):
    """Shared constructor, validation and result assembly of the fixed-number classes."""

    _count_name = 'k'
    _pattern = ''
    _metadata_key = 'degree'

    def __init__(
        self,
        count: Count,
        *,
        allow_self_connections: bool = True,
        with_replacement: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.count = _check_count(count, self._count_name)
        self.allow_self_connections = bool(allow_self_connections)
        self.with_replacement = bool(with_replacement)
        self.weight_init = weight
        self.delay_init = delay

    def _exclude_self(self, pre_num: int, post_num: int) -> bool:
        # Self-connections are only defined between equally sized populations
        # (the same convention as ``Random``).
        return not self.allow_self_connections and pre_num == post_num

    def _check_available(self, k: int, avail: int, what: str, extra: str):
        if self.with_replacement:
            if k > 0 and avail == 0:
                raise ValueError(f'{type(self).__name__}: no {what} available to draw from{extra}.')
        elif k > avail:
            raise ValueError(
                f'{type(self).__name__}: {self._count_name}={k} exceeds the {avail} available '
                f'{what}{extra}; use a smaller {self._count_name} or with_replacement=True.'
            )

    def _result(self, pre, post, pre_size, post_size, pre_positions, post_positions, count):
        n = int(pre.shape[0])
        metadata = {
            'pattern': self._pattern,
            self._metadata_key: count,
            'with_replacement': self.with_replacement,
            'allow_self_connections': self.allow_self_connections,
            'weight_initialization': self.weight_init,
            'delay_initialization': self.delay_init,
        }
        weights = delays = None
        if n > 0:
            kw = dict(pre_size=pre_size, post_size=post_size, pre_positions=pre_positions,
                      post_positions=post_positions, rng=self.rng)
            if callable(self.weight_init) or callable(self.delay_init):
                distances = _edge_distances(pre_positions, post_positions, pre, post)
                if distances is not None:
                    kw['distances'] = distances
            weights = param(self.weight_init, n, param_type='weight', **kw)
            delays = param(self.delay_init, n, param_type='delay', **kw)
        return ConnectionResult(
            pre.astype(np.int64, copy=False),
            post.astype(np.int64, copy=False),
            pre_size=pre_size,
            post_size=post_size,
            weights=weights,
            delays=delays,
            pre_positions=pre_positions,
            post_positions=post_positions,
            model_type='point',
            metadata=metadata,
        )


class FixedOutDegree(_FixedNumber):
    """Connect every presynaptic neuron to exactly ``k`` postsynaptic neurons.

    The ``k`` targets of each presynaptic neuron are drawn uniformly from the
    postsynaptic population, independently across presynaptic neurons.

    Parameters
    ----------
    k : int or float
        Out-degree. An int is an absolute count; a float in ``(0, 1]`` is a
        fraction of the allowed targets, resolved as ``round(k * post_num)``
        (``round(k * (post_num - 1))`` when self-connections are excluded).
    allow_self_connections : bool, default True
        If False and the populations have equal size, neuron ``i`` never
        targets neuron ``i`` (the ``k`` targets are drawn from the other
        ``post_num - 1`` neurons). Ignored when ``pre_num != post_num``.
    with_replacement : bool, default False
        If True, targets are drawn with replacement, so a pair may appear more
        than once (duplicate edges) and ``k`` may exceed ``post_num``.
    weight : Initializer, optional
        Weight initialization (initializer, scalar, Quantity or array), drawn
        once per edge through :func:`braintools.init.param`. When both position
        arrays are given, per-edge ``distances`` are passed to the initializer
        so distance-dependent weights work. If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization, handled like ``weight``. If None, no delays.
    seed : int, optional
        Seed of the generator's private ``np.random.RandomState``.
    **kwargs
        Further arguments of :class:`~braintools.conn.Connectivity`
        (``pre_size``, ``post_size``).

    Raises
    ------
    TypeError
        If ``k`` is neither an int nor a float.
    ValueError
        If ``k`` is negative or a float outside ``(0, 1]``; at generation time
        if ``k`` exceeds the available targets (``post_num``, or
        ``post_num - 1`` without self-connections) and
        ``with_replacement=False``.

    See Also
    --------
    FixedInDegree : Fixed number of inputs per postsynaptic neuron.
    FixedTotalNum : Fixed total number of edges.
    FixedProb : Independent per-pair connection probability.
    ConnectionResult.to_fixed_num_per_pre : Export to ``brainevent.FixedNumPerPre``.

    Notes
    -----
    Without replacement each row is a uniformly random ``k``-subset, drawn by
    a vectorised sampler that redraws only duplicated entries (or, when
    ``k > post_num / 2``, takes the ``k`` smallest of i.i.d. uniform keys in row
    chunks of at most ``2**22`` pairs). Memory is ``O(pre_num * k)``; time is
    ``O(pre_num * k)``, or ``O(pre_num * post_num)`` on the key path, where
    ``k`` is already a large fraction of ``post_num``. Edges are returned sorted by
    presynaptic then postsynaptic index. ``metadata`` records ``'degree'``
    (the resolved int ``k``), ``'with_replacement'`` and
    ``'allow_self_connections'``.

    Generation is host-side NumPy and is not traceable by ``jax.jit``; the
    exported brainevent matrices are.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> from braintools.conn import FixedOutDegree
        >>> conn = FixedOutDegree(10, weight=0.5 * u.nS, seed=0)
        >>> result = conn(pre_size=100, post_size=200)
        >>> result.n_connections
        1000
        >>> np.unique(np.bincount(result.pre_indices, minlength=100))
        array([10])
        >>> w = result.to_fixed_num_per_pre()
        >>> w.indices.shape
        (100, 10)

        >>> # k as a fraction of post_num, no self-connections
        >>> result = FixedOutDegree(0.1, allow_self_connections=False, seed=0)(50, 50)
        >>> result.metadata['degree']
        5
        >>> bool(np.any(result.pre_indices == result.post_indices))
        False
    """

    __module__ = 'braintools.conn'
    _count_name = 'k'
    _pattern = 'fixed_out_degree'
    _metadata_key = 'degree'

    def __init__(
        self,
        k: Count,
        *,
        allow_self_connections: bool = True,
        with_replacement: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            k,
            allow_self_connections=allow_self_connections,
            with_replacement=with_replacement,
            weight=weight,
            delay=delay,
            seed=seed,
            **kwargs
        )

    @property
    def k(self) -> Count:
        """The out-degree as given (int count or fraction)."""
        return self.count

    def generate(
        self,
        pre_size: Size,
        post_size: Size,
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate the fixed out-degree connections."""
        pre_num, post_num = _size_count(pre_size), _size_count(post_size)
        exclude = self._exclude_self(pre_num, post_num)
        k = _resolve_count(self.count, post_num - int(exclude))
        self._check_available(
            k, post_num - int(exclude), 'post neurons',
            f' (post_num={post_num}{", self-connections excluded" if exclude else ""})'
        )
        post = _sample_rows(self.rng, pre_num, post_num, k, exclude, self.with_replacement)
        pre: np.ndarray = np.repeat(np.arange(pre_num, dtype=np.int64), k)
        return self._result(pre, post.reshape(-1), pre_size, post_size, pre_positions, post_positions, k)


class FixedInDegree(_FixedNumber):
    """Connect every postsynaptic neuron to exactly ``k`` presynaptic neurons.

    The ``k`` sources of each postsynaptic neuron are drawn uniformly from the
    presynaptic population, independently across postsynaptic neurons.

    Parameters
    ----------
    k : int or float
        In-degree. An int is an absolute count; a float in ``(0, 1]`` is a
        fraction of the allowed sources, resolved as ``round(k * pre_num)``
        (``round(k * (pre_num - 1))`` when self-connections are excluded).
    allow_self_connections : bool, default True
        If False and the populations have equal size, neuron ``j`` never
        receives from neuron ``j`` (sources are drawn from the other
        ``pre_num - 1`` neurons). Ignored when ``pre_num != post_num``.
    with_replacement : bool, default False
        If True, sources are drawn with replacement, so a pair may appear more
        than once (duplicate edges) and ``k`` may exceed ``pre_num``.
    weight : Initializer, optional
        Weight initialization (initializer, scalar, Quantity or array), drawn
        once per edge through :func:`braintools.init.param`. When both position
        arrays are given, per-edge ``distances`` are passed to the initializer.
        If None, no weights are generated.
    delay : Initializer, optional
        Delay initialization, handled like ``weight``. If None, no delays.
    seed : int, optional
        Seed of the generator's private ``np.random.RandomState``.
    **kwargs
        Further arguments of :class:`~braintools.conn.Connectivity`
        (``pre_size``, ``post_size``).

    Raises
    ------
    TypeError
        If ``k`` is neither an int nor a float.
    ValueError
        If ``k`` is negative or a float outside ``(0, 1]``; at generation time
        if ``k`` exceeds the available sources (``pre_num``, or
        ``pre_num - 1`` without self-connections) and
        ``with_replacement=False``.

    See Also
    --------
    FixedOutDegree : Fixed number of outputs per presynaptic neuron.
    FixedTotalNum : Fixed total number of edges.
    ConnectionResult.to_fixed_num_per_post : Export to ``brainevent.FixedNumPerPost``.
    Network : Populations and projections built from connectivities.

    Notes
    -----
    Uses the same sampler as :class:`FixedOutDegree` with the roles of pre and
    post swapped; memory is ``O(post_num * k)``. Edges are returned
    grouped by postsynaptic neuron (sorted by post, then pre index), which is
    the layout of ``brainevent.FixedNumPerPost``. ``metadata`` records
    ``'degree'`` (the resolved int ``k``), ``'with_replacement'`` and
    ``'allow_self_connections'``.

    Generation is host-side NumPy and is not traceable by ``jax.jit``; the
    exported brainevent matrices are.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from braintools.conn import FixedInDegree
        >>> result = FixedInDegree(20, weight=0.1, seed=1)(pre_size=200, post_size=50)
        >>> np.unique(np.bincount(result.post_indices, minlength=50))
        array([20])
        >>> w = result.to_fixed_num_per_post()
        >>> w.shape, w.indices.shape
        ((200, 50), (50, 20))
        >>> drive = jnp.ones(200) @ w  # each post neuron sums 20 inputs of 0.1
        >>> bool(jnp.allclose(drive, 2.0))
        True
    """

    __module__ = 'braintools.conn'
    _count_name = 'k'
    _pattern = 'fixed_in_degree'
    _metadata_key = 'degree'

    def __init__(
        self,
        k: Count,
        *,
        allow_self_connections: bool = True,
        with_replacement: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            k,
            allow_self_connections=allow_self_connections,
            with_replacement=with_replacement,
            weight=weight,
            delay=delay,
            seed=seed,
            **kwargs
        )

    @property
    def k(self) -> Count:
        """The in-degree as given (int count or fraction)."""
        return self.count

    def generate(
        self,
        pre_size: Size,
        post_size: Size,
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate the fixed in-degree connections."""
        pre_num, post_num = _size_count(pre_size), _size_count(post_size)
        exclude = self._exclude_self(pre_num, post_num)
        k = _resolve_count(self.count, pre_num - int(exclude))
        self._check_available(
            k, pre_num - int(exclude), 'pre neurons',
            f' (pre_num={pre_num}{", self-connections excluded" if exclude else ""})'
        )
        pre = _sample_rows(self.rng, post_num, pre_num, k, exclude, self.with_replacement)
        post: np.ndarray = np.repeat(np.arange(post_num, dtype=np.int64), k)
        return self._result(pre.reshape(-1), post, pre_size, post_size, pre_positions, post_positions, k)


class FixedTotalNum(_FixedNumber):
    """Draw exactly ``n`` edges uniformly over all allowed (pre, post) pairs.

    Parameters
    ----------
    n : int or float
        Number of edges. An int is an absolute count; a float in ``(0, 1]`` is
        a fraction of the allowed pairs, resolved as
        ``round(n * pre_num * post_num)`` (``round(n * pre_num * (post_num - 1))``
        when self-connections are excluded).
    allow_self_connections : bool, default True
        If False and the populations have equal size, pairs ``(i, i)`` are
        never drawn. Ignored when ``pre_num != post_num``.
    with_replacement : bool, default False
        If True, the ``n`` pairs are drawn independently with replacement, so a
        pair may appear more than once and ``n`` may exceed the number of pairs.
    weight : Initializer, optional
        Weight initialization (initializer, scalar, Quantity or array), drawn
        once per edge through :func:`braintools.init.param`. If None, no
        weights are generated.
    delay : Initializer, optional
        Delay initialization, handled like ``weight``. If None, no delays.
    seed : int, optional
        Seed of the generator's private ``np.random.RandomState``.
    **kwargs
        Further arguments of :class:`~braintools.conn.Connectivity`
        (``pre_size``, ``post_size``).

    Raises
    ------
    TypeError
        If ``n`` is neither an int nor a float.
    ValueError
        If ``n`` is negative or a float outside ``(0, 1]``; at generation time
        if ``n`` exceeds the number of allowed pairs and
        ``with_replacement=False``.

    See Also
    --------
    FixedOutDegree, FixedInDegree, FixedProb

    Notes
    -----
    The pairs are ``n`` distinct flat indices into the (virtual)
    ``pre_num x post_num`` matrix (``pre_num x (post_num - 1)`` without
    self-connections, remapped by shifting), drawn by rejection of duplicates
    when ``n`` is at most half the pairs and by a permutation of at most
    ``2 n`` values otherwise, so memory is ``O(n)``. Unlike
    :class:`FixedProb`, the edge count is exact rather than binomial. Edges are
    returned sorted by presynaptic then postsynaptic index. ``metadata``
    records ``'n'`` (the resolved int count), ``'with_replacement'`` and
    ``'allow_self_connections'``.

    Generation is host-side NumPy and is not traceable by ``jax.jit``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from braintools.conn import FixedTotalNum
        >>> result = FixedTotalNum(500, allow_self_connections=False, seed=0)(100, 100)
        >>> result.n_connections
        500
        >>> codes = result.pre_indices * 100 + result.post_indices
        >>> np.unique(codes).size, bool(np.any(result.pre_indices == result.post_indices))
        (500, False)
    """

    __module__ = 'braintools.conn'
    _count_name = 'n'
    _pattern = 'fixed_total_num'
    _metadata_key = 'n'

    def __init__(
        self,
        n: Count,
        *,
        allow_self_connections: bool = True,
        with_replacement: bool = False,
        weight: Optional[Initializer] = None,
        delay: Optional[Initializer] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            n,
            allow_self_connections=allow_self_connections,
            with_replacement=with_replacement,
            weight=weight,
            delay=delay,
            seed=seed,
            **kwargs
        )

    @property
    def n(self) -> Count:
        """The number of edges as given (int count or fraction)."""
        return self.count

    def generate(
        self,
        pre_size: Size,
        post_size: Size,
        pre_positions: Optional[np.ndarray] = None,
        post_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> ConnectionResult:
        """Generate the fixed-total-number connections."""
        pre_num, post_num = _size_count(pre_size), _size_count(post_size)
        exclude = self._exclude_self(pre_num, post_num)
        cols = post_num - int(exclude)
        total = pre_num * cols
        n = _resolve_count(self.count, total)
        self._check_available(
            n, total, 'pairs',
            f' ({pre_num} x {post_num}{", self-connections excluded" if exclude else ""})'
        )
        if self.with_replacement:
            flat = np.sort(randint(self.rng, total, n)) if n > 0 else np.zeros(0, np.int64)
        else:
            flat = sample_flat_without_replacement(self.rng, total, n)
        pre, post = np.divmod(flat, max(cols, 1))
        if exclude:
            post = post + (post >= pre)
        return self._result(pre, post, pre_size, post_size, pre_positions, post_positions, n)
