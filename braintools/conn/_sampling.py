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

"""Vectorised, memory-bounded sampling primitives shared by the connectivity generators.

All functions are host-side NumPy and take the generator's ``rng`` (a legacy
``np.random.RandomState`` or a ``np.random.Generator``). None of them builds an
``n_pre x n_post`` array: work on all-pairs quantities is split into row chunks
of at most ``MAX_CHUNK_PAIRS`` pairs.
"""

from typing import Iterator, Optional, Tuple, Union

import numpy as np

__all__ = [
    'MAX_CHUNK_PAIRS',
    'row_chunks',
    'randint',
    'sample_rows_without_replacement',
    'sample_flat_without_replacement',
    'bernoulli_pairs',
]

#: Largest number of (pre, post) pairs held at once by chunked computations.
MAX_CHUNK_PAIRS = 2 ** 22

RNG = Union[np.random.RandomState, np.random.Generator]


def row_chunks(n_rows: int, n_cols: int, max_pairs: int = MAX_CHUNK_PAIRS) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` row ranges whose blocks hold at most ``max_pairs`` pairs.

    Parameters
    ----------
    n_rows, n_cols : int
        Shape of the (virtual) pair matrix.
    max_pairs : int, default MAX_CHUNK_PAIRS
        Upper bound on ``(stop - start) * n_cols`` (at least one row per chunk).

    Yields
    ------
    tuple of int
        Consecutive half-open row ranges covering ``[0, n_rows)``.
    """
    step = max(1, int(max_pairs) // max(int(n_cols), 1))
    for start in range(0, int(n_rows), step):
        yield start, min(start + step, int(n_rows))


def randint(rng: RNG, high: int, size) -> np.ndarray:
    """Draw int64 integers uniformly from ``[0, high)`` with either RNG API."""
    if isinstance(rng, np.random.Generator):
        return rng.integers(0, high, size=size, dtype=np.int64)
    return rng.randint(0, high, size=size, dtype=np.int64)


def _random(rng: RNG, size) -> np.ndarray:
    return rng.random(size) if isinstance(rng, np.random.Generator) else rng.random_sample(size)


def sample_rows_without_replacement(
    rng: RNG,
    n_rows: int,
    n_cols: int,
    k: int,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw ``k`` distinct column indices for every row, uniformly.

    Parameters
    ----------
    rng : np.random.RandomState or np.random.Generator
        Random source.
    n_rows : int
        Number of rows (e.g. pre neurons for a fixed out-degree).
    n_cols : int
        Size of the population sampled from.
    k : int
        Number of distinct indices per row.
    exclude : np.ndarray of int, optional
        Shape ``(n_rows,)``; ``exclude[i]`` is never drawn for row ``i`` (used
        to forbid self-connections). Values must lie in ``[0, n_cols)``.

    Returns
    -------
    np.ndarray
        ``(n_rows, k)`` int64 array; each row is sorted and has distinct values.

    Raises
    ------
    ValueError
        If ``k`` is negative or larger than the number of allowed columns.

    Notes
    -----
    For ``k`` up to half the allowed columns, ``k`` values are drawn with
    replacement and duplicated values are redrawn until every row is distinct.
    The procedure treats all column labels symmetrically, so each row is a
    uniformly random ``k``-subset. Above half, the ``k`` smallest of i.i.d.
    uniform keys are taken in row chunks of at most ``MAX_CHUNK_PAIRS`` pairs.
    Excluded columns are removed by drawing from ``n_cols - 1`` values and
    shifting those at or above ``exclude[i]`` up by one.
    """
    n_rows, n_cols, k = int(n_rows), int(n_cols), int(k)
    avail = n_cols - (1 if exclude is not None else 0)
    if k < 0:
        raise ValueError(f'k must be non-negative, got {k}.')
    if k > avail:
        raise ValueError(
            f'Cannot draw {k} distinct indices per row from {avail} allowed values '
            f'(population of {n_cols}{", self excluded" if exclude is not None else ""}).'
        )
    if k == 0 or n_rows == 0:
        return np.zeros((n_rows, k), dtype=np.int64)

    if 2 * k > avail:
        out: np.ndarray = np.empty((n_rows, k), dtype=np.int64)
        for start, stop in row_chunks(n_rows, avail):
            keys = _random(rng, (stop - start, avail))
            idx = np.argpartition(keys, k - 1, axis=1)[:, :k] if k < avail else np.argsort(keys, axis=1)
            out[start:stop] = np.sort(idx, axis=1)
    else:
        out = np.sort(randint(rng, avail, (n_rows, k)), axis=1)
        while True:
            dup = out[:, 1:] == out[:, :-1]
            n_dup = int(dup.sum())
            if n_dup == 0:
                break
            out[:, 1:][dup] = randint(rng, avail, n_dup)
            out.sort(axis=1)

    if exclude is not None:
        exclude = np.asarray(exclude, dtype=np.int64).reshape(n_rows, 1)
        out = out + (out >= exclude)
    return out


def _sorted_unique(x: np.ndarray) -> np.ndarray:
    # ``np.unique`` on int64 takes a hash-based path in recent NumPy that is ~20x
    # slower than sorting for millions of values; the result is the same.
    y = np.sort(x)
    if y.size < 2:
        return y
    return y[np.concatenate(([True], y[1:] != y[:-1]))]


def sample_flat_without_replacement(rng: RNG, total: int, n: int) -> np.ndarray:
    """Draw ``n`` distinct integers from ``[0, total)`` uniformly, sorted.

    Parameters
    ----------
    rng : np.random.RandomState or np.random.Generator
        Random source.
    total : int
        Size of the index space (may exceed the int32 range).
    n : int
        Number of distinct values.

    Returns
    -------
    np.ndarray
        Sorted int64 array of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``n`` is negative or larger than ``total``.

    Notes
    -----
    Memory is ``O(n)`` when ``n <= total / 2`` (rejection of duplicates);
    otherwise a permutation of ``total <= 2 n`` values is taken.
    """
    total, n = int(total), int(n)
    if n < 0 or n > total:
        raise ValueError(f'Cannot draw {n} distinct values from {total}.')
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if 2 * n > total:
        return np.sort(rng.permutation(total)[:n].astype(np.int64))
    vals = _sorted_unique(randint(rng, total, n))
    while vals.size < n:
        vals = _sorted_unique(np.concatenate([vals, randint(rng, total, n - vals.size)]))
    return vals


def bernoulli_pairs(
    rng: RNG,
    n_pre: int,
    n_post: int,
    p: float,
    allow_self_connections: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample independent Bernoulli(``p``) connections of an ``n_pre x n_post`` block.

    The number of edges is drawn from ``Binomial(M, p)`` over the ``M`` allowed
    pairs, and that many distinct pairs are then drawn uniformly, which has
    exactly the distribution of independent per-pair Bernoulli draws without
    materialising the pair matrix.

    Parameters
    ----------
    rng : np.random.RandomState or np.random.Generator
        Random source.
    n_pre, n_post : int
        Block shape.
    p : float
        Connection probability in ``[0, 1]``.
    allow_self_connections : bool, default True
        If False, pairs ``(i, i)`` are excluded (requires ``n_pre == n_post``).

    Returns
    -------
    pre, post : np.ndarray
        int64 indices, sorted by ``pre`` then ``post``.

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``, or self-connections are excluded for a
        non-square block.
    """
    n_pre, n_post = int(n_pre), int(n_post)
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f'p must be in [0, 1], got {p}.')
    if not allow_self_connections and n_pre != n_post:
        raise ValueError('Excluding self-connections requires n_pre == n_post.')
    cols = n_post if allow_self_connections else n_post - 1
    total = n_pre * max(cols, 0)
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    n_edges = int(rng.binomial(total, p))
    flat = sample_flat_without_replacement(rng, total, n_edges)
    pre, post = np.divmod(flat, cols)
    if not allow_self_connections:
        post = post + (post >= pre)
    return pre.astype(np.int64), post.astype(np.int64)
