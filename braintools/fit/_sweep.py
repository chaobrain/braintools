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

"""Vectorised parameter sweeps."""

from typing import Any, Callable, Dict, Optional, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import set_module_as

__all__ = [
    'sweep',
]


def _reshape(tree: Any, shape: Tuple[int, ...]) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.reshape(shape + tuple(leaf.shape[1:])),
        tree,
        is_leaf=u.math.is_quantity,
    )


@set_module_as('braintools.fit')
def sweep(
    fn: Callable[[Dict[str, Any]], Any],
    grid: Dict[str, Any],
    *,
    mode: str = 'product',
    batch_size: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Evaluate a pure function over a grid of parameter values.

    Parameters
    ----------
    fn : callable
        ``fn(params) -> PyTree``, a pure (``vmap``-able) function of a dict
        of scalar parameters.
    grid : dict
        ``{name: values}`` with 1-D arrays (or ``brainunit`` Quantities) of
        candidate values.
    mode : {'product', 'zip'}, default 'product'
        ``'product'`` evaluates the cartesian product of the value arrays (grid
        shape ``(len(v1), len(v2), ...)`` in the order of ``grid``); ``'zip'``
        pairs equal-length arrays element-wise (grid shape ``(n,)``).
    batch_size : int, optional
        Number of grid points evaluated together by one ``vmap``-ped call.
        The chunks run one after another through :func:`jax.lax.map`, so the
        peak memory is bounded by ``batch_size`` evaluations; a final partial
        chunk is handled when ``batch_size`` does not divide the grid size.
        ``None`` evaluates the whole grid in a single ``vmap``.

    Returns
    -------
    outputs : PyTree
        The outputs of ``fn`` stacked over the grid: every leaf has shape
        ``grid_shape + leaf_shape``.
    table : dict
        The flattened parameter table ``{name: values}``; every entry has
        length ``prod(grid_shape)`` and row ``i`` is the input of output
        ``outputs.reshape(-1, ...)[i]``.

    Raises
    ------
    ValueError
        If ``grid`` is empty, a value array is not 1-D or is empty, ``mode``
        is unknown, the arrays have different lengths in ``'zip'`` mode, or
        ``batch_size`` is not a positive integer.

    See Also
    --------
    Fitter : Optimise the parameters instead of scanning them.

    Notes
    -----
    The grid shape is static, so ``sweep`` can be called inside
    ``jax.jit``; it is differentiable with respect to values ``fn`` closes
    over.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from braintools.fit import sweep
        >>> out, table = sweep(lambda p: p['a'] * p['b'],
        ...                    {'a': jnp.array([1., 2.]), 'b': jnp.array([10., 20., 30.])})
        >>> out
        Array([[10., 20., 30.],
               [20., 40., 60.]], dtype=float32)
        >>> table['b']
        Array([10., 20., 30., 10., 20., 30.], dtype=float32)
        >>>
        >>> out, _ = sweep(lambda p: p['a'] + p['b'],
        ...                {'a': jnp.arange(5.), 'b': jnp.ones(5)}, mode='zip', batch_size=2)
        >>> out
        Array([1., 2., 3., 4., 5.], dtype=float32)
    """
    if not isinstance(grid, dict) or not grid:
        raise ValueError('`grid` must be a non-empty dict {name: 1-D values}.')
    if mode not in ('product', 'zip'):
        raise ValueError(f"`mode` must be 'product' or 'zip', got {mode!r}.")
    if batch_size is not None and (
        isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1
    ):
        raise ValueError(f'`batch_size` must be a positive integer or None, got {batch_size!r}.')

    names = list(grid)
    values = {}
    for name in names:
        v = grid[name]
        v = v if isinstance(v, u.Quantity) else jnp.asarray(v)
        if v.ndim != 1:
            raise ValueError(f'Grid values of {name!r} must be 1-D, got shape {v.shape}.')
        if v.shape[0] == 0:
            raise ValueError(f'Grid values of {name!r} are empty.')
        values[name] = v

    lengths = [values[n].shape[0] for n in names]
    if mode == 'zip':
        if len(set(lengths)) != 1:
            raise ValueError(
                "In 'zip' mode all grid arrays must have the same length; got "
                + ', '.join(f'{n}: {k}' for n, k in zip(names, lengths)) + '.'
            )
        grid_shape: Tuple[int, ...] = (lengths[0],)
        table = dict(values)
    else:
        grid_shape = tuple(lengths)
        index = jnp.meshgrid(*[jnp.arange(k) for k in lengths], indexing='ij')
        table = {n: values[n][idx.reshape(-1)] for n, idx in zip(names, index)}

    if batch_size is None:
        outputs = jax.vmap(fn)(table)
    else:
        outputs = jax.lax.map(fn, table, batch_size=int(batch_size))
    return _reshape(outputs, grid_shape), table
