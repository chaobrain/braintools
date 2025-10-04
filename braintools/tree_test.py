import jax
import jax.numpy as jnp
import numpy as np

from braintools.tree import (
    scale,
    mul,
    shift,
    add,
    sub,
    dot,
    sum as tree_sum,
    squared_norm,
    concat,
    split,
    idx as tree_idx,
    expand,
    take,
    as_numpy,
)


def _assert_trees_allclose(a, b, rtol=1e-6, atol=1e-6):
    la, ta = jax.tree_util.tree_flatten(a)
    lb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb, "PyTree structures differ"
    for xa, xb in zip(la, lb):
        np.testing.assert_allclose(np.asarray(xa), np.asarray(xb), rtol=rtol, atol=atol)


def _make_tree_pair():
    # Simple matching-structure PyTrees for elementwise ops
    t1 = {
        'a': jnp.array([1.0, 2.0, 3.0]),
        'b': {
            'c': jnp.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        }
    }
    t2 = {
        'a': jnp.array([10.0, 20.0, 30.0]),
        'b': {
            'c': jnp.array([[2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0]])
        }
    }
    return t1, t2


def test_scale_and_mul_scalar():
    t1, _ = _make_tree_pair()
    s = 2.5
    out_scale = scale(t1, s)
    out_mul = mul(t1, s)

    expected = {
        'a': t1['a'] * s,
        'b': {'c': t1['b']['c'] * s}
    }
    _assert_trees_allclose(out_scale, expected)
    _assert_trees_allclose(out_mul, expected)


def test_shift_and_add_scalar():
    t1, _ = _make_tree_pair()
    s = 3.0
    out_shift = shift(t1, s)
    out_add = add(t1, s)

    expected = {
        'a': t1['a'] + s,
        'b': {'c': t1['b']['c'] + s}
    }
    _assert_trees_allclose(out_shift, expected)
    _assert_trees_allclose(out_add, expected)


def test_mul_and_add_pytree_and_sub():
    t1, t2 = _make_tree_pair()

    out_mul = mul(t1, t2)
    expected_mul = {
        'a': t1['a'] * t2['a'],
        'b': {'c': t1['b']['c'] * t2['b']['c']}
    }
    _assert_trees_allclose(out_mul, expected_mul)

    out_add = add(t1, t2)
    expected_add = {
        'a': t1['a'] + t2['a'],
        'b': {'c': t1['b']['c'] + t2['b']['c']}
    }
    _assert_trees_allclose(out_add, expected_add)

    out_sub = sub(t1, t2)
    expected_sub = {
        'a': t1['a'] - t2['a'],
        'b': {'c': t1['b']['c'] - t2['b']['c']}
    }
    _assert_trees_allclose(out_sub, expected_sub)


def test_dot_sum_squared_norm():
    t1, t2 = _make_tree_pair()

    # Manual expectations
    expected_sum = jnp.sum(t1['a']) + jnp.sum(t1['b']['c'])
    expected_dot = jnp.sum(t1['a'] * t2['a']) + jnp.sum(t1['b']['c'] * t2['b']['c'])
    expected_sqnorm = jnp.sum(t1['a'] ** 2) + jnp.sum(t1['b']['c'] ** 2)

    out_sum = tree_sum(t1)
    out_dot = dot(t1, t2)
    out_sq = squared_norm(t1)

    np.testing.assert_allclose(np.asarray(out_sum), np.asarray(expected_sum))
    np.testing.assert_allclose(np.asarray(out_dot), np.asarray(expected_dot))
    np.testing.assert_allclose(np.asarray(out_sq), np.asarray(expected_sqnorm))

    # Consistency: squared_norm == dot(x, x)
    np.testing.assert_allclose(np.asarray(out_sq), np.asarray(dot(t1, t1)))


def test_concat_axis0():
    t1, t2 = _make_tree_pair()

    out0 = concat([t1, t2], axis=0)
    expected0 = {
        'a': jnp.concatenate([t1['a'], t2['a']], axis=0),
        'b': {'c': jnp.concatenate([t1['b']['c'], t2['b']['c']], axis=0)}
    }
    _assert_trees_allclose(out0, expected0)


def test_concat_axis1_on_2d_tree():
    t1 = {'m': jnp.arange(6.0).reshape(2, 3)}
    t2 = {'m': (jnp.arange(6.0).reshape(2, 3) + 10.0)}
    out = concat([t1, t2], axis=1)
    expected = {'m': jnp.concatenate([t1['m'], t2['m']], axis=1)}
    _assert_trees_allclose(out, expected)


def test_split_with_remainder():
    tree = {
        'a': jnp.arange(7.0),
        'b': {'c': jnp.arange(14.0).reshape(7, 2)},
    }
    parts = split(tree, (2, 3))
    assert isinstance(parts, tuple) and len(parts) == 3

    p0, p1, p2 = parts
    _assert_trees_allclose(p0, {'a': tree['a'][:2], 'b': {'c': tree['b']['c'][:2]}})
    _assert_trees_allclose(p1, {'a': tree['a'][2:5], 'b': {'c': tree['b']['c'][2:5]}})
    _assert_trees_allclose(p2, {'a': tree['a'][5:], 'b': {'c': tree['b']['c'][5:]}})


def test_idx_int_and_slice():
    tree = {
        'a': jnp.array([1.0, 2.0, 3.0, 4.0]),
        'b': {'c': jnp.arange(12.0).reshape(4, 3)},
    }

    out_int = tree_idx(tree, 2)
    expected_int = {'a': tree['a'][2], 'b': {'c': tree['b']['c'][2]}}
    # Wrap scalars to arrays for uniform comparison
    out_int_n = jax.tree_util.tree_map(lambda x: np.asarray(x), out_int)
    exp_int_n = jax.tree_util.tree_map(lambda x: np.asarray(x), expected_int)
    _assert_trees_allclose(out_int_n, exp_int_n)

    out_sl = tree_idx(tree, slice(1, 3))
    expected_sl = {'a': tree['a'][1:3], 'b': {'c': tree['b']['c'][1:3]}}
    _assert_trees_allclose(out_sl, expected_sl)


def test_expand_shapes():
    tree = {
        'a': jnp.array([1.0, 2.0, 3.0]),  # (3,)
        'b': {'c': jnp.ones((2, 3))},  # (2, 3)
    }
    out = expand(tree, axis=0)
    assert out['a'].shape == (1, 3)
    assert out['b']['c'].shape == (1, 2, 3)


def test_take_slice_and_indices():
    tree = {
        'a': jnp.array([10.0, 20.0, 30.0, 40.0]),
        'b': {'c': jnp.arange(24.0).reshape(4, 6)},
    }

    # Slice along axis 0
    out_slice = take(tree, slice(1, 3), axis=0)
    exp_slice = {'a': tree['a'][1:3], 'b': {'c': tree['b']['c'][1:3]}}
    _assert_trees_allclose(out_slice, exp_slice)

    # Indices along axis 0
    idxs = jnp.array([0, 2])
    out_idx0 = take(tree, idxs, axis=0)
    exp_idx0 = {'a': jnp.take(tree['a'], idxs, axis=0), 'b': {'c': jnp.take(tree['b']['c'], idxs, axis=0)}}
    _assert_trees_allclose(out_idx0, exp_idx0)


def test_as_numpy_types():
    t1, _ = _make_tree_pair()
    out = as_numpy(t1)
    leaves, _ = jax.tree_util.tree_flatten(out)
    assert all(isinstance(x, np.ndarray) for x in leaves)
