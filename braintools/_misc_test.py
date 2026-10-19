# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import unittest

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintools._misc import randn_like, set_module_as, tree_map


class TestRandnLike(unittest.TestCase):
    """``randn_like`` must mirror ``y``'s shape *and* float dtype.

    Regression test: it was implemented as ``brainstate.random.randn(*shape)``,
    which silently returned float32 for every input -- so with ``jax_enable_x64``
    on, a float64 SDE state received float32 Brownian increments, and float16 /
    bfloat16 states were promoted to float32.
    """

    def test_float_dtypes_are_preserved(self):
        # float16/bfloat16 are not affected by the x64 flag, so these assert the
        # dtype relationship directly and fail on the old float32-always behaviour.
        for dtype in (jnp.float16, jnp.float32, jnp.bfloat16):
            out = randn_like(jnp.zeros((2, 3), dtype))
            self.assertEqual(np.asarray(out).dtype, dtype)

    def test_dtype_matches_input_dtype(self):
        # Note: float64 is omitted here because without ``jax_enable_x64`` it is
        # silently truncated to float32 on construction (and jax warns about it);
        # the relationship still holds whenever x64 is on.
        for dtype in (jnp.float16, jnp.float32, jnp.bfloat16):
            y = jnp.zeros(4, dtype)
            self.assertEqual(np.asarray(randn_like(y)).dtype, u.math.get_dtype(y))

    def test_shape_is_preserved(self):
        for shape in ((3,), (2, 4), (2, 3, 5), ()):
            self.assertEqual(np.asarray(randn_like(jnp.zeros(shape))).shape, shape)

    def test_non_float_inputs_do_not_raise(self):
        # ``random.normal`` rejects integer dtypes, so they must fall back.
        for y in (jnp.zeros(3, jnp.int32), jnp.zeros(3, jnp.bool_)):
            out = np.asarray(randn_like(y))
            self.assertEqual(out.shape, (3,))
            self.assertTrue(np.issubdtype(out.dtype, np.floating))

    def test_quantity_uses_mantissa_dtype(self):
        y = jnp.zeros(3, jnp.float32) * u.mV
        self.assertEqual(np.asarray(randn_like(y)).dtype, jnp.float32)

    def test_samples_differ_between_calls(self):
        a = np.asarray(randn_like(jnp.zeros(64)))
        b = np.asarray(randn_like(jnp.zeros(64)))
        self.assertFalse(np.allclose(a, b))


class TestTreeMap(unittest.TestCase):

    def test_matches_jax_tree_map_on_plain_trees(self):
        tree = {'a': jnp.arange(3.0), 'b': (jnp.ones((2, 2)), [jnp.arange(4.0)])}
        for f in (lambda x: x * 2.0, lambda x: x + 1.0, lambda x: jnp.sum(x)):
            got = jax.tree_util.tree_leaves(tree_map(f, tree))
            ref = jax.tree_util.tree_leaves(jax.tree.map(f, tree))
            for a, b in zip(got, ref):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_matches_jax_tree_map_with_rest(self):
        t1 = {'a': jnp.arange(3.0), 'b': (jnp.ones((2, 2)), [jnp.arange(4.0)])}
        t2 = {'a': jnp.ones(3) * 3, 'b': (jnp.ones((2, 2)) * 2, [jnp.ones(4) * 5])}
        got = jax.tree_util.tree_leaves(tree_map(lambda x, y: x * y, t1, t2))
        ref = jax.tree_util.tree_leaves(jax.tree.map(lambda x, y: x * y, t1, t2))
        for a, b in zip(got, ref):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b))

    def test_quantity_is_a_single_leaf(self):
        tree = {'v': jnp.array([1.0, 2.0, 3.0]) * u.mV, 'w': jnp.arange(2.0)}
        out = tree_map(lambda x: x * 2.0, tree)
        # the Quantity stays whole (unit preserved) rather than being split apart
        self.assertEqual(u.get_unit(out['v']), u.mV)
        np.testing.assert_allclose(np.asarray(u.get_mantissa(out['v'])), [2.0, 4.0, 6.0])
        np.testing.assert_allclose(np.asarray(out['w']), [0.0, 2.0])


class TestSetModuleAs(unittest.TestCase):

    def test_sets_module_on_function_and_class(self):
        def fn(x):
            return x

        class Klass:
            pass

        set_module_as('some.mod')(fn)
        set_module_as('some.mod')(Klass)
        self.assertEqual(fn.__module__, 'some.mod')
        self.assertEqual(Klass.__module__, 'some.mod')