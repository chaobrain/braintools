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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.fit._objective import Objective, Term


def mse(p, t):
    return jnp.mean((p - t) ** 2)


def mae(p, t):
    return jnp.mean(jnp.abs(p - t))


PRED = {'v': jnp.array([0., 2.]), 'r': jnp.array([1., 1.])}
TARGET = {'v': jnp.array([0., 0.]), 'r': jnp.array([0., 2.])}


def test_term_value_weight_and_name():
    t = Term(mse)
    assert t.name == 'mse' and t.weight == 1.0 and t.key is None
    np.testing.assert_allclose(float(t(jnp.array([1., 2.]), jnp.array([1., 4.]))), 2.)
    t2 = 3. * t
    assert isinstance(t2, Term) and t2.weight == 3. and t.weight == 1.
    np.testing.assert_allclose(float(t2(jnp.array([1., 2.]), jnp.array([1., 4.]))), 6.)
    assert (t * 2.).weight == 2.
    assert Term(mse, key='v').name == 'v'
    assert Term(mse, key='v', name='custom').name == 'custom'
    assert Term(functools.partial(mse)).name == 'partial'
    assert 'mse' in repr(t)


def test_term_key_selects_dict_entries():
    np.testing.assert_allclose(float(Term(mse, key='v')(PRED, TARGET)), 2.)
    np.testing.assert_allclose(float(Term(mae, key='r')(PRED, TARGET)), 1.)
    # integer keys index tuple predictions
    np.testing.assert_allclose(float(Term(mse, key=1)((jnp.zeros(2), jnp.ones(2)), (0., 0.))), 1.)


def test_objective_composition_and_breakdown():
    obj = 0.5 * Term(mse, key='v', name='mse_v') + Term(mae, key='r')
    assert isinstance(obj, Objective) and len(obj) == 2
    np.testing.assert_allclose(float(obj(PRED, TARGET)), 0.5 * 2. + 1.)
    parts = obj.breakdown(PRED, TARGET)
    assert list(parts) == ['mse_v', 'r']
    np.testing.assert_allclose(float(parts['mse_v']), 1.)
    np.testing.assert_allclose(sum(float(v) for v in parts.values()), float(obj(PRED, TARGET)))
    # objective + term, term + objective, objective + objective, scalar * objective
    obj3 = obj + Term(mse, key='r')
    assert len(obj3) == 3
    assert len(Term(mse, key='r') + obj) == 3
    assert len(obj + obj) == 4
    scaled = 2. * obj
    np.testing.assert_allclose(float(scaled(PRED, TARGET)), 2. * float(obj(PRED, TARGET)))
    np.testing.assert_allclose(float((obj * 2.)(PRED, TARGET)), float(scaled(PRED, TARGET)))
    assert [t.name for t in obj] == ['mse_v', 'r']
    assert 'mse_v' in repr(obj)


def test_duplicate_names_get_suffixes():
    obj = Term(mse) + Term(mse) + Term(mse)
    assert obj.names == ('mse', 'mse_1', 'mse_2')
    assert list(obj.breakdown(jnp.ones(2), jnp.zeros(2))) == ['mse', 'mse_1', 'mse_2']


def test_sum_builtin_and_callables():
    obj = sum([Term(mse), Term(mae)])
    assert isinstance(obj, Objective) and len(obj) == 2
    assert sum([obj]) is obj
    obj2 = Objective(mse, Term(mae), obj)
    assert len(obj2) == 4
    np.testing.assert_allclose(float(Objective(mse)(jnp.ones(2), jnp.zeros(2))), 1.)


def test_invalid_operands():
    t = Term(mse)
    with pytest.raises(TypeError):
        t * t
    with pytest.raises(TypeError):
        t + 1.
    with pytest.raises(TypeError):
        1. + t
    with pytest.raises(TypeError):
        Objective(t) * jnp.ones(2)
    with pytest.raises(TypeError):
        Objective(t) + 'x'
    with pytest.raises(TypeError):
        'x' + Objective(t)
    with pytest.raises(TypeError, match='callable'):
        Term(1.)
    with pytest.raises(TypeError, match='scalar'):
        Term(mse, weight=jnp.ones(2))
    with pytest.raises(TypeError, match='scalar'):
        Term(mse, weight=True)
    with pytest.raises(TypeError, match='scalar'):
        Term(mse, weight='1')
    with pytest.raises(ValueError, match='at least one'):
        Objective()
    with pytest.raises(TypeError, match='Term'):
        Objective(3)


def test_jax_array_weight():
    t = jnp.asarray(0.25) * Term(mse)
    np.testing.assert_allclose(float(t(jnp.ones(2) * 2., jnp.zeros(2))), 1.)


def test_jit_vmap_grad():
    obj = 0.5 * Term(mse, key='v') + Term(mae, key='r')
    f = jax.jit(obj)
    np.testing.assert_allclose(float(f(PRED, TARGET)), float(obj(PRED, TARGET)))
    batched = jax.tree.map(lambda x: jnp.stack([x, x + 1.]), PRED)
    out = jax.vmap(obj, in_axes=(0, None))(batched, TARGET)
    assert out.shape == (2,)
    np.testing.assert_allclose(float(out[0]), float(obj(PRED, TARGET)))
    g = jax.grad(lambda v: obj({'v': v, 'r': PRED['r']}, TARGET))(PRED['v'])
    np.testing.assert_allclose(np.asarray(g), 0.5 * 2. * np.asarray(PRED['v'] - TARGET['v']) / 2.)
    br = jax.jit(obj.breakdown)(PRED, TARGET)
    np.testing.assert_allclose(float(br['r']), 1.)


def test_unique_names_never_collide_and_breakdown_sums_to_total():
    obj = Term(mse, name='a') + Term(mse, weight=2., name='a') + Term(mse, weight=3., name='a_1')
    assert len(set(obj.names)) == 3
    parts = obj.breakdown(jnp.ones(2), jnp.zeros(2))
    assert len(parts) == 3
    np.testing.assert_allclose(sum(float(v) for v in parts.values()), float(obj(jnp.ones(2), jnp.zeros(2))))
    obj2 = Term(mse, name='a_1') + Term(mse, name='a') + Term(mse, name='a') + Term(mse, name='a_1')
    assert len(set(obj2.names)) == 4
    assert obj2.names[0] == 'a_1' and obj2.names[1] == 'a'
