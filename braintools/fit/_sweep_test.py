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

import itertools

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.fit._sweep import sweep


def f(p):
    return p['a'] * 10. + p['b']


A = jnp.array([1., 2., 3.])
B = jnp.array([0.1, 0.2])


def test_product_matches_itertools():
    out, table = sweep(f, {'a': A, 'b': B})
    assert out.shape == (3, 2)
    for (i, a), (j, b) in itertools.product(enumerate(np.asarray(A)), enumerate(np.asarray(B))):
        np.testing.assert_allclose(float(out[i, j]), a * 10. + b, rtol=1e-6)
    ref = list(itertools.product(np.asarray(A), np.asarray(B)))
    np.testing.assert_allclose(np.asarray(table['a']), [r[0] for r in ref])
    np.testing.assert_allclose(np.asarray(table['b']), [r[1] for r in ref])
    np.testing.assert_allclose(np.asarray(out).reshape(-1), np.asarray(f(table)), rtol=1e-6)


def test_zip_mode():
    out, table = sweep(f, {'a': A, 'b': jnp.array([0.1, 0.2, 0.3])}, mode='zip')
    np.testing.assert_allclose(np.asarray(out), [10.1, 20.2, 30.3], rtol=1e-6)
    assert table['a'].shape == (3,)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 6, 7, 100])
def test_batch_size_not_dividing_grid(batch_size):
    ref, _ = sweep(f, {'a': A, 'b': B})
    out, _ = sweep(f, {'a': A, 'b': B}, batch_size=batch_size)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-6)


def test_pytree_outputs_and_trailing_shapes():
    def g(p):
        return {'vec': p['a'] * jnp.arange(4.), 'sum': p['a'] + p['b']}

    out, _ = sweep(g, {'a': A, 'b': B}, batch_size=4)
    assert out['vec'].shape == (3, 2, 4) and out['sum'].shape == (3, 2)
    np.testing.assert_allclose(np.asarray(out['vec'][2, 1]), 3. * np.arange(4.))


def test_quantity_grid_and_outputs():
    def g(p):
        return p['v'] * 2.

    out, table = sweep(g, {'v': jnp.array([1., 2.]) * u.mV, 'k': jnp.array([1., 2., 3.])})
    assert isinstance(out, u.Quantity) and out.shape == (2, 3)
    np.testing.assert_allclose(out.to_decimal(u.mV)[1], 4.)
    assert table['v'].unit == u.mV


def test_three_dimensional_grid():
    out, table = sweep(lambda p: p['x'] + p['y'] + p['z'],
                       {'x': jnp.arange(2.), 'y': jnp.arange(3.), 'z': jnp.arange(4.)}, batch_size=5)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(float(out[1, 2, 3]), 6.)
    assert table['z'].shape == (24,)


def test_validation():
    with pytest.raises(ValueError, match='same length'):
        sweep(f, {'a': A, 'b': B}, mode='zip')
    with pytest.raises(ValueError, match='mode'):
        sweep(f, {'a': A, 'b': B}, mode='outer')
    with pytest.raises(ValueError, match='non-empty'):
        sweep(f, {})
    with pytest.raises(ValueError, match='1-D'):
        sweep(f, {'a': jnp.ones((2, 2)), 'b': B})
    with pytest.raises(ValueError, match='1-D'):
        sweep(f, {'a': 1., 'b': B})
    with pytest.raises(ValueError, match='empty'):
        sweep(f, {'a': jnp.zeros(0), 'b': B})
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match='batch_size'):
            sweep(f, {'a': A, 'b': B}, batch_size=bad)


def test_jit_and_grad():
    @jax.jit
    def total(scale):
        out, _ = sweep(lambda p: scale * p['a'] ** 2, {'a': A}, batch_size=2)
        return jnp.sum(out)

    np.testing.assert_allclose(float(total(2.)), 2. * float(jnp.sum(A ** 2)), rtol=1e-6)
    np.testing.assert_allclose(float(jax.grad(total)(2.)), float(jnp.sum(A ** 2)), rtol=1e-6)


def test_vmap_over_closure():
    def run(scale):
        out, _ = sweep(lambda p: scale * p['a'], {'a': A}, batch_size=2)
        return out

    res = jax.vmap(run)(jnp.array([1., 2.]))
    np.testing.assert_allclose(np.asarray(res), np.stack([A, 2. * A]))
