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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools.spike
from braintools.spike._tuning import gaussian_tuning


def _closed_form(x, centers, sigma):
    x, centers, sigma = (np.asarray(a, dtype=np.float64) for a in (x, centers, sigma))
    return np.exp(-0.5 * ((x - centers) / sigma) ** 2)


def test_peak_is_one_at_center():
    centers = jnp.linspace(0., 1., 5)
    r = gaussian_tuning(0.5, centers, 0.25)
    assert r.shape == (5,)
    assert float(r[2]) == 1.0
    np.testing.assert_allclose(r, _closed_form(0.5, centers, 0.25), rtol=1e-6)


def test_symmetric_about_center():
    centers = jnp.array([0.0])
    left = gaussian_tuning(-0.3, centers, 0.2)
    right = gaussian_tuning(0.3, centers, 0.2)
    np.testing.assert_array_equal(left, right)


def test_one_sigma_value():
    np.testing.assert_allclose(gaussian_tuning(1.0, 0.0, 1.0), np.exp(-0.5), rtol=1e-6)


def test_population_axis_via_explicit_newaxis():
    centers = jnp.linspace(-1., 1., 7)
    x = jnp.linspace(-2., 2., 12).reshape(3, 4)
    r = gaussian_tuning(x[..., None], centers, 0.3)
    assert r.shape == (3, 4, 7)
    np.testing.assert_allclose(r, _closed_form(np.asarray(x)[..., None], centers, 0.3), rtol=1e-5)


def test_plain_broadcasting_without_axis_insertion():
    # x with the same length as centers is evaluated elementwise.
    x = jnp.array([0.0, 0.5, 1.0])
    centers = jnp.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(gaussian_tuning(x, centers, 0.5), _closed_form(x, centers, 0.5), rtol=1e-6)


def test_per_center_sigma():
    centers = jnp.array([0.0, 1.0])
    sigma = jnp.array([0.1, 1.0])
    np.testing.assert_allclose(gaussian_tuning(0.5, centers, sigma), _closed_form(0.5, centers, sigma), rtol=1e-6)


def test_python_scalars():
    assert float(gaussian_tuning(1.0, 1.0, 0.5)) == 1.0
    assert float(gaussian_tuning(1, 1, 2)) == 1.0


def test_output_in_unit_interval_and_far_values_vanish():
    r = gaussian_tuning(jnp.array([1e3, -1e3])[:, None], jnp.linspace(0., 1., 5), 0.1)
    assert np.all(np.asarray(r) >= 0.) and np.all(np.asarray(r) <= 1.)
    np.testing.assert_array_equal(r, 0.)


def test_quantity_inputs_give_dimensionless_array():
    centers = jnp.linspace(0., 1., 5)
    r_unit = gaussian_tuning(0.3 * u.mV, centers * u.mV, 0.2 * u.mV)
    r_plain = gaussian_tuning(0.3, centers, 0.2)
    assert not isinstance(r_unit, u.Quantity)
    np.testing.assert_allclose(r_unit, r_plain, rtol=1e-6)


def test_quantity_inputs_with_compatible_scales():
    centers = jnp.linspace(0., 1., 5) * u.mV
    r = gaussian_tuning(500. * u.uvolt, centers, 0.25 * u.mV)
    np.testing.assert_allclose(r, _closed_form(0.5, np.linspace(0., 1., 5), 0.25), rtol=1e-5)


def test_mismatched_dimensions_raise():
    centers = jnp.linspace(0., 1., 5)
    with pytest.raises(u.UnitMismatchError):
        gaussian_tuning(0.5 * u.mV, centers * u.ms, 0.25 * u.mV)


@pytest.mark.parametrize('sigma', [0.25, 0.25 * u.ms])
def test_non_dimensionless_ratio_raises(sigma):
    centers = jnp.linspace(0., 1., 5) * u.mV
    with pytest.raises(TypeError, match='dimensionless'):
        gaussian_tuning(0.5 * u.mV, centers, sigma)


def test_zero_sigma_gives_nan_at_center_zero_elsewhere():
    r = np.asarray(gaussian_tuning(0.5, jnp.linspace(0., 1., 5), 0.))
    assert np.isnan(r[2])
    np.testing.assert_array_equal(np.delete(r, 2), 0.)


def test_jit_vmap_and_grad():
    centers = jnp.linspace(0., 1., 5)
    f = jax.jit(lambda x: gaussian_tuning(x, centers, 0.2))
    np.testing.assert_allclose(f(0.4), gaussian_tuning(0.4, centers, 0.2), rtol=1e-6)
    batched = jax.vmap(f)(jnp.array([0.1, 0.6]))
    assert batched.shape == (2, 5)
    # d/dx exp(-(x-c)^2 / 2s^2) = -(x-c)/s^2 * r ; zero at the center.
    g = jax.grad(lambda x: gaussian_tuning(x, 0.5, 0.2))(0.5)
    assert float(g) == 0.0
    g = jax.grad(lambda x: gaussian_tuning(x, 0.0, 1.0))(1.0)
    np.testing.assert_allclose(g, -np.exp(-0.5), rtol=1e-6)


def test_not_exported_from_spike_package():
    assert not hasattr(braintools.spike, 'gaussian_tuning')
    assert 'gaussian_tuning' not in braintools.spike.__all__
