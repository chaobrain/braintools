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

import copy

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.quad import _registry
from braintools.quad._odeint import odeint
from braintools.quad._ode_integrator import ode_euler_step


@pytest.fixture
def x64():
    prev = jax.config.read('jax_enable_x64')
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', prev)


def decay(y, t):
    return -y


def oscillator(s, t, w):
    return {'x': s['v'], 'v': -w ** 2 * s['x']}


class TestExactSolutions:
    def test_linear_decay(self):
        ts, ys = odeint(decay, jnp.array(1.0), 0.0, 2.0, 0.05)
        assert ts.shape == ys.shape == (41,)
        np.testing.assert_allclose(ts, np.linspace(0, 2, 41), atol=1e-6)
        np.testing.assert_allclose(ys, np.exp(-np.asarray(ts)), rtol=1e-6)
        assert float(ys[0]) == 1.0

    def test_harmonic_oscillator_pytree(self, x64):
        w = 2.0
        ts, ys = odeint(oscillator, {'x': jnp.array(1.0), 'v': jnp.array(0.0)}, 0.0, 3.0, 0.01, w)
        assert set(ys) == {'x', 'v'} and ys['x'].shape == (301,)
        t = np.asarray(ts)
        np.testing.assert_allclose(ys['x'], np.cos(w * t), atol=1e-8)
        np.testing.assert_allclose(ys['v'], -w * np.sin(w * t), atol=1e-8)

    def test_vector_state_and_kwargs(self):
        def f(y, t, a, b=0.0):
            return -a * y + b

        y0 = jnp.array([1.0, 2.0, 3.0])
        ts, ys = odeint(f, y0, 0.0, 1.0, 0.01, 2.0, b=1.0)
        assert ys.shape == (101, 3)
        exact = 0.5 + (np.asarray(y0) - 0.5) * np.exp(-2.0)
        np.testing.assert_allclose(ys[-1], exact, rtol=1e-5)

    def test_time_dependent_rhs(self, x64):
        ts, ys = odeint(lambda y, t: y * jnp.cos(t), jnp.array(1.0), 0.0, 2.0, 0.01)
        np.testing.assert_allclose(ys, np.exp(np.sin(np.asarray(ts))), rtol=1e-9)

    def test_backwards(self, x64):
        ts, ys = odeint(decay, jnp.array(1.0), 1.0, 0.0, -0.01)
        assert ts.shape == (101,)
        np.testing.assert_allclose(ts[-1], 0.0, atol=1e-12)
        np.testing.assert_allclose(ys[-1], np.e, rtol=1e-9)


# (method, expected order, step counts)
_ORDERS = [
    ('euler', 1, (32, 64)),
    ('rk2', 2, (32, 64)),
    ('midpoint', 2, (32, 64)),
    ('ralston2', 2, (32, 64)),
    ('heun', 3, (32, 64)),
    ('rk3', 3, (32, 64)),
    ('ralston3', 3, (32, 64)),
    ('ssprk33', 3, (32, 64)),
    ('rk23', 3, (32, 64)),
    ('rk4', 4, (32, 64)),
    ('rk4_38', 4, (32, 64)),
    ('rk45', 5, (16, 32)),
    ('rkf45', 5, (32, 64)),
    ('dopri5', 5, (16, 32)),
    ('dopri8', 8, (8, 16)),
]


@pytest.mark.parametrize('method,order,ns', _ORDERS, ids=[m for m, _, _ in _ORDERS])
def test_observed_order_of_convergence(method, order, ns, x64):
    T = 2.0
    exact = np.exp(np.sin(T))
    errs = []
    for n in ns:
        _, ys = odeint(lambda y, t: y * jnp.cos(t), jnp.array(1.0), 0.0, T, T / n, method=method)
        errs.append(abs(float(ys[-1]) - exact))
    observed = np.log2(errs[0] / errs[1])
    assert abs(observed - order) < 0.3, (method, observed, errs)


class TestSaving:
    def test_save_every_dividing(self):
        ts1, ys1 = odeint(decay, jnp.array(1.0), 0.0, 1.0, 0.1)
        ts5, ys5 = odeint(decay, jnp.array(1.0), 0.0, 1.0, 0.1, save_every=5)
        assert ts5.shape == ys5.shape == (3,)
        np.testing.assert_allclose(ts5, ts1[::5], atol=1e-7)
        np.testing.assert_allclose(ys5, ys1[::5], rtol=1e-6)

    def test_save_every_not_dividing(self):
        # n = 10, save_every = 3 -> 3 saved steps, stop at t = 0.9 (step 10 is not run)
        ts, ys = odeint(decay, jnp.array(1.0), 0.0, 1.0, 0.1, save_every=3)
        assert ts.shape == ys.shape == (4,)
        np.testing.assert_allclose(ts, [0.0, 0.3, 0.6, 0.9], atol=1e-6)
        ts9, ys9 = odeint(decay, jnp.array(1.0), 0.0, 0.9, 0.1)
        np.testing.assert_allclose(ys[-1], ys9[-1], rtol=1e-6)
        np.testing.assert_allclose(ys, ys9[::3], rtol=1e-6)

    def test_t1_equals_t0(self):
        ts, ys = odeint(decay, jnp.array([1.0, 2.0]), 0.5, 0.5, 0.1)
        assert ts.shape == (1,) and ys.shape == (1, 2)
        np.testing.assert_allclose(ts, [0.5])
        np.testing.assert_allclose(ys[0], [1.0, 2.0])

    def test_fewer_steps_than_save_every(self):
        ts, ys = odeint(oscillator, {'x': jnp.array(1.0), 'v': jnp.array(0.0)}, 0.0, 0.2, 0.1, 1.0,
                        save_every=3)
        assert ts.shape == (1,) and ys['x'].shape == (1,) and ys['v'].shape == (1,)

    def test_n_steps_zero(self):
        ts, ys = odeint(decay, jnp.array(1.0), 0.0, dt=0.1, n_steps=0)
        assert ts.shape == ys.shape == (1,)

    def test_rounding_of_step_count(self):
        # (1.04 - 0) / 0.1 rounds to 10 steps
        ts, _ = odeint(decay, jnp.array(1.0), 0.0, 1.04, 0.1)
        assert ts.shape == (11,)
        np.testing.assert_allclose(ts[-1], 1.0, atol=1e-6)


class TestUnits:
    def test_quantity_state_and_time(self):
        tau, v_rest, R, I = 10. * u.ms, -65. * u.mV, 10. * u.Mohm, 0.5 * u.nA

        def neuron(s, t, I):
            return {'V': (v_rest - s['V'] + R * I) / tau}

        ts, ys = odeint(neuron, {'V': -70. * u.mV}, 0. * u.ms, 20. * u.ms, 0.05 * u.ms, I)
        assert u.get_unit(ts) == u.ms and u.get_unit(ys['V']) == u.mV
        assert ys['V'].shape == (401,)
        v_inf = v_rest + R * I
        exact = v_inf + (-70. * u.mV - v_inf) * u.math.exp(-ts / tau)
        np.testing.assert_allclose(ys['V'].to_decimal(u.mV), exact.to_decimal(u.mV), rtol=1e-5)

    def test_mixed_time_units(self):
        ts, ys = odeint(lambda y, t: -y / (1. * u.ms), jnp.array(1.0), 0. * u.ms, 1. * u.ms, 1e-5 * u.second)
        assert ts.shape == (101,)
        np.testing.assert_allclose(ys[-1], np.exp(-1.0), rtol=1e-5)

    def test_incompatible_units(self):
        with pytest.raises(u.UnitMismatchError):
            odeint(decay, jnp.array(1.0), 0. * u.ms, 1. * u.ms, 0.1 * u.mV)

    def test_expeuler_with_units(self):
        f = lambda V, t: (-65. * u.mV - V) / (10. * u.ms)
        ts, ys = odeint(f, -70. * u.mV, 0. * u.ms, 5. * u.ms, 0.1 * u.ms, method='expeuler')
        exact = -65. - 5. * np.exp(-0.5)
        np.testing.assert_allclose(ys[-1].to_decimal(u.mV), exact, rtol=1e-5)


class TestValidation:
    def test_missing_dt(self):
        with pytest.raises(ValueError, match='dt'):
            odeint(decay, 1.0, 0.0, 1.0)

    def test_missing_t1_and_n_steps(self):
        with pytest.raises(ValueError, match='n_steps'):
            odeint(decay, 1.0, 0.0, dt=0.1)

    def test_both_t1_and_n_steps(self):
        with pytest.raises(ValueError, match='not both'):
            odeint(decay, 1.0, 0.0, 1.0, 0.1, n_steps=10)

    @pytest.mark.parametrize('n', [-1, 2.0, True])
    def test_bad_n_steps(self, n):
        with pytest.raises(ValueError, match='n_steps'):
            odeint(decay, 1.0, 0.0, dt=0.1, n_steps=n)

    @pytest.mark.parametrize('k', [0, -2, 1.5, True])
    def test_bad_save_every(self, k):
        with pytest.raises(ValueError, match='save_every'):
            odeint(decay, 1.0, 0.0, 1.0, 0.1, save_every=k)

    @pytest.mark.parametrize('dt', [0.0, jnp.array(0.0)])
    def test_zero_dt(self, dt):
        with pytest.raises(ValueError, match='dt'):
            odeint(decay, 1.0, 0.0, 1.0, dt)

    def test_wrong_direction(self):
        with pytest.raises(ValueError, match='negative'):
            odeint(decay, 1.0, 1.0, 0.0, 0.1)

    def test_unknown_method(self):
        with pytest.raises(KeyError, match='rk4'):
            odeint(decay, 1.0, 0.0, 1.0, 0.1, method='rk5')

    def test_traced_t1_needs_n_steps(self):
        with pytest.raises(ValueError, match='n_steps'):
            jax.jit(lambda t1: odeint(decay, 1.0, 0.0, t1, 0.1))(1.0)


class TestMethods:
    def test_callable_method(self):
        _, ys_a = odeint(decay, jnp.array(1.0), 0.0, 1.0, 0.1, method=ode_euler_step)
        _, ys_b = odeint(decay, jnp.array(1.0), 0.0, 1.0, 0.1, method='euler')
        np.testing.assert_allclose(ys_a, ys_b)
        np.testing.assert_allclose(ys_a[-1], 0.9 ** 10, rtol=1e-6)

    def test_registered_method(self):
        saved = copy.deepcopy({k: dict(v) for k, v in _registry._REGISTRY.items()})
        try:
            def backward_euler_linear(f, y, t, *args, **kwargs):
                # exact implicit Euler for f(y) = -y
                return y / (1 + brainstate.environ.get_dt())

            _registry.register_method('implicit_decay', backward_euler_linear)
            _, ys = odeint(decay, jnp.array(1.0), 0.0, 1.0, 0.1, method='implicit_decay')
            np.testing.assert_allclose(ys[-1], 1.1 ** -10, rtol=1e-6)
        finally:
            _registry._REGISTRY.clear()
            _registry._REGISTRY.update(saved)


class TestTransforms:
    def test_jit_with_n_steps(self):
        @jax.jit
        def run(y0, t0, dt):
            return odeint(decay, y0, t0, None, dt, n_steps=20, save_every=4)

        ts, ys = run(jnp.array(2.0), 1.0, 0.05)
        ts_ref, ys_ref = odeint(decay, jnp.array(2.0), 1.0, 2.0, 0.05, save_every=4)
        assert ts.shape == (6,)
        np.testing.assert_allclose(ts, ts_ref, atol=1e-6)
        np.testing.assert_allclose(ys, ys_ref, rtol=1e-6)

    def test_jit_with_concrete_t1(self):
        run = jax.jit(lambda y0: odeint(decay, y0, 0.0, 1.0, 0.1)[1])
        np.testing.assert_allclose(run(jnp.array(1.0))[-1], np.exp(-1.0), rtol=1e-5)

    def test_vmap_over_y0_and_args(self):
        f = lambda y, t, a: -a * y
        y0 = jnp.array([1.0, 2.0, 3.0])
        a = jnp.array([0.5, 1.0, 2.0])
        ys = jax.vmap(lambda y, k: odeint(f, y, 0.0, 1.0, 0.01, k)[1])(y0, a)
        assert ys.shape == (3, 101)
        np.testing.assert_allclose(ys[:, -1], np.asarray(y0) * np.exp(-np.asarray(a)), rtol=1e-5)

    def test_grad_matches_analytic(self, x64):
        def loss(k):
            return odeint(lambda y, t, k: -k * y, jnp.array(1.0), 0.0, 2.0, 0.01, k)[1][-1]

        k = 0.7
        g = jax.grad(loss)(k)
        np.testing.assert_allclose(g, -2.0 * np.exp(-2.0 * k), rtol=1e-8)

    def test_grad_wrt_y0_and_dt(self, x64):
        def loss(y0, dt):
            return odeint(decay, y0, 0.0, None, dt, n_steps=50)[1][-1]

        gy, gdt = jax.grad(loss, argnums=(0, 1))(1.5, 0.02)
        # y_N = y0 * e^{-N dt}: dy/dy0 = e^{-1}, dy/ddt = -N y0 e^{-1}
        np.testing.assert_allclose(gy, np.exp(-1.0), rtol=1e-8)
        np.testing.assert_allclose(gdt, -50 * 1.5 * np.exp(-1.0), rtol=1e-6)

    @pytest.mark.parametrize('save_every', [1, 3])
    def test_checkpoint_same_values_and_gradient(self, save_every, x64):
        w0 = {'x': jnp.array(1.0), 'v': jnp.array(0.5)}

        def loss(w, checkpoint):
            _, ys = odeint(oscillator, w0, 0.0, 1.0, 0.01, w, save_every=save_every, checkpoint=checkpoint)
            return jnp.sum(ys['x'] ** 2) + jnp.sum(ys['v'])

        v0, g0 = jax.value_and_grad(loss)(1.3, False)
        v1, g1 = jax.value_and_grad(loss)(1.3, True)
        np.testing.assert_allclose(v1, v0, rtol=1e-12)
        np.testing.assert_allclose(g1, g0, rtol=1e-10)
        assert np.isfinite(g0) and g0 != 0
