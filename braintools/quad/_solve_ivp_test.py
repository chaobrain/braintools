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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.quad._solve_ivp import ODESolution, solve_ivp


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


def lotka_volterra(y, t):
    return jnp.stack([1.5 * y[0] - y[0] * y[1], -3.0 * y[1] + y[0] * y[1]])


def fitzhugh_nagumo(y, t):
    v, w = y[0], y[1]
    return jnp.stack([v - v ** 3 / 3 - w + 0.5, (v + 0.7 - 0.8 * w) / 12.5])


def _lv_np(t, y):
    return np.array([1.5 * y[0] - y[0] * y[1], -3.0 * y[1] + y[0] * y[1]])


def _fhn_np(t, y):
    v, w = y
    return np.array([v - v ** 3 / 3 - w + 0.5, (v + 0.7 - 0.8 * w) / 12.5])


def final(sol):
    n = int(sol.n_steps)
    return jax.tree.map(lambda a: a[n], sol.ys, is_leaf=u.math.is_quantity)


class TestExactSolutions:
    def test_linear_decay_t_eval(self, x64):
        te = jnp.linspace(0.0, 3.0, 13)
        sol = solve_ivp(decay, (0.0, 3.0), jnp.array(2.0), rtol=1e-9, atol=1e-12, t_eval=te)
        assert isinstance(sol, ODESolution)
        assert bool(sol.success)
        np.testing.assert_allclose(sol.ts, te)
        np.testing.assert_allclose(sol.ys, 2.0 * np.exp(-np.asarray(te)), rtol=1e-6)

    def test_harmonic_oscillator_pytree(self, x64):
        w = 3.0
        y0 = {'x': jnp.array(1.0), 'v': jnp.array(0.0)}
        sol = solve_ivp(oscillator, (0.0, 5.0), y0, w, rtol=1e-10, atol=1e-12, max_steps=2048)
        assert bool(sol.success)
        n = int(sol.n_steps)
        assert sol.ys['x'].shape == (2049,) and sol.ts.shape == (2049,)
        t = np.asarray(sol.ts[:n + 1])
        np.testing.assert_allclose(sol.ys['x'][:n + 1], np.cos(w * t), atol=1e-8)
        np.testing.assert_allclose(sol.ys['v'][:n + 1], -w * np.sin(w * t), atol=3e-8)
        # padding repeats the final accepted value
        assert float(sol.ts[n]) == 5.0
        np.testing.assert_array_equal(sol.ts[n:], 5.0)
        np.testing.assert_array_equal(sol.ys['x'][n:], sol.ys['x'][n])
        # accepted times strictly increase
        assert np.all(np.diff(t) > 0)

    def test_cubic_solution_is_interpolated_exactly(self, x64):
        # y = t^3 is reproduced by any RK of order >= 3 and by cubic Hermite interpolation
        te = jnp.array([0.0, 0.37, 1.1, 1.9, 2.0])
        sol = solve_ivp(lambda y, t: 3 * t ** 2 * jnp.ones_like(y), (0.0, 2.0), jnp.zeros(2), t_eval=te)
        assert sol.ys.shape == (5, 2)
        np.testing.assert_allclose(sol.ys, np.repeat(np.asarray(te)[:, None] ** 3, 2, axis=1), atol=1e-12)

    @pytest.mark.parametrize('method', ['dopri5', 'rk45', 'rkf45', 'rk23', 'bs32', 'dopri8', 'rk45_dopri',
                                        'rk87_dopri'])
    def test_every_adaptive_method(self, method, x64):
        sol = solve_ivp(decay, (0.0, 2.0), jnp.array([1.0, 3.0]), method=method, rtol=1e-8, atol=1e-10,
                        max_steps=1024)
        assert bool(sol.success)
        np.testing.assert_allclose(final(sol), np.array([1.0, 3.0]) * np.exp(-2.0), rtol=1e-6)


@pytest.mark.parametrize('rhs,fun,y0,T', [
    (lotka_volterra, _lv_np, [10.0, 5.0], 10.0),
    (fitzhugh_nagumo, _fhn_np, [-1.0, 1.0], 50.0),
], ids=['lotka_volterra', 'fitzhugh_nagumo'])
@pytest.mark.parametrize('ours,theirs,rtol,atol', [
    ('dopri5', 'RK45', 1e-6, 1e-9),
    ('dopri5', 'RK45', 1e-8, 1e-10),
    ('dopri8', 'DOP853', 1e-8, 1e-10),
])
def test_matches_scipy(rhs, fun, y0, T, ours, theirs, rtol, atol, x64):
    integrate = pytest.importorskip('scipy.integrate')
    te = np.linspace(0.0, T, 11)
    ref = integrate.solve_ivp(fun, (0, T), y0, method='DOP853', rtol=1e-13, atol=1e-13, t_eval=te).y.T
    sp = integrate.solve_ivp(fun, (0, T), y0, method=theirs, rtol=rtol, atol=atol, t_eval=te).y.T

    sol = solve_ivp(rhs, (0.0, T), jnp.asarray(y0), method=ours, rtol=rtol, atol=atol)
    assert bool(sol.success)
    err_end = np.max(np.abs(np.asarray(final(sol)) - ref[-1]))
    err_end_sp = np.max(np.abs(sp[-1] - ref[-1]))
    # same accuracy class as SciPy at the same tolerances
    assert err_end < 10 * max(err_end_sp, 10 * atol), (err_end, err_end_sp)
    assert err_end < 1e3 * rtol

    # steps land on the t_eval points, so they are as accurate as the steps themselves
    sol_te = solve_ivp(rhs, (0.0, T), jnp.asarray(y0), method=ours, rtol=rtol, atol=atol,
                       t_eval=jnp.asarray(te))
    assert bool(sol_te.success)
    err_te = np.max(np.abs(np.asarray(sol_te.ys) - ref))
    err_te_sp = np.max(np.abs(sp - ref))
    assert err_te < 5 * err_te_sp, (err_te, err_te_sp)


class TestEdgeCases:
    def test_stiff_hits_max_steps(self, x64):
        def van_der_pol(y, t, mu=1000.0):
            return jnp.stack([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])

        sol = solve_ivp(van_der_pol, (0.0, 3000.0), jnp.array([2.0, 0.0]), max_steps=100)
        assert not bool(sol.success)
        assert int(sol.n_steps) + int(sol.n_rejected) == 100
        assert np.all(np.isfinite(sol.ys)) and np.all(np.isfinite(sol.ts))
        n = int(sol.n_steps)
        assert float(sol.ts[n]) < 3000.0
        np.testing.assert_array_equal(sol.ys[n:], np.broadcast_to(sol.ys[n], sol.ys[n:].shape))

    def test_step_size_collapse(self):
        f = lambda y, t: jnp.where(t > 0.5, jnp.nan, -y)
        sol = solve_ivp(f, (0.0, 1.0), jnp.array(1.0), max_steps=500, t_eval=jnp.array([0.25, 0.5, 1.0]))
        assert not bool(sol.success)
        assert int(sol.n_steps) + int(sol.n_rejected) < 500
        assert np.all(np.isfinite(sol.ys))
        np.testing.assert_allclose(sol.ys[0], np.exp(-0.25), rtol=1e-4)

    def test_t_eval_at_span_ends(self, x64):
        te = jnp.array([0.0, 1.0])
        sol = solve_ivp(decay, (0.0, 1.0), jnp.array([1.0, 2.0]), t_eval=te, rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(sol.ys[0], [1.0, 2.0])
        sol_steps = solve_ivp(decay, (0.0, 1.0), jnp.array([1.0, 2.0]), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(sol.ys[1], final(sol_steps), rtol=1e-14)

    def test_backwards(self, x64):
        te = jnp.array([2.0, 1.5, 0.5, 0.0])
        sol = solve_ivp(decay, (2.0, 0.0), jnp.array(1.0), t_eval=te, rtol=1e-9, atol=1e-12)
        assert bool(sol.success)
        np.testing.assert_allclose(sol.ys, np.exp(2.0 - np.asarray(te)), rtol=1e-6)
        steps = solve_ivp(decay, (2.0, 0.0), jnp.array(1.0), rtol=1e-9, atol=1e-12)
        n = int(steps.n_steps)
        assert float(steps.ts[n]) == 0.0
        assert np.all(np.diff(np.asarray(steps.ts[:n + 1])) < 0)

    def test_empty_span(self):
        sol = solve_ivp(decay, (1.0, 1.0), jnp.array([1.0, 2.0]), max_steps=8)
        assert bool(sol.success) and int(sol.n_steps) == 0 and int(sol.n_rejected) == 0
        np.testing.assert_array_equal(sol.ys, np.tile([1.0, 2.0], (9, 1)))
        sol = solve_ivp(decay, (1.0, 1.0), jnp.array(3.0), max_steps=8, t_eval=jnp.array([1.0, 1.0]))
        np.testing.assert_array_equal(sol.ys, [3.0, 3.0])

    def test_unsorted_t_eval(self, x64):
        te = jnp.array([0.9, 0.1, 0.5])
        sol = solve_ivp(decay, (0.0, 1.0), jnp.array(1.0), t_eval=te, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(sol.ys, np.exp(-np.asarray(te)), rtol=1e-6)

    def test_empty_t_eval(self):
        sol = solve_ivp(lambda s, t: {'a': -s['a']}, (0.0, 1.0), {'a': jnp.ones(3)}, t_eval=jnp.zeros((0,)),
                        max_steps=64)
        assert bool(sol.success) and sol.ts.shape == (0,) and sol.ys['a'].shape == (0, 3)

    def test_dense_t_eval_lands_on_points(self, x64):
        te = jnp.linspace(0.0, 4.0, 41)
        sparse = solve_ivp(fitzhugh_nagumo, (0.0, 4.0), jnp.array([-1.0, 1.0]), rtol=1e-7, atol=1e-10)
        dense = solve_ivp(fitzhugh_nagumo, (0.0, 4.0), jnp.array([-1.0, 1.0]), rtol=1e-7, atol=1e-10,
                          t_eval=te)
        # about one extra step per t_eval point
        assert int(sparse.n_steps) <= int(dense.n_steps) <= int(sparse.n_steps) + 41 + 5
        np.testing.assert_allclose(dense.ys[-1], final(sparse), rtol=1e-6)
        # values at t_eval are step values: integrating to te[k] alone gives the same accuracy
        to_mid = solve_ivp(fitzhugh_nagumo, (0.0, 2.0), jnp.array([-1.0, 1.0]), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(dense.ys[20], final(to_mid), atol=1e-6)

    def test_integer_state(self):
        sol = solve_ivp(decay, (0.0, 1.0), jnp.array(1), t_eval=jnp.array([1.0]))
        assert jnp.issubdtype(sol.ys.dtype, jnp.floating)
        np.testing.assert_allclose(sol.ys, [np.exp(-1.0)], rtol=1e-4)

    def test_dt0_traced_quantity(self, x64):
        run = jax.jit(lambda h: solve_ivp(lambda y, t: -y / (1. * u.ms), (0. * u.ms, 1. * u.ms),
                                          jnp.array(1.0), dt0=h * u.ms, max_steps=128))
        sol = run(0.01)
        np.testing.assert_allclose(sol.ts[1].to_decimal(u.ms), 0.01, rtol=1e-12)

    def test_dt0(self, x64):
        sol = solve_ivp(decay, (0.0, 1.0), jnp.array(1.0), dt0=1e-3, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(float(sol.ts[1]), 1e-3, rtol=1e-12)
        np.testing.assert_allclose(final(sol), np.exp(-1.0), rtol=1e-7)

    def test_constant_rhs(self, x64):
        # zero derivative exercises the degenerate branches of the initial-step heuristic
        sol = solve_ivp(lambda y, t: jnp.zeros_like(y), (0.0, 1.0), jnp.zeros(3), max_steps=64)
        assert bool(sol.success)
        np.testing.assert_array_equal(final(sol), np.zeros(3))


class TestUnits:
    def test_quantity_time_and_state(self, x64):
        tau, v_rest = 10. * u.ms, -65. * u.mV

        def f(s, t):
            return {'V': (v_rest - s['V']) / tau, 'n': -s['n'] / tau}

        y0 = {'V': -70. * u.mV, 'n': jnp.array(1.0)}
        te = jnp.array([0.0, 5.0, 20.0]) * u.ms
        sol = solve_ivp(f, (0. * u.ms, 20. * u.ms), y0, rtol=1e-9, atol={'V': 1e-6 * u.mV, 'n': 1e-9}, t_eval=te)
        assert bool(sol.success)
        assert u.get_unit(sol.ts) == u.ms and u.get_unit(sol.ys['V']) == u.mV
        assert not isinstance(sol.ys['n'], u.Quantity)
        t = np.array([0.0, 5.0, 20.0])
        np.testing.assert_allclose(sol.ys['V'].to_decimal(u.mV), -65.0 - 5.0 * np.exp(-t / 10), rtol=1e-7)
        np.testing.assert_allclose(sol.ys['n'], np.exp(-t / 10), rtol=1e-6)

    def test_quantity_time_units_converted(self, x64):
        f = lambda y, t: -y / (1. * u.ms)
        sol = solve_ivp(f, (0. * u.ms, 0.002 * u.second), jnp.array(1.0), dt0=0.01 * u.ms,
                        t_eval=jnp.array([1.0, 2.0]) * u.ms, rtol=1e-9, atol=1e-12)
        assert u.get_unit(sol.ts) == u.ms
        np.testing.assert_allclose(sol.ys, np.exp([-1.0, -2.0]), rtol=1e-6)

    def test_atol_pytree(self, x64):
        y0 = {'a': jnp.array(1.0), 'b': 1.0 * u.mV}
        f = lambda s, t: {'a': -s['a'] / (1. * u.second), 'b': -s['b'] / (1. * u.second)}
        sol = solve_ivp(f, (0.0 * u.second, 1.0 * u.second), y0, atol={'a': 1e-10, 'b': 1e-10 * u.volt},
                        rtol=1e-8)
        assert bool(sol.success)
        np.testing.assert_allclose(final(sol)['b'].to_decimal(u.mV), np.exp(-1.0), rtol=1e-6)

    def test_wrong_derivative_unit(self):
        with pytest.raises(u.UnitMismatchError):
            solve_ivp(lambda y, t: -y, (0. * u.ms, 1. * u.ms), 1.0 * u.mV)

    def test_atol_wrong_unit(self):
        with pytest.raises(u.UnitMismatchError):
            solve_ivp(decay, (0.0, 1.0), 1.0 * u.mV, atol=1e-6 * u.ms)
        # a single Quantity atol cannot serve leaves of different units
        with pytest.raises(u.UnitMismatchError):
            solve_ivp(lambda s, t: {'a': -s['a'], 'b': -s['b']}, (0.0, 1.0),
                      {'a': 1.0 * u.mV, 'b': jnp.array(1.0)}, atol=1e-6 * u.mV)

    def test_plain_state_needs_rate_with_quantity_time(self):
        # with a Quantity time axis the derivative of a plain state has unit 1/[t]
        with pytest.raises(u.UnitMismatchError):
            solve_ivp(decay, (0. * u.ms, 1. * u.ms), jnp.array(1.0))


class TestValidation:
    def test_unknown_method(self):
        with pytest.raises(KeyError, match='dopri5'):
            solve_ivp(decay, (0.0, 1.0), 1.0, method='dopri4')

    def test_non_adaptive_method(self):
        with pytest.raises(ValueError, match='no embedded error estimate'):
            solve_ivp(decay, (0.0, 1.0), 1.0, method='rk4')

    def test_method_must_be_a_name(self):
        with pytest.raises(TypeError):
            solve_ivp(decay, (0.0, 1.0), 1.0, method=lambda *a: None)

    @pytest.mark.parametrize('span', [(0.0,), (0.0, 1.0, 2.0), 1.0])
    def test_bad_t_span(self, span):
        with pytest.raises(ValueError, match='t_span'):
            solve_ivp(decay, span, 1.0)

    @pytest.mark.parametrize('n', [0, -3, 2.5, True])
    def test_bad_max_steps(self, n):
        with pytest.raises(ValueError, match='max_steps'):
            solve_ivp(decay, (0.0, 1.0), 1.0, max_steps=n)

    def test_t_eval_outside_span(self):
        with pytest.raises(ValueError, match='inside t_span'):
            solve_ivp(decay, (0.0, 1.0), 1.0, t_eval=jnp.array([0.5, 1.5]))
        with pytest.raises(ValueError, match='inside t_span'):
            solve_ivp(decay, (1.0, 0.0), 1.0, t_eval=jnp.array([-0.1]))

    def test_t_eval_not_1d(self):
        with pytest.raises(ValueError, match='1-D'):
            solve_ivp(decay, (0.0, 1.0), 1.0, t_eval=jnp.zeros((2, 2)))

    def test_zero_dt0(self):
        with pytest.raises(ValueError, match='dt0'):
            solve_ivp(decay, (0.0, 1.0), 1.0, dt0=0.0)

    def test_atol_structure_mismatch(self):
        with pytest.raises(ValueError, match='atol'):
            solve_ivp(decay, (0.0, 1.0), {'a': 1.0, 'b': 2.0, 'c': 3.0}, atol=[1e-6, 1e-6])

    def test_rhs_structure_mismatch(self):
        with pytest.raises(ValueError, match='leaves'):
            solve_ivp(lambda y, t: (y, y), (0.0, 1.0), jnp.array(1.0))

    def test_complex_state(self):
        with pytest.raises(ValueError, match='real'):
            solve_ivp(decay, (0.0, 1.0), jnp.array(1.0 + 1.0j))


class TestTransforms:
    def test_jit_with_traced_span_and_state(self, x64):
        @jax.jit
        def run(y0, t1, k):
            return solve_ivp(lambda y, t, k: -k * y, (0.0, t1), y0, k, rtol=1e-9, atol=1e-12,
                             max_steps=256, t_eval=jnp.linspace(0.0, 1.0, 3) * t1)

        sol = run(jnp.array(2.0), 1.5, 0.8)
        assert bool(sol.success)
        np.testing.assert_allclose(sol.ys, 2.0 * np.exp(-0.8 * np.array([0.0, 0.75, 1.5])), rtol=1e-7)

    def test_jit_backwards_traced(self, x64):
        run = jax.jit(lambda t0, t1: solve_ivp(decay, (t0, t1), jnp.array(1.0), max_steps=128,
                                               t_eval=jnp.stack([t1])).ys)
        np.testing.assert_allclose(run(1.0, 0.0), [np.e], rtol=1e-5)
        np.testing.assert_allclose(run(0.0, 1.0), [np.exp(-1.0)], rtol=1e-5)

    def test_vmap_over_parameters(self, x64):
        k = jnp.array([0.5, 1.0, 4.0])
        run = jax.vmap(lambda k: solve_ivp(lambda y, t, k: -k * y, (0.0, 1.0), jnp.array(1.0), k,
                                           rtol=1e-9, atol=1e-12, max_steps=256,
                                           t_eval=jnp.array([1.0])))
        sol = run(k)
        assert sol.ys.shape == (3, 1)
        assert np.all(np.asarray(sol.success))
        # stiffer problems take more steps
        assert int(sol.n_steps[2]) > int(sol.n_steps[0])
        np.testing.assert_allclose(sol.ys[:, 0], np.exp(-np.asarray(k)), rtol=1e-7)

    def test_grad_matches_finite_differences(self, x64):
        def loss(p):
            k, y0 = p
            sol = solve_ivp(lambda y, t, k: -k * y + jnp.sin(t), (0.0, 2.0), y0, k, rtol=1e-11, atol=1e-13,
                            max_steps=2048, t_eval=jnp.array([0.7, 2.0]))
            return jnp.sum(sol.ys ** 2)

        p = (0.7, 1.3)
        g = jax.grad(lambda k, y0: loss((k, y0)), argnums=(0, 1))(*p)
        eps = 1e-5
        fd_k = (loss((p[0] + eps, p[1])) - loss((p[0] - eps, p[1]))) / (2 * eps)
        fd_y = (loss((p[0], p[1] + eps)) - loss((p[0], p[1] - eps))) / (2 * eps)
        np.testing.assert_allclose(g[0], fd_k, rtol=1e-6)
        np.testing.assert_allclose(g[1], fd_y, rtol=1e-6)

    def test_grad_wrt_t_eval(self, x64):
        def y_at(t):
            return solve_ivp(decay, (0.0, 2.0), jnp.array(1.0), rtol=1e-10, atol=1e-12,
                             t_eval=jnp.stack([t, 1.5]), max_steps=512).ys[0]

        np.testing.assert_allclose(jax.grad(y_at)(0.8), -np.exp(-0.8), rtol=1e-6)

    def test_grad_wrt_t1_and_pytree(self, x64):
        w = 2.0

        def x_at_end(t1):
            sol = solve_ivp(oscillator, (0.0, t1), {'x': jnp.array(1.0), 'v': jnp.array(0.0)}, w,
                            rtol=1e-10, atol=1e-12, max_steps=1024)
            return sol.ys['x'][sol.n_steps]

        t1 = 1.1
        np.testing.assert_allclose(x_at_end(t1), np.cos(w * t1), atol=1e-8)
        np.testing.assert_allclose(jax.grad(x_at_end)(t1), -w * np.sin(w * t1), rtol=1e-6)

    def test_grad_through_quantities(self, x64):
        def loss(tau_ms):
            f = lambda V, t, tau: (-65. * u.mV - V) / tau
            sol = solve_ivp(f, (0. * u.ms, 10. * u.ms), -70. * u.mV, tau_ms * u.ms, rtol=1e-10,
                            atol=1e-10 * u.mV, t_eval=jnp.array([10.0]) * u.ms)
            return sol.ys[0].to_decimal(u.mV)

        tau = 5.0
        # V(10) = -65 - 5 exp(-10 / tau)  =>  dV/dtau = -5 exp(-10/tau) * 10 / tau^2
        np.testing.assert_allclose(jax.grad(loss)(tau), -5 * np.exp(-10 / tau) * 10 / tau ** 2, rtol=1e-6)


class TestRobustGradients:
    """Rejected attempts must not leak NaN into reverse-mode gradients."""

    @staticmethod
    def _y_end(y0, dt0):
        f = lambda y, t: -jnp.sqrt(y)
        sol = solve_ivp(f, (0., 1.5), y0, dt0=dt0, rtol=1e-8, atol=1e-10, max_steps=200,
                        t_eval=jnp.array([1.5]))
        return sol.ys[0], sol

    def test_nan_in_rejected_attempt(self, x64):
        # dt0 = 1.5 overshoots into y < 0 (sqrt -> NaN) and is rejected twice
        y, sol = self._y_end(jnp.asarray(1.0), 1.5)
        assert int(sol.n_rejected) >= 1 and bool(sol.success)
        np.testing.assert_allclose(y, 0.0625, rtol=1e-6)
        # y(t) = (sqrt(y0) - t / 2)^2  =>  dy/dy0 = (sqrt(y0) - t / 2) / sqrt(y0) = 0.25
        g = jax.grad(lambda y0: self._y_end(y0, 1.5)[0])(jnp.asarray(1.0))
        np.testing.assert_allclose(g, 0.25, rtol=1e-6)

    def test_nan_in_rejected_attempt_vmapped(self, x64):
        y0 = jnp.array([1.0, 1.2])
        g = jax.vmap(jax.grad(lambda y0: self._y_end(y0, 1.5)[0]))(y0)
        np.testing.assert_allclose(g, (np.sqrt(y0) - 0.75) / np.sqrt(y0), rtol=1e-6)

    def test_forward_unchanged_by_rejections(self, x64):
        y_rej, _ = self._y_end(jnp.asarray(1.0), 1.5)
        y_auto, _ = self._y_end(jnp.asarray(1.0), None)
        np.testing.assert_allclose(y_rej, y_auto, rtol=1e-7)


def _temp_bytes(fn, *args):
    compiled = jax.jit(fn).lower(*args).compile()
    try:
        stats = compiled.memory_analysis()
    except Exception:  # pragma: no cover - backend without memory analysis
        stats = None
    if stats is None:  # pragma: no cover
        pytest.skip('memory_analysis is not available on this backend')
    return stats.temp_size_in_bytes


class TestMemory:
    N = 4000

    @staticmethod
    def _f(y, t):
        return -y + 0.1 * jnp.tanh(jnp.roll(y, 1))

    def test_t_eval_forward_memory_independent_of_max_steps(self):
        te = jnp.linspace(0., 1., 10)
        y0 = jax.ShapeDtypeStruct((self.N,), jnp.float32)
        state = self.N * 4

        def fwd(max_steps):
            return lambda y: solve_ivp(self._f, (0., 1.), y, t_eval=te, max_steps=max_steps).ys

        small, large = _temp_bytes(fwd(256), y0), _temp_bytes(fwd(4096), y0)
        # a few copies of the state plus the (10, N) output buffer, not max_steps copies
        assert large < 64 * state, large / state
        assert large < small + 4 * state, (small / state, large / state)

    def test_checkpoint_bounds_gradient_memory(self):
        te = jnp.linspace(0., 1., 10)
        y0 = jax.ShapeDtypeStruct((self.N,), jnp.float32)
        state = self.N * 4

        def loss(checkpoint):
            return jax.grad(lambda y: jnp.sum(solve_ivp(self._f, (0., 1.), y, t_eval=te, max_steps=4096,
                                                        checkpoint=checkpoint).ys[-1]))

        plain = _temp_bytes(loss(False), y0)
        ckpt = _temp_bytes(loss(True), y0)
        assert ckpt < plain / 8, (plain / state, ckpt / state)
        assert ckpt < 2000 * state, ckpt / state

    def test_checkpoint_same_values_and_gradient(self, x64):
        def loss(k, checkpoint):
            sol = solve_ivp(lambda y, t, k: -k * jnp.sqrt(jnp.abs(y) + 1.0), (0.0, 2.0), jnp.array([1.0, 2.0]),
                            k, rtol=1e-8, atol=1e-10, max_steps=300, t_eval=jnp.array([0.5, 2.0]),
                            checkpoint=checkpoint)
            return jnp.sum(sol.ys ** 2), sol

        (v0, s0), g0 = jax.value_and_grad(loss, has_aux=True)(0.7, False)
        (v1, s1), g1 = jax.value_and_grad(loss, has_aux=True)(0.7, True)
        np.testing.assert_allclose(v1, v0, rtol=1e-12)
        np.testing.assert_allclose(g1, g0, rtol=1e-10)
        assert int(s1.n_steps) == int(s0.n_steps)
        steps0 = solve_ivp(decay, (0.0, 1.0), jnp.array(1.0), max_steps=64)
        steps1 = solve_ivp(decay, (0.0, 1.0), jnp.array(1.0), max_steps=64, checkpoint=True)
        np.testing.assert_allclose(steps1.ys, steps0.ys, rtol=1e-12)
