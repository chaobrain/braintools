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
import optax
import pytest

from braintools.optim._multistart import MultiStartResult, multistart_init, multistart_minimize


def rosenbrock(p):
    return (1. - p['x']) ** 2 + 100. * (p['y'] - p['x'] ** 2) ** 2


def wavy(x):
    return jnp.sin(3. * x) + 0.1 * x ** 2


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


# ---------------------------------------------------------------------------
# reference problems
# ---------------------------------------------------------------------------

def test_rosenbrock_many_starts():
    key = jax.random.PRNGKey(0)
    x0 = multistart_init(key, {'x': (-2., 2.), 'y': (-1., 3.)}, 16)
    res = multistart_minimize(rosenbrock, x0, max_iter=300)
    assert isinstance(res, MultiStartResult)
    assert res.fun.shape == (16,) and res.converged.shape == (16,)
    assert res.history.shape == (16, 300)
    assert bool(res.converged.all())
    np.testing.assert_allclose(np.asarray(res.x['x']), 1., atol=1e-2)
    np.testing.assert_allclose(np.asarray(res.x['y']), 1., atol=2e-2)
    np.testing.assert_allclose(float(res.best_x['x']), 1., atol=1e-3)
    assert float(res.best_fun) == float(res.fun.min())
    assert int(res.best_index) == int(jnp.argmin(res.fun))


def test_rosenbrock_matches_scipy(x64):
    scipy_optimize = pytest.importorskip('scipy.optimize')
    x0 = {'x': jnp.array([-1.2, 0.5]), 'y': jnp.array([1., -0.5])}
    res = multistart_minimize(rosenbrock, x0, max_iter=500, tol=1e-10)
    for s in range(2):
        ref = scipy_optimize.minimize(
            lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2,
            np.array([float(x0['x'][s]), float(x0['y'][s])]), method='BFGS', tol=1e-12,
        )
        np.testing.assert_allclose([float(res.x['x'][s]), float(res.x['y'][s])], ref.x, atol=1e-5)


def test_multimodal_finds_all_minima_and_best():
    grid = np.linspace(-3, 3, 600001)
    v = np.sin(3 * grid) + 0.1 * grid ** 2
    idx = np.where((v[1:-1] < v[:-2]) & (v[1:-1] < v[2:]))[0] + 1
    local_minima = grid[idx]
    res = multistart_minimize(wavy, jnp.linspace(-2.9, 2.9, 30), bounds=(-3., 3.))
    found = np.asarray(res.x)
    interior = found[np.abs(found) < 2.99]
    for m in local_minima:
        assert np.min(np.abs(interior - m)) < 1e-3
    for xf in interior:
        assert np.min(np.abs(local_minima - xf)) < 1e-3
    global_min = local_minima[np.argmin(np.sin(3 * local_minima) + 0.1 * local_minima ** 2)]
    np.testing.assert_allclose(float(res.best_x), global_min, atol=1e-3)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method, lr', [('adam', 0.05), ('gd', 0.1)])
def test_first_order_methods(method, lr):
    fun = lambda x: jnp.sum((x - jnp.array([1., -2.])) ** 2)
    x0 = jnp.array([[0., 0.], [3., 3.], [-1., 4.]])
    # fixed-step methods stop on the gradient test only, so ``tol`` must be
    # reachable in float32
    res = multistart_minimize(fun, x0, method=method, learning_rate=lr, max_iter=2000, tol=1e-4)
    np.testing.assert_allclose(np.asarray(res.x), np.tile([1., -2.], (3, 1)), atol=1e-3)
    assert bool(res.converged.all())


def test_optax_transformation_method():
    fun = lambda x: (x - 3.) ** 2
    res = multistart_minimize(fun, jnp.array([0., 10.]), method=optax.sgd(0.1), max_iter=500, tol=1e-10)
    np.testing.assert_allclose(np.asarray(res.x), 3., atol=1e-3)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match='Unknown method'):
        multistart_minimize(wavy, jnp.zeros(2), method='newton')
    with pytest.raises(TypeError, match='method'):
        multistart_minimize(wavy, jnp.zeros(2), method=3)


def test_history_is_monotone_for_lbfgs_and_padded():
    res = multistart_minimize(rosenbrock, {'x': jnp.array([-1.2]), 'y': jnp.array([1.])}, max_iter=100)
    hist = np.asarray(res.history[0])
    n = int(res.n_iter[0])
    assert n < 100
    assert np.all(np.diff(hist[:n]) <= 1e-6)
    np.testing.assert_allclose(hist[n:], float(res.fun[0]))


def test_max_iter_reached_not_converged():
    res = multistart_minimize(rosenbrock, {'x': jnp.array([-1.2]), 'y': jnp.array([1.])}, max_iter=3)
    assert int(res.n_iter[0]) == 3
    assert not bool(res.converged[0])


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

def test_all_starts_already_at_optimum():
    fun = lambda x: jnp.sum((x - 2.) ** 2)
    res = multistart_minimize(fun, jnp.full((4, 3), 2.))
    assert bool(res.converged.all())
    np.testing.assert_array_equal(np.asarray(res.n_iter), 0)
    np.testing.assert_allclose(np.asarray(res.x), 2.)
    np.testing.assert_allclose(np.asarray(res.history), 0.)


def test_non_finite_starts_are_never_best():
    # log(x) is nan for x < 0: those starts are not finite from the outset
    fun = lambda x: (jnp.log(x) - 1.) ** 2 + 5.
    x0 = jnp.array([-1., 0.5, -3., 4.])
    res = multistart_minimize(fun, x0)
    conv = np.asarray(res.converged)
    assert not conv[0] and not conv[2]
    assert conv[1] and conv[3]
    np.testing.assert_array_equal(np.asarray(res.n_iter)[[0, 2]], 0)
    assert int(res.best_index) in (1, 3)
    assert np.isfinite(float(res.best_fun))
    np.testing.assert_allclose(float(res.best_x), np.e, rtol=1e-3)


def test_all_starts_non_finite():
    res = multistart_minimize(lambda x: jnp.log(x), jnp.array([-1., -2.]))
    assert not bool(res.converged.any())
    assert int(res.best_index) == 0
    assert not np.isfinite(float(res.best_fun))


def test_loss_becoming_non_finite_keeps_last_finite_point():
    # gradient descent with a huge step jumps into the nan region x < 0
    fun = lambda x: jnp.where(x > 0., (x - 1.) ** 2, jnp.nan)
    res = multistart_minimize(fun, jnp.array([3.]), method='gd', learning_rate=10., max_iter=20)
    assert not bool(res.converged[0])
    assert np.isfinite(float(res.fun[0]))
    np.testing.assert_allclose(float(res.x[0]), 3.)
    assert int(res.n_iter[0]) == 1


def test_x0_validation():
    with pytest.raises(ValueError, match='leading'):
        multistart_minimize(rosenbrock, {'x': jnp.zeros(2), 'y': jnp.zeros(3)})
    with pytest.raises(ValueError, match='leading'):
        multistart_minimize(wavy, jnp.asarray(1.))
    with pytest.raises(ValueError, match='no leaves'):
        multistart_minimize(wavy, {})
    with pytest.raises(ValueError, match='max_iter'):
        multistart_minimize(wavy, jnp.zeros(2), max_iter=0)


def test_fun_must_be_scalar_and_unitless():
    with pytest.raises(ValueError, match='scalar'):
        multistart_minimize(lambda x: x * jnp.ones(2), jnp.zeros(2))
    with pytest.raises(TypeError, match='dimensionless'):
        multistart_minimize(lambda x: x * u.mV, jnp.zeros(2))
    res = multistart_minimize(lambda x: (x - 1.) ** 2 * u.Quantity(1.), jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(res.x), 1., atol=1e-3)


def test_integer_x0_is_promoted():
    res = multistart_minimize(lambda x: (x - 1.5) ** 2, jnp.array([0, 4]))
    np.testing.assert_allclose(np.asarray(res.x), 1.5, atol=1e-3)


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------

def test_bounded_minimum_on_the_boundary():
    # unconstrained minimum at 5 lies outside [0, 2]: the bounded result tends to 2
    fun = lambda x: (x - 5.) ** 2
    res = multistart_minimize(fun, jnp.array([0.5, 1.5]), bounds=(0., 2.), max_iter=200)
    assert bool((res.x <= 2.).all() & (res.x >= 0.).all())
    np.testing.assert_allclose(np.asarray(res.x), 2., atol=1e-2)


def test_pytree_bounds_with_none_and_one_sided():
    fun = lambda p: (p['a'] - 0.3) ** 2 + jnp.sum((p['b'] + 1.) ** 2) + (p['c'] - 4.) ** 2 + (p['d'] + 2.) ** 2
    x0 = {'a': jnp.array([0.1, 0.9]), 'b': jnp.zeros((2, 2)),
          'c': jnp.array([2., 10.]), 'd': jnp.array([-5., -1.])}
    bounds = {'a': (0., 1.), 'b': None, 'c': (1., None), 'd': (None, 0.)}
    res = multistart_minimize(fun, x0, bounds=bounds, max_iter=300)
    np.testing.assert_allclose(np.asarray(res.x['a']), 0.3, atol=1e-3)
    np.testing.assert_allclose(np.asarray(res.x['b']), -1., atol=1e-3)
    np.testing.assert_allclose(np.asarray(res.x['c']), 4., atol=1e-3)
    np.testing.assert_allclose(np.asarray(res.x['d']), -2., atol=1e-3)
    assert bool((res.x['d'] <= 0.).all())


def test_array_bounds_and_start_on_bound():
    fun = lambda x: jnp.sum((x - jnp.array([0.5, 2.5])) ** 2)
    bounds = (jnp.array([0., 2.]), jnp.array([1., 3.]))
    x0 = jnp.array([[0., 3.], [0.2, 2.2]])  # first start sits on the bounds
    res = multistart_minimize(fun, x0, bounds=bounds, max_iter=200)
    assert np.all(np.isfinite(np.asarray(res.x)))
    np.testing.assert_allclose(np.asarray(res.best_x), [0.5, 2.5], atol=1e-3)


def test_quantity_x0_and_bounds():
    fun = lambda v: ((v - (-60. * u.mV)) / u.mV) ** 2
    x0 = jnp.array([-75., -50.]) * u.mV
    res = multistart_minimize(fun, x0, bounds=(-0.08 * u.volt, -40. * u.mV), max_iter=200)
    assert isinstance(res.x, u.Quantity) and res.x.unit == u.mV
    np.testing.assert_allclose(res.best_x.to_decimal(u.mV), -60., atol=1e-2)


def test_bound_validation():
    with pytest.raises(ValueError, match='low < high'):
        multistart_minimize(wavy, jnp.zeros(2), bounds=(1., 0.))
    with pytest.raises(ValueError, match='finite'):
        multistart_minimize(wavy, jnp.zeros(2), bounds=(0., jnp.inf))
    with pytest.raises(ValueError, match='mirror'):
        multistart_minimize(rosenbrock, {'x': jnp.zeros(2), 'y': jnp.zeros(2)}, bounds={'x': (0., 1.)})
    with pytest.raises(ValueError, match='pair'):
        multistart_minimize(wavy, jnp.zeros(2), bounds=[(0., 1.)])
    with pytest.raises(TypeError, match='unit'):
        multistart_minimize(wavy, jnp.zeros(2), bounds=(0. * u.mV, 1. * u.mV))
    with pytest.raises(TypeError, match='compatible'):
        multistart_minimize(lambda v: (v / u.mV) ** 2, jnp.zeros(2) * u.mV, bounds=(0., 1.))


# ---------------------------------------------------------------------------
# transformations
# ---------------------------------------------------------------------------

def test_jit_wrapping():
    @jax.jit
    def solve(x0):
        return multistart_minimize(wavy, x0, bounds=(-3., 3.)).best_x

    np.testing.assert_allclose(float(solve(jnp.linspace(-2.5, 2.5, 5))), -0.5122, atol=1e-3)


def test_jit_with_traced_bounds():
    @jax.jit
    def solve(x0, hi):
        return multistart_minimize(lambda x: (x - 5.) ** 2, x0, bounds=(0., hi)).best_x

    np.testing.assert_allclose(float(solve(jnp.array([0.5, 1.]), 2.)), 2., atol=1e-2)


def test_vmap_over_problems():
    def solve(c):
        return multistart_minimize(lambda x: (x - c) ** 2, jnp.array([-1., 1.])).best_x

    cs = jnp.array([0.5, 2., -3.])
    np.testing.assert_allclose(np.asarray(jax.vmap(solve)(cs)), np.asarray(cs), atol=1e-4)


# ---------------------------------------------------------------------------
# multistart_init
# ---------------------------------------------------------------------------

def test_multistart_init_shapes_ranges_and_units():
    key = jax.random.PRNGKey(1)
    bounds = {'a': (0., 1.), 'b': (jnp.zeros(3), jnp.full(3, 2.)), 'c': (-70. * u.mV, -0.05 * u.volt)}
    x0 = multistart_init(key, bounds, 5)
    assert x0['a'].shape == (5,) and x0['b'].shape == (5, 3)
    assert isinstance(x0['c'], u.Quantity) and x0['c'].unit == u.mV
    c = x0['c'].to_decimal(u.mV)
    assert np.all((c >= -70.) & (c < -50.))
    assert np.all((np.asarray(x0['b']) >= 0.) & (np.asarray(x0['b']) < 2.))
    single = multistart_init(key, (0., 1.), 3)
    assert single.shape == (3,)


def test_multistart_init_reproducible_and_vmappable():
    b = {'a': (0., 1.)}
    k = jax.random.PRNGKey(3)
    np.testing.assert_array_equal(np.asarray(multistart_init(k, b, 4)['a']),
                                  np.asarray(multistart_init(k, b, 4)['a']))
    keys = jax.random.split(k, 2)
    out = jax.vmap(lambda kk: multistart_init(kk, b, 4)['a'])(keys)
    assert out.shape == (2, 4)
    jitted = jax.jit(lambda kk: multistart_init(kk, b, 4))(k)
    np.testing.assert_allclose(np.asarray(jitted['a']), np.asarray(multistart_init(k, b, 4)['a']))


def test_multistart_init_validation():
    with pytest.raises(ValueError, match='n_starts'):
        multistart_init(jax.random.PRNGKey(0), (0., 1.), 0)
    with pytest.raises(ValueError, match='n_starts'):
        multistart_init(jax.random.PRNGKey(0), (0., 1.), True)
    with pytest.raises(ValueError, match='no'):
        multistart_init(jax.random.PRNGKey(0), {}, 2)
    with pytest.raises(ValueError, match='pair'):
        multistart_init(jax.random.PRNGKey(0), {'a': 1.}, 2)


def test_float64_precision(x64):
    res = multistart_minimize(rosenbrock, {'x': jnp.array([-1.2]), 'y': jnp.array([1.])}, tol=1e-12)
    assert res.fun.dtype == jnp.float64
    np.testing.assert_allclose(float(res.best_x['x']), 1., atol=1e-6)


# ---------------------------------------------------------------------------
# regressions from review
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['lbfgs', 'adam'])
def test_starts_on_the_bounds_move_to_the_optimum(method):
    # a start exactly on a bound must not be reported converged at iteration 0
    res = multistart_minimize(lambda x: (x - 0.5) ** 2, jnp.array([0., 1., .2]), bounds=(0., 1.),
                              method=method, learning_rate=0.1, max_iter=2000)
    np.testing.assert_allclose(np.asarray(res.x), 0.5, atol=1e-2)
    assert np.all(np.asarray(res.n_iter) > 0)


def test_start_on_bound_with_gd_is_not_reported_converged():
    # plain gradient descent crawls where the sigmoid is flat, but must not
    # claim convergence there
    res = multistart_minimize(lambda x: (x - 0.5) ** 2, jnp.array([0., 1.]), bounds=(0., 1.),
                              method='gd', learning_rate=0.1, max_iter=50)
    assert not bool(res.converged.any())
    np.testing.assert_array_equal(np.asarray(res.n_iter), 50)


def test_gradient_test_is_in_the_constrained_space():
    # at a bounded optimum on the boundary the constrained gradient stays large:
    # convergence must come from the loss test, not from a vanishing sigmoid slope
    res = multistart_minimize(lambda x: (x - 5.) ** 2, jnp.array([1.999]), bounds=(0., 2.))
    assert int(res.n_iter[0]) > 0


def test_diverged_start_is_never_best():
    # start 0 walks into a nan region (its last finite loss, -2.5, is lower than
    # the proper local minimum of start 1, 10.)
    g = lambda x: jnp.where(x >= 3., jnp.nan, jnp.where(x < -5., (x + 8.) ** 2 + 10., -x))
    res = multistart_minimize(g, jnp.array([0., -10.]), method='gd', learning_rate=0.5, max_iter=100)
    assert not bool(res.converged[0])
    assert int(res.best_index) == 1
    np.testing.assert_allclose(float(res.best_fun), 10., atol=1e-4)


def test_all_diverged_falls_back_to_lowest_finite_loss():
    g = lambda x: jnp.where(x >= 3., jnp.nan, -x)
    res = multistart_minimize(g, jnp.array([0., 1.]), method='gd', learning_rate=0.5, max_iter=100)
    assert not bool(res.converged.any())
    assert np.isfinite(float(res.best_fun))
    assert float(res.best_fun) == float(jnp.min(res.fun))


def test_first_order_methods_ignore_loss_change_test():
    # a large constant offset makes the relative loss change tiny long before
    # the minimum; adam/gd must rely on the gradient test only
    res = multistart_minimize(lambda x: (x - 3.) ** 2 + 1e4, jnp.array([0.]), method='adam',
                              learning_rate=1e-2, max_iter=2000)
    np.testing.assert_allclose(float(res.x[0]), 3., atol=2e-2)
