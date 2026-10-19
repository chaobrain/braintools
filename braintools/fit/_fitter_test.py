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

import time

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from braintools.fit._fitter import Fitter, FitResult
from braintools.fit._objective import Term
from braintools.optim._multistart import MultiStartResult
from braintools.quad import ode_rk4_step

T = jnp.linspace(0., 1., 40)


def mse(pred, target):
    return jnp.mean((pred - target) ** 2)


def decay(p):
    return p['a'] * jnp.exp(-T / p['tau'])


TRUE = {'a': 2.0, 'tau': 0.3}
TARGET = decay(TRUE)
SPACE = {'a': (0., 5.), 'tau': (0.05, 1.)}


class Decay(brainstate.nn.Module):
    def __init__(self, a=1., tau=0.5, reg=None):
        super().__init__()
        self.a = brainstate.nn.Param(a, t=brainstate.nn.SigmoidT(0., 5.))
        self.tau = brainstate.nn.Param(tau, t=brainstate.nn.SigmoidT(0.05, 1.), reg=reg)

    def __call__(self):
        return self.a.value() * jnp.exp(-T / self.tau.value())


def predict(model):
    return model()


# ---------------------------------------------------------------------------
# functional mode, 'grad'
# ---------------------------------------------------------------------------

def test_functional_grad_recovers_parameters():
    res = Fitter(decay, mse, SPACE).fit(TARGET, n_starts=4, key=jax.random.PRNGKey(0), n_steps=100)
    assert isinstance(res, FitResult) and res.backend == 'grad'
    assert isinstance(res.raw, MultiStartResult)
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-3)
    np.testing.assert_allclose(float(res.best_params['tau']), 0.3, rtol=1e-3)
    assert res.params['a'].shape == (4,) and res.losses.shape == (4,)
    assert res.history.shape == (4, 100) and res.converged.shape == (4,)
    assert res.best_loss == pytest.approx(float(jnp.min(res.losses)))
    assert np.all((np.asarray(res.params['tau']) > 0.05) & (np.asarray(res.params['tau']) < 1.))
    assert 'grad' in repr(res) and 'n_starts=4' in repr(res)


def test_single_start_defaults_to_midpoint_or_init():
    fitter = Fitter(decay, mse, SPACE, init={'tau': 0.5})
    res = fitter.fit(TARGET, n_steps=1)
    hist0 = float(res.history[0, 0])
    assert res.losses.shape == (1,)
    # the first step started from a=2.5 (midpoint) and tau=0.5 (init)
    start_loss = float(mse(decay({'a': 2.5, 'tau': 0.5}), TARGET))
    assert hist0 < start_loss
    assert fitter.names == ('a', 'tau')


def test_single_start_equals_vmapped_start():
    x0 = {'a': jnp.array([0.5, 4., 2.5, 1.]), 'tau': jnp.array([0.9, 0.1, 0.5, 0.2])}
    fitter = Fitter(decay, mse, SPACE)
    batch = fitter.fit(TARGET, x0=x0, n_steps=60)
    for i in range(4):
        one = fitter.fit(TARGET, x0={k: v[i:i + 1] for k, v in x0.items()}, n_steps=60)
        np.testing.assert_allclose(float(one.params['a'][0]), float(batch.params['a'][i]), rtol=1e-4)
        np.testing.assert_allclose(float(one.params['tau'][0]), float(batch.params['tau'][i]), rtol=1e-4)
        np.testing.assert_allclose(np.asarray(one.history[0]), np.asarray(batch.history[i]),
                                   rtol=1e-3, atol=1e-7)


def test_draws_are_reproducible_and_require_key():
    fitter = Fitter(decay, mse, SPACE)
    r1 = fitter.fit(TARGET, n_starts=3, key=jax.random.PRNGKey(5), n_steps=2)
    r2 = fitter.fit(TARGET, n_starts=3, key=jax.random.PRNGKey(5), n_steps=2)
    np.testing.assert_array_equal(np.asarray(r1.params['a']), np.asarray(r2.params['a']))
    with pytest.raises(ValueError, match='key'):
        fitter.fit(TARGET, n_starts=3)


def test_quantity_bounds_restore_units():
    t = jnp.linspace(0., 50., 30) * u.ms

    def simulate(p):
        return jnp.exp(-t / p['tau']) * p['v']

    true = {'tau': 12. * u.ms, 'v': -3. * u.mV}
    target = simulate(true)

    def objective(pred, y):
        return jnp.mean(((pred - y) / u.mV) ** 2)

    space = {'tau': (2. * u.ms, 0.05 * u.second), 'v': (-10. * u.mV, 0. * u.mV)}
    res = Fitter(simulate, objective, space).fit(target, n_starts=4, key=jax.random.PRNGKey(1))
    assert isinstance(res.best_params['tau'], u.Quantity)
    assert res.best_params['tau'].unit == u.ms
    np.testing.assert_allclose(res.best_params['tau'].to_decimal(u.ms), 12., rtol=1e-3)
    np.testing.assert_allclose(res.best_params['v'].to_decimal(u.mV), -3., rtol=1e-3)
    assert res.params['v'].shape == (4,) and res.params['v'].unit == u.mV


def test_unbounded_and_one_sided_parameters():
    def simulate(p):
        return p['a'] * jnp.exp(-T / p['tau']) + p['c']

    target = simulate({'a': 2., 'tau': 0.3, 'c': 0.5})
    space = {'a': None, 'tau': (0.01, None), 'c': (0., 1.)}
    res = Fitter(simulate, mse, space, init={'a': 1., 'tau': 0.6}).fit(target, n_starts=3,
                                                                      key=jax.random.PRNGKey(2))
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-3)
    np.testing.assert_allclose(float(res.best_params['tau']), 0.3, rtol=1e-3)
    np.testing.assert_allclose(float(res.best_params['c']), 0.5, rtol=1e-3)


def test_array_parameter():
    w_true = jnp.array([0.5, -1., 2.])
    X = jnp.stack([jnp.ones(10), jnp.linspace(0, 1, 10), jnp.linspace(0, 1, 10) ** 2], 1)

    res = Fitter(lambda p: X @ p['w'], mse, {'w': (-3. * jnp.ones(3), 3. * jnp.ones(3))}).fit(X @ w_true)
    np.testing.assert_allclose(np.asarray(res.best_params['w']), np.asarray(w_true), atol=1e-3)


def test_dict_prediction_with_objective_terms():
    def simulate(p):
        return {'x': p['a'] * T, 'y': p['b'] * jnp.ones(3)}

    target = {'x': 1.5 * T, 'y': -0.5 * jnp.ones(3)}
    obj = Term(mse, key='x') + 2. * Term(mse, key='y')
    res = Fitter(simulate, obj, {'a': (0., 3.), 'b': (-1., 1.)}).fit(target)
    np.testing.assert_allclose(float(res.best_params['a']), 1.5, rtol=1e-3)
    np.testing.assert_allclose(float(res.best_params['b']), -0.5, rtol=1e-3)


def test_transient_is_trimmed():
    def simulate(p):
        # the first five samples are garbage that would bias the fit
        return jnp.concatenate([jnp.full(5, 100.), p['a'] * T])

    target = jnp.concatenate([jnp.zeros(5), 1.2 * T])[5:]
    fitter = Fitter(simulate, mse, {'a': (0., 3.)}, transient=5)
    np.testing.assert_allclose(float(fitter.fit(target).best_params['a']), 1.2, rtol=1e-3)
    res = Fitter(simulate, mse, {'a': (0., 3.)}).fit(target, transient=5)
    np.testing.assert_allclose(float(res.best_params['a']), 1.2, rtol=1e-3)


@pytest.mark.parametrize('optimizer', [
    'adam',
    {'method': 'adam', 'learning_rate': 0.05, 'tol': 1e-9},
    optax.adam(0.05),
])
def test_grad_optimizer_options(optimizer):
    res = Fitter(decay, mse, SPACE, optimizer=optimizer).fit(TARGET, n_steps=400)
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=2e-2)


def test_grad_with_braintools_optax_optimizer():
    from braintools.optim import Adam
    res = Fitter(decay, mse, SPACE, optimizer=Adam(lr=0.05)).fit(TARGET, n_steps=400)
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=2e-2)


# ---------------------------------------------------------------------------
# scipy and nevergrad backends
# ---------------------------------------------------------------------------

def test_scipy_backend_two_parameters():
    pytest.importorskip('scipy')
    res = Fitter(decay, mse, SPACE, backend='scipy').fit(TARGET, n_starts=2, key=jax.random.PRNGKey(0),
                                                          n_steps=50)
    assert res.backend == 'scipy' and len(res.raw) == 2
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-3)
    np.testing.assert_allclose(float(res.best_params['tau']), 0.3, rtol=1e-3)
    assert res.history.shape == (2, 50)
    assert bool(res.converged.any())


def test_scipy_backend_gradient_free_method_and_options():
    res = Fitter(decay, mse, SPACE, backend='scipy',
                 optimizer={'method': 'Powell', 'tol': 1e-10, 'maxfev': 4000}).fit(TARGET, n_steps=500)
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-2)


def test_nevergrad_backend_two_parameters():
    pytest.importorskip('nevergrad')
    res = Fitter(decay, mse, SPACE, backend='nevergrad').fit(TARGET, n_starts=16, key=jax.random.PRNGKey(0),
                                                              n_steps=40)
    assert res.backend == 'nevergrad'
    assert res.history.shape == (16, 40) and res.losses.shape == (16,)
    assert res.params['a'].shape == (16,)
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=0.05)
    np.testing.assert_allclose(float(res.best_params['tau']), 0.3, rtol=0.05)
    assert res.best_loss == pytest.approx(float(np.min(np.asarray(res.history))))


def test_nevergrad_options_and_quantity_bounds():
    pytest.importorskip('nevergrad')
    t = jnp.linspace(0., 1., 20) * u.second

    def simulate(p):
        return jnp.exp(-t / p['tau'])

    target = simulate({'tau': 300. * u.ms})
    res = Fitter(simulate, mse, {'tau': (50. * u.ms, 1. * u.second)}, backend='nevergrad',
                 optimizer={'method': 'OnePlusOne', 'n_sample': 4}).fit(target, key=jax.random.PRNGKey(3),
                                                                        n_steps=60)
    assert res.best_params['tau'].unit == u.ms
    assert res.history.shape == (4, 60)
    np.testing.assert_allclose(res.best_params['tau'].to_decimal(u.ms), 300., rtol=0.05)


def test_nevergrad_requires_box_and_key():
    pytest.importorskip('nevergrad')
    fitter = Fitter(decay, mse, {'a': None, 'tau': (0.05, 1.)}, init={'a': 1.}, backend='nevergrad')
    with pytest.raises(ValueError, match='finite'):
        fitter.fit(TARGET, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match='key'):
        Fitter(decay, mse, SPACE, backend='nevergrad').fit(TARGET)


# ---------------------------------------------------------------------------
# module mode
# ---------------------------------------------------------------------------

def test_module_grad_writes_back_only_at_the_end():
    model = Decay()
    before = {k: float(v) for k, v in (('a', model.a.value()), ('tau', model.tau.value()))}
    fitter = Fitter.from_module(model, predict, mse)
    assert fitter.names == ('a', 'tau')
    loss_z, loss_x = fitter._loss_functions(TARGET, None)
    float(loss_x({'a': jnp.asarray(2.), 'tau': jnp.asarray(0.3)}))
    jax.vmap(loss_z)({'a': jnp.zeros(3), 'tau': jnp.zeros(3)})
    # evaluating candidates does not touch the model
    assert float(model.a.value()) == before['a'] and float(model.tau.value()) == before['tau']
    res = fitter.fit(TARGET, n_starts=4, key=jax.random.PRNGKey(0))
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-3)
    np.testing.assert_allclose(float(res.best_params['tau']), 0.3, rtol=1e-3)
    np.testing.assert_allclose(float(model.a.value()), float(res.best_params['a']), rtol=1e-6)
    np.testing.assert_allclose(float(model.tau.value()), float(res.best_params['tau']), rtol=1e-6)


def test_module_single_start_uses_current_value():
    model = Decay(a=2., tau=0.3)
    res = Fitter.from_module(model, predict, mse).fit(TARGET, n_steps=5)
    assert int(res.raw.n_iter[0]) == 0
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-5)


def test_module_reg_loss_is_added():
    reg = brainstate.nn.L2Reg(weight=0.1)
    model = Decay(a=2., tau=0.3, reg=reg)
    fitter = Fitter.from_module(model, predict, mse)
    _, loss_x = fitter._loss_functions(TARGET, None)
    value = float(loss_x({'a': jnp.asarray(2.), 'tau': jnp.asarray(0.3)}))
    np.testing.assert_allclose(value, float(model.reg_loss()), rtol=1e-4)
    assert value > 0.


def test_module_without_fit_params_raises():
    class Fixed(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = brainstate.nn.Param(1., fit=False)

    with pytest.raises(ValueError, match='no trainable'):
        Fitter.from_module(Fixed(), lambda m: m.a.value(), mse)


def test_module_quantity_param_and_nested_names():
    class Inner(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.tau = brainstate.nn.Param(20. * u.ms, t=brainstate.nn.SigmoidT(2. * u.ms, 50. * u.ms))

    class Outer(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

    t = jnp.linspace(0., 60., 30) * u.ms
    model = Outer()
    pred = lambda m: jnp.exp(-t / m.inner.tau.value())
    target = jnp.exp(-t / (8. * u.ms))
    res = Fitter.from_module(model, pred, mse).fit(target, n_starts=3, key=jax.random.PRNGKey(0))
    assert list(res.best_params) == ['inner.tau']
    assert res.best_params['inner.tau'].unit == u.ms
    np.testing.assert_allclose(res.best_params['inner.tau'].to_decimal(u.ms), 8., rtol=1e-3)
    np.testing.assert_allclose(model.inner.tau.value().to_decimal(u.ms), 8., rtol=1e-3)


def test_module_scipy_and_nevergrad_backends():
    pytest.importorskip('nevergrad')
    model = Decay()
    res = Fitter.from_module(model, predict, mse, backend='scipy').fit(TARGET, n_starts=2,
                                                                       key=jax.random.PRNGKey(1))
    np.testing.assert_allclose(float(model.tau.value()), 0.3, rtol=1e-3)
    assert res.backend == 'scipy'
    model = Decay()
    res = Fitter.from_module(model, predict, mse, backend='nevergrad').fit(
        TARGET, n_starts=16, key=jax.random.PRNGKey(1), n_steps=40)
    np.testing.assert_allclose(float(model.tau.value()), float(res.best_params['tau']), rtol=1e-4)
    np.testing.assert_allclose(float(res.best_params['tau']), 0.3, rtol=0.05)


def test_module_search_space_and_unbounded_transforms():
    class Mixed(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = brainstate.nn.Param(1.)  # IdentityT: no box
            self.b = brainstate.nn.Param(0.5, t=brainstate.nn.ClipT(0., 1.))
            self.c = brainstate.nn.Param(1., t=brainstate.nn.SoftplusT(0.))

    pred = lambda m: m.a.value() * T + m.b.value() + m.c.value() * T ** 2
    target = 0.7 * T + 0.2 + 1.5 * T ** 2
    model = Mixed()
    fitter = Fitter.from_module(model, pred, mse, search_space={'c': (0.1, 3.)})
    boxes = {s.name: s.box for s in fitter._specs}
    assert boxes['a'] is None
    np.testing.assert_allclose([float(boxes['b'][0]), float(boxes['b'][1])], [0., 1.])
    np.testing.assert_allclose([float(boxes['c'][0]), float(boxes['c'][1])], [0.1, 3.])
    res = fitter.fit(target, n_starts=3, key=jax.random.PRNGKey(0), n_steps=300)
    np.testing.assert_allclose(float(res.best_params['a']), 0.7, atol=1e-2)
    np.testing.assert_allclose(float(res.best_params['b']), 0.2, atol=1e-2)
    np.testing.assert_allclose(float(res.best_params['c']), 1.5, atol=1e-2)
    with pytest.raises(ValueError, match='finite'):
        Fitter.from_module(Mixed(), pred, mse, backend='nevergrad').fit(target, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match='unknown'):
        Fitter.from_module(Mixed(), pred, mse, search_space={'zzz': (0., 1.)})
    with pytest.raises(ValueError, match='pair'):
        Fitter.from_module(Mixed(), pred, mse, search_space={'a': 1.})


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_constructor_validation():
    with pytest.raises(ValueError, match='backend'):
        Fitter(decay, mse, SPACE, backend='sgd')
    with pytest.raises(TypeError, match='simulate'):
        Fitter(1, mse, SPACE)
    with pytest.raises(TypeError, match='objective'):
        Fitter(decay, None, SPACE)
    with pytest.raises(TypeError, match='predict'):
        Fitter.from_module(Decay(), None, mse)
    with pytest.raises(ValueError, match='space'):
        Fitter(decay, mse, {})
    with pytest.raises(ValueError, match='init'):
        Fitter(decay, mse, {'a': None})
    with pytest.raises(ValueError, match='init'):
        Fitter(decay, mse, {'a': (0., None)})
    with pytest.raises(ValueError, match='unknown'):
        Fitter(decay, mse, SPACE, init={'zzz': 1.})
    with pytest.raises(ValueError, match='pair'):
        Fitter(decay, mse, {'a': [0., 1.]})
    with pytest.raises(ValueError, match='transient'):
        Fitter(decay, mse, SPACE, transient=-1)
    with pytest.raises(TypeError, match='optimizer'):
        Fitter(decay, mse, SPACE, optimizer=3)
    with pytest.raises(TypeError, match='optimizer'):
        Fitter(decay, mse, SPACE, backend='scipy', optimizer=optax.adam(0.1))
    with pytest.raises(TypeError, match='unit'):
        Fitter(decay, mse, {'a': (0. * u.mV, 1.), 'tau': (0.1, 1.)})


def test_fit_validation():
    fitter = Fitter(decay, mse, SPACE)
    with pytest.raises(ValueError, match='n_starts'):
        fitter.fit(TARGET, n_starts=0)
    with pytest.raises(ValueError, match='n_steps'):
        fitter.fit(TARGET, n_steps=True)
    with pytest.raises(ValueError, match='keys'):
        fitter.fit(TARGET, x0={'a': jnp.ones(2)})
    with pytest.raises(ValueError, match='shape'):
        fitter.fit(TARGET, x0={'a': jnp.ones(()), 'tau': jnp.ones(())})
    with pytest.raises(ValueError, match='same number'):
        fitter.fit(TARGET, x0={'a': jnp.ones(2), 'tau': 0.5 * jnp.ones(3)})
    with pytest.raises(ValueError, match='Unknown options'):
        Fitter(decay, mse, SPACE, optimizer={'beta': 1.}).fit(TARGET)
    with pytest.raises(TypeError, match='dimensionless'):
        Fitter(decay, lambda p, y: mse(p, y) * u.mV, SPACE).fit(TARGET)


# ---------------------------------------------------------------------------
# exit criteria: parameter recovery with n_starts=8
# ---------------------------------------------------------------------------

def _hopf_rhs(y, t, a, omega):
    x, v = y
    r2 = x * x + v * v
    return (a - r2) * x - omega * v, (a - r2) * v + omega * x


def _simulate_hopf(params, dt=0.05, n_steps=200):
    """Stuart-Landau (Hopf normal form) oscillator; returns the x trace."""

    def step(y, k):
        with brainstate.environ.context(dt=dt):
            y = ode_rk4_step(_hopf_rhs, y, k * dt, params['a'], params['omega'])
        return y, y[0]

    _, xs = jax.lax.scan(step, (jnp.asarray(0.5), jnp.asarray(0.)), jnp.arange(n_steps))
    return xs


def test_recover_hopf_oscillator():
    truth = {'a': 0.8, 'omega': 1.5}
    target = _simulate_hopf(truth)
    start = time.perf_counter()
    res = Fitter(_simulate_hopf, mse, {'a': (0.2, 2.0), 'omega': (0.5, 3.0)}).fit(
        target, n_starts=8, key=jax.random.PRNGKey(0), n_steps=200)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.
    np.testing.assert_allclose(float(res.best_params['a']), truth['a'], rtol=0.05)
    np.testing.assert_allclose(float(res.best_params['omega']), truth['omega'], rtol=0.05)


def _wc_sigmoid(x, gain, theta):
    return 1. / (1. + jnp.exp(-gain * (x - theta)))


class WilsonCowan(brainstate.nn.Module):
    """Two-population Wilson-Cowan rate model driven by a sinusoidal input."""

    def __init__(self, w_ee, w_ei, dt=0.1, n_steps=400):
        super().__init__()
        self.w_ee = brainstate.nn.Param(w_ee, t=brainstate.nn.SigmoidT(5., 20.))
        self.w_ei = brainstate.nn.Param(w_ei, t=brainstate.nn.SigmoidT(5., 20.))
        self.w_ie, self.w_ii = 13., 11.
        self.tau_e, self.tau_i = 1., 2.
        self.dt, self.n_steps = dt, n_steps

    def rhs(self, y, t, w_ee, w_ei):
        e, i = y
        drive = 1.5 + jnp.sin(0.5 * t)
        de = (-e + _wc_sigmoid(w_ee * e - w_ei * i + drive, 1.3, 4.)) / self.tau_e
        di = (-i + _wc_sigmoid(self.w_ie * e - self.w_ii * i, 2., 3.7)) / self.tau_i
        return de, di

    def rates(self):
        w_ee, w_ei = self.w_ee.value(), self.w_ei.value()

        def step(y, k):
            with brainstate.environ.context(dt=self.dt):
                y = ode_rk4_step(self.rhs, y, k * self.dt, w_ee, w_ei)
            return y, jnp.stack(y)

        _, rates = jax.lax.scan(step, (jnp.asarray(0.1), jnp.asarray(0.05)), jnp.arange(self.n_steps))
        return rates


def test_recover_wilson_cowan_couplings():
    target = WilsonCowan(16., 12.).rates()
    model = WilsonCowan(10., 8.)
    start = time.perf_counter()
    res = Fitter.from_module(model, lambda m: m.rates(), mse).fit(
        target, n_starts=8, key=jax.random.PRNGKey(0), n_steps=200)
    elapsed = time.perf_counter() - start
    assert elapsed < 60.
    np.testing.assert_allclose(float(res.best_params['w_ee']), 16., rtol=0.05)
    np.testing.assert_allclose(float(res.best_params['w_ei']), 12., rtol=0.05)
    np.testing.assert_allclose(float(model.w_ee.value()), float(res.best_params['w_ee']), rtol=1e-5)


def test_module_with_other_states_and_array_param_nevergrad():
    pytest.importorskip('nevergrad')

    class Lin(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = brainstate.nn.Param(jnp.zeros(2), t=brainstate.nn.SigmoidT(-2., 2.))
            self.offset = brainstate.State(jnp.asarray(0.25))  # read, not fitted

        def __call__(self):
            return self.w.value()[0] * T + self.w.value()[1] + self.offset.value

    target = 0.8 * T - 0.5 + 0.25
    model = Lin()
    res = Fitter.from_module(model, lambda m: m(), mse).fit(target, n_starts=2, key=jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.asarray(res.best_params['w']), [0.8, -0.5], atol=1e-3)
    assert float(model.offset.value) == 0.25
    res = Fitter.from_module(Lin(), lambda m: m(), mse, backend='nevergrad').fit(
        target, n_starts=16, key=jax.random.PRNGKey(0), n_steps=60)
    assert res.params['w'].shape == (16, 2)
    np.testing.assert_allclose(np.asarray(res.best_params['w']), [0.8, -0.5], atol=0.1)


def test_dimensionless_quantity_objective_is_accepted():
    res = Fitter(decay, lambda p, y: mse(p, y) * u.Quantity(1.), SPACE).fit(TARGET)
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-3)


def test_start_on_bound_moves_in_fitter():
    # explicit starts exactly on the box bounds must not stall
    x0 = {'a': jnp.array([0., 5.]), 'tau': jnp.array([0.05, 1.])}
    res = Fitter(decay, mse, SPACE).fit(TARGET, x0=x0, n_steps=200)
    np.testing.assert_allclose(np.asarray(res.params['a']), 2., rtol=1e-2)
    np.testing.assert_allclose(np.asarray(res.params['tau']), 0.3, rtol=1e-2)


def test_module_param_at_transform_bound_moves():
    model = Decay(a=5., tau=1.)  # both at the upper bound of their SigmoidT
    res = Fitter.from_module(model, predict, mse).fit(TARGET, n_steps=200)
    np.testing.assert_allclose(float(res.best_params['a']), 2., rtol=1e-2)


def test_nevergrad_does_not_consume_global_numpy_rng():
    pytest.importorskip('nevergrad')
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    fitter = Fitter(decay, mse, SPACE, backend='nevergrad')
    r1 = fitter.fit(TARGET, n_starts=4, key=jax.random.PRNGKey(0), n_steps=3)
    assert np.random.rand() == expected
    # reproducible from the key alone, whatever the global generator holds
    np.random.seed(7)
    r2 = fitter.fit(TARGET, n_starts=4, key=jax.random.PRNGKey(0), n_steps=3)
    np.testing.assert_array_equal(np.asarray(r1.history), np.asarray(r2.history))
