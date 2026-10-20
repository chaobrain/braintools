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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintools.optim._base import Optimizer, OptimState
from braintools.optim._online import FORCE, LMS, RLS, _OnlineReadout


@pytest.fixture
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


def make_problem(n=120, n_in=5, n_out=2, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_in))
    W = rng.normal(size=(n_in, n_out))
    T = X @ W + noise * rng.normal(size=(n, n_out))
    return jnp.asarray(X), jnp.asarray(T), W


def run_batches(opt, w, X, T, batch):
    """Feed (X, T) to ``opt`` in consecutive batches inside a for_loop."""
    n = X.shape[0] // batch
    Xb = X[:n * batch].reshape(n, batch, X.shape[1])
    Tb = T[:n * batch].reshape(n, batch, T.shape[1])
    return brainstate.transform.for_loop(lambda x, t: opt.step(x, None, t), Xb, Tb)


# ---------------------------------------------------------------------------
# RLS
# ---------------------------------------------------------------------------

def test_rls_is_optimizer():
    opt = RLS()
    assert isinstance(opt, Optimizer)
    assert RLS.__module__ == 'braintools.optim'
    assert FORCE.__module__ == 'braintools.optim'
    assert LMS.__module__ == 'braintools.optim'


def test_rls_large_p0_converges_to_least_squares(x64):
    X, T, _ = make_problem()
    w = brainstate.ParamState(jnp.zeros((5, 2)))
    opt = RLS(alpha=1e-8).register_trainable_weights({'readout': w})
    run_batches(opt, w, X, T, 1)
    lstsq = np.linalg.lstsq(np.asarray(X), np.asarray(T), rcond=None)[0]
    np.testing.assert_allclose(w.value, lstsq, atol=1e-6)


def test_rls_equals_ridge_without_intercept(x64):
    X, T, _ = make_problem()
    alpha = 3.0
    w = brainstate.ParamState(jnp.zeros((5, 2)))
    opt = RLS(alpha=alpha).register_trainable_weights({'readout': w})
    run_batches(opt, w, X, T, 4)
    Xn, Tn = np.asarray(X), np.asarray(T)
    ridge = np.linalg.solve(Xn.T @ Xn + alpha * np.eye(5), Xn.T @ Tn)
    np.testing.assert_allclose(w.value, ridge, atol=1e-10)
    # P is the inverse of the regularised input correlation.
    np.testing.assert_allclose(opt.P['readout'].value, np.linalg.inv(Xn.T @ Xn + alpha * np.eye(5)),
                               atol=1e-10)


@pytest.mark.parametrize('batch', [4, 10, 30])
def test_rls_batch_size_equivalence(x64, batch):
    X, T, _ = make_problem()
    w1 = brainstate.ParamState(jnp.zeros((5, 2)))
    opt1 = RLS(alpha=0.5).register_trainable_weights({'w': w1})
    run_batches(opt1, w1, X, T, 1)
    wb = brainstate.ParamState(jnp.zeros((5, 2)))
    optb = RLS(alpha=0.5).register_trainable_weights({'w': wb})
    run_batches(optb, wb, X, T, batch)
    np.testing.assert_allclose(wb.value, w1.value, atol=1e-10)
    np.testing.assert_allclose(optb.P['w'].value, opt1.P['w'].value, atol=1e-10)


def test_rls_forgetting_is_exponentially_weighted_ridge(x64):
    X, T, _ = make_problem(n=60)
    alpha, lam, B = 2.0, 0.9, 6
    w = brainstate.ParamState(jnp.zeros((5, 2)))
    opt = RLS(alpha=alpha, forgetting=lam).register_trainable_weights({'w': w})
    run_batches(opt, w, X, T, B)
    K = 60 // B
    weights = np.repeat(lam ** np.arange(K - 1, -1, -1), B)  # one factor per batch
    Xn, Tn = np.asarray(X), np.asarray(T)
    A = (Xn * weights[:, None]).T @ Xn + lam ** K * alpha * np.eye(5)
    ref = np.linalg.solve(A, (Xn * weights[:, None]).T @ Tn)
    np.testing.assert_allclose(w.value, ref, atol=1e-9)


def test_rls_tracks_changing_target_with_forgetting(x64):
    rng = np.random.default_rng(1)
    X = jnp.asarray(rng.normal(size=(400, 3)))
    W1, W2 = np.array([[1.], [0.], [-1.]]), np.array([[-2.], [1.], [0.5]])
    T = jnp.concatenate([X[:200] @ W1, X[200:] @ W2])
    for lam, close in ((0.95, True), (1.0, False)):
        w = brainstate.ParamState(jnp.zeros((3, 1)))
        opt = RLS(alpha=1.0, forgetting=lam).register_trainable_weights({'w': w})
        run_batches(opt, w, X, T, 1)
        assert np.allclose(w.value, W2, atol=1e-3) == close


def test_rls_outputs_argument_matches_none(x64):
    X, T, _ = make_problem(n=20)
    wa = brainstate.ParamState(jnp.ones((5, 2)))
    wb = brainstate.ParamState(jnp.ones((5, 2)))
    oa = RLS().register_trainable_weights({'w': wa})
    ob = RLS().register_trainable_weights({'w': wb})
    ea = oa.step(X[:4], X[:4] @ wa.value, T[:4])
    eb = ob.step(X[:4], None, T[:4])
    np.testing.assert_allclose(ea, eb)
    np.testing.assert_allclose(ea, X[:4] @ jnp.ones((5, 2)) - T[:4])  # a-priori error
    np.testing.assert_allclose(wa.value, wb.value)


def test_rls_single_sample_1d_inputs(x64):
    X, T, _ = make_problem(n=10)
    wa = brainstate.ParamState(jnp.zeros((5, 2)))
    wb = brainstate.ParamState(jnp.zeros((5, 2)))
    oa = RLS().register_trainable_weights({'w': wa})
    ob = RLS().register_trainable_weights({'w': wb})
    e = oa.step(X[0], None, T[0])
    assert e.shape == (2,)
    eb = ob.step(X[:1], None, T[:1])
    assert eb.shape == (1, 2)
    np.testing.assert_allclose(wa.value, wb.value)


def test_rls_jit_matches_eager(x64):
    X, T, _ = make_problem(n=12)
    wa = brainstate.ParamState(jnp.zeros((5, 2)))
    wb = brainstate.ParamState(jnp.zeros((5, 2)))
    oa = RLS(alpha=0.3).register_trainable_weights({'w': wa})
    ob = RLS(alpha=0.3).register_trainable_weights({'w': wb})
    step = brainstate.transform.jit(lambda x, t: ob.step(x, None, t))
    for i in range(3):
        oa.step(X[4 * i:4 * i + 4], None, T[4 * i:4 * i + 4])
        step(X[4 * i:4 * i + 4], T[4 * i:4 * i + 4])
    np.testing.assert_allclose(wb.value, wa.value, atol=1e-12)
    np.testing.assert_allclose(ob.P['w'].value, oa.P['w'].value, atol=1e-12)


def test_rls_float32_weights_keep_dtype():
    X, T, _ = make_problem(n=8)
    w = brainstate.ParamState(jnp.zeros((5, 2), jnp.float32))
    opt = RLS().register_trainable_weights({'w': w})
    opt.step(X.astype(jnp.float32), None, T.astype(jnp.float32))
    assert w.value.dtype == jnp.float32
    assert opt.P['w'].value.dtype == jnp.float32


def test_rls_p_is_optim_state_and_reregistration_resets(x64):
    X, T, _ = make_problem(n=8)
    w = brainstate.ParamState(jnp.zeros((5, 2)))
    opt = RLS(alpha=2.).register_trainable_weights({'w': w})
    assert isinstance(opt.P['w'], OptimState)
    np.testing.assert_allclose(opt.P['w'].value, np.eye(5) / 2.)
    opt.step(X, None, T)
    assert not np.allclose(opt.P['w'].value, np.eye(5) / 2.)
    opt.register_trainable_weights({'w': w})
    np.testing.assert_allclose(opt.P['w'].value, np.eye(5) / 2.)


# ---------------------------------------------------------------------------
# several weights
# ---------------------------------------------------------------------------

def test_dict_inputs_train_independent_readouts(x64):
    X, T, _ = make_problem()
    Z = jnp.asarray(np.random.default_rng(3).normal(size=(120, 3)))
    TZ = Z @ jnp.array([[1.], [2.], [3.]])
    wa = brainstate.ParamState(jnp.zeros((5, 2)))
    wb = brainstate.ParamState(jnp.zeros((3, 1)))
    opt = RLS(alpha=1e-8).register_trainable_weights({'a': wa, 'b': wb})

    def body(x, t, z, tz):
        return opt.step({'a': x, 'b': z}, None, {'a': t, 'b': tz})

    errs = brainstate.transform.for_loop(body, X[:, None], T[:, None], Z[:, None], TZ[:, None])
    assert set(errs) == {'a', 'b'}
    assert errs['a'].shape == (120, 1, 2) and errs['b'].shape == (120, 1, 1)
    np.testing.assert_allclose(wa.value, np.linalg.lstsq(X, T, rcond=None)[0], atol=1e-6)
    np.testing.assert_allclose(wb.value, [[1.], [2.], [3.]], atol=1e-6)


def test_dict_subset_updates_only_given_weights(x64):
    X, T, _ = make_problem(n=8)
    wa = brainstate.ParamState(jnp.zeros((5, 2)))
    wb = brainstate.ParamState(jnp.zeros((5, 2)))
    opt = LMS(lr=0.01).register_trainable_weights({'a': wa, 'b': wb})
    opt.step({'a': X}, {'a': None}, {'a': T})
    assert not np.allclose(wa.value, 0.)
    np.testing.assert_array_equal(wb.value, 0.)


def test_multiple_weights_require_dict():
    wa = brainstate.ParamState(jnp.zeros((2, 1)))
    wb = brainstate.ParamState(jnp.zeros((2, 1)))
    opt = RLS().register_trainable_weights({'a': wa, 'b': wb})
    with pytest.raises(ValueError, match='must be a dict'):
        opt.step(jnp.ones((1, 2)), None, jnp.ones((1, 1)))
    with pytest.raises(ValueError, match='Unknown weight'):
        opt.step({'c': jnp.ones((1, 2))}, None, {'c': jnp.ones((1, 1))})
    with pytest.raises(ValueError, match='must contain the key'):
        opt.step({'a': jnp.ones((1, 2))}, None, {'b': jnp.ones((1, 1))})


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_constructor_validation():
    for bad in (0., -1., 'x'):
        with pytest.raises(ValueError, match='alpha'):
            RLS(alpha=bad)
    for bad in (0., 1.5, -0.1):
        with pytest.raises(ValueError, match='forgetting'):
            RLS(forgetting=bad)
    with pytest.raises(ValueError, match='alpha'):
        FORCE(alpha=-1.)
    with pytest.raises(ValueError, match='lr'):
        LMS(lr=-0.1)
    with pytest.raises(ValueError, match='lr'):
        LMS(lr=None)


@pytest.mark.parametrize('cls', [RLS, LMS])
def test_registration_validation(cls):
    opt = cls()
    with pytest.raises(TypeError, match='dict'):
        opt.register_trainable_weights([brainstate.ParamState(jnp.zeros((2, 1)))])
    with pytest.raises(TypeError, match='brainstate.State'):
        opt.register_trainable_weights({'w': jnp.zeros((2, 1))})
    with pytest.raises(ValueError, match='n_in, n_out'):
        opt.register_trainable_weights({'w': brainstate.ParamState(jnp.zeros(3))})
    with pytest.raises(TypeError, match='unit'):
        opt.register_trainable_weights({'w': brainstate.ParamState(jnp.zeros((2, 1)) * u.mV)})
    s = brainstate.ParamState(jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match='already registered'):
        opt.register_trainable_weights({'a': s, 'b': s})


@pytest.mark.parametrize('cls', [RLS, LMS])
def test_step_validation(cls):
    opt = cls()
    with pytest.raises(ValueError, match='register_trainable_weights'):
        opt.step(jnp.ones((1, 2)), None, jnp.ones((1, 1)))
    opt.register_trainable_weights({'w': brainstate.ParamState(jnp.zeros((2, 1)))})
    with pytest.raises(ValueError, match='inputs'):
        opt.step(jnp.ones((1, 3)), None, jnp.ones((1, 1)))
    with pytest.raises(ValueError, match='inputs'):
        opt.step(jnp.ones((1, 1, 2)), None, jnp.ones((1, 1)))
    with pytest.raises(ValueError, match='targets'):
        opt.step(jnp.ones((2, 2)), None, jnp.ones((1, 1)))
    with pytest.raises(ValueError, match='outputs'):
        opt.step(jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 1)))
    with pytest.raises(u.UnitMismatchError):
        opt.step(jnp.ones((1, 2)) * u.mV, None, jnp.ones((1, 1)))
    # dimensionless Quantities are accepted
    err = opt.step(u.Quantity(jnp.ones((1, 2))), None, u.Quantity(jnp.ones((1, 1))))
    assert err.shape == (1, 1)


def test_base_readout_delta_not_implemented():
    opt = _OnlineReadout()
    opt.register_trainable_weights({'w': brainstate.ParamState(jnp.zeros((2, 1)))})
    with pytest.raises(NotImplementedError):
        opt.step(jnp.ones((1, 2)), None, jnp.ones((1, 1)))


@pytest.mark.parametrize('cls', [RLS, LMS])
def test_update_is_alias_of_step(x64, cls):
    X, T, _ = make_problem(n=6)
    wa = brainstate.ParamState(jnp.zeros((5, 2)))
    wb = brainstate.ParamState(jnp.zeros((5, 2)))
    oa = cls().register_trainable_weights({'w': wa})
    ob = cls().register_trainable_weights({'w': wb})
    np.testing.assert_allclose(oa.update(X, None, T), ob.step(X, None, T))
    np.testing.assert_allclose(wa.value, wb.value)


# ---------------------------------------------------------------------------
# LMS
# ---------------------------------------------------------------------------

def test_lms_single_step_formula(x64):
    X, T, _ = make_problem(n=4)
    W0 = jnp.asarray(np.random.default_rng(2).normal(size=(5, 2)))
    w = brainstate.ParamState(W0)
    opt = LMS(lr=0.05).register_trainable_weights({'w': w})
    E = opt.step(X, None, T)
    np.testing.assert_allclose(E, X @ W0 - T)
    np.testing.assert_allclose(w.value, W0 - 0.05 * X.T @ (X @ W0 - T), atol=1e-12)


def test_lms_converges_for_small_lr(x64):
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(3000, 4)))
    W = rng.normal(size=(4, 1))
    T = X @ W
    w = brainstate.ParamState(jnp.zeros((4, 1)))
    opt = LMS(lr=0.01).register_trainable_weights({'w': w})
    errs = run_batches(opt, w, X, T, 1)
    np.testing.assert_allclose(w.value, W, atol=1e-6)
    assert float(jnp.abs(errs[-1]).max()) < 1e-6 < float(jnp.abs(errs[0]).max())


def test_lms_diverges_for_large_lr(x64):
    X, T, _ = make_problem(n=50, noise=0.)
    w = brainstate.ParamState(jnp.zeros((5, 2)))
    opt = LMS(lr=5.).register_trainable_weights({'w': w})
    errs = run_batches(opt, w, X, T, 1)
    assert not bool(jnp.all(jnp.abs(errs[-1]) < 1.))


# ---------------------------------------------------------------------------
# FORCE
# ---------------------------------------------------------------------------

def test_force_learns_sine_with_chaotic_rate_network():
    N, g, dt, tau = 200, 1.5, 1.0, 10.0
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
    J = g * jax.random.normal(k1, (N, N)) / jnp.sqrt(N)
    w_fb = jax.random.uniform(k2, (N,), minval=-1., maxval=1.)
    x = brainstate.HiddenState(0.5 * jax.random.normal(k3, (N,)))
    w = brainstate.ParamState(jnp.zeros((N, 1)))
    opt = FORCE(alpha=1.0).register_trainable_weights({'readout': w})

    def run(target, train):
        r = jnp.tanh(x.value)
        z = r @ w.value
        x.value = x.value + dt / tau * (-x.value + J @ r + w_fb * z[0])
        if train:
            opt.step(r[None], z[None], target[None])
        return z[0], r

    n_train, n_test = 4000, 1000
    t = jnp.arange(n_train + n_test) * dt
    target = jnp.sin(2 * jnp.pi * t / 100.)[:, None]
    # Before training the network is chaotic: rates keep fluctuating.
    _, rates = brainstate.transform.for_loop(lambda y: run(y, False), target[:300])
    assert float(jnp.std(rates[-100:], axis=0).mean()) > 0.1
    z_train, _ = brainstate.transform.for_loop(lambda y: run(y, True), target[:n_train])
    z_test, _ = brainstate.transform.for_loop(lambda y: run(y, False), target[n_train:])
    train_mse = float(jnp.mean((z_train[-500:] - target[n_train - 500:n_train, 0]) ** 2))
    test_mse = float(jnp.mean((z_test - target[n_train:, 0]) ** 2))
    assert train_mse < 0.01
    assert test_mse < 0.05
    assert opt.P['readout'].value.shape == (N, N)
