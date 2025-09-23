import types

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest


def _patch_scipy(monkeypatch):
    import braintools.optim._scipy_optimizer as so
    # Ensure constructor does not fail on missing SciPy
    monkeypatch.setattr(so, "minimize", object(), raising=False)

    # Replace the SciPy wrapper with a cheap stub that evaluates the objective
    def fake_minimize_with_jax(x0, loss_fun, **kwargs):
        val = float(np.asarray(loss_fun(x0)))
        return types.SimpleNamespace(fun=val, x=x0)

    monkeypatch.setattr(so, "scipy_minimize_with_jax", fake_minimize_with_jax, raising=False)


def test_tuple_bounds_structure(monkeypatch):
    _patch_scipy(monkeypatch)
    from braintools.optim import ScipyOptimizer

    def loss(x, y):
        return (x - 1.0) ** 2 + (y + 2.0) ** 2

    bounds = [(-5.0, 5.0), (-3.0, 3.0)]
    opt = ScipyOptimizer(loss_fun=loss, bounds=bounds, method='L-BFGS-B')
    res = opt.minimize(n_iter=2)

    assert isinstance(res.x, (tuple, list)) and len(res.x) == 2
    assert jnp.shape(res.x[0]) == () and jnp.shape(res.x[1]) == ()


def test_dict_bounds_structure(monkeypatch):
    _patch_scipy(monkeypatch)
    from braintools.optim import ScipyOptimizer

    def loss(**p):
        return (p['a'] - 1.0) ** 2 + (p['b'] + 2.0) ** 2

    bounds = {"a": (-5.0, 5.0), "b": (-3.0, 3.0)}
    opt = ScipyOptimizer(loss_fun=loss, bounds=bounds, method='TNC')
    res = opt.minimize(n_iter=2)

    assert isinstance(res.x, dict)
    assert set(res.x.keys()) == {"a", "b"}
    assert jnp.shape(res.x['a']) == () and jnp.shape(res.x['b']) == ()


def test_array_bounds_and_shapes(monkeypatch):
    _patch_scipy(monkeypatch)
    from braintools.optim import ScipyOptimizer

    def loss(v1, v2):
        return jnp.sum(v1 ** 2) + jnp.sum((v2 - 1.0) ** 2)

    bounds = [
        (jnp.array([-2.0, -1.0]), jnp.array([2.0, 3.0])),
        (jnp.zeros((2,)), jnp.ones((2,)) * 5.0),
    ]
    opt = ScipyOptimizer(loss_fun=loss, bounds=bounds, method='L-BFGS-B')
    res = opt.minimize(n_iter=2)

    assert isinstance(res.x, (tuple, list)) and len(res.x) == 2
    assert res.x[0].shape == (2,)
    assert res.x[1].shape == (2,)


def test_units_conversion_and_validation(monkeypatch):
    _patch_scipy(monkeypatch)
    from braintools.optim import ScipyOptimizer

    def loss(x):
        return jnp.sum(x ** 2)

    # Compatible units (mV vs uV)
    bounds = [(-1.0 * u.mV, 1000.0 * u.mV)]
    opt = ScipyOptimizer(loss_fun=loss, bounds=bounds, method='Powell')
    res = opt.minimize(n_iter=1)
    assert isinstance(res.x, (tuple, list)) and res.x[0].shape == ()

    # Incompatible units should raise during initialization
    with pytest.raises(Exception):
        ScipyOptimizer(loss_fun=loss, bounds=[(1. * u.mV, 1.0 * u.nA)], method='Powell')


def test_bounds_none_raises(monkeypatch):
    _patch_scipy(monkeypatch)
    from braintools.optim import ScipyOptimizer

    def loss(x):
        return x ** 2

    with pytest.raises(ValueError):
        ScipyOptimizer(loss_fun=loss, bounds=None, method='L-BFGS-B')


def test_mismatched_shapes_raise(monkeypatch):
    _patch_scipy(monkeypatch)
    from braintools.optim import ScipyOptimizer

    def loss(x):
        return jnp.sum(x ** 2)

    with pytest.raises(AssertionError):
        ScipyOptimizer(
            loss_fun=loss,
            bounds=[(jnp.array([0.0, 0.0]), jnp.array([1.0]))],
            method='L-BFGS-B',
        )
