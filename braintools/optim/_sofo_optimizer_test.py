# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# ==============================================================================

import jax
import jax.numpy as jnp
import pytest

import brainstate
import braintools


class MLP(brainstate.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.l1 = brainstate.nn.Linear(n_in, n_hidden)
        self.l2 = brainstate.nn.Linear(n_hidden, n_out)

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))


def _fixed_regression_batch(n=32, n_in=8, n_out=3, seed=0):
    rng = brainstate.random.RandomState(seed)
    x = rng.randn(n, n_in)
    y = rng.randn(n, n_out)
    return x, y


def test_sofo_reduces_mse_loss():
    brainstate.random.seed(0)
    model = MLP(8, 16, 3)
    x, y = _fixed_regression_batch()
    loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
    opt = braintools.optim.SOFO(model, loss_fn, lr=2e-2, loss='mse',
                                tangent_size=64, damping=1e-5, key=jax.random.PRNGKey(0))
    opt.register_trainable_weights(model.states(brainstate.ParamState))

    @brainstate.transform.jit
    def step(bx, by):
        return opt.step(bx, by)

    first = float(step(x, y))
    last = first
    for _ in range(20):
        last = float(step(x, y))
    assert last < first


def test_sofo_reduces_ce_loss():
    brainstate.random.seed(1)
    model = MLP(8, 16, 4)
    rng = brainstate.random.RandomState(2)
    x = rng.randn(40, 8)
    y = rng.randint(0, 4, (40,))
    loss_fn = lambda pred, target: braintools.metric.softmax_cross_entropy_with_integer_labels(pred, target).mean()
    opt = braintools.optim.SOFO(model, loss_fn, lr=2e-2, loss='ce',
                                tangent_size=64, damping=1e-6, key=jax.random.PRNGKey(0))
    opt.register_trainable_weights(model.states(brainstate.ParamState))

    @brainstate.transform.jit
    def step(bx, by):
        return opt.step(bx, by)

    first = float(step(x, y))
    last = first
    for _ in range(30):
        last = float(step(x, y))
    assert last < first


def test_sofo_direction_structure_and_finite():
    brainstate.random.seed(0)
    model = MLP(8, 16, 3)
    x, y = _fixed_regression_batch()
    loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
    opt = braintools.optim.SOFO(model, loss_fn, lr=1e-2, tangent_size=32, key=jax.random.PRNGKey(0))
    params = model.states(brainstate.ParamState)
    opt.register_trainable_weights(params)

    grads, preds = opt._compute_direction(x, y)
    assert set(grads.keys()) == set(params.keys())
    assert preds.shape == y.shape
    for leaf in jax.tree.leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))


def test_sofo_step_before_register_raises():
    model = MLP(4, 8, 2)
    loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
    opt = braintools.optim.SOFO(model, loss_fn)
    with pytest.raises(ValueError):
        opt.step(jnp.zeros((2, 4)), jnp.zeros((2, 2)))


def test_sofo_unknown_loss_raises():
    brainstate.random.seed(0)
    model = MLP(4, 8, 2)
    x, y = _fixed_regression_batch(n=8, n_in=4, n_out=2)
    loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)
    opt = braintools.optim.SOFO(model, loss_fn, loss='bad', key=jax.random.PRNGKey(0))
    opt.register_trainable_weights(model.states(brainstate.ParamState))
    with pytest.raises(ValueError):
        opt.step(x, y)


def test_sofo_deterministic_with_fixed_key():
    # Determinism holds for a fixed set of parameter States: same instance + same key ->
    # identical direction (the internal grad dict is keyed by id(State), so cross-instance
    # ordering is not guaranteed -- this is inherent to the design).
    brainstate.random.seed(7)
    model = MLP(6, 10, 2)
    x, y = _fixed_regression_batch(n=16, n_in=6, n_out=2)
    loss_fn = lambda pred, target: jnp.mean((pred - target) ** 2)

    opt = braintools.optim.SOFO(model, loss_fn, tangent_size=32, key=jax.random.PRNGKey(123))
    opt.register_trainable_weights(model.states(brainstate.ParamState))

    g1, _ = opt._compute_direction(x, y)
    g2, _ = opt._compute_direction(x, y)
    l1, l2 = jax.tree.leaves(g1), jax.tree.leaves(g2)
    assert len(l1) == len(l2) and len(l1) > 0
    for a, b in zip(l1, l2):
        assert jnp.allclose(a, b)

    # a different key produces a different direction (same parameter States)
    opt2 = braintools.optim.SOFO(model, loss_fn, tangent_size=32, key=jax.random.PRNGKey(999))
    opt2.register_trainable_weights(model.states(brainstate.ParamState))
    g3, _ = opt2._compute_direction(x, y)
    l3 = jax.tree.leaves(g3)
    assert any(not jnp.allclose(a, b) for a, b in zip(l1, l3))


# ---------------------------------------------------------------------------
# SOFOScan
# ---------------------------------------------------------------------------

class LinearRNN(brainstate.nn.Module):
    """h_t = W_h h_{t-1} + W_x x_t ; output = W_o h_t. Latent passed explicitly.

    Uses default (affine) Linear layers -- the mse GGN is identity regardless of the bias, so
    the direction test is unaffected. Avoids relying on a bias-disable kwarg whose name may
    differ across brainstate versions.
    """

    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.wh = brainstate.nn.Linear(n_hidden, n_hidden)
        self.wx = brainstate.nn.Linear(n_in, n_hidden)
        self.wo = brainstate.nn.Linear(n_hidden, n_out)

    def __call__(self, latent, inputs):
        new_latent = self.wh(latent) + self.wx(inputs)
        output = self.wo(new_latent)
        return new_latent, output


def _seq_batch(T=5, batch=4, n_in=3, n_out=2, seed=0):
    rng = brainstate.random.RandomState(seed)
    xs = rng.randn(T, batch, n_in)
    ys = rng.randn(T, batch, n_out)
    return xs, ys


def test_sofoscan_reduces_loss():
    brainstate.random.seed(0)
    n_in, n_hidden, n_out, batch = 3, 6, 2, 4
    cell = LinearRNN(n_in, n_hidden, n_out)
    loss_fn = lambda out, label: jnp.mean((out - label) ** 2)
    opt = braintools.optim.SOFOScan(cell, loss_fn, lr=1e-2, loss='mse',
                                    tangent_size=64, damping=1e-5, key=jax.random.PRNGKey(0))
    opt.register_trainable_weights(cell.states(brainstate.ParamState))
    xs, ys = _seq_batch(T=5, batch=batch, n_in=n_in, n_out=n_out)
    z0 = jnp.zeros((batch, n_hidden))

    @brainstate.transform.jit
    def step(z, b):
        return opt.step(z, b)

    first = float(step(z0, (xs, ys)))
    last = first
    for _ in range(30):
        last = float(step(z0, (xs, ys)))
    assert last < first


def test_sofoscan_direction_matches_gradient_under_high_damping():
    """In the high-damping limit SOFO reduces to (subspace-projected) gradient descent, so its
    direction must align with the true gradient. This validates the scan + forward-mode-JVP math
    (at low damping the natural-gradient direction differs from the plain gradient by the GGN
    preconditioner, so it is not a good oracle)."""
    brainstate.random.seed(3)
    n_in, n_hidden, n_out, batch = 3, 5, 2, 4
    cell = LinearRNN(n_in, n_hidden, n_out)
    loss_fn = lambda out, label: jnp.mean((out - label) ** 2)
    opt = braintools.optim.SOFOScan(cell, loss_fn, lr=1e-2, loss='mse',
                                    tangent_size=400, damping=10.0, key=jax.random.PRNGKey(0))
    params = cell.states(brainstate.ParamState)
    opt.register_trainable_weights(params)
    xs, ys = _seq_batch(T=4, batch=batch, n_in=n_in, n_out=n_out, seed=5)
    z0 = jnp.zeros((batch, n_hidden))

    h, _ = opt._compute_direction(z0, (xs, ys))

    # true gradient of the total (collapsed) scan loss via reverse-mode
    def total_loss():
        latent = z0
        outs = []
        for t in range(xs.shape[0]):
            latent, out = cell(latent, xs[t])
            outs.append(out)
        preds = jnp.concatenate(outs, axis=0)
        return jnp.mean((preds - ys.reshape(-1, ys.shape[-1])) ** 2)

    true_grads = brainstate.transform.grad(total_loss, grad_states=params)()

    lh, lt = jax.tree.leaves(h), jax.tree.leaves(true_grads)
    dot = sum(jnp.sum(a * b) for a, b in zip(lh, lt))
    nh = jnp.sqrt(sum(jnp.sum(a ** 2) for a in lh))
    ng = jnp.sqrt(sum(jnp.sum(b ** 2) for b in lt))
    cos = float(dot / (nh * ng + 1e-12))
    assert cos > 0.7


def test_sofoscan_step_before_register_raises():
    cell = LinearRNN(3, 4, 2)
    loss_fn = lambda out, label: jnp.mean((out - label) ** 2)
    opt = braintools.optim.SOFOScan(cell, loss_fn)
    xs = jnp.zeros((2, 4, 3))
    ys = jnp.zeros((2, 4, 2))
    with pytest.raises(ValueError):
        opt.step(jnp.zeros((4, 4)), (xs, ys))
