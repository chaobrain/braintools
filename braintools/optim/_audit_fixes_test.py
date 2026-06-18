# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Regression tests for the 2026-06-18 ``braintools.optim`` audit fixes.

Each test maps to an issue ID from ``docs/braintools-optim-issues-found-20260618.md``
and asserts the *corrected* behavior (i.e. it fails against the pre-fix code).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import brainstate

import braintools.optim as O


def _applied_lrs(scheduler_factory, n_steps):
    """Effective LR applied to a unit gradient by SGD(momentum=0) each step.

    With momentum=0 and weight_decay=0, ``SGD`` applies ``-lr * grad``; with a unit
    gradient the per-step parameter delta is exactly ``-lr``. This probes the LR that
    is *actually applied to the gradient* (not merely reported by ``current_lr``).
    """
    w = brainstate.ParamState(jnp.zeros((1,)))
    sch = scheduler_factory()
    opt = O.SGD(lr=sch, momentum=0.0)
    opt.register_trainable_weights({'w': w})
    applied = []
    for _ in range(n_steps):
        before = float(w.value[0])
        opt.step({'w': jnp.ones((1,))})
        applied.append(-(float(w.value[0]) - before))
        sch.step()
    return applied


# --------------------------------------------------------------------------- #
# Scheduler fixes
# --------------------------------------------------------------------------- #

class TestSchedulerAppliedLR:
    def test_S3_first_step_uses_scheduled_lr_linear_warmup(self):
        # S-3: the very first applied LR must be the warmup start (0.1), not base_lr.
        applied = _applied_lrs(
            lambda: O.LinearLR(base_lr=1.0, start_factor=0.1, end_factor=1.0, total_iters=5), 6
        )
        assert applied[0] == pytest.approx(0.1, abs=1e-5)
        assert applied[-1] == pytest.approx(1.0, abs=1e-5)
        assert applied == sorted(applied)  # monotonic ramp-up

    def test_S2_S4_sequential_applies_and_is_continuous(self):
        # S-2/S-4: SequentialLR must actually apply the active sub-scheduler's LR,
        # with the milestone-relative epoch (so ExponentialLR starts at its base).
        applied = _applied_lrs(
            lambda: O.SequentialLR(
                schedulers=[O.ConstantLR(base_lr=1.0, factor=1.0, total_iters=0),
                            O.ExponentialLR(base_lr=1.0, gamma=0.5)],
                milestones=[3],
            ), 7
        )
        # first phase: constant 1.0
        assert all(a == pytest.approx(1.0, abs=1e-5) for a in applied[:4])
        # second phase: 0.5**0, 0.5**1, 0.5**2 -> smooth, no discontinuous jump to 0.5**5
        assert applied[4] == pytest.approx(0.5, abs=1e-5)
        assert applied[5] == pytest.approx(0.25, abs=1e-5)
        assert applied[6] == pytest.approx(0.125, abs=1e-5)

    def test_S2_warm_restarts_applied(self):
        # S-2: CosineAnnealingWarmRestarts must apply its schedule (not stay at base_lr).
        applied = _applied_lrs(lambda: O.CosineAnnealingWarmRestarts(base_lr=1.0, T_0=3, eta_min=0.0), 6)
        assert applied[0] == pytest.approx(1.0, abs=1e-5)
        assert applied[1] < applied[0]            # decaying within the cycle
        assert applied[3] == pytest.approx(1.0, abs=1e-5)  # restart back to base

    def test_S6_onecycle_starts_at_initial_lr(self):
        # S-6: at epoch 0 OneCycleLR yields initial_lr = max_lr / div_factor.
        sch = O.OneCycleLR(max_lr=1.0, total_steps=100, div_factor=25.0)
        assert float(sch.get_lr()[0]) == pytest.approx(0.04, abs=1e-6)


class TestSchedulerStructural:
    def test_S5_chained_advances_last_epoch(self):
        # S-5: ChainedScheduler.last_epoch must advance (else epoch-driven loops hang).
        ch = O.ChainedScheduler([O.LinearLR(base_lr=1.0, start_factor=0.1, total_iters=3),
                                 O.StepLR(base_lr=1.0, step_size=2, gamma=0.5)])
        start = int(ch.last_epoch.value)
        for _ in range(4):
            ch.step()
        assert int(ch.last_epoch.value) == start + 4

    def test_S5_chained_combines_multiplicatively(self):
        # S-5: the combined LR is the product of each sub-scheduler's factor.
        a = O.ConstantLR(base_lr=1.0, factor=0.5, total_iters=100)   # factor 0.5
        b = O.ConstantLR(base_lr=1.0, factor=0.2, total_iters=100)   # factor 0.2
        ch = O.ChainedScheduler([a, b])
        assert float(ch.get_lr()[0]) == pytest.approx(0.1, abs=1e-6)  # 1.0 * 0.5 * 0.2

    def test_S8_get_last_lr_exists(self):
        # S-8: both LRScheduler and OptaxOptimizer expose get_last_lr().
        sch = O.StepLR(base_lr=0.1, step_size=30, gamma=0.1)
        assert sch.get_last_lr()[0] == pytest.approx(0.1)
        opt = O.SGD(lr=0.1)
        w = brainstate.ParamState(jnp.zeros((1,)))
        opt.register_trainable_weights({'w': w})
        opt.add_scheduler(O.StepLR(base_lr=0.1, step_size=1, gamma=0.1))
        assert opt.get_last_lr()[0] == pytest.approx(0.1)

    def test_S9_warmup_cosine_multigroup(self):
        # S-9: per-group base_lrs honored (not collapsed to group 0).
        sch = O.WarmupCosineSchedule(base_lr=[1.0, 0.1], warmup_steps=0, total_steps=100)
        lrs = [float(x) for x in sch.get_lr()]
        assert lrs[0] == pytest.approx(1.0, abs=1e-5)
        assert lrs[1] == pytest.approx(0.1, abs=1e-5)

    @pytest.mark.parametrize("factory", [
        lambda: O.StepLR(step_size=0),
        lambda: O.CosineAnnealingLR(T_max=0),
        lambda: O.PolynomialLR(total_iters=0),
        lambda: O.CyclicLR(step_size_up=0),
        lambda: O.OneCycleLR(max_lr=1.0, total_steps=10, pct_start=0.0),
        lambda: O.WarmupScheduler(warmup_epochs=-1),
        lambda: O.CosineAnnealingWarmRestarts(T_0=0),
    ])
    def test_S7_zero_period_raises(self, factory):
        # S-7: degenerate periods raise ValueError instead of ZeroDivisionError/NaN.
        with pytest.raises(ValueError):
            factory()

    def test_S14_cosine_holds_at_eta_min_past_tmax(self):
        # S-14: past T_max the LR holds at eta_min (no oscillation back up).
        sch = O.CosineAnnealingLR(base_lr=1.0, T_max=10, eta_min=0.0)
        sch.last_epoch.value = 10
        assert float(sch.get_lr()[0]) == pytest.approx(0.0, abs=1e-6)
        sch.last_epoch.value = 15
        assert float(sch.get_lr()[0]) == pytest.approx(0.0, abs=1e-6)
        sch.last_epoch.value = 20
        assert float(sch.get_lr()[0]) == pytest.approx(0.0, abs=1e-6)

    def test_S15_plateau_state_roundtrip(self):
        # S-15: ReduceLROnPlateau persists best/num_bad_epochs/cooldown_counter.
        sch = O.ReduceLROnPlateau(base_lr=1.0, patience=1, factor=0.5)
        for m in [1.0, 1.0, 1.0, 1.0]:
            sch.step(m)
        sd = sch.state_dict()
        assert 'best' in sd and 'num_bad_epochs' in sd and 'cooldown_counter' in sd
        sch2 = O.ReduceLROnPlateau(base_lr=1.0, patience=1, factor=0.5)
        sch2.load_state_dict(sd)
        assert float(sch2.best.value) == pytest.approx(float(sch.best.value))

    def test_S15_warm_restarts_state_roundtrip(self):
        sch = O.CosineAnnealingWarmRestarts(base_lr=1.0, T_0=5)
        for _ in range(3):
            sch.step()
        sd = sch.state_dict()
        assert 'T_cur' in sd and 'T_i' in sd
        sch2 = O.CosineAnnealingWarmRestarts(base_lr=1.0, T_0=5)
        sch2.load_state_dict(sd)
        assert int(sch2.T_cur.value) == int(sch.T_cur.value)

    def test_S17_attach_optimizer_error_message(self):
        with pytest.raises(TypeError, match="OptaxOptimizer"):
            O.StepLR(base_lr=0.1).attach_optimizer("not an optimizer")


# --------------------------------------------------------------------------- #
# Optimizer algorithm fixes
# --------------------------------------------------------------------------- #

def _descend(optimizer, ref_tx, steps=3, p0=(1.0, 2.0), grad=(0.5, -0.3)):
    """Run ``optimizer`` and an optax reference ``ref_tx`` on the same problem."""
    w = brainstate.ParamState(jnp.array(p0))
    optimizer.register_trainable_weights({'w': w})
    g = jnp.array(grad)
    for _ in range(steps):
        optimizer.step({'w': g})
    p = jnp.array(p0)
    st = ref_tx.init(p)
    for _ in range(steps):
        u, st = ref_tx.update(g, st, p)
        p = optax.apply_updates(p, u)
    return w.value, p


class TestOptimizerAlgorithms:
    def test_O1_adagrad_is_rss_not_rms(self):
        # O-1: Adagrad must match optax.adagrad (root-sum-of-squares accumulator).
        ours, ref = _descend(O.Adagrad(lr=0.1, initial_accumulator_value=0.1, eps=1e-7),
                             optax.adagrad(0.1))
        assert jnp.allclose(ours, ref, atol=1e-6)

    def test_O12_adagrad_lr_decay(self):
        # O-12: lr_decay must actually decay the step (clr = lr / (1 + t*lr_decay)).
        no_decay = _descend(O.Adagrad(lr=0.1, lr_decay=0.0, initial_accumulator_value=0.1, eps=1e-7),
                            optax.adagrad(0.1))[0]
        with_decay_w = brainstate.ParamState(jnp.array([1.0, 2.0]))
        opt = O.Adagrad(lr=0.1, lr_decay=0.5, initial_accumulator_value=0.1, eps=1e-7)
        opt.register_trainable_weights({'w': with_decay_w})
        for _ in range(3):
            opt.step({'w': jnp.array([0.5, -0.3])})
        # With decay the parameters move less (smaller effective LR over time).
        assert jnp.all(jnp.abs(with_decay_w.value - 1.0) <= jnp.abs(no_decay - 1.0) + 1e-9)
        assert not jnp.allclose(with_decay_w.value, no_decay)

    def test_O5_rprop_matches_optax(self):
        ours, ref = _descend(O.Rprop(lr=0.1), optax.rprop(0.1))
        assert jnp.allclose(ours, ref, atol=1e-6)

    def test_O8_fromage_matches_optax(self):
        ours, ref = _descend(O.Fromage(lr=0.1), optax.fromage(0.1))
        assert jnp.allclose(ours, ref, atol=1e-6)

    def test_O7_lamb_matches_optax(self):
        ours, ref = _descend(O.Lamb(lr=0.1, weight_decay=0.01),
                             optax.lamb(0.1, weight_decay=0.01))
        assert jnp.allclose(ours, ref, atol=1e-6)

    def test_O11_novograd_folds_weight_decay(self):
        ours, ref = _descend(O.Novograd(lr=0.1, weight_decay=0.01),
                             optax.novograd(0.1, weight_decay=0.01))
        assert jnp.allclose(ours, ref, atol=1e-5)

    def test_O6_adafactor_decay_rate_positive_default(self):
        # O-6: the default decay_rate must be positive (0.8), not -0.8.
        assert O.Adafactor().decay_rate == pytest.approx(0.8)

    def test_O18_adafactor_zero_lr_not_silently_replaced(self):
        # O-18: lr=0.0 must not be silently turned into 1e-3 by ``lr or 1e-3``.
        opt = O.Adafactor(lr=0.0)
        assert float(opt.base_lr) == 0.0


class TestWeightDecayCoupling:
    def test_O2_adam_differs_from_adamw(self):
        # O-2: with weight decay, coupled Adam must differ from decoupled AdamW.
        a, _ = _descend(O.Adam(lr=0.1, weight_decay=0.1), optax.identity(), steps=3)
        aw, _ = _descend(O.AdamW(lr=0.1, weight_decay=0.1), optax.identity(), steps=3)
        assert not jnp.allclose(a, aw)

    def test_O2_adam_coupled_matches_reference(self):
        # Coupled L2 Adam == chain(add_decayed_weights, scale_by_adam, scale(-lr)).
        ref = optax.chain(optax.add_decayed_weights(0.1), optax.scale_by_adam(), optax.scale(-0.1))
        ours, refp = _descend(O.Adam(lr=0.1, weight_decay=0.1), ref)
        assert jnp.allclose(ours, refp, atol=1e-6)

    def test_O2_adamw_decoupled_matches_optax(self):
        ours, ref = _descend(O.AdamW(lr=0.1, weight_decay=0.1), optax.adamw(0.1, weight_decay=0.1))
        assert jnp.allclose(ours, ref, atol=1e-6)

    def test_O2_sgd_coupled_weight_decay(self):
        # Coupled SGD: update = -lr * (grad + wd * param). First step from p0 with g.
        w = brainstate.ParamState(jnp.array([2.0]))
        opt = O.SGD(lr=0.1, momentum=0.0, weight_decay=0.5)
        opt.register_trainable_weights({'w': w})
        opt.step({'w': jnp.array([1.0])})
        # expected: 2.0 - 0.1 * (1.0 + 0.5 * 2.0) = 2.0 - 0.2 = 1.8
        assert float(w.value[0]) == pytest.approx(1.8, abs=1e-6)


class TestOptimizerInfra:
    def test_O3_lookahead_runs_without_crash(self):
        # O-3: Lookahead must not crash (previously double-applied base + LookaheadParams).
        w = brainstate.ParamState(jnp.array([1.0, 2.0]))
        opt = O.Lookahead(base_optimizer=optax.sgd(0.1), sync_period=3, alpha=0.5)
        opt.register_trainable_weights({'w': w})
        for _ in range(7):
            opt.step({'w': jnp.array([0.5, -0.3])})
        assert jnp.all(jnp.isfinite(w.value))

    def test_O4_add_param_group_updates_all_groups(self):
        a = brainstate.ParamState(jnp.array([1.0]))
        b = brainstate.ParamState(jnp.array([1.0]))
        opt = O.Adam(lr=0.1)
        opt.register_trainable_weights({'a': a})
        opt.add_param_group({'b': b}, lr=0.5)
        opt.step({'a': jnp.array([1.0]), 'b': jnp.array([1.0])})
        assert float(a.value[0]) < 1.0       # default group updated
        assert float(b.value[0]) < 1.0       # added group updated
        # higher-lr group must move more
        assert (1.0 - float(b.value[0])) > (1.0 - float(a.value[0]))

    def test_O4_add_param_group_before_register_raises(self):
        b = brainstate.ParamState(jnp.array([1.0]))
        opt = O.Adam(lr=0.1)
        with pytest.raises(ValueError):
            opt.add_param_group({'b': b})

    def test_O9_manual_current_lr_affects_applied_update(self):
        # O-9: setting current_lr must change the LR actually applied during step.
        w = brainstate.ParamState(jnp.zeros(1))
        opt = O.SGD(lr=0.1, momentum=0.0)
        opt.register_trainable_weights({'w': w})
        opt.current_lr = 0.5
        opt.step({'w': jnp.ones(1)})
        assert float(w.value[0]) == pytest.approx(-0.5, abs=1e-5)

    def test_O13_checkpoint_resume_equivalent(self):
        g = {'a': jnp.array([1.0]), 'b': jnp.array([1.0])}

        def make():
            a = brainstate.ParamState(jnp.array([1.0]))
            b = brainstate.ParamState(jnp.array([1.0]))
            o = O.Adam(lr=0.1)
            o.register_trainable_weights({'a': a})
            o.add_param_group({'b': b}, lr=0.3)
            return o, a, b

        ref, ra, rb = make()
        for _ in range(3):
            ref.step(g)

        ck, ca, cb = make()
        ck.step(g)
        sd = ck.state_dict()
        a2 = brainstate.ParamState(ca.value)
        b2 = brainstate.ParamState(cb.value)
        opt2 = O.Adam(lr=0.1)
        opt2.register_trainable_weights({'a': a2})
        opt2.add_param_group({'b': b2}, lr=0.3)
        opt2.load_state_dict(sd)
        for _ in range(2):
            opt2.step(g)
        assert jnp.allclose(ra.value, a2.value, atol=1e-6)
        assert jnp.allclose(rb.value, b2.value, atol=1e-6)

    def test_O10_lbfgs_basic_step(self):
        w = brainstate.ParamState(jnp.array([1.0, 2.0]))
        opt = O.LBFGS(lr=1.0)
        opt.register_trainable_weights({'w': w})
        opt.step({'w': jnp.array([0.5, -0.3])})
        assert jnp.all(jnp.isfinite(w.value))


# --------------------------------------------------------------------------- #
# UniqueStateManager / nested-parameter fixes
# --------------------------------------------------------------------------- #

class TestNestedParameters:
    def test_U1_nested_dict_register_and_step(self):
        # U-1: nested dict of params must register and update (previously crashed in
        # the write-back loop with "'dict' object has no attribute 'value'").
        a = brainstate.ParamState(jnp.array([1.0]))
        b = brainstate.ParamState(jnp.array([2.0]))
        opt = O.SGD(lr=0.1, momentum=0.0)
        opt.register_trainable_weights({'layer1': {'w': a}, 'layer2': {'w': b}})
        opt.step({'layer1': {'w': jnp.array([1.0])}, 'layer2': {'w': jnp.array([1.0])}})
        assert float(a.value[0]) == pytest.approx(0.9, abs=1e-6)
        assert float(b.value[0]) == pytest.approx(1.9, abs=1e-6)

    def test_U1_deeply_nested(self):
        c = brainstate.ParamState(jnp.array([1.0]))
        opt = O.Adam(lr=0.1)
        opt.register_trainable_weights({'enc': {'block0': {'w': c}}})
        opt.step({'enc': {'block0': {'w': jnp.array([1.0])}}})
        assert jnp.isfinite(c.value[0]) and float(c.value[0]) < 1.0

    def test_U1_state_value_is_pytree(self):
        # A State whose value is itself a dict (like brainstate ParamState holding
        # {'weight','bias'}) must round-trip through a step without misalignment.
        s = brainstate.ParamState({'weight': jnp.ones((2, 2)), 'bias': jnp.zeros((2,))})
        opt = O.SGD(lr=0.1, momentum=0.0)
        opt.register_trainable_weights({'dense': s})
        opt.step({'dense': {'weight': jnp.ones((2, 2)), 'bias': jnp.ones((2,))}})
        assert isinstance(s.value, dict)
        assert jnp.allclose(s.value['weight'], 0.9)
        assert jnp.allclose(s.value['bias'], -0.1)

    def test_U1_nested_with_brainstate_model(self):
        model = brainstate.nn.Linear(4, 3)
        opt = O.Adam(lr=0.01)
        opt.register_trainable_weights(model.states(brainstate.ParamState))

        def loss():
            return jnp.sum(model(jnp.ones((2, 4))) ** 2)

        grads = brainstate.transform.grad(loss, model.states(brainstate.ParamState))()
        before = list(model.states(brainstate.ParamState).values())[0].value['weight'].copy()
        opt.step(grads)
        after = list(model.states(brainstate.ParamState).values())[0].value['weight']
        assert not jnp.allclose(before, after)


# --------------------------------------------------------------------------- #
# SOFO fixes
# --------------------------------------------------------------------------- #

class TestSOFO:
    def _model(self):
        lin = brainstate.nn.Linear(4, 2)
        return lin

    def test_F1_damping_floor_no_nan(self):
        # F-1: degenerate inputs/targets must not produce NaN/Inf updates.
        model = self._model()
        loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)
        opt = O.SOFO(model, loss_fn, lr=1e-2, tangent_size=8, damping=0.0,
                     key=brainstate.random.split_key())
        opt.register_trainable_weights(model.states(brainstate.ParamState))
        opt.step(jnp.zeros((3, 4)), jnp.zeros((3, 2)))
        w = list(model.states(brainstate.ParamState).values())[0].value['weight']
        assert jnp.all(jnp.isfinite(w))

    def test_F2_fixed_key_resamples_each_step(self):
        # F-2: with a fixed key, consecutive steps must use different tangents (folded
        # with the step count), so the applied directions differ step to step.
        import jax
        model = self._model()
        loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)
        opt = O.SOFO(model, loss_fn, lr=1e-1, tangent_size=8, key=jax.random.PRNGKey(0))
        opt.register_trainable_weights(model.states(brainstate.ParamState))
        x = jnp.ones((4, 4))
        y = jnp.ones((4, 2))
        w = lambda: list(model.states(brainstate.ParamState).values())[0].value['weight']
        w0 = w().copy()
        opt.step(x, y)
        d1 = w() - w0
        w1 = w().copy()
        opt.step(x, y)
        d2 = w() - w1
        # If the key were not folded with the step count the two raw tangent batches
        # would be identical; the resulting updates must not be proportional/identical.
        assert not jnp.allclose(d1, d2)

    def test_sofo_reduces_loss(self):
        model = self._model()
        loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)
        opt = O.SOFO(model, loss_fn, lr=5e-2, tangent_size=16,
                     key=brainstate.random.split_key())
        opt.register_trainable_weights(model.states(brainstate.ParamState))
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 2))
        first = float(opt.step(x, y))
        for _ in range(15):
            last = float(opt.step(x, y))
        assert last < first


# --------------------------------------------------------------------------- #
# ScipyOptimizer fixes
# --------------------------------------------------------------------------- #

class TestScipyOptimizer:
    def test_X1_tnc_slsqp_run_under_float32_default(self):
        # X-1: TNC/SLSQP Cython kernels need float64 buffers; the wrapper must cast
        # x0/jac/bounds to float64 so these methods don't crash under JAX float32.
        pytest.importorskip('scipy')
        from braintools.optim import ScipyOptimizer

        def loss(x, y):
            return (x - 1.0) ** 2 + (y + 2.0) ** 2

        bounds = [(-5.0, 5.0), (-3.0, 3.0)]
        for method in ('TNC', 'SLSQP'):
            opt = ScipyOptimizer(loss_fun=loss, bounds=bounds, method=method)
            res = opt.minimize(n_iter=3)
            assert res is not None
            assert np.isfinite(res.fun)
            # convex quadratic → gradient method converges near the optimum
            assert res.fun < 1e-2, (method, res.fun)

    def test_X2_minimize_returns_result_on_nonfinite_loss(self, monkeypatch):
        # X-2: `inf < inf` / `nan < inf` are False, so a plain `<` left best_res=None
        # for diverging losses. minimize() must return a result, not None.
        import types
        import braintools.optim._scipy_optimizer as so

        def fake_minimize_with_jax(x0, loss_fun, **kwargs):
            return types.SimpleNamespace(fun=float('inf'), x=x0)

        monkeypatch.setattr(so, 'scipy_minimize_with_jax', fake_minimize_with_jax, raising=False)
        from braintools.optim import ScipyOptimizer

        def loss(x):
            return jnp.inf * jnp.sum(x ** 2)

        opt = ScipyOptimizer(loss_fun=loss, bounds=[(-1.0, 1.0)], method='L-BFGS-B')
        res = opt.minimize(n_iter=3)
        assert res is not None
        assert not np.isfinite(res.fun)

    def test_X3_gradient_free_method_emits_no_jac_warning(self):
        # X-3: gradient-free methods must not be handed a Jacobian (which triggers a
        # per-iteration "does not use gradient information" RuntimeWarning).
        pytest.importorskip('scipy')
        import warnings
        from braintools.optim import ScipyOptimizer

        def loss(x, y):
            return (x - 1.0) ** 2 + (y + 2.0) ** 2

        opt = ScipyOptimizer(loss_fun=loss, bounds=[(-5.0, 5.0), (-3.0, 3.0)], method='Powell')
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            opt.minimize(n_iter=1)
        msgs = [str(w.message) for w in rec]
        assert not any('does not use gradient' in m for m in msgs), msgs


# --------------------------------------------------------------------------- #
# NevergradOptimizer fixes
# --------------------------------------------------------------------------- #

class TestNevergradOptimizer:
    def test_N1_all_nan_losses_do_not_crash(self):
        # N-1: an all-NaN iteration must not crash via np.nanargmin's "All-NaN slice";
        # the optimizer falls back to Nevergrad's recommendation (with a warning).
        pytest.importorskip('nevergrad')
        import warnings
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return jnp.full(x.shape, jnp.nan)

        opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                 n_sample=6, method='OnePlusOne')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = opt.minimize(n_iter=1, verbose=False)
        assert len(res) == 2
        assert np.all(np.isfinite(np.asarray(res)))

    def test_N1_all_nan_emits_warning(self):
        # N-1: the all-NaN fallback should warn rather than silently continue.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return jnp.full(x.shape, jnp.nan)

        opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                 n_sample=6, method='OnePlusOne')
        with pytest.warns(RuntimeWarning):
            opt.minimize(n_iter=1, verbose=False)

    def test_N3_seed_makes_runs_reproducible(self):
        # N-3: with a seed, the ask/tell sampling is reproducible across runs.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return x ** 2 + y ** 2

        def run():
            opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                     n_sample=8, method='RandomSearch', budget=100, seed=42)
            return opt.minimize(n_iter=2, verbose=False)

        r1 = run()
        r2 = run()
        assert np.allclose(np.asarray(r1), np.asarray(r2))

    def test_N3_different_seeds_differ(self):
        # N-3 sanity: different seeds should generally explore differently.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return x ** 2 + y ** 2

        def run(seed):
            opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                     n_sample=8, method='RandomSearch', budget=100, seed=seed)
            return opt.minimize(n_iter=2, verbose=False)

        assert not np.allclose(np.asarray(run(1)), np.asarray(run(2)))

    def test_N5_budget_exceeded_warns(self):
        # N-5: n_iter*n_sample evals are not capped by `budget`; warn on a mismatch.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return x ** 2 + y ** 2

        opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                 n_sample=8, method='OnePlusOne', budget=5)
        with pytest.warns(RuntimeWarning, match='budget'):
            opt.minimize(n_iter=2, verbose=False)


# --------------------------------------------------------------------------- #
# Every optimizer takes a real .step() (audit "missing feature #1")
# --------------------------------------------------------------------------- #

class TestStepAllOptimizers:
    """Construct, register and ``.step()`` every optimizer end-to-end.

    The audit noted CI "almost never calls ``.step()`` for the exotic optimizers",
    hiding a class of ``default_tx``/step bugs. This descends a 2-D parameter with
    a fixed gradient for a few steps and asserts updates stay finite and move.
    """

    # Each factory accepts ``**k`` so the same registry can be exercised with and
    # without gradient clipping (every ``default_tx`` has a clip branch).
    OPTIMIZERS = [
        ('SGD', lambda **k: O.SGD(lr=0.05, momentum=0.9, **k)),
        ('SGD_nesterov', lambda **k: O.SGD(lr=0.05, momentum=0.9, nesterov=True, **k)),
        ('SGD_wd', lambda **k: O.SGD(lr=0.05, weight_decay=0.01, **k)),
        ('Momentum', lambda **k: O.Momentum(lr=0.05, momentum=0.9, **k)),
        ('MomentumNesterov', lambda **k: O.MomentumNesterov(lr=0.05, momentum=0.9, **k)),
        ('Adam', lambda **k: O.Adam(lr=0.05, **k)),
        ('Adam_wd', lambda **k: O.Adam(lr=0.05, weight_decay=0.01, **k)),
        ('Adam_amsgrad', lambda **k: O.Adam(lr=0.05, amsgrad=True, **k)),
        ('AdamW', lambda **k: O.AdamW(lr=0.05, weight_decay=0.01, **k)),
        ('Adagrad', lambda **k: O.Adagrad(lr=0.05, weight_decay=0.01, **k)),
        ('Adadelta', lambda **k: O.Adadelta(lr=0.5, **k)),
        ('RMSprop', lambda **k: O.RMSprop(lr=0.05, **k)),
        ('RMSprop_centered', lambda **k: O.RMSprop(lr=0.05, momentum=0.9, centered=True, **k)),
        ('Adamax', lambda **k: O.Adamax(lr=0.05, **k)),
        ('Nadam', lambda **k: O.Nadam(lr=0.05, **k)),
        ('RAdam', lambda **k: O.RAdam(lr=0.05, **k)),
        ('Lamb', lambda **k: O.Lamb(lr=0.05, weight_decay=0.01, **k)),
        ('Lars', lambda **k: O.Lars(lr=0.05, weight_decay=0.01, **k)),
        ('Yogi', lambda **k: O.Yogi(lr=0.05, **k)),
        ('Rprop', lambda **k: O.Rprop(lr=0.05, **k)),
        ('Adafactor', lambda **k: O.Adafactor(lr=0.05, **k)),
        ('AdaBelief', lambda **k: O.AdaBelief(lr=0.05, **k)),
        ('Lion', lambda **k: O.Lion(lr=0.05, **k)),
        ('SM3', lambda **k: O.SM3(lr=0.05, **k)),
        ('Novograd', lambda **k: O.Novograd(lr=0.05, weight_decay=0.01, **k)),
        ('Fromage', lambda **k: O.Fromage(lr=0.05, **k)),
        ('LBFGS', lambda **k: O.LBFGS(lr=1.0, **k)),
        ('Lookahead', lambda **k: O.Lookahead(optax.adam(0.05), sync_period=2, **k)),
    ]

    @pytest.mark.parametrize('name,factory', OPTIMIZERS, ids=[n for n, _ in OPTIMIZERS])
    def test_step_updates_and_finite(self, name, factory):
        w = brainstate.ParamState(jnp.array([[1.0, -2.0], [0.5, 3.0]]))
        opt = factory()
        opt.register_trainable_weights({'w': w})
        before = w.value.copy()
        g = {'w': jnp.array([[0.3, -0.4], [0.2, -0.1]])}
        for _ in range(3):
            opt.step(g)
        assert jnp.all(jnp.isfinite(w.value)), f'{name} produced non-finite params'
        assert not jnp.allclose(w.value, before), f'{name} did not update params'

    @pytest.mark.parametrize('name,factory', OPTIMIZERS, ids=[n for n, _ in OPTIMIZERS])
    def test_step_with_grad_clipping(self, name, factory):
        # Exercise the grad_clip_norm / grad_clip_value branches in every default_tx.
        w = brainstate.ParamState(jnp.array([[1.0, -2.0], [0.5, 3.0]]))
        opt = factory(grad_clip_norm=1.0, grad_clip_value=0.5)
        opt.register_trainable_weights({'w': w})
        before = w.value.copy()
        g = {'w': jnp.array([[30.0, -40.0], [20.0, -10.0]])}  # large → triggers clipping
        for _ in range(3):
            opt.step(g)
        assert jnp.all(jnp.isfinite(w.value)), f'{name} non-finite under clipping'
        assert not jnp.allclose(w.value, before), f'{name} did not update under clipping'

    @pytest.mark.parametrize('name,factory', OPTIMIZERS, ids=[n for n, _ in OPTIMIZERS])
    def test_state_dict_roundtrip(self, name, factory):
        # state_dict/load_state_dict must restore the optimizer's internal buffers
        # (momentum/variance/step count) so checkpoint→resume is exact (O-13).
        w = brainstate.ParamState(jnp.array([[1.0, -2.0], [0.5, 3.0]]))
        opt = factory()
        opt.register_trainable_weights({'w': w})
        g = {'w': jnp.array([[0.3, -0.4], [0.2, -0.1]])}
        opt.step(g)
        opt.step(g)
        sd = opt.state_dict()

        # Fresh optimizer with the param set to the SAME current value: only the
        # optimizer state distinguishes them, so a matching next step proves the
        # round-trip restored that state.
        w2 = brainstate.ParamState(w.value.copy())
        opt2 = factory()
        opt2.register_trainable_weights({'w': w2})
        opt2.load_state_dict(sd)

        opt.step(g)
        opt2.step(g)
        assert jnp.allclose(w.value, w2.value, atol=1e-5), name


# --------------------------------------------------------------------------- #
# Multi-group step: params outside any added group still update (lines 541-552)
# --------------------------------------------------------------------------- #

class TestMultiGroupStep:
    def test_default_params_update_alongside_added_group(self):
        w1 = brainstate.ParamState(jnp.array([1.0, 2.0]))
        w2 = brainstate.ParamState(jnp.array([3.0, 4.0]))
        opt = O.Adam(lr=0.1)
        opt.register_trainable_weights({'w1': w1})
        opt.add_param_group({'w2': w2}, lr=0.3)
        b1, b2 = w1.value.copy(), w2.value.copy()
        opt.step({'w1': jnp.ones(2), 'w2': jnp.ones(2)})
        # w1 belongs to no explicit added group → flows through the main tx;
        # w2 is in the added group with a larger LR → moves further.
        assert not jnp.allclose(w1.value, b1)
        assert not jnp.allclose(w2.value, b2)
        assert jnp.all(jnp.isfinite(w1.value)) and jnp.all(jnp.isfinite(w2.value))


# --------------------------------------------------------------------------- #
# SOFO update() data-alias and momentum branches
# --------------------------------------------------------------------------- #

class _ScanCell(brainstate.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.wh = brainstate.nn.Linear(n_hidden, n_hidden)
        self.wx = brainstate.nn.Linear(n_in, n_hidden)
        self.wo = brainstate.nn.Linear(n_hidden, n_out)

    def __call__(self, latent, inputs):
        new_latent = self.wh(latent) + self.wx(inputs)
        return new_latent, self.wo(new_latent)


class TestSOFOContract:
    def test_sofo_update_is_data_alias_of_step(self):
        # F-3: SOFO.update(inputs, targets) is the documented data-alias of step().
        model = brainstate.nn.Linear(4, 2)
        loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)
        opt = O.SOFO(model, loss_fn, lr=1e-2, tangent_size=8,
                     key=brainstate.random.split_key())
        opt.register_trainable_weights(model.states(brainstate.ParamState))
        loss = opt.update(jnp.ones((4, 4)), jnp.ones((4, 2)))
        assert jnp.isfinite(loss)

    def test_sofo_momentum_nesterov_steps(self):
        # Exercise the momentum/nesterov default_tx branch of SOFO.
        model = brainstate.nn.Linear(4, 2)
        loss_fn = lambda pred, y: jnp.mean((pred - y) ** 2)
        opt = O.SOFO(model, loss_fn, lr=2e-2, tangent_size=16, momentum=0.9,
                     nesterov=True, key=brainstate.random.split_key())
        opt.register_trainable_weights(model.states(brainstate.ParamState))
        x, y = jnp.ones((8, 4)), jnp.ones((8, 2))
        first = float(opt.step(x, y))
        for _ in range(10):
            last = float(opt.step(x, y))
        assert jnp.isfinite(last) and last < first

    def test_sofoscan_update_is_data_alias_of_step(self):
        # F-3 for SOFOScan: update(z_init, batch) aliases step().
        brainstate.random.seed(0)
        n_in, n_hidden, n_out, batch, T = 3, 5, 2, 4, 5
        cell = _ScanCell(n_in, n_hidden, n_out)
        loss_fn = lambda out, label: jnp.mean((out - label) ** 2)
        opt = O.SOFOScan(cell, loss_fn, lr=1e-2, tangent_size=32,
                         key=brainstate.random.split_key())
        opt.register_trainable_weights(cell.states(brainstate.ParamState))
        rng = brainstate.random.RandomState(0)
        xs = rng.randn(T, batch, n_in)
        ys = rng.randn(T, batch, n_out)
        z0 = jnp.zeros((batch, n_hidden))
        loss = opt.update(z0, (xs, ys))
        assert jnp.isfinite(loss)


# --------------------------------------------------------------------------- #
# LBFGS line-search construction + update(value=, value_fn=) path (O-17)
# --------------------------------------------------------------------------- #

class TestLBFGSLinesearch:
    @staticmethod
    def _quadratic(params):
        # value_fn receives the params pytree {'w': array}; minimise sum(w**2).
        return sum(jnp.sum(v ** 2) for v in params.values())

    @pytest.mark.parametrize('linesearch', ['zoom', 'backtracking'])
    def test_linesearch_descends(self, linesearch):
        # String linesearch names resolve to optax transforms and the
        # value/value_fn update path drives the loss down on a convex problem.
        w = brainstate.ParamState(jnp.array([2.0, -3.0, 1.5]))
        opt = O.LBFGS(lr=1.0, linesearch=linesearch)
        opt.register_trainable_weights({'w': w})

        def loss_fn(params):
            return self._quadratic(params)

        first = last = None
        for _ in range(8):
            value = float(self._quadratic({'w': w.value}))
            grads = {'w': 2.0 * w.value}
            opt.update(grads, value=value, value_fn=loss_fn)
            last = float(self._quadratic({'w': w.value}))
            if first is None:
                first = value
        assert jnp.all(jnp.isfinite(w.value))
        assert last < first, f'{linesearch} line-search did not reduce the loss'

    def test_linesearch_accepts_optax_transform(self):
        # A pre-built optax GradientTransformationExtraArgs is accepted as-is.
        ls = optax.scale_by_zoom_linesearch(max_linesearch_steps=20)
        w = brainstate.ParamState(jnp.array([1.0, -1.0]))
        opt = O.LBFGS(lr=1.0, linesearch=ls)
        opt.register_trainable_weights({'w': w})

        def loss_fn(params):
            return self._quadratic(params)

        value = float(self._quadratic({'w': w.value}))
        opt.update({'w': 2.0 * w.value}, value=value, value_fn=loss_fn)
        assert jnp.all(jnp.isfinite(w.value))

    def test_invalid_linesearch_string_raises(self):
        with pytest.raises(ValueError, match='Unknown linesearch'):
            O.LBFGS(linesearch='nope')

    def test_invalid_linesearch_type_raises(self):
        with pytest.raises(ValueError, match='Unknown linesearch'):
            O.LBFGS(linesearch=12345)


# --------------------------------------------------------------------------- #
# Misc real (non-defensive) branches: Adam beta1/beta2 deprecation, Lookahead WD
# --------------------------------------------------------------------------- #

class TestMiscBranches:
    def test_adam_beta1_beta2_deprecated_override(self):
        # The legacy beta1/beta2 kwargs must warn and override the betas tuple.
        with pytest.warns(DeprecationWarning, match='beta1'):
            opt = O.Adam(lr=0.05, beta1=0.8)
        assert opt.betas[0] == 0.8
        with pytest.warns(DeprecationWarning, match='beta2'):
            opt = O.Adam(lr=0.05, beta2=0.99)
        assert opt.betas[1] == 0.99
        # Both override and the optimizer still steps cleanly.
        w = brainstate.ParamState(jnp.array([1.0, -2.0]))
        opt.register_trainable_weights({'w': w})
        opt.step({'w': jnp.array([0.5, -0.5])})
        assert jnp.all(jnp.isfinite(w.value))

    def test_lookahead_weight_decay_folds_into_fast(self):
        # weight_decay > 0 chains add_decayed_weights into the fast optimizer.
        w = brainstate.ParamState(jnp.array([1.0, -2.0, 3.0]))
        opt = O.Lookahead(optax.sgd(0.1), sync_period=2, alpha=0.5, weight_decay=0.01)
        opt.register_trainable_weights({'w': w})
        before = w.value.copy()
        for _ in range(4):
            opt.step({'w': jnp.array([0.2, -0.3, 0.1])})
        assert jnp.all(jnp.isfinite(w.value))
        assert not jnp.allclose(w.value, before)
