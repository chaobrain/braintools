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

"""Regression tests for the 2026-06-18 ``braintools.optim`` *re-audit* fixes.

Each test maps to an issue ID (R-1 .. R-17) from
``docs/braintools-optim-issues-found-20260618.md`` and asserts the *corrected* behavior
(i.e. it would fail against the pre-fix code).
"""

import numpy as np
import jax.numpy as jnp
import optax
import pytest

import brainstate

import braintools.optim as O


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


# --------------------------------------------------------------------------- #
# R-1: SM3 must apply momentum exactly once (match optax.sm3)
# --------------------------------------------------------------------------- #

class TestSM3Momentum:
    def test_R1_sm3_matches_optax(self):
        ours, ref = _descend(O.SM3(lr=0.1, momentum=0.9, eps=1e-8), optax.sm3(0.1, 0.9))
        assert jnp.allclose(ours, ref, atol=1e-6)

    def test_R1_momentum_zero_disables_smoothing(self):
        # Previously momentum=0 still applied the internal scale_by_sm3 b1=0.9.
        w0, _ = _descend(O.SM3(lr=0.1, momentum=0.0), optax.identity())
        w9, _ = _descend(O.SM3(lr=0.1, momentum=0.9), optax.identity())
        assert not jnp.allclose(w0, w9)
        # momentum=0 ⇒ matches optax.sm3 with momentum=0
        ours, ref = _descend(O.SM3(lr=0.1, momentum=0.0, eps=1e-8), optax.sm3(0.1, 0.0))
        assert jnp.allclose(ours, ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# R-2: RMSprop(centered=True) must use the variance estimate (scale_by_stddev)
# --------------------------------------------------------------------------- #

class TestRMSpropCentered:
    def test_R2_centered_matches_optax(self):
        ours, ref = _descend(O.RMSprop(lr=0.1, alpha=0.9, eps=1e-8, centered=True),
                             optax.rmsprop(0.1, decay=0.9, eps=1e-8, centered=True))
        assert jnp.allclose(ours, ref, atol=1e-6)

    def test_R2_centered_differs_from_uncentered(self):
        wt, _ = _descend(O.RMSprop(lr=0.1, alpha=0.9, centered=True), optax.identity())
        wf, _ = _descend(O.RMSprop(lr=0.1, alpha=0.9, centered=False), optax.identity())
        assert not jnp.allclose(wt, wf)


# --------------------------------------------------------------------------- #
# R-3: Nadam.momentum_decay must drive PyTorch-style scheduled momentum
# --------------------------------------------------------------------------- #

class TestNadamScheduledMomentum:
    def test_R3_matches_torch_nadam(self):
        torch = pytest.importorskip('torch')
        lr, b1, b2, eps, psi = 0.01, 0.9, 0.999, 1e-8, 4e-3
        grads = [np.array([0.5, -0.3, 0.2], np.float32),
                 np.array([0.1, 0.4, -0.6], np.float32),
                 np.array([-0.2, 0.05, 0.3], np.float32),
                 np.array([0.3, -0.1, 0.15], np.float32)]
        p0 = np.array([1.0, 2.0, -1.0], np.float32)

        w = brainstate.ParamState(jnp.array(p0))
        opt = O.Nadam(lr=lr, betas=(b1, b2), eps=eps, momentum_decay=psi)
        opt.register_trainable_weights({'w': w})
        for g in grads:
            opt.step({'w': jnp.array(g)})
        ours = np.asarray(w.value)

        tp = torch.tensor(p0.copy(), requires_grad=True)
        topt = torch.optim.NAdam([tp], lr=lr, betas=(b1, b2), eps=eps,
                                 momentum_decay=psi, weight_decay=0.0)
        for g in grads:
            topt.zero_grad()
            tp.grad = torch.tensor(g.copy())
            topt.step()
        ref = tp.detach().numpy()
        assert np.allclose(ours, ref, atol=1e-5), (ours, ref)

    def test_R3_momentum_decay_actually_changes_result(self):
        # Two different momentum_decay values must yield different trajectories
        # (the parameter was previously inert).
        a, _ = _descend(O.Nadam(lr=0.05, momentum_decay=4e-3), optax.identity(), steps=5)
        b, _ = _descend(O.Nadam(lr=0.05, momentum_decay=2e-1), optax.identity(), steps=5)
        assert not jnp.allclose(a, b)


# --------------------------------------------------------------------------- #
# R-4: LBFGS docstring parameter name (scale_init_precond) is the real one
# --------------------------------------------------------------------------- #

class TestLBFGSDocParam:
    @pytest.mark.parametrize('flag', [True, False])
    def test_R4_scale_init_precond_constructs(self, flag):
        opt = O.LBFGS(lr=1.0, scale_init_precond=flag)
        assert opt.scale_init_precond is flag

    def test_R4_scale_init_hess_rejected(self):
        with pytest.raises(TypeError):
            O.LBFGS(lr=1.0, scale_init_hess=True)


# --------------------------------------------------------------------------- #
# R-5: LBFGS honors an LRScheduler (no longer frozen at construction)
# --------------------------------------------------------------------------- #

class TestLBFGSSchedule:
    @staticmethod
    def _run(make_lr, steps=4):
        w = brainstate.ParamState(jnp.array([2.0, -3.0, 1.5]))
        opt = O.LBFGS(lr=make_lr())
        opt.register_trainable_weights({'w': w})
        sch = opt.lr_scheduler
        traj = []
        for _ in range(steps):
            opt.step({'w': 2.0 * w.value})
            traj.append(np.asarray(w.value).copy())
            sch.step()
        return traj

    def test_R5_scheduler_changes_trajectory(self):
        base = self._run(lambda: 1.0)  # float -> constant lr 1.0
        decay = self._run(lambda: O.StepLR(base_lr=1.0, step_size=1, gamma=0.1))
        # epoch-0 LR is identical (gamma**0 == 1), so the first step matches ...
        assert np.allclose(base[0], decay[0], atol=1e-6)
        # ... but once the schedule decays, the trajectories diverge.
        assert not np.allclose(base[3], decay[3])


# --------------------------------------------------------------------------- #
# R-6: CosineAnnealingWarmRestarts.step(epoch=...) jumps to the absolute epoch
# --------------------------------------------------------------------------- #

class TestWarmRestartsEpochJump:
    @staticmethod
    def _stepwise(T0, Tmult, n):
        s = O.CosineAnnealingWarmRestarts(base_lr=1.0, T_0=T0, T_mult=Tmult)
        for _ in range(n):
            s.step()
        return int(s.T_cur.value), int(s.T_i.value)

    @staticmethod
    def _jump(T0, Tmult, n):
        s = O.CosineAnnealingWarmRestarts(base_lr=1.0, T_0=T0, T_mult=Tmult)
        s.step(epoch=n)
        return int(s.T_cur.value), int(s.T_i.value)

    @pytest.mark.parametrize('Tmult', [1, 2, 3])
    @pytest.mark.parametrize('n', [1, 3, 5, 7, 12, 20])
    def test_R6_epoch_jump_matches_stepwise(self, Tmult, n):
        assert self._jump(3, Tmult, n) == self._stepwise(3, Tmult, n)


# --------------------------------------------------------------------------- #
# R-7: scheduler enum params validated fail-fast in __init__
# --------------------------------------------------------------------------- #

class TestSchedulerEnumValidation:
    def test_R7_cyclic_mode(self):
        with pytest.raises(ValueError, match='mode'):
            O.CyclicLR(mode='nope')

    def test_R7_cyclic_scale_mode(self):
        with pytest.raises(ValueError, match='scale_mode'):
            O.CyclicLR(scale_mode='nope')

    def test_R7_onecycle_anneal(self):
        with pytest.raises(ValueError, match='anneal_strategy'):
            O.OneCycleLR(max_lr=1.0, total_steps=10, anneal_strategy='nope')

    def test_R7_plateau_mode(self):
        with pytest.raises(ValueError, match='mode'):
            O.ReduceLROnPlateau(mode='minimize')

    def test_R7_plateau_threshold_mode(self):
        with pytest.raises(ValueError, match='threshold_mode'):
            O.ReduceLROnPlateau(threshold_mode='relative')


# --------------------------------------------------------------------------- #
# R-8: optimizer hyperparameter validation
# --------------------------------------------------------------------------- #

class TestHyperparamValidation:
    def test_R8_negative_lr(self):
        with pytest.raises(ValueError, match='learning rate'):
            O.SGD(lr=-1.0)

    def test_R8_negative_weight_decay(self):
        with pytest.raises(ValueError, match='weight_decay'):
            O.Adam(lr=0.1, weight_decay=-0.1)

    def test_R8_nonpositive_grad_clip_norm(self):
        with pytest.raises(ValueError, match='grad_clip_norm'):
            O.Adam(lr=0.1, grad_clip_norm=0.0)

    def test_R8_negative_grad_clip_value(self):
        with pytest.raises(ValueError, match='grad_clip_value'):
            O.Adam(lr=0.1, grad_clip_value=-1.0)

    def test_R8_scheduler_lr_not_rejected(self):
        # A scheduler lr must still be accepted (only float lr is range-checked).
        opt = O.SGD(lr=O.StepLR(base_lr=0.1, step_size=1))
        assert opt is not None


# --------------------------------------------------------------------------- #
# R-12: base OptaxOptimizer.default_tx uses coupled weight decay (like Adam)
# --------------------------------------------------------------------------- #

class TestBaseCoupledWeightDecay:
    def test_R12_base_default_tx_is_coupled(self):
        # Base fallback (scale_by_adam) with weight decay must equal coupled L2 Adam:
        # chain(add_decayed_weights, scale_by_adam, scale(-lr)).
        ref = optax.chain(optax.add_decayed_weights(0.1), optax.scale_by_adam(), optax.scale(-0.1))
        ours, refp = _descend(O.OptaxOptimizer(lr=0.1, weight_decay=0.1), ref)
        assert jnp.allclose(ours, refp, atol=1e-6)


# --------------------------------------------------------------------------- #
# R-9/R-10: documented examples are now runnable (construction does not raise)
# --------------------------------------------------------------------------- #

class TestDocExamplesRunnable:
    def test_R9_adafactor_positive_decay_rate(self):
        opt = O.Adafactor(lr=0.05, decay_rate=0.8)
        assert opt.decay_rate == pytest.approx(0.8)

    def test_R10_onecycle_find_max_lr_pct_start(self):
        # The find_max_lr docstring example uses pct_start=0.99 (must be < 1.0).
        sch = O.OneCycleLR(max_lr=10.0, total_steps=100, div_factor=1e8,
                           final_div_factor=1.0, pct_start=0.99)
        assert 0.0 < sch.pct_start < 1.0
        with pytest.raises(ValueError):
            O.OneCycleLR(max_lr=10.0, total_steps=100, pct_start=1.0)
