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

# -*- coding: utf-8 -*-

import pytest
import jax
import jax.numpy as jnp
import brainstate
import braintools.optim
import numpy as np


# ==============================================================================
# Test StepLR Scheduler
# ==============================================================================

class TestStepLR:
    """Test StepLR scheduler"""

    def test_basic_step_lr(self):
        """Test basic StepLR functionality"""
        scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=10, gamma=0.1)

        # Initial learning rate
        assert scheduler.current_lrs.value[0] == 0.1

        # After 9 steps, should still be 0.1
        for _ in range(9):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.1)

        # After 10th step, should be 0.01
        scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.01)

        # After 20th step, should be 0.001
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.001)

    def test_step_lr_with_optimizer(self):
        """Test StepLR integration with optimizer"""
        scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=5, gamma=0.5)
        optimizer = braintools.optim.Adam(lr=scheduler)

        # Create a simple model
        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.1)

        # Step scheduler
        for _ in range(5):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.05)

    def test_step_lr_jit(self):
        """Test StepLR with JIT compilation"""
        scheduler = braintools.optim.StepLR(base_lr=1.0, step_size=10, gamma=0.1)

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        # Initial lr
        assert scheduler.current_lrs.value[0] == 1.0

        # Run jitted steps
        for i in range(15):
            lr = jit_step()
            if i < 9:
                assert np.isclose(lr, 1.0)
            else:
                assert np.isclose(lr, 0.1)

    def test_step_lr_multiple_param_groups(self):
        """Test StepLR with multiple learning rates"""
        scheduler = braintools.optim.StepLR(base_lr=[0.1, 0.01], step_size=5, gamma=0.1)

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 0.1)
        assert np.isclose(scheduler.current_lrs.value[1], 0.01)

        # Step and check decay
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.01)
        assert np.isclose(scheduler.current_lrs.value[1], 0.001)

    def test_original(self):
        """Original test from existing code"""
        optimizer = braintools.optim.Adam(braintools.optim.StepLR(0.1))
        optimizer.lr_apply(lambda lr: lr * 0.5)
        assert optimizer.current_lr == 0.05


# ==============================================================================
# Test MultiStepLR Scheduler
# ==============================================================================

class TestMultiStepLR:
    """Test MultiStepLR scheduler"""

    def test_basic_multistep_lr(self):
        """Test basic MultiStepLR functionality"""
        scheduler = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[10, 20, 30],
            gamma=0.1
        )

        # Initial learning rate
        assert scheduler.current_lrs.value[0] == 1.0

        # Before first milestone (epoch 10)
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.1), \
            f"Expected 0.1, got {scheduler.current_lrs.value[0]}"

        # Before second milestone (epoch 20)
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.01), \
            f"Expected 0.01, got {scheduler.current_lrs.value[0]}"

        # Before third milestone (epoch 30)
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.001), \
            f"Expected 0.001, got {scheduler.current_lrs.value[0]}"

        # After all milestones
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.001), \
            f"LR should remain constant after all milestones"

    def test_multistep_lr_with_optimizer(self):
        """Test MultiStepLR integration with optimizer"""
        scheduler = braintools.optim.MultiStepLR(
            base_lr=0.1,
            milestones=[5, 10],
            gamma=0.1
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.1)

        # Step to first milestone
        for _ in range(5):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.01), \
            f"Expected 0.01 at milestone 5, got {optimizer.current_lr}"

        # Step to second milestone
        for _ in range(5):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.001), \
            f"Expected 0.001 at milestone 10, got {optimizer.current_lr}"

    def test_multistep_lr_jit(self):
        """Test MultiStepLR with JIT compilation"""
        scheduler = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[5, 10],
            gamma=0.5
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        # Initial lr
        assert scheduler.current_lrs.value[0] == 1.0

        # Run jitted steps and verify LR at each stage
        # After step i, we're at epoch i+1, so milestone is reached at step (milestone-1)
        for i in range(15):
            lr = jit_step()
            # After step i, we're at epoch i+1
            # milestone 5 is reached after step 4 (epoch becomes 5)
            # milestone 10 is reached after step 9 (epoch becomes 10)
            if i < 4:  # epochs 1-4, before milestone 5
                assert np.isclose(lr, 1.0), f"Step {i} (epoch {i+1}): expected 1.0, got {lr}"
            elif i < 9:  # epochs 5-9, after milestone 5, before milestone 10
                assert np.isclose(lr, 0.5), f"Step {i} (epoch {i+1}): expected 0.5, got {lr}"
            else:  # epochs 10+, after milestone 10
                assert np.isclose(lr, 0.25), f"Step {i} (epoch {i+1}): expected 0.25, got {lr}"

    def test_multistep_lr_multiple_param_groups(self):
        """Test MultiStepLR with multiple learning rates"""
        scheduler = braintools.optim.MultiStepLR(
            base_lr=[1.0, 0.1],
            milestones=[5, 10],
            gamma=0.1
        )

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1)

        # Step to first milestone
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.1)
        assert np.isclose(scheduler.current_lrs.value[1], 0.01)

        # Step to second milestone
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.01)
        assert np.isclose(scheduler.current_lrs.value[1], 0.001)

    def test_multistep_lr_empty_milestones(self):
        """Test MultiStepLR with no milestones"""
        scheduler = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[],
            gamma=0.1
        )

        # LR should remain constant with no milestones
        initial_lr = scheduler.current_lrs.value[0]
        for _ in range(20):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], initial_lr), \
            "LR should not change without milestones"

    def test_multistep_lr_single_milestone(self):
        """Test MultiStepLR with a single milestone"""
        scheduler = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[10],
            gamma=0.5
        )

        # Before milestone
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5)

        # After milestone
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5), \
            "LR should remain constant after last milestone"

    def test_multistep_lr_state_dict(self):
        """Test MultiStepLR state dict save/load"""
        scheduler1 = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[10, 20],
            gamma=0.1
        )

        # Run some steps
        for _ in range(15):
            scheduler1.step()

        # Save state
        state_dict = scheduler1.state_dict()

        # Create new scheduler and load state
        scheduler2 = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[10, 20],
            gamma=0.1
        )
        scheduler2.load_state_dict(state_dict)

        # Verify state matches
        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

        # Verify they continue identically
        for _ in range(10):
            scheduler1.step()
            scheduler2.step()
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_multistep_lr_jit_with_optimizer(self):
        """Test MultiStepLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.MultiStepLR(
            base_lr=0.1,
            milestones=[5, 10],
            gamma=0.1
        )
        optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        @brainstate.transform.jit
        def train_step(x):
            # Forward pass
            y = model(x)
            loss = jnp.sum(y ** 2)

            # Backward pass
            grads = brainstate.transform.grad(
                lambda: jnp.sum(model(x) ** 2),
                grad_states=model.states(brainstate.ParamState)
            )()

            # Update
            optimizer.step(grads)

            # Step scheduler
            scheduler.step()

            return loss, optimizer.current_lr

        # Run training steps
        x = jnp.ones((1, 10))

        # Before first milestone (epochs 1-4, steps 0-3)
        for i in range(4):
            loss, lr = train_step(x)
            assert np.isclose(lr, 0.1), f"Step {i} (epoch {i+1}): expected LR 0.1, got {lr}"

        # After first milestone, before second (epochs 5-9, steps 4-8)
        for i in range(4, 9):
            loss, lr = train_step(x)
            assert np.isclose(lr, 0.01), f"Step {i} (epoch {i+1}): expected LR 0.01, got {lr}"

        # After second milestone (epochs 10+, steps 9+)
        for i in range(9, 15):
            loss, lr = train_step(x)
            assert np.isclose(lr, 0.001), f"Step {i} (epoch {i+1}): expected LR 0.001, got {lr}"

    def test_multistep_lr_various_gamma_values(self):
        """Test MultiStepLR with different gamma values"""
        # Test with gamma = 0.5
        scheduler = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[5, 10],
            gamma=0.5
        )

        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5)

        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.25)

        # Test with gamma = 0.2
        scheduler2 = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[5, 10],
            gamma=0.2
        )

        for _ in range(5):
            scheduler2.step()
        assert np.isclose(scheduler2.current_lrs.value[0], 0.2)

        for _ in range(5):
            scheduler2.step()
        assert np.isclose(scheduler2.current_lrs.value[0], 0.04)

    def test_multistep_lr_close_milestones(self):
        """Test MultiStepLR with closely spaced milestones"""
        scheduler = braintools.optim.MultiStepLR(
            base_lr=1.0,
            milestones=[2, 3, 4],
            gamma=0.5
        )

        # Step through each milestone
        expected_lrs = [1.0, 1.0, 0.5, 0.25, 0.125]
        for i, expected_lr in enumerate(expected_lrs):
            if i > 0:
                scheduler.step()
            current_lr = scheduler.current_lrs.value[0]
            assert np.isclose(current_lr, expected_lr), \
                f"Step {i}: expected {expected_lr}, got {current_lr}"


# ==============================================================================
# Test ExponentialLR Scheduler
# ==============================================================================

class TestExponentialLR:
    """Test ExponentialLR scheduler"""

    def test_basic_exponential_lr(self):
        """Test basic ExponentialLR functionality"""
        scheduler = braintools.optim.ExponentialLR(base_lr=1.0, gamma=0.9)

        # Initial learning rate
        assert scheduler.current_lrs.value[0] == 1.0

        # After each step, lr should be multiplied by gamma
        expected_lrs = [1.0]
        for i in range(10):
            scheduler.step()
            expected_lrs.append(expected_lrs[-1] * 0.9)
            assert np.isclose(scheduler.current_lrs.value[0], expected_lrs[-1]), \
                f"Step {i+1}: expected {expected_lrs[-1]}, got {scheduler.current_lrs.value[0]}"

    def test_exponential_lr_with_optimizer(self):
        """Test ExponentialLR integration with optimizer"""
        scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        optimizer = braintools.optim.Adam(lr=scheduler)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.1)

        # Step scheduler 5 times
        expected_lr = 0.1
        for _ in range(5):
            scheduler.step()
            expected_lr *= 0.95

        assert np.isclose(optimizer.current_lr, expected_lr), \
            f"Expected {expected_lr}, got {optimizer.current_lr}"

    def test_exponential_lr_jit(self):
        """Test ExponentialLR with JIT compilation"""
        scheduler = braintools.optim.ExponentialLR(base_lr=1.0, gamma=0.95)

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        # Initial lr
        assert scheduler.current_lrs.value[0] == 1.0

        # Run jitted steps
        expected_lr = 1.0
        for i in range(10):
            expected_lr *= 0.95
            lr = jit_step()
            assert np.isclose(lr, expected_lr), \
                f"Step {i} (epoch {i+1}): expected {expected_lr}, got {lr}"

    def test_exponential_lr_multiple_param_groups(self):
        """Test ExponentialLR with multiple learning rates"""
        scheduler = braintools.optim.ExponentialLR(base_lr=[1.0, 0.1], gamma=0.9)

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1)

        # Step and check decay
        for _ in range(5):
            scheduler.step()

        assert np.isclose(scheduler.current_lrs.value[0], 1.0 * (0.9 ** 5))
        assert np.isclose(scheduler.current_lrs.value[1], 0.1 * (0.9 ** 5))

    def test_exponential_lr_state_dict(self):
        """Test ExponentialLR state dict save/load"""
        scheduler1 = braintools.optim.ExponentialLR(base_lr=1.0, gamma=0.9)

        # Run some steps
        for _ in range(7):
            scheduler1.step()

        # Save state
        state_dict = scheduler1.state_dict()

        # Create new scheduler and load state
        scheduler2 = braintools.optim.ExponentialLR(base_lr=1.0, gamma=0.9)
        scheduler2.load_state_dict(state_dict)

        # Verify state matches
        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_exponential_lr_gamma_near_one(self):
        """Test ExponentialLR with gamma very close to 1.0"""
        scheduler = braintools.optim.ExponentialLR(base_lr=1.0, gamma=0.99)

        # LR should decay very slowly
        for _ in range(10):
            scheduler.step()

        # After 10 steps with gamma=0.99: lr = 1.0 * 0.99^10 ≈ 0.904
        assert np.isclose(scheduler.current_lrs.value[0], 1.0 * (0.99 ** 10))
        assert scheduler.current_lrs.value[0] > 0.9

    def test_exponential_lr_jit_with_optimizer(self):
        """Test ExponentialLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.ExponentialLR(base_lr=0.1, gamma=0.95)
        optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        @brainstate.transform.jit
        def train_step(x):
            y = model(x)
            loss = jnp.sum(y ** 2)

            grads = brainstate.transform.grad(
                lambda: jnp.sum(model(x) ** 2),
                grad_states=model.states(brainstate.ParamState)
            )()

            optimizer.step(grads)
            scheduler.step()

            return loss, optimizer.current_lr

        x = jnp.ones((1, 10))
        expected_lr = 0.1

        for i in range(10):
            expected_lr *= 0.95
            loss, lr = train_step(x)
            assert np.isclose(lr, expected_lr), \
                f"Step {i} (epoch {i+1}): expected LR {expected_lr}, got {lr}"


# ==============================================================================
# Test ExponentialDecayLR Scheduler
# ==============================================================================

class TestExponentialDecayLR:
    """Test ExponentialDecayLR scheduler"""

    def test_basic_exponential_decay_lr(self):
        """Test basic ExponentialDecayLR functionality"""
        scheduler = braintools.optim.ExponentialDecayLR(
            base_lr=1.0,
            decay_steps=10,
            decay_rate=0.5,
            staircase=False
        )

        # Initial learning rate
        assert scheduler.current_lrs.value[0] == 1.0

        # After decay_steps, lr should be approximately decay_rate * base_lr
        for _ in range(10):
            scheduler.step()

        # With staircase=False, it's continuous: lr = base_lr * decay_rate^(step/decay_steps)
        assert np.isclose(scheduler.current_lrs.value[0], 0.5, rtol=1e-5), \
            f"Expected ~0.5, got {scheduler.current_lrs.value[0]}"

    def test_exponential_decay_lr_staircase(self):
        """Test ExponentialDecayLR with staircase=True"""
        scheduler = braintools.optim.ExponentialDecayLR(
            base_lr=1.0,
            decay_steps=10,
            decay_rate=0.5,
            staircase=True
        )

        # Before decay_steps, lr should remain constant
        for _ in range(9):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0), \
            f"Expected 1.0, got {scheduler.current_lrs.value[0]}"

        # At decay_steps, lr should drop to decay_rate * base_lr
        scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5), \
            f"Expected 0.5, got {scheduler.current_lrs.value[0]}"

        # Continue to next decay boundary
        for _ in range(9):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5)

        scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.25)

    def test_exponential_decay_lr_with_optimizer(self):
        """Test ExponentialDecayLR integration with optimizer"""
        scheduler = braintools.optim.ExponentialDecayLR(
            base_lr=0.1,
            decay_steps=5,
            decay_rate=0.5,
            staircase=True
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.1)

        # Step to first decay
        for _ in range(5):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.05)

    def test_exponential_decay_lr_jit(self):
        """Test ExponentialDecayLR with JIT compilation"""
        scheduler = braintools.optim.ExponentialDecayLR(
            base_lr=1.0,
            decay_steps=5,
            decay_rate=0.5,
            staircase=True
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        # Test staircase behavior under JIT
        # After step i, epoch = i+1. With decay_steps=5:
        # epochs 1-4: floor((i+1)/5) = 0, lr = 1.0 * 0.5^0 = 1.0
        # epochs 5-9: floor((i+1)/5) = 1, lr = 1.0 * 0.5^1 = 0.5
        # epochs 10-14: floor((i+1)/5) = 2, lr = 1.0 * 0.5^2 = 0.25
        # epochs 15+: floor((i+1)/5) = 3, lr = 1.0 * 0.5^3 = 0.125
        for i in range(16):
            lr = jit_step()
            epoch = i + 1
            if epoch < 5:
                assert np.isclose(lr, 1.0), f"Step {i} (epoch {epoch}): expected 1.0, got {lr}"
            elif epoch < 10:
                assert np.isclose(lr, 0.5), f"Step {i} (epoch {epoch}): expected 0.5, got {lr}"
            elif epoch < 15:
                assert np.isclose(lr, 0.25), f"Step {i} (epoch {epoch}): expected 0.25, got {lr}"
            else:
                assert np.isclose(lr, 0.125), f"Step {i} (epoch {epoch}): expected 0.125, got {lr}"

    def test_exponential_decay_lr_continuous(self):
        """Test ExponentialDecayLR with continuous decay (staircase=False)"""
        scheduler = braintools.optim.ExponentialDecayLR(
            base_lr=1.0,
            decay_steps=10,
            decay_rate=0.5,
            staircase=False
        )

        # LR should decay smoothly
        prev_lr = scheduler.current_lrs.value[0]
        for _ in range(20):
            scheduler.step()
            current_lr = scheduler.current_lrs.value[0]
            assert current_lr < prev_lr, "LR should decrease monotonically"
            prev_lr = current_lr

    def test_exponential_decay_lr_state_dict(self):
        """Test ExponentialDecayLR state dict save/load"""
        scheduler1 = braintools.optim.ExponentialDecayLR(
            base_lr=1.0,
            decay_steps=10,
            decay_rate=0.5,
            staircase=True
        )

        for _ in range(12):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.ExponentialDecayLR(
            base_lr=1.0,
            decay_steps=10,
            decay_rate=0.5,
            staircase=True
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_exponential_decay_lr_multiple_param_groups(self):
        """Test ExponentialDecayLR with multiple learning rates"""
        scheduler = braintools.optim.ExponentialDecayLR(
            base_lr=[1.0, 0.1],
            decay_steps=5,
            decay_rate=0.5,
            staircase=True
        )

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1)

        # Step to first decay
        for _ in range(5):
            scheduler.step()

        assert np.isclose(scheduler.current_lrs.value[0], 0.5)
        assert np.isclose(scheduler.current_lrs.value[1], 0.05)


# ==============================================================================
# Test CosineAnnealingLR Scheduler
# ==============================================================================

class TestCosineAnnealingLR:
    """Test CosineAnnealingLR scheduler"""

    def test_basic_cosine_annealing_lr(self):
        """Test basic CosineAnnealingLR functionality"""
        scheduler = braintools.optim.CosineAnnealingLR(
            base_lr=1.0,
            T_max=10,
            eta_min=0.0
        )

        # Initial learning rate
        assert scheduler.current_lrs.value[0] == 1.0

        # At T_max/2, lr should be around base_lr/2
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5, atol=0.1), \
            f"At T_max/2, expected ~0.5, got {scheduler.current_lrs.value[0]}"

        # At T_max, lr should be eta_min
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.0, atol=1e-5), \
            f"At T_max, expected {0.0}, got {scheduler.current_lrs.value[0]}"

    def test_cosine_annealing_lr_with_optimizer(self):
        """Test CosineAnnealingLR integration with optimizer"""
        scheduler = braintools.optim.CosineAnnealingLR(
            base_lr=0.1,
            T_max=20,
            eta_min=0.001
        )
        optimizer = braintools.optim.AdamW(lr=scheduler, weight_decay=0.01)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.1)

        # Step to middle
        for _ in range(10):
            scheduler.step()

        # Should be around midpoint between base_lr and eta_min
        mid_lr = (0.1 + 0.001) / 2
        assert np.isclose(optimizer.current_lr, mid_lr, atol=0.01)

        # Step to end
        for _ in range(10):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.001, atol=1e-5)

    def test_cosine_annealing_lr_jit(self):
        """Test CosineAnnealingLR with JIT compilation"""
        scheduler = braintools.optim.CosineAnnealingLR(
            base_lr=1.0,
            T_max=100,
            eta_min=0.1
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        lrs = []
        for _ in range(100):
            lrs.append(jit_step())

        # Check that it follows cosine pattern
        assert lrs[-1] < lrs[0], "LR should decrease"
        assert lrs[-1] >= 0.1, f"LR should not go below eta_min, got {lrs[-1]}"

        # Check monotonic decrease in first half
        for i in range(49):
            assert lrs[i] >= lrs[i+1] - 1e-6, \
                f"LR should decrease monotonically in first half at step {i}"

    def test_cosine_annealing_lr_multiple_param_groups(self):
        """Test CosineAnnealingLR with multiple learning rates"""
        scheduler = braintools.optim.CosineAnnealingLR(
            base_lr=[1.0, 0.1],
            T_max=10,
            eta_min=0.01
        )

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1)

        # Step to T_max
        for _ in range(10):
            scheduler.step()

        assert np.isclose(scheduler.current_lrs.value[0], 0.01)
        assert np.isclose(scheduler.current_lrs.value[1], 0.01)

    def test_cosine_annealing_lr_state_dict(self):
        """Test CosineAnnealingLR state dict save/load"""
        scheduler1 = braintools.optim.CosineAnnealingLR(
            base_lr=1.0,
            T_max=100,
            eta_min=0.0
        )

        for _ in range(50):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.CosineAnnealingLR(
            base_lr=1.0,
            T_max=100,
            eta_min=0.0
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_cosine_annealing_lr_symmetry(self):
        """Test that CosineAnnealingLR follows cosine curve symmetry"""
        scheduler = braintools.optim.CosineAnnealingLR(
            base_lr=1.0,
            T_max=20,
            eta_min=0.0
        )

        lrs = [scheduler.current_lrs.value[0]]
        for _ in range(20):
            scheduler.step()
            lrs.append(scheduler.current_lrs.value[0])

        # Check that LR at T_max/4 and 3*T_max/4 are symmetric around midpoint
        lr_quarter = lrs[5]
        lr_three_quarter = lrs[15]
        midpoint = 0.5

        # Both should be approximately equidistant from midpoint
        dist1 = abs(lr_quarter - midpoint)
        dist2 = abs(lr_three_quarter - midpoint)
        assert np.isclose(dist1, dist2, atol=0.1), \
            f"Cosine should be symmetric: {lr_quarter} and {lr_three_quarter}"

    def test_cosine_annealing_lr_jit_with_optimizer(self):
        """Test CosineAnnealingLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.CosineAnnealingLR(
            base_lr=0.1,
            T_max=20,
            eta_min=0.001
        )
        optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        @brainstate.transform.jit
        def train_step(x):
            y = model(x)
            loss = jnp.sum(y ** 2)

            grads = brainstate.transform.grad(
                lambda: jnp.sum(model(x) ** 2),
                grad_states=model.states(brainstate.ParamState)
            )()

            optimizer.step(grads)
            scheduler.step()

            return loss, optimizer.current_lr

        x = jnp.ones((1, 10))
        lrs = []

        for i in range(20):
            loss, lr = train_step(x)
            lrs.append(lr)

        # Check LR follows cosine pattern
        assert lrs[0] > lrs[-1], "LR should decrease from start to end"
        assert lrs[-1] >= 0.001 - 1e-5, "LR should not go below eta_min"

        # Check monotonic decrease in first half
        for i in range(9):
            assert lrs[i] >= lrs[i+1] - 1e-6, \
                f"LR should decrease in first half at step {i}"

    def test_cosine_annealing_lr_small_tmax(self):
        """Test CosineAnnealingLR with very small T_max"""
        scheduler = braintools.optim.CosineAnnealingLR(
            base_lr=1.0,
            T_max=2,
            eta_min=0.0
        )

        # Initial
        assert scheduler.current_lrs.value[0] == 1.0

        # After 1 step
        scheduler.step()
        assert scheduler.current_lrs.value[0] < 1.0
        assert scheduler.current_lrs.value[0] > 0.0

        # After 2 steps (at T_max)
        scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.0, atol=1e-5)


# ==============================================================================
# Test PolynomialLR Scheduler
# ==============================================================================

class TestPolynomialLR:
    """Test PolynomialLR scheduler"""

    def test_basic_polynomial_lr(self):
        """Test basic PolynomialLR functionality"""
        scheduler = braintools.optim.PolynomialLR(
            base_lr=1.0,
            total_iters=10,
            power=2.0
        )

        # Initial learning rate
        assert scheduler.current_lrs.value[0] == 1.0

        # Step through and check polynomial decay
        for i in range(10):
            scheduler.step()
            # lr = base_lr * (1 - min(epoch, total_iters) / total_iters) ^ power
            epoch = i + 1
            expected_lr = 1.0 * ((1 - min(epoch, 10) / 10) ** 2.0)
            assert np.isclose(scheduler.current_lrs.value[0], expected_lr), \
                f"Step {i} (epoch {epoch}): expected {expected_lr}, got {scheduler.current_lrs.value[0]}"

        # After total_iters, lr should be 0
        assert np.isclose(scheduler.current_lrs.value[0], 0.0, atol=1e-6)

    def test_polynomial_lr_linear_decay(self):
        """Test PolynomialLR with power=1.0 (linear decay)"""
        scheduler = braintools.optim.PolynomialLR(
            base_lr=1.0,
            total_iters=10,
            power=1.0
        )

        # Initial
        assert scheduler.current_lrs.value[0] == 1.0

        # At half way, should be 0.5
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5), \
            f"Expected 0.5, got {scheduler.current_lrs.value[0]}"

        # At end, should be 0
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.0, atol=1e-6)

    def test_polynomial_lr_with_optimizer(self):
        """Test PolynomialLR integration with optimizer"""
        scheduler = braintools.optim.PolynomialLR(
            base_lr=0.1,
            total_iters=20,
            power=2.0
        )
        optimizer = braintools.optim.Adam(lr=scheduler)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.1)

        # Step halfway
        for _ in range(10):
            scheduler.step()

        # Should be significantly reduced
        assert optimizer.current_lr < 0.1
        assert optimizer.current_lr > 0.0

    def test_polynomial_lr_jit(self):
        """Test PolynomialLR with JIT compilation"""
        scheduler = braintools.optim.PolynomialLR(
            base_lr=1.0,
            total_iters=10,
            power=1.0  # Linear decay
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        # Test linear decay under JIT
        for i in range(10):
            lr = jit_step()
            expected = 1.0 - (i + 1) / 10
            assert np.isclose(lr, expected, atol=1e-6), \
                f"Step {i}: expected {expected}, got {lr}"

    def test_polynomial_lr_multiple_param_groups(self):
        """Test PolynomialLR with multiple learning rates"""
        scheduler = braintools.optim.PolynomialLR(
            base_lr=[1.0, 0.1],
            total_iters=10,
            power=2.0
        )

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1)

        # Step to middle
        for _ in range(5):
            scheduler.step()

        # Both should decay proportionally
        assert scheduler.current_lrs.value[0] > scheduler.current_lrs.value[1]

        # Step to end
        for _ in range(5):
            scheduler.step()

        assert np.isclose(scheduler.current_lrs.value[0], 0.0, atol=1e-6)
        assert np.isclose(scheduler.current_lrs.value[1], 0.0, atol=1e-6)

    def test_polynomial_lr_state_dict(self):
        """Test PolynomialLR state dict save/load"""
        scheduler1 = braintools.optim.PolynomialLR(
            base_lr=1.0,
            total_iters=20,
            power=2.0
        )

        for _ in range(8):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.PolynomialLR(
            base_lr=1.0,
            total_iters=20,
            power=2.0
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_polynomial_lr_different_powers(self):
        """Test PolynomialLR with different power values"""
        powers = [0.5, 1.0, 2.0, 3.0]

        for power in powers:
            scheduler = braintools.optim.PolynomialLR(
                base_lr=1.0,
                total_iters=10,
                power=power
            )

            # Step to middle
            for _ in range(5):
                scheduler.step()

            # All should be decreasing but at different rates
            assert scheduler.current_lrs.value[0] < 1.0
            assert scheduler.current_lrs.value[0] > 0.0

    def test_polynomial_lr_jit_with_optimizer(self):
        """Test PolynomialLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.PolynomialLR(
            base_lr=0.1,
            total_iters=10,
            power=1.0
        )
        optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        @brainstate.transform.jit
        def train_step(x):
            y = model(x)
            loss = jnp.sum(y ** 2)

            grads = brainstate.transform.grad(
                lambda: jnp.sum(model(x) ** 2),
                grad_states=model.states(brainstate.ParamState)
            )()

            optimizer.step(grads)
            scheduler.step()

            return loss, optimizer.current_lr

        x = jnp.ones((1, 10))
        prev_lr = 0.1

        for i in range(10):
            loss, lr = train_step(x)
            assert lr <= prev_lr + 1e-6, "LR should decrease monotonically"
            prev_lr = lr


# ==============================================================================
# Test WarmupScheduler
# ==============================================================================

class TestWarmupScheduler:
    """Test WarmupScheduler"""

    def test_basic_warmup(self):
        """Test basic WarmupScheduler functionality"""
        scheduler = braintools.optim.WarmupScheduler(
            base_lr=1.0,
            warmup_epochs=10,
            warmup_start_lr=0.0
        )

        # Initial learning rate should be warmup_start_lr
        assert np.isclose(scheduler.current_lrs.value[0], 0.0)

        # After warmup_epochs/2, should be around base_lr/2
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5, atol=0.1), \
            f"At warmup/2, expected ~0.5, got {scheduler.current_lrs.value[0]}"

        # After warmup_epochs, should be base_lr
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0), \
            f"After warmup, expected 1.0, got {scheduler.current_lrs.value[0]}"

        # After warmup, should stay at base_lr
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

    def test_warmup_with_nonzero_start(self):
        """Test WarmupScheduler with non-zero start LR"""
        scheduler = braintools.optim.WarmupScheduler(
            base_lr=1.0,
            warmup_epochs=10,
            warmup_start_lr=0.1
        )

        # Initial should be warmup_start_lr
        assert np.isclose(scheduler.current_lrs.value[0], 0.1)

        # Linear interpolation from 0.1 to 1.0
        for i in range(10):
            scheduler.step()

        # Should reach base_lr after warmup
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

    def test_warmup_with_optimizer(self):
        """Test WarmupScheduler integration with optimizer"""
        scheduler = braintools.optim.WarmupScheduler(
            base_lr=0.1,
            warmup_epochs=5,
            warmup_start_lr=0.01
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.01)

        # After warmup
        for _ in range(5):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.1)

    def test_warmup_jit(self):
        """Test WarmupScheduler with JIT compilation"""
        scheduler = braintools.optim.WarmupScheduler(
            base_lr=1.0,
            warmup_epochs=10,
            warmup_start_lr=0.1
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        lrs = []
        for _ in range(15):
            lrs.append(jit_step())

        # Should be increasing during warmup
        for i in range(9):
            assert lrs[i] <= lrs[i+1] + 1e-6, \
                f"LR should increase during warmup at step {i}"

        # After warmup, should stay at base_lr
        for i in range(10, 14):
            assert np.isclose(lrs[i], 1.0), \
                f"After warmup, LR should be 1.0, got {lrs[i]}"

    def test_warmup_multiple_param_groups(self):
        """Test WarmupScheduler with multiple learning rates"""
        scheduler = braintools.optim.WarmupScheduler(
            base_lr=[1.0, 0.1],
            warmup_epochs=10,
            warmup_start_lr=0.0
        )

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 0.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.0)

        # After warmup
        for _ in range(10):
            scheduler.step()

        assert np.isclose(scheduler.current_lrs.value[0], 1.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1)

    def test_warmup_state_dict(self):
        """Test WarmupScheduler state dict save/load"""
        scheduler1 = braintools.optim.WarmupScheduler(
            base_lr=1.0,
            warmup_epochs=10,
            warmup_start_lr=0.1
        )

        for _ in range(7):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.WarmupScheduler(
            base_lr=1.0,
            warmup_epochs=10,
            warmup_start_lr=0.1
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_warmup_single_epoch(self):
        """Test WarmupScheduler with warmup_epochs=1"""
        scheduler = braintools.optim.WarmupScheduler(
            base_lr=1.0,
            warmup_epochs=1,
            warmup_start_lr=0.5
        )

        # Initial
        assert np.isclose(scheduler.current_lrs.value[0], 0.5)

        # After 1 step
        scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

    def test_warmup_jit_with_optimizer(self):
        """Test WarmupScheduler with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.WarmupScheduler(
            base_lr=0.1,
            warmup_epochs=5,
            warmup_start_lr=0.01
        )
        optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        @brainstate.transform.jit
        def train_step(x):
            y = model(x)
            loss = jnp.sum(y ** 2)

            grads = brainstate.transform.grad(
                lambda: jnp.sum(model(x) ** 2),
                grad_states=model.states(brainstate.ParamState)
            )()

            optimizer.step(grads)
            scheduler.step()

            return loss, optimizer.current_lr

        x = jnp.ones((1, 10))
        lrs = []

        for i in range(10):
            loss, lr = train_step(x)
            lrs.append(lr)

        # Check warmup phase
        for i in range(4):
            assert lrs[i] < lrs[i+1], f"LR should increase during warmup at step {i}"

        # After warmup
        for i in range(5, 9):
            assert np.isclose(lrs[i], 0.1), f"After warmup, LR should be 0.1, got {lrs[i]}"


# ==============================================================================
# Test CyclicLR Scheduler
# ==============================================================================

class TestCyclicLR:
    """Test CyclicLR scheduler"""

    def test_basic_cyclic_lr_triangular(self):
        """Test basic CyclicLR with triangular mode"""
        scheduler = braintools.optim.CyclicLR(
            base_lr=0.1,
            max_lr=1.0,
            step_size_up=10,
            mode='triangular'
        )

        # Initial learning rate should be base_lr
        assert np.isclose(scheduler.current_lrs.value[0], 0.1)

        # Should increase towards max_lr during step_size_up
        prev_lr = scheduler.current_lrs.value[0]
        for _ in range(10):
            scheduler.step()
            current_lr = scheduler.current_lrs.value[0]
            assert current_lr >= prev_lr - 1e-6, "LR should increase during upward phase"
            prev_lr = current_lr

        # At peak, should be near max_lr
        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.1)

        # Should decrease back to base_lr
        for _ in range(10):
            scheduler.step()
            current_lr = scheduler.current_lrs.value[0]
            assert current_lr <= prev_lr + 1e-6, "LR should decrease during downward phase"
            prev_lr = current_lr

    def test_cyclic_lr_triangular2_mode(self):
        """Test CyclicLR with triangular2 mode"""
        scheduler = braintools.optim.CyclicLR(
            base_lr=0.1,
            max_lr=1.0,
            step_size_up=5,
            mode='triangular2'
        )

        # First cycle
        for _ in range(10):
            scheduler.step()

        # Get max LR of first cycle
        first_cycle_max = scheduler.current_lrs.value[0]

        # Second cycle - max should be reduced
        for _ in range(5):
            scheduler.step()

        second_cycle_max = scheduler.current_lrs.value[0]

        # In triangular2 mode, amplitude decreases by half each cycle
        # So second cycle max should be less than first
        assert second_cycle_max < first_cycle_max

    def test_cyclic_lr_exp_range_mode(self):
        """Test CyclicLR with exp_range mode"""
        scheduler = braintools.optim.CyclicLR(
            base_lr=0.1,
            max_lr=1.0,
            step_size_up=5,
            mode='exp_range',
            gamma=0.99
        )

        # Should still cycle but with exponential decay
        for _ in range(20):
            scheduler.step()

        # LR should be between base and max
        lr = scheduler.current_lrs.value[0]
        assert lr >= 0.05  # Accounting for decay
        assert lr <= 1.0

    def test_cyclic_lr_with_optimizer(self):
        """Test CyclicLR integration with optimizer"""
        scheduler = braintools.optim.CyclicLR(
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=5,
            mode='triangular'
        )
        optimizer = braintools.optim.Adam(lr=scheduler)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial lr
        assert np.isclose(optimizer.current_lr, 0.01)

        # Step and verify cycling
        initial_lr = optimizer.current_lr
        for _ in range(5):
            scheduler.step()

        # Should have increased
        assert optimizer.current_lr > initial_lr

    def test_cyclic_lr_jit(self):
        """Test CyclicLR with JIT compilation"""
        scheduler = braintools.optim.CyclicLR(
            base_lr=0.1,
            max_lr=1.0,
            step_size_up=5,
            mode='triangular'
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        lrs = []
        for _ in range(20):
            lrs.append(jit_step())

        # Check it cycles - should see ups and downs
        assert max(lrs) > min(lrs)

        # Check first increase
        assert lrs[4] > lrs[0], "LR should increase in first phase"

    def test_cyclic_lr_multiple_param_groups(self):
        """Test CyclicLR with multiple learning rates"""
        scheduler = braintools.optim.CyclicLR(
            base_lr=[0.1, 0.01],
            max_lr=[1.0, 0.1],
            step_size_up=5,
            mode='triangular'
        )

        # Check initial lrs
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 0.1)
        assert np.isclose(scheduler.current_lrs.value[1], 0.01)

        # Both should cycle proportionally
        for _ in range(5):
            scheduler.step()

        # Both should have increased
        assert scheduler.current_lrs.value[0] > 0.1
        assert scheduler.current_lrs.value[1] > 0.01

    def test_cyclic_lr_state_dict(self):
        """Test CyclicLR state dict save/load"""
        scheduler1 = braintools.optim.CyclicLR(
            base_lr=0.1,
            max_lr=1.0,
            step_size_up=5,
            mode='triangular'
        )

        for _ in range(7):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.CyclicLR(
            base_lr=0.1,
            max_lr=1.0,
            step_size_up=5,
            mode='triangular'
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_cyclic_lr_step_size_down(self):
        """Test CyclicLR with custom step_size_down"""
        scheduler = braintools.optim.CyclicLR(
            base_lr=0.1,
            max_lr=1.0,
            step_size_up=5,
            step_size_down=10,
            mode='triangular'
        )

        # Go through one complete cycle
        lrs = []
        for _ in range(15):
            scheduler.step()
            lrs.append(scheduler.current_lrs.value[0])

        # Should go up in 5 steps, down in 10 steps
        assert lrs[4] > lrs[0]  # Increasing phase
        assert lrs[14] < lrs[4]  # Decreasing phase

    def test_cyclic_lr_jit_with_optimizer(self):
        """Test CyclicLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.CyclicLR(
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=5,
            mode='triangular'
        )
        optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        @brainstate.transform.jit
        def train_step(x):
            y = model(x)
            loss = jnp.sum(y ** 2)

            grads = brainstate.transform.grad(
                lambda: jnp.sum(model(x) ** 2),
                grad_states=model.states(brainstate.ParamState)
            )()

            optimizer.step(grads)
            scheduler.step()

            return loss, optimizer.current_lr

        x = jnp.ones((1, 10))
        lrs = []

        for i in range(15):
            loss, lr = train_step(x)
            lrs.append(lr)

        # Check cycling behavior
        assert max(lrs) > min(lrs), "LR should cycle"
        assert lrs[4] > lrs[0], "LR should increase initially"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
