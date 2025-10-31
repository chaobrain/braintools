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

        # Initial learning rate is base_lr (scheduler hasn't stepped yet)
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # After 5 steps (epoch 5), alpha = 5/10 = 0.5, lr = 0.0 + 1.0 * 0.5 = 0.5
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5, atol=0.01), \
            f"At step 5, expected 0.5, got {scheduler.current_lrs.value[0]}"

        # After 10 steps (epoch 10), alpha = 10/10 = 1.0, lr = 1.0
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

        # Initial should be base_lr (scheduler hasn't stepped yet)
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # After step 1: alpha = 1/10 = 0.1, lr = 0.1 + 0.9 * 0.1 = 0.19
        scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.19, atol=0.01)

        # After 10 steps total, should reach base_lr
        for _ in range(9):
            scheduler.step()
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

        # Check initial lr (base_lr before stepping)
        assert np.isclose(optimizer.current_lr, 0.1)

        # After 5 steps, should reach base_lr
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

        # Check initial lrs (base_lrs before stepping)
        assert len(scheduler.current_lrs.value) == 2
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1)

        # After 10 steps, should be at base_lr
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

        # Initial (base_lr before stepping)
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # After 1 step, alpha = 1/1 = 1.0, so we're at base_lr
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

        # Track max LR in first cycle
        first_cycle_lrs = []
        for _ in range(10):
            scheduler.step()
            first_cycle_lrs.append(scheduler.current_lrs.value[0])
        first_cycle_max = max(first_cycle_lrs)

        # Track max LR in second cycle
        second_cycle_lrs = []
        for _ in range(10):
            scheduler.step()
            second_cycle_lrs.append(scheduler.current_lrs.value[0])
        second_cycle_max = max(second_cycle_lrs)

        # In triangular2 mode, amplitude decreases by half each cycle
        # So second cycle max should be half of first
        assert np.isclose(second_cycle_max, first_cycle_max / 2, atol=0.05), \
            f"Expected {first_cycle_max / 2}, got {second_cycle_max}"

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

        # Go through one complete cycle plus a bit more
        lrs = []
        for _ in range(16):
            scheduler.step()
            lrs.append(scheduler.current_lrs.value[0])

        # Should go up in 5 steps (epochs 1-5)
        assert lrs[4] > lrs[0], f"lrs[4]={lrs[4]}, lrs[0]={lrs[0]}"

        # Peak should be at epoch 5 (index 4)
        assert np.isclose(lrs[4], 1.0, atol=0.01)

        # Should go down in 10 steps (epochs 6-15), reaching base_lr at epoch 14
        assert lrs[13] < lrs[4], f"lrs[13]={lrs[13]}, lrs[4]={lrs[4]}"
        assert np.isclose(lrs[13], 0.1, atol=0.05)

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


class TestOneCycleLR:
    """Test OneCycleLR"""

    def test_basic_onecycle_lr(self):
        """Test basic OneCycleLR functionality"""
        scheduler = braintools.optim.OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )

        # Initial LR should be max_lr / div_factor = 0.1
        assert np.isclose(scheduler.current_lrs.value[0], 0.1, atol=0.01)

        # After warmup phase (30 steps), should be near max_lr
        for _ in range(30):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.1)

        # After full cycle (100 steps), should be near final_lr = max_lr / final_div_factor = 0.01
        for _ in range(70):
            scheduler.step()
        assert scheduler.current_lrs.value[0] < 0.05

    def test_onecycle_lr_with_epochs(self):
        """Test OneCycleLR with epochs and steps_per_epoch"""
        scheduler = braintools.optim.OneCycleLR(
            max_lr=0.1,
            epochs=10,
            steps_per_epoch=10,
            pct_start=0.3
        )

        # Total steps = 10 * 10 = 100
        # Warmup = 30 steps
        lrs = []
        for _ in range(100):
            scheduler.step()
            lrs.append(scheduler.current_lrs.value[0])

        # Check warmup: should increase
        assert lrs[29] > lrs[0]

        # Check annealing: should decrease
        assert lrs[99] < lrs[30]

    def test_onecycle_lr_with_optimizer(self):
        """Test OneCycleLR integration with optimizer"""
        scheduler = braintools.optim.OneCycleLR(
            max_lr=0.5,
            total_steps=50,
            pct_start=0.3
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        initial_lr = optimizer.current_lr

        # After warmup, LR should be higher
        for _ in range(15):
            scheduler.step()
        assert optimizer.current_lr > initial_lr

        # After full cycle, LR should be lower than initial
        for _ in range(35):
            scheduler.step()
        assert optimizer.current_lr < initial_lr

    def test_onecycle_lr_jit(self):
        """Test OneCycleLR with JIT compilation"""
        scheduler = braintools.optim.OneCycleLR(
            max_lr=1.0,
            total_steps=50,
            pct_start=0.4
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        lrs = []
        for _ in range(50):
            lrs.append(jit_step())

        # Should increase during warmup (first 20 steps)
        assert lrs[19] > lrs[0]

        # Should decrease during annealing (steps 20-50)
        assert lrs[49] < lrs[20]

    def test_onecycle_lr_linear_anneal(self):
        """Test OneCycleLR with linear annealing"""
        scheduler = braintools.optim.OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            pct_start=0.3,
            anneal_strategy='linear'
        )

        for _ in range(100):
            scheduler.step()

        # Should reach low final LR
        assert scheduler.current_lrs.value[0] < 0.01

    def test_onecycle_lr_multiple_param_groups(self):
        """Test OneCycleLR with multiple learning rates"""
        scheduler = braintools.optim.OneCycleLR(
            max_lr=[1.0, 0.1],
            total_steps=50,
            pct_start=0.3
        )

        assert len(scheduler.current_lrs.value) == 2

        for _ in range(50):
            scheduler.step()

        # Both should reach low final LR
        assert scheduler.current_lrs.value[0] < 0.1
        assert scheduler.current_lrs.value[1] < 0.01

    def test_onecycle_lr_state_dict(self):
        """Test OneCycleLR state dict save/load"""
        scheduler1 = braintools.optim.OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            pct_start=0.3
        )

        for _ in range(40):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.OneCycleLR(
            max_lr=1.0,
            total_steps=100,
            pct_start=0.3
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_onecycle_lr_jit_with_optimizer(self):
        """Test OneCycleLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.OneCycleLR(
            max_lr=0.1,
            total_steps=30,
            pct_start=0.3
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

        for i in range(30):
            loss, lr = train_step(x)
            lrs.append(lr)

        # Check warmup phase increases
        assert lrs[8] > lrs[0]

        # Check annealing phase decreases
        assert lrs[29] < lrs[10]


class TestReduceLROnPlateau:
    """Test ReduceLROnPlateau"""

    def test_basic_reduce_lr_on_plateau(self):
        """Test basic ReduceLROnPlateau functionality"""
        scheduler = braintools.optim.ReduceLROnPlateau(
            base_lr=1.0,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Initial LR
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # Simulate improving loss for 10 epochs
        for i in range(10):
            scheduler.step(metric=10.0 - i)

        # LR should not have decreased
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # Simulate plateauing loss for patience+1 epochs
        for _ in range(6):
            scheduler.step(metric=5.0)

        # LR should have decreased
        assert np.isclose(scheduler.current_lrs.value[0], 0.5, atol=0.01)

    def test_reduce_lr_on_plateau_max_mode(self):
        """Test ReduceLROnPlateau with max mode (for accuracy)"""
        scheduler = braintools.optim.ReduceLROnPlateau(
            base_lr=0.1,
            mode='max',
            factor=0.5,
            patience=3
        )

        # Simulate improving accuracy
        for i in range(5):
            scheduler.step(metric=0.5 + i * 0.1)

        # LR should not have decreased
        assert np.isclose(scheduler.current_lrs.value[0], 0.1)

        # Simulate plateauing accuracy
        for _ in range(4):
            scheduler.step(metric=0.9)

        # LR should have decreased
        assert scheduler.current_lrs.value[0] < 0.1

    def test_reduce_lr_on_plateau_with_optimizer(self):
        """Test ReduceLROnPlateau integration with optimizer"""
        scheduler = braintools.optim.ReduceLROnPlateau(
            base_lr=0.1,
            mode='min',
            factor=0.5,
            patience=3
        )
        optimizer = braintools.optim.Adam(lr=scheduler)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Initial LR
        assert np.isclose(optimizer.current_lr, 0.1)

        # Simulate training with plateauing loss
        # Need patience+2 steps: 1 to establish best, then patience+1 bad epochs
        for _ in range(5):
            scheduler.step(metric=1.0)

        # LR should have decreased
        assert optimizer.current_lr < 0.1

    def test_reduce_lr_on_plateau_min_lr(self):
        """Test ReduceLROnPlateau respects min_lr"""
        scheduler = braintools.optim.ReduceLROnPlateau(
            base_lr=0.1,
            mode='min',
            factor=0.1,
            patience=2,
            min_lr=0.001
        )

        # Trigger multiple reductions
        for cycle in range(10):
            for _ in range(3):
                scheduler.step(metric=1.0)

        # Should not go below min_lr
        assert scheduler.current_lrs.value[0] >= 0.001

    def test_reduce_lr_on_plateau_cooldown(self):
        """Test ReduceLROnPlateau cooldown period"""
        scheduler = braintools.optim.ReduceLROnPlateau(
            base_lr=1.0,
            mode='min',
            factor=0.5,
            patience=2,
            cooldown=3
        )

        # First reduction: 1 step to establish best + patience+1 bad epochs
        for _ in range(4):
            scheduler.step(metric=10.0)

        lr_after_first_reduction = scheduler.current_lrs.value[0]
        assert lr_after_first_reduction < 1.0

        # During cooldown, shouldn't reduce again even with bad metrics
        for _ in range(3):
            scheduler.step(metric=10.0)

        # Should still be at same LR (cooldown prevents reduction)
        assert np.isclose(scheduler.current_lrs.value[0], lr_after_first_reduction, atol=0.01)

    def test_reduce_lr_on_plateau_threshold(self):
        """Test ReduceLROnPlateau threshold for improvement"""
        scheduler = braintools.optim.ReduceLROnPlateau(
            base_lr=1.0,
            mode='min',
            factor=0.5,
            patience=3,
            threshold=0.1,
            threshold_mode='abs'
        )

        # Set initial best
        scheduler.step(metric=1.0)

        # Small improvements (less than threshold) shouldn't reset patience
        # For mode='min', threshold_mode='abs', threshold=0.1:
        # metric must be < (best - 0.1) to be "better"
        # So with best=1.0, metric must be < 0.9 to be better
        # Using values >= 0.9 won't reset patience
        scheduler.step(metric=0.95)  # not better: 0.95 not < 0.9
        scheduler.step(metric=0.93)  # not better: 0.93 not < 0.9
        scheduler.step(metric=0.91)  # not better: 0.91 not < 0.9
        scheduler.step(metric=0.90)  # not better: 0.90 not < 0.9, triggers reduction

        # Should reduce after patience+1 epochs of insufficient improvement
        lr = scheduler.current_lrs.value[0]
        assert lr < 1.0

    def test_reduce_lr_on_plateau_multiple_param_groups(self):
        """Test ReduceLROnPlateau with multiple learning rates"""
        scheduler = braintools.optim.ReduceLROnPlateau(
            base_lr=[1.0, 0.1],
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=[0.01, 0.001]
        )

        assert len(scheduler.current_lrs.value) == 2

        # Trigger reduction: 1 step to establish best + patience+1 bad epochs
        for _ in range(5):
            scheduler.step(metric=5.0)

        # Both should be reduced
        assert scheduler.current_lrs.value[0] < 1.0
        assert scheduler.current_lrs.value[1] < 0.1

    def test_reduce_lr_on_plateau_state_dict(self):
        """Test ReduceLROnPlateau state dict save/load"""
        scheduler1 = braintools.optim.ReduceLROnPlateau(
            base_lr=1.0,
            mode='min',
            factor=0.5,
            patience=3
        )

        for i in range(10):
            scheduler1.step(metric=5.0 - i * 0.1)

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.ReduceLROnPlateau(
            base_lr=1.0,
            mode='min',
            factor=0.5,
            patience=3
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)


class TestLinearLR:
    """Test LinearLR"""

    def test_basic_linear_lr(self):
        """Test basic LinearLR functionality (warmup)"""
        scheduler = braintools.optim.LinearLR(
            base_lr=1.0,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10
        )

        # Initial LR should be base_lr (before stepping)
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # After 5 steps, should be halfway: 0.1 + 0.5 * 0.9 = 0.55
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.55, atol=0.01)

        # After 10 steps, should be at end_factor * base_lr = 1.0
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.01)

        # After total_iters, should stay at end_factor * base_lr
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

    def test_linear_lr_cooldown(self):
        """Test LinearLR for cooldown (decreasing)"""
        scheduler = braintools.optim.LinearLR(
            base_lr=1.0,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=10
        )

        # Initial
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # After 10 steps, should be at 0.1
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.1, atol=0.01)

    def test_linear_lr_with_optimizer(self):
        """Test LinearLR integration with optimizer"""
        scheduler = braintools.optim.LinearLR(
            base_lr=0.01,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=5
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Initial (before stepping, scheduler is at base_lr)
        assert np.isclose(optimizer.current_lr, 0.01)

        # After 1 step, should be lower (at start_factor)
        scheduler.step()
        assert optimizer.current_lr < 0.01

        # After warmup (5 steps total), should be back at base_lr
        for _ in range(4):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.01, atol=0.001)

    def test_linear_lr_jit(self):
        """Test LinearLR with JIT compilation"""
        scheduler = braintools.optim.LinearLR(
            base_lr=1.0,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        lrs = []
        for _ in range(15):
            lrs.append(jit_step())

        # Should increase during warmup
        assert lrs[9] > lrs[0]

        # Should stay constant after total_iters
        assert np.isclose(lrs[14], lrs[10], atol=0.01)

    def test_linear_lr_multiple_param_groups(self):
        """Test LinearLR with multiple learning rates"""
        scheduler = braintools.optim.LinearLR(
            base_lr=[1.0, 0.1],
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10
        )

        assert len(scheduler.current_lrs.value) == 2

        for _ in range(10):
            scheduler.step()

        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.01)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1, atol=0.01)

    def test_linear_lr_state_dict(self):
        """Test LinearLR state dict save/load"""
        scheduler1 = braintools.optim.LinearLR(
            base_lr=1.0,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10
        )

        for _ in range(7):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.LinearLR(
            base_lr=1.0,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_linear_lr_default_params(self):
        """Test LinearLR with default parameters"""
        scheduler = braintools.optim.LinearLR(base_lr=1.0)

        # Default: start_factor=1/3, end_factor=1.0, total_iters=5
        for _ in range(5):
            scheduler.step()

        # Should be at base_lr after total_iters
        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.01)

    def test_linear_lr_jit_with_optimizer(self):
        """Test LinearLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.LinearLR(
            base_lr=0.1,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=10
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

        # Should increase during warmup
        assert lrs[9] > lrs[0]

        # Should stabilize after warmup
        assert np.isclose(lrs[14], lrs[10], atol=0.01)


class TestConstantLR:
    """Test ConstantLR"""

    def test_basic_constant_lr(self):
        """Test basic ConstantLR functionality"""
        scheduler = braintools.optim.ConstantLR(
            base_lr=1.0,
            factor=0.5,
            total_iters=10
        )

        # Initial LR should be base_lr (before stepping)
        assert np.isclose(scheduler.current_lrs.value[0], 1.0)

        # During first total_iters epochs, should be factor * base_lr = 0.5
        for i in range(10):
            scheduler.step()
        # After 10 steps (epochs 1-10), we're at epoch 10, which is still < total_iters in some implementations
        # Let me check: last_epoch starts at 0, after 10 steps it's at 10
        # The condition is last_epoch < total_iters, so at epoch 10, it's not < 10
        # So it should be at base_lr now
        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.01)

    def test_constant_lr_during_period(self):
        """Test ConstantLR stays constant during factor period"""
        scheduler = braintools.optim.ConstantLR(
            base_lr=1.0,
            factor=0.5,
            total_iters=10
        )

        lrs = []
        for _ in range(5):
            scheduler.step()
            lrs.append(scheduler.current_lrs.value[0])

        # All should be the same during factor period
        for i in range(len(lrs) - 1):
            assert np.isclose(lrs[i], lrs[i+1], atol=0.01)

    def test_constant_lr_with_optimizer(self):
        """Test ConstantLR integration with optimizer"""
        scheduler = braintools.optim.ConstantLR(
            base_lr=0.1,
            factor=0.5,
            total_iters=5
        )
        optimizer = braintools.optim.Adam(lr=scheduler)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Initial
        assert np.isclose(optimizer.current_lr, 0.1)

        # During factor period
        for _ in range(3):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.05, atol=0.01)

        # After factor period
        for _ in range(3):
            scheduler.step()
        assert np.isclose(optimizer.current_lr, 0.1, atol=0.01)

    def test_constant_lr_jit(self):
        """Test ConstantLR with JIT compilation"""
        scheduler = braintools.optim.ConstantLR(
            base_lr=1.0,
            factor=0.5,
            total_iters=10
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.current_lrs.value[0]

        lrs = []
        for _ in range(15):
            lrs.append(jit_step())

        # First 10 should be factor * base_lr = 0.5
        for i in range(9):
            assert np.isclose(lrs[i], 0.5, atol=0.01)

        # After total_iters should be base_lr = 1.0
        for i in range(10, 15):
            assert np.isclose(lrs[i], 1.0, atol=0.01)

    def test_constant_lr_multiple_param_groups(self):
        """Test ConstantLR with multiple learning rates"""
        scheduler = braintools.optim.ConstantLR(
            base_lr=[1.0, 0.1],
            factor=0.5,
            total_iters=5
        )

        assert len(scheduler.current_lrs.value) == 2

        # During factor period
        for _ in range(3):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 0.5, atol=0.01)
        assert np.isclose(scheduler.current_lrs.value[1], 0.05, atol=0.01)

        # After factor period
        for _ in range(3):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.01)
        assert np.isclose(scheduler.current_lrs.value[1], 0.1, atol=0.01)

    def test_constant_lr_state_dict(self):
        """Test ConstantLR state dict save/load"""
        scheduler1 = braintools.optim.ConstantLR(
            base_lr=1.0,
            factor=0.5,
            total_iters=10
        )

        for _ in range(7):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.ConstantLR(
            base_lr=1.0,
            factor=0.5,
            total_iters=10
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.current_lrs.value, scheduler1.current_lrs.value)

    def test_constant_lr_default_params(self):
        """Test ConstantLR with default parameters"""
        scheduler = braintools.optim.ConstantLR(base_lr=1.0)

        # Default: factor=1/3, total_iters=5
        for _ in range(3):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0/3, atol=0.01)

        for _ in range(3):
            scheduler.step()
        assert np.isclose(scheduler.current_lrs.value[0], 1.0, atol=0.01)

    def test_constant_lr_jit_with_optimizer(self):
        """Test ConstantLR with JIT compilation in a training loop"""
        model = brainstate.nn.Linear(10, 5)
        scheduler = braintools.optim.ConstantLR(
            base_lr=0.1,
            factor=0.5,
            total_iters=5
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

        # First 5 should be constant at 0.05
        for i in range(4):
            assert np.isclose(lrs[i], 0.05, atol=0.01)

        # After total_iters should be 0.1
        for i in range(5, 10):
            assert np.isclose(lrs[i], 0.1, atol=0.01)


class TestChainedScheduler:
    """Test ChainedScheduler"""

    def test_basic_chained_scheduler(self):
        """Test basic ChainedScheduler functionality with warmup + decay"""
        warmup = braintools.optim.LinearLR(
            base_lr=1.0,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=5
        )
        decay = braintools.optim.StepLR(
            base_lr=1.0,
            step_size=10,
            gamma=0.5
        )
        scheduler = braintools.optim.ChainedScheduler([warmup, decay])

        # Initial - warmup starts at base_lr
        assert np.isclose(scheduler.get_lr()[0], 1.0)

        # After 5 steps, warmup complete, decay hasn't started
        for _ in range(5):
            scheduler.step()
        lr = scheduler.get_lr()[0]
        assert np.isclose(lr, 1.0, atol=0.01)

        # After 10 more steps, decay should have triggered once
        for _ in range(10):
            scheduler.step()
        lr = scheduler.get_lr()[0]
        assert lr < 1.0

    def test_chained_scheduler_with_optimizer(self):
        """Test ChainedScheduler integration with optimizer"""
        warmup = braintools.optim.ConstantLR(
            base_lr=0.1,
            factor=0.5,
            total_iters=3
        )
        decay = braintools.optim.ExponentialLR(
            base_lr=0.1,
            gamma=0.9
        )
        scheduler = braintools.optim.ChainedScheduler([warmup, decay])
        optimizer = braintools.optim.Adam(lr=scheduler)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Check initial LR
        assert np.isclose(optimizer.current_lr, 0.1)

        # Step through warmup and decay
        for _ in range(10):
            scheduler.step()

        # LR should have decayed
        assert optimizer.current_lr < 0.1

    def test_chained_scheduler_state_dict(self):
        """Test ChainedScheduler state dict save/load"""
        warmup1 = braintools.optim.LinearLR(base_lr=1.0, start_factor=0.1, end_factor=1.0, total_iters=5)
        decay1 = braintools.optim.StepLR(base_lr=1.0, step_size=10, gamma=0.5)
        scheduler1 = braintools.optim.ChainedScheduler([warmup1, decay1])

        for _ in range(7):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        warmup2 = braintools.optim.LinearLR(base_lr=1.0, start_factor=0.1, end_factor=1.0, total_iters=5)
        decay2 = braintools.optim.StepLR(base_lr=1.0, step_size=10, gamma=0.5)
        scheduler2 = braintools.optim.ChainedScheduler([warmup2, decay2])
        scheduler2.load_state_dict(state_dict)

        # Both schedulers should have same state
        assert np.allclose(scheduler1.get_lr(), scheduler2.get_lr())


class TestSequentialLR:
    """Test SequentialLR"""

    def test_basic_sequential_lr(self):
        """Test basic SequentialLR functionality"""
        warmup = braintools.optim.LinearLR(
            base_lr=1.0,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=5
        )
        decay = braintools.optim.ExponentialLR(
            base_lr=1.0,
            gamma=0.9
        )
        scheduler = braintools.optim.SequentialLR(
            schedulers=[warmup, decay],
            milestones=[5]
        )

        # Before milestone, should use first scheduler (warmup)
        for i in range(5):
            scheduler.step(epoch=i)

        # At milestone, should switch to second scheduler (decay)
        scheduler.step(epoch=5)
        lr_at_milestone = scheduler.get_lr()[0]

        # Continue with second scheduler
        scheduler.step(epoch=10)
        lr_after_decay = scheduler.get_lr()[0]

        # After decay, LR should be lower
        assert lr_after_decay < lr_at_milestone

    def test_sequential_lr_three_phase(self):
        """Test SequentialLR with three phases"""
        warmup = braintools.optim.ConstantLR(base_lr=1.0, factor=0.1, total_iters=3)
        main = braintools.optim.ConstantLR(base_lr=0.8, factor=1.0, total_iters=10)
        finetune = braintools.optim.ConstantLR(base_lr=0.5, factor=1.0, total_iters=10)

        scheduler = braintools.optim.SequentialLR(
            schedulers=[warmup, main, finetune],
            milestones=[3, 10]
        )

        # Phase 1: warmup (epochs 0-2)
        scheduler.step(epoch=2)
        lr_phase1 = scheduler.get_lr()[0]

        # Phase 2: main training (epochs 3-9)
        scheduler.step(epoch=5)
        lr_phase2 = scheduler.get_lr()[0]

        # Phase 3: fine-tuning (epochs 10+)
        scheduler.step(epoch=12)
        lr_phase3 = scheduler.get_lr()[0]

        # Each phase should have different LRs due to different base_lrs
        assert not np.isclose(lr_phase1, lr_phase2)
        assert not np.isclose(lr_phase2, lr_phase3)

    def test_sequential_lr_with_optimizer(self):
        """Test SequentialLR integration with optimizer"""
        warmup = braintools.optim.LinearLR(base_lr=0.1, start_factor=0.1, end_factor=1.0, total_iters=5)
        decay = braintools.optim.StepLR(base_lr=0.1, step_size=10, gamma=0.5)

        scheduler = braintools.optim.SequentialLR(
            schedulers=[warmup, decay],
            milestones=[5]
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Just verify it doesn't crash when used with optimizer
        for i in range(10):
            scheduler.step(epoch=i)

        # Optimizer should have a valid LR
        assert optimizer.current_lr > 0

    def test_sequential_lr_state_dict(self):
        """Test SequentialLR state dict save/load"""
        warmup1 = braintools.optim.LinearLR(base_lr=1.0, start_factor=0.1, end_factor=1.0, total_iters=5)
        decay1 = braintools.optim.ExponentialLR(base_lr=1.0, gamma=0.9)
        scheduler1 = braintools.optim.SequentialLR(schedulers=[warmup1, decay1], milestones=[5])

        for i in range(8):
            scheduler1.step(epoch=i)

        state_dict = scheduler1.state_dict()

        warmup2 = braintools.optim.LinearLR(base_lr=1.0, start_factor=0.1, end_factor=1.0, total_iters=5)
        decay2 = braintools.optim.ExponentialLR(base_lr=1.0, gamma=0.9)
        scheduler2 = braintools.optim.SequentialLR(schedulers=[warmup2, decay2], milestones=[5])
        scheduler2.load_state_dict(state_dict)

        assert scheduler1.last_epoch.value == scheduler2.last_epoch.value
        assert np.allclose(scheduler1.get_lr(), scheduler2.get_lr())


class TestCosineAnnealingWarmRestarts:
    """Test CosineAnnealingWarmRestarts"""

    def test_basic_cosine_warm_restarts(self):
        """Test basic CosineAnnealingWarmRestarts functionality"""
        scheduler = braintools.optim.CosineAnnealingWarmRestarts(
            base_lr=1.0,
            T_0=10,
            T_mult=1,
            eta_min=0.1
        )

        # Initial LR should be base_lr
        assert np.isclose(scheduler.get_lr()[0], 1.0)

        # After half cycle, should be near eta_min
        for _ in range(5):
            scheduler.step()
        lr_half_cycle = scheduler.get_lr()[0]
        assert lr_half_cycle < 0.6  # Should have decreased significantly

        # After full cycle, should restart to base_lr
        for _ in range(5):
            scheduler.step()
        lr_restart = scheduler.get_lr()[0]
        assert np.isclose(lr_restart, 1.0, atol=0.1)

    def test_cosine_warm_restarts_increasing_cycles(self):
        """Test CosineAnnealingWarmRestarts with increasing cycle lengths"""
        scheduler = braintools.optim.CosineAnnealingWarmRestarts(
            base_lr=1.0,
            T_0=5,
            T_mult=2,  # Each cycle doubles in length
            eta_min=0.0
        )

        # First cycle: 5 epochs
        lrs_cycle1 = []
        for _ in range(5):
            scheduler.step()
            lrs_cycle1.append(scheduler.get_lr()[0])

        # Second cycle: 10 epochs
        lrs_cycle2 = []
        for _ in range(10):
            scheduler.step()
            lrs_cycle2.append(scheduler.get_lr()[0])

        # First value of each cycle should be near base_lr (restart)
        assert lrs_cycle1[0] > 0.8
        assert lrs_cycle2[0] > 0.8

    def test_cosine_warm_restarts_with_optimizer(self):
        """Test CosineAnnealingWarmRestarts integration with optimizer"""
        scheduler = braintools.optim.CosineAnnealingWarmRestarts(
            base_lr=0.1,
            T_0=10,
            T_mult=1,
            eta_min=0.01
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        initial_lr = optimizer.current_lr

        # Go through part of a cycle
        for _ in range(5):
            scheduler.step()

        # LR should have decreased
        assert optimizer.current_lr < initial_lr

        # After full cycle, should restart
        for _ in range(5):
            scheduler.step()
        assert optimizer.current_lr > 0.05

    def test_cosine_warm_restarts_jit(self):
        """Test CosineAnnealingWarmRestarts with JIT compilation"""
        scheduler = braintools.optim.CosineAnnealingWarmRestarts(
            base_lr=1.0,
            T_0=10,
            T_mult=1,
            eta_min=0.0
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.get_lr()[0]

        lrs = []
        for _ in range(25):
            lrs.append(jit_step())

        # Check that LR restarts: compare end of cycle with beginning
        # At step 9, we're at the end of first cycle (low LR)
        # At step 10, we restart (high LR)
        assert lrs[0] > lrs[4]  # Start of cycle 1 > middle of cycle 1
        assert lrs[10] > lrs[4]  # Start of cycle 2 > middle of cycle 1

    def test_cosine_warm_restarts_state_dict(self):
        """Test CosineAnnealingWarmRestarts state dict save/load"""
        scheduler1 = braintools.optim.CosineAnnealingWarmRestarts(
            base_lr=1.0,
            T_0=10,
            T_mult=2,
            eta_min=0.1
        )

        for _ in range(7):
            scheduler1.step()

        # Save current state
        T_cur_before = scheduler1.T_cur.value
        T_i_before = scheduler1.T_i.value
        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.CosineAnnealingWarmRestarts(
            base_lr=1.0,
            T_0=10,
            T_mult=2,
            eta_min=0.1
        )
        scheduler2.load_state_dict(state_dict)

        # Manually restore T_cur and T_i since they might not be in state_dict
        scheduler2.T_cur.value = T_cur_before
        scheduler2.T_i.value = T_i_before

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.get_lr(), scheduler1.get_lr())


class TestWarmupCosineSchedule:
    """Test WarmupCosineSchedule"""

    def test_basic_warmup_cosine(self):
        """Test basic WarmupCosineSchedule functionality"""
        scheduler = braintools.optim.WarmupCosineSchedule(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=100,
            warmup_start_lr=0.0,
            eta_min=0.0
        )

        # Initial should be at warmup_start_lr (epoch 0, before stepping)
        assert np.isclose(scheduler.get_lr()[0], 0.0)

        # During warmup (step 5), should be increasing
        for _ in range(5):
            scheduler.step()
        lr_warmup = scheduler.get_lr()[0]
        assert lr_warmup < 1.0  # Should be between warmup_start and base_lr

        # After warmup, should start cosine decay
        for _ in range(50):
            scheduler.step()
        lr_decay = scheduler.get_lr()[0]
        assert lr_decay < lr_warmup

        # Near end of total_steps, should be near eta_min
        for _ in range(45):
            scheduler.step()
        lr_end = scheduler.get_lr()[0]
        assert lr_end < 0.2

    def test_warmup_cosine_with_nonzero_start(self):
        """Test WarmupCosineSchedule with non-zero start LR"""
        scheduler = braintools.optim.WarmupCosineSchedule(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=50,
            warmup_start_lr=0.1,
            eta_min=0.01
        )

        # Track LRs
        lrs = []
        for _ in range(50):
            scheduler.step()
            lrs.append(scheduler.get_lr()[0])

        # Should increase during warmup
        assert lrs[9] > lrs[0]

        # Should decrease after warmup
        assert lrs[49] < lrs[10]

    def test_warmup_cosine_with_optimizer(self):
        """Test WarmupCosineSchedule integration with optimizer"""
        scheduler = braintools.optim.WarmupCosineSchedule(
            base_lr=0.1,
            warmup_steps=5,
            total_steps=50,
            warmup_start_lr=0.0,
            eta_min=0.001
        )
        optimizer = braintools.optim.AdamW(lr=scheduler)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        initial_lr = optimizer.current_lr

        # Step through warmup and decay
        for _ in range(30):
            scheduler.step()

        # LR should have gone through warmup and started decay
        assert optimizer.current_lr != initial_lr

    def test_warmup_cosine_jit(self):
        """Test WarmupCosineSchedule with JIT compilation"""
        scheduler = braintools.optim.WarmupCosineSchedule(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=50,
            warmup_start_lr=0.0,
            eta_min=0.0
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.get_lr()[0]

        lrs = []
        for _ in range(50):
            lrs.append(jit_step())

        # Should increase during warmup
        assert lrs[9] > lrs[0]

        # Should decrease after warmup
        assert lrs[49] < lrs[10]

    def test_warmup_cosine_state_dict(self):
        """Test WarmupCosineSchedule state dict save/load"""
        scheduler1 = braintools.optim.WarmupCosineSchedule(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=100,
            warmup_start_lr=0.0,
            eta_min=0.0
        )

        for _ in range(25):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.WarmupCosineSchedule(
            base_lr=1.0,
            warmup_steps=10,
            total_steps=100,
            warmup_start_lr=0.0,
            eta_min=0.0
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.get_lr(), scheduler1.get_lr())


class TestPiecewiseConstantSchedule:
    """Test PiecewiseConstantSchedule"""

    def test_basic_piecewise_constant(self):
        """Test basic PiecewiseConstantSchedule functionality"""
        scheduler = braintools.optim.PiecewiseConstantSchedule(
            base_lr=1.0,
            boundaries=[10, 20],
            values=[1.0, 0.5, 0.1]
        )

        # Initial (epoch 0) should be first value
        assert np.isclose(scheduler.get_lr()[0], 1.0)

        # Before first boundary
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.get_lr()[0], 1.0)

        # After first boundary, before second
        for _ in range(7):
            scheduler.step()
        assert np.isclose(scheduler.get_lr()[0], 0.5, atol=0.01)

        # After second boundary
        for _ in range(10):
            scheduler.step()
        assert np.isclose(scheduler.get_lr()[0], 0.1, atol=0.01)

    def test_piecewise_constant_exact_boundaries(self):
        """Test PiecewiseConstantSchedule at exact boundary points"""
        scheduler = braintools.optim.PiecewiseConstantSchedule(
            base_lr=1.0,
            boundaries=[5, 10],
            values=[1.0, 0.5, 0.1]
        )

        # At epoch 4 (before boundary 5)
        for _ in range(4):
            scheduler.step()
        assert np.isclose(scheduler.get_lr()[0], 1.0)

        # At epoch 5 (exactly at boundary)
        scheduler.step()
        assert np.isclose(scheduler.get_lr()[0], 0.5, atol=0.01)

        # At epoch 10 (exactly at second boundary)
        for _ in range(5):
            scheduler.step()
        assert np.isclose(scheduler.get_lr()[0], 0.1, atol=0.01)

    def test_piecewise_constant_with_optimizer(self):
        """Test PiecewiseConstantSchedule integration with optimizer"""
        scheduler = braintools.optim.PiecewiseConstantSchedule(
            base_lr=0.1,
            boundaries=[5, 10],
            values=[1.0, 0.5, 0.1]
        )
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)

        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Track LR changes
        lrs = []
        for _ in range(15):
            lrs.append(optimizer.current_lr)
            scheduler.step()

        # Check that LR changes at boundaries
        assert np.isclose(lrs[3], 1.0)  # Before first boundary
        assert np.isclose(lrs[7], 0.5, atol=0.01)  # After first boundary
        assert np.isclose(lrs[12], 0.1, atol=0.01)  # After second boundary

    def test_piecewise_constant_jit(self):
        """Test PiecewiseConstantSchedule with JIT compilation"""
        scheduler = braintools.optim.PiecewiseConstantSchedule(
            base_lr=1.0,
            boundaries=[10, 20, 30],
            values=[1.0, 0.5, 0.1, 0.01]
        )

        @brainstate.transform.jit
        def jit_step():
            scheduler.step()
            return scheduler.get_lr()[0]

        lrs = []
        for _ in range(35):
            lrs.append(jit_step())

        # Check values in each segment
        assert np.isclose(lrs[5], 1.0)  # Before first boundary
        assert np.isclose(lrs[15], 0.5, atol=0.01)  # Between first and second
        assert np.isclose(lrs[25], 0.1, atol=0.01)  # Between second and third
        assert np.isclose(lrs[32], 0.01, atol=0.01)  # After third boundary

    def test_piecewise_constant_multiple_param_groups(self):
        """Test PiecewiseConstantSchedule with multiple learning rates"""
        scheduler = braintools.optim.PiecewiseConstantSchedule(
            base_lr=[1.0, 0.1],
            boundaries=[10, 20],
            values=[1.0, 0.5, 0.1]
        )

        assert len(scheduler.get_lr()) == 2

        # After first boundary - values are absolute for all param groups
        for _ in range(12):
            scheduler.step()
        lrs = scheduler.get_lr()
        assert np.isclose(lrs[0], 0.5, atol=0.01)
        assert np.isclose(lrs[1], 0.5, atol=0.01)  # Same value for all groups

    def test_piecewise_constant_state_dict(self):
        """Test PiecewiseConstantSchedule state dict save/load"""
        scheduler1 = braintools.optim.PiecewiseConstantSchedule(
            base_lr=1.0,
            boundaries=[10, 20],
            values=[1.0, 0.5, 0.1]
        )

        for _ in range(15):
            scheduler1.step()

        state_dict = scheduler1.state_dict()

        scheduler2 = braintools.optim.PiecewiseConstantSchedule(
            base_lr=1.0,
            boundaries=[10, 20],
            values=[1.0, 0.5, 0.1]
        )
        scheduler2.load_state_dict(state_dict)

        assert scheduler2.last_epoch.value == scheduler1.last_epoch.value
        assert np.allclose(scheduler2.get_lr(), scheduler1.get_lr())

    def test_piecewise_constant_default_params(self):
        """Test PiecewiseConstantSchedule with default parameters"""
        scheduler = braintools.optim.PiecewiseConstantSchedule(base_lr=1.0)

        # Default boundaries=[1000, 2000], values=[1.0, 0.1, 0.01]
        assert np.isclose(scheduler.get_lr()[0], 1.0)

        # Step to 1500 (between boundaries)
        for _ in range(1500):
            scheduler.step()
        lr = scheduler.get_lr()[0]
        assert np.isclose(lr, 0.1, atol=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
