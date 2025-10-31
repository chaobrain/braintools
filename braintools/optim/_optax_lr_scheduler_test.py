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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
