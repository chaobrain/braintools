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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
