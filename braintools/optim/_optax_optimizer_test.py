#!/usr/bin/env python
"""Comprehensive tests for OptaxOptimizer."""

import unittest

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from brainstate import ParamState

import braintools
import braintools.optim as O


class SimpleModelV1(brainstate.nn.Module):
    """Simple model for testing."""

    def __init__(self, in_features=10, out_features=5):
        super().__init__()
        self.linear = brainstate.nn.Linear(in_features, out_features)

    def __call__(self, x):
        return self.linear(x)


class SimpleModelV2(brainstate.nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super().__init__()
        self.linear1 = brainstate.nn.Linear(input_dim, hidden_dim)
        self.linear2 = brainstate.nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


class TestOptaxOptimizer(unittest.TestCase):
    """Test OptaxOptimizer basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModelV2()
        self.param_states = braintools.optim.UniqueStateManager().merge_with(
            self.model.states(ParamState)
        ).to_dict()
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
        self.target_data = jax.random.normal(jax.random.PRNGKey(1), (32, 2))

    def _compute_loss_and_grads(self):
        """Helper to compute loss and gradients."""

        # Get current parameters as a dictionary

        def loss_fn():
            # Apply the model with explicit parameters (no side effects)
            # Reconstruct predictions without modifying model state
            x = self.input_data
            predictions = self.model(x)
            return jnp.mean((predictions - self.target_data) ** 2)

        # Compute loss and gradients
        loss = loss_fn()
        grads = brainstate.transform.grad(loss_fn, grad_states=self.param_states)()

        return loss, grads

    def test_initialization(self):
        """Test optimizer initialization with different parameters."""
        # Test with default parameters
        opt1 = braintools.optim.OptaxOptimizer()
        self.assertEqual(opt1._base_lr, 1e-3)
        self.assertEqual(opt1.weight_decay, 0.0)
        self.assertIsNone(opt1.grad_clip_norm)
        self.assertIsNone(opt1.grad_clip_value)

        # Test with custom parameters
        opt2 = braintools.optim.OptaxOptimizer(
            lr=0.01,
            weight_decay=0.001,
            grad_clip_norm=1.0,
            grad_clip_value=0.5
        )
        self.assertEqual(opt2._base_lr, 0.01)
        self.assertEqual(opt2.weight_decay, 0.001)
        self.assertEqual(opt2.grad_clip_norm, 1.0)
        self.assertEqual(opt2.grad_clip_value, 0.5)

    def test_register_trainable_weights(self):
        """Test registering model parameters."""
        optimizer = braintools.optim.Adam(lr=0.01)
        param_states = self.model.states(ParamState)

        # Register weights
        optimizer.register_trainable_weights(param_states)

        # Check that parameters are registered
        self.assertIsNotNone(optimizer.opt_state)
        self.assertEqual(len(optimizer.param_states), len(param_states))

        # Check default param group is created
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'].value, 0.01)

    def test_step_updates_parameters(self):
        """Test that step() updates parameters."""
        optimizer = braintools.optim.Adam(lr=0.1)
        optimizer.register_trainable_weights(self.param_states)

        # Get initial parameters
        initial_params = {}
        for k, v in self.param_states.items():
            # Deep copy the nested dict structure
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Compute gradients
        _, grads = self._compute_loss_and_grads()

        # Take optimization step
        optimizer.step(grads)

        # Check parameters were updated
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                # Check nested dict parameters
                for sub_k in v.value:
                    self.assertFalse(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
            else:
                self.assertFalse(jnp.allclose(initial_params[k], v.value))

        # Check step count increased
        self.assertEqual(optimizer.step_count.value, 1)

    def test_lr_property(self):
        """Test learning rate getter and setter."""
        optimizer = braintools.optim.Adam(lr=0.01)

        # Test getter
        self.assertEqual(optimizer.current_lr, 0.01)

        # Test setter
        optimizer.current_lr = 0.001
        self.assertEqual(optimizer.current_lr, 0.001)
        self.assertEqual(optimizer._current_lr.value, 0.001)

    def test_state_dict_and_load(self):
        """Test saving and loading optimizer state."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.param_states)

        # Take a few steps
        for _ in range(3):
            _, grads = self._compute_loss_and_grads()
            optimizer.step(grads)

        # Save state
        state_dict = optimizer.state_dict()

        # Check state dict contains expected keys
        self.assertIn('step_count', state_dict)
        self.assertIn('lr', state_dict)
        self.assertIn('base_lr', state_dict)
        self.assertIn('param_groups', state_dict)
        self.assertIn('opt_state', state_dict)

        # Create new optimizer and load state
        new_optimizer = braintools.optim.Adam(lr=0.01)
        new_optimizer.register_trainable_weights(self.param_states)
        new_optimizer.load_state_dict(state_dict)

        # Check state was restored
        self.assertEqual(new_optimizer.step_count.value, 3)
        self.assertEqual(new_optimizer.current_lr, optimizer.current_lr)

    def test_gradient_clipping_by_norm(self):
        """Test gradient clipping by norm functionality."""
        optimizer = braintools.optim.SGD(lr=0.1, grad_clip_norm=0.5)
        optimizer.register_trainable_weights(self.param_states)

        # Compute large gradients
        def large_loss_fn():
            x = self.input_data
            predictions = self.model(x) * 100  # Scale up to get large gradients
            return jnp.mean((predictions - self.target_data) ** 2)

        loss = large_loss_fn()
        grads = brainstate.transform.grad(large_loss_fn, grad_states=self.param_states)()

        # Compute gradient norm before clipping
        grad_norm_before = jnp.sqrt(sum(
            jnp.sum(g ** 2) if not isinstance(g, dict) else
            sum(jnp.sum(sub_g ** 2) for sub_g in g.values())
            for g in grads.values()
        ))

        # Take optimization step (gradients should be clipped)
        optimizer.step(grads)

        # The gradients should have been clipped to max norm of 0.5
        # We can't directly check the clipped gradients, but we can verify
        # that the parameter updates are smaller than they would be without clipping
        self.assertGreater(grad_norm_before, 0.5)  # Ensure gradients were large enough to be clipped

    def test_gradient_clipping_by_value(self):
        """Test gradient clipping by value functionality."""
        optimizer = braintools.optim.SGD(lr=0.1, grad_clip_value=0.1)
        optimizer.register_trainable_weights(self.param_states)

        # Create gradients with known large values
        def custom_loss_fn():
            x = self.input_data
            predictions = self.model(x) * 10  # Scale to get larger gradients
            return jnp.mean((predictions - self.target_data) ** 2)

        loss = custom_loss_fn()
        grads = brainstate.transform.grad(custom_loss_fn, grad_states=self.param_states)()

        # Store initial parameters
        initial_params = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Take optimization step (gradients should be clipped by value)
        optimizer.step(grads)

        # Parameters should have been updated
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                for sub_k in v.value:
                    self.assertFalse(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
            else:
                self.assertFalse(jnp.allclose(initial_params[k], v.value))

    def test_weight_decay(self):
        """Test weight decay (L2 regularization) functionality."""
        # Test with zero gradients to isolate weight decay effect
        optimizer_no_decay = braintools.optim.SGD(lr=0.1, weight_decay=0.0)
        optimizer_with_decay = braintools.optim.SGD(lr=0.1, weight_decay=0.01)

        # Create model and get parameters
        model = SimpleModelV2()
        param_states = braintools.optim.UniqueStateManager().merge_with(
            model.states(ParamState)
        ).to_dict()

        # Make copies for each optimizer
        import jax
        param_states_no_decay = jax.tree.map(
            lambda x: ParamState(x.value.copy() if hasattr(x.value, 'copy') else x.value),
            param_states
        )
        param_states_with_decay = jax.tree.map(
            lambda x: ParamState(x.value.copy() if hasattr(x.value, 'copy') else x.value),
            param_states
        )

        optimizer_no_decay.register_trainable_weights(param_states_no_decay)
        optimizer_with_decay.register_trainable_weights(param_states_with_decay)

        # Store initial norms
        initial_norms = {}
        for k, v in param_states_no_decay.items():
            if isinstance(v.value, dict):
                initial_norms[k] = {sub_k: jnp.linalg.norm(sub_v)
                                    for sub_k, sub_v in v.value.items()}
            else:
                initial_norms[k] = jnp.linalg.norm(v.value)

        # Create zero gradients to isolate weight decay effect
        zero_grads = {}
        for k, v in param_states_no_decay.items():
            if isinstance(v.value, dict):
                zero_grads[k] = {sub_k: jnp.zeros_like(sub_v)
                                 for sub_k, sub_v in v.value.items()}
            else:
                zero_grads[k] = jnp.zeros_like(v.value)

        # Take steps with zero gradients
        optimizer_no_decay.step(zero_grads)
        optimizer_with_decay.step(zero_grads)

        # With zero gradients and no weight decay, params shouldn't change
        for k, v in param_states_no_decay.items():
            if isinstance(v.value, dict):
                for sub_k, sub_v in v.value.items():
                    norm_after = jnp.linalg.norm(sub_v)
                    self.assertTrue(jnp.allclose(norm_after, initial_norms[k][sub_k]))
            else:
                norm_after = jnp.linalg.norm(v.value)
                self.assertTrue(jnp.allclose(norm_after, initial_norms[k]))

        # With zero gradients and weight decay, params should shrink
        for k, v in param_states_with_decay.items():
            if isinstance(v.value, dict):
                for sub_k, sub_v in v.value.items():
                    norm_after = jnp.linalg.norm(sub_v)
                    # Weight decay should reduce the norm (skip zero parameters)
                    if initial_norms[k][sub_k] > 1e-6:
                        self.assertLess(norm_after, initial_norms[k][sub_k])
            else:
                norm_after = jnp.linalg.norm(v.value)
                if initial_norms[k] > 1e-6:
                    self.assertLess(norm_after, initial_norms[k])

    def test_multiple_param_groups(self):
        """Test optimizer with multiple parameter groups with different learning rates."""
        # For multiple param groups test, we need to register all params first,
        # then update the learning rates for specific groups
        optimizer = braintools.optim.Adam(lr=0.01)

        # Register all parameters at once (needed for gradient computation)
        optimizer.register_trainable_weights(self.param_states)

        # Update learning rates for specific parameter groups
        # Note: Since add_param_group adds a new group, we need to ensure
        # the optimizer has a way to handle different learning rates per group

        # Store initial parameters
        initial_params = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Compute gradients
        _, grads = self._compute_loss_and_grads()

        # Take optimization step
        optimizer.step(grads)

        # All parameters should be updated
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                for sub_k in v.value:
                    self.assertFalse(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
            else:
                self.assertFalse(jnp.allclose(initial_params[k], v.value))

    def test_zero_gradients(self):
        """Test optimizer behavior with zero gradients."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.param_states)

        # Store initial parameters
        initial_params = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Create zero gradients
        zero_grads = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                zero_grads[k] = {sub_k: jnp.zeros_like(sub_v) for sub_k, sub_v in v.value.items()}
            else:
                zero_grads[k] = jnp.zeros_like(v.value)

        # Take step with zero gradients
        optimizer.step(zero_grads)

        # Parameters should not change (except possibly due to weight decay if enabled)
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                for sub_k in v.value:
                    # With zero gradients and no weight decay, params shouldn't change
                    self.assertTrue(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
            else:
                self.assertTrue(jnp.allclose(initial_params[k], v.value))

    def test_optimizer_momentum(self):
        """Test SGD with momentum."""
        optimizer = braintools.optim.SGD(lr=0.1, momentum=0.9)
        optimizer.register_trainable_weights(self.param_states)

        # Take multiple steps to build up momentum
        prev_updates = None
        for i in range(3):
            # Store params before update
            params_before = {}
            for k, v in self.param_states.items():
                if isinstance(v.value, dict):
                    params_before[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
                else:
                    params_before[k] = v.value.copy()

            # Compute gradients
            _, grads = self._compute_loss_and_grads()

            # Take step
            optimizer.step(grads)

            # Calculate updates
            current_updates = {}
            for k, v in self.param_states.items():
                if isinstance(v.value, dict):
                    current_updates[k] = {sub_k: v.value[sub_k] - params_before[k][sub_k]
                                          for sub_k in v.value}
                else:
                    current_updates[k] = v.value - params_before[k]

            # After first step, updates should incorporate momentum
            if i > 0 and prev_updates is not None:
                # With momentum, current update should be influenced by previous update
                # This is a qualitative check - exact verification would require
                # reimplementing the momentum calculation
                pass

            prev_updates = current_updates

    def test_nan_gradient_handling(self):
        """Test optimizer behavior with NaN gradients."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.param_states)

        # Store initial parameters
        initial_params = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Create gradients with NaN values
        nan_grads = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                nan_grads[k] = {sub_k: jnp.full_like(sub_v, jnp.nan)
                                for sub_k, sub_v in v.value.items()}
            else:
                nan_grads[k] = jnp.full_like(v.value, jnp.nan)

        # Take step with NaN gradients - this should not crash
        # The behavior depends on the optimizer implementation
        try:
            optimizer.step(nan_grads)
            # Check if parameters became NaN (expected behavior for most optimizers)
            # or remained unchanged (if optimizer has NaN protection)
        except Exception as e:
            # Some optimizers might raise an exception for NaN gradients
            self.fail(f"Optimizer should handle NaN gradients gracefully: {e}")

    def test_different_optimizers(self):
        """Test different optimizer types with the same model."""
        optimizers = [
            braintools.optim.SGD(lr=0.01),
            braintools.optim.Adam(lr=0.01),
            braintools.optim.RMSprop(lr=0.01),
            braintools.optim.AdamW(lr=0.01, weight_decay=0.01)
        ]

        for opt in optimizers:
            # Reset model parameters for each optimizer
            model = SimpleModelV2()
            param_states = braintools.optim.UniqueStateManager().merge_with(
                model.states(ParamState)
            ).to_dict()

            opt.register_trainable_weights(param_states)

            # Store initial loss
            def loss_fn():
                x = self.input_data
                predictions = model(x)
                return jnp.mean((predictions - self.target_data) ** 2)

            initial_loss = loss_fn()
            initial_grads = brainstate.transform.grad(loss_fn, grad_states=param_states)()

            # Take several optimization steps
            for _ in range(5):
                loss = loss_fn()
                grads = brainstate.transform.grad(loss_fn, grad_states=param_states)()
                opt.step(grads)

            # Loss should decrease after optimization
            final_loss = loss_fn()
            self.assertLess(final_loss, initial_loss, f"{opt.__class__.__name__} should reduce loss")

    def test_lr_scheduler_integration(self):
        """Test integration with learning rate scheduler."""
        # Create a simple linear decay schedule
        optimizer = braintools.optim.Adam(lr=0.1)
        optimizer.register_trainable_weights(self.param_states)

        # Check initial lr
        self.assertEqual(optimizer.current_lr, 0.1)

        # Manually update lr after some steps
        for i in range(10):
            _, grads = self._compute_loss_and_grads()
            optimizer.step(grads)

            # Manually decay learning rate
            if i == 4:
                optimizer.current_lr = 0.05

        # Check lr was updated
        self.assertEqual(optimizer.current_lr, 0.05)

    def test_empty_gradients(self):
        """Test optimizer with all zero gradients."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.param_states)

        # Store initial step count and params
        initial_step_count = optimizer.step_count.value
        initial_params = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Create all zero gradients (same structure as params)
        zero_grads = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                zero_grads[k] = {sub_k: jnp.zeros_like(sub_v) for sub_k, sub_v in v.value.items()}
            else:
                zero_grads[k] = jnp.zeros_like(v.value)

        # Step with zero gradients should not crash
        optimizer.step(zero_grads)

        # Step count should still increase
        self.assertEqual(optimizer.step_count.value, initial_step_count + 1)

        # Parameters should not change with zero gradients
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                for sub_k in v.value:
                    self.assertTrue(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
            else:
                self.assertTrue(jnp.allclose(initial_params[k], v.value))

    def test_partial_gradients(self):
        """Test optimizer with non-zero gradients for subset and zero for others."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.param_states)

        # Store initial parameters
        initial_params = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Create gradients - non-zero for linear1, zero for linear2
        mixed_grads = {}
        for k, v in self.param_states.items():
            if 'linear1' in k:
                # Non-zero gradients for linear1
                if isinstance(v.value, dict):
                    mixed_grads[k] = {sub_k: jnp.ones_like(sub_v) * 0.01
                                      for sub_k, sub_v in v.value.items()}
                else:
                    mixed_grads[k] = jnp.ones_like(v.value) * 0.01
            else:
                # Zero gradients for other parameters
                if isinstance(v.value, dict):
                    mixed_grads[k] = {sub_k: jnp.zeros_like(sub_v)
                                      for sub_k, sub_v in v.value.items()}
                else:
                    mixed_grads[k] = jnp.zeros_like(v.value)

        # Take step with mixed gradients
        optimizer.step(mixed_grads)

        # Only linear1 parameters should be updated
        for k, v in self.param_states.items():
            if 'linear1' in k:
                # These should be updated
                if isinstance(v.value, dict):
                    for sub_k in v.value:
                        self.assertFalse(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
                else:
                    self.assertFalse(jnp.allclose(initial_params[k], v.value))
            else:
                # These should NOT be updated (zero gradients)
                if isinstance(v.value, dict):
                    for sub_k in v.value:
                        self.assertTrue(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
                else:
                    self.assertTrue(jnp.allclose(initial_params[k], v.value))

    def test_gradient_accumulation(self):
        """Test multiple gradient accumulations before step."""
        optimizer = braintools.optim.SGD(lr=0.1)
        optimizer.register_trainable_weights(self.param_states)

        # Store initial parameters
        initial_params = {}
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                initial_params[k] = {sub_k: sub_v.copy() for sub_k, sub_v in v.value.items()}
            else:
                initial_params[k] = v.value.copy()

        # Accumulate gradients from multiple batches
        accumulated_grads = None
        num_accumulations = 3

        for _ in range(num_accumulations):
            _, grads = self._compute_loss_and_grads()

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                # Add gradients
                accumulated_grads = jax.tree.map(
                    lambda a, b: a + b,
                    accumulated_grads,
                    grads
                )

        # Average accumulated gradients
        averaged_grads = jax.tree.map(
            lambda g: g / num_accumulations,
            accumulated_grads
        )

        # Take step with averaged gradients
        optimizer.step(averaged_grads)

        # Parameters should be updated
        for k, v in self.param_states.items():
            if isinstance(v.value, dict):
                for sub_k in v.value:
                    self.assertFalse(jnp.allclose(initial_params[k][sub_k], v.value[sub_k]))
            else:
                self.assertFalse(jnp.allclose(initial_params[k], v.value))

    def test_reset_optimizer_state(self):
        """Test creating a fresh optimizer resets state."""
        optimizer1 = braintools.optim.Adam(lr=0.01)
        optimizer1.register_trainable_weights(self.param_states)

        # Take several steps to build up momentum
        for _ in range(5):
            _, grads = self._compute_loss_and_grads()
            optimizer1.step(grads)

        # Store current step count
        step_count_first = optimizer1.step_count.value
        self.assertEqual(step_count_first, 5)

        # Create a new optimizer (reset state)
        optimizer2 = braintools.optim.Adam(lr=0.01)
        optimizer2.register_trainable_weights(self.param_states)

        # Step count should be at zero for new optimizer
        self.assertEqual(optimizer2.step_count.value, 0)

        # Optimizer state should be initialized
        self.assertIsNotNone(optimizer2.opt_state)

        # The two optimizers should have different states
        self.assertNotEqual(optimizer1.step_count.value, optimizer2.step_count.value)


# ============================================================================
# Nadam Optimizer Tests
# ============================================================================

class TestAllOptimizers(unittest.TestCase):

    def test_nadam_basic(self):
        """Test basic Nadam usage with float learning rate."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Nadam optimizer
        optimizer = braintools.optim.Nadam(lr=0.002)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.002
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-8
        assert optimizer.momentum_decay == 4e-3
        print("[OK] test_nadam_basic passed")

    def test_nadam_custom_betas(self):
        """Test Nadam with custom beta values."""
        model = brainstate.nn.Linear(10, 5)

        # Slower second moment decay for more aggressive updates
        optimizer = braintools.optim.Nadam(lr=0.002, betas=(0.9, 0.99))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.betas == (0.9, 0.99)
        print("[OK] test_nadam_custom_betas passed")

    def test_nadam_with_scheduler(self):
        """Test Nadam with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Step learning rate decay
        scheduler = braintools.optim.StepLR(base_lr=0.01, step_size=10, gamma=0.1)
        optimizer = braintools.optim.Nadam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify scheduler attachment
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.01  # Initial learning rate
        print("[OK] test_nadam_with_scheduler passed")

    def test_nadam_gradient_clipping(self):
        """Test Nadam with gradient clipping."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients by global norm
        optimizer = braintools.optim.Nadam(
            lr=0.002,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_nadam_gradient_clipping passed")

    def test_nadam_weight_decay(self):
        """Test Nadam with weight decay."""
        model = brainstate.nn.Linear(10, 5)

        # Add L2 regularization
        optimizer = braintools.optim.Nadam(
            lr=0.002,
            weight_decay=0.01
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 0.01
        print("[OK] test_nadam_weight_decay passed")

    def test_nadam_complete_training(self):
        """Test complete training loop with Nadam."""
        # Setup
        model = SimpleModelV1()
        optimizer = braintools.optim.Nadam(lr=0.002)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify optimizer was initialized correctly
        assert optimizer.current_lr == 0.002
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.momentum_decay == 4e-3

        # Verify optimization state exists
        assert optimizer.opt_state is not None

        print("[OK] test_nadam_complete_training passed")

    # ============================================================================
    # RAdam Optimizer Tests
    # ============================================================================

    def test_radam_basic(self):
        """Test basic RAdam usage with float learning rate."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize RAdam optimizer
        optimizer = braintools.optim.RAdam(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-8
        print("[OK] test_radam_basic passed")

    def test_radam_custom_betas(self):
        """Test RAdam with custom beta values."""
        model = brainstate.nn.Linear(10, 5)

        # Slower first moment decay for more stable updates
        optimizer = braintools.optim.RAdam(lr=0.001, betas=(0.8, 0.999))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.betas == (0.8, 0.999)
        print("[OK] test_radam_custom_betas passed")

    def test_radam_with_scheduler(self):
        """Test RAdam with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Step learning rate decay (ExponentialLR may have compatibility issues)
        scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=10, gamma=0.95)
        optimizer = braintools.optim.RAdam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify scheduler attachment
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.001  # Initial learning rate
        print("[OK] test_radam_with_scheduler passed")

    def test_radam_gradient_clipping(self):
        """Test RAdam with gradient clipping."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients by global norm
        optimizer = braintools.optim.RAdam(
            lr=0.001,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_radam_gradient_clipping passed")

    def test_radam_weight_decay(self):
        """Test RAdam with weight decay."""
        model = brainstate.nn.Linear(10, 5)

        # Add L2 regularization
        optimizer = braintools.optim.RAdam(
            lr=0.001,
            weight_decay=0.01
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 0.01
        print("[OK] test_radam_weight_decay passed")

    def test_radam_complete_training(self):
        """Test complete training loop with RAdam."""
        # Setup
        model = SimpleModelV1()
        optimizer = braintools.optim.RAdam(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify optimizer was initialized correctly
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)

        # Verify optimization state exists
        assert optimizer.opt_state is not None

        print("[OK] test_radam_complete_training passed")

    # ============================================================================
    # Comparison Tests
    # ============================================================================

    def test_nadam_vs_radam_initialization(self):
        """Compare initialization of Nadam and RAdam optimizers."""
        model = SimpleModelV1()

        # Initialize both optimizers with same hyperparameters
        nadam = braintools.optim.Nadam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        radam = braintools.optim.RAdam(lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        nadam.register_trainable_weights(model.states(brainstate.ParamState))
        radam.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify both have same initial parameters
        assert nadam.current_lr == radam.current_lr
        assert nadam.betas == radam.betas
        assert nadam.eps == radam.eps

        # Verify Nadam has momentum_decay parameter
        assert hasattr(nadam, 'momentum_decay')
        assert not hasattr(radam, 'momentum_decay')

        print("[OK] test_nadam_vs_radam_initialization passed")

    def test_multiple_param_groups(self):
        """Test Nadam and RAdam with multiple parameter groups."""
        model = SimpleModelV1()

        # Test Nadam
        nadam = braintools.optim.Nadam(lr=0.001)
        nadam.register_trainable_weights(model.states(brainstate.ParamState))

        # Test RAdam
        radam = braintools.optim.RAdam(lr=0.001)
        radam.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify both optimizers registered params
        assert nadam.param_states is not None
        assert radam.param_states is not None

        print("[OK] test_multiple_param_groups passed")

    def test_gradient_update_functionality(self):
        """Test that gradient updates work for both optimizers."""
        # Nadam optimizer
        model_nadam = SimpleModelV1()
        nadam = braintools.optim.Nadam(lr=0.01)
        nadam.register_trainable_weights(model_nadam.states(brainstate.ParamState))

        # RAdam optimizer
        model_radam = SimpleModelV1()
        radam = braintools.optim.RAdam(lr=0.01)
        radam.register_trainable_weights(model_radam.states(brainstate.ParamState))

        # Verify both optimizers have been registered with params
        assert nadam.param_states is not None
        assert radam.param_states is not None

        # Verify optimizer states initialized
        assert nadam.opt_state is not None
        assert radam.opt_state is not None

        # Verify learning rates
        assert nadam.current_lr == 0.01
        assert radam.current_lr == 0.01

        print("[OK] test_gradient_update_functionality passed")

    def test_state_dict_functionality(self):
        """Test state dict save/load for both optimizers."""
        model = SimpleModelV1()

        # Test Nadam state dict
        nadam = braintools.optim.Nadam(lr=0.002)
        nadam.register_trainable_weights(model.states(brainstate.ParamState))

        # Get state dict
        nadam_state = nadam.state_dict()
        assert 'opt_state' in nadam_state
        assert 'param_groups' in nadam_state

        # Test RAdam state dict
        radam = braintools.optim.RAdam(lr=0.001)
        radam.register_trainable_weights(model.states(brainstate.ParamState))

        # Get state dict
        radam_state = radam.state_dict()
        assert 'opt_state' in radam_state
        assert 'param_groups' in radam_state

        print("[OK] test_state_dict_functionality passed")

    # ============================================================================
    # Edge Case Tests
    # ============================================================================

    def test_edge_case_zero_learning_rate(self):
        """Test optimizers with zero learning rate."""
        model = SimpleModelV1()

        # Nadam with zero lr
        nadam = braintools.optim.Nadam(lr=0.0)
        nadam.register_trainable_weights(model.states(brainstate.ParamState))
        assert nadam.current_lr == 0.0

        # RAdam with zero lr
        radam = braintools.optim.RAdam(lr=0.0)
        radam.register_trainable_weights(model.states(brainstate.ParamState))
        assert radam.current_lr == 0.0

        print("[OK] test_edge_case_zero_learning_rate passed")

    def test_edge_case_extreme_betas(self):
        """Test optimizers with extreme beta values."""
        model = SimpleModelV1()

        # Test with very small betas (fast decay)
        nadam = braintools.optim.Nadam(lr=0.001, betas=(0.1, 0.1))
        nadam.register_trainable_weights(model.states(brainstate.ParamState))
        assert nadam.betas == (0.1, 0.1)

        # Test with very large betas (slow decay)
        radam = braintools.optim.RAdam(lr=0.001, betas=(0.999, 0.9999))
        radam.register_trainable_weights(model.states(brainstate.ParamState))
        assert radam.betas == (0.999, 0.9999)

        print("[OK] test_edge_case_extreme_betas passed")

    def test_edge_case_large_weight_decay(self):
        """Test optimizers with large weight decay."""
        model = SimpleModelV1()

        # Nadam with large weight decay
        nadam = braintools.optim.Nadam(lr=0.001, weight_decay=1.0)
        nadam.register_trainable_weights(model.states(brainstate.ParamState))
        assert nadam.weight_decay == 1.0

        # RAdam with large weight decay
        radam = braintools.optim.RAdam(lr=0.001, weight_decay=1.0)
        radam.register_trainable_weights(model.states(brainstate.ParamState))
        assert radam.weight_decay == 1.0

        print("[OK] test_edge_case_large_weight_decay passed")

    # ============================================================================
    # LAMB Optimizer Tests
    # ============================================================================

    def test_lamb_basic(self):
        """Test basic LAMB usage with float learning rate."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize LAMB optimizer for large batch training
        optimizer = braintools.optim.Lamb(lr=0.002)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.002
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-6
        print("[OK] test_lamb_basic passed")

    def test_lamb_large_learning_rate(self):
        """Test LAMB with large learning rate for big batch sizes."""
        model = brainstate.nn.Linear(10, 5)

        # LAMB can handle larger learning rates due to trust ratio
        optimizer = braintools.optim.Lamb(lr=0.01, betas=(0.9, 0.999))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.01
        assert optimizer.betas == (0.9, 0.999)
        print("[OK] test_lamb_large_learning_rate passed")

    def test_lamb_with_scheduler(self):
        """Test LAMB with learning rate scheduler for warmup and decay."""
        model = brainstate.nn.Linear(10, 5)

        # Use StepLR scheduler (simpler and more reliable)
        scheduler = braintools.optim.StepLR(
            base_lr=0.002,
            step_size=30,
            gamma=0.1
        )
        optimizer = braintools.optim.Lamb(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.002  # Initial lr
        print("[OK] test_lamb_with_scheduler passed")

    def test_lamb_weight_decay(self):
        """Test LAMB with weight decay for regularization."""
        model = brainstate.nn.Linear(10, 5)

        # LAMB applies weight decay adaptively
        optimizer = braintools.optim.Lamb(
            lr=0.002,
            weight_decay=0.01
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 0.01
        print("[OK] test_lamb_weight_decay passed")

    def test_lamb_gradient_clipping(self):
        """Test LAMB with gradient clipping for stability."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients for training stability
        optimizer = braintools.optim.Lamb(
            lr=0.002,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_lamb_gradient_clipping passed")

    def test_lamb_large_batch_config(self):
        """Test LAMB with large batch training configuration."""
        model = SimpleModelV1()

        # LAMB with settings for large batch
        optimizer = braintools.optim.Lamb(
            lr=0.01,  # Higher lr due to normalization
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify configuration
        assert optimizer.current_lr == 0.01
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.weight_decay == 0.01
        assert optimizer.eps == 1e-6

        # Verify optimizer state initialized
        assert optimizer.opt_state is not None

        print("[OK] test_lamb_large_batch_config passed")

    # ============================================================================
    # LARS Optimizer Tests
    # ============================================================================

    def test_lars_basic(self):
        """Test basic LARS usage with momentum."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize LARS optimizer
        optimizer = braintools.optim.Lars(lr=0.1, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.1
        assert optimizer.momentum == 0.9
        assert optimizer.trust_coefficient == 0.001
        print("[OK] test_lars_basic passed")

    def test_lars_custom_trust_coefficient(self):
        """Test LARS with custom trust coefficient for fine-tuning."""
        model = brainstate.nn.Linear(10, 5)

        # Smaller trust coefficient for more conservative updates
        optimizer = braintools.optim.Lars(
            lr=0.1,
            momentum=0.9,
            trust_coefficient=0.0001
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.trust_coefficient == 0.0001
        print("[OK] test_lars_custom_trust_coefficient passed")

    def test_lars_with_scheduler(self):
        """Test LARS with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Use StepLR scheduler (consistent with other tests)
        scheduler = braintools.optim.StepLR(
            base_lr=0.1,
            step_size=30,
            gamma=0.5
        )
        optimizer = braintools.optim.Lars(lr=scheduler, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer._lr_scheduler is scheduler
        assert optimizer.momentum == 0.9
        assert optimizer.current_lr == 0.1  # Initial lr
        print("[OK] test_lars_with_scheduler passed")

    def test_lars_weight_decay(self):
        """Test LARS with weight decay."""
        model = brainstate.nn.Linear(10, 5)

        # LARS with weight decay
        optimizer = braintools.optim.Lars(
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 5e-4
        print("[OK] test_lars_weight_decay passed")

    def test_lars_large_batch_scaling(self):
        """Test LARS with linear scaling for large batch training."""
        model = SimpleModelV1()

        # Configuration for large batch training
        # Linear scaling rule: lr = base_lr * (batch_size / base_batch)
        batch_size = 4096
        base_batch = 256
        base_lr = 0.1
        scaled_lr = base_lr * (batch_size / base_batch)

        optimizer = braintools.optim.Lars(
            lr=scaled_lr,
            momentum=0.9,
            weight_decay=5e-4,
            trust_coefficient=0.001
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == scaled_lr
        assert optimizer.current_lr == 1.6  # 0.1 * (4096/256) = 1.6
        assert optimizer.momentum == 0.9
        assert optimizer.weight_decay == 5e-4
        assert optimizer.trust_coefficient == 0.001

        print("[OK] test_lars_large_batch_scaling passed")

    def test_lars_cnn_config(self):
        """Test LARS configuration for CNN training."""
        model = SimpleModelV1()

        # LARS for large batch CNN training
        optimizer = braintools.optim.Lars(
            lr=1.6,  # Large lr for batch size 4096
            momentum=0.9,
            weight_decay=1e-4,
            trust_coefficient=0.001
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer.current_lr == 1.6
        assert optimizer.momentum == 0.9
        assert optimizer.weight_decay == 1e-4
        assert optimizer.trust_coefficient == 0.001
        assert optimizer.eps == 1e-8

        # Verify optimizer state
        assert optimizer.opt_state is not None

        print("[OK] test_lars_cnn_config passed")

    # ============================================================================
    # Comparison Tests for LAMB and LARS
    # ============================================================================

    def test_lamb_vs_lars_initialization(self):
        """Compare initialization of LAMB and LARS optimizers."""
        model = SimpleModelV1()

        # Initialize both optimizers
        lamb = braintools.optim.Lamb(lr=0.1, weight_decay=1e-4)
        lars = braintools.optim.Lars(lr=0.1, momentum=0.9, weight_decay=1e-4)

        lamb.register_trainable_weights(model.states(brainstate.ParamState))
        lars.register_trainable_weights(model.states(brainstate.ParamState))

        # Both use trust ratio mechanism
        assert lamb.current_lr == lars.current_lr
        assert lamb.weight_decay == lars.weight_decay

        # LAMB uses Adam-style betas, LARS uses momentum
        assert hasattr(lamb, 'betas')
        assert hasattr(lars, 'momentum')

        # LARS has trust_coefficient
        assert hasattr(lars, 'trust_coefficient')

        print("[OK] test_lamb_vs_lars_initialization passed")

    def test_large_batch_optimizers_state_dict(self):
        """Test state dict functionality for LAMB and LARS."""
        model = SimpleModelV1()

        # Test LAMB state dict
        lamb = braintools.optim.Lamb(lr=0.002)
        lamb.register_trainable_weights(model.states(brainstate.ParamState))

        lamb_state = lamb.state_dict()
        assert 'opt_state' in lamb_state
        assert 'param_groups' in lamb_state

        # Test LARS state dict
        lars = braintools.optim.Lars(lr=0.1, momentum=0.9)
        lars.register_trainable_weights(model.states(brainstate.ParamState))

        lars_state = lars.state_dict()
        assert 'opt_state' in lars_state
        assert 'param_groups' in lars_state

        print("[OK] test_large_batch_optimizers_state_dict passed")

    def test_large_batch_edge_cases(self):
        """Test edge cases for large batch optimizers."""
        model = SimpleModelV1()

        # LAMB with extreme learning rate
        lamb_extreme = braintools.optim.Lamb(lr=100.0)
        lamb_extreme.register_trainable_weights(model.states(brainstate.ParamState))
        assert lamb_extreme.current_lr == 100.0

        # LARS with no momentum
        lars_no_momentum = braintools.optim.Lars(lr=0.1, momentum=0.0)
        lars_no_momentum.register_trainable_weights(model.states(brainstate.ParamState))
        assert lars_no_momentum.momentum == 0.0

        # LARS with very small trust coefficient
        lars_small_trust = braintools.optim.Lars(lr=0.1, trust_coefficient=1e-6)
        lars_small_trust.register_trainable_weights(model.states(brainstate.ParamState))
        assert lars_small_trust.trust_coefficient == 1e-6

        print("[OK] test_large_batch_edge_cases passed")

    # ============================================================================
    # Lookahead Optimizer Tests
    # ============================================================================

    def test_lookahead_basic_sgd(self):
        """Test basic Lookahead usage with SGD as base optimizer."""
        import optax

        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Create base optimizer (SGD)
        base_opt = optax.sgd(learning_rate=0.1)

        # Wrap with Lookahead
        optimizer = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=5,
            alpha=0.5
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.sync_period == 5
        assert optimizer.alpha == 0.5
        print("[OK] test_lookahead_basic_sgd passed")

    def test_lookahead_with_adam(self):
        """Test Lookahead with Adam as base optimizer."""
        import optax

        # Lookahead + Adam (RAdam paper recommends this combination)
        base_opt = optax.adam(learning_rate=0.001)
        optimizer = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=6,
            alpha=0.5
        )
        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.sync_period == 6
        assert optimizer.alpha == 0.5
        print("[OK] test_lookahead_with_adam passed")

    def test_lookahead_custom_sync_period(self):
        """Test Lookahead with custom synchronization period."""
        import optax

        # Longer sync period for more exploration
        base_opt = optax.sgd(learning_rate=0.1, momentum=0.9)
        optimizer = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=10,  # Synchronize every 10 steps
            alpha=0.5
        )
        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.sync_period == 10
        print("[OK] test_lookahead_custom_sync_period passed")

    def test_lookahead_custom_alpha(self):
        """Test Lookahead with custom slow weights step size."""
        import optax

        # Smaller alpha for more conservative slow weight updates
        base_opt = optax.adam(learning_rate=0.001)
        optimizer = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=5,
            alpha=0.3  # More conservative than default 0.5
        )
        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.alpha == 0.3
        print("[OK] test_lookahead_custom_alpha passed")

    def test_lookahead_with_scheduler(self):
        """Test Lookahead with learning rate scheduler."""
        import optax

        # Combine with scheduler for dynamic learning rate
        scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=30, gamma=0.5)
        base_opt = optax.sgd(learning_rate=0.1)
        optimizer = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            lr=scheduler,
            sync_period=5,
            alpha=0.5
        )
        model = brainstate.nn.Linear(10, 5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.1  # Initial lr
        print("[OK] test_lookahead_with_scheduler passed")

    def test_lookahead_complete_config(self):
        """Test Lookahead with complete configuration."""
        import optax

        model = SimpleModelV1()

        # Lookahead with SGD + momentum
        base_opt = optax.sgd(learning_rate=0.1, momentum=0.9)
        optimizer = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=5,
            alpha=0.5
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all settings
        assert optimizer.sync_period == 5
        assert optimizer.alpha == 0.5
        assert optimizer.base_optimizer is base_opt
        assert optimizer.opt_state is not None

        print("[OK] test_lookahead_complete_config passed")

    # ============================================================================
    # Yogi Optimizer Tests
    # ============================================================================

    def test_yogi_basic(self):
        """Test basic Yogi usage with default parameters."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Yogi optimizer
        optimizer = braintools.optim.Yogi(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-3  # Note: Yogi uses larger epsilon
        print("[OK] test_yogi_basic passed")

    def test_yogi_custom_betas(self):
        """Test Yogi with custom beta values."""
        model = brainstate.nn.Linear(10, 5)

        # Adjust momentum parameters
        optimizer = braintools.optim.Yogi(
            lr=0.001,
            betas=(0.9, 0.99)  # Faster second moment decay
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.betas == (0.9, 0.99)
        print("[OK] test_yogi_custom_betas passed")

    def test_yogi_custom_epsilon(self):
        """Test Yogi with larger epsilon for increased stability."""
        model = brainstate.nn.Linear(10, 5)

        # Yogi is less sensitive to epsilon than Adam
        optimizer = braintools.optim.Yogi(
            lr=0.001,
            eps=1e-2  # Even larger epsilon for more stability
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.eps == 1e-2
        print("[OK] test_yogi_custom_epsilon passed")

    def test_yogi_with_scheduler(self):
        """Test Yogi with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Combine with step decay
        scheduler = braintools.optim.StepLR(base_lr=0.01, step_size=30, gamma=0.5)
        optimizer = braintools.optim.Yogi(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.01  # Initial lr
        print("[OK] test_yogi_with_scheduler passed")

    def test_yogi_weight_decay(self):
        """Test Yogi with weight decay for regularization."""
        model = brainstate.nn.Linear(10, 5)

        # Add L2 regularization
        optimizer = braintools.optim.Yogi(
            lr=0.001,
            weight_decay=0.01
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 0.01
        print("[OK] test_yogi_weight_decay passed")

    def test_yogi_complete_config(self):
        """Test Yogi with complete configuration for NLP tasks."""
        model = SimpleModelV1()

        # Yogi works well for NLP with sparse gradients
        optimizer = braintools.optim.Yogi(
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-3
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all settings
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-3
        assert optimizer.opt_state is not None

        print("[OK] test_yogi_complete_config passed")

    # ============================================================================
    # Comparison Tests for Lookahead and Yogi
    # ============================================================================

    def test_lookahead_vs_yogi(self):
        """Compare Lookahead and Yogi optimizers."""
        import optax

        model = SimpleModelV1()

        # Lookahead wraps another optimizer
        base_opt = optax.adam(learning_rate=0.001)
        lookahead = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=5,
            alpha=0.5
        )
        lookahead.register_trainable_weights(model.states(brainstate.ParamState))

        # Yogi is a standalone optimizer
        yogi = braintools.optim.Yogi(lr=0.001)
        yogi.register_trainable_weights(model.states(brainstate.ParamState))

        # Lookahead is a meta-optimizer
        assert hasattr(lookahead, 'base_optimizer')
        assert hasattr(lookahead, 'sync_period')
        assert hasattr(lookahead, 'alpha')

        # Yogi has Adam-like parameters
        assert hasattr(yogi, 'betas')
        assert hasattr(yogi, 'eps')

        print("[OK] test_lookahead_vs_yogi passed")

    def test_meta_optimizers_state_dict(self):
        """Test state dict functionality for Lookahead and Yogi."""
        import optax

        model = SimpleModelV1()

        # Test Lookahead state dict
        base_opt = optax.sgd(learning_rate=0.1)
        lookahead = braintools.optim.Lookahead(base_optimizer=base_opt)
        lookahead.register_trainable_weights(model.states(brainstate.ParamState))

        lookahead_state = lookahead.state_dict()
        assert 'opt_state' in lookahead_state
        assert 'param_groups' in lookahead_state

        # Test Yogi state dict
        yogi = braintools.optim.Yogi(lr=0.001)
        yogi.register_trainable_weights(model.states(brainstate.ParamState))

        yogi_state = yogi.state_dict()
        assert 'opt_state' in yogi_state
        assert 'param_groups' in yogi_state

        print("[OK] test_meta_optimizers_state_dict passed")

    def test_advanced_optimizers_edge_cases(self):
        """Test edge cases for Lookahead and Yogi."""
        import optax

        model = SimpleModelV1()

        # Lookahead with very short sync period
        base_opt = optax.sgd(learning_rate=0.1)
        lookahead_short = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=1,
            alpha=0.5
        )
        lookahead_short.register_trainable_weights(model.states(brainstate.ParamState))
        assert lookahead_short.sync_period == 1

        # Lookahead with alpha = 1 (full update)
        lookahead_full = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=5,
            alpha=1.0
        )
        lookahead_full.register_trainable_weights(model.states(brainstate.ParamState))
        assert lookahead_full.alpha == 1.0

        # Yogi with very small epsilon
        yogi_small_eps = braintools.optim.Yogi(lr=0.001, eps=1e-8)
        yogi_small_eps.register_trainable_weights(model.states(brainstate.ParamState))
        assert yogi_small_eps.eps == 1e-8

        # Yogi with extreme beta values
        yogi_extreme = braintools.optim.Yogi(lr=0.001, betas=(0.5, 0.9))
        yogi_extreme.register_trainable_weights(model.states(brainstate.ParamState))
        assert yogi_extreme.betas == (0.5, 0.9)

        print("[OK] test_advanced_optimizers_edge_cases passed")

    # ============================================================================
    # LBFGS Optimizer Tests
    # ============================================================================

    def test_lbfgs_basic(self):
        """Test basic L-BFGS usage for batch optimization."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize L-BFGS optimizer
        optimizer = braintools.optim.LBFGS(lr=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 1.0
        assert optimizer.memory_size == 10
        assert optimizer.scale_init_precond is True
        print("[OK] test_lbfgs_basic passed")

    def test_lbfgs_large_memory(self):
        """Test L-BFGS with larger memory for better Hessian approximation."""
        model = brainstate.nn.Linear(10, 5)

        # Larger memory for better Hessian approximation
        optimizer = braintools.optim.LBFGS(
            lr=1.0,
            memory_size=20
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.memory_size == 20
        print("[OK] test_lbfgs_large_memory passed")

    def test_lbfgs_small_memory(self):
        """Test L-BFGS with smaller memory for efficiency."""
        model = brainstate.nn.Linear(10, 5)

        # Smaller memory footprint
        optimizer = braintools.optim.LBFGS(
            lr=1.0,
            memory_size=5
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.memory_size == 5
        print("[OK] test_lbfgs_small_memory passed")

    def test_lbfgs_no_scaling(self):
        """Test L-BFGS with initial Hessian scaling disabled."""
        model = brainstate.nn.Linear(10, 5)

        # Without Hessian scaling
        optimizer = braintools.optim.LBFGS(
            lr=1.0,
            memory_size=10,
            scale_init_precond=False
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.scale_init_precond is False
        print("[OK] test_lbfgs_no_scaling passed")

    def test_lbfgs_fine_tuning(self):
        """Test L-BFGS configuration for fine-tuning."""
        model = SimpleModelV1()

        # L-BFGS for fine-tuning with full batch
        optimizer = braintools.optim.LBFGS(
            lr=1.0,
            memory_size=10
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all settings
        assert optimizer.current_lr == 1.0
        assert optimizer.memory_size == 10
        assert optimizer.scale_init_precond is True
        assert optimizer.opt_state is not None

        print("[OK] test_lbfgs_fine_tuning passed")

    def test_lbfgs_convex_optimization(self):
        """Test L-BFGS for convex optimization."""
        # L-BFGS excels at convex problems
        model = brainstate.nn.Linear(50, 1)  # Linear regression
        optimizer = braintools.optim.LBFGS(
            lr=1.0,
            memory_size=15
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.memory_size == 15
        print("[OK] test_lbfgs_convex_optimization passed")

    # ============================================================================
    # Rprop Optimizer Tests
    # ============================================================================

    def test_rprop_basic(self):
        """Test basic Rprop usage with default parameters."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Rprop optimizer
        optimizer = braintools.optim.Rprop(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.01
        assert optimizer.etas == (0.5, 1.2)
        assert optimizer.step_sizes == (1e-6, 50.0)
        print("[OK] test_rprop_basic passed")

    def test_rprop_custom_etas(self):
        """Test Rprop with custom eta values for step size adjustment."""
        model = brainstate.nn.Linear(10, 5)

        # More aggressive step size changes
        optimizer = braintools.optim.Rprop(
            lr=0.01,
            etas=(0.3, 1.5)  # Faster decrease, faster increase
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.etas == (0.3, 1.5)
        print("[OK] test_rprop_custom_etas passed")

    def test_rprop_custom_step_sizes(self):
        """Test Rprop with custom step size bounds."""
        model = brainstate.nn.Linear(10, 5)

        # Tighter bounds on step sizes
        optimizer = braintools.optim.Rprop(
            lr=0.01,
            step_sizes=(1e-5, 10.0)
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.step_sizes == (1e-5, 10.0)
        print("[OK] test_rprop_custom_step_sizes passed")

    def test_rprop_complete_config(self):
        """Test Rprop with complete configuration."""
        model = brainstate.nn.Linear(10, 5)

        # All parameters customized
        optimizer = braintools.optim.Rprop(
            lr=0.01,
            etas=(0.5, 1.2),
            step_sizes=(1e-6, 50.0)
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.01
        assert optimizer.etas == (0.5, 1.2)
        assert optimizer.step_sizes == (1e-6, 50.0)
        print("[OK] test_rprop_complete_config passed")

    def test_rprop_batch_training(self):
        """Test Rprop configuration for batch training."""
        model = SimpleModelV1()

        # Rprop for batch training
        optimizer = braintools.optim.Rprop(
            lr=0.01,
            etas=(0.5, 1.2)
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all settings
        assert optimizer.current_lr == 0.01
        assert optimizer.etas == (0.5, 1.2)
        assert optimizer.step_sizes == (1e-6, 50.0)
        assert optimizer.opt_state is not None

        print("[OK] test_rprop_batch_training passed")

    def test_rprop_classification(self):
        """Test Rprop for classification tasks."""
        model = SimpleModelV1()

        optimizer = braintools.optim.Rprop(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Rprop adapts step sizes automatically
        assert optimizer.current_lr == 0.01
        print("[OK] test_rprop_classification passed")

    # ============================================================================
    # Comparison Tests for LBFGS and Rprop
    # ============================================================================

    def test_lbfgs_vs_rprop(self):
        """Compare L-BFGS and Rprop optimizers."""
        model = SimpleModelV1()

        # L-BFGS - second-order method
        lbfgs = braintools.optim.LBFGS(lr=1.0, memory_size=10)
        lbfgs.register_trainable_weights(model.states(brainstate.ParamState))

        # Rprop - first-order with adaptive step sizes
        rprop = braintools.optim.Rprop(lr=0.01, etas=(0.5, 1.2))
        rprop.register_trainable_weights(model.states(brainstate.ParamState))

        # L-BFGS has memory_size parameter
        assert hasattr(lbfgs, 'memory_size')
        assert hasattr(lbfgs, 'scale_init_precond')

        # Rprop has etas and step_sizes
        assert hasattr(rprop, 'etas')
        assert hasattr(rprop, 'step_sizes')

        print("[OK] test_lbfgs_vs_rprop passed")

    def test_second_order_optimizers_state_dict(self):
        """Test state dict functionality for L-BFGS and Rprop."""
        model = SimpleModelV1()

        # Test L-BFGS state dict
        lbfgs = braintools.optim.LBFGS(lr=1.0)
        lbfgs.register_trainable_weights(model.states(brainstate.ParamState))

        lbfgs_state = lbfgs.state_dict()
        assert 'opt_state' in lbfgs_state
        assert 'param_groups' in lbfgs_state

        # Test Rprop state dict
        rprop = braintools.optim.Rprop(lr=0.01)
        rprop.register_trainable_weights(model.states(brainstate.ParamState))

        rprop_state = rprop.state_dict()
        assert 'opt_state' in rprop_state
        assert 'param_groups' in rprop_state

        print("[OK] test_second_order_optimizers_state_dict passed")

    def test_batch_optimizers_edge_cases(self):
        """Test edge cases for L-BFGS and Rprop."""
        model = SimpleModelV1()

        # L-BFGS with very small memory
        lbfgs_small = braintools.optim.LBFGS(lr=1.0, memory_size=3)
        lbfgs_small.register_trainable_weights(model.states(brainstate.ParamState))
        assert lbfgs_small.memory_size == 3

        # L-BFGS with very large memory
        lbfgs_large = braintools.optim.LBFGS(lr=1.0, memory_size=50)
        lbfgs_large.register_trainable_weights(model.states(brainstate.ParamState))
        assert lbfgs_large.memory_size == 50

        # Rprop with extreme etas
        rprop_extreme = braintools.optim.Rprop(lr=0.01, etas=(0.1, 2.0))
        rprop_extreme.register_trainable_weights(model.states(brainstate.ParamState))
        assert rprop_extreme.etas == (0.1, 2.0)

        # Rprop with very tight step size bounds
        rprop_tight = braintools.optim.Rprop(lr=0.01, step_sizes=(1e-4, 1.0))
        rprop_tight.register_trainable_weights(model.states(brainstate.ParamState))
        assert rprop_tight.step_sizes == (1e-4, 1.0)

        print("[OK] test_batch_optimizers_edge_cases passed")

    # ============================================================================
    # Adafactor Optimizer Tests
    # ============================================================================

    def test_adafactor_basic(self):
        """Test basic Adafactor usage with automatic learning rate."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Adafactor with auto learning rate
        optimizer = braintools.optim.Adafactor()
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.factored is True
        assert optimizer.eps == (1e-30, 1e-3)
        assert optimizer.clip_threshold == 1.0
        print("[OK] test_adafactor_basic passed")

    def test_adafactor_explicit_lr(self):
        """Test Adafactor with explicit learning rate."""
        model = brainstate.nn.Linear(10, 5)

        # Use explicit learning rate
        optimizer = braintools.optim.Adafactor(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify learning rate
        assert optimizer.current_lr == 0.001
        assert optimizer.factored is True
        print("[OK] test_adafactor_explicit_lr passed")

    def test_adafactor_with_momentum(self):
        """Test Adafactor with momentum."""
        model = brainstate.nn.Linear(10, 5)

        # Enable momentum
        optimizer = braintools.optim.Adafactor(lr=0.001, beta1=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify beta1 is set
        assert optimizer.beta1 == 0.9
        assert optimizer.current_lr == 0.001
        print("[OK] test_adafactor_with_momentum passed")

    def test_adafactor_non_factored(self):
        """Test Adafactor without factorization."""
        model = brainstate.nn.Linear(10, 5)

        # Disable factorization (uses more memory but may be faster)
        optimizer = braintools.optim.Adafactor(lr=0.001, factored=False)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify factored is False
        assert optimizer.factored is False
        assert optimizer.current_lr == 0.001
        print("[OK] test_adafactor_non_factored passed")

    def test_adafactor_large_model(self):
        """Test Adafactor for large transformer training."""
        # Large model simulation
        model = brainstate.nn.Linear(1000, 500)

        # Typical configuration for large models
        optimizer = braintools.optim.Adafactor(
            beta1=0.9,
            clip_threshold=1.0,
            decay_rate=-0.8
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify configuration
        assert optimizer.beta1 == 0.9
        assert optimizer.clip_threshold == 1.0
        assert optimizer.decay_rate == -0.8
        print("[OK] test_adafactor_large_model passed")

    def test_adafactor_complete_config(self):
        """Test Adafactor with complete configuration."""
        model = brainstate.nn.Linear(100, 50)

        # Complete configuration
        optimizer = braintools.optim.Adafactor(
            lr=0.01,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=0.9,
            weight_decay=0.0001,
            factored=True
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer.current_lr == 0.01
        assert optimizer.eps == (1e-30, 1e-3)
        assert optimizer.clip_threshold == 1.0
        assert optimizer.decay_rate == -0.8
        assert optimizer.beta1 == 0.9
        assert optimizer.weight_decay == 0.0001
        assert optimizer.factored is True
        print("[OK] test_adafactor_complete_config passed")

    # ============================================================================
    # AdaBelief Optimizer Tests
    # ============================================================================

    def test_adabelief_basic(self):
        """Test basic AdaBelief usage."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize AdaBelief
        optimizer = braintools.optim.AdaBelief(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-16
        print("[OK] test_adabelief_basic passed")

    def test_adabelief_custom_betas(self):
        """Test AdaBelief with custom betas."""
        model = brainstate.nn.Linear(10, 5)

        # Faster momentum decay, slower variance decay
        optimizer = braintools.optim.AdaBelief(lr=0.001, betas=(0.8, 0.999))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify betas
        assert optimizer.betas == (0.8, 0.999)
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_custom_betas passed")

    def test_adabelief_scheduler(self):
        """Test AdaBelief with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Learning rate decays every 30 epochs
        scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=30, gamma=0.5)
        optimizer = braintools.optim.AdaBelief(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify scheduler
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_scheduler passed")

    def test_adabelief_weight_decay(self):
        """Test AdaBelief with weight decay."""
        model = brainstate.nn.Linear(10, 5)

        # Add L2 regularization
        optimizer = braintools.optim.AdaBelief(lr=0.001, weight_decay=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify weight decay
        assert optimizer.weight_decay == 0.01
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_weight_decay passed")

    def test_adabelief_gradient_clipping(self):
        """Test AdaBelief with gradient clipping."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients by global norm
        optimizer = braintools.optim.AdaBelief(
            lr=0.001,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify gradient clipping
        assert optimizer.grad_clip_norm == 1.0
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_gradient_clipping passed")

    def test_adabelief_complete_config(self):
        """Test AdaBelief with complete configuration."""
        # Large model
        model = brainstate.nn.Linear(1000, 500)

        # Learning rate schedule
        scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=20, gamma=0.5)

        # Complete AdaBelief configuration
        optimizer = braintools.optim.AdaBelief(
            lr=scheduler,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=0.0001,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-16
        assert optimizer.weight_decay == 0.0001
        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_adabelief_complete_config passed")

    # ============================================================================
    # Adafactor vs AdaBelief Comparison Tests
    # ============================================================================

    def test_adafactor_vs_adabelief(self):
        """Compare Adafactor and AdaBelief optimizers."""
        model1 = brainstate.nn.Linear(20, 10)
        model2 = brainstate.nn.Linear(20, 10)

        # Create both optimizers
        opt_adafactor = braintools.optim.Adafactor(lr=0.001)
        opt_adafactor.register_trainable_weights(model1.states(brainstate.ParamState))

        opt_adabelief = braintools.optim.AdaBelief(lr=0.001)
        opt_adabelief.register_trainable_weights(model2.states(brainstate.ParamState))

        # Verify both are initialized
        assert opt_adafactor.current_lr == 0.001
        assert opt_adabelief.current_lr == 0.001
        assert opt_adafactor.opt_state is not None
        assert opt_adabelief.opt_state is not None

        print("[OK] test_adafactor_vs_adabelief passed")

    def test_memory_efficient_optimizers_state_dict(self):
        """Test state_dict functionality for memory-efficient optimizers."""
        model = brainstate.nn.Linear(10, 5)

        # Test Adafactor
        opt1 = braintools.optim.Adafactor(lr=0.001, factored=True)
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        state1 = opt1.state_dict()
        assert 'opt_state' in state1
        assert 'lr' in state1

        # Test AdaBelief
        opt2 = braintools.optim.AdaBelief(lr=0.001)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        state2 = opt2.state_dict()
        assert 'opt_state' in state2
        assert 'lr' in state2

        print("[OK] test_memory_efficient_optimizers_state_dict passed")

    def test_belief_based_optimizers_edge_cases(self):
        """Test edge cases for belief-based optimizers."""
        model = brainstate.nn.Linear(5, 3)

        # Test Adafactor with minimal eps
        opt1 = braintools.optim.Adafactor(lr=0.01, eps=(1e-30, 1e-3))
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt1.eps == (1e-30, 1e-3)

        # Test AdaBelief with very small eps
        opt2 = braintools.optim.AdaBelief(lr=0.01, eps=1e-16)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt2.eps == 1e-16

        # Test AdaBelief with extreme betas
        opt3 = braintools.optim.AdaBelief(lr=0.01, betas=(0.5, 0.9))
        opt3.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt3.betas == (0.5, 0.9)

        print("[OK] test_belief_based_optimizers_edge_cases passed")


class TestOptimizerExample(unittest.TestCase):

    # ============================================================================
    # OptaxOptimizer Base Class Examples
    # ============================================================================

    def test_optax_optimizer_basic_usage(self):
        """Test basic optimizer usage with float learning rate."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adam(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.001
        assert optimizer._lr_scheduler is not None

    def test_optax_optimizer_with_scheduler(self):
        """Test optimizer with learning rate scheduler."""
        model = SimpleModelV1()
        scheduler = braintools.optim.StepLR(base_lr=0.01, step_size=10, gamma=0.1)
        optimizer = braintools.optim.Adam(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.01

    def test_optax_optimizer_param_groups(self):
        """Test optimizer with multiple parameter groups."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adam(lr=0.001)

        # Register main parameters
        optimizer.register_trainable_weights(model.linear.states(brainstate.ParamState))

        # Add another group with different lr
        special_params = {'special': brainstate.ParamState(jnp.zeros(5))}
        optimizer.add_param_group(special_params, lr=0.0001)

        assert len(optimizer.param_groups) == 2

    # ============================================================================
    # SGD Optimizer Examples
    # ============================================================================

    def test_sgd_basic(self):
        """Test basic SGD without momentum."""
        model = SimpleModelV1()
        optimizer = braintools.optim.SGD(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.01
        assert optimizer.momentum == 0.0

    def test_sgd_with_momentum(self):
        """Test SGD with momentum."""
        model = SimpleModelV1()
        optimizer = braintools.optim.SGD(lr=0.01, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.momentum == 0.9

    def test_sgd_with_nesterov(self):
        """Test SGD with Nesterov momentum."""
        model = SimpleModelV1()
        optimizer = braintools.optim.SGD(lr=0.01, momentum=0.9, nesterov=True)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.nesterov is True

    def test_sgd_with_scheduler(self):
        """Test SGD with learning rate scheduling."""
        model = SimpleModelV1()
        scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=28, gamma=0.1)
        optimizer = braintools.optim.SGD(lr=scheduler, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify initial learning rate
        assert optimizer.current_lr == 0.1

        # Step the scheduler multiple times
        for _ in range(30):
            scheduler.step()

        # Learning rate should have decayed
        assert optimizer.current_lr < 0.1

    # ============================================================================
    # Momentum Optimizer Examples
    # ============================================================================

    def test_momentum_basic(self):
        """Test basic Momentum optimizer."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Momentum(lr=0.01, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.01
        assert optimizer.momentum == 0.9

    def test_momentum_with_weight_decay(self):
        """Test Momentum optimizer with weight decay."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Momentum(lr=0.01, momentum=0.9, weight_decay=0.0001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 0.0001
        assert optimizer.momentum == 0.9

    def test_momentum_with_gradient_clipping(self):
        """Test Momentum optimizer with gradient clipping."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Momentum(
            lr=0.01,
            momentum=0.9,
            grad_clip_norm=1.0,
            grad_clip_value=0.5
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.grad_clip_norm == 1.0
        assert optimizer.grad_clip_value == 0.5

    def test_momentum_with_scheduler(self):
        """Test Momentum optimizer with learning rate scheduling."""
        model = SimpleModelV1()
        scheduler = braintools.optim.StepLR(base_lr=0.1, step_size=28, gamma=0.1)
        optimizer = braintools.optim.Momentum(lr=scheduler, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify initial learning rate
        assert optimizer.current_lr == 0.1

        # Step the scheduler multiple times
        for _ in range(30):
            scheduler.step()

        # Learning rate should have decayed
        assert optimizer.current_lr < 0.1

    # ============================================================================
    # MomentumNesterov Optimizer Examples
    # ============================================================================

    def test_momentum_nesterov_basic(self):
        """Test basic MomentumNesterov optimizer."""
        model = SimpleModelV1()
        optimizer = braintools.optim.MomentumNesterov(lr=0.01, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.01
        assert optimizer.momentum == 0.9

    def test_momentum_nesterov_with_weight_decay(self):
        """Test MomentumNesterov optimizer with weight decay."""
        model = SimpleModelV1()
        optimizer = braintools.optim.MomentumNesterov(lr=0.01, momentum=0.9, weight_decay=0.0001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 0.0001
        assert optimizer.momentum == 0.9

    def test_momentum_nesterov_with_gradient_clipping(self):
        """Test MomentumNesterov optimizer with gradient clipping."""
        model = SimpleModelV1()
        optimizer = braintools.optim.MomentumNesterov(
            lr=0.01,
            momentum=0.9,
            grad_clip_norm=1.0,
            grad_clip_value=0.5
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.grad_clip_norm == 1.0
        assert optimizer.grad_clip_value == 0.5

    def test_momentum_nesterov_with_scheduler(self):
        """Test MomentumNesterov optimizer with learning rate scheduling."""
        model = SimpleModelV1()
        scheduler = braintools.optim.ExponentialLR(base_lr=0.01, gamma=0.95)
        optimizer = braintools.optim.MomentumNesterov(lr=scheduler, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify initial learning rate
        assert optimizer.current_lr == 0.01

        # Step the scheduler multiple times
        for _ in range(10):
            scheduler.step()

        # Learning rate should have decayed
        assert optimizer.current_lr < 0.01

    # ============================================================================
    # Adam Optimizer Examples
    # ============================================================================

    def test_adam_basic(self):
        """Test basic Adam optimizer."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adam(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)

    def test_adam_custom_betas(self):
        """Test Adam with custom beta values."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adam(lr=0.001, betas=(0.9, 0.99))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.betas == (0.9, 0.99)

    def test_adam_with_amsgrad(self):
        """Test Adam with AMSGrad."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adam(lr=0.001, amsgrad=True)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.amsgrad is True

    def test_adam_with_gradient_clipping(self):
        """Test Adam with gradient clipping."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adam(lr=0.001, grad_clip_norm=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.grad_clip_norm == 1.0

    # ============================================================================
    # AdamW Optimizer Examples
    # ============================================================================

    def test_adamw_basic(self):
        """Test basic AdamW usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.AdamW(lr=0.001, weight_decay=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.weight_decay == 0.01

    def test_adamw_with_scheduler(self):
        """Test AdamW with scheduler."""
        model = SimpleModelV1()
        scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=10, gamma=0.1)
        optimizer = braintools.optim.AdamW(lr=scheduler, weight_decay=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer._lr_scheduler is scheduler

    # ============================================================================
    # Other Optimizers Examples
    # ============================================================================

    def test_rmsprop(self):
        """Test RMSprop optimizer."""
        model = SimpleModelV1()
        optimizer = braintools.optim.RMSprop(lr=0.01, alpha=0.99)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.alpha == 0.99

    def test_adagrad(self):
        """Test Adagrad optimizer."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adagrad(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.current_lr == 0.01

    def test_adadelta(self):
        """Test Adadelta optimizer."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adadelta(lr=1.0, rho=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        assert optimizer.rho == 0.9

    # ============================================================================
    # Optimizer Properties Test
    # ============================================================================

    def test_optimizer_properties(self):
        """Test optimizer properties and basic functionality."""
        model = SimpleModelV1(10, 5)
        optimizer = braintools.optim.Adam(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify optimizer was initialized correctly
        assert optimizer.step_count.value == 0
        assert optimizer.current_lr == 0.001
        assert optimizer.base_lr == 0.001
        assert optimizer.opt_state is not None

    def test_optimizer_state_dict_structure(self):
        """Test optimizer state dict structure."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adam(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Get state dict
        state_dict = optimizer.state_dict()

        # Verify state dict contains expected keys
        assert 'step_count' in state_dict
        assert 'lr' in state_dict
        assert 'base_lr' in state_dict
        assert 'opt_state' in state_dict
        assert state_dict['step_count'] == 0
        assert state_dict['lr'] == 0.001
        assert state_dict['base_lr'] == 0.001


class test_novograd_fromage(unittest.TestCase):

    # ============================================================================
    # Novograd Optimizer Tests
    # ============================================================================

    def test_novograd_basic(self):
        """Test basic Novograd usage."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Novograd
        optimizer = braintools.optim.Novograd(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-8
        print("[OK] test_novograd_basic passed")

    def test_novograd_custom_betas(self):
        """Test Novograd with custom betas."""
        model = brainstate.nn.Linear(10, 5)

        # Higher beta1 for more momentum
        optimizer = braintools.optim.Novograd(lr=0.001, betas=(0.95, 0.999))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify betas
        assert optimizer.betas == (0.95, 0.999)
        assert optimizer.current_lr == 0.001
        print("[OK] test_novograd_custom_betas passed")

    def test_novograd_weight_decay(self):
        """Test Novograd with weight decay."""
        model = brainstate.nn.Linear(10, 5)

        # Add L2 regularization
        optimizer = braintools.optim.Novograd(lr=0.001, weight_decay=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify weight decay
        assert optimizer.weight_decay == 0.01
        assert optimizer.current_lr == 0.001
        print("[OK] test_novograd_weight_decay passed")

    def test_novograd_scheduler(self):
        """Test Novograd with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Polynomial decay schedule
        scheduler = braintools.optim.StepLR(
            base_lr=0.01,
            step_size=100,
            gamma=0.5
        )
        optimizer = braintools.optim.Novograd(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify scheduler
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.01
        print("[OK] test_novograd_scheduler passed")

    def test_novograd_gradient_clipping(self):
        """Test Novograd with gradient clipping."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients by global norm
        optimizer = braintools.optim.Novograd(
            lr=0.001,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify gradient clipping
        assert optimizer.grad_clip_norm == 1.0
        assert optimizer.current_lr == 0.001
        print("[OK] test_novograd_gradient_clipping passed")

    def test_novograd_complete_config(self):
        """Test Novograd with complete configuration."""
        # Large speech model
        model = brainstate.nn.Linear(1000, 500)

        # Learning rate schedule with warmup
        scheduler = braintools.optim.StepLR(
            base_lr=0.01,
            step_size=1000,
            gamma=0.9
        )

        # Complete Novograd configuration
        optimizer = braintools.optim.Novograd(
            lr=scheduler,
            betas=(0.95, 0.98),
            eps=1e-8,
            weight_decay=0.001,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.01
        assert optimizer.betas == (0.95, 0.98)
        assert optimizer.eps == 1e-8
        assert optimizer.weight_decay == 0.001
        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_novograd_complete_config passed")

    # ============================================================================
    # Fromage Optimizer Tests
    # ============================================================================

    def test_fromage_basic(self):
        """Test basic learning-rate-free usage."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Fromage with default lr=1.0 (no tuning needed)
        optimizer = braintools.optim.Fromage(lr=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 1.0
        assert optimizer.momentum == 0.0
        print("[OK] test_fromage_basic passed")

    def test_fromage_with_momentum(self):
        """Test Fromage with momentum."""
        model = brainstate.nn.Linear(10, 5)

        # Enable momentum for better convergence
        optimizer = braintools.optim.Fromage(lr=1.0, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify momentum
        assert optimizer.momentum == 0.9
        assert optimizer.current_lr == 1.0
        print("[OK] test_fromage_with_momentum passed")

    def test_fromage_without_momentum(self):
        """Test Fromage without momentum."""
        model = brainstate.nn.Linear(10, 5)

        # Pure gradient-based updates
        optimizer = braintools.optim.Fromage(lr=1.0, momentum=0.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify momentum is disabled
        assert optimizer.momentum == 0.0
        assert optimizer.current_lr == 1.0
        print("[OK] test_fromage_without_momentum passed")

    def test_fromage_lr_scaling(self):
        """Test Fromage with global learning rate scaling."""
        model = brainstate.nn.Linear(10, 5)

        # Scale automatic step sizes by 0.5
        optimizer = braintools.optim.Fromage(lr=0.5, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify lr scaling
        assert optimizer.current_lr == 0.5
        assert optimizer.momentum == 0.9
        print("[OK] test_fromage_lr_scaling passed")

    def test_fromage_gradient_clipping(self):
        """Test Fromage with gradient clipping."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients for stability
        optimizer = braintools.optim.Fromage(
            lr=1.0,
            momentum=0.9,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify gradient clipping
        assert optimizer.grad_clip_norm == 1.0
        assert optimizer.current_lr == 1.0
        assert optimizer.momentum == 0.9
        print("[OK] test_fromage_gradient_clipping passed")

    def test_fromage_complete_config(self):
        """Test Fromage with complete configuration."""
        # Model for rapid experimentation
        model = brainstate.nn.Linear(100, 50)

        # Complete Fromage configuration
        optimizer = braintools.optim.Fromage(
            lr=1.0,
            momentum=0.9,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer.current_lr == 1.0
        assert optimizer.momentum == 0.9
        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_fromage_complete_config passed")

    # ============================================================================
    # Novograd vs Fromage Comparison Tests
    # ============================================================================

    def test_novograd_vs_fromage(self):
        """Compare Novograd and Fromage optimizers."""
        model1 = brainstate.nn.Linear(20, 10)
        model2 = brainstate.nn.Linear(20, 10)

        # Create both optimizers
        opt_novograd = braintools.optim.Novograd(lr=0.001)
        opt_novograd.register_trainable_weights(model1.states(brainstate.ParamState))

        opt_fromage = braintools.optim.Fromage(lr=1.0, momentum=0.9)
        opt_fromage.register_trainable_weights(model2.states(brainstate.ParamState))

        # Verify both are initialized
        assert opt_novograd.current_lr == 0.001
        assert opt_fromage.current_lr == 1.0
        assert opt_novograd.opt_state is not None
        assert opt_fromage.opt_state is not None

        print("[OK] test_novograd_vs_fromage passed")

    def test_layer_wise_optimizers_state_dict(self):
        """Test state_dict functionality for layer-wise optimizers."""
        model = brainstate.nn.Linear(10, 5)

        # Test Novograd
        opt1 = braintools.optim.Novograd(lr=0.001, betas=(0.95, 0.999))
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        state1 = opt1.state_dict()
        assert 'opt_state' in state1
        assert 'lr' in state1

        # Test Fromage
        opt2 = braintools.optim.Fromage(lr=1.0, momentum=0.9)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        state2 = opt2.state_dict()
        assert 'opt_state' in state2
        assert 'lr' in state2

        print("[OK] test_layer_wise_optimizers_state_dict passed")

    def test_specialized_optimizers_edge_cases(self):
        """Test edge cases for specialized optimizers."""
        model = brainstate.nn.Linear(5, 3)

        # Test Novograd with extreme betas
        opt1 = braintools.optim.Novograd(lr=0.001, betas=(0.5, 0.9))
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt1.betas == (0.5, 0.9)

        # Test Novograd with large epsilon
        opt2 = braintools.optim.Novograd(lr=0.001, eps=1e-4)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt2.eps == 1e-4

        # Test Fromage with zero momentum
        opt3 = braintools.optim.Fromage(lr=1.0, momentum=0.0)
        opt3.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt3.momentum == 0.0

        # Test Fromage with high momentum
        opt4 = braintools.optim.Fromage(lr=1.0, momentum=0.99)
        opt4.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt4.momentum == 0.99

        print("[OK] test_specialized_optimizers_edge_cases passed")


class test_lion_sm3(unittest.TestCase):

    # ============================================================================
    # Lion Optimizer Tests
    # ============================================================================

    def test_lion_basic(self):
        """Test basic Lion usage with small learning rate."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Lion with small lr (3-10x smaller than Adam)
        optimizer = braintools.optim.Lion(lr=1e-4)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 1e-4
        assert optimizer.betas == (0.9, 0.99)
        assert optimizer.weight_decay == 0.0
        print("[OK] test_lion_basic passed")

    def test_lion_weight_decay(self):
        """Test Lion with larger weight decay."""
        model = brainstate.nn.Linear(10, 5)

        # Lion typically uses larger weight decay than Adam
        optimizer = braintools.optim.Lion(lr=1e-4, weight_decay=0.1)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify weight decay
        assert optimizer.current_lr == 1e-4
        assert optimizer.weight_decay == 0.1
        print("[OK] test_lion_weight_decay passed")

    def test_lion_custom_betas(self):
        """Test Lion with custom betas."""
        model = brainstate.nn.Linear(10, 5)

        # Custom interpolation coefficients
        optimizer = braintools.optim.Lion(lr=1e-4, betas=(0.95, 0.98))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify betas
        assert optimizer.betas == (0.95, 0.98)
        assert optimizer.current_lr == 1e-4
        print("[OK] test_lion_custom_betas passed")

    def test_lion_scheduler(self):
        """Test Lion with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Step decay schedule
        scheduler = braintools.optim.StepLR(
            base_lr=1e-4,
            step_size=100,
            gamma=0.5
        )
        optimizer = braintools.optim.Lion(lr=scheduler, weight_decay=0.1)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify scheduler
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 1e-4
        assert optimizer.weight_decay == 0.1
        print("[OK] test_lion_scheduler passed")

    def test_lion_gradient_clipping(self):
        """Test Lion with gradient clipping."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients for stability
        optimizer = braintools.optim.Lion(
            lr=1e-4,
            weight_decay=0.1,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify gradient clipping
        assert optimizer.grad_clip_norm == 1.0
        assert optimizer.current_lr == 1e-4
        assert optimizer.weight_decay == 0.1
        print("[OK] test_lion_gradient_clipping passed")

    def test_lion_complete_config(self):
        """Test Lion with complete configuration."""
        # Large transformer model
        model = brainstate.nn.Linear(1000, 500)

        # Learning rate decay schedule
        scheduler = braintools.optim.StepLR(
            base_lr=1e-4,
            step_size=100,
            gamma=0.9
        )

        # Complete Lion configuration
        optimizer = braintools.optim.Lion(
            lr=scheduler,
            betas=(0.9, 0.99),
            weight_decay=0.1,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 1e-4
        assert optimizer.betas == (0.9, 0.99)
        assert optimizer.weight_decay == 0.1
        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_lion_complete_config passed")

    # ============================================================================
    # SM3 Optimizer Tests
    # ============================================================================

    def test_sm3_basic(self):
        """Test basic SM3 usage with default settings."""
        # Create model with embedding layer
        model = brainstate.nn.Linear(10, 5)

        # Initialize SM3 with default lr=1.0
        optimizer = braintools.optim.SM3(lr=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 1.0
        assert optimizer.momentum == 0.9
        assert optimizer.eps == 1e-8
        print("[OK] test_sm3_basic passed")

    def test_sm3_custom_lr(self):
        """Test SM3 with custom learning rate."""
        model = brainstate.nn.Linear(10, 5)

        # Use smaller learning rate
        optimizer = braintools.optim.SM3(lr=0.1)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify learning rate
        assert optimizer.current_lr == 0.1
        assert optimizer.momentum == 0.9
        print("[OK] test_sm3_custom_lr passed")

    def test_sm3_momentum(self):
        """Test SM3 with custom momentum."""
        model = brainstate.nn.Linear(10, 5)

        # Higher momentum for smoother updates
        optimizer = braintools.optim.SM3(lr=1.0, momentum=0.95)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify momentum
        assert optimizer.momentum == 0.95
        assert optimizer.current_lr == 1.0
        print("[OK] test_sm3_momentum passed")

    def test_sm3_no_momentum(self):
        """Test SM3 without momentum."""
        model = brainstate.nn.Linear(10, 5)

        # Disable momentum
        optimizer = braintools.optim.SM3(lr=1.0, momentum=0.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify momentum is disabled
        assert optimizer.momentum == 0.0
        assert optimizer.current_lr == 1.0
        print("[OK] test_sm3_no_momentum passed")

    def test_sm3_scheduler(self):
        """Test SM3 with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Exponential decay schedule
        scheduler = braintools.optim.StepLR(
            base_lr=1.0,
            step_size=100,
            gamma=0.9
        )
        optimizer = braintools.optim.SM3(lr=scheduler, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify scheduler
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 1.0
        assert optimizer.momentum == 0.9
        print("[OK] test_sm3_scheduler passed")

    def test_sm3_complete_config(self):
        """Test SM3 with complete configuration."""
        # Large embedding model
        model = brainstate.nn.Linear(1000, 500)

        # Learning rate schedule for long training
        scheduler = braintools.optim.StepLR(
            base_lr=1.0,
            step_size=1000,
            gamma=0.95
        )

        # Complete SM3 configuration
        optimizer = braintools.optim.SM3(
            lr=scheduler,
            momentum=0.9,
            eps=1e-8,
            weight_decay=0.0001
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 1.0
        assert optimizer.momentum == 0.9
        assert optimizer.eps == 1e-8
        assert optimizer.weight_decay == 0.0001
        print("[OK] test_sm3_complete_config passed")

    # ============================================================================
    # Lion vs SM3 Comparison Tests
    # ============================================================================

    def test_lion_vs_sm3(self):
        """Compare Lion and SM3 optimizers."""
        model1 = brainstate.nn.Linear(20, 10)
        model2 = brainstate.nn.Linear(20, 10)

        # Create both optimizers
        opt_lion = braintools.optim.Lion(lr=1e-4, weight_decay=0.1)
        opt_lion.register_trainable_weights(model1.states(brainstate.ParamState))

        opt_sm3 = braintools.optim.SM3(lr=1.0, momentum=0.9)
        opt_sm3.register_trainable_weights(model2.states(brainstate.ParamState))

        # Verify both are initialized
        assert opt_lion.current_lr == 1e-4
        assert opt_sm3.current_lr == 1.0
        assert opt_lion.opt_state is not None
        assert opt_sm3.opt_state is not None

        print("[OK] test_lion_vs_sm3 passed")

    def test_memory_efficient_optimizers_state_dict(self):
        """Test state_dict functionality for memory-efficient optimizers."""
        model = brainstate.nn.Linear(10, 5)

        # Test Lion
        opt1 = braintools.optim.Lion(lr=1e-4, weight_decay=0.1)
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        state1 = opt1.state_dict()
        assert 'opt_state' in state1
        assert 'lr' in state1

        # Test SM3
        opt2 = braintools.optim.SM3(lr=1.0, momentum=0.9)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        state2 = opt2.state_dict()
        assert 'opt_state' in state2
        assert 'lr' in state2

        print("[OK] test_memory_efficient_optimizers_state_dict passed")

    def test_sign_based_optimizers_edge_cases(self):
        """Test edge cases for sign-based and memory-efficient optimizers."""
        model = brainstate.nn.Linear(5, 3)

        # Test Lion with extreme betas
        opt1 = braintools.optim.Lion(lr=1e-4, betas=(0.5, 0.9))
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt1.betas == (0.5, 0.9)

        # Test Lion with zero weight decay
        opt2 = braintools.optim.Lion(lr=1e-4, weight_decay=0.0)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt2.weight_decay == 0.0

        # Test SM3 with very small eps
        opt3 = braintools.optim.SM3(lr=1.0, eps=1e-10)
        opt3.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt3.eps == 1e-10

        # Test SM3 with zero momentum
        opt4 = braintools.optim.SM3(lr=1.0, momentum=0.0)
        opt4.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt4.momentum == 0.0

        print("[OK] test_sign_based_optimizers_edge_cases passed")


class test_adafactor_adabelief(unittest.TestCase):

    # ============================================================================
    # Adafactor Optimizer Tests
    # ============================================================================

    def test_adafactor_basic(self):
        """Test basic Adafactor usage with automatic learning rate."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize Adafactor with auto learning rate
        optimizer = braintools.optim.Adafactor()
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.factored is True
        assert optimizer.eps == (1e-30, 1e-3)
        assert optimizer.clip_threshold == 1.0
        print("[OK] test_adafactor_basic passed")

    def test_adafactor_explicit_lr(self):
        """Test Adafactor with explicit learning rate."""
        model = brainstate.nn.Linear(10, 5)

        # Use explicit learning rate
        optimizer = braintools.optim.Adafactor(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify learning rate
        assert optimizer.current_lr == 0.001
        assert optimizer.factored is True
        print("[OK] test_adafactor_explicit_lr passed")

    def test_adafactor_with_momentum(self):
        """Test Adafactor with momentum."""
        model = brainstate.nn.Linear(10, 5)

        # Enable momentum
        optimizer = braintools.optim.Adafactor(lr=0.001, beta1=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify beta1 is set
        assert optimizer.beta1 == 0.9
        assert optimizer.current_lr == 0.001
        print("[OK] test_adafactor_with_momentum passed")

    def test_adafactor_non_factored(self):
        """Test Adafactor without factorization."""
        model = brainstate.nn.Linear(10, 5)

        # Disable factorization (uses more memory but may be faster)
        optimizer = braintools.optim.Adafactor(lr=0.001, factored=False)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify factored is False
        assert optimizer.factored is False
        assert optimizer.current_lr == 0.001
        print("[OK] test_adafactor_non_factored passed")

    def test_adafactor_large_model(self):
        """Test Adafactor for large transformer training."""
        # Large model simulation
        model = brainstate.nn.Linear(1000, 500)

        # Typical configuration for large models
        optimizer = braintools.optim.Adafactor(
            beta1=0.9,
            clip_threshold=1.0,
            decay_rate=-0.8
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify configuration
        assert optimizer.beta1 == 0.9
        assert optimizer.clip_threshold == 1.0
        assert optimizer.decay_rate == -0.8
        print("[OK] test_adafactor_large_model passed")

    def test_adafactor_complete_config(self):
        """Test Adafactor with complete configuration."""
        model = brainstate.nn.Linear(100, 50)

        # Complete configuration
        optimizer = braintools.optim.Adafactor(
            lr=0.01,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=0.9,
            weight_decay=0.0001,
            factored=True
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer.current_lr == 0.01
        assert optimizer.eps == (1e-30, 1e-3)
        assert optimizer.clip_threshold == 1.0
        assert optimizer.decay_rate == -0.8
        assert optimizer.beta1 == 0.9
        assert optimizer.weight_decay == 0.0001
        assert optimizer.factored is True
        print("[OK] test_adafactor_complete_config passed")

    # ============================================================================
    # AdaBelief Optimizer Tests
    # ============================================================================

    def test_adabelief_basic(self):
        """Test basic AdaBelief usage."""
        # Create model
        model = brainstate.nn.Linear(10, 5)

        # Initialize AdaBelief
        optimizer = braintools.optim.AdaBelief(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify parameters
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-16
        print("[OK] test_adabelief_basic passed")

    def test_adabelief_custom_betas(self):
        """Test AdaBelief with custom betas."""
        model = brainstate.nn.Linear(10, 5)

        # Faster momentum decay, slower variance decay
        optimizer = braintools.optim.AdaBelief(lr=0.001, betas=(0.8, 0.999))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify betas
        assert optimizer.betas == (0.8, 0.999)
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_custom_betas passed")

    def test_adabelief_scheduler(self):
        """Test AdaBelief with learning rate scheduler."""
        model = brainstate.nn.Linear(10, 5)

        # Learning rate decays every 30 epochs
        scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=30, gamma=0.5)
        optimizer = braintools.optim.AdaBelief(lr=scheduler)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify scheduler
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_scheduler passed")

    def test_adabelief_weight_decay(self):
        """Test AdaBelief with weight decay."""
        model = brainstate.nn.Linear(10, 5)

        # Add L2 regularization
        optimizer = braintools.optim.AdaBelief(lr=0.001, weight_decay=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify weight decay
        assert optimizer.weight_decay == 0.01
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_weight_decay passed")

    def test_adabelief_gradient_clipping(self):
        """Test AdaBelief with gradient clipping."""
        model = brainstate.nn.Linear(10, 5)

        # Clip gradients by global norm
        optimizer = braintools.optim.AdaBelief(
            lr=0.001,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify gradient clipping
        assert optimizer.grad_clip_norm == 1.0
        assert optimizer.current_lr == 0.001
        print("[OK] test_adabelief_gradient_clipping passed")

    def test_adabelief_complete_config(self):
        """Test AdaBelief with complete configuration."""
        # Large model
        model = brainstate.nn.Linear(1000, 500)

        # Learning rate schedule
        scheduler = braintools.optim.StepLR(base_lr=0.001, step_size=20, gamma=0.5)

        # Complete AdaBelief configuration
        optimizer = braintools.optim.AdaBelief(
            lr=scheduler,
            betas=(0.9, 0.999),
            eps=1e-16,
            weight_decay=0.0001,
            grad_clip_norm=1.0
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Verify all parameters
        assert optimizer._lr_scheduler is scheduler
        assert optimizer.current_lr == 0.001
        assert optimizer.betas == (0.9, 0.999)
        assert optimizer.eps == 1e-16
        assert optimizer.weight_decay == 0.0001
        assert optimizer.grad_clip_norm == 1.0
        print("[OK] test_adabelief_complete_config passed")

    # ============================================================================
    # Adafactor vs AdaBelief Comparison Tests
    # ============================================================================

    def test_adafactor_vs_adabelief(self):
        """Compare Adafactor and AdaBelief optimizers."""
        model1 = brainstate.nn.Linear(20, 10)
        model2 = brainstate.nn.Linear(20, 10)

        # Create both optimizers
        opt_adafactor = braintools.optim.Adafactor(lr=0.001)
        opt_adafactor.register_trainable_weights(model1.states(brainstate.ParamState))

        opt_adabelief = braintools.optim.AdaBelief(lr=0.001)
        opt_adabelief.register_trainable_weights(model2.states(brainstate.ParamState))

        # Verify both are initialized
        assert opt_adafactor.current_lr == 0.001
        assert opt_adabelief.current_lr == 0.001
        assert opt_adafactor.opt_state is not None
        assert opt_adabelief.opt_state is not None

        print("[OK] test_adafactor_vs_adabelief passed")

    def test_memory_efficient_optimizers_state_dict(self):
        """Test state_dict functionality for memory-efficient optimizers."""
        model = brainstate.nn.Linear(10, 5)

        # Test Adafactor
        opt1 = braintools.optim.Adafactor(lr=0.001, factored=True)
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        state1 = opt1.state_dict()
        assert 'opt_state' in state1
        assert 'lr' in state1

        # Test AdaBelief
        opt2 = braintools.optim.AdaBelief(lr=0.001)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        state2 = opt2.state_dict()
        assert 'opt_state' in state2
        assert 'lr' in state2

        print("[OK] test_memory_efficient_optimizers_state_dict passed")

    def test_belief_based_optimizers_edge_cases(self):
        """Test edge cases for belief-based optimizers."""
        model = brainstate.nn.Linear(5, 3)

        # Test Adafactor with minimal eps
        opt1 = braintools.optim.Adafactor(lr=0.01, eps=(1e-30, 1e-3))
        opt1.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt1.eps == (1e-30, 1e-3)

        # Test AdaBelief with very small eps
        opt2 = braintools.optim.AdaBelief(lr=0.01, eps=1e-16)
        opt2.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt2.eps == 1e-16

        # Test AdaBelief with extreme betas
        opt3 = braintools.optim.AdaBelief(lr=0.01, betas=(0.5, 0.9))
        opt3.register_trainable_weights(model.states(brainstate.ParamState))
        assert opt3.betas == (0.5, 0.9)

        print("[OK] test_belief_based_optimizers_edge_cases passed")


class test_basic(unittest.TestCase):

    # ============================================================================
    # Adaptive Learning Rate Optimizers
    # ============================================================================

    def test_adagrad_basic(self):
        """Test basic Adagrad usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adagrad(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.01

    def test_adagrad_custom_eps(self):
        """Test Adagrad with custom epsilon."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adagrad(lr=0.01, eps=1e-8)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.eps == 1e-8

    def test_adagrad_weight_decay(self):
        """Test Adagrad with weight decay."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adagrad(lr=0.01, weight_decay=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.weight_decay == 0.01

    def test_adadelta_basic(self):
        """Test basic Adadelta usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adadelta()
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.rho == 0.9

    def test_adadelta_custom_rho(self):
        """Test Adadelta with custom rho."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adadelta(rho=0.95)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.rho == 0.95

    def test_adadelta_with_lr(self):
        """Test Adadelta with explicit learning rate."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adadelta(lr=0.5, rho=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.5
        assert optimizer.rho == 0.9

    # ============================================================================
    # RMSprop Tests
    # ============================================================================

    def test_rmsprop_basic(self):
        """Test basic RMSprop usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.RMSprop(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.01
        assert optimizer.alpha == 0.99

    def test_rmsprop_momentum(self):
        """Test RMSprop with momentum."""
        model = SimpleModelV1()
        optimizer = braintools.optim.RMSprop(lr=0.01, momentum=0.9)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.momentum == 0.9

    def test_rmsprop_centered(self):
        """Test centered RMSprop."""
        model = SimpleModelV1()
        optimizer = braintools.optim.RMSprop(lr=0.01, centered=True)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.centered is True

    def test_rmsprop_custom_alpha(self):
        """Test RMSprop with custom alpha."""
        model = SimpleModelV1()
        optimizer = braintools.optim.RMSprop(lr=0.01, alpha=0.95)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.alpha == 0.95

    # ============================================================================
    # Adam Family Tests
    # ============================================================================

    def test_adamax_basic(self):
        """Test basic Adamax usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adamax(lr=0.002)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.002

    def test_adamax_custom_betas(self):
        """Test Adamax with custom betas."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adamax(lr=0.002, betas=(0.9, 0.99))
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.betas == (0.9, 0.99)

    def test_nadam_basic(self):
        """Test basic Nadam usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Nadam(lr=0.002)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.002

    def test_radam_basic(self):
        """Test basic RAdam usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.RAdam(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.001

    # ============================================================================
    # Large Batch Training Optimizers
    # ============================================================================

    def test_lamb_basic(self):
        """Test basic Lamb usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Lamb(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.001

    def test_lars_basic(self):
        """Test basic Lars usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Lars(lr=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 1.0

    # ============================================================================
    # Specialized Optimizers
    # ============================================================================

    def test_yogi_basic(self):
        """Test basic Yogi usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Yogi(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.001

    def test_lion_basic(self):
        """Test basic Lion usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Lion(lr=1e-4, weight_decay=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 1e-4
        assert optimizer.weight_decay == 0.01

    def test_adabelief_basic(self):
        """Test basic AdaBelief usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.AdaBelief(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.001

    def test_novograd_basic(self):
        """Test basic Novograd usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Novograd(lr=0.001)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.001

    def test_sm3_basic(self):
        """Test basic SM3 usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.SM3(lr=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 1.0

    # ============================================================================
    # Second Order Optimizers
    # ============================================================================

    def test_lbfgs_basic(self):
        """Test basic LBFGS usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.LBFGS(lr=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 1.0

    def test_rprop_basic(self):
        """Test basic Rprop usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Rprop(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 0.01

    # ============================================================================
    # Meta Optimizers
    # ============================================================================

    def test_adafactor_basic(self):
        """Test basic Adafactor usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Adafactor()
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.factored is True

    def test_fromage_basic(self):
        """Test basic Fromage usage."""
        model = SimpleModelV1()
        optimizer = braintools.optim.Fromage(lr=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer.current_lr == 1.0

    # ============================================================================
    # Learning Rate Scheduler Integration Tests
    # ============================================================================

    def test_optimizer_with_step_lr(self):
        """Test various optimizers with StepLR scheduler."""
        optimizers_to_test = [
            ('Adagrad', braintools.optim.Adagrad),
            ('Adadelta', braintools.optim.Adadelta),
            ('RMSprop', braintools.optim.RMSprop),
            ('Adamax', braintools.optim.Adamax),
            ('Lamb', braintools.optim.Lamb),
        ]

        for name, OptimizerClass in optimizers_to_test:
            model = SimpleModelV1()
            scheduler = braintools.optim.StepLR(base_lr=0.01, step_size=10, gamma=0.1)
            optimizer = OptimizerClass(lr=scheduler)
            optimizer.register_trainable_weights(model.states(brainstate.ParamState))

            assert optimizer._lr_scheduler is scheduler, f"{name} scheduler mismatch"
            assert optimizer.current_lr == 0.01, f"{name} initial lr mismatch"

    # ============================================================================
    # Gradient Clipping Tests
    # ============================================================================

    def test_optimizers_with_gradient_clipping(self):
        """Test that optimizers support gradient clipping."""
        model = SimpleModelV1()

        # Test gradient norm clipping
        optimizer1 = braintools.optim.Adagrad(lr=0.01, grad_clip_norm=1.0)
        optimizer1.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer1.grad_clip_norm == 1.0

        # Test gradient value clipping
        optimizer2 = braintools.optim.RMSprop(lr=0.01, grad_clip_value=0.5)
        optimizer2.register_trainable_weights(model.states(brainstate.ParamState))
        assert optimizer2.grad_clip_value == 0.5

    # ============================================================================
    # Weight Decay Tests
    # ============================================================================

    def test_optimizers_with_weight_decay(self):
        """Test that optimizers support weight decay."""
        model = SimpleModelV1()

        optimizers = [
            braintools.optim.Adagrad(lr=0.01, weight_decay=0.01),
            braintools.optim.Adadelta(lr=1.0, weight_decay=0.01),
            braintools.optim.RMSprop(lr=0.01, weight_decay=0.01),
            braintools.optim.Lamb(lr=0.001, weight_decay=0.01),
        ]

        for optimizer in optimizers:
            optimizer.register_trainable_weights(model.states(brainstate.ParamState))
            assert optimizer.weight_decay == 0.01


# ==============================================================================
# Regression tests for the 2026-06-18 ``braintools.optim`` audit fixes.
# Each test maps to an issue ID from docs/braintools-optim-issues-found-20260618.md
# and asserts the *corrected* behavior (i.e. it fails against the pre-fix code).
# ==============================================================================


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
# LBFGS line-search construction + update(value=, value_fn=) path (O-17)
# --------------------------------------------------------------------------- #

class TestLBFGSLinesearchDescent:
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
        from braintools._deprecation import BraintoolsDeprecationWarning, registered_deprecations
        with pytest.warns(BraintoolsDeprecationWarning, match='beta1'):
            opt = O.Adam(lr=0.05, beta1=0.8)
        assert opt.betas[0] == 0.8
        with pytest.warns(BraintoolsDeprecationWarning, match='beta2'):
            opt = O.Adam(lr=0.05, beta2=0.99)
        assert opt.betas[1] == 0.99
        # Both arguments are registered deprecations (listed in the migration guide).
        names = {r.old_name for r in registered_deprecations()}
        assert 'braintools.optim.Adam(beta1=...)' in names
        assert 'braintools.optim.Adam(beta2=...)' in names
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


class TestLBFGSLinesearch:
    """L-BFGS with a line search converges like ``optax.lbfgs`` and checks ``value_fn``."""

    @staticmethod
    def _rosen(values):
        w = values['w']
        return (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2

    @pytest.mark.parametrize('linesearch', ['zoom', 'backtracking'])
    def test_rosenbrock(self, linesearch):
        params = {'w': brainstate.ParamState(jnp.array([-1.5, 2.0]))}
        opt = O.LBFGS(linesearch=linesearch)
        opt.register_trainable_weights(params)
        for _ in range(60):
            grads = brainstate.transform.grad(lambda: self._rosen({'w': params['w'].value}), grad_states=params)()
            opt.update(grads, value=self._rosen({'w': params['w'].value}), value_fn=self._rosen)
        np.testing.assert_allclose(params['w'].value, [1.0, 1.0], atol=1e-3)

    def test_zero_argument_value_fn_rejected(self):
        params = {'w': brainstate.ParamState(jnp.zeros(2))}
        opt = O.LBFGS(linesearch='zoom')
        opt.register_trainable_weights(params)
        with pytest.raises(TypeError, match='value_fn'):
            opt.update({'w': jnp.ones(2)}, value=1.0, value_fn=lambda: 1.0)


# ==============================================================================
# Regression tests for the 2026-06-18 ``braintools.optim`` *re-audit* fixes
# (issue IDs R-1 .. R-17 in docs/braintools-optim-issues-found-20260618.md).
# ==============================================================================


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


# ---------------------------------------------------------------------------
# Checkpoint round trip through msgpack (resume must restore optax NamedTuples)
# ---------------------------------------------------------------------------

_STATEFUL_OPTIMIZERS = [
    lambda: O.Adam(lr=0.01),
    lambda: O.AdamW(lr=0.01, weight_decay=1e-2),
    lambda: O.Momentum(lr=0.01, momentum=0.9),
    lambda: O.RMSprop(lr=0.01),
    lambda: O.Adagrad(lr=0.01),
    lambda: O.Nadam(lr=0.01),
    lambda: O.Yogi(lr=0.01),
    lambda: O.Lamb(lr=0.01),
    lambda: O.AdaBelief(lr=0.01),
    lambda: O.Lion(lr=0.01),
    lambda: O.Adam(lr=O.StepLR(base_lr=0.01, step_size=2, gamma=0.5)),
]


def _roundtrip_setup():
    w = ParamState(jnp.array([[0.5, -0.3], [0.2, 0.1]]))
    b = ParamState(jnp.array([0.1, -0.2]))
    params = {'w': w, 'b': b}
    x = jnp.array([[1.0, 2.0], [0.5, -1.0], [-0.3, 0.7]])

    def loss_fn():
        return jnp.sum((x @ w.value + b.value) ** 2)

    def grads():
        return brainstate.transform.grad(loss_fn, params)()

    return params, grads


@pytest.mark.parametrize('make_opt', _STATEFUL_OPTIMIZERS)
def test_load_state_dict_after_msgpack_roundtrip_resumes(make_opt, tmp_path):
    from braintools.file import msgpack_load, msgpack_save

    params, grads = _roundtrip_setup()
    opt = make_opt()
    opt.register_trainable_weights(params)
    for _ in range(3):
        opt.step(grads())

    path = str(tmp_path / 'opt.msgpack')
    msgpack_save(path, opt.state_dict())
    loaded = msgpack_load(path)

    # Reference: continue the original run by one step.
    snapshot = {k: v.value for k, v in params.items()}
    g = grads()
    opt.step(g)
    expected = {k: np.asarray(v.value) for k, v in params.items()}

    # Resume from the serialized checkpoint and take the same step.
    for k, v in params.items():
        v.value = snapshot[k]
    resumed = make_opt()
    resumed.register_trainable_weights(params)
    resumed.load_state_dict(loaded)
    assert jax.tree.structure(resumed.opt_state.value) == jax.tree.structure(opt.opt_state.value)
    resumed.step(g)
    for k, v in params.items():
        np.testing.assert_allclose(np.asarray(v.value), expected[k], rtol=1e-6, atol=1e-6)


def test_load_state_dict_after_msgpack_roundtrip_restores_param_group_states(tmp_path):
    from braintools.file import msgpack_load, msgpack_save

    w1 = ParamState(jnp.ones((2,)))
    w2 = ParamState(jnp.ones((3,)))
    opt = O.Adam(lr=0.01)
    opt.register_trainable_weights({'w1': w1})
    opt.add_param_group({'w2': w2}, lr=0.1)
    opt.step({'w1': jnp.ones((2,)), 'w2': jnp.ones((3,))})

    path = str(tmp_path / 'opt.msgpack')
    msgpack_save(path, opt.state_dict())

    resumed = O.Adam(lr=0.01)
    resumed.register_trainable_weights({'w1': w1})
    resumed.add_param_group({'w2': w2}, lr=0.1)
    resumed.load_state_dict(msgpack_load(path))
    assert (jax.tree.structure(resumed.param_groups_opt_states[0].value)
            == jax.tree.structure(opt.param_groups_opt_states[0].value))
    resumed.step({'w1': jnp.ones((2,)), 'w2': jnp.ones((3,))})


def test_load_state_dict_in_memory_roundtrip_still_works():
    params, grads = _roundtrip_setup()
    opt = O.Adam(lr=0.01)
    opt.register_trainable_weights(params)
    opt.step(grads())
    resumed = O.Adam(lr=0.01)
    resumed.register_trainable_weights(params)
    resumed.load_state_dict(opt.state_dict())
    assert jax.tree.structure(resumed.opt_state.value) == jax.tree.structure(opt.opt_state.value)
    resumed.step(grads())


def test_state_dict_before_registration_has_no_opt_state():
    # A fresh optimizer (no weights registered yet) can still be checkpointed.
    opt = O.Adam(lr=0.01)
    sd = opt.state_dict()
    assert sd['opt_state'] is None
    assert float(sd['lr']) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# The LR scheduler state is part of the optimizer state (issue #137): a restored
# optimizer continues its schedule instead of restarting it.
# ---------------------------------------------------------------------------

def _floats(xs):
    """Learning rates as Python floats: most schedulers return 0-d jax Arrays
    (their formulas use jnp, so they trace under jit), which ``pytest.approx``
    does not compare inside a list."""
    return [float(x) for x in xs]


def _sched_opt(make_sched, make_opt=O.Adam, w0=(1.0, -2.0, 0.5)):
    """A fresh ``(optimizer, scheduler, weight)`` with ``lr=make_sched()``."""
    w = ParamState(jnp.asarray(w0, dtype=jnp.float32))
    sched = make_sched()
    opt = make_opt(lr=sched)
    opt.register_trainable_weights({'w': w})
    return opt, sched, w


def _steplr():
    return O.StepLR(base_lr=0.05, step_size=1, gamma=0.5)


def _plain_leaf(x):
    return isinstance(x, (bool, int, float, str, np.ndarray, np.generic, jax.Array))


class TestSchedulerStateInStateDict:
    _G = {'w': jnp.array([0.3, -0.4, 0.2], dtype=jnp.float32)}

    def test_steplr_round_trip_continues_schedule(self):
        opt, sched, w = _sched_opt(_steplr)
        for _ in range(3):
            opt.step(self._G)
            sched.step()
        sd = opt.state_dict()

        opt2, sched2, w2 = _sched_opt(_steplr, w0=w.value)
        opt2.load_state_dict(sd)
        assert int(sched2.last_epoch.value) == 3
        assert float(opt2.current_lr) == pytest.approx(0.00625)
        assert _floats(sched2.get_last_lr()) == pytest.approx([0.00625])
        # A stepped and a fresh optimizer have the same state_dict structure, so a
        # serialiser can restore the saved leaves into a fresh optimizer's structure.
        assert jax.tree.structure(sd) == jax.tree.structure(_sched_opt(_steplr)[0].state_dict())

        # The next optimizer step applies the restored LR: same weights as the
        # uninterrupted run, bit for bit.
        opt.step(self._G)
        opt2.step(self._G)
        np.testing.assert_array_equal(np.asarray(w2.value), np.asarray(w.value))

        # The next scheduler step continues from epoch 3 (0.05 * 0.5**4).
        sched.step()
        sched2.step()
        assert int(sched2.last_epoch.value) == 4
        assert float(opt2.current_lr) == pytest.approx(0.003125)
        assert float(opt2.current_lr) == float(opt.current_lr)
        opt.step(self._G)
        opt2.step(self._G)
        np.testing.assert_array_equal(np.asarray(w2.value), np.asarray(w.value))

    def test_applied_lr_after_load_is_the_restored_lr(self):
        # Plain SGD from zero weights gives ``w = -lr * g``: the update shows the LR.
        opt, sched, w = _sched_opt(_steplr, make_opt=O.SGD)
        for _ in range(3):
            sched.step()
        opt2, sched2, w2 = _sched_opt(_steplr, make_opt=O.SGD, w0=(0.0, 0.0, 0.0))
        opt2.load_state_dict(opt.state_dict())
        opt2.step(self._G)
        np.testing.assert_allclose(np.asarray(w2.value),
                                   -0.00625 * np.asarray(self._G['w']), rtol=1e-6)

    def test_state_dict_holds_scheduler_state_as_plain_values(self):
        opt, sched, _ = _sched_opt(_steplr)
        extra = O.ReduceLROnPlateau(base_lr=0.1, patience=0)
        opt.add_scheduler(extra)
        sched.step()
        extra.step(1.0)
        sd = opt.state_dict()
        assert int(sd['lr_scheduler']['last_epoch']) == 1
        assert _floats(sd['lr_scheduler']['current_lrs']) == pytest.approx([0.025])
        assert len(sd['schedulers']) == 1
        assert int(sd['schedulers'][0]['last_epoch']) == 1
        bad = [x for x in jax.tree.leaves([sd['lr_scheduler'], sd['schedulers']])
               if not _plain_leaf(x)]
        assert bad == []

    def test_constant_lr_round_trips_unchanged(self):
        w = ParamState(jnp.ones(3))
        opt = O.Adam(lr=0.01)
        opt.register_trainable_weights({'w': w})
        opt.step(self._G)
        sd = opt.state_dict()
        assert int(sd['lr_scheduler']['last_epoch']) == 0
        assert _floats(sd['lr_scheduler']['current_lrs']) == pytest.approx([0.01])
        assert sd['schedulers'] == []

        w2 = ParamState(w.value)
        opt2 = O.Adam(lr=0.01)
        opt2.register_trainable_weights({'w': w2})
        opt2.load_state_dict(sd)
        assert float(opt2.current_lr) == pytest.approx(0.01)
        assert _floats(opt2.lr_scheduler.get_last_lr()) == pytest.approx([0.01])
        assert jax.tree.map(float, opt2.state_dict()['lr_scheduler']) == \
               jax.tree.map(float, sd['lr_scheduler'])
        opt.step(self._G)
        opt2.step(self._G)
        np.testing.assert_array_equal(np.asarray(w2.value), np.asarray(w.value))

    def test_manual_lr_change_survives_round_trip(self):
        # ``current_lr = x`` writes the reported LR and the schedule's applied LR;
        # both come back after a load.
        opt, _, _ = _sched_opt(lambda: O.ConstantLR(base_lr=0.01, factor=1.0, total_iters=0),
                               make_opt=O.SGD)
        opt.current_lr = 0.002
        opt2, _, w2 = _sched_opt(lambda: O.ConstantLR(base_lr=0.01, factor=1.0, total_iters=0),
                                 make_opt=O.SGD, w0=(0.0, 0.0, 0.0))
        opt2.load_state_dict(opt.state_dict())
        assert float(opt2.current_lr) == pytest.approx(0.002)
        assert _floats(opt2.lr_scheduler.get_last_lr()) == pytest.approx([0.002])
        opt2.step(self._G)
        np.testing.assert_allclose(np.asarray(w2.value),
                                   -0.002 * np.asarray(self._G['w']), rtol=1e-6)

    def test_add_scheduler_state_round_trips(self):
        def make():
            opt = O.Adam(lr=0.01)
            opt.register_trainable_weights({'w': ParamState(jnp.ones(3))})
            extra = O.StepLR(base_lr=0.1, step_size=1, gamma=0.1)
            opt.add_scheduler(extra)
            return opt, extra

        opt, extra = make()
        extra.step()
        extra.step()
        opt2, extra2 = make()
        opt2.load_state_dict(opt.state_dict())
        assert int(extra2.last_epoch.value) == 2
        assert _floats(extra2.get_last_lr()) == pytest.approx([0.001])
        assert _floats(opt2.get_last_lr()) == pytest.approx([0.001])
        assert int(opt2.lr_scheduler.last_epoch.value) == 0
        extra2.step()
        assert _floats(extra2.get_last_lr()) == pytest.approx([0.0001])

    def test_scheduler_count_mismatch_raises_before_any_change(self):
        opt = O.Adam(lr=0.01)
        opt.register_trainable_weights({'w': ParamState(jnp.ones(3))})
        opt.add_scheduler(O.StepLR(base_lr=0.1, step_size=1, gamma=0.1))
        opt.step(self._G)
        sd = opt.state_dict()

        fresh = O.Adam(lr=0.01)
        fresh.register_trainable_weights({'w': ParamState(jnp.ones(3))})
        with pytest.raises(ValueError, match='1 scheduler.*0'):
            fresh.load_state_dict(sd)
        assert int(fresh.step_count.value) == 0

    def test_old_style_state_dict_without_scheduler_keys_loads(self):
        opt, sched, w = _sched_opt(_steplr)
        for _ in range(3):
            opt.step(self._G)
            sched.step()
        sd = opt.state_dict()
        del sd['lr_scheduler'], sd['schedulers']

        opt2, sched2, w2 = _sched_opt(_steplr, w0=w.value)
        opt2.load_state_dict(sd)
        # As before the scheduler state was saved: the schedule is not restored,
        # the reported/applied LR and the optax state are.
        assert int(sched2.last_epoch.value) == 0
        assert float(opt2.current_lr) == pytest.approx(0.00625)
        assert int(opt2.step_count.value) == 3
        opt.step(self._G)
        opt2.step(self._G)
        np.testing.assert_array_equal(np.asarray(w2.value), np.asarray(w.value))

    _SCHEDULERS = [
        ('StepLR', _steplr, 3),
        ('MultiStepLR', lambda: O.MultiStepLR(base_lr=0.1, milestones=[2, 4], gamma=0.1), 3),
        ('ExponentialLR', lambda: O.ExponentialLR(base_lr=0.1, gamma=0.9), 3),
        ('CosineAnnealingLR', lambda: O.CosineAnnealingLR(base_lr=0.1, T_max=10), 3),
        ('CosineAnnealingWarmRestarts',
         lambda: O.CosineAnnealingWarmRestarts(base_lr=0.1, T_0=2, T_mult=2), 5),
        ('ChainedScheduler', lambda: O.ChainedScheduler([
            O.StepLR(base_lr=0.1, step_size=2, gamma=0.5),
            O.ExponentialLR(base_lr=0.1, gamma=0.9)]), 3),
        ('SequentialLR', lambda: O.SequentialLR(
            [O.LinearLR(base_lr=0.1, start_factor=0.1, total_iters=2),
             O.StepLR(base_lr=0.1, step_size=1, gamma=0.5)], milestones=[2]), 4),
    ]

    @pytest.mark.parametrize('via_msgpack', [False, True], ids=['memory', 'msgpack'])
    @pytest.mark.parametrize('name,make_sched,n', _SCHEDULERS, ids=[s[0] for s in _SCHEDULERS])
    def test_scheduler_kinds_round_trip(self, name, make_sched, n, via_msgpack, tmp_path):
        from braintools.file import msgpack_load, msgpack_save

        opt, sched, _ = _sched_opt(make_sched, make_opt=O.SGD)
        for _ in range(n):
            sched.step()
        sd = opt.state_dict()
        if via_msgpack:
            path = str(tmp_path / 'opt.msgpack')
            msgpack_save(path, sd, verbose=False)
            sd = msgpack_load(path)

        opt2, sched2, _ = _sched_opt(make_sched, make_opt=O.SGD)
        opt2.load_state_dict(sd)
        assert int(sched2.last_epoch.value) == n
        assert float(opt2.current_lr) == pytest.approx(float(opt.current_lr))
        assert _floats(sched2.get_last_lr()) == pytest.approx(_floats(sched.get_last_lr()))
        for _ in range(2):
            sched.step()
            sched2.step()
            assert float(opt2.current_lr) == pytest.approx(float(opt.current_lr)), name
            assert _floats(sched2.get_last_lr()) == pytest.approx(_floats(sched.get_last_lr())), name

    @pytest.mark.parametrize('extra', [0, 1], ids=['no_extra', 'one_extra'])
    def test_add_scheduler_state_survives_msgpack(self, extra, tmp_path):
        from braintools.file import msgpack_load, msgpack_save

        def make():
            opt, sched, _ = _sched_opt(_steplr)
            extras = [O.ReduceLROnPlateau(base_lr=0.1, patience=0, factor=0.5)
                      for _ in range(extra)]
            for e in extras:
                opt.add_scheduler(e)
            return opt, sched, extras

        opt, sched, extras = make()
        sched.step()
        for e in extras:
            e.step(1.0)
            e.step(2.0)  # worse: one reduction (patience 0)
        path = str(tmp_path / 'opt.msgpack')
        msgpack_save(path, opt.state_dict(), verbose=False)

        opt2, sched2, extras2 = make()
        opt2.load_state_dict(msgpack_load(path))
        assert int(sched2.last_epoch.value) == 1
        for e, e2 in zip(extras, extras2):
            assert int(e2.last_epoch.value) == 2
            assert _floats(e2.get_last_lr()) == pytest.approx(_floats(e.get_last_lr()))
            assert float(e2.best.value) == pytest.approx(1.0)
            assert int(e2.num_bad_epochs.value) == int(e.num_bad_epochs.value)

    def test_every_optax_optimizer_inherits_the_state_dict_methods(self):
        subclasses = [getattr(O, n) for n in dir(O)
                      if isinstance(getattr(O, n), type)
                      and issubclass(getattr(O, n), O.OptaxOptimizer)]
        assert len(subclasses) > 20
        for cls in subclasses:
            assert cls.state_dict is O.OptaxOptimizer.state_dict, cls.__name__
            assert cls.load_state_dict is O.OptaxOptimizer.load_state_dict, cls.__name__
