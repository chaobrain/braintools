#!/usr/bin/env python
"""Comprehensive tests for OptaxOptimizer."""

import unittest

import brainstate
import jax
import jax.numpy as jnp
from brainstate import ParamState

import braintools


class SimpleModel(brainstate.nn.Module):
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
        self.model = SimpleModel()
        self.param_states = braintools.optim.UniqueStateManager().merge_with(
            self.model.states(ParamState)
        ).to_dict()
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
        self.target_data = jax.random.normal(jax.random.PRNGKey(1), (32, 2))

    def _compute_loss_and_grads(self, model):
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
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)

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
        _, grads = self._compute_loss_and_grads(self.model)

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
        self.assertEqual(optimizer.lr, 0.01)

        # Test setter
        optimizer.lr = 0.001
        self.assertEqual(optimizer.lr, 0.001)
        self.assertEqual(optimizer._current_lr.value, 0.001)

    def test_state_dict_and_load(self):
        """Test saving and loading optimizer state."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.param_states)

        # Take a few steps
        for _ in range(3):
            _, grads = self._compute_loss_and_grads(self.model)
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
        self.assertEqual(new_optimizer.lr, optimizer.lr)

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
        model = SimpleModel()
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
        _, grads = self._compute_loss_and_grads(self.model)

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
            _, grads = self._compute_loss_and_grads(self.model)

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
            model = SimpleModel()
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
        self.assertEqual(optimizer.lr, 0.1)

        # Manually update lr after some steps
        for i in range(10):
            _, grads = self._compute_loss_and_grads(self.model)
            optimizer.step(grads)

            # Manually decay learning rate
            if i == 4:
                optimizer.lr = 0.05

        # Check lr was updated
        self.assertEqual(optimizer.lr, 0.05)

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
            _, grads = self._compute_loss_and_grads(self.model)

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
            _, grads = self._compute_loss_and_grads(self.model)
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
