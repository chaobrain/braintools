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
