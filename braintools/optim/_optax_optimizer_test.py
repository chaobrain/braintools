# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import unittest

import brainstate
import jax
import jax.numpy as jnp
import optax

import braintools


class SimpleModel(brainstate.nn.Module):
    """Simple model for testing optimizers."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.linear1 = brainstate.nn.Linear(input_dim, hidden_dim)
        self.linear2 = brainstate.nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        x = self.linear1(x)
        x = jnp.tanh(x)
        return self.linear2(x)


class TestOptaxOptimizer(unittest.TestCase):
    """Test base OptaxOptimizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (32, 10))
        self.target_data = jax.random.normal(jax.random.PRNGKey(1), (32, 5))

    @brainstate.compile.jit(static_argnums=0)
    def _compute_loss_and_grads(self):
        """Helper to compute loss and gradients."""

        def loss_fn(params):
            # Temporarily set model parameters
            for k, v in params.items():
                self.model.states()[k].value = v

            predictions = self.model(self.input_data)
            loss = jnp.mean((predictions - self.target_data) ** 2)
            return loss

        params = {k: v.value for k, v in self.model.states(brainstate.ParamState).items()}
        loss = loss_fn(params)
        grads = brainstate.transform.grad(loss_fn)(params)
        return loss, grads

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = braintools.optim.OptaxOptimizer(lr=0.01)
        self.assertEqual(optimizer._base_lr, 0.01)
        self.assertEqual(optimizer._current_lr, 0.01)
        self.assertEqual(optimizer.step_count.value, 0)
        self.assertIsNone(optimizer.opt_state)

    def test_custom_tx(self):
        """Test initialization with custom optax transformation."""
        tx = optax.adam(0.001)
        optimizer = braintools.optim.OptaxOptimizer(tx=tx)
        self.assertIsNotNone(optimizer.tx)

    def test_register_trainable_weights(self):
        """Test registering trainable weights."""
        optimizer = braintools.optim.Adam(lr=0.01)
        param_states = self.model.states(brainstate.ParamState)
        optimizer.register_trainable_weights(param_states)

        self.assertIsNotNone(optimizer.opt_state)
        self.assertEqual(len(optimizer.param_states), len(param_states))
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_step_updates_parameters(self):
        """Test that step() updates parameters."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.model.states(brainstate.ParamState))

        # Get initial parameters
        initial_params = {k: v.value.copy() for k, v in self.model.states(brainstate.ParamState).items()}

        # Compute gradients and update
        _, grads = self._compute_loss_and_grads()
        optimizer.step(grads)

        def check(a, b):
            self.assertTrue(not jnp.allclose(a, b))

        # Check parameters were updated
        for k, v in self.model.states(brainstate.ParamState).items():
            jax.tree.map(check, initial_params[k], v.value)

        self.assertEqual(optimizer.step_count.value, 1)

    def test_lr_property(self):
        """Test learning rate property getter and setter."""
        optimizer = braintools.optim.Adam(lr=0.01)
        self.assertEqual(optimizer.lr, 0.01)

        optimizer.lr = 0.001
        self.assertEqual(optimizer.lr, 0.001)
        self.assertEqual(optimizer._current_lr, 0.001)

    def test_state_dict_and_load(self):
        """Test state dict saving and loading."""
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(self.model.states(brainstate.ParamState))

        # Take a few steps
        for _ in range(3):
            _, grads = self._compute_loss_and_grads()
            optimizer.step(grads)

        # Save state
        state_dict = optimizer.state_dict()
        self.assertEqual(state_dict["step_count"], 3)
        self.assertEqual(state_dict["lr"], 0.01)

        # Create new optimizer and load state
        new_optimizer = braintools.optim.Adam(lr=0.02)
        new_optimizer.register_trainable_weights(self.model.states(brainstate.ParamState))
        new_optimizer.load_state_dict(state_dict)

        self.assertEqual(new_optimizer.step_count.value, 3)
        self.assertEqual(new_optimizer.lr, 0.01)


class TestOptimizers(unittest.TestCase):
    """Test individual optimizer classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.input_data = jax.random.normal(jax.random.PRNGKey(0), (32, 10))
        self.target_data = jax.random.normal(jax.random.PRNGKey(1), (32, 5))

    def _create_simple_model(self):
        """Create a simple model for testing."""
        class SimpleParamModel(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jax.random.normal(jax.random.PRNGKey(0), (10, 5)))
                self.b = brainstate.ParamState(jnp.zeros(5))

            def __call__(self, x):
                return x @ self.w.value + self.b.value

        return SimpleParamModel()

    def _test_optimizer_basic(self, optimizer_class, **kwargs):
        """Helper to test basic optimizer functionality."""
        model = self._create_simple_model()
        optimizer = optimizer_class(**kwargs)
        param_states = model.states(brainstate.ParamState)
        optimizer.register_trainable_weights(param_states)

        # Check initialization
        self.assertIsNotNone(optimizer.opt_state)
        self.assertEqual(optimizer.step_count.value, 0)

        # Perform a few steps
        for i in range(3):
            grads = {k: jax.random.normal(jax.random.PRNGKey(i), v.value.shape) * 0.01
                     for k, v in param_states.items()}
            optimizer.step(grads)

        self.assertEqual(optimizer.step_count.value, 3)
        return optimizer

    def test_sgd(self):
        """Test SGD optimizer."""
        optimizer = self._test_optimizer_basic(braintools.optim.SGD, lr=0.1)
        self.assertEqual(optimizer.lr, 0.1)

    def test_sgd_with_momentum(self):
        """Test SGD with momentum."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.SGD,
            lr=0.1,
            momentum=0.9,
            nesterov=True
        )
        self.assertIsNotNone(optimizer.tx)

    def test_adam(self):
        """Test Adam optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Adam,
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.assertEqual(optimizer.lr, 0.01)

    def test_adamw(self):
        """Test AdamW optimizer with weight decay."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.AdamW,
            lr=0.01,
            weight_decay=0.01
        )
        self.assertIsNotNone(optimizer.tx)

    def test_adagrad(self):
        """Test Adagrad optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Adagrad,
            lr=0.1,
            eps=1e-10
        )
        self.assertIsNotNone(optimizer.tx)

    def test_adadelta(self):
        """Test Adadelta optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Adadelta,
            lr=1.0,
            rho=0.9,
            eps=1e-6
        )
        self.assertIsNotNone(optimizer.tx)

    def test_rmsprop(self):
        """Test RMSprop optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.RMSprop,
            lr=0.01,
            alpha=0.99,
            eps=1e-8
        )
        self.assertIsNotNone(optimizer.tx)

    def test_adamax(self):
        """Test Adamax optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Adamax,
            lr=0.002,
            betas=(0.9, 0.999)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_nadam(self):
        """Test Nadam optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Nadam,
            lr=0.002,
            betas=(0.9, 0.999)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_radam(self):
        """Test RAdam optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.RAdam,
            lr=0.001,
            betas=(0.9, 0.999),
            threshold=5.0
        )
        self.assertIsNotNone(optimizer.tx)

    def test_lamb(self):
        """Test LAMB optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Lamb,
            lr=0.001,
            betas=(0.9, 0.999)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_lars(self):
        """Test LARS optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Lars,
            lr=0.1,
            momentum=0.9,
            trust_coefficient=1e-3
        )
        self.assertIsNotNone(optimizer.tx)

    def test_yogi(self):
        """Test Yogi optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Yogi,
            lr=0.01,
            betas=(0.9, 0.999)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_lbfgs(self):
        """Test L-BFGS optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.LBFGS,
            lr=1.0,
            memory_size=10
        )
        self.assertIsNotNone(optimizer.tx)

    def test_rprop(self):
        """Test Rprop optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Rprop,
            lr=0.01,
            etas=(0.5, 1.2)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_adafactor(self):
        """Test Adafactor optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Adafactor,
            lr=None,
            decay_rate=0.8
        )
        self.assertIsNotNone(optimizer.tx)

    def test_adabelief(self):
        """Test AdaBelief optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.AdaBelief,
            lr=0.001,
            betas=(0.9, 0.999)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_lion(self):
        """Test Lion optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Lion,
            lr=0.0001,
            betas=(0.9, 0.99)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_sm3(self):
        """Test SM3 optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.SM3,
            lr=1.0,
            momentum=0.0
        )
        self.assertIsNotNone(optimizer.tx)

    def test_novograd(self):
        """Test Novograd optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Novograd,
            lr=0.001,
            betas=(0.95, 0.98)
        )
        self.assertIsNotNone(optimizer.tx)

    def test_fromage(self):
        """Test Fromage optimizer."""
        optimizer = self._test_optimizer_basic(
            braintools.optim.Fromage,
            lr=0.01,
            min_norm=1e-6
        )
        self.assertIsNotNone(optimizer.tx)

    def test_lookahead(self):
        """Test Lookahead wrapper optimizer."""
        base_opt = optax.adam(0.001)
        model = self._create_simple_model()
        optimizer = braintools.optim.Lookahead(
            base_optimizer=base_opt,
            sync_period=5,
            slow_step_size=0.5
        )
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))
        self.assertIsNotNone(optimizer.tx)


class TestOptimizerFeatures(unittest.TestCase):
    """Test optimizer features like gradient clipping, weight decay, etc."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()

    def _create_simple_model(self):
        """Create a simple model for testing."""
        class SimpleParamModel(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = brainstate.ParamState(jax.random.normal(jax.random.PRNGKey(0), (10, 5)))
                self.b = brainstate.ParamState(jnp.zeros(5))

            def __call__(self, x):
                return x @ self.w.value + self.b.value

        return SimpleParamModel()

    def test_gradient_clipping_by_norm(self):
        """Test gradient clipping by global norm."""
        model = self._create_simple_model()
        optimizer = braintools.optim.Adam(lr=0.01, grad_clip_norm=1.0)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Create large gradients that should be clipped
        large_grads = {k: jnp.ones_like(v.value) * 100
                       for k, v in model.states(brainstate.ParamState).items()}

        # Step should not raise an error
        optimizer.step(large_grads)
        self.assertEqual(optimizer.step_count.value, 1)

    def test_gradient_clipping_by_value(self):
        """Test gradient clipping by value."""
        model = self._create_simple_model()
        optimizer = braintools.optim.Adam(lr=0.01, grad_clip_value=0.5)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Create gradients with large values
        large_grads = {k: jnp.ones_like(v.value) * 10
                       for k, v in model.states(brainstate.ParamState).items()}

        optimizer.step(large_grads)
        self.assertEqual(optimizer.step_count.value, 1)

    def test_weight_decay(self):
        """Test weight decay effect."""
        # Create two identical models
        model1 = self._create_simple_model()
        model2 = self._create_simple_model()

        # Sync initial parameters
        for (k1, v1), (k2, v2) in zip(
            model1.states(brainstate.ParamState).items(),
            model2.states(brainstate.ParamState).items()
        ):
            v2.value = v1.value.copy()

        # Create optimizers with and without weight decay
        opt_no_decay = braintools.optim.Adam(lr=0.01, weight_decay=0.0)
        opt_with_decay = braintools.optim.Adam(lr=0.01, weight_decay=0.1)

        opt_no_decay.register_trainable_weights(model1.states(brainstate.ParamState))
        opt_with_decay.register_trainable_weights(model2.states(brainstate.ParamState))

        # Apply same gradients
        grads = {k: jnp.ones_like(v.value) * 0.1
                 for k, v in model1.states(brainstate.ParamState).items()}

        opt_no_decay.step(grads)
        opt_with_decay.step(grads)

        # Parameters should be different due to weight decay
        for (k1, v1), (k2, v2) in zip(
            model1.states(brainstate.ParamState).items(),
            model2.states(brainstate.ParamState).items()
        ):
            self.assertFalse(jnp.allclose(v1.value, v2.value))

    def test_param_groups(self):
        """Test parameter groups with different learning rates."""
        model = self._create_simple_model()
        optimizer = braintools.optim.Adam(lr=0.01)

        # Get parameter states
        param_states = model.states(brainstate.ParamState)

        # Register first parameter with default lr
        first_param = {k: v for i, (k, v) in enumerate(param_states.items()) if i == 0}
        optimizer.register_trainable_weights(first_param)

        # Add second parameter group with different lr
        second_param = {k: v for i, (k, v) in enumerate(param_states.items()) if i == 1}
        optimizer.add_param_group(second_param, lr=0.001, weight_decay=0.01)

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        self.assertEqual(optimizer.param_groups[1]['lr'], 0.001)
        self.assertEqual(optimizer.param_groups[1]['weight_decay'], 0.01)

    def test_zero_grad_compatibility(self):
        """Test zero_grad method (no-op in JAX)."""
        model = self._create_simple_model()
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        # Should not raise an error
        optimizer.zero_grad()

    def test_multiple_step_updates(self):
        """Test multiple parameter updates."""
        model = self._create_simple_model()
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        initial_step_count = optimizer.step_count.value
        num_updates = 5

        for i in range(num_updates):
            grads = {k: jax.random.normal(jax.random.PRNGKey(i), v.value.shape) * 0.01
                     for k, v in model.states(brainstate.ParamState).items()}
            optimizer.step(grads)

        self.assertEqual(optimizer.step_count.value, initial_step_count + num_updates)

    def test_partial_gradients(self):
        """Test optimizer behavior with partial gradients."""
        model = self._create_simple_model()
        optimizer = braintools.optim.Adam(lr=0.01)
        param_states = model.states(brainstate.ParamState)
        optimizer.register_trainable_weights(param_states)

        # Create gradients for only the first parameter
        partial_grads = {}
        for i, (k, v) in enumerate(param_states.items()):
            if i == 0:
                partial_grads[k] = jax.random.normal(jax.random.PRNGKey(0), v.value.shape) * 0.01
                break

        # Should handle partial gradients gracefully
        optimizer.step(partial_grads)
        self.assertEqual(optimizer.step_count.value, 1)

    def test_update_method_backward_compatibility(self):
        """Test that update() method works for backward compatibility."""
        model = self._create_simple_model()
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        grads = {k: jax.random.normal(jax.random.PRNGKey(0), v.value.shape) * 0.01
                 for k, v in model.states(brainstate.ParamState).items()}

        # Both update() and step() should work
        optimizer.update(grads)
        self.assertEqual(optimizer.step_count.value, 1)

        optimizer.step(grads)
        self.assertEqual(optimizer.step_count.value, 2)


class TestOptimizerErrors(unittest.TestCase):
    """Test error handling in optimizers."""

    def test_step_without_registration(self):
        """Test that step() raises error without registered weights."""
        optimizer = braintools.optim.Adam(lr=0.01)

        with self.assertRaises(ValueError) as context:
            optimizer.step({})

        self.assertIn("register_trainable_weights", str(context.exception))

    def test_invalid_gradient_transformation(self):
        """Test error with invalid gradient transformation."""
        with self.assertRaises(TypeError):
            optimizer = braintools.optim.OptaxOptimizer(tx="not_a_transformation")

    def test_register_invalid_states(self):
        """Test error when registering invalid states."""
        optimizer = braintools.optim.Adam(lr=0.01)

        # Test with non-dict
        with self.assertRaises(TypeError):
            optimizer.register_trainable_weights([1, 2, 3])

        # Test with dict containing non-State values
        with self.assertRaises(TypeError):
            optimizer.register_trainable_weights({"key": "not_a_state"})

    def test_step_with_none_grads_and_no_closure(self):
        """Test error when step() is called with None grads and no closure."""
        model = SimpleModel()
        optimizer = braintools.optim.Adam(lr=0.01)
        optimizer.register_trainable_weights(model.states(brainstate.ParamState))

        with self.assertRaises(ValueError) as context:
            optimizer.step(None)

        self.assertIn("Either grads or closure", str(context.exception))


if __name__ == "__main__":
    unittest.main()
