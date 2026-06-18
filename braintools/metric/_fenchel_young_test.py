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
# ========================================================================

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax.scipy.special import logsumexp

import braintools


def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
    """An argmax one-hot function for arbitrary shapes."""
    inputs_flat = jnp.reshape(inputs, (-1))
    flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
    return jnp.reshape(flat_one_hot, inputs.shape)


class FenchelYoungTest(parameterized.TestCase):

    def test_fenchel_young_reg(self):
        # Checks the behavior of the Fenchel-Young loss.
        fy_loss = braintools.metric.make_fenchel_young_loss(logsumexp)
        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 2)
        theta_true = jax.random.uniform(rngs[0], (8, 5))
        y_true = jax.vmap(jax.nn.softmax)(theta_true)
        theta_random = jax.random.uniform(rngs[1], (8, 5))
        y_random = jax.vmap(jax.nn.softmax)(theta_random)
        grad_random = jax.vmap(jax.grad(fy_loss))(theta_random, y_true)
        # Checks that the gradient of the loss takes the correct form.
        self.assertTrue(jnp.allclose(grad_random, y_random - y_true, rtol=1e-4))
        y_one_hot = jax.vmap(one_hot_argmax)(theta_true)
        int_one_hot = jnp.where(y_one_hot == 1.)[1]
        loss_one_hot = jax.vmap(fy_loss)(theta_random, y_one_hot)
        log_loss = jax.vmap(braintools.metric.softmax_cross_entropy_with_integer_labels)(theta_random, int_one_hot)
        # Checks that the FY loss associated to logsumexp is correct.
        self.assertTrue(jnp.allclose(loss_one_hot, log_loss, rtol=1e-4))
        # Checks that vmapping or not is equivalent.
        loss_one_hot_no_vmap = fy_loss(theta_random, y_one_hot)
        self.assertTrue(jnp.allclose(loss_one_hot, loss_one_hot_no_vmap, rtol=1e-4))

    def test_logsumexp_value_matches_analytic_reference(self):
        # Pin the forward value against an explicit logsumexp reference computed
        # in float64, instead of relying on any float32 overflow coincidence.
        fy_loss = braintools.metric.make_fenchel_young_loss(logsumexp)
        scores = jnp.array([[2.0, 1.0, 0.5], [1.5, 2.5, 1.0]])
        targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        loss = fy_loss(scores, targets)
        self.assertEqual(loss.shape, (2,))

        scores_np = np.asarray(scores, dtype=np.float64)
        targets_np = np.asarray(targets, dtype=np.float64)
        ref = (
            np.log(np.sum(np.exp(scores_np), axis=-1))
            - np.sum(targets_np * scores_np, axis=-1)
        )
        np.testing.assert_allclose(np.asarray(loss), ref, rtol=1e-5, atol=1e-6)

    def test_gradient_equals_softmax_minus_target(self):
        # For max_fun = logsumexp the analytic gradient w.r.t. the scores is
        # softmax(scores) - targets. Compare autodiff against this reference.
        fy_loss = braintools.metric.make_fenchel_young_loss(logsumexp)
        rng = jax.random.PRNGKey(7)
        rngs = jax.random.split(rng, 2)
        scores = jax.random.normal(rngs[0], (4, 6))
        targets = jax.vmap(jax.nn.softmax)(jax.random.normal(rngs[1], (4, 6)))

        grad = jax.grad(lambda s, t: fy_loss(s, t).sum())(scores, targets)
        expected = jax.nn.softmax(scores, axis=-1) - targets
        np.testing.assert_allclose(
            np.asarray(grad), np.asarray(expected), rtol=1e-5, atol=1e-6
        )

    def test_args_kwargs_forwarding(self):
        # F2: extra positional/keyword arguments must be forwarded to max_fun.
        # A buggy implementation (vectorize treating args as core inputs) would
        # raise or mis-broadcast here.
        def scaled_logsumexp(scores, scale, *, offset=0.0):
            return scale * logsumexp(scores) + offset

        fy_loss = braintools.metric.make_fenchel_young_loss(scaled_logsumexp)
        scores = jnp.array([[2.0, 1.0, 0.5], [1.5, 2.5, 1.0]])
        targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        scale, offset = 2.0, 0.5
        loss = fy_loss(scores, targets, scale, offset=offset)
        self.assertEqual(loss.shape, (2,))

        scores_np = np.asarray(scores, dtype=np.float64)
        targets_np = np.asarray(targets, dtype=np.float64)
        ref = (
            scale * np.log(np.sum(np.exp(scores_np), axis=-1)) + offset
            - np.sum(targets_np * scores_np, axis=-1)
        )
        np.testing.assert_allclose(np.asarray(loss), ref, rtol=1e-5, atol=1e-6)

    def test_args_forwarding_keyword_only(self):
        # Forwarding via **kwargs only.
        def custom_max(scores, *, temperature=1.0):
            return temperature * logsumexp(scores / temperature)

        fy_loss = braintools.metric.make_fenchel_young_loss(custom_max)
        scores = jnp.array([2.0, 1.0, 0.5])
        targets = jnp.array([1.0, 0.0, 0.0])

        loss = fy_loss(scores, targets, temperature=2.0)
        self.assertEqual(loss.shape, ())

        scores_np = np.asarray(scores, dtype=np.float64)
        temperature = 2.0
        ref = (
            temperature * np.log(np.sum(np.exp(scores_np / temperature)))
            - np.sum(np.asarray(targets, dtype=np.float64) * scores_np)
        )
        np.testing.assert_allclose(float(loss), float(ref), rtol=1e-5, atol=1e-6)

    def test_scalar_max_fun_signature(self):
        # F5: a max_fun returning a scalar per core call (jnp.max, NOT
        # keepdims=True which would yield shape (1,)) is consistent with the
        # "(n)->()" vectorize signature.
        fy_loss = braintools.metric.make_fenchel_young_loss(jnp.max)
        scores = jnp.array([[2.0, 1.0, 0.5], [1.5, 2.5, 1.0]])
        targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        loss = fy_loss(scores, targets)
        self.assertEqual(loss.shape, (2,))
        expected = jnp.max(scores, axis=-1) - jnp.sum(targets * scores, axis=-1)
        np.testing.assert_allclose(np.asarray(loss), np.asarray(expected), rtol=1e-5)

    def test_jit_smoke(self):
        fy_loss = braintools.metric.make_fenchel_young_loss(logsumexp)
        scores = jnp.array([[2.0, 1.0, 0.5], [1.5, 2.5, 1.0]])
        targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        jitted = jax.jit(fy_loss)
        np.testing.assert_allclose(
            np.asarray(jitted(scores, targets)),
            np.asarray(fy_loss(scores, targets)),
            rtol=1e-6,
        )
