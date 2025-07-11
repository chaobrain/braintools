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
# ========================================================================

import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.scipy.special import logsumexp

from braintools.metric import _classification
from braintools.metric import _fenchel_young


def one_hot_argmax(inputs: jnp.ndarray) -> jnp.ndarray:
    """An argmax one-hot function for arbitrary shapes."""
    inputs_flat = jnp.reshape(inputs, (-1))
    flat_one_hot = jax.nn.one_hot(jnp.argmax(inputs_flat), inputs_flat.shape[0])
    return jnp.reshape(flat_one_hot, inputs.shape)


class FenchelYoungTest(parameterized.TestCase):

    def test_fenchel_young_reg(self):
        # Checks the behavior of the Fenchel-Young loss.
        fy_loss = (_fenchel_young.make_fenchel_young_loss(logsumexp))
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
        log_loss = jax.vmap(_classification.softmax_cross_entropy_with_integer_labels)(theta_random, int_one_hot)
        # Checks that the FY loss associated to logsumexp is correct.
        self.assertTrue(jnp.allclose(loss_one_hot, log_loss, rtol=1e-4))
        # Checks that vmapping or not is equivalent.
        loss_one_hot_no_vmap = fy_loss(theta_random, y_one_hot)
        self.assertTrue(jnp.allclose(loss_one_hot, loss_one_hot_no_vmap, rtol=1e-4))
