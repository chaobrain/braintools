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


import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import braintools

# Export symbols from math for conciser test value definitions.
exp = math.exp
log = math.log
logloss = lambda x: log(1.0 + exp(-x))
sigmoid = lambda x: 1.0 / (1.0 + exp(-x))


class RankingLossesTest(parameterized.TestCase):

    @parameterized.parameters([
        {
            "loss_fn": braintools.metric.ranking_softmax_loss,
            "expected_value": -(
                log(exp(2.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
                + log(exp(1.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
            ),
        },
    ])
    def test_computes_loss_value(self, loss_fn, expected_value):
        scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
        labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])

        loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

        np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

    @parameterized.parameters([
        {
            "loss_fn": braintools.metric.ranking_softmax_loss,
            "expected_value": -(
                (-2.1e26 - (0.0 + -2.1e26 + 3.4e37 + 42.0))
                + (3.4e37 - (0.0 + -2.1e26 + 3.4e37 + 42.0))
            ),
        },
    ])
    def test_computes_loss_with_extreme_inputs(self, loss_fn, expected_value):
        scores = jnp.asarray([0.0, -2.1e26, 3.4e37, 42.0])
        labels = jnp.asarray([0.0, 1.0, 1.0, 0.0])

        loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

        np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

    @parameterized.parameters([
        {"loss_fn": braintools.metric.ranking_softmax_loss, "expected_value": 0.0},
    ])
    def test_computes_loss_for_zero_labels(self, loss_fn, expected_value):
        scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
        labels = jnp.asarray([0.0, 0.0, 0.0, 0.0])

        loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

        np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

    @parameterized.parameters([
        {
            "loss_fn": braintools.metric.ranking_softmax_loss,
            "expected_value": -(
                2.0 * log(exp(2.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
                + log(exp(1.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
            ),
        },
    ])
    def test_computes_weighted_loss_value(self, loss_fn, expected_value):
        scores = jnp.asarray([0.0, 3.0, 2.0, 1.0])
        labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])
        weights = jnp.asarray([1.0, 1.0, 2.0, 1.0])

        loss = loss_fn(scores, labels, weights=weights, reduce_fn=jnp.sum)

        np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

    @parameterized.parameters([
        {
            "loss_fn": braintools.metric.ranking_softmax_loss,
            "expected_value": [
                -(
                    log(exp(2.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
                    + log(exp(1.0) / (exp(0.0) + exp(3.0) + exp(1.0) + exp(2.0)))
                ),
                -(
                    2.0
                    * log(exp(3.0) / (exp(3.0) + exp(1.0) + exp(4.0) + exp(2.0)))
                    + log(exp(4.0) / (exp(3.0) + exp(1.0) + exp(4.0) + exp(2.0)))
                ),
            ],
        },
    ])
    def test_computes_loss_value_with_vmap(self, loss_fn, expected_value):
        scores = jnp.asarray([[0.0, 3.0, 1.0, 2.0], [3.0, 1.0, 4.0, 2.0]])
        labels = jnp.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0]])

        loss_fn = functools.partial(loss_fn, reduce_fn=jnp.sum)
        vmap_loss_fn = jax.vmap(loss_fn, in_axes=(0, 0), out_axes=0)
        loss = vmap_loss_fn(scores, labels)

        np.testing.assert_allclose(jnp.asarray(expected_value), loss, rtol=1e-3)

    @parameterized.parameters([
        {
            "loss_fn": braintools.metric.ranking_softmax_loss,
            "expected_value": [
                -log(exp(2.0) / (exp(2.0) + exp(1.0) + exp(3.0))),
                -log(exp(1.5) / (exp(1.0) + exp(0.5) + exp(1.5))),
            ],
            "normalizer": 2.0,
        },
    ])
    def test_computes_reduced_loss(self, loss_fn, expected_value, normalizer):
        scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected_value = jnp.asarray(expected_value)

        mean_loss = loss_fn(scores, labels, reduce_fn=jnp.mean)
        sum_loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

        np.testing.assert_allclose(
            mean_loss, jnp.sum(expected_value) / normalizer, rtol=1e-3
        )
        np.testing.assert_allclose(sum_loss, jnp.sum(expected_value), rtol=1e-3)

    @parameterized.parameters([
        {"loss_fn": braintools.metric.ranking_softmax_loss, "expected_shape": (2,)},
    ])
    def test_computes_unreduced_loss(self, loss_fn, expected_shape):
        scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        none_loss = loss_fn(scores, labels, reduce_fn=None)
        sum_loss = loss_fn(scores, labels, reduce_fn=jnp.sum)

        self.assertEqual(none_loss.shape, expected_shape)
        self.assertEqual(jnp.sum(none_loss), sum_loss)

    @parameterized.parameters([
        braintools.metric.ranking_softmax_loss,
    ])
    def test_computes_loss_value_with_where(self, loss_fn):
        scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
        labels = jnp.asarray([0.0, 0.0, 2.0, 1.0])
        where = jnp.asarray([True, True, True, False])
        expected_scores = jnp.asarray([0.0, 3.0, 1.0])
        expected_labels = jnp.asarray([0.0, 0.0, 2.0])

        loss = loss_fn(scores, labels, where=where)
        expected_loss = loss_fn(expected_scores, expected_labels)

        np.testing.assert_allclose(expected_loss, loss, rtol=1e-3)

    @parameterized.parameters([
        braintools.metric.ranking_softmax_loss,
    ])
    def test_computes_loss_value_with_all_masked(self, loss_fn):
        scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
        labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])
        where = jnp.asarray([False, False, False, False])

        loss = loss_fn(scores, labels, where=where)

        np.testing.assert_allclose(jnp.asarray(0.0), loss, rtol=1e-3)

    @parameterized.parameters([
        braintools.metric.ranking_softmax_loss,
    ])
    def test_computes_loss_with_arbitrary_batch_dimensions(self, loss_fn):
        scores = jnp.asarray([2.0, 3.0, 1.0])
        labels = jnp.asarray([0.0, 0.0, 1.0])
        where = jnp.asarray([False, True, True])
        original_loss = loss_fn(scores, labels, where=where)

        scores = jnp.asarray([[[[2.0, 3.0, 1.0]]]])
        labels = jnp.asarray([[[[0.0, 0.0, 1.0]]]])
        where = jnp.asarray([[[[False, True, True]]]])
        batched_loss = loss_fn(scores, labels, where=where)

        np.testing.assert_allclose(original_loss, batched_loss, rtol=1e-3)

    @parameterized.parameters([
        braintools.metric.ranking_softmax_loss,
    ])
    def test_grad_does_not_return_nan_for_zero_labels(self, loss_fn):
        scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
        labels = jnp.asarray([0.0, 0.0, 0.0, 0.0])

        grads = jax.grad(loss_fn)(scores, labels, reduce_fn=jnp.mean)

        np.testing.assert_array_equal(
            jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads))
        )

    @parameterized.parameters([
        braintools.metric.ranking_softmax_loss,
    ])
    def test_grad_does_not_return_nan_with_all_masked(self, loss_fn):
        scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
        labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])
        where = jnp.asarray([False, False, False, False])

        grads = jax.grad(loss_fn)(scores, labels, where=where, reduce_fn=jnp.mean)

        np.testing.assert_array_equal(
            jnp.isnan(grads), jnp.zeros_like(jnp.isnan(grads))
        )

    @parameterized.parameters([
        braintools.metric.ranking_softmax_loss,
    ])
    def test_ignores_lists_containing_only_invalid_items(self, loss_fn):
        scores = jnp.asarray([[0.0, 3.0, 1.0, 2.0], [3.0, 1.0, 4.0, 2.0]])
        labels = jnp.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0]])
        mask = jnp.asarray([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=jnp.bool_)

        output = loss_fn(scores, labels, where=mask)
        expected = loss_fn(scores[0, :], labels[0, :])

        np.testing.assert_allclose(output, expected, rtol=1e-3)

    def test_pinned_numeric_value(self):
        # Pin the loss on a small known input against an explicit float64
        # logsumexp reference (raw labels weighting log_softmax, no label
        # normalization).
        scores = jnp.asarray([2.0, 1.0, 3.0])
        labels = jnp.asarray([1.0, 0.0, 0.0])

        loss = braintools.metric.ranking_softmax_loss(
            scores, labels, reduction='sum'
        )

        scores_np = np.asarray(scores, dtype=np.float64)
        labels_np = np.asarray(labels, dtype=np.float64)
        log_softmax = scores_np - np.log(np.sum(np.exp(scores_np)))
        ref = -np.sum(labels_np * log_softmax)
        np.testing.assert_allclose(float(loss), float(ref), rtol=1e-5)
        # Matches the module docstring's pinned value of 1.408.
        np.testing.assert_allclose(float(loss), 1.408, atol=1e-3)

    def test_labels_are_raw_weights_not_normalized(self):
        # F9: doubling a single label should double its contribution to the
        # loss (raw multiplicative weighting), which would NOT hold if labels
        # were softmax-normalized.
        scores = jnp.asarray([2.0, 1.0, 3.0])
        labels1 = jnp.asarray([1.0, 0.0, 0.0])
        labels2 = jnp.asarray([2.0, 0.0, 0.0])

        loss1 = braintools.metric.ranking_softmax_loss(
            scores, labels1, reduction='sum'
        )
        loss2 = braintools.metric.ranking_softmax_loss(
            scores, labels2, reduction='sum'
        )
        np.testing.assert_allclose(float(loss2), 2.0 * float(loss1), rtol=1e-5)

    @parameterized.parameters(['none', 'mean', 'sum'])
    def test_reduction_string_options(self, reduction):
        scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        out = braintools.metric.ranking_softmax_loss(
            scores, labels, reduction=reduction
        )
        per_list = braintools.metric.ranking_softmax_loss(
            scores, labels, reduction='none'
        )
        if reduction == 'none':
            self.assertEqual(out.shape, (2,))
        elif reduction == 'sum':
            np.testing.assert_allclose(float(out), float(jnp.sum(per_list)), rtol=1e-5)
        elif reduction == 'mean':
            np.testing.assert_allclose(float(out), float(jnp.mean(per_list)), rtol=1e-5)

    def test_reduction_string_matches_reduce_fn(self):
        scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        for reduction, reduce_fn in [('mean', jnp.mean), ('sum', jnp.sum)]:
            a = braintools.metric.ranking_softmax_loss(
                scores, labels, reduction=reduction
            )
            b = braintools.metric.ranking_softmax_loss(
                scores, labels, reduce_fn=reduce_fn
            )
            np.testing.assert_allclose(float(a), float(b), rtol=1e-6)

    def test_invalid_reduction_raises(self):
        scores = jnp.array([2.0, 1.0, 3.0])
        labels = jnp.array([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            braintools.metric.ranking_softmax_loss(
                scores, labels, reduction='median'
            )

    def test_accepts_python_list_inputs(self):
        # F13: Python list inputs must not raise AttributeError.
        list_loss = braintools.metric.ranking_softmax_loss(
            [2.0, 1.0, 3.0], [1.0, 0.0, 0.0], reduction='sum'
        )
        array_loss = braintools.metric.ranking_softmax_loss(
            jnp.asarray([2.0, 1.0, 3.0]),
            jnp.asarray([1.0, 0.0, 0.0]),
            reduction='sum',
        )
        np.testing.assert_allclose(float(list_loss), float(array_loss), rtol=1e-6)

    def test_list_inputs_with_where_and_weights(self):
        loss = braintools.metric.ranking_softmax_loss(
            [2.0, 1.0, 3.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            where=[True, True, True, False],
            weights=[1.0, 1.0, 2.0, 1.0],
            reduction='sum',
        )
        self.assertFalse(bool(jnp.isnan(loss)))

    def test_empty_mask_row_no_nan_value(self):
        # F14: a fully-masked list must not yield NaN in the forward value,
        # even though masked logits are set to -inf (0 * -inf = NaN without the
        # explicit pre-sum guard).
        scores = jnp.asarray([[0.0, 3.0, 1.0, 2.0], [3.0, 1.0, 4.0, 2.0]])
        labels = jnp.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0]])
        where = jnp.asarray([[True, True, True, True], [False, False, False, False]])

        per_list = braintools.metric.ranking_softmax_loss(
            scores, labels, where=where, reduction='none'
        )
        self.assertFalse(bool(jnp.any(jnp.isnan(per_list))))
        # The fully-masked second list contributes exactly 0.
        np.testing.assert_allclose(float(per_list[1]), 0.0, atol=1e-6)

    def test_empty_mask_grad_no_nan(self):
        scores = jnp.asarray([0.0, 3.0, 1.0, 2.0])
        labels = jnp.asarray([0.0, 0.0, 1.0, 1.0])
        where = jnp.asarray([False, False, False, False])

        grads = jax.grad(
            lambda s, l: braintools.metric.ranking_softmax_loss(
                s, l, where=where, reduction='mean'
            )
        )(scores, labels)
        self.assertFalse(bool(jnp.any(jnp.isnan(grads))))

    def test_robust_mean_with_partial(self):
        # F10/F11: the empty-mask guard must trigger for a functools.partial of
        # jnp.mean (the old identity check `reduce_fn is jnp.mean` would fail).
        scores = jnp.asarray([[0.0, 3.0, 1.0, 2.0], [3.0, 1.0, 4.0, 2.0]])
        labels = jnp.asarray([[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 1.0, 0.0]])
        where = jnp.asarray([[True, True, True, True], [False, False, False, False]])

        partial_mean = functools.partial(jnp.mean)
        out = braintools.metric.ranking_softmax_loss(
            scores, labels, where=where, reduce_fn=partial_mean
        )
        self.assertFalse(bool(jnp.isnan(out)))
        # Only the first (valid) list contributes; mean over valid lists equals
        # that list's loss.
        expected = braintools.metric.ranking_softmax_loss(
            scores[0], labels[0], reduction='sum'
        )
        np.testing.assert_allclose(float(out), float(expected), rtol=1e-5)

    def test_reduction_takes_precedence_over_reduce_fn(self):
        scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        # reduction='sum' should override reduce_fn=jnp.mean.
        out = braintools.metric.ranking_softmax_loss(
            scores, labels, reduction='sum', reduce_fn=jnp.mean
        )
        ref = braintools.metric.ranking_softmax_loss(
            scores, labels, reduce_fn=jnp.sum
        )
        np.testing.assert_allclose(float(out), float(ref), rtol=1e-6)

    def test_jit_smoke(self):
        scores = jnp.array([[2.0, 1.0, 3.0], [1.0, 0.5, 1.5]])
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        fn = jax.jit(
            functools.partial(braintools.metric.ranking_softmax_loss, reduction='mean')
        )
        np.testing.assert_allclose(
            float(fn(scores, labels)),
            float(braintools.metric.ranking_softmax_loss(scores, labels, reduction='mean')),
            rtol=1e-6,
        )

    def test_leading_dims_unreduced_shape(self):
        # F17: unreduced loss has shape equal to ALL leading dims, not a single
        # batch dim.
        scores = jnp.zeros((2, 3, 4))
        labels = jnp.zeros((2, 3, 4)).at[..., 0].set(1.0)
        out = braintools.metric.ranking_softmax_loss(
            scores, labels, reduction='none'
        )
        self.assertEqual(out.shape, (2, 3))
