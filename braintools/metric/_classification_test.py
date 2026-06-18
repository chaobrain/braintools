# This file is modified from [optax/losses](https://github.com/google-deepmind/optax).
# The copyright notice is as follows:
#
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import braintools


class SoftmaxCrossEntropyTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
        self.ts = np.array([[0., 1., 0.], [1., 0., 0.]], dtype=np.float32)
        # taken expected outputs from rlax.
        self.exp = np.array([9.00013, 3.0696733], dtype=np.float32)

    def test_scalar(self):
        """Tests for a full batch."""
        np.testing.assert_allclose(
            braintools.metric.softmax_cross_entropy(self.ys[0], self.ts[0]),
            self.exp[0], atol=1e-4
        )

    def test_batched(self):
        """Tests for a full batch."""
        np.testing.assert_allclose(
            braintools.metric.softmax_cross_entropy(self.ys, self.ts),
            self.exp, atol=1e-4
        )


class SoftmaxCrossEntropyWithIntegerLabelsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
        self.ts = np.array([1, 0], dtype=np.int32)

    def test_consistent_with_softmax_cross_entropy_scalar(self):
        """Tests for a scalar."""
        exp = braintools.metric.softmax_cross_entropy(
            self.ys[0], jax.nn.one_hot(self.ts[0], 3))
        np.testing.assert_allclose(
            braintools.metric.softmax_cross_entropy_with_integer_labels(self.ys[0], self.ts[0]),
            exp, rtol=1e-6
        )

    def test_consistent_with_softmax_cross_entropy_batched(self):
        """Tests for a full batch."""
        exp = braintools.metric.softmax_cross_entropy(
            self.ys, jax.nn.one_hot(self.ts, 3))
        np.testing.assert_allclose(
            braintools.metric.softmax_cross_entropy_with_integer_labels(self.ys, self.ts),
            exp, rtol=1e-6
        )


class SigmoidCrossEntropyTest(parameterized.TestCase):

    @parameterized.parameters(
        dict(preds=np.array([-1e+09, -1e-09]),
             labels=np.array([1., 0.]),
             expected=5e+08),
        dict(preds=np.array([-1e+09, -1e-09]),
             labels=np.array([0., 1.]),
             expected=0.3465736),
        dict(preds=np.array([1e+09, 1e-09]),
             labels=np.array([1., 0.]),
             expected=0.3465736),
        dict(preds=np.array([1e+09, 1e-09]),
             labels=np.array([0., 1.]),
             expected=5e+08),
        dict(preds=np.array([-1e+09, 1e-09]),
             labels=np.array([1., 0.]),
             expected=5e+08),
        dict(preds=np.array([-1e+09, 1e-09]),
             labels=np.array([0., 1.]),
             expected=0.3465736),
        dict(preds=np.array([1e+09, -1e-09]),
             labels=np.array([1., 0.]),
             expected=0.3465736),
        dict(preds=np.array([1e+09, -1e-09]),
             labels=np.array([0., 1.]),
             expected=5e+08),
        dict(preds=np.array([0., 0.]),
             labels=np.array([1., 0.]),
             expected=0.6931472),
        dict(preds=np.array([0., 0.]),
             labels=np.array([0., 1.]),
             expected=0.6931472),
    )
    def testSigmoidCrossEntropy(self, preds, labels, expected):
        tested = jnp.mean(
            braintools.metric.sigmoid_binary_cross_entropy(preds, labels))
        np.testing.assert_allclose(tested, expected, rtol=1e-6, atol=1e-6)


class PolyLossTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.logits = np.array([0.14, 1.456, 2.356, -0.124, -2.47])
        self.labels = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

        self.batched_logits = np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]])
        self.batched_labels = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]])
        # all expected values are computed using tf version of `poly1_cross_entropy`
        # see page 10 here https://arxiv.org/pdf/2204.12511.pdf for more

    @parameterized.parameters(
        dict(eps=2, expected=4.5317),
        dict(eps=1, expected=3.7153),
        dict(eps=-1, expected=2.0827),
        dict(eps=0, expected=2.8990),
        dict(eps=-0.5, expected=2.4908),
        dict(eps=1.15, expected=3.8378),
        dict(eps=1.214, expected=3.8900),
        dict(eps=5.45, expected=7.3480),
    )
    def test_scalar(self, eps, expected):
        np.testing.assert_allclose(
            (braintools.metric.poly_loss_cross_entropy)(
                self.logits, self.labels, epsilon=eps
            ),
            expected,
            atol=1e-4,
        )

    @parameterized.parameters(
        dict(eps=2, expected=np.array([0.4823, 1.2567])),
        dict(eps=1, expected=np.array([0.3261, 1.0407])),
        dict(eps=0, expected=np.array([0.1698, 0.8247])),
        dict(eps=-0.5, expected=np.array([0.0917, 0.7168])),
        dict(eps=1.15, expected=np.array([0.3495, 1.0731])),
        dict(eps=1.214, expected=np.array([0.3595, 1.0870])),
        dict(eps=5.45, expected=np.array([1.0211, 2.0018])),
    )
    def test_batched(self, eps, expected):
        np.testing.assert_allclose(
            (braintools.metric.poly_loss_cross_entropy)(
                self.batched_logits, self.batched_labels, epsilon=eps
            ),
            expected,
            atol=1e-4,
        )

    @parameterized.parameters(
        dict(
            logits=np.array(
                [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0], [0.134, 1.234, 3.235]]
            ),
            labels=np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2], [0.34, 0.33, 0.33]]
            ),
        ),
        dict(
            logits=np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]),
            labels=np.array([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]),
        ),
        dict(
            logits=np.array(
                [[4.0, 2.0, 1.0, 0.134, 1.3515], [0.0, 5.0, 1.0, 0.5215, 5.616]]
            ),
            labels=np.array(
                [[0.5, 0.0, 0.0, 0.0, 0.5], [0.0, 0.12, 0.2, 0.56, 0.12]]
            ),
        ),
        dict(logits=np.array([1.89, 2.39]), labels=np.array([0.34, 0.66])),
        dict(logits=np.array([0.314]), labels=np.array([1.0])),
    )
    def test_equals_to_cross_entropy_when_eps0(self, logits, labels):
        np.testing.assert_allclose(
            (braintools.metric.poly_loss_cross_entropy)(
                logits, labels, epsilon=0.0),
            (braintools.metric.softmax_cross_entropy)(
                logits, labels),
            atol=1e-4,
        )


class HingeTest(parameterized.TestCase):

    def test_binary(self):
        label = jnp.array(1)
        signed_label = jnp.array(2.0 * label - 1.0)
        score = jnp.array(10.)

        def reference_impl(label, logit):
            return jax.nn.relu(1 - logit * (2.0 * label - 1.0))

        expected = reference_impl(label, score)
        result = braintools.metric.hinge_loss(score, signed_label)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_binary(self):
        labels = jnp.array([1, 0])
        signed_labels = jnp.array(2.0 * labels - 1.0)
        scores = jnp.array([10., 20.])

        def reference_impl(label, logit):
            return jax.nn.relu(1 - logit * (2.0 * label - 1.0))

        expected = jax.vmap(reference_impl)(labels, scores)
        # no need to vmap the optax loss. leading dimensions automatically handled.
        result = braintools.metric.hinge_loss(scores, signed_labels)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_multi_class(self):
        label = jnp.array(1)
        scores = jnp.array([10., 3.])

        def reference_impl(label, scores):
            one_hot_label = jax.nn.one_hot(label, scores.shape[-1])
            return jnp.max(scores + 1.0 - one_hot_label) - scores[label]

        expected = reference_impl(label, scores)
        result = braintools.metric.multiclass_hinge_loss(scores, label)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_multi_class(self):
        label = jnp.array([1, 0])
        scores = jnp.array([[10., 3.], [11., -2.]])

        def reference_impl(label, scores):
            one_hot_label = jax.nn.one_hot(label, scores.shape[-1])
            return jnp.max(scores + 1.0 - one_hot_label) - scores[label]

        expected = jax.vmap(reference_impl)(label, scores)
        # no need to vmap the optax loss. leading dimensions automatically handled.
        result = braintools.metric.multiclass_hinge_loss(scores, label)
        np.testing.assert_allclose(result, expected, atol=1e-4)


class ConvexKLDivergenceTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.log_ps = np.array([
            [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
            [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
        ])
        self.qs = np.array(
            [[0.2, 0.2, 0.2, 0.1, 0.15, 0.15], [0.05, 0.03, 0.02, 0.3, 0.5, 0.0]]
        )

        # Computed convex kullback-leibler divergence of P from Q.
        self.exp = np.array([0.88757247, 0.859308])

    def test_scalar(self):
        np.testing.assert_allclose(
            (braintools.metric.convex_kl_divergence)(
                self.log_ps[0], self.qs[0]),
            self.exp[0],
            atol=1e-4,
        )

    def test_batched(self):
        np.testing.assert_allclose(
            (braintools.metric.convex_kl_divergence)(
                self.log_ps, self.qs),
            self.exp,
            atol=1e-4,
        )


class PerceptronTest(parameterized.TestCase):

    def test_binary(self):
        label = jnp.array(1)
        signed_label = jnp.array(2.0 * label - 1.0)
        score = jnp.array(10.)

        def reference_impl(label, logit) -> float:
            return jax.nn.relu(- logit * (2.0 * label - 1.0))

        expected = reference_impl(label, score)
        result = braintools.metric.perceptron_loss(score, signed_label)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_binary(self):
        labels = jnp.array([1, 0])
        signed_labels = jnp.array(2.0 * labels - 1.0)
        scores = jnp.array([10., 20.])

        def reference_impl(label, logit) -> float:
            return jax.nn.relu(- logit * (2.0 * label - 1.0))

        expected = jax.vmap(reference_impl)(labels, scores)
        # no need to vmap the optax loss. leading dimensions automatically handled.
        result = braintools.metric.perceptron_loss(scores, signed_labels)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_multi_class(self):
        label = jnp.array(1)
        scores = jnp.array([10., 3.])

        def reference_impl(label, scores):
            return jnp.max(scores) - scores[label]

        expected = reference_impl(label, scores)
        result = braintools.metric.multiclass_perceptron_loss(scores, label)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_multi_class(self):
        label = jnp.array([1, 0])
        scores = jnp.array([[10., 3.], [11., -2.]])

        def reference_impl(label, scores):
            return jnp.max(scores) - scores[label]

        expected = jax.vmap(reference_impl)(label, scores)
        # no need to vmap the optax loss. leading dimensions automatically handled.
        result = braintools.metric.multiclass_perceptron_loss(scores, label)
        np.testing.assert_allclose(result, expected, atol=1e-4)


class KLDivergenceTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.log_ps = np.array(
            [[-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
             [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971]])
        self.qs = np.array([[0.2, 0.2, 0.2, 0.1, 0.15, 0.15],
                            [0.05, 0.03, 0.02, 0.3, 0.5, 0.]])
        # Computed kullback-leibler divergence of P from Q.
        self.exp = np.array([0.8875577, 0.7592807])

    def test_scalar(self):
        np.testing.assert_allclose(
            (braintools.metric.kl_divergence)(self.log_ps[0], self.qs[0]),
            self.exp[0],
            atol=1e-4)

    def test_batched(self):
        np.testing.assert_allclose(
            (braintools.metric.kl_divergence)(self.log_ps, self.qs),
            self.exp,
            atol=1e-4)


class KLDivergenceWithLogTargetsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.log_ps = np.array(
            [[-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
             [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971]])
        self.qs = np.array([[-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
                            [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026]])
        # Computed kullback-leibler divergence of P from Q.
        self.exp = np.array([0.8875625, 0.7187435584901326])

    def test_scalar(self):
        np.testing.assert_allclose(
            (braintools.metric.kl_divergence_with_log_targets)(
                self.log_ps[0], self.qs[0]),
            self.exp[0],
            atol=1e-4)

    def test_batched(self):
        np.testing.assert_allclose(
            (braintools.metric.kl_divergence_with_log_targets)(
                self.log_ps, self.qs),
            self.exp,
            atol=1e-4)


def _lengths_to_paddings(lengths, maxlength: int):
    indices = jnp.arange(maxlength).reshape((1,) * lengths.ndim + (maxlength,))
    lengths = jnp.expand_dims(lengths, axis=-1)
    elem_valid = indices < lengths
    return np.logical_not(elem_valid).astype(np.float32)


def _average_ctc_loss(
    logprobs,
    logprob_paddings,
    labels,
    label_paddings
):
    return jnp.average(braintools.metric.ctc_loss(logprobs, logprob_paddings, labels, label_paddings))


class CTCTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(1234)
        self._rtol = 5e-3 if jax.default_backend() != 'cpu' else 1e-6

    def test_with_one_to_one_alignment(self):
        # when inputsteps and outputsteps are equal, no blank will be allowed.
        batchsize = 8
        steps = 50
        nclasses = 40
        logits = np.random.randn(batchsize, steps, nclasses)
        labels = np.random.uniform(
            1, nclasses, size=(batchsize, steps)).astype(np.int32)

        # This function only covers the cases without same-label repetition.
        # `test_repeat_with_one_to_one_alignment` below complements those cases.
        # So, redraw the samples for satisfying the non-repetition constraint.
        for n in range(labels.shape[0]):
            for t in range(1, labels.shape[1]):
                while labels[n, t] == labels[n, t - 1]:
                    labels[n, t] = np.random.uniform(1, nclasses)

        results = (braintools.metric.ctc_loss_with_forward_probs)(
            logits, np.zeros(logits.shape[:2]),
            labels, np.zeros(labels.shape))
        (per_seq_loss, logalpha_blank, logalpha_emit) = results

        logprobs = jax.nn.log_softmax(logits)
        for b in range(batchsize):
            p = 0.0
            for t in range(steps):
                p += logprobs[b, t, labels[b, t]]
            np.testing.assert_allclose(
                np.array(-p), per_seq_loss[b], rtol=self._rtol)

            # Check forward-probabilities.
            # 1. All-phi path: logalpha_blank[-1, b, 0] must be a probability of
            #   the path that outputs blank symbols for all the frames.
            np.testing.assert_allclose(logalpha_blank[-1, b, 0],
                                       np.sum(logprobs[b, :, 0]),
                                       rtol=self._rtol)

            # 2. After emitting all the labels
            #   the negated loss must be identical with the forward probability of
            #   paths after consuming all the labels (because one-to-one alignment
            #   doesn't allow extra blank symbols)
            np.testing.assert_allclose(logalpha_emit[-1, b, steps - 1],
                                       -per_seq_loss[b],
                                       rtol=self._rtol)
            #   and, this forward probability must be copied to the blank forward
            #   probability of the next step.
            np.testing.assert_allclose(logalpha_blank[-1, b, steps],
                                       -per_seq_loss[b],
                                       rtol=self._rtol)

    def test_with_one_to_one_alignment_and_paddings(self):
        batch_size = 5
        nclasses = 13
        steps = 7
        logits = np.random.normal(size=[batch_size, steps, nclasses])
        logprobs = jax.nn.log_softmax(logits)

        labels = []
        for n in range(batch_size):
            row = list(range(1, nclasses))
            np.random.shuffle(row)
            labels.append(row[:steps])
        labels = np.array(labels)

        lengths = np.random.randint(3, 6, size=(batch_size,))
        paddings = _lengths_to_paddings(lengths, steps)

        actual_loss = (braintools.metric.ctc_loss)(
            logits, paddings, labels, paddings)

        value_and_grad = (jax.value_and_grad(_average_ctc_loss))
        unused_avg_loss, actual_gradients = value_and_grad(
            logits, paddings, labels, paddings)

        for n in range(batch_size):
            expected_loss = -sum(logprobs[n, t, k]
                                 for t, k in enumerate(labels[n, :lengths[n]]))
            np.testing.assert_allclose(expected_loss, actual_loss[n], rtol=self._rtol)

            expected_gradients = np.array(jax.nn.softmax(logits[n]))
            expected_gradients[lengths[n]:] = 0.0
            for t, k in enumerate(labels[n, :lengths[n]]):
                expected_gradients[t, k] -= 1.0
            expected_gradients /= batch_size
            np.testing.assert_allclose(
                expected_gradients, actual_gradients[n], rtol=self._rtol)

    def test_repeat_with_one_to_one_alignment(self):
        # test if it can correctly handle the same-label repetition.
        nclasses = 5
        labels = np.array([
            [1, 2, 2, 3],
            [2, 3, 4, 4],
            [1, 1, 1, 1],
            [1, 1, 2, 3],
            [1, 1, 1, 2],
        ])
        expected_alignment = [  # expected minimal alignment
            [1, 2, 0, 2, 3],
            [2, 3, 4, 0, 4],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 2, 3],
            [1, 0, 1, 0, 1, 2],
        ]
        batch_size = len(labels)
        label_lens = np.array([4] * batch_size)
        label_steps = 6
        # Designed to have two padding elements on the right.
        labels = np.pad(labels, [(0, 0), (0, label_steps - labels.shape[1])])
        label_paddings = _lengths_to_paddings(label_lens, label_steps)

        logit_lengths = np.array([len(seq) for seq in expected_alignment])
        logit_steps = max(logit_lengths)
        logits = np.random.randn(batch_size, logit_steps, nclasses)
        logit_paddings = _lengths_to_paddings(logit_lengths, logit_steps)

        per_seq_loss = (braintools.metric.ctc_loss)(
            logits, logit_paddings, labels, label_paddings)

        logprobs = jax.nn.log_softmax(logits)
        for n in range(batch_size):
            expected_loss = -sum(logprobs[n, t, k]
                                 for t, k in enumerate(expected_alignment[n]))
            np.testing.assert_allclose(
                jnp.array(expected_loss), per_seq_loss[n], rtol=self._rtol)


class SigmoidFocalLossTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ys = np.array([[2.0, 0.1, -2.0], [0.3, -0.1, 1.2]], dtype=np.float32)
        self.ts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        self._rtol = 5e-3 if jax.default_backend() != 'cpu' else 1e-6

        logit = lambda x: jnp.log(x / (1.0 - x))
        self.large_ys = logit(jnp.array([0.9, 0.98, 0.3, 0.99]))
        self.small_ys = logit(jnp.array([0.1, 0.02, 0.09, 0.15]))
        self.ones_ts = jnp.array([1.0, 1.0, 1.0, 1.0])

    def test_focal_equals_ce(self):
        """If gamma == 0 and alpha == 0 we expect a CE loss."""
        np.testing.assert_allclose(
            (braintools.metric.sigmoid_focal_loss)(
                self.ys, self.ts, gamma=0.0
            ),
            braintools.metric.sigmoid_binary_cross_entropy(self.ys, self.ts),
            rtol=self._rtol,
        )

    def test_scale(self):
        """This test should catch problems with p_t."""
        gamma = 2
        focal_loss = (braintools.metric.sigmoid_focal_loss)(
            self.ys, self.ts, gamma=gamma
        )
        p = jax.nn.sigmoid(self.ys)
        ce_loss = braintools.metric.sigmoid_binary_cross_entropy(self.ys, self.ts)
        p_t = p * self.ts + (1 - p) * (1 - self.ts)
        scale = (1 - p_t) ** gamma
        focal_scale = focal_loss / ce_loss
        np.testing.assert_allclose(focal_scale, scale, rtol=self._rtol)

    def test_large_logit_fl_less_than_ce(self):
        """If gamma == 2 and alpha == 0.5, the impact of large logits is reduced."""
        focal_loss = (braintools.metric.sigmoid_focal_loss)(
            self.large_ys, self.ones_ts, gamma=2, alpha=0.5
        )
        ce_loss = braintools.metric.sigmoid_binary_cross_entropy(
            self.large_ys, self.ones_ts
        )
        loss_ratio = ce_loss / focal_loss
        expected_ratio = 2.0 / ((1.0 - jax.nn.sigmoid(self.large_ys)) ** 2)
        np.testing.assert_allclose(loss_ratio, expected_ratio, rtol=self._rtol)

    def test_small_logit_fl_less_than_ce(self):
        """If gamma == 2, small logits retain their weight."""
        focal_loss = (braintools.metric.sigmoid_focal_loss)(
            self.small_ys, self.ones_ts, gamma=2
        )
        ce_loss = braintools.metric.sigmoid_binary_cross_entropy(
            self.small_ys, self.ones_ts
        )
        loss_ratio = ce_loss / focal_loss
        expected_ratio = 1.0 / ((1.0 - jax.nn.sigmoid(self.small_ys)) ** 2)
        np.testing.assert_allclose(loss_ratio, expected_ratio, rtol=self._rtol)

    def test_alpha_one(self):
        """Test if re-weighting with alpha=1 is ok."""
        np.testing.assert_allclose(
            (braintools.metric.sigmoid_focal_loss)(
                self.ys, self.ts, gamma=0.0, alpha=1
            ),
            braintools.metric.sigmoid_binary_cross_entropy(self.ys, self.ts)
            * self.ts,
            rtol=self._rtol,
        )

    def test_ignore_positive(self):
        """If alpha == 0 positive examples do not matter."""
        focal_loss = (braintools.metric.sigmoid_focal_loss)(
            self.ys, self.ts, alpha=0
        )
        ce_loss = braintools.metric.sigmoid_binary_cross_entropy(self.ys, self.ts)
        assert all(ce_loss[self.ts == 1] > 0)
        assert all(focal_loss[self.ts == 1] == 0)

    def test_ignore_negative(self):
        """If alpha == 1 negative examples do not matter."""
        focal_loss = (braintools.metric.sigmoid_focal_loss)(
            self.ys, self.ts, alpha=1
        )
        ce_loss = braintools.metric.sigmoid_binary_cross_entropy(self.ys, self.ts)
        assert all(ce_loss[self.ts == 0] > 0)
        assert all(focal_loss[self.ts == 0] == 0)

    def test_alpha_none_equals_unweighted(self):
        """alpha=None must take the unweighted (no class re-weighting) path."""
        # alpha=None should be identical to passing alpha that disables weighting.
        loss_none = braintools.metric.sigmoid_focal_loss(
            self.ys, self.ts, alpha=None, gamma=2.0
        )
        p = jax.nn.sigmoid(self.ys)
        ce_loss = braintools.metric.sigmoid_binary_cross_entropy(self.ys, self.ts)
        p_t = p * self.ts + (1 - p) * (1 - self.ts)
        expected = ce_loss * ((1 - p_t) ** 2.0)
        np.testing.assert_allclose(loss_none, expected, rtol=self._rtol)

    def test_alpha_negative_path(self):
        """A negative alpha (< 0) disables weighting just like alpha=None."""
        loss_neg = braintools.metric.sigmoid_focal_loss(
            self.ys, self.ts, alpha=-1.0, gamma=2.0
        )
        loss_none = braintools.metric.sigmoid_focal_loss(
            self.ys, self.ts, alpha=None, gamma=2.0
        )
        np.testing.assert_allclose(loss_neg, loss_none, rtol=self._rtol)

    def test_docstring_example_values(self):
        """Pin the values used in the de-indented docstring example (A14)."""
        logits = jnp.array([2.0, -1.0, 0.5, -2.0])
        labels = jnp.array([1.0, 0.0, 1.0, 0.0])
        loss = braintools.metric.sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0)
        np.testing.assert_allclose(
            loss, [0.00045089, 0.01699354, 0.01689337, 0.00135267], rtol=1e-4
        )
        loss_unweighted = braintools.metric.sigmoid_focal_loss(
            logits, labels, alpha=None, gamma=2.0
        )
        np.testing.assert_allclose(
            loss_unweighted, [0.00180356, 0.02265805, 0.06757348, 0.00180356], rtol=1e-4
        )


class NLLLossTest(parameterized.TestCase):
    """Tests for ``nll_loss`` (A16/A17, TEST-A)."""

    def test_docstring_scalar_value(self):
        # A17 sign bug regression: target=1, log_probs=log([0.1,0.7,0.2])
        # NLL must be -log(0.7) = +0.35667497 (positive!).
        log_probs = jnp.log(jnp.array([0.1, 0.7, 0.2]))
        loss = braintools.metric.nll_loss(log_probs, 1)
        np.testing.assert_allclose(float(loss), 0.35667497, rtol=1e-5)
        # explicitly assert it is the *negative* log-likelihood, i.e. positive.
        self.assertGreater(float(loss), 0.0)

    def test_nll_is_nonnegative(self):
        # log-probabilities are <= 0, so the negative log-likelihood is >= 0.
        log_probs = jax.nn.log_softmax(
            jnp.array([[2.0, 1.0, 0.1], [0.3, -1.2, 4.0]]), axis=-1
        )
        loss = braintools.metric.nll_loss(log_probs, jnp.array([0, 2]))
        self.assertTrue(bool(jnp.all(loss >= 0.0)))

    def test_batch_values(self):
        log_probs = jnp.log(jnp.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]]))
        targets = jnp.array([1, 2])
        loss = braintools.metric.nll_loss(log_probs, targets)
        np.testing.assert_allclose(
            loss, [-np.log(0.7), -np.log(0.4)], rtol=1e-5
        )

    def test_equivalence_with_integer_label_cross_entropy(self):
        # nll_loss(log_softmax(logits), labels) == softmax_cross_entropy_with_integer_labels
        logits = jnp.array([[10., 1., -2.], [1., 4., 0.2]])
        labels = jnp.array([1, 0])
        nll = braintools.metric.nll_loss(jax.nn.log_softmax(logits, axis=-1), labels)
        ce = braintools.metric.softmax_cross_entropy_with_integer_labels(logits, labels)
        np.testing.assert_allclose(nll, ce, rtol=1e-5)

    def test_nd_targets(self):
        # PyTorch convention: log_probs (N, C, d1), target (N, d1); classes on axis=1.
        log_probs = jax.nn.log_softmax(
            jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4)), axis=1
        )
        target = jnp.array([[0, 1, 2, 0], [2, 2, 1, 0]])
        loss = braintools.metric.nll_loss(log_probs, target)
        self.assertEqual(loss.shape, (2, 4))
        # spot check against an explicit gather
        for n in range(2):
            for d in range(4):
                np.testing.assert_allclose(
                    float(loss[n, d]),
                    float(-log_probs[n, int(target[n, d]), d]),
                    rtol=1e-5,
                )

    def test_invalid_shape_raises(self):
        # scalar target requires 1-D log_probs
        with self.assertRaises(ValueError):
            braintools.metric.nll_loss(jnp.zeros((2, 3)), jnp.asarray(1))
        # 1-D target requires 2-D log_probs
        with self.assertRaises(ValueError):
            braintools.metric.nll_loss(jnp.zeros((3,)), jnp.array([0, 1]))
        # N-D target requires log_probs with exactly one extra (class) axis
        with self.assertRaises(ValueError):
            braintools.metric.nll_loss(jnp.zeros((2, 3)), jnp.zeros((2, 4), dtype=jnp.int32))

    def test_jit_smoke(self):
        log_probs = jax.nn.log_softmax(
            jnp.array([[2.0, 1.0, 0.1], [0.3, -1.2, 4.0]]), axis=-1
        )
        targets = jnp.array([0, 2])
        jitted = jax.jit(braintools.metric.nll_loss)
        np.testing.assert_allclose(
            jitted(log_probs, targets),
            braintools.metric.nll_loss(log_probs, targets),
            rtol=1e-6,
        )


class KLDivergenceGradientTest(parameterized.TestCase):
    """Gradient (NaN-safety) tests for the KL variants at exact zeros (A8, TEST-KL)."""

    def test_kl_divergence_grad_at_zero_targets(self):
        log_preds = jnp.log(jnp.array([0.6, 0.3, 0.1]))
        targets = jnp.array([0.7, 0.3, 0.0])  # contains an exact zero
        grad = jax.grad(
            lambda t: braintools.metric.kl_divergence(log_preds, t).sum()
        )(targets)
        # The single-`where` form would leak a NaN here; the double-`where` keeps
        # the gradient finite everywhere, including at the exact zero target.
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        # With log(t) frozen to 0 on the zero branch, the local gradient at the
        # zero target reduces to d/dt [t*(0 - log p)] = -log p (finite).
        np.testing.assert_allclose(float(grad[2]), float(-log_preds[2]), rtol=1e-5)

    def test_kl_divergence_grad_wrt_log_predictions(self):
        log_preds = jnp.log(jnp.array([0.6, 0.3, 0.1]))
        targets = jnp.array([0.7, 0.3, 0.0])
        grad = jax.grad(
            lambda lp: braintools.metric.kl_divergence(lp, targets).sum()
        )(log_preds)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        # d/d log_pred [t*(log t - log_pred)] = -t
        np.testing.assert_allclose(grad, -targets, rtol=1e-5)

    def test_kl_divergence_with_log_targets_grad_at_neg_inf(self):
        log_preds = jnp.log(jnp.array([0.6, 0.3, 0.1]))
        log_targets = jnp.log(jnp.array([0.7, 0.3, 0.0]))  # last entry is -inf
        grad = jax.grad(
            lambda lp: braintools.metric.kl_divergence_with_log_targets(lp, log_targets).sum()
        )(log_preds)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        # d/d log_pred [exp(log_t)*(log_t - log_pred)] = -exp(log_t) = -target
        np.testing.assert_allclose(grad, -jnp.exp(log_targets), rtol=1e-5)

    def test_convex_kl_divergence_grad_at_zero_targets(self):
        log_preds = jnp.log(jnp.array([0.6, 0.3, 0.1]))
        targets = jnp.array([0.7, 0.3, 0.0])
        grad = jax.grad(
            lambda t: braintools.metric.convex_kl_divergence(log_preds, t).sum()
        )(targets)
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_kl_divergence_value_hand_computed(self):
        # KL(P||Q) = sum_i P_i (log P_i - log Q_i), with the 0*log0 term = 0.
        log_preds = jnp.log(jnp.array([0.6, 0.3, 0.1]))
        targets = jnp.array([0.7, 0.3, 0.0])
        expected = (
            0.7 * (np.log(0.7) - np.log(0.6))
            + 0.3 * (np.log(0.3) - np.log(0.3))
            + 0.0  # zero target contributes nothing
        )
        np.testing.assert_allclose(
            float(braintools.metric.kl_divergence(log_preds, targets)),
            expected,
            rtol=1e-5,
        )


class HingeLossDocstringTest(parameterized.TestCase):
    """Pin the corrected hinge_loss docstring output (A19)."""

    def test_expected_output(self):
        predictions = jnp.array([1.0, -0.5, 2.0])
        targets = jnp.array([1, -1, 1])
        loss = braintools.metric.hinge_loss(predictions, targets)
        # 1 - (-0.5)*(-1) = 0.5 for the middle element (not 1.5).
        np.testing.assert_allclose(loss, [0.0, 0.5, 0.0], atol=1e-6)


class AssertDtypeHelpersTest(parameterized.TestCase):
    """assert_is_float / assert_is_int raise explicit errors, not bare assert (A6)."""

    def test_assert_is_float_raises_typeerror_on_int(self):
        from braintools.metric._classification import assert_is_float
        with self.assertRaises(TypeError):
            assert_is_float(jnp.array([1, 2, 3], dtype=jnp.int32))

    def test_assert_is_float_passes_on_float(self):
        from braintools.metric._classification import assert_is_float
        # should not raise
        assert_is_float(jnp.array([1.0, 2.0], dtype=jnp.float32))

    def test_assert_is_int_raises_typeerror_on_float(self):
        from braintools.metric._classification import assert_is_int
        with self.assertRaises(TypeError):
            assert_is_int(jnp.array([1.0, 2.0], dtype=jnp.float32))

    def test_softmax_ce_with_integer_labels_does_not_mutate_input(self):
        # A3: ensure the function does not mutate a NumPy caller array in place.
        logits = np.array([[2.0, 1.0, 0.1], [1.0, 4.0, 0.2]], dtype=np.float32)
        original = logits.copy()
        labels = np.array([0, 1], dtype=np.int32)
        braintools.metric.softmax_cross_entropy_with_integer_labels(logits, labels)
        np.testing.assert_array_equal(logits, original)
