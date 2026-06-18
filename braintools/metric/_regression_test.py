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


import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import braintools
from braintools.metric import safe_norm


class SquaredErrorTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ys = jnp.array([-2., -1., 0.5, 1.])
        self.ts = jnp.array([-1.5, 0., -1, 1.])
        # compute expected outputs in numpy.
        self.exp = (self.ts - self.ys) ** 2

    def test_scalar(self):
        np.testing.assert_allclose(braintools.metric.squared_error(self.ys[0], self.ts[0]), self.exp[0])

    def test_batched(self):
        np.testing.assert_allclose(braintools.metric.squared_error(self.ys, self.ts), self.exp)

    def test_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            _ = braintools.metric.squared_error(self.ys, jnp.expand_dims(self.ts, axis=-1))

    def test_invalid_reduction_raises(self):
        # The shared _reduce helper rejects unknown reduction strings.
        with self.assertRaises(ValueError):
            _ = braintools.metric.squared_error(self.ys, self.ts, reduction='invalid')


class L2LossTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ys = jnp.array([-2., -1., 0.5, 1.])
        self.ts = jnp.array([-1.5, 0., -1, 1.])
        # compute expected outputs in numpy.
        self.exp = 0.5 * (self.ts - self.ys) ** 2

    def test_scalar(self):
        np.testing.assert_allclose((braintools.metric.l2_loss)(self.ys[0], self.ts[0]), self.exp[0])

    def test_batched(self):
        np.testing.assert_allclose((braintools.metric.l2_loss)(self.ys, self.ts), self.exp)

    def test_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            _ = (braintools.metric.l2_loss)(self.ys, jnp.expand_dims(self.ts, axis=-1))


class HuberLossTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ys = np.array([-2.0, 0.5, 0., 0.5, 2.0, 4.0, 132.])
        self.ts = np.array([0.0, -0.5, 0., 1., 1.0, 2.0, 0.3])
        # computed expected outputs manually.
        self.exp = np.array([1.5, 0.5, 0., 0.125, 0.5, 1.5, 131.2])

    def test_scalar(self):
        np.testing.assert_allclose((braintools.metric.huber_loss)(self.ys[0], self.ts[0], delta=1.0), self.exp[0])

    def test_batched(self):
        np.testing.assert_allclose((braintools.metric.huber_loss)(self.ys, self.ts, delta=1.0), self.exp)


class LogCoshTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        # Test large values for overflow
        self.ys = jnp.array([500, -2., -1., 0.5, 1.])
        self.ts = jnp.array([-200, -1.5, 0., -1, 1.])
        # computed using tensorflow.keras.losses.log_cosh v2.4.1
        self.exp = jnp.array([699.3068, 0.12011445, 0.4337809, 0.85544014, 0.])
        self.exp_ys_only = jnp.array(
            [499.30685, 1.3250027, 0.4337809, 0.12011451, 0.43378082])

    def test_scalar(self):
        out = (braintools.metric.log_cosh)(self.ys[0], self.ts[0])
        np.testing.assert_allclose(out, self.exp[0], atol=1e-5)

    def test_batched(self):
        out = (braintools.metric.log_cosh)(self.ys, self.ts)
        np.testing.assert_allclose(out, self.exp, atol=1e-5)

    def test_scalar_predictions_only(self):
        out = (braintools.metric.log_cosh)(self.ys[0])
        np.testing.assert_allclose(out, self.exp_ys_only[0], atol=1e-5)

    def test_batched_predictions_only(self):
        out = (braintools.metric.log_cosh)(self.ys)
        np.testing.assert_allclose(out, self.exp_ys_only, atol=1e-5)


class CosineDistanceTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
        self.ts = np.array([[0., 1.2, 0.2], [1., -0.3, 0.]], dtype=np.float32)
        # distance computed expected output from `scipy 1.20`.
        self.exp = np.array([0.9358251989, 1.0464068465], dtype=np.float32)

    def test_scalar_distance(self):
        """Tests for a full batch."""
        np.testing.assert_allclose((braintools.metric.cosine_distance)(self.ys[0], self.ts[0]), self.exp[0], atol=1e-4)

    def test_scalar_similarity(self):
        """Tests for a full batch."""
        np.testing.assert_allclose((braintools.metric.cosine_similarity)(self.ys[0], self.ts[0]), 1. - self.exp[0],
                                   atol=1e-4)

    def test_batched_distance(self):
        """Tests for a full batch."""
        np.testing.assert_allclose((braintools.metric.cosine_distance)(self.ys, self.ts), self.exp, atol=1e-4)

    def test_batched_similarity(self):
        """Tests for a full batch."""
        np.testing.assert_allclose((braintools.metric.cosine_similarity)(self.ys, self.ts), 1. - self.exp, atol=1e-4)

    def test_zero_vector_value_finite(self):
        """B5: zero-vector cosine similarity must be finite (no NaN)."""
        zero_vec = jnp.array([0.0, 0.0, 0.0])
        normal_vec = jnp.array([1.0, 2.0, 3.0])
        sim = braintools.metric.cosine_similarity(zero_vec, normal_vec)
        self.assertTrue(bool(jnp.isfinite(sim)))
        np.testing.assert_allclose(sim, 0.0, atol=1e-4)

    def test_zero_vector_gradient_finite(self):
        """B5: gradient through a zero-vector must be finite (no NaN/inf)."""
        normal_vec = jnp.array([1.0, 2.0, 3.0])

        def f(z):
            return braintools.metric.cosine_similarity(z, normal_vec)

        grad = jax.grad(f)(jnp.array([0.0, 0.0, 0.0]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_default_epsilon_is_nonzero(self):
        """B5: default epsilon must enable zero-vector safety (not 0.)."""
        import inspect
        sig = inspect.signature(braintools.metric.cosine_similarity)
        self.assertGreater(sig.parameters['epsilon'].default, 0.0)


class AbsoluteErrorTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.pred = jnp.array([1.0, 2.0, 3.0])
        self.target = jnp.array([1.1, 1.9, 3.2])

    def test_none_reduction(self):
        out = braintools.metric.absolute_error(self.pred, self.target, reduction='none')
        np.testing.assert_allclose(out, jnp.abs(self.pred - self.target), atol=1e-6)

    def test_default_mean(self):
        np.testing.assert_allclose(
            braintools.metric.absolute_error(self.pred, self.target),
            jnp.mean(jnp.abs(self.pred - self.target)), atol=1e-6)

    def test_axis_reduction(self):
        pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        target = jnp.array([[1.1, 1.9], [2.8, 4.2]])
        out = braintools.metric.absolute_error(pred, target, axis=1)
        np.testing.assert_allclose(out, jnp.mean(jnp.abs(pred - target), axis=1), atol=1e-6)

    def test_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            braintools.metric.absolute_error(self.pred, jnp.expand_dims(self.target, -1))

    def test_predictions_only(self):
        out = braintools.metric.absolute_error(self.pred, reduction='none')
        np.testing.assert_allclose(out, jnp.abs(self.pred), atol=1e-6)


class L2NormTest(parameterized.TestCase):

    def test_value(self):
        np.testing.assert_allclose(
            braintools.metric.l2_norm(jnp.array([3.0, 4.0])), 5.0, atol=1e-6)

    def test_with_targets(self):
        pred = jnp.array([3.0, 4.0])
        target = jnp.array([0.0, 0.0])
        np.testing.assert_allclose(braintools.metric.l2_norm(pred, target), 5.0, atol=1e-6)

    def test_axis(self):
        X = jnp.array([[3.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(
            braintools.metric.l2_norm(X, axis=1), np.array([5.0, 10.0]), atol=1e-6)

    def test_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            braintools.metric.l2_norm(jnp.array([3.0, 4.0]), jnp.array([[0.0, 0.0]]))


class L1LossTest(parameterized.TestCase):

    def test_default_reduction_is_mean(self):
        """B9: l1_loss default reduction must be 'mean', matching its docstring."""
        import inspect
        sig = inspect.signature(braintools.metric.l1_loss)
        self.assertEqual(sig.parameters['reduction'].default, 'mean')

    def test_is_true_mae(self):
        """B10: l1_loss must compute mean absolute error, not a per-row L1 sum."""
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.5, 2.5], [2.0, 5.0]])
        # per-sample MAE: row0 mean(|.5|,|.5|)=0.5, row1 mean(|1|,|1|)=1.0; mean=0.75
        np.testing.assert_allclose(braintools.metric.l1_loss(logits, targets), 0.75, atol=1e-6)

    def test_none_reduction_per_sample(self):
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.5, 2.5], [2.0, 5.0]])
        out = braintools.metric.l1_loss(logits, targets, reduction='none')
        np.testing.assert_allclose(out, np.array([0.5, 1.0]), atol=1e-6)

    def test_equals_mean_abs_for_single_feature(self):
        """B9/B10: mean reduction over single-feature rows equals jnp.mean(|pred-target|)."""
        pred = jnp.array([1.0, 2.0, 3.0]).reshape(-1, 1)
        target = jnp.array([1.5, 2.5, 2.0]).reshape(-1, 1)
        expected = jnp.mean(jnp.abs(pred - target))
        np.testing.assert_allclose(braintools.metric.l1_loss(pred, target), expected, atol=1e-6)

    def test_sum_reduction(self):
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.5, 2.5], [2.0, 5.0]])
        # sum of per-sample MAEs: 0.5 + 1.0 = 1.5
        np.testing.assert_allclose(braintools.metric.l1_loss(logits, targets, reduction='sum'), 1.5, atol=1e-6)

    def test_l1loss_class_default_mean(self):
        from braintools.metric._regression import L1Loss
        loss = L1Loss()
        self.assertEqual(loss.reduction, 'mean')
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.5, 2.5], [2.0, 5.0]])
        np.testing.assert_allclose(loss.update(logits, targets), 0.75, atol=1e-6)


class LogCoshStabilityTest(parameterized.TestCase):

    def test_large_input_no_inf_nan(self):
        """B1: very large |x| must not produce inf/nan in the forward value."""
        x = jnp.array([500.0, -500.0, 1000.0, -1000.0])
        out = braintools.metric.log_cosh(x)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    def test_large_input_finite_gradient(self):
        """B1: gradient at large |x| must be finite (and bounded by ~1)."""
        def f(x):
            return braintools.metric.log_cosh(x, reduction='sum')

        grad = jax.grad(f)(jnp.array([500.0, -500.0, 3.0, 0.0]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertTrue(bool(jnp.all(jnp.abs(grad) <= 1.0 + 1e-5)))

    def test_matches_reference_small_inputs(self):
        """Stable form must match the analytic log(cosh(x)) for normal inputs."""
        errs = jnp.array([-2., -1., 0.5, 1., 3.])
        reference = jnp.logaddexp(errs, -errs) - jnp.log(2.0)
        np.testing.assert_allclose(braintools.metric.log_cosh(errs), reference, atol=1e-6)

    def test_reduction_variants(self):
        """B3/B4: reduction/axis variants produce correct shapes/values."""
        x = jnp.array([[0.0, 1.0], [-1.0, 2.0]])
        none_out = braintools.metric.log_cosh(x)
        self.assertEqual(none_out.shape, (2, 2))
        np.testing.assert_allclose(
            braintools.metric.log_cosh(x, reduction='mean'), jnp.mean(none_out), atol=1e-6)
        np.testing.assert_allclose(
            braintools.metric.log_cosh(x, reduction='sum'), jnp.sum(none_out), atol=1e-6)
        np.testing.assert_allclose(
            braintools.metric.log_cosh(x, axis=1, reduction='mean'),
            jnp.mean(none_out, axis=1), atol=1e-6)


class HuberReductionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.pred = jnp.array([1.0, 2.0, 5.0])
        self.target = jnp.array([1.1, 1.9, 3.0])

    def test_none_default_shape(self):
        """B3/B4: default reduction='none' preserves element-wise shape."""
        out = braintools.metric.huber_loss(self.pred, self.target)
        self.assertEqual(out.shape, (3,))

    def test_mean_and_sum(self):
        none_out = braintools.metric.huber_loss(self.pred, self.target)
        np.testing.assert_allclose(
            braintools.metric.huber_loss(self.pred, self.target, reduction='mean'),
            jnp.mean(none_out), atol=1e-6)
        np.testing.assert_allclose(
            braintools.metric.huber_loss(self.pred, self.target, reduction='sum'),
            jnp.sum(none_out), atol=1e-6)

    def test_axis_reduction(self):
        pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        target = jnp.array([[1.1, 1.9], [2.8, 4.2]])
        none_out = braintools.metric.huber_loss(pred, target)
        np.testing.assert_allclose(
            braintools.metric.huber_loss(pred, target, axis=1, reduction='mean'),
            jnp.mean(none_out, axis=1), atol=1e-6)

    def test_quantity_delta(self):
        """B2: huber_loss must work with a brainunit.Quantity delta."""
        import brainunit as u
        errs = jnp.array([0.5, 2.0]) * u.mV
        out = braintools.metric.huber_loss(errs, delta=1.0 * u.mV)
        # 0.5*0.5^2=0.125 ; 0.5*1^2 + 1*(2-1)=1.5, units mV^2
        np.testing.assert_allclose(u.get_magnitude(out), np.array([0.125, 1.5]), atol=1e-6)


class L2LossReductionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.pred = jnp.array([1.0, 2.0, 3.0])
        self.target = jnp.array([1.5, 2.5, 2.0])

    def test_none_default(self):
        """B3/B4: default reduction='none' preserves element-wise shape/values."""
        out = braintools.metric.l2_loss(self.pred, self.target)
        np.testing.assert_allclose(out, 0.5 * (self.pred - self.target) ** 2, atol=1e-6)

    def test_mean_and_sum(self):
        none_out = 0.5 * (self.pred - self.target) ** 2
        np.testing.assert_allclose(
            braintools.metric.l2_loss(self.pred, self.target, reduction='mean'),
            jnp.mean(none_out), atol=1e-6)
        np.testing.assert_allclose(
            braintools.metric.l2_loss(self.pred, self.target, reduction='sum'),
            jnp.sum(none_out), atol=1e-6)

    def test_axis_reduction(self):
        pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        target = jnp.array([[1.5, 2.5], [2.0, 5.0]])
        none_out = 0.5 * (pred - target) ** 2
        np.testing.assert_allclose(
            braintools.metric.l2_loss(pred, target, axis=1, reduction='sum'),
            jnp.sum(none_out, axis=1), atol=1e-6)


class SafeNormTest(parameterized.TestCase):

    def test_zero_vector_floored(self):
        """B7/B8: zero vector returns min_norm, not 0."""
        x = jnp.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(safe_norm(x, min_norm=1e-8), 1e-8, atol=1e-12)

    def test_nonzero_value(self):
        np.testing.assert_allclose(safe_norm(jnp.array([3.0, 4.0]), min_norm=1e-8), 5.0, atol=1e-6)

    def test_axis_none_squeeze(self):
        """B7: axis=None reduces over all axes to a scalar."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        out = safe_norm(X, min_norm=1e-8, axis=None)
        self.assertEqual(out.shape, ())
        np.testing.assert_allclose(out, jnp.linalg.norm(X), atol=1e-5)

    def test_axis_handling(self):
        X = jnp.array([[3.0, 4.0], [0.0, 0.0]])
        # row 0 -> 5.0; row 1 -> floored to min_norm
        out = safe_norm(X, min_norm=0.1, axis=1)
        np.testing.assert_allclose(out, np.array([5.0, 0.1]), atol=1e-6)

    def test_keepdims(self):
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        out = safe_norm(X, min_norm=0.1, axis=1, keepdims=True)
        self.assertEqual(out.shape, (2, 1))

    def test_zero_vector_gradient_finite(self):
        """B7/B8: gradient at a zero vector must be finite (no NaN)."""
        grad = jax.grad(lambda x: safe_norm(x, min_norm=1e-6))(jnp.array([0.0, 0.0, 0.0]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))

    def test_nonzero_gradient(self):
        grad = jax.grad(lambda x: safe_norm(x, min_norm=1e-6))(jnp.array([3.0, 4.0]))
        np.testing.assert_allclose(grad, np.array([0.6, 0.8]), atol=1e-5)

    def test_jit(self):
        f = jax.jit(lambda x: safe_norm(x, min_norm=1e-8))
        np.testing.assert_allclose(f(jnp.array([3.0, 4.0])), 5.0, atol=1e-6)
        np.testing.assert_allclose(f(jnp.array([0.0, 0.0])), 1e-8, atol=1e-12)
