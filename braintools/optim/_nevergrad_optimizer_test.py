# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import unittest
from unittest.mock import patch

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

import braintools
import braintools.optim as O


class TestNevergradOptimizer(unittest.TestCase):

    def setUp(self):
        # Define a simple quadratic loss function for testing
        def loss_fun(x):
            return jnp.sum(x ** 2)

        self.loss_fun = loss_fun

        # Define a batched version for multiple samples
        def batched_loss_fun(*params):
            return jnp.array([jnp.sum(jnp.asarray(p) ** 2) for p in zip(*params)])

        self.batched_loss_fun = batched_loss_fun

        # Simple bounds for scalar parameters
        self.scalar_bounds = [(-5.0, 5.0), (-3.0, 3.0)]

        # Bounds for array parameters
        self.array_bounds = [
            (jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0])),
            (jnp.array([-3.0, -3.0]), jnp.array([3.0, 3.0]))
        ]

        # Dictionary bounds
        self.dict_bounds = {
            'param1': (-5.0, 5.0),
            'param2': (-3.0, 3.0)
        }

        # Dictionary bounds with arrays
        self.dict_array_bounds = {
            'param1': (jnp.array([-5.0, -5.0]), jnp.array([5.0, 5.0])),
            'param2': (jnp.array([-3.0, -3.0]), jnp.array([3.0, 3.0]))
        }

    def test_initializationWorksWithScalarBounds(self):
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.scalar_bounds,
            n_sample=10,
            method='DE'
        )

        self.assertIsInstance(optimizer, braintools.optim.Optimizer)
        self.assertEqual(optimizer.n_sample, 10)
        self.assertEqual(optimizer.method, 'DE')

    def test_initializationWorksWithArrayBounds(self):
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.array_bounds,
            n_sample=10,
            method='DE'
        )

        self.assertIsInstance(optimizer, braintools.optim.Optimizer)
        self.assertEqual(optimizer.n_sample, 10)

    def test_initializationWorksWithDictBounds(self):
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.dict_bounds,
            n_sample=10,
            method='DE'
        )

        self.assertIsInstance(optimizer, braintools.optim.Optimizer)
        self.assertEqual(optimizer.n_sample, 10)

    def test_initializationWorksWithDictArrayBounds(self):
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.dict_array_bounds,
            n_sample=10,
            method='DE'
        )

        self.assertIsInstance(optimizer, braintools.optim.Optimizer)
        self.assertEqual(optimizer.n_sample, 10)

    def test_raiseErrorWithInvalidBoundsType(self):
        with self.assertRaises(ValueError):
            braintools.optim.NevergradOptimizer(
                batched_loss_fun=self.batched_loss_fun,
                bounds=123,  # Invalid bounds type
                n_sample=10
            )

    def test_raiseErrorWithInvalidBoundsDimensions(self):
        with self.assertRaises(AssertionError):
            braintools.optim.NevergradOptimizer(
                batched_loss_fun=self.batched_loss_fun,
                bounds=[(-5.0, 5.0, 0.0)],  # Should be tuple of 2
                n_sample=10
            )

    def test_raiseErrorWithMismatchedArrayBoundShapes(self):
        with self.assertRaises(AssertionError):
            braintools.optim.NevergradOptimizer(
                batched_loss_fun=self.batched_loss_fun,
                bounds=[(jnp.array([-5.0, -5.0]), jnp.array([5.0]))],  # Mismatched shapes
                n_sample=10
            )

    def test_raiseErrorWithUnitMismatch(self):
        with self.assertRaises(Exception):  # Unit mismatch error
            braintools.optim.NevergradOptimizer(
                batched_loss_fun=self.batched_loss_fun,
                bounds=[(u.Quantity(1.0, 'mV'), u.Quantity(5.0, 'nA'))],  # Different units
                n_sample=10
            )

    def test_raiseErrorWithNonCallableLossFunction(self):
        with self.assertRaises(AssertionError):
            braintools.optim.NevergradOptimizer(
                batched_loss_fun="not a function",  # Not callable
                bounds=self.scalar_bounds,
                n_sample=10
            )

    def test_raiseErrorWithNegativeSampleSize(self):
        with self.assertRaises(AssertionError):
            braintools.optim.NevergradOptimizer(
                batched_loss_fun=self.batched_loss_fun,
                bounds=self.scalar_bounds,
                n_sample=-5  # Negative sample size
            )

    def test_minimizationReturnsExpectedResult(self):
        # For a simple quadratic function, optimization should find values close to zero
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.scalar_bounds,
            n_sample=10,
            method='OnePlusOne'  # Using a simpler algorithm for testing
        )

        # Suppress output during test
        with patch('builtins.print'):
            result = optimizer.minimize(n_iter=3, verbose=True)

        # Check that results are within bounds
        for i, bound in enumerate(self.scalar_bounds):
            self.assertGreaterEqual(result[i], bound[0])
            self.assertLessEqual(result[i], bound[1])

    def test_minimizationWithDictBoundsReturnsExpectedResult(self):
        def dict_batched_loss_fun(**params):
            return jnp.array([jnp.sum(p1 ** 2 + p2 ** 2)
                              for p1, p2 in zip(params['param1'], params['param2'])])

        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=dict_batched_loss_fun,
            bounds=self.dict_bounds,
            n_sample=10,
            method='OnePlusOne'
        )

        # Suppress output during test
        with patch('builtins.print'):
            result = optimizer.minimize(n_iter=3, verbose=True)

        # Check that we got a dictionary result with the right keys
        self.assertIsInstance(result, dict)
        self.assertIn('param1', result)
        self.assertIn('param2', result)

        # Check bounds
        self.assertGreaterEqual(result['param1'], self.dict_bounds['param1'][0])
        self.assertLessEqual(result['param1'], self.dict_bounds['param1'][1])
        self.assertGreaterEqual(result['param2'], self.dict_bounds['param2'][0])
        self.assertLessEqual(result['param2'], self.dict_bounds['param2'][1])

    def test_usesNevergradRecommendationWhenSpecified(self):
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.scalar_bounds,
            n_sample=10,
            method='OnePlusOne',
            use_nevergrad_recommendation=True
        )

        # Mock the optimizer's provide_recommendation method
        with patch.object(optimizer, 'initialize'):
            with patch.object(optimizer, '_one_trial') as mock_trial:
                optimizer.minimize(n_iter=1, verbose=False)
                # Check that _one_trial was called with choice_best=True
                mock_trial.assert_called_with(choice_best=True)

    def test_correctNumberOfIterationsPerformed(self):
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.scalar_bounds,
            n_sample=10,
            method='OnePlusOne'
        )

        with patch.object(optimizer, '_one_trial') as mock_trial:
            mock_trial.return_value = [0, 0]  # Mock return value
            optimizer.minimize(n_iter=5, verbose=False)
            self.assertEqual(mock_trial.call_count, 5)

    def test_raiseErrorWithInvalidIterationCount(self):
        optimizer = braintools.optim.NevergradOptimizer(
            batched_loss_fun=self.batched_loss_fun,
            bounds=self.scalar_bounds,
            n_sample=10
        )

        with self.assertRaises(AssertionError):
            optimizer.minimize(n_iter=-1)  # Negative iteration count

        with self.assertRaises(AssertionError):
            optimizer.minimize(n_iter="not a number")  # Not an integer


# ==============================================================================
# Regression tests for the 2026-06-18 ``braintools.optim`` audit fixes.
# Each test maps to an issue ID from docs/braintools-optim-issues-found-20260618.md
# and asserts the *corrected* behavior (i.e. it fails against the pre-fix code).
# ==============================================================================


# --------------------------------------------------------------------------- #
# NevergradOptimizer fixes
# --------------------------------------------------------------------------- #

class TestNevergradOptimizerAuditFixes:
    def test_N1_all_nan_losses_do_not_crash(self):
        # N-1: an all-NaN iteration must not crash via np.nanargmin's "All-NaN slice";
        # the optimizer falls back to Nevergrad's recommendation (with a warning).
        pytest.importorskip('nevergrad')
        import warnings
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return jnp.full(x.shape, jnp.nan)

        opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                 n_sample=6, method='OnePlusOne')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = opt.minimize(n_iter=1, verbose=False)
        assert len(res) == 2
        assert np.all(np.isfinite(np.asarray(res)))

    def test_N1_all_nan_emits_warning(self):
        # N-1: the all-NaN fallback should warn rather than silently continue.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return jnp.full(x.shape, jnp.nan)

        opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                 n_sample=6, method='OnePlusOne')
        with pytest.warns(RuntimeWarning):
            opt.minimize(n_iter=1, verbose=False)

    def test_N3_seed_makes_runs_reproducible(self):
        # N-3: with a seed, the ask/tell sampling is reproducible across runs.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return x ** 2 + y ** 2

        def run():
            opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                     n_sample=8, method='RandomSearch', budget=100, seed=42)
            return opt.minimize(n_iter=2, verbose=False)

        r1 = run()
        r2 = run()
        assert np.allclose(np.asarray(r1), np.asarray(r2))

    def test_N3_different_seeds_differ(self):
        # N-3 sanity: different seeds should generally explore differently.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return x ** 2 + y ** 2

        def run(seed):
            opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                     n_sample=8, method='RandomSearch', budget=100, seed=seed)
            return opt.minimize(n_iter=2, verbose=False)

        assert not np.allclose(np.asarray(run(1)), np.asarray(run(2)))

    def test_N5_budget_exceeded_warns(self):
        # N-5: n_iter*n_sample evals are not capped by `budget`; warn on a mismatch.
        pytest.importorskip('nevergrad')
        from braintools.optim import NevergradOptimizer

        def batched_loss_fun(x, y):
            return x ** 2 + y ** 2

        opt = NevergradOptimizer(batched_loss_fun, [(-5.0, 5.0), (-3.0, 3.0)],
                                 n_sample=8, method='OnePlusOne', budget=5)
        with pytest.warns(RuntimeWarning, match='budget'):
            opt.minimize(n_iter=2, verbose=False)


class TestNevergradUnitStripping:
    """With quantity bounds the loss receives plain numbers in the lower bound's unit."""

    def test_loss_receives_plain_numbers(self):
        import brainunit as u
        seen = {}

        def loss(tau, v):
            seen['types'] = (type(tau), type(v))
            seen['range'] = (float(jnp.min(tau)), float(jnp.max(tau)))
            return (tau - 20.0) ** 2 + (v + 50.0) ** 2  # ms and mV as plain numbers

        opt = O.NevergradOptimizer(loss, bounds=[(5.0 * u.ms, 0.05 * u.second), (-60.0 * u.mV, -40.0 * u.mV)],
                                   n_sample=16, method='DE', seed=0)
        tau, v = opt.minimize(n_iter=30, verbose=False)
        assert not any(issubclass(t, u.Quantity) for t in seen['types'])
        assert 5.0 <= seen['range'][0] and seen['range'][1] <= 50.0  # 0.05 s converted to ms
        # the returned best parameters carry the bounds' units again
        assert isinstance(tau, u.Quantity) and isinstance(v, u.Quantity)
        np.testing.assert_allclose(tau.to_decimal(u.ms), 20.0, atol=1.0)
        np.testing.assert_allclose(v.to_decimal(u.mV), -50.0, atol=1.0)
