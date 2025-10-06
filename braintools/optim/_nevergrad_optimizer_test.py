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

import braintools


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
