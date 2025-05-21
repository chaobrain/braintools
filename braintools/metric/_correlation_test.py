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

# -*- coding: utf-8 -*-


import unittest
from functools import partial

import brainstate as bst
import jax.numpy as jnp
from jax import jit

import braintools as bt

bst.environ.set(dt=0.1)


class TestCrossCorrelation(unittest.TestCase):
    def test_c(self):
        spikes = jnp.asarray([[1, 0, 1, 0, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]]).T
        cc1 = bt.metric.cross_correlation(spikes, 1., dt=1.)
        f_cc = jit(partial(bt.metric.cross_correlation, bin=1, dt=1.))
        cc2 = f_cc(spikes)
        print(cc1, cc2)
        self.assertTrue(cc1 == cc2)
        bst.util.clear_buffer_memory()

    def test_cc(self):
        spikes = jnp.ones((1000, 10))
        cc1 = bt.metric.cross_correlation(spikes, 1.)
        self.assertTrue(cc1 == 1.)

        spikes = jnp.zeros((1000, 10))
        cc2 = bt.metric.cross_correlation(spikes, 1.)
        self.assertTrue(cc2 == 0.)

        bst.util.clear_buffer_memory()

    def test_cc2(self):
        spikes = bst.random.randint(0, 2, (1000, 10))
        print(bt.metric.cross_correlation(spikes, 1.))
        print(bt.metric.cross_correlation(spikes, 0.5))
        bst.util.clear_buffer_memory()

    def test_cc3(self):
        spikes = bst.random.random((1000, 100)) < 0.8
        print(bt.metric.cross_correlation(spikes, 1.))
        print(bt.metric.cross_correlation(spikes, 0.5))
        bst.util.clear_buffer_memory()

    def test_cc4(self):
        spikes = bst.random.random((1000, 100)) < 0.2
        print(bt.metric.cross_correlation(spikes, 1.))
        print(bt.metric.cross_correlation(spikes, 0.5))
        bst.util.clear_buffer_memory()

    def test_cc5(self):
        spikes = bst.random.random((1000, 100)) < 0.05
        print(bt.metric.cross_correlation(spikes, 1.))
        print(bt.metric.cross_correlation(spikes, 0.5))
        bst.util.clear_buffer_memory()


class TestVoltageFluctuation(unittest.TestCase):
    def test_vf1(self):
        voltages = bst.random.normal(0, 10, size=(100, 10))
        print(bt.metric.voltage_fluctuation(voltages))

        with bst.environ.context(precision=64):
            voltages = jnp.ones((100, 10))
            r1 = bt.metric.voltage_fluctuation(voltages)

            jit_f = jit(partial(bt.metric.voltage_fluctuation))
            jit_f = jit(lambda a: bt.metric.voltage_fluctuation(a))
            r2 = jit_f(voltages)
            print(r1, r2)  # TODO: JIT results are different?
            # self.assertTrue(r1 == r2)

        bst.util.clear_buffer_memory()


class TestFunctionalConnectivity(unittest.TestCase):
    def test_cf1(self):
        act = bst.random.random((10000, 3))
        r1 = bt.metric.functional_connectivity(act)

        jit_f = jit(partial(bt.metric.functional_connectivity))
        r2 = jit_f(act)

        self.assertTrue(jnp.allclose(r1, r2))
        bst.util.clear_buffer_memory()


class TestMatrixCorrelation(unittest.TestCase):
    def test_mc(self):
        A = bst.random.random((100, 100))
        B = bst.random.random((100, 100))
        r1 = (bt.metric.matrix_correlation(A, B))

        jit_f = jit(partial(bt.metric.matrix_correlation))
        r2 = jit_f(A, B)
        self.assertTrue(jnp.allclose(r1, r2))
        bst.util.clear_buffer_memory()
