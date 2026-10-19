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

"""Sample counts and window indices must not suffer floating-point off-by-one errors."""

import unittest

import brainstate
import brainunit as u
import numpy as np

import braintools.input as inp
from braintools.input._steps import ceil_steps, floor_steps


class TestStepHelpers(unittest.TestCase):
    def test_near_integer_ratios_snap(self):
        self.assertEqual(ceil_steps(0.3, 0.1), 3)  # 0.3 / 0.1 = 3.0000000000000004
        self.assertEqual(floor_steps(0.3, 0.1), 3)  # not 2
        self.assertEqual(ceil_steps(1.1, 0.1), 11)

    def test_true_fractions_round_outward(self):
        self.assertEqual(ceil_steps(0.25, 0.1), 3)
        self.assertEqual(floor_steps(0.25, 0.1), 2)

    def test_quantities(self):
        self.assertEqual(ceil_steps(0.3 * u.ms, 0.1 * u.ms), 3)
        self.assertEqual(ceil_steps(300 * u.ms, 0.1 * u.ms), 3000)


class TestGeneratorLengths(unittest.TestCase):
    def test_every_generator_has_duration_over_dt_samples(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            for k in list(range(1, 400)) + [1234, 2999, 3001]:
                d = k * 0.1 * u.ms
                lengths = {
                    'ramp': inp.ramp(0., 1., d).shape[0],
                    'Ramp': inp.Ramp(0., 1., d)().shape[0],
                    'step': inp.step([0., 1.], [0. * u.ms, 0. * u.ms], d).shape[0],
                    'sinusoidal': inp.sinusoidal(1., 10 * u.Hz, d).shape[0],
                    'square': inp.square(1., 10 * u.Hz, d).shape[0],
                    'gaussian_pulse': inp.gaussian_pulse(1., d / 2, 1 * u.ms, d).shape[0],
                    'poisson': inp.poisson(10 * u.Hz, d, seed=0).shape[0],
                    'section': inp.section([0., 1.], [d, d]).shape[0] // 2,
                }
                for name, n in lengths.items():
                    self.assertEqual(n, k, f'{name} with duration {k * 0.1:.1f} ms')

    def test_window_start_index(self):
        # t_start = 0.3 ms must start at sample 3 (int(2.9999...) used to give 2)
        with brainstate.environ.context(dt=0.1 * u.ms):
            x = np.asarray(inp.square(1., 1000 * u.Hz, 1. * u.ms, t_start=0.3 * u.ms, bias=True))
            self.assertEqual(int(np.flatnonzero(x)[0]), 3)
            x = np.asarray(inp.sinusoidal(1., 1000 * u.Hz, 1. * u.ms, t_start=0.3 * u.ms, bias=True))
            np.testing.assert_allclose(x[:4], [0., 0., 0., 1.], atol=1e-6)  # starts at phase 0
            s = np.asarray(inp.step([0., 1.], [0. * u.ms, 0.3 * u.ms], 1. * u.ms))
            self.assertEqual(int(np.flatnonzero(s)[0]), 3)

    def test_other_time_units(self):
        with brainstate.environ.context(dt=1e-4 * u.second):
            self.assertEqual(inp.ramp(0., 1., 0.3 * u.ms).shape[0], 3)
            self.assertEqual(inp.Constant([(1., 0.3 * u.ms)])().shape[0], 3)


class TestCompositeChannels(unittest.TestCase):
    def test_single_channel_broadcasts_over_channels(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            noise = inp.WienerProcess(10 * u.ms, n=3, sigma=1., seed=0)
            offset = inp.Constant([(1., 10 * u.ms)])
            out = (noise + offset)()
            self.assertEqual(out.shape, (100, 3))
            np.testing.assert_allclose(out - noise(), 1., rtol=1e-6)

    def test_multichannel_padding(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            short = inp.WienerProcess(5 * u.ms, n=2, sigma=1., seed=0)
            long = inp.WienerProcess(10 * u.ms, n=2, sigma=1., seed=1)
            out = (short + long)()
            self.assertEqual(out.shape, (100, 2))
            np.testing.assert_allclose(out[50:], long()[50:])


class TestCompositeDivision(unittest.TestCase):
    def test_zero_divisor_keeps_numerator(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            out = (inp.Constant([(3.0, 2 * u.ms)]) / inp.Constant([(0.0, 1 * u.ms), (2.0, 1 * u.ms)]))()
            np.testing.assert_allclose(out, [3.0] * 10 + [1.5] * 10)

    def test_divisor_with_units(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            ratio = (inp.Constant([(3.0 * u.nA, 1 * u.ms)]) / inp.Constant([(1.5 * u.nA, 1 * u.ms)]))()
            np.testing.assert_allclose(np.asarray(ratio), 2.0)
            g = (inp.Constant([(3.0 * u.nA, 1 * u.ms)]) / inp.Constant([(2.0 * u.mV, 1 * u.ms)]))()
            np.testing.assert_allclose(g.to_decimal(u.uS), 1.5)


if __name__ == '__main__':
    unittest.main()
