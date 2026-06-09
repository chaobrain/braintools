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

import unittest

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import braintools
from braintools.visualize._animation import Camera


class TestAnimator(unittest.TestCase):
    """Coverage tests for braintools.visualize._animation."""

    def setUp(self):
        np.random.seed(0)
        # Time-first spiking data: [num_steps, H, W]
        self.data = np.random.rand(5, 8, 8)

    def tearDown(self):
        plt.close('all')

    def test_animator_default(self):
        fig, ax = plt.subplots()
        anim = braintools.visualize.animator(self.data, fig, ax)
        self.assertIsInstance(anim, ArtistAnimation)

    def test_animator_num_steps_and_interval(self):
        fig, ax = plt.subplots()
        anim = braintools.visualize.animator(
            self.data, fig, ax, num_steps=3, interval=20, cmap='viridis'
        )
        self.assertIsInstance(anim, ArtistAnimation)
        # num_steps frames captured
        self.assertEqual(len(anim._framedata), 3)


class TestCamera(unittest.TestCase):
    """Direct coverage tests for the Camera helper class."""

    def tearDown(self):
        plt.close('all')

    def test_snap_and_animate(self):
        fig, ax = plt.subplots()
        camera = Camera(fig)
        for i in range(3):
            ax.plot(np.arange(5), np.arange(5) + i)
            frame = camera.snap()
            self.assertIsInstance(frame, list)
        anim = camera.animate(interval=50)
        self.assertIsInstance(anim, ArtistAnimation)
        self.assertEqual(len(camera._photos), 3)

    def test_snap_with_legend(self):
        fig, ax = plt.subplots()
        camera = Camera(fig)
        ax.plot([0, 1, 2], [0, 1, 2], label='line')
        ax.legend()
        # legend_ branch inside snap() must be exercised
        frame = camera.snap()
        self.assertIsInstance(frame, list)
        anim = camera.animate()
        self.assertIsInstance(anim, ArtistAnimation)


if __name__ == '__main__':
    unittest.main()
