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

"""Regression tests for the 2026-06-18 ``braintools.visualize`` audit fixes.

Each test pins a specific bug that was found and fixed; see
``docs/braintools-visualize-issues-found-20260618.md``.
"""

import os
import tempfile
import unittest
import warnings
from unittest import mock

import brainstate
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

from braintools.visualize import (
    line_plot, raster_plot, animate_1D, animate_2D, remove_axis,
    firing_rate_map, neural_trajectory, correlation_matrix,
    brain_surface_3d, apply_style,
)
from braintools.visualize._plots import _resolve_dt


class TestAxesTargeting(unittest.TestCase):
    """A1/A2: labels/limits/title must land on the passed Axes."""

    def tearDown(self):
        plt.close('all')

    def test_line_plot_labels_go_to_passed_axes(self):
        fig, (a1, a2) = plt.subplots(1, 2)
        out = line_plot(np.arange(10), np.random.randn(10, 3), plot_ids=[0, 1],
                        ax=a1, xlabel='TT', ylabel='YY', title='ZZ',
                        xlim=(0, 5), ylim=(-2, 2))
        self.assertIs(out, a1)
        self.assertEqual(a1.get_xlabel(), 'TT')
        self.assertEqual(a1.get_ylabel(), 'YY')
        self.assertEqual(a1.get_title(), 'ZZ')
        self.assertEqual(a1.get_xlim(), (0.0, 5.0))
        # the sibling axes must be untouched
        self.assertEqual(a2.get_xlabel(), '')

    def test_raster_plot_labels_go_to_passed_axes(self):
        fig, (a1, a2) = plt.subplots(1, 2)
        sp = (np.random.rand(20, 5) > 0.7).astype(float)
        out = raster_plot(np.arange(20), sp, ax=a1, xlabel='TT', ylabel='NN',
                          xlim=(0, 10), ylim=(0, 5), title='RR')
        self.assertIs(out, a1)
        self.assertEqual(a1.get_xlabel(), 'TT')
        self.assertEqual(a1.get_title(), 'RR')
        self.assertEqual(a2.get_xlabel(), '')

    def test_line_plot_no_ax_uses_current_axes(self):
        fig, ax = plt.subplots()
        out = line_plot(np.arange(10), np.random.randn(10), xlabel='t')
        self.assertEqual(ax.get_xlabel(), 't')


class TestResolveDt(unittest.TestCase):
    """B1: dt resolution must not crash outside a brainstate context."""

    def test_explicit_dt(self):
        self.assertEqual(_resolve_dt(0.25), 0.25)

    def test_environ_dt(self):
        with brainstate.environ.context(dt=0.123):
            self.assertEqual(_resolve_dt(None), 0.123)

    def test_fallback_when_no_environ(self):
        with mock.patch.object(brainstate.environ, 'get_dt',
                               side_effect=KeyError('dt')):
            self.assertEqual(_resolve_dt(None), 1.0)

    def test_animate_2d_without_environ(self):
        with mock.patch.object(brainstate.environ, 'get_dt',
                               side_effect=KeyError('dt')):
            anim = animate_2D(np.random.rand(6, 9), net_size=(3, 3), show=False)
        self.assertIsNotNone(anim)
        plt.close('all')

    def test_animate_1d_without_environ(self):
        with mock.patch.object(brainstate.environ, 'get_dt',
                               side_effect=KeyError('dt')):
            fig = animate_1D(np.random.randn(6, 12), show=False)
        self.assertIsNotNone(fig)
        plt.close('all')


class TestAnimate1DStaticVars(unittest.TestCase):
    """A3: static_vars must work with auto-ylim and render without KeyError."""

    def tearDown(self):
        plt.close('all')

    def test_static_ndarray_auto_ylim(self):
        # no explicit ylim -> exercises the auto-ylim path that used to KeyError
        fig = animate_1D(np.random.randn(8, 20), static_vars=np.random.randn(20),
                         dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_static_list_of_arrays_auto_ylim(self):
        fig = animate_1D(np.random.randn(8, 20),
                         static_vars=[np.random.randn(20), np.random.randn(20)],
                         dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_static_data_dict_auto_ylim(self):
        fig = animate_1D(np.random.randn(8, 20),
                         static_vars=[{'data': np.random.randn(20), 'legend': 's'}],
                         dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_static_renders_to_gif(self):
        # rendering frames exercises the frame() closure for static vars
        gif = os.path.join(tempfile.mkdtemp(), 'static_auto.gif')
        animate_1D(np.random.randn(5, 12), static_vars=np.random.randn(12),
                   dt=0.1, save_path=gif)
        self.assertTrue(os.path.exists(gif))

    def test_static_dict_without_ys_or_data_raises(self):
        with self.assertRaises(ValueError):
            animate_1D(np.random.randn(5, 12),
                       static_vars=[{'legend': 'oops'}], dt=0.1, show=False)


class TestAnimateSaveBranches(unittest.TestCase):
    """Cover the gif/mp4/other save dispatch (ffmpeg unavailable -> mock save)."""

    def tearDown(self):
        plt.close('all')

    def test_animate_2d_gif(self):
        gif = os.path.join(tempfile.mkdtemp(), 'a.gif')
        animate_2D(np.random.rand(5, 9), net_size=(3, 3), dt=0.1, save_path=gif)
        self.assertTrue(os.path.exists(gif))

    def test_animate_2d_show_branch(self):
        # save_path is None and show=True -> plt.show() (a no-op under Agg)
        anim = animate_2D(np.random.rand(4, 9), net_size=(3, 3), dt=0.1, show=True)
        self.assertIsNotNone(anim)

    def test_animate_2d_mp4_branch(self):
        with mock.patch.object(animation.FuncAnimation, 'save') as msave:
            animate_2D(np.random.rand(5, 9), net_size=(3, 3), dt=0.1,
                       save_path='movie.mp4')
            msave.assert_called_once()

    def test_animate_2d_other_extension_branch(self):
        with mock.patch.object(animation.FuncAnimation, 'save') as msave:
            animate_2D(np.random.rand(5, 9), net_size=(3, 3), dt=0.1,
                       save_path='movie.avi')
            args, kwargs = msave.call_args
            self.assertTrue(args[0].endswith('.mp4'))

    def test_animate_1d_mp4_branch(self):
        with mock.patch.object(animation.FuncAnimation, 'save') as msave:
            animate_1D(np.random.randn(5, 12), dt=0.1, save_path='line.mp4')
            msave.assert_called_once()

    def test_animate_1d_other_extension_branch(self):
        with mock.patch.object(animation.FuncAnimation, 'save') as msave:
            animate_1D(np.random.randn(5, 12), dt=0.1, save_path='line.avi')
            args, kwargs = msave.call_args
            self.assertTrue(args[0].endswith('.mp4'))


class TestRemoveAxisBlank(unittest.TestCase):
    """B2: remove_axis() with no spine names blanks the panel."""

    def tearDown(self):
        plt.close('all')

    def test_no_args_blanks_panel(self):
        fig, ax = plt.subplots()
        remove_axis(ax)
        self.assertTrue(all(not s.get_visible() for s in ax.spines.values()))
        self.assertEqual(len(ax.get_xticks()), 0)
        self.assertEqual(len(ax.get_yticks()), 0)

    def test_named_spines_still_work(self):
        fig, ax = plt.subplots()
        remove_axis(ax, 'top', 'right')
        self.assertFalse(ax.spines['top'].get_visible())
        self.assertFalse(ax.spines['right'].get_visible())
        self.assertTrue(ax.spines['left'].get_visible())

    def test_invalid_spine_message(self):
        fig, ax = plt.subplots()
        with self.assertRaises(ValueError):
            remove_axis(ax, 'middle')


class TestFiringRateMapNonSquare(unittest.TestCase):
    """A4: rectangular grid_size must not raise IndexError."""

    def tearDown(self):
        plt.close('all')

    def test_tall_grid(self):
        rates = np.random.rand(300) * 10
        pos = np.random.rand(300, 2)
        ax = firing_rate_map(rates, positions=pos, grid_size=(10, 25))
        self.assertIsNotNone(ax)

    def test_wide_grid(self):
        rates = np.random.rand(300) * 10
        pos = np.random.rand(300, 2)
        ax = firing_rate_map(rates, positions=pos, grid_size=(25, 10))
        self.assertIsNotNone(ax)

    def test_boundary_points_included(self):
        # points exactly on the max edge must be binned, not dropped
        rates = np.array([1.0, 2.0, 3.0, 4.0])
        pos = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        ax = firing_rate_map(rates, positions=pos, grid_size=(2, 3))
        self.assertIsNotNone(ax)


class TestNeuralTrajectory3DColorbar(unittest.TestCase):
    """D3: 3D time-colored trajectory gets a colorbar like the 2D path."""

    def tearDown(self):
        plt.close('all')

    def test_3d_time_color_adds_colorbar(self):
        fig = plt.figure()
        data = np.random.randn(40, 3)
        ax = neural_trajectory(data, dims=(0, 1, 2), time_color=True)
        # the 3D axes plus a colorbar axes
        self.assertGreaterEqual(len(ax.figure.axes), 2)


class TestCorrelationSpearmanTwoFeatures(unittest.TestCase):
    """A5: spearman correlation for exactly two features must be a 2x2 matrix."""

    def tearDown(self):
        plt.close('all')

    def test_two_feature_matrix_shape(self):
        ax = correlation_matrix(np.random.randn(100, 2), method='spearman')
        img = ax.images[0].get_array()
        self.assertEqual(img.shape, (2, 2))

    def test_many_feature_spearman(self):
        ax = correlation_matrix(np.random.randn(100, 5), method='spearman')
        self.assertEqual(ax.images[0].get_array().shape, (5, 5))


class TestBrainSurfaceColormap(unittest.TestCase):
    """A6: brain_surface_3d must not use the removed plt.cm.get_cmap."""

    def tearDown(self):
        plt.close('all')

    def test_no_deprecation_warning(self):
        verts = np.random.randn(20, 3)
        faces = np.random.randint(0, 20, (10, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('error', matplotlib.MatplotlibDeprecationWarning)
            ax = brain_surface_3d(verts, faces, values=np.random.rand(20))
        self.assertIsNotNone(ax)


class TestApplyStyleContextManager(unittest.TestCase):
    """B4: apply_style works both as a plain call and as a context manager."""

    def tearDown(self):
        plt.rcdefaults()
        plt.close('all')

    def test_context_manager_restores_rcparams(self):
        plt.rcdefaults()
        base_grid = plt.rcParams['axes.grid']
        with apply_style('publication'):
            pass
        self.assertEqual(plt.rcParams['axes.grid'], base_grid)

    def test_plain_call_applies_and_persists(self):
        plt.rcdefaults()
        apply_style('dark')
        # dark_style sets a dark figure facecolor; it must persist (no restore)
        self.assertEqual(plt.rcParams['figure.facecolor'], '#2E2E2E')

    def test_invalid_style_raises(self):
        with self.assertRaises(ValueError):
            apply_style('does-not-exist')


if __name__ == '__main__':
    unittest.main()
