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
import os
import tempfile
import unittest
from unittest import mock

import brainstate
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib import animation

import braintools
from braintools.visualize._plots import (
    _resolve_dt,
    line_plot,
    raster_plot,
    animate_1D,
    animate_2D,
    remove_axis,
)


class TestExistingPlots(unittest.TestCase):
    """Test enhanced existing plotting functions."""

    def setUp(self):
        """Set up test data."""
        self.time = np.linspace(0, 10, 100)
        self.values = np.random.randn(100, 3)
        self.spike_matrix = np.random.binomial(1, 0.05, (100, 10))

    def test_line_plot(self):
        """Test enhanced line plot."""
        ax = braintools.visualize.line_plot(
            self.time,
            self.values,
            plot_ids=[0, 1],
            colors=['red', 'blue'],
            alpha=0.8,
            legend='Signal'
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_line_plot_errors(self):
        """Test line plot error handling."""
        with self.assertRaises(ValueError):
            braintools.visualize.line_plot([], self.values)

        with self.assertRaises(ValueError):
            braintools.visualize.line_plot(self.time[:50], self.values)

    def test_raster_plot(self):
        """Test enhanced raster plot."""
        ax = braintools.visualize.raster_plot(
            self.time,
            self.spike_matrix,
            alpha=0.8,
            color='red'
        )
        self.assertIsNotNone(ax)
        plt.close()

    def test_raster_plot_errors(self):
        """Test raster plot error handling."""
        with self.assertRaises(ValueError):
            braintools.visualize.raster_plot(None, self.spike_matrix)

        with self.assertRaises(ValueError):
            braintools.visualize.raster_plot(self.time, [])


class TestLinePlotConversion(unittest.TestCase):
    """Issue 1: ``line_plot`` reshaped before converting -> crashed on lists."""

    def tearDown(self):
        plt.close('all')

    def test_nested_list_val_matrix(self):
        # Previously raised AttributeError: 'list' object has no attribute 'reshape'
        ax = line_plot(list(range(5)), [[1], [2], [3], [4], [5]], plot_ids=[0])
        self.assertEqual(len(ax.get_lines()), 1)

    def test_flat_list_inputs(self):
        ax = line_plot([0, 1, 2], [1.0, 2.0, 3.0], plot_ids=[0])
        self.assertEqual(len(ax.get_lines()), 1)

    def test_ndarray_still_works(self):
        ax = line_plot(np.arange(10), np.random.randn(10, 3), plot_ids=[0, 1, 2])
        self.assertEqual(len(ax.get_lines()), 3)


class TestAnimate1DNoMutation(unittest.TestCase):
    """Issue 3: ``animate_1D`` mutated the caller's input dictionaries."""

    def tearDown(self):
        plt.close('all')

    def test_single_dict_not_mutated(self):
        d = {'ys': np.random.rand(10, 20)}
        before = set(d)
        animate_1D(d, show=False)
        self.assertEqual(set(d), before)

    def test_list_of_dicts_not_mutated(self):
        v = [{'ys': np.random.rand(5, 8), 'legend': 'a'}]
        before = set(v[0])
        animate_1D(v, show=False)
        self.assertEqual(set(v[0]), before)
        self.assertEqual(v[0]['legend'], 'a')

    def test_ndarray_input_works(self):
        fig = animate_1D(np.random.rand(6, 12), show=False)
        self.assertIsNotNone(fig)

    def test_unknown_dynamic_type_raises(self):
        with self.assertRaises(ValueError):
            animate_1D("not-an-array", show=False)


class TestAnimate2DShapeHandling(unittest.TestCase):
    """Issue 8: cryptic unpack error for non-2D input / size mismatch."""

    def tearDown(self):
        plt.close('all')

    def test_2d_input(self):
        anim = animate_2D(np.random.rand(10, 25), net_size=(5, 5), show=False)
        self.assertIsNotNone(anim)

    def test_3d_input_accepted(self):
        anim = animate_2D(np.random.rand(10, 5, 5), net_size=(5, 5), show=False)
        self.assertIsNotNone(anim)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            animate_2D(np.random.rand(10, 30), net_size=(5, 5), show=False)

    def test_3d_frame_mismatch_raises(self):
        with self.assertRaises(ValueError):
            animate_2D(np.random.rand(10, 4, 4), net_size=(5, 5), show=False)

    def test_bad_ndim_raises(self):
        with self.assertRaises(ValueError):
            animate_2D(np.random.rand(10), net_size=(5, 5), show=False)


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


class TestLinePlotExtra(unittest.TestCase):
    """Cover the branch-heavy parts of line_plot."""

    def setUp(self):
        self.time = np.linspace(0, 10, 50)
        self.values = np.random.randn(50, 4)

    def tearDown(self):
        plt.close('all')

    def test_default_plot_ids_none(self):
        # plot_ids None -> defaults to [0]
        ax = line_plot(self.time, self.values)
        self.assertIsNotNone(ax)

    def test_plot_ids_int(self):
        # plot_ids as int is wrapped into a list
        ax = line_plot(self.time, self.values, plot_ids=2)
        self.assertIsNotNone(ax)

    def test_plot_ids_ndarray(self):
        ax = line_plot(self.time, self.values, plot_ids=np.array([0, 1, 3]))
        self.assertIsNotNone(ax)

    def test_plot_ids_type_error(self):
        with self.assertRaises(TypeError):
            line_plot(self.time, self.values, plot_ids="bad")

    def test_invalid_plot_ids_value_error(self):
        with self.assertRaises(ValueError):
            line_plot(self.time, self.values, plot_ids=[99])
        with self.assertRaises(ValueError):
            line_plot(self.time, self.values, plot_ids=[-1])

    def test_with_existing_ax_and_all_options(self):
        fig, ax = plt.subplots()
        out = line_plot(
            self.time,
            self.values,
            plot_ids=[0, 1, 2],
            ax=ax,
            xlim=(0, 5),
            ylim=(-3, 3),
            xlabel='t',
            ylabel='v',
            legend='sig',
            title='My Title',
            colors=['red'],  # fewer colors than ids -> triggers cycling
            alpha=0.5,
            linewidth=2.0,
        )
        self.assertIs(out, ax)

    def test_single_id_legend(self):
        # single plot id with legend uses the legend string directly
        fig, ax = plt.subplots()
        out = line_plot(self.time, self.values, plot_ids=[1], ax=ax, legend='only')
        self.assertIs(out, ax)

    def test_colors_enough(self):
        fig, ax = plt.subplots()
        out = line_plot(
            self.time, self.values, plot_ids=[0, 1], ax=ax,
            colors=['red', 'blue', 'green'],
        )
        self.assertIs(out, ax)

    def test_show_branch(self):
        # show=True path (Agg show is a no-op)
        ax = line_plot(self.time, self.values, show=True)
        self.assertIsNotNone(ax)

    def test_1d_val_matrix_reshape(self):
        # 1D val_matrix gets reshaped to (N, 1)
        ax = line_plot(self.time, self.values[:, 0])
        self.assertIsNotNone(ax)

    def test_none_val_matrix_raises(self):
        with self.assertRaises(ValueError):
            line_plot(self.time, None)


class TestRasterPlotExtra(unittest.TestCase):
    """Cover raster_plot optional branches."""

    def setUp(self):
        self.time = np.linspace(0, 10, 60)
        self.spikes = np.random.binomial(1, 0.1, (60, 8))

    def tearDown(self):
        plt.close('all')

    def test_with_ax_and_options(self):
        fig, ax = plt.subplots()
        out = raster_plot(
            self.time,
            self.spikes,
            ax=ax,
            marker='|',
            markersize=4,
            color='blue',
            xlabel='time',
            ylabel='idx',
            xlim=(0, 5),
            ylim=(0, 8),
            title='Raster',
            alpha=0.6,
        )
        self.assertIs(out, ax)

    def test_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            raster_plot(self.time[:30], self.spikes)

    def test_show_branch(self):
        ax = raster_plot(self.time, self.spikes, show=True)
        self.assertIsNotNone(ax)

    def test_no_labels(self):
        # falsy labels skip the set calls
        fig, ax = plt.subplots()
        out = raster_plot(self.time, self.spikes, ax=ax, xlabel='', ylabel='')
        self.assertIs(out, ax)


class TestAnimate1D(unittest.TestCase):
    """Exercise animate_1D input-format branches."""

    def tearDown(self):
        plt.close('all')

    def test_ndarray_input(self):
        data = np.random.randn(8, 20)
        fig = animate_1D(data, dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_dict_input_with_legend(self):
        data = {'ys': np.random.randn(8, 20), 'legend': 'v'}
        fig = animate_1D(data, dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_dict_input_no_legend(self):
        data = {'ys': np.random.randn(8, 20)}
        fig = animate_1D(data, dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_list_of_dicts_and_arrays(self):
        v1 = {'ys': np.random.randn(8, 15), 'legend': 'a', 'xs': np.arange(15)}
        v2 = np.random.randn(8, 15)
        fig = animate_1D([v1, v2], dt=0.2, xlim=(0, 14),
                         xlabel='x', ylabel='y', show=False)
        self.assertIsNotNone(fig)

    def test_list_dict_without_legend_or_xs(self):
        # dict entry in a list missing both 'legend' and 'xs' -> default branches
        v1 = {'ys': np.random.randn(8, 15)}
        fig = animate_1D([v1], dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_positive_data_autoscale(self):
        # all-positive data drives the ylim_min>0 / ylim_max>0 branches
        data = np.abs(np.random.randn(8, 20)) + 1.0
        fig = animate_1D(data, dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_list_of_arrays(self):
        v1 = np.random.randn(8, 12)
        v2 = np.random.randn(8, 12)
        fig = animate_1D([v1, v2], dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_static_vars_array_list(self):
        # static vars stored without 'ys'; pass explicit ylim to avoid the
        # auto-ylim path that would index var['ys'] on static entries.
        dyn = np.random.randn(8, 10)
        static_arr = np.random.randn(10)
        fig = animate_1D(dyn, static_vars=[static_arr], dt=0.1,
                         ylim=(-5, 5), show=False)
        self.assertIsNotNone(fig)

    def test_static_var_ndarray_direct(self):
        dyn = np.random.randn(8, 10)
        static_arr = np.random.randn(10)
        fig = animate_1D(dyn, static_vars=static_arr, dt=0.1,
                         ylim=(-5, 5), show=False)
        self.assertIsNotNone(fig)

    def test_static_var_dict_with_data(self):
        # static dict carrying 'data' (1D); explicit ylim avoids auto-ylim
        # which would index the missing 'ys' key for this entry.
        dyn = np.random.randn(8, 10)
        static = {'data': np.random.randn(10), 'legend': 's'}
        fig = animate_1D(dyn, static_vars=[static], dt=0.1,
                         ylim=(-5, 5), show=False)
        self.assertIsNotNone(fig)

    def test_static_var_dict_with_ys(self):
        # static dict carrying 'ys' participates in auto-ylim and is rendered in
        # the frame closure (covers the static-var plotting branch).
        import os
        import tempfile
        dyn = {'ys': np.random.randn(5, 10), 'legend': 'd'}
        static = {'ys': np.random.randn(10), 'legend': 's'}
        gif_path = os.path.join(tempfile.mkdtemp(), 'static.gif')
        fig = animate_1D(dyn, static_vars=static, dt=0.1, save_path=gif_path)
        self.assertTrue(os.path.exists(gif_path))

    def test_static_dict_in_list_without_legend(self):
        dyn = np.random.randn(8, 10)
        static = {'data': np.random.randn(10)}  # no 'legend'
        fig = animate_1D(dyn, static_vars=[static], dt=0.1,
                         ylim=(-5, 5), show=False)
        self.assertIsNotNone(fig)

    def test_static_dict_without_legend(self):
        # static dict (not in a list) with 'ys' but no 'legend'
        dyn = np.random.randn(8, 10)
        static = {'ys': np.random.randn(10)}
        fig = animate_1D(dyn, static_vars=static, dt=0.1,
                         ylim=(-5, 5), show=False)
        self.assertIsNotNone(fig)

    def test_bad_element_in_static_list_raises(self):
        dyn = np.random.randn(8, 10)
        with self.assertRaises(ValueError):
            animate_1D(dyn, static_vars=["bad"], dt=0.1, ylim=(-1, 1),
                       show=False)

    def test_unknown_static_type_raises(self):
        dyn = np.random.randn(8, 10)
        with self.assertRaises(ValueError):
            animate_1D(dyn, static_vars="bad", dt=0.1, ylim=(-1, 1), show=False)

    def test_unknown_dynamic_type_in_list_raises(self):
        with self.assertRaises(ValueError):
            animate_1D(["bad"], dt=0.1, show=False)

    def test_explicit_ylim(self):
        data = np.random.randn(8, 20)
        fig = animate_1D(data, dt=0.1, ylim=(-5, 5), show=False)
        self.assertIsNotNone(fig)

    def test_negative_data_autoscale(self):
        # all-negative data drives the ylim_min/ylim_max negative branches
        data = -np.abs(np.random.randn(8, 20)) - 1.0
        fig = animate_1D(data, dt=0.1, show=False)
        self.assertIsNotNone(fig)

    def test_default_dt(self):
        data = np.random.randn(6, 10)
        with brainstate.environ.context(dt=0.1):
            fig = animate_1D(data, show=False)
        self.assertIsNotNone(fig)

    def test_save_gif_renders_frames(self):
        # Saving forces frame rendering, covering the frame() closure and the
        # gif save branch. matplotlib falls back to the Pillow writer when
        # imagemagick is unavailable.
        import os
        import tempfile
        data = {'ys': np.random.randn(5, 12), 'legend': 'v'}
        tmpdir = tempfile.mkdtemp()
        gif_path = os.path.join(tmpdir, 'anim.gif')
        fig = animate_1D(
            data, dt=0.1, xlim=(0, 11), xlabel='x', ylabel='y',
            save_path=gif_path,
        )
        self.assertTrue(os.path.exists(gif_path))


class TestRemoveAxis(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_invalid_position_raises(self):
        fig, ax = plt.subplots()
        with self.assertRaises(ValueError):
            remove_axis(ax, 'not-a-side')

    def test_valid_position_hides_spine(self):
        # Valid positions hide the corresponding spines.
        fig, ax = plt.subplots()
        remove_axis(ax, 'left', 'top')
        self.assertFalse(ax.spines['left'].get_visible())
        self.assertFalse(ax.spines['top'].get_visible())


class TestAnimate2D(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def test_animate_2d_runs(self):
        # values: (num_step, num_neuron) reshaped to a (height, width) grid.
        values = np.random.rand(5, 12)
        anim = animate_2D(values, net_size=(3, 4), dt=0.1, show=False)
        self.assertIsNotNone(anim)


def test_animations_start_at_first_frame_and_animate_1d_keeps_animation():
    import gc
    import re
    import braintools
    vals = np.arange(12.0).reshape(3, 4)
    anim = braintools.visualize.animate_2D(vals, net_size=(2, 2), dt=1.0, show=False)
    first = []
    anim._func(0)
    first.append(anim._fig.texts[0].get_text())
    assert first == ['Time: 1.00 ms']
    assert list(anim.new_frame_seq())[0] == 0
    fig = braintools.visualize.animate_1D(vals, dt=1.0, show=False)
    gc.collect()
    assert list(fig.animation.new_frame_seq())[0] == 0
    assert len(re.findall(r'frames\[\d+\]', fig.animation.to_jshtml())) == 3  # renders all 3 frames
    plt.close('all')



class TestRasterPlotSpikeTrains(unittest.TestCase):
    """``raster_plot(ts, sp_matrix)`` also accepts a SpikeTrains as ``ts``."""

    def setUp(self):
        import brainunit as u
        from braintools.spike import SpikeTrains
        self.u = u
        self.SpikeTrains = SpikeTrains
        self.spikes = np.zeros((40, 3))
        self.spikes[::4, 0] = 1.
        self.spikes[7, 2] = 1.
        self.ts = np.arange(40) * 0.5

    def tearDown(self):
        plt.close('all')

    def _points(self, ax):
        pts = np.asarray(ax.collections[0].get_offsets())
        return pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def test_matches_legacy_positional_call(self):
        legacy = self._points(raster_plot(self.ts, self.spikes))
        fig, ax = plt.subplots()
        st = self.SpikeTrains.from_dense(self.spikes, dt=0.5 * self.u.ms)
        new = self._points(raster_plot(st, ax=ax))
        np.testing.assert_allclose(new, legacy, rtol=1e-6)

    def test_quantity_times_plotted_in_ms(self):
        st = self.SpikeTrains.from_dense(self.spikes, dt=0.0005 * self.u.second)
        pts = self._points(raster_plot(st))
        np.testing.assert_allclose(pts[:, 0].max(), 36 * 0.5, rtol=1e-5)

    def test_non_time_unit_kept_as_is(self):
        st = self.SpikeTrains.from_dense(self.spikes, dt=1. * self.u.mV)
        pts = self._points(raster_plot(st))
        np.testing.assert_allclose(pts[:, 0].max(), 36., rtol=1e-6)

    def test_unitless_spike_trains(self):
        st = self.SpikeTrains.from_dense(self.spikes, dt=0.5)
        pts = self._points(raster_plot(st))
        np.testing.assert_allclose(pts, self._points(raster_plot(self.ts, self.spikes)), rtol=1e-6)

    def test_spike_trains_and_sp_matrix_raises(self):
        st = self.SpikeTrains.from_dense(self.spikes, dt=0.5)
        with self.assertRaises(ValueError):
            raster_plot(st, self.spikes)

    def test_missing_sp_matrix_still_raises(self):
        with self.assertRaises(ValueError):
            raster_plot(self.ts)
