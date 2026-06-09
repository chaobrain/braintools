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

import brainstate
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

import braintools
from braintools.visualize._plots import (
    line_plot,
    raster_plot,
    animate_1D,
    animate_2D,
    remove_axis,
)


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


if __name__ == '__main__':
    unittest.main()
