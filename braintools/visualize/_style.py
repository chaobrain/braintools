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


try:
    import matplotlib.pyplot as plt
    from matplotlib import RcParams, rcParams
    import scienceplots  # noqa: F401
    from braintools._misc import set_module_as


    @set_module_as('braintools.visualize')
    def exclude(rc: RcParams, keys: list):
        rc_new = RcParams()
        for key in rc.keys():
            for k in keys:
                if k in key:
                    break
            else:
                rc_new._set(key, rc[key])
        return rc_new


    style = exclude(plt.style.library['notebook'], ['font.family', 'mathtext.fontset', 'size', 'width'])
    plt.style.core.update_nested_dict(plt.style.library, {'notebook2': style})
    plt.style.core.available[:] = sorted(plt.style.library.keys())


    @set_module_as('braintools.visualize')
    def plot_style1(fontsize=22, axes_edgecolor='black', figsize='5,4', lw=1):
        """Plot style for publication.

        Parameters
        ----------
        fontsize : int
            The font size.
        axes_edgecolor : str
            The exes edge color.
        figsize : str, tuple
            The figure size.
        lw : int
            Line width.
        """
        rcParams['text.latex.preamble'] = [r"\usepackage{amsmath, lmodern}"]
        params = {
            'text.usetex': True,
            'font.family': 'lmodern',
            # 'text.latex.unicode': True,
            'text.color': 'black',
            'xtick.labelsize': fontsize - 2,
            'ytick.labelsize': fontsize - 2,
            'axes.labelsize': fontsize,
            'axes.labelweight': 'bold',
            'axes.edgecolor': axes_edgecolor,
            'axes.titlesize': fontsize,
            'axes.titleweight': 'bold',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.grid': False,
            'axes.facecolor': 'white',
            'lines.linewidth': lw,
            "figure.figsize": figsize,
        }
        rcParams.update(params)

except:
    pass

finally:
    pass
