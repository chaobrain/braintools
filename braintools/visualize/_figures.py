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


# -*- coding: utf-8 -*-


from typing import Union

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from braintools._misc import set_module_as

__all__ = [
    'get_figure',
]


@set_module_as('braintools.visualize')
def get_figure(
    row_num: int,
    col_num: int,
    row_len: Union[int, float] = 3,
    col_len: Union[int, float] = 6
):
    """Get the constrained_layout figure.

    Parameters
    ----------
    row_num : int
        The row number of the figure.
    col_num : int
        The column number of the figure.
    row_len : int, float
        The length of each row.
    col_len : int, float
        The length of each column.

    Returns
    -------
    fig_and_gs : tuple
        Figure and GridSpec.
    """
    fig = plt.figure(figsize=(col_num * col_len, row_num * row_len), constrained_layout=True)
    gs = GridSpec(row_num, col_num, figure=fig)
    return fig, gs
