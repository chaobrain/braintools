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
from typing import Optional, Union, Callable

import jax.numpy as jnp

from .context import Context
from .feature import Feature

__all__ = [
    'label',
    'match_label',
    'comparison_label',
]


# =============================================================================
# Output Label Helpers
# =============================================================================

def label(
    value: Union[int, str, Callable[[Context], int]]
) -> Callable[[Context, Optional[Feature]], int]:
    """
    Create an output label specification.

    Parameters
    ----------
    value : int, str, or callable
        - int: Static label value
        - str: Context key containing the label
        - callable: Function (ctx) -> label

    Returns
    -------
    Callable
        Label function (ctx, feature) -> int

    Examples
    --------
    .. code-block:: python

        >>> from braintools.cogtask import Context, label
        >>> ctx = Context()
        >>> ctx.update(ground_truth=1, choice=2)
        >>> label(0)(ctx, None)  # static label, e.g. during fixation
        0
        >>> label('ground_truth')(ctx, None)  # read from the context
        1
        >>> label(lambda ctx: ctx['choice'] + 1)(ctx, None)  # custom rule
        3
    """
    if isinstance(value, int):
        def get_label(ctx: Context, feature: Optional[Feature]) -> int:
            return value

        get_label.__name__ = f"label({value})"
    elif isinstance(value, str):
        def get_label(ctx: Context, feature: Optional[Feature]) -> int:
            return ctx.get(value, 0)

        get_label.__name__ = f"label('{value}')"
    else:
        # Callable
        def get_label(ctx: Context, feature: Optional[Feature]) -> int:
            return value(ctx)

        get_label.__name__ = "label(<callable>)"

    return get_label


def match_label(
    match_key: str,
    match_label: int = 1,
    nonmatch_label: int = 2
) -> Callable[[Context, Optional[Feature]], int]:
    """
    Create a label for match/non-match tasks.

    Parameters
    ----------
    match_key : str
        Context key containing boolean match status (True/False).
    match_label : int
        Label for match trials (default 1).
    nonmatch_label : int
        Label for non-match trials (default 2).

    Returns
    -------
    Callable
        Label function (ctx, feature) -> int

    Examples
    --------
    .. code-block:: python

        >>> from braintools.cogtask import Context, match_label
        >>> spec = match_label('is_match')  # e.g. outputs={'label': spec}
        >>> ctx = Context()
        >>> ctx['is_match'] = True
        >>> int(spec(ctx, None))
        1
        >>> ctx['is_match'] = False
        >>> int(spec(ctx, None))
        2
        >>> int(match_label('is_match', match_label=3, nonmatch_label=4)(ctx, None))
        4
    """

    def get_label(ctx: Context, feature: Optional[Feature]):
        is_match = ctx.get(match_key, False)
        return jnp.where(is_match, match_label, nonmatch_label)

    get_label.__name__ = f"match_label('{match_key}')"
    return get_label


def comparison_label(
    comparison_key: str,
    greater_label: int = 1,
    less_label: int = 2
) -> Callable[[Context, Optional[Feature]], int]:
    """
    Create a label for comparison tasks.

    Parameters
    ----------
    comparison_key : str
        Context key containing boolean comparison result (True if greater).
    greater_label : int
        Label when comparison is True/greater (default 1).
    less_label : int
        Label when comparison is False/less (default 2).

    Returns
    -------
    Callable
        Label function (ctx, feature) -> int

    Examples
    --------
    .. code-block:: python

        >>> from braintools.cogtask import Context, comparison_label
        >>> spec = comparison_label('test_greater_than_sample')
        >>> ctx = Context()
        >>> ctx['test_greater_than_sample'] = True
        >>> int(spec(ctx, None))
        1
        >>> ctx['test_greater_than_sample'] = False
        >>> int(spec(ctx, None))
        2
    """

    def get_label(ctx: Context, feature: Optional[Feature]):
        is_greater = ctx.get(comparison_key, False)
        return jnp.where(is_greater, greater_label, less_label)

    get_label.__name__ = f"comparison_label('{comparison_key}')"
    return get_label
