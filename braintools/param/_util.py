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

"""
Utility functions for parameter module.

This module provides helper functions for extracting parameter values
and normalizing shape specifications.
"""

import brainstate

__all__ = [
    'get_param',
    'get_size',
]


def get_param(param):
    """
    Extract the underlying value from a parameter.

    If the parameter is a ``brainstate.State`` instance, returns its ``.value``
    attribute. Otherwise, returns the parameter unchanged.

    Parameters
    ----------
    param : State or any
        A parameter that may be wrapped in a State object.

    Returns
    -------
    any
        The unwrapped parameter value.

    Examples
    --------
    >>> import brainstate
    >>> state = brainstate.ParamState(1.0)
    >>> get_param(state)
    1.0
    >>> get_param(2.0)
    2.0
    """
    if isinstance(param, brainstate.State):
        return param.value
    else:
        return param


def get_size(size):
    """
    Normalize a size specification to a tuple.

    Converts various size representations (int, tuple, list) to a consistent
    tuple format for use with array creation functions.

    Parameters
    ----------
    size : int, tuple, or list
        Size specification. If int, converted to single-element tuple.
        If tuple or list, converted to tuple.

    Returns
    -------
    tuple
        Size as a tuple.

    Raises
    ------
    ValueError
        If size is not an int, tuple, or list.

    Examples
    --------
    >>> get_size(5)
    (5,)
    >>> get_size((3, 4))
    (3, 4)
    >>> get_size([2, 3, 4])
    (2, 3, 4)
    """
    if isinstance(size, int):
        return (size,)
    elif isinstance(size, (tuple, list)):
        return tuple(size)
    else:
        raise ValueError(f"size must be int, tuple, or list, got {type(size).__name__}")
