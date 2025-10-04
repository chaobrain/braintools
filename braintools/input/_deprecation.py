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

"""Deprecation utilities for the input module."""

import functools
import warnings
from typing import Callable, Type

from braintools._misc import set_module_as


@set_module_as('braintools.input')
def deprecated_alias(old_name: str, new_name: str, version: str = "1.0.0"):
    """Create a deprecation warning decorator for functions or classes.

    Parameters
    ----------
    old_name : str
        The old deprecated name
    new_name : str
        The new recommended name
    version : str
        Version when the deprecation will be removed
    """

    def decorator(func_or_class):
        if isinstance(func_or_class, type):
            # For classes
            original_init = func_or_class.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(
                    f"`{old_name}` is deprecated and will be removed in version {version}. "
                    f"Please use `{new_name}` instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                original_init(self, *args, **kwargs)

            func_or_class.__init__ = new_init
            return func_or_class
        else:
            # For functions
            @functools.wraps(func_or_class)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"`{old_name}` is deprecated and will be removed in version {version}. "
                    f"Please use `{new_name}` instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
                return func_or_class(*args, **kwargs)

            return wrapper

    return decorator


@set_module_as('braintools.input')
def create_deprecated_class(new_class: Type, old_name: str, new_name: str, version: str = "1.0.0") -> Type:
    """Create a deprecated alias for a class.

    Parameters
    ----------
    new_class : Type
        The new class to alias
    old_name : str
        The old deprecated name
    new_name : str
        The new recommended name
    version : str
        Version when the deprecation will be removed

    Returns
    -------
    Type
        A new class that inherits from the original with a deprecation warning
    """

    class DeprecatedClass(new_class):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"`{old_name}` is deprecated and will be removed in version {version}. "
                f"Please use `{new_name}` instead.",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(*args, **kwargs)

    DeprecatedClass.__name__ = old_name
    DeprecatedClass.__qualname__ = old_name
    return DeprecatedClass


@set_module_as('braintools.input')
def create_deprecated_function(new_func: Callable, old_name: str, new_name: str, version: str = "1.0.0") -> Callable:
    """Create a deprecated alias for a function.

    Parameters
    ----------
    new_func : Callable
        The new function to alias
    old_name : str
        The old deprecated name
    new_name : str
        The new recommended name
    version : str
        Version when the deprecation will be removed

    Returns
    -------
    Callable
        A wrapper function that calls the original with a deprecation warning
    """

    @functools.wraps(new_func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"`{old_name}()` is deprecated and will be removed in version {version}. "
            f"Please use `{new_name}()` instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return new_func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    return wrapper
