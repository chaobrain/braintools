# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``braintools.cogtask.utils``."""

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest


"""Package-level tests for ``braintools.cogtask``."""

import importlib.util

import braintools
import braintools.cogtask as ct


def test_version_reexports_braintools_version():
    assert ct.__version__ == braintools.__version__
    assert ct.__version_info__ == braintools.__version_info__
    assert ct.__version__ != '0.0.1'


def test_version_in_all():
    assert '__version__' in ct.__all__
    assert '__version_info__' in ct.__all__


def test_private_version_module_removed():
    assert importlib.util.find_spec('braintools.cogtask._version') is None


def test_all_names_resolve():
    for name in ct.__all__:
        assert hasattr(ct, name), name
