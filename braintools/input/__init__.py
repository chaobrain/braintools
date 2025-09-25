# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


"""
This module provides various methods to form current inputs.
You can access them through ``braintools.input.XXX``.

The module now supports two APIs:
1. **Composable API (recommended)**: Object-oriented API that allows combining inputs
2. **Functional API**: Traditional function-based API for backward compatibility

Composable API Example:
-----------------------
>>> import brainunit as u
>>> from braintools.input import RampInput, SinusoidalInput
>>> ramp = RampInput(0, 1, 500 * u.ms)
>>> sine = SinusoidalInput(0.5, 10 * u.Hz, 500 * u.ms)
>>> combined = ramp + sine  # Combine inputs
>>> scaled = combined.scale(0.5)  # Transform result
>>> array = scaled()  # Generate the array

Functional API Example:
-----------------------
>>> import brainunit as u
>>> from braintools.input import ramp_input, sinusoidal_input
>>> ramp = ramp_input(0, 1, 500 * u.ms)
>>> sine = sinusoidal_input(0.5, 10 * u.Hz, 500 * u.ms)
>>> combined = ramp + sine  # Simple array addition
"""

# Import functional API for backward compatibility
from ._basic import *
from ._basic import __all__ as basic_all2
# Import base classes for composable API
from ._composable_base import *
from ._composable_base import __all__ as base_all
# Import composable classes
from ._composable_basic import *
from ._composable_basic import __all__ as basic_all
from ._pulses import *
from ._pulses import __all__ as pulses_all2
from ._composable_pulses import *
from ._composable_pulses import __all__ as pulses_all
from ._stochastic import *
from ._stochastic import __all__ as stochastic_all2
from ._composable_stochastic import *
from ._composable_stochastic import __all__ as stochastic_all
from ._waveforms import *
from ._waveforms import __all__ as waveforms_all2
from ._composable_waveforms import *
from ._composable_waveforms import __all__ as waveforms_all

# Define __all__ for both APIs
__all__ = base_all + basic_all + waveforms_all + pulses_all + stochastic_all
__all__ = __all__ + basic_all2 + waveforms_all2 + pulses_all2 + stochastic_all2
del (base_all, basic_all, waveforms_all, pulses_all, stochastic_all)
del (basic_all2, waveforms_all2, pulses_all2, stochastic_all2)
