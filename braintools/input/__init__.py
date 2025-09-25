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
"""

# Import from categorized modules
from .basic import *
from .basic import __all__ as basic_all

from .waveforms import *
from .waveforms import __all__ as waveforms_all

from .pulses import *
from .pulses import __all__ as pulses_all

from .stochastic import *
from .stochastic import __all__ as stochastic_all

# Keep importing from currents for backward compatibility (if it still exists)
try:
    from .currents import *
    from .currents import __all__ as currents_all
    __all__ = currents_all
except ImportError:
    # If currents.py doesn't exist, combine all the module exports
    __all__ = basic_all + waveforms_all + pulses_all + stochastic_all
