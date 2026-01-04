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

"""
File I/O and Checkpointing Utilities.

This module provides utilities for file input/output operations and model checkpointing,
specifically designed for neuroscience and machine learning workflows.

**Key Features:**

- **MATLAB File I/O**: Load and parse .mat files with automatic type conversion
- **Model Checkpointing**: Save and restore model states using efficient msgpack serialization
- **Async Saving**: Non-blocking checkpoint saves for better performance
- **Type Safety**: Proper handling of BrainUnit quantities and BrainState objects
- **Flexible Loading**: Support for mismatch handling between saved and current model structures

**Quick Start:**

.. code-block:: python

    from braintools.file import load_matfile, msgpack_save, msgpack_load

    # Load MATLAB data
    data = load_matfile('experiment_data.mat')

    # Save model checkpoint
    msgpack_save('model_checkpoint.pkl', model_state)

    # Load checkpoint back
    restored_state = msgpack_load('model_checkpoint.pkl', target=model_state)

**MATLAB File Loading:**

.. code-block:: python

    from braintools.file import load_matfile

    # Load with default settings (excludes MATLAB headers)
    data = load_matfile('data.mat')

    # Include MATLAB metadata
    data = load_matfile('data.mat', header_info=False)

    # Access nested structures (automatically converted to Python dicts/lists)
    spike_times = data['trial_data']['spike_times']

**Model Checkpointing:**

.. code-block:: python

    import brainstate as bst
    from braintools.file import msgpack_save, msgpack_load, AsyncManager

    # Simple synchronous save
    msgpack_save('checkpoint.pkl', model.state_dict())

    # Load checkpoint with mismatch handling
    state = msgpack_load('checkpoint.pkl', target=model.state_dict(), mismatch='warn')

    # Async saving for large models (non-blocking)
    with AsyncManager() as manager:
        msgpack_save('checkpoint.pkl', model.state_dict(), async_manager=manager)
        # Continue training while save happens in background

    # Custom serialization for user-defined types
    from braintools.file import msgpack_register_serialization

    def my_type_to_dict(obj):
        return {'data': obj.data}

    def my_type_from_dict(obj, state_dict, mismatch='error'):
        obj.data = state_dict['data']
        return obj

    msgpack_register_serialization(MyType, my_type_to_dict, my_type_from_dict)

"""

# MATLAB file I/O
from ._matfile import (
    load_matfile,
)

# Checkpointing utilities
from ._msg_checkpoint import (
    msgpack_from_state_dict,
    msgpack_to_state_dict,
    msgpack_register_serialization,
    msgpack_save,
    msgpack_load,
    AsyncManager,
)

__all__ = [
    # MATLAB I/O
    'load_matfile',

    # Checkpointing
    'msgpack_from_state_dict',
    'msgpack_to_state_dict',
    'msgpack_register_serialization',
    'msgpack_save',
    'msgpack_load',
    'AsyncManager',
]
