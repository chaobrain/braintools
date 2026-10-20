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
- **Neuro data formats** (optional extras): HDF5 with unit metadata
  (``pip install braintools[hdf5]``), NWB units tables and time series
  (``braintools[nwb]``), and NEO spike-train interoperability (``braintools[neo]``)

**Quick Start:**

.. code-block:: python

    from braintools.file import load_matfile, msgpack_save, msgpack_load

    # Load MATLAB data
    data = load_matfile('experiment_data.mat')

    # Save model checkpoint
    msgpack_save('model_checkpoint.msgpack', model_state)

    # Load checkpoint back
    restored_state = msgpack_load('model_checkpoint.msgpack', target=model_state)

**MATLAB File Loading:**

.. code-block:: python

    from braintools.file import load_matfile, save_matfile

    # Load with default settings (excludes MATLAB headers)
    data = load_matfile('data.mat')

    # Include MATLAB metadata
    data = load_matfile('data.mat', include_header=True)

    # Access nested structures (automatically converted to Python dicts/lists)
    spike_times = data['trial_data']['spike_times']

    # Save data back to a MATLAB .mat file
    save_matfile('out.mat', {'spike_times': spike_times})

**Model Checkpointing:**

.. code-block:: python

    import brainstate
    from braintools.file import msgpack_save, msgpack_load, AsyncManager

    # Simple synchronous save
    msgpack_save('checkpoint.msgpack', model.states())

    # Load checkpoint with mismatch handling
    state = msgpack_load('checkpoint.msgpack', target=model.states(), mismatch='warn')  # restores in place

    # Async saving for large models (non-blocking)
    with AsyncManager() as manager:
        msgpack_save('checkpoint.msgpack', model.states(), async_manager=manager)
        # Continue training while save happens in background

    # Custom serialization for user-defined types
    from braintools.file import msgpack_register_serialization

    def my_type_to_dict(obj):
        return {'data': obj.data}

    def my_type_from_dict(obj, state_dict, mismatch='error'):
        obj.data = state_dict['data']
        return obj

    msgpack_register_serialization(MyType, my_type_to_dict, my_type_from_dict)

**Neuro data formats:**

.. code-block:: python

    import brainunit as u
    import jax.numpy as jnp
    from braintools.spike import SpikeTrains
    from braintools.file import save_hdf5, load_hdf5, save_nwb_units, load_nwb_units, to_neo

    trains = SpikeTrains.from_list([jnp.array([1.0, 5.0]) * u.ms, jnp.array([2.0]) * u.ms])

    # Nested pytrees of arrays, Quantities and SpikeTrains (needs h5py)
    save_hdf5('run.h5', {'v': jnp.zeros(3) * u.mV, 'spikes': trains})
    restored = load_hdf5('run.h5')

    # NWB units table (needs pynwb); times are stored in seconds
    save_nwb_units('session.nwb', trains, session_description='demo', identifier='run-1')
    trains_s = load_nwb_units('session.nwb')

    # NEO spike trains (needs neo and quantities)
    neo_trains = to_neo(trains)

"""

# MATLAB file I/O
from ._matfile import (
    load_matfile,
    save_matfile,
)

# Checkpointing utilities
from ._msg_checkpoint import (
    msgpack_from_state_dict,
    msgpack_to_state_dict,
    msgpack_register_serialization,
    msgpack_save,
    msgpack_load,
    AsyncManager,
    AlreadyExistsError,
    InvalidCheckpointPath,
)

# Optional neuro data formats (the backends are imported lazily)
from ._hdf5 import (
    save_hdf5,
    load_hdf5,
)
from ._neo import (
    to_neo,
    from_neo,
)
from ._nwb import (
    save_nwb_units,
    load_nwb_units,
    save_nwb_timeseries,
    load_nwb_timeseries,
)

__all__ = [
    # MATLAB I/O
    'load_matfile',
    'save_matfile',

    # Checkpointing
    'msgpack_from_state_dict',
    'msgpack_to_state_dict',
    'msgpack_register_serialization',
    'msgpack_save',
    'msgpack_load',
    'AsyncManager',
    'AlreadyExistsError',
    'InvalidCheckpointPath',

    # HDF5, NEO and NWB
    'save_hdf5',
    'load_hdf5',
    'to_neo',
    'from_neo',
    'save_nwb_units',
    'load_nwb_units',
    'save_nwb_timeseries',
    'load_nwb_timeseries',
]
