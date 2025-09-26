# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
IO functions for reading and writing connectivity data in various formats.

Supports:
- SONATA format (Blue Brain Project standard)
- NWB (Neurodata Without Borders)
- HDF5 (Hierarchical Data Format)
- Parquet (columnar storage)
- Zarr (chunked array storage)
"""

from typing import Dict, Optional, Tuple, Any, Union

import numpy as np

from braintools._misc import set_module_as

__all__ = [
    # Readers
    'from_sonata',
    'from_nwb',
    'from_hdf5',
    'from_parquet',
    'from_zarr',
    'from_csv',
    'from_dict',

    # Writers
    'to_sonata',
    'to_nwb',
    'to_hdf5',
    'to_parquet',
    'to_zarr',
    'to_csv',
    'to_dict',
]


# ----------------------
# Reading Functions
# ----------------------

@set_module_as('braintools.conn')
def from_sonata(
    filepath: str,
    population: Optional[str] = None,
    edge_group: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load connectivity from SONATA format.
    
    SONATA is the standard format used by the Blue Brain Project and 
    Allen Institute for large-scale brain simulations.
    
    Parameters
    ----------
    filepath : str
        Path to SONATA file.
    population : str, optional
        Name of the population to load. If None, loads first population.
    edge_group : str, optional
        Name of the edge group. If None, loads first edge group.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict
        Additional properties (weights, delays, etc).
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for SONATA format. Install with: pip install h5py")

    with h5py.File(filepath, 'r') as f:
        # Navigate to edges group
        edges = f.get('edges', f.get('connectivity', f))

        if edge_group is None:
            # Use first edge group
            edge_group = list(edges.keys())[0]

        edge_data = edges[edge_group]

        # Get source and target indices
        if 'source_node_id' in edge_data:
            pre_indices = edge_data['source_node_id'][:]
        else:
            pre_indices = edge_data['src'][:]

        if 'target_node_id' in edge_data:
            post_indices = edge_data['target_node_id'][:]
        else:
            post_indices = edge_data['tgt'][:]

        # Get additional properties
        properties = {}
        for key in edge_data.keys():
            if key not in ['source_node_id', 'target_node_id', 'src', 'tgt']:
                properties[key] = edge_data[key][:]

    return pre_indices, post_indices, properties


@set_module_as('braintools.conn')
def from_nwb(
    filepath: str,
    electrode_group: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load connectivity from NWB (Neurodata Without Borders) format.
    
    Parameters
    ----------
    filepath : str
        Path to NWB file.
    electrode_group : str, optional
        Name of electrode group to load.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict
        Additional properties.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for NWB format. Install with: pip install h5py")

    with h5py.File(filepath, 'r') as f:
        # NWB has different structure - look for connectivity in processing or analysis
        if 'processing' in f and 'connectivity' in f['processing']:
            conn_data = f['processing']['connectivity']
        elif 'analysis' in f and 'connectivity' in f['analysis']:
            conn_data = f['analysis']['connectivity']
        else:
            # Fallback to general search
            conn_data = f

        # Extract connectivity
        pre_indices = conn_data.get('pre_indices', conn_data.get('source', []))[:]
        post_indices = conn_data.get('post_indices', conn_data.get('target', []))[:]

        properties = {}
        for key in ['weight', 'delay', 'synapse_type']:
            if key in conn_data:
                properties[key] = conn_data[key][:]

    return pre_indices, post_indices, properties


@set_module_as('braintools.conn')
def from_hdf5(
    filepath: str,
    pre_key: str = 'pre_indices',
    post_key: str = 'post_indices',
    group: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load connectivity from HDF5 format.
    
    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    pre_key : str
        Key for presynaptic indices.
    post_key : str
        Key for postsynaptic indices.
    group : str, optional
        HDF5 group containing the data.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict
        Additional properties.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")

    with h5py.File(filepath, 'r') as f:
        if group:
            data = f[group]
        else:
            data = f

        pre_indices = data[pre_key][:]
        post_indices = data[post_key][:]

        # Get other properties
        properties = {}
        for key in data.keys():
            if key not in [pre_key, post_key]:
                try:
                    properties[key] = data[key][:]
                except:
                    pass  # Skip non-array data

    return pre_indices, post_indices, properties


@set_module_as('braintools.conn')
def from_parquet(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load connectivity from Parquet format.
    
    Parquet is efficient for large-scale columnar data storage.
    
    Parameters
    ----------
    filepath : str
        Path to Parquet file.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict
        Additional properties.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet format. Install with: pip install pandas pyarrow")

    df = pd.read_parquet(filepath)

    # Find pre/post columns
    pre_col = None
    post_col = None
    for col in df.columns:
        if 'pre' in col.lower() or 'source' in col.lower():
            pre_col = col
        elif 'post' in col.lower() or 'target' in col.lower():
            post_col = col

    if pre_col is None or post_col is None:
        # Use first two columns as fallback
        pre_col = df.columns[0]
        post_col = df.columns[1]

    pre_indices = df[pre_col].values
    post_indices = df[post_col].values

    # Get other columns as properties
    properties = {}
    for col in df.columns:
        if col not in [pre_col, post_col]:
            properties[col] = df[col].values

    return pre_indices, post_indices, properties


@set_module_as('braintools.conn')
def from_zarr(
    filepath: str,
    pre_key: str = 'pre_indices',
    post_key: str = 'post_indices'
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load connectivity from Zarr format.
    
    Zarr is efficient for chunked array storage.
    
    Parameters
    ----------
    filepath : str
        Path to Zarr store.
    pre_key : str
        Key for presynaptic indices.
    post_key : str
        Key for postsynaptic indices.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict
        Additional properties.
    """
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr is required. Install with: pip install zarr")

    store = zarr.open(filepath, mode='r')

    pre_indices = np.array(store[pre_key])
    post_indices = np.array(store[post_key])

    # Get other arrays as properties
    properties = {}
    for key in store.keys():
        if key not in [pre_key, post_key]:
            properties[key] = np.array(store[key])

    return pre_indices, post_indices, properties


@set_module_as('braintools.conn')
def from_csv(
    filepath: str,
    pre_col: Union[str, int] = 0,
    post_col: Union[str, int] = 1,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load connectivity from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file.
    pre_col : str or int
        Column name or index for presynaptic indices.
    post_col : str or int
        Column name or index for postsynaptic indices.
    **kwargs
        Additional arguments passed to np.loadtxt or pd.read_csv.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict
        Additional properties.
    """
    try:
        import pandas as pd
        df = pd.read_csv(filepath, **kwargs)

        if isinstance(pre_col, int):
            pre_indices = df.iloc[:, pre_col].values
        else:
            pre_indices = df[pre_col].values

        if isinstance(post_col, int):
            post_indices = df.iloc[:, post_col].values
        else:
            post_indices = df[post_col].values

        # Get other columns
        properties = {}
        for i, col in enumerate(df.columns):
            if (isinstance(pre_col, str) and col == pre_col) or (isinstance(pre_col, int) and i == pre_col):
                continue
            if (isinstance(post_col, str) and col == post_col) or (isinstance(post_col, int) and i == post_col):
                continue
            properties[col] = df[col].values

    except ImportError:
        # Fallback to numpy
        data = np.loadtxt(filepath, delimiter=',', **kwargs)
        pre_indices = data[:, pre_col].astype(int)
        post_indices = data[:, post_col].astype(int)
        properties = {}
        if data.shape[1] > 2:
            properties['weights'] = data[:, 2]

    return pre_indices, post_indices, properties


@set_module_as('braintools.conn')
def from_dict(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load connectivity from dictionary.
    
    Parameters
    ----------
    data : dict
        Dictionary with 'pre_indices' and 'post_indices' keys.
        
    Returns
    -------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict
        Additional properties.
    """
    pre_indices = np.asarray(data.get('pre_indices', data.get('pre', data.get('source'))))
    post_indices = np.asarray(data.get('post_indices', data.get('post', data.get('target'))))

    properties = {}
    skip_keys = ['pre_indices', 'post_indices', 'pre', 'post', 'source', 'target']
    for key, value in data.items():
        if key not in skip_keys:
            properties[key] = np.asarray(value) if hasattr(value, '__len__') else value

    return pre_indices, post_indices, properties


# ----------------------
# Writing Functions
# ----------------------

@set_module_as('braintools.conn')
def to_sonata(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    filepath: str,
    properties: Optional[Dict] = None,
    population_name: str = 'default',
    overwrite: bool = False
):
    """Save connectivity in SONATA format.
    
    Parameters
    ----------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    filepath : str
        Path to save file.
    properties : dict, optional
        Additional properties to save.
    population_name : str
        Name of the population.
    overwrite : bool
        Whether to overwrite existing file.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for SONATA format. Install with: pip install h5py")

    mode = 'w' if overwrite else 'a'

    with h5py.File(filepath, mode) as f:
        # Create edges group
        edges = f.require_group('edges')
        pop_group = edges.require_group(population_name)

        # Save indices
        if 'source_node_id' in pop_group:
            del pop_group['source_node_id']
        if 'target_node_id' in pop_group:
            del pop_group['target_node_id']

        pop_group.create_dataset('source_node_id', data=pre_indices, dtype='i8')
        pop_group.create_dataset('target_node_id', data=post_indices, dtype='i8')

        # Save properties
        if properties:
            for key, value in properties.items():
                if key in pop_group:
                    del pop_group[key]
                pop_group.create_dataset(key, data=value)

        # Add metadata
        pop_group.attrs['format'] = 'SONATA'
        pop_group.attrs['version'] = '0.1'


@set_module_as('braintools.conn')
def to_nwb(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    filepath: str,
    properties: Optional[Dict] = None,
    overwrite: bool = False
):
    """Save connectivity in NWB format.
    
    Parameters
    ----------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    filepath : str
        Path to save file.
    properties : dict, optional
        Additional properties to save.
    overwrite : bool
        Whether to overwrite existing file.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for NWB format. Install with: pip install h5py")

    mode = 'w' if overwrite else 'a'

    with h5py.File(filepath, mode) as f:
        # Create NWB structure
        processing = f.require_group('processing')
        conn_module = processing.require_group('connectivity')

        # Save connectivity
        if 'pre_indices' in conn_module:
            del conn_module['pre_indices']
        if 'post_indices' in conn_module:
            del conn_module['post_indices']

        conn_module.create_dataset('pre_indices', data=pre_indices)
        conn_module.create_dataset('post_indices', data=post_indices)

        # Save properties
        if properties:
            for key, value in properties.items():
                if key in conn_module:
                    del conn_module[key]
                conn_module.create_dataset(key, data=value)

        # Add NWB metadata
        f.attrs['nwb_version'] = '2.0'
        conn_module.attrs['description'] = 'Synaptic connectivity data'


@set_module_as('braintools.conn')
def to_hdf5(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    filepath: str,
    properties: Optional[Dict] = None,
    group: Optional[str] = None,
    overwrite: bool = False
):
    """Save connectivity in HDF5 format.
    
    Parameters
    ----------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    filepath : str
        Path to save file.
    properties : dict, optional
        Additional properties to save.
    group : str, optional
        HDF5 group to store data in.
    overwrite : bool
        Whether to overwrite existing file.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")

    mode = 'w' if overwrite else 'a'

    with h5py.File(filepath, mode) as f:
        if group:
            g = f.require_group(group)
        else:
            g = f

        # Save connectivity
        if 'pre_indices' in g:
            del g['pre_indices']
        if 'post_indices' in g:
            del g['post_indices']

        g.create_dataset('pre_indices', data=pre_indices)
        g.create_dataset('post_indices', data=post_indices)

        # Save properties
        if properties:
            for key, value in properties.items():
                if key in g:
                    del g[key]
                g.create_dataset(key, data=value)


@set_module_as('braintools.conn')
def to_parquet(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    filepath: str,
    properties: Optional[Dict] = None
):
    """Save connectivity in Parquet format.
    
    Parameters
    ----------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    filepath : str
        Path to save file.
    properties : dict, optional
        Additional properties to save.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for Parquet. Install with: pip install pandas pyarrow")

    data = {
        'pre_indices': pre_indices,
        'post_indices': post_indices
    }

    if properties:
        data.update(properties)

    df = pd.DataFrame(data)
    df.to_parquet(filepath, index=False)


@set_module_as('braintools.conn')
def to_zarr(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    filepath: str,
    properties: Optional[Dict] = None,
    overwrite: bool = False
):
    """Save connectivity in Zarr format.
    
    Parameters
    ----------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    filepath : str
        Path to save file.
    properties : dict, optional
        Additional properties to save.
    overwrite : bool
        Whether to overwrite existing store.
    """
    try:
        import zarr
    except ImportError:
        raise ImportError("zarr is required. Install with: pip install zarr")

    mode = 'w' if overwrite else 'a'
    store = zarr.open(filepath, mode=mode)

    # Save connectivity
    store['pre_indices'] = pre_indices
    store['post_indices'] = post_indices

    # Save properties
    if properties:
        for key, value in properties.items():
            store[key] = value


@set_module_as('braintools.conn')
def to_csv(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    filepath: str,
    properties: Optional[Dict] = None
):
    """Save connectivity in CSV format.
    
    Parameters
    ----------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    filepath : str
        Path to save file.
    properties : dict, optional
        Additional properties to save.
    """
    data = np.column_stack([pre_indices, post_indices])

    header = 'pre_indices,post_indices'

    if properties:
        for key, value in properties.items():
            data = np.column_stack([data, value])
            header += f',{key}'

    np.savetxt(filepath, data, delimiter=',', header=header, comments='', fmt='%g')


@set_module_as('braintools.conn')
def to_dict(
    pre_indices: np.ndarray,
    post_indices: np.ndarray,
    properties: Optional[Dict] = None
) -> Dict[str, Any]:
    """Convert connectivity to dictionary.
    
    Parameters
    ----------
    pre_indices : np.ndarray
        Source neuron indices.
    post_indices : np.ndarray
        Target neuron indices.
    properties : dict, optional
        Additional properties.
        
    Returns
    -------
    dict
        Dictionary with connectivity data.
    """
    data = {
        'pre_indices': pre_indices,
        'post_indices': post_indices
    }

    if properties:
        data.update(properties)

    return data
