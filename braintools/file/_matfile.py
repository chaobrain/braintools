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

import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy.io import loadmat, savemat

try:  # SciPy >= 1.8 exposes ``mat_struct`` from the public namespace.
    from scipy.io.matlab import mat_struct
except ImportError:  # pragma: no cover - fallback for very old SciPy releases
    from scipy.io.matlab.mio5_params import mat_struct

__all__ = [
    'load_matfile',
    'save_matfile',
]

# HDF5 files (used by MATLAB v7.3 ``.mat`` files) start with this 8-byte signature.
_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'

PathLike = Union[str, os.PathLike]


def load_matfile(
    filename: PathLike,
    include_header: bool = False,
    struct_as_record: bool = False,
    squeeze_me: bool = True,
    verbose: bool = False,
    *,
    header_info: Optional[bool] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    A simple function to load a .mat file using scipy from Python.
    It uses a recursive approach for parsing properly Matlab objects.

    This function recursively converts MATLAB data structures to Python types:
    - MATLAB structs → Python dictionaries
    - MATLAB cell arrays (object arrays) → Python lists
    - Numeric arrays → NumPy arrays

    Parameters
    ----------
    filename : str or os.PathLike
        The path to the .mat file to be loaded.
    include_header : bool, optional
        If True, includes the MATLAB header keys ('__header__', '__version__',
        '__globals__') in the output. If False (default), excludes them.
    struct_as_record : bool, optional
        Whether to load MATLAB structs as numpy record arrays, by default False.
        Passed to scipy.io.loadmat.
    squeeze_me : bool, optional
        Whether to squeeze unit matrix dimensions, by default True.
        Passed to scipy.io.loadmat.
    verbose : bool, optional
        If True, print loading information. Defaults to False.
    header_info : bool, optional
        .. deprecated::
            Use ``include_header`` instead. Note that ``header_info`` had the
            *inverse* meaning of its name: the old ``header_info=True``
            corresponds to ``include_header=False``.
    **kwargs
        Additional keyword arguments passed to scipy.io.loadmat.

    Returns
    -------
    dict
        A dictionary with the content of the .mat file, with MATLAB-specific
        structures converted to Python types.

    Raises
    ------
    TypeError
        If filename is not a string or path-like object.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the path is not a file, or if loading fails.
    NotImplementedError
        If the file is a MATLAB v7.3 (HDF5) file, which scipy cannot read.

    See Also
    --------
    save_matfile : Save a dictionary to a MATLAB .mat file.
    scipy.io.loadmat : Underlying MATLAB file loader.

    Examples
    --------
    >>> data = load_matfile('experiment_data.mat')
    >>> print(data.keys())
    dict_keys(['trial_data', 'timestamps', 'spike_times'])

    Include the MATLAB metadata headers:

    >>> data = load_matfile('experiment_data.mat', include_header=True)
    """
    # Backward-compatible handling of the deprecated, inverted ``header_info`` flag.
    if header_info is not None:
        warnings.warn(
            "`header_info` is deprecated and its sense was the inverse of its name; "
            "use `include_header` instead (the old `header_info=True` corresponds to "
            "`include_header=False`).",
            DeprecationWarning,
            stacklevel=2,
        )
        include_header = not header_info

    # Input validation
    if not isinstance(filename, (str, os.PathLike)):
        raise TypeError(
            f'filename must be a string or path-like object, got {type(filename).__name__}'
        )

    # File existence check
    if not os.path.exists(filename):
        raise FileNotFoundError(f'MATLAB file not found: {filename}')
    if not os.path.isfile(filename):
        raise ValueError(f'Path is not a file: {filename}')

    # MATLAB v7.3 files are HDF5 and cannot be read by scipy.io.loadmat; detect
    # them up front so users get an actionable error instead of an opaque failure.
    with open(filename, 'rb') as fh:
        if fh.read(8) == _HDF5_MAGIC:
            raise NotImplementedError(
                f'"{filename}" is a MATLAB v7.3 (HDF5) file, which scipy.io.loadmat '
                f'cannot read. Use an HDF5-aware reader such as `h5py` or the `mat73` '
                f'package to load v7.3 files.'
            )

    if verbose:
        print(f'Loading MATLAB file from {filename}')

    def parse_mat(element: Any) -> Union[List, Dict[str, Any], np.ndarray, Any]:
        """Recursively parse MATLAB data structures to Python types.

        Parameters
        ----------
        element : Any
            MATLAB data structure element to parse

        Returns
        -------
        Union[List, Dict[str, Any], np.ndarray, Any]
            Parsed Python data structure
        """
        # MATLAB cell arrays (object arrays) → Python lists
        # Using isinstance() for proper subclass support
        # Using ndim for safer dimension checking
        if isinstance(element, np.ndarray) and element.dtype == np.object_ and element.ndim > 0:
            return [parse_mat(entry) for entry in element]

        # MATLAB structs → Python dictionaries
        if isinstance(element, mat_struct):
            return {fn: parse_mat(getattr(element, fn)) for fn in element._fieldnames}

        # Regular numeric arrays, scalars, or other types → return as-is
        return element

    # Load MATLAB file with error handling
    try:
        mat = loadmat(
            filename,
            struct_as_record=struct_as_record,
            squeeze_me=squeeze_me,
            **kwargs
        )
    except Exception as e:
        raise ValueError(f'Failed to load MATLAB file "{filename}": {e}') from e
    # Parse loaded data and filter headers unless explicitly requested.
    dict_output = {}
    for key, value in mat.items():
        if include_header or not key.startswith('__'):
            dict_output[key] = parse_mat(value)

    return dict_output


def save_matfile(
    filename: PathLike,
    data: Dict[str, Any],
    verbose: bool = False,
    **kwargs
) -> None:
    """
    Save a dictionary of variables to a MATLAB ``.mat`` file via scipy.

    This is the counterpart of :func:`load_matfile`. It is a thin, validated
    wrapper around :func:`scipy.io.savemat`.

    Parameters
    ----------
    filename : str or os.PathLike
        Destination path for the ``.mat`` file.
    data : dict
        Mapping from MATLAB variable names (str) to array-like values.
    verbose : bool, optional
        If True, print saving information. Defaults to False.
    **kwargs
        Additional keyword arguments passed to :func:`scipy.io.savemat`
        (e.g. ``do_compression``, ``oned_as``, ``format``).

    Raises
    ------
    TypeError
        If filename is not path-like, or data is not a dict.
    ValueError
        If saving fails.

    See Also
    --------
    load_matfile : Load a MATLAB .mat file.
    scipy.io.savemat : Underlying MATLAB file writer.

    Examples
    --------
    >>> import numpy as np
    >>> save_matfile('out.mat', {'x': np.arange(3), 'label': 'trial-1'})
    """
    if not isinstance(filename, (str, os.PathLike)):
        raise TypeError(
            f'filename must be a string or path-like object, got {type(filename).__name__}'
        )
    if not isinstance(data, dict):
        raise TypeError(
            f'data must be a dict mapping variable names to values, got {type(data).__name__}'
        )

    if verbose:
        print(f'Saving MATLAB file to {filename}')

    try:
        savemat(filename, data, **kwargs)
    except Exception as e:
        raise ValueError(f'Failed to save MATLAB file "{filename}": {e}') from e
