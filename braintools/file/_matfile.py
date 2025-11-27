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
from typing import Any, Dict, List, Union

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import mio5_params

__all__ = [
    'load_matfile',
]


def load_matfile(
    filename: str,
    header_info: bool = True,
    struct_as_record: bool = False,
    squeeze_me: bool = True,
    verbose: bool = True,
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
    filename : str
        The path to the .mat file to be loaded.
    header_info : bool, optional
        If True (default), excludes MATLAB header keys ('__header__',
        '__version__', '__globals__') from the output. If False, includes them.
    struct_as_record : bool, optional
        Whether to load MATLAB structs as numpy record arrays, by default False.
        Passed to scipy.io.loadmat.
    squeeze_me : bool, optional
        Whether to squeeze unit matrix dimensions, by default True.
        Passed to scipy.io.loadmat.
    verbose : bool, optional
        If True (default), print loading information.
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

    See Also
    --------
    scipy.io.loadmat : Underlying MATLAB file loader

    Examples
    --------
    >>> data = load_matfile('experiment_data.mat')
    >>> print(data.keys())
    dict_keys(['trial_data', 'timestamps', 'spike_times'])
    """
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
        if isinstance(element, mio5_params.mat_struct):
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
    # Parse loaded data and filter headers if requested
    dict_output = {}
    for key, value in mat.items():
        # Skip header keys if header_info=True (default)
        if not header_info or not key.startswith('__'):
            dict_output[key] = parse_mat(value)

    return dict_output
