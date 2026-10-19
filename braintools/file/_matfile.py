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
import tempfile
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


def _is_hdf5(filename: PathLike) -> bool:
    """Return True if ``filename`` is an HDF5 file.

    The HDF5 superblock may sit at byte 0 or, after a user block, at 512, 1024,
    2048, ... MATLAB v7.3 files use a 512-byte user block holding the text header
    ``MATLAB 7.3 MAT-file ...``, so their signature is at offset 512, not 0.
    """
    size = os.path.getsize(filename)
    with open(filename, 'rb') as fh:
        offset = 0
        while offset + len(_HDF5_MAGIC) <= size:
            fh.seek(offset)
            if fh.read(len(_HDF5_MAGIC)) == _HDF5_MAGIC:
                return True
            offset = 512 if offset == 0 else offset * 2
    return False


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
    .. code-block:: python

        >>> import os
        >>> import tempfile
        >>> import numpy as np
        >>> from braintools.file import load_matfile, save_matfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'data.mat')
        >>> save_matfile(path, {'spike_times': np.array([0.1, 0.5]), 'label': 'trial-1'})
        >>> data = load_matfile(path)
        >>> sorted(data.keys())
        ['label', 'spike_times']

    Include the MATLAB metadata headers:

    .. code-block:: python

        >>> data = load_matfile(path, include_header=True)
        >>> '__header__' in data
        True
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
    v73_message = (
        f'"{filename}" is a MATLAB v7.3 (HDF5) file, which scipy.io.loadmat '
        f'cannot read. Use an HDF5-aware reader such as `h5py` or the `mat73` '
        f'package to load v7.3 files.'
    )
    if _is_hdf5(filename):
        raise NotImplementedError(v73_message)

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
    except NotImplementedError as e:
        # scipy's own v7.3 check (reads the version from the MATLAB header).
        raise NotImplementedError(v73_message) from e
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
    .. code-block:: python

        >>> import os
        >>> import tempfile
        >>> import numpy as np
        >>> from braintools.file import save_matfile
        >>> path = os.path.join(tempfile.mkdtemp(), 'out.mat')
        >>> save_matfile(path, {'x': np.arange(3), 'label': 'trial-1'})
        >>> os.path.exists(path)
        True
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

    # Write to a unique temporary file in the destination directory, then
    # atomically rename it into place. ``savemat`` writes directly to its target,
    # so a failure part-way through (e.g. an unsavable value such as ``None``)
    # would otherwise truncate an existing good .mat file in place -- leaving a
    # file that still loads without error but is silently missing variables.
    # ``os.replace`` is atomic on both POSIX and Windows, and the temp file lives
    # in the same directory so the rename never crosses a filesystem.
    # ``savemat`` appends ``.mat`` to an extensionless name (``appendmat=True`` by
    # default). The temp file below always ends in ``.mat``, so apply that rule to
    # the final name here, or ``save_matfile('out', ...)`` would write ``out``.
    filename = os.fspath(filename)
    if kwargs.get('appendmat', True) and not filename.endswith('.mat'):
        filename = filename + '.mat'
    target_dir = os.path.dirname(os.path.abspath(filename))
    try:
        fd, tmp_filename = tempfile.mkstemp(suffix='.mat', dir=target_dir)
        os.close(fd)
    except OSError as e:
        raise ValueError(f'Failed to save MATLAB file "{filename}": {e}') from e

    try:
        savemat(tmp_filename, data, **kwargs)
        # mkstemp creates the file with 0600; keep the destination's existing
        # permissions when overwriting, otherwise use the default a plain
        # ``open(..., 'wb')`` would have given (0666 masked by the umask).
        try:
            if os.path.exists(filename):
                mode = os.stat(filename).st_mode & 0o777
            else:
                umask = os.umask(0)
                os.umask(umask)
                mode = 0o666 & ~umask
            os.chmod(tmp_filename, mode)
        except OSError:
            pass
        os.replace(tmp_filename, filename)
    except Exception as e:
        if os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except OSError:
                pass
        raise ValueError(f'Failed to save MATLAB file "{filename}": {e}') from e
