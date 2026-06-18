``braintools.file`` module
===========================

.. currentmodule:: braintools.file
.. automodule:: braintools.file

Utilities for loading and saving experiment artifacts, including MATLAB
archives and high-performance MsgPack checkpoints.

MATLAB I/O
----------

:func:`load_matfile` reads a ``.mat`` file (via :func:`scipy.io.loadmat`),
recursively converting MATLAB structs to dicts and cell arrays to lists. Pass
``include_header=True`` to keep the ``__header__`` / ``__version__`` /
``__globals__`` metadata. MATLAB v7.3 (HDF5) files are not supported and raise
``NotImplementedError``. :func:`save_matfile` writes a dict of variables back to
a ``.mat`` file.

Checkpointing
-------------

:func:`msgpack_save` / :func:`msgpack_load` serialize PyTrees (including
:class:`brainunit.Quantity` and :class:`brainstate.State` leaves) to and from
``msgpack``. Notes:

- **Mismatch handling.** When a ``target`` is given, ``mismatch`` controls what
  happens on a structural difference (dict keys, list/tuple length, namedtuple
  fields, unit, array shape): ``'error'`` (default) raises, ``'warn'`` warns and
  keeps the target's value, ``'ignore'`` keeps it silently.
- **In-place State restore.** ``State`` leaves are restored in place: the
  template's ``.value`` is mutated. Pass a throwaway template to preserve the
  original.
- **Large arrays.** Arrays above ~1 GiB are transparently chunked to bypass the
  msgpack 2 GiB leaf limit. ``msgpack_load`` accepts an optional ``max_size``
  guard (``None`` = unlimited).
- **Async saves.** :class:`AsyncManager` runs the file write in the background;
  serialization happens synchronously so the on-disk snapshot is consistent.
- **Custom types.** Register handlers with
  :func:`msgpack_register_serialization`.

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    load_matfile
    save_matfile
    msgpack_from_state_dict
    msgpack_to_state_dict
    msgpack_register_serialization
    msgpack_save
    msgpack_load
    AsyncManager
    AlreadyExistsError
    InvalidCheckpointPath
