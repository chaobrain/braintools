# File Processing

BrainTools file helpers handle the two kinds of files most experiments touch: model checkpoints and MATLAB data.

## What you'll find here

- A **checkpointing** tutorial covering `msgpack_save` / `msgpack_load`, loading into a template to recover structure and types, restoring `brainstate` models and random-number state in place, `brainunit` quantities, `mismatch='error' | 'warn' | 'ignore'` for evolving models, custom class serialization, `AsyncManager` background writes, error handling, and a periodic-checkpoint-and-resume pattern.
- A **MATLAB files** tutorial covering `load_matfile` and `save_matfile`: how structs, struct arrays, cell arrays, and logicals are converted, the `squeeze_me` / `struct_as_record` options, common round-trip pitfalls, header information, and MATLAB v7.3 files.

For the full list of functions and parameters, see the {doc}`API reference <../apis/file>`.

```{toctree}
:maxdepth: 1

msgpack_checkpointing.ipynb
load_matfile.ipynb
```
