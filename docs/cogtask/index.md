# Cognitive Tasks

A modular, composable framework for constructing cognitive tasks for neural-network training and computational neuroscience simulations.

## What you'll find here

- A **quickstart** showing how to sample single trials and JIT/`vmap`-friendly batches from a pre-built task.
- A **building-blocks tutorial** covering features, declarative phases, encoders, label helpers, and both the instance-based and class-based `Task` definition patterns — plus vector-output tasks, conditional control flow, and custom encoders.
- A **variable-length trial sequences** tutorial covering `VariableDuration`, conditional control flow (`If` / `Switch` / `While`), the packed-buffer-plus-mask layout returned by `sample_trial` and `batch_sample(..., return_mask=True)`, masked losses, and the `TruncExp` / `UniformDuration` duration samplers.

For the full list of pre-built tasks, encoders, phases, and utilities, see the {doc}`API reference <../apis/cogtask>`.

```{toctree}
:maxdepth: 1

01_quickstart.ipynb
02_building_custom_tasks.ipynb
03_variable_length_trials.ipynb
```
