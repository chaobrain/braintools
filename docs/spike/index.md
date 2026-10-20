# Spike Data

`braintools.spike` holds spike data and the tools that produce and combine it. The `SpikeTrains` container stores the spike times of a population in a padded, fixed-shape layout that works under `jax.jit` and `jax.vmap`. The encoders turn numbers or stimuli into spike rasters, and the bitwise operations combine spike tensors and stay differentiable for surrogate-gradient spikes.

## What you'll find here

- **Spike trains**: the dense and event forms of a `SpikeTrains` and the conversions between them (`from_dense`, `from_events`, `from_list`, `to_dense`, `to_events`, `to_list`), binning, windows and neuron selections, units with `dt` and `t_start`, the capacity `K` and how overflow is reported under `jit` and `vmap`, the encoders, the bitwise operations, plotting a raster, ISI histogram and PSTH with `braintools.visualize`, and spike statistics (CV, Fano factor, event-aligned PSTH) from `braintools.analysis.spike`.

The {doc}`spike-encoding tutorial <../misc/spike_encoding>` covers all ten encoders in detail. For the full list of classes and functions, see the {doc}`API reference <../apis/spike>`.

```{toctree}
:maxdepth: 1

01_spike_trains.ipynb
```
