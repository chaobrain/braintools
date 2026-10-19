# Spike Encoding

Convert numbers and time series into spike trains for spiking networks. The tutorial covers all ten encoders in `braintools.spike`: rate codes (`RateEncoder`, `PoissonEncoder`, `BernoulliEncoder`, `PopulationEncoder`), timing codes (`LatencyEncoder`, `RankOrderEncoder`, `SpikeCountEncoder`), signal codes (`DeltaEncoder`, `StepCurrentEncoder`), and symbol codes (`TemporalEncoder`). Each is checked against the rates, timings, or counts it should produce, and the tutorial ends with a guide to choosing one.

The encoders moved from the top level to `braintools.spike` in 0.4.0. The old names (`braintools.RateEncoder` and so on) still work but emit a `BraintoolsDeprecationWarning`, and will be removed in 1.0.0.

```{toctree}
:maxdepth: 1

spike_encoding.ipynb
```

## From rasters to spike times: `SpikeTrains`

Encoders return dense rasters with shape `(n_time, n_neurons)`. Most of those entries are zero. `braintools.spike.SpikeTrains` stores only the spike times, one padded row per neuron. Row `n` holds the `counts[n]` spike times of neuron `n` in ascending order, followed by `+inf` padding up to a fixed `capacity`.

Because the shapes are fixed, a `SpikeTrains` is a JAX pytree. You can build and use it inside `jax.jit` and `jax.vmap`. When `dt` carries a unit, the spike times carry it too:

```python
import brainstate
import brainunit as u
import jax.numpy as jnp
import braintools
from braintools.spike import SpikeTrains

brainstate.environ.set(dt=0.1 * u.ms)
brainstate.random.seed(0)
raster = braintools.spike.PoissonEncoder()(jnp.array([20., 50., 100.]), n_time=10_000)  # 1 s

st = SpikeTrains.from_dense(raster)       # dt comes from brainstate.environ
st.counts                                 # spikes per neuron
st.bin(100 * u.ms)                        # (10, 3) spike counts in 100 ms bins
st.window(200 * u.ms, 500 * u.ms)         # spikes in [200 ms, 500 ms), same capacity
times, ids = st.to_events()               # flat, time-sorted events (host-only)
st.to_dense()                             # back to the boolean raster
braintools.visualize.spike_raster(st)     # the spike plots accept SpikeTrains directly
```

Two other constructors build a `SpikeTrains` from other formats:

- `SpikeTrains.from_events(times, neuron_ids, n_neurons)` takes flat event arrays.
- `SpikeTrains.from_list([...])` takes one array per neuron, and the arrays may differ in length.

`capacity` is the padded row length. It is part of the array shapes, so it must be fixed when a function is traced:

- With concrete data, it defaults to the largest spike count.
- Inside `jit` or `vmap`, pass it explicitly, for example `SpikeTrains.from_dense(raster, capacity=200)`.
- If a traced neuron fires more often than `capacity`, its extra spikes are dropped and `overflow` is set. Call `st.check()` outside the transform to turn that flag into an error.
