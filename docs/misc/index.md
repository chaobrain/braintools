# Spike Encoding

Convert numbers and time series into spike trains for spiking networks. The tutorial covers all ten encoders in `braintools`: rate codes (`RateEncoder`, `PoissonEncoder`, `BernoulliEncoder`, `PopulationEncoder`), timing codes (`LatencyEncoder`, `RankOrderEncoder`, `SpikeCountEncoder`), signal codes (`DeltaEncoder`, `StepCurrentEncoder`), and symbol codes (`TemporalEncoder`). Each is checked against the rates, timings, or counts it should produce, and the tutorial ends with a guide to choosing one.

```{toctree}
:maxdepth: 1

spike_encoding.ipynb
```
