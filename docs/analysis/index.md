# Neural Data Analysis

`braintools.analysis` collects measures for analysing recorded or simulated neural activity. They are grouped by the kind of data they take: `braintools.analysis.spike` for spike trains, `braintools.analysis.spectral` for spectra and coupling of continuous signals, `braintools.analysis.field` for local field potentials and current source density, `braintools.analysis.population` for multichannel activity, and `braintools.analysis.stats` for general correlation statistics. Each measure is checked in these tutorials against a case with a known answer.

These functions used to live in `braintools.metric`; the old names still work in 0.4 but warn when called and will be removed in 1.0.

## What you'll find here

- **Firing rates and population synchrony**: rasters, `firing_rate`, the chance level of the spike-count coherence $\kappa$, Golomb's $\chi^2$, and functional connectivity (and its dynamics).
- **Spike-train distances and synchrony**: Victor-Purpura and van Rossum distances, coincidence measures versus STTC for independent trains, phase locking, and burst synchrony.
- **LFP analysis**: generating an LFP from spikes, Welch spectra and frequency resolution, coherence, spectral entropy, phase-amplitude coupling, phase coherence, and current source density.

For the full list of functions and parameters, see the {doc}`API reference <../apis/analysis>`.

```{toctree}
:maxdepth: 1

05_spiking_metrics.ipynb
06_spiking_sync.ipynb
07_lfp_analysis.ipynb
```
