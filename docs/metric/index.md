# Metric Function

`braintools.metric` collects losses for training models and measures for analysing neural activity. The losses are pure JAX functions that match their optax counterparts; the neural measures are checked in these tutorials against cases with a known answer.

## What you'll find here

- **Classification losses**: sigmoid and softmax cross-entropy, focal loss, label smoothing, and a gradient-descent example.
- **Regression losses**: L1, L2, Huber, and log-cosh; the differing default reductions; and how loss shape controls robustness to outliers.
- **Ranking**: the listwise softmax loss with masking, weights, and reductions, and a small learning-to-rank example.
- **Cosine similarity**: aligned pairs versus pairwise matrices, retrieval, and comparing population activity patterns.
- **Firing rates and population synchrony**: `raster_plot`, `firing_rate`, the chance level of the spike-count coherence $\kappa$, Golomb's $\chi^2$, and functional connectivity (and its dynamics).
- **Spike-train distances and synchrony**: Victor-Purpura and van Rossum distances, coincidence measures versus STTC for independent trains, phase locking, and burst synchrony.
- **LFP analysis**: generating an LFP from spikes, Welch spectra and frequency resolution, coherence, spectral entropy, phase-amplitude coupling, phase coherence, and current source density.

For the full list of functions and parameters, see the {doc}`API reference <../apis/metric>`.

```{toctree}
:maxdepth: 1

01_classification_losses.ipynb
02_regression_losses.ipynb
03_ranking_learning_to_rank.ipynb
04_pairwise_embedding_similarity.ipynb
05_spiking_metrics.ipynb
06_spiking_sync.ipynb
07_lfp_analysis.ipynb
```
