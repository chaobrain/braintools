# Metric Function

`braintools.metric` collects losses and metrics for training models. The losses are pure JAX functions that match their optax counterparts.

The measures for analysing neural activity (firing rates, spike-train synchrony and distances, LFP analysis, functional connectivity) moved to `braintools.analysis` in 0.4; their tutorials are now under {doc}`Neural Data Analysis <../analysis/index>`. The old `braintools.metric` names still work but warn when called and will be removed in 1.0.

## What you'll find here

- **Classification losses**: sigmoid and softmax cross-entropy, focal loss, label smoothing, and a gradient-descent example.
- **Regression losses**: L1, L2, Huber, and log-cosh; the differing default reductions; and how loss shape controls robustness to outliers.
- **Ranking**: the listwise softmax loss with masking, weights, and reductions, and a small learning-to-rank example.
- **Cosine similarity**: aligned pairs versus pairwise matrices, retrieval, and comparing population activity patterns.

For the full list of functions and parameters, see the {doc}`API reference <../apis/metric>`.

```{toctree}
:maxdepth: 1

01_classification_losses.ipynb
02_regression_losses.ipynb
03_ranking_learning_to_rank.ipynb
04_pairwise_embedding_similarity.ipynb
```
