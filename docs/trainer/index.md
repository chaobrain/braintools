# Training Loop

`braintools.trainer` is a lightweight, PyTorch-Lightning-style training loop for `brainstate` models: a `LightningModule` defines the training, validation and test steps, and a `Trainer` runs them with batching, a jit-compiled gradient step, callbacks, checkpoints and logging. The module is in maintenance mode: bugs are fixed, but no new features are planned.

## What you'll find here

- A **training loop** tutorial: a small classifier written as a `LightningModule`, `DataLoader` batching of plain arrays, validation metrics, `EarlyStopping`, `ModelCheckpoint` with `save_top_k`, a `CSVLogger` whose `metrics.csv` is read back with pandas and plotted, inspecting a checkpoint with `load_checkpoint`, and evaluating a fresh model from the best checkpoint with `Trainer.test(..., ckpt_path=...)` and `Trainer.predict`.

For the full list of classes and parameters, see the {doc}`API reference <../apis/trainer>`.

```{toctree}
:maxdepth: 1

01_training_loop.ipynb
```
