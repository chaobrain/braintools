# braintools.visualize.morphology (VIS-3)

## Goal

Move braincell's morphology and topology plotting (`braincell.vis`, about
11k lines and 6.8k lines of tests) into braintools as
`braintools.visualize.morphology`, so that other simulators can reuse it.
The plots must not depend on braincell: they take neutral data types that
a simulator fills in. braincell keeps its `vis2d`/`vis3d` methods and turns
them into thin wrappers in a later task (VIS-4a).

## Decisions

1. **A subpackage that is not imported by `import braintools`.** It pulls in
   matplotlib collections, and optionally networkx, PyVista and Plotly.
   `import braintools.visualize.morphology` loads none of the optional
   libraries; each backend imports its library on first use. Reason: keep
   `import braintools` as fast as before.
2. **Neutral, frozen data types** (`BranchData`, `MorphologyData`,
   `RegionData`, `LocsetData`, `TopologyGraph`), with geometry stored as
   read-only float64 arrays in micrometres. A simulator either converts its
   objects or implements `__morphology_data__()` / `__topology_data__()`;
   `as_morphology_data` / `as_topology_graph` normalise both. Reason: an
   explicit, validated contract instead of duck typing on simulator
   objects; tests build data directly.
3. **The layout and scene code moves unchanged.** It walks a tree
   (`node.parent`, `node.children`), so a private `_tree.MorphologyTree`
   view over `MorphologyData` provides that interface, built once per data
   instance. The changes to moved files are limited to imports, type
   annotations and how the tree is obtained. Plots are pixel-identical to
   braincell's for every test fixture.
4. **The layout cache keys on the data's structure** (`structure_key`),
   including the order of children, which the old key missed.
5. **Values state their space when ambiguous.** `ValueSpec.space` is
   `"branch"`, `"segment"`, `"centerline"` or `None` (inferred from the
   length, as before).
6. **Topology graphs are plotted from `TopologyGraph`.**
   `plot_topology_graph(graph, frame=...)` reads values, highlight
   fractions and the value label from the graph; giving them both in the
   graph and as arguments is an error. Branch-level graphs carry no values.
7. **Removed while moving:** the `legacy` root layout, the `set_defaults`
   alias, and SWC-based tests (SWC reading stays in braincell).
8. **Dependencies:** a `morphology` extra (`networkx`, `plotly`,
   `pyvista`); a missing optional backend raises an error naming
   `pip install braintools[morphology]`. `braintools.visualize` no longer
   imports Plotly at import time.
9. **Version 0.5.0.** This is a new public subpackage.

## Steps

1. Copy the modules and tests; add `_data.py`, `_tree.py` and the test
   fixture builders (`_testing.py`); adapt imports.
2. Add `ValueSpec.space`, the structural cache key and the
   `TopologyGraph` input; remove the legacy pieces.
3. Docs: `docs/apis/visualize_morphology.rst`, runnable examples, See Also
   links from the matching `braintools.visualize` functions; the API
   coverage test and the doctest runner also walk the lazy subpackage.
4. `pyproject.toml` extra, changelog, version; add braincell (main) to the
   downstream workflow.
5. Pixel baselines from braincell (same tolerance).

## Verify

- `pytest braintools/visualize`, bare `mypy`,
  `python scripts/run_doctests.py braintools.visualize.morphology`,
  `sphinx-build -W`, and the API coverage test pass.
- `import braintools.visualize.morphology` imports no braincell, networkx,
  PyVista or Plotly (subprocess test).
- The same plots drawn by braincell and by this subpackage are
  pixel-identical for the shared fixtures.
