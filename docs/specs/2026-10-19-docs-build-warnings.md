# Fix documentation build errors & warnings (+ README badge)

## Context
`python -m sphinx -b html docs docs/_build/html` succeeds (exit 0) but prints ~380 WARNING/ERROR lines
(log: `/tmp/bt_docs.log`). Goal: a clean build. The user also asked to remove the
"Documentation Status" (readthedocs) badge from README.md.
All work goes on the worktree branch `worktree-fix-docs-build`. Per AGENTS.md rule 7, first write this
plan to `docs/specs/2026-10-19-docs-build-warnings.md`.

## Root causes and fixes (by count)

1. **~150 "Field list ends without a blank line"** (conn, init, input, optim, trainer, cogtask, surrogate).
   Confirmed by an isolated build: caused by a bug in `sphinx_autodoc_typehints` 3.9.6. Its `_remove_preexisting_type`
   counts blank lines as part of the `:type x:` entry it replaces, so it deletes the blank line before `.. rubric:: Examples`.
   Fix (user chose this): new `docs/_ext/docstring_fixes.py` with a function `restore_field_list_blank_line(lines)`
   plus an `autodoc-process-docstring` handler connected with `priority=600` (after typehints' default 500).
   It inserts `''` after the last line of a field list (`:field ...:` + indented continuation lines) when the next line is
   non-blank, not indented, and not another field. In conf.py, add the `_ext` directory to `sys.path` and register the extension.
   Unit test in `docs/_ext/docstring_fixes_test.py` (the AllToAll-style lines from the isolated build → blank line restored;
   field list followed by a blank line → unchanged; consecutive fields → unchanged).

2. **~160 "Footnote [n] is not referenced"**: 78 docstrings in `braintools/{optim,metric,surrogate,conn}` define
   `.. [n]` references without citing them. Fix (user chose this): add `[n]_` markers at a natural spot
   (extended summary or Notes, never the one-line summary), one per reference entry. Get the exact list of objects from the
   build log.

3. **18 "unknown output mime type application/vnd.plotly.v1+json"** (`docs/visualize/05_interactive_visualization.ipynb`):
   every such output also has `text/html`, which renders. Add `"mystnb.unknown_mime_type"` to `suppress_warnings`.

4. **16 "autosummary: stub file not found"** (`apis/input.rst`, `apis/surrogate.rst`): `section`/`Section`, `sigmoid`/`Sigmoid`, etc.
   collide on case-insensitive filesystems (`/mnt/d`, Windows, macOS). Add `autosummary_filename_map` in conf.py that maps each
   lowercase function, whose name matches a class in the same module case-insensitively, to `<fullname>-function`. Build the map
   automatically by scanning `braintools.input` and `braintools.surrogate`.

5. **12 "duplicate object description"** (OptaxOptimizer, LightningModule, Logger): napoleon's `Attributes` and `Methods` sections
   generate directives that `:members:` also generates. Fix: set `napoleon_use_ivar = True` in conf.py, and remove the redundant
   `Methods` section from `OptaxOptimizer` (`braintools/optim/_optax_optimizer.py:103`). `:members:` already documents those methods.
   Check whether the `Methods` sections in `input/_composable_base.py:71` and `surrogate/_base.py:126` cause duplicates
   after the rebuild, and treat them the same way if they do.

6. **7 "Unknown target name: 1"** (`apis/surrogate.rst:61`): the one-line summaries of GaussianGrad, LogTailedRelu,
   MultiGaussianGrad, QPseudoSpike, ReluGrad, S2NN and SlayerGrad contain `[1]_`. Move the citation into the
   extended summary.

7. **3 "Undefined substitution |x|"** (surrogate InvSquareGrad, ReluGrad, ReluGrad.surrogate_grad): put `|x|` inside
   `:math:` or a literal.

8. **~30 one-off docutils errors**: Unexpected indentation, Block quote/Bullet list ends without a blank line, Inline emphasis/literal
   start-string (in cogtask.working_memory, LatencyEncoder, optim.{ScipyOptimizer, CyclicLR, NevergradOptimizer, PolynomialLR,
   ReduceLROnPlateau}, quad.{dde_*, imex_*}, init.{DistanceProfile, Initialization}, input.{NoisySinusoidal, OUProcess,
   WienerProcess}). Fix each docstring directly, using blank lines before lists and indented blocks, and escaping or using literals for `*`/`` ` ``.

9. **Config/toctree odds and ends**
   - conf.py: rename deprecated `jupyter_execute_notebooks` → `nb_execution_mode` and `execution_timeout` → `nb_execution_timeout`.
   - `misc/spike_encoding.md` and `.ipynb` share a docname: rename the `.md` to `misc/index.md` (matching the other tutorial
     folders) and update `docs/index.rst`.
   - `apis/tree.rst` isn't in any toctree: add `apis/tree.rst` to the API Reference toctree in `docs/index.rst`.
   - Forward ref `'OptaxOptimizer'` in `braintools/optim/_optax_lr_scheduler.py:110`: add an `if TYPE_CHECKING:` import of
     `OptaxOptimizer`.

10. **README.md**: delete the `<a href='https://brainx.chaobrain.com/braintools/'>…Documentation Status…</a>` block (lines 10–12).

## Verification
- `pytest docs/_ext/docstring_fixes_test.py` and the existing tests of touched modules (`braintools/optim`, `surrogate`, `quad`, `input`)
  to confirm no regressions from docstring edits.
- Full rebuild from `docs/`: `python -m sphinx -b html -j 8 . _build/html > /tmp/bt_docs2.log 2>&1` (~6 min), then
  `grep -cE "WARNING|ERROR" /tmp/bt_docs2.log` should be 0. Explain any leftovers.
- Spot-check a rendered page (e.g. `generated/braintools.conn.AllToAll.html`) to confirm the Examples rubric and parameter types render.

## Follow-up (2026-10-19)

- **sphinx-autodoc-typehints upgraded to 3.17.2** (pinned `>=3.17.2` in `requirements-doc.txt`). 3.17.2 fixes the
  "removes the blank line after a replaced `:type x:` field" bug upstream (verified with an isolated build that warned
  under 3.9.6), so the `docs/_ext/docstring_fixes.py` workaround from item 1 was removed.
- **Case-collision stub names** moved from conf.py into `docs/_ext/autosummary_names.py`, now applied to every module named
  by `.. currentmodule::` in `docs/apis/*.rst`, not just `input`/`surrogate`. A class keeps the plain stub name and the
  colliding names get a `-function` / `-class` / `-object` suffix. Submodules are ignored. Tests: `docs/_ext/autosummary_names_test.py`,
  including one that asserts every documented module's stub names are unique case-insensitively.
- **CI**: new `.github/workflows/docs.yml` (push to main, PRs) runs the helper tests and
  `sphinx-build -W --keep-going`. `deploy-docs.yml` also builds with `-W --keep-going`, so a release can't ship with warnings.
