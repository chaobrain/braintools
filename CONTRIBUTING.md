# Contributing to BrainTools

We welcome contributions of all sizes, from typo fixes to major features. This
document explains how to get started, what we expect from contributors, and how
we review and release changes.

## Ways to Contribute

- Report bugs or request features by opening GitHub issues with clear context
  and reproduction steps.
- Improve documentation, tutorials, and examples.
- Fix defects, refactor internals, or add new functionality.
- Help other community members by reviewing pull requests or answering
  questions.

## Development Workflow

1. **Fork and clone** the repository. Configure the upstream remote so you can
   pull future updates.
2. **Create a feature branch**: `git switch -c feature/my-improvement`.
3. **Install dependencies** using an isolated environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -e .[testing]
   pip install -r requirements-doc.txt  # Only if you plan to build docs
   ```
4. **Run the test suite** early and often:
   ```bash
   pytest
   ```
5. **Format and lint** your changes before committing:
   ```bash
   pre-commit install
   pre-commit run --all-files
   ```
6. **Write clear commit messages** in the present tense, e.g.
   `Add distance-aware connectivity helper`.
7. **Open a pull request** against `main`. Reference related issues and describe
   the motivation, implementation details, and testing performed.

## Coding Guidelines

- Python code must be compatible with CPython 3.10 and newer.
- Follow the rules enforced by `ruff` and the project configuration. Prefer
  descriptive variable names and concise functions.
- Keep modules focused; extract shared logic into utilities instead of duplicating
  code.
- Add or update unit tests for new behavior. Contributions without coverage will
  likely be delayed.
- Document public APIs using reStructuredText or Markdown docstrings and update
  relevant guides in `docs/` when functionality changes.

## Documentation

To preview the documentation locally:

```bash
pip install -r requirements-doc.txt
sphinx-build -b html docs docs/_build/html
```

Please proofread content for clarity and provide screenshots or diagrams when
it helps users understand the feature. Keep narrative tutorials executable by
running the code blocks end-to-end.

## Pull Request Checklist

- [ ] Tests pass locally (`pytest`).
- [ ] `pre-commit run --all-files` reports no issues.
- [ ] New or changed behavior is covered by tests and documentation.
- [ ] Changelog entry added when the change is user-facing.
- [ ] Contributors listed in `AUTHORS.md` updated if applicable.

## Release Process

Core maintainers cut releases on demand. If your change should be part of the
next release, mention it in your pull request. Release steps include:

1. Update `CHANGELOG.md` with highlights of the release.
2. Bump `braintools.__version__`.
3. Build artifacts: `python -m pip wheel . --no-deps -w dist` and
   `python -m build` (PEP 517 source distribution).
4. Run smoke tests on the built wheel in a clean environment.
5. Publish to PyPI and create a GitHub release tag.

## Community Expectations

All participants must follow the project [Code of Conduct](CODE_OF_CONDUCT.md).
If you encounter issues or need clarification, open a discussion or email
maintainers at conduct@braintools.dev.

Thank you for helping BrainTools grow!
