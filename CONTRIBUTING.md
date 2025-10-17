# Contributing to `vseros_b`

Thank you for taking the time to contribute! This project is organised as a
research toolkit: experiments live under `src/vseros_b/exps`, candidate
generators under `src/vseros_b/candidates`, and shared utilities reside in the
`features`, `utils`, and `registry` subpackages.

## Adding a new experiment

1. **Create a module** in `src/vseros_b/exps/expXYZ_<short_name>.py` that
   subclasses `BaseExperiment` and implements at least `fit` and `evaluate`.
2. **Keep imports local** to the package. Use relative imports (e.g.
   `from ..candidates import pop_decay`) rather than absolute paths.
3. **Register** the experiment by adding it to `DEFAULT_EXPERIMENTS` in
   `src/vseros_b/registry/default.py`. This makes it discoverable via the
   registry helpers and orchestrator notebooks.
4. **Save artefacts** under the paths provided by `BaseExperiment.io`. Use the
   helpers in `artifacts.py` for consistent logging and W&B integration.

## Extending feature builders

Feature logic lives under `src/vseros_b/features/`:

- `sources.py` should expose pure data transforms that accept pandas DataFrames
  or candidate maps and return feature frames.
- `builders.py` combines those frames into matrices consumed by rankers.
- `schema.py` defines feature metadata (`FeatureSchema`, `FeatureDefinition`).

When introducing new feature sets, document them in `schema.py`, provide a
builder in `builders.py`, and cover the main aggregation logic in
`sources.py`.

## Coding guidelines

- Follow the formatting and linting rules enforced by the pre-commit hooks
  (`black`, `isort`, `flake8`). Run `pre-commit run --all-files` locally before
  opening a pull request.
- Prefer type hints; most modules are annotated and checked by static tooling.
- Avoid hard-coded paths. Configuration must flow through `config.py` and be
  overridable via environment variables (see the existing `Paths` dataclass).
- Keep notebooks lightweight: heavier processing belongs in modules under
  `src/vseros_b` which can then be imported from notebooks.

## Reporting bugs or requesting features

Please open an issue with:

- A short summary of the problem.
- Steps to reproduce (for bugs) or motivation/use-case (for features).
- Any relevant logs or stack traces.

We appreciate your help in improving the experimentation stack!
