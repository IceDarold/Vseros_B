# Vseros_B – RecSys experimentation toolkit

## Overview

This repository contains a modular experimentation stack for the Vseros_B Stage 1
recall track. It focuses on building reproducible candidate generators and
rankers that deliver top-20 recommendations for every user. The codebase is
centred around a reusable `BaseExperiment` abstraction which orchestrates data
loading, fitting, candidate generation, evaluation, and artefact persistence.

## Repository layout

```
vseros_b/
├─ pyproject.toml              # project metadata and formatting rules
├─ requirements.txt            # Kaggle-friendly dependency lock
├─ README.md
├─ CONTRIBUTING.md             # contribution guidelines
├─ VERSION                     # exported package version (used in W&B runs)
├─ src/
│  └─ vseros_b/
│     ├─ __init__.py
│     ├─ config.py             # column names, paths, quick-mode toggles
│     ├─ artifacts.py          # artefact I/O + W&B helpers
│     ├─ metrics.py
│     ├─ base_exp.py
│     ├─ data.py               # load_and_prepare()
│     ├─ registry/
│     │  ├─ __init__.py        # central experiment registry
│     │  └─ default.py         # default experiment registrations
│     ├─ features/             # feature generation building blocks
│     │  ├─ __init__.py
│     │  ├─ builders.py
│     │  ├─ sources.py
│     │  ├─ embeddings.py
│     │  └─ schema.py
│     ├─ candidates/           # candidate generators
│     │  ├─ pop_decay.py, trending.py, covis.py, item2vec_lite.py, ppr.py, …
│     ├─ exps/                 # experiment implementations
│     │  ├─ exp101_pop_decay.py
│     │  ├─ …
│     │  └─ exp108_recall_fusion.py
│     └─ utils/
│        ├─ logging.py
│        └─ seeds.py
├─ notebooks/
│  ├─ orchestrator.ipynb
│  └─ lightgcn_train.ipynb
├─ scripts/
│  ├─ bootstrap_kaggle.py
│  └─ check_env.py
└─ .pre-commit-config.yaml
```

## Configuration

All runtime paths are controlled from `config.py`. The defaults assume the
following layout relative to the project root:

```
./data/train_data.pq
./data/sample_submission.csv
./artifacts/
```

Set the `VSEROS_B_*` environment variables to override any path, for example:

```bash
export VSEROS_B_DATA_DIR=/kaggle/input/vseros-data
export VSEROS_B_ARTIFACT_DIR=/kaggle/working/artifacts
```

Running `vseros_b.artifacts.ensure_project_dirs()` or invoking
`load_and_prepare()` will create the required directories.

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare the dataset (copy `train_data.pq` and `sample_submission.csv` into
   the configured data directory).
3. Load data and run a baseline experiment inside a Python session or notebook:
   ```python
   from pathlib import Path

   from vseros_b import registry
   from vseros_b.data import load_and_prepare

   context = load_and_prepare()
   exp_cls = registry.get("exp101_pop_decay")
   experiment = exp_cls()
   experiment.fit(context)
   metrics = experiment.evaluate(context)
   print(metrics.head())
   ```
4. Artefacts and metrics will appear under `artifacts/` as defined by the
   configuration.

## Adding new experiments

- Implement the experiment in `src/vseros_b/exps/` by subclassing
  `BaseExperiment` and reusing helpers from `candidates/`, `features/`, and
  `metrics.py`.
- Register the experiment in `src/vseros_b/registry/default.py` so that it is
  available through the dynamic registry API.
- Document any additional dependencies in `requirements.txt` if they are needed
  for Kaggle submissions.

## Tooling

- Formatting and linting are enforced via `.pre-commit-config.yaml`
  (Black, isort, Flake8, trailing whitespace).
- The package exposes its semantic version through the `VERSION` file and the
  `vseros_b.__version__` attribute.
- Scripts under `scripts/` provide convenience entry points for Kaggle/Colab
  environments (`bootstrap_kaggle.py`) and quick environment diagnostics
  (`check_env.py`).

Happy experimenting!
