# Vseros_B – RecSys experimentation toolkit

## Overview
This repository hosts a recommendation-system experimentation stack for the Vseros_B Stage 1 recall track. It focuses on generating top‑20 candidate items for each user based on historical click logs provided by T‑Bank. The codebase is centred around a reusable `BaseExperiment` abstraction that orchestrates loading data, fitting models (where required), generating candidate lists, computing recall metrics, and persisting artefacts.

### Dataset snapshot
- **train_data.pq** – 47 days of user–item interactions; the historical split used for training/validation.
- **sample_submission.csv** – template listing all target users; for each, the system must return 20 predicted item IDs.
- **Task** – predict the items a user will click during the final 7‑day horizon immediately after the training period.
- **Metric** – mean Average Precision at 20 (mAP@20).

### Repository layout
```
├── notebooks/               # ad‑hoc exploration reports
└── src/vseros_b/
    ├── base_exp.py          # BaseExperiment with fit/candidates/evaluate/save lifecycle
    ├── config.py            # central configuration (paths, run modes, defaults)
    ├── data.py              # data loading, caching, splits
    ├── artifacts.py         # artefact management helpers
    ├── metrics.py           # recall/precision/NDCG metrics
    ├── pop_decay.py, trending.py, covis.py, ...
    ├── exp10x_*.py          # concrete experiment definitions (popularity, trending, covisitation,
    │                        # item2vec, LightGCN import, Personalized PageRank, recall fusion, …)
    └── utils (various)      # supporting algorithm-specific utilities
```
Each `exp10x_*` module subclasses `BaseExperiment` and encapsulates the workflow for a single candidate generator. Results, metrics, and auxiliary data are saved through `artifacts.py`, making runs reproducible and shareable.

## Suggested improvements to streamline experimentation

### 1. Package structure
- Split `src/vseros_b` into subpackages (e.g. `data/`, `experiments/`, `models/`, `pipelines/`, `utils/`) to clarify responsibilities and reduce module length.
- Expose experiment classes through an `experiments/registry.py` to enable dynamic lookup (`get_experiment("exp105_item2vec")`).

### 2. Configuration management
- Migrate the current config module to structured configs (Pydantic, dataclasses, or Hydra/OmegaConf) with YAML overrides. This would simplify quick parameter sweeps, artifact naming, and reproducibility.

### 3. Unified CLI entrypoint
- Provide a single CLI (`python -m vseros_b.run --exp exp105_item2vec --config confs/item2vec.yaml --mode quick`) that triggers the full lifecycle.
- Bundle typical experiment sequences (train → generate → evaluate) into reusable “playbooks.”

### 4. Artefact conventions
- Extend `artifacts.py` to standardize directory layout: `${ARTIFACT_ROOT}/{experiment}/{run_id}/{stage}` with metadata (timestamp, parameters, git SHA) for reproducibility.
- Add utilities to list previous runs, load the latest artefacts, and compare metrics across experiments.

### 5. Metrics and benchmarking
- Centralize evaluation by wrapping `metrics.py` in a service that can benchmark multiple experiments on the same split and produce dashboards/tables.
- Track summary statistics (coverage, diversity) alongside mAP@20.

### 6. Documentation and onboarding
- Flesh out module-level docstrings describing data expectations, caching scheme, and experiment contracts.
- Provide quick-start instructions: where to place datasets, how to run the baseline experiment, and how to contribute new candidate generators.
- Maintain a changelog or roadmap capturing tested hypotheses and planned improvements.

### 7. Quality gates
- Introduce type checking (`mypy`) and linting (`ruff`/`flake8`) to keep experiments consistent and catch regressions early.
- Prepare smoke tests (e.g. via `pytest`) that instantiate each experiment in quick mode to ensure dependency sanity.

These refinements should make the platform more maintainable, accelerate iterative experimentation, and ease collaboration among teammates.
