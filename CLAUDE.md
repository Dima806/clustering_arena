# Project: clustering_arena

## Identity
Clustering algorithm comparison. 5 notebooks, 1 Streamlit app, src/ library.
8 algorithms × 5 real datasets. Provocative thesis: K-Means loses on 4/5.

## Stack
Python 3.11, sklearn, hdbscan, umap-learn, matplotlib, seaborn, streamlit.
Deps: uv (pyproject.toml). No pip.
Constraint: 2-CPU Codespace — n_jobs=2 max, datasets < 50MB combined.

## Structure
notebooks/ — 01_eda, 02_comparison, 03_density, 04_hierarchical, 05_decision
src/ — datasets.py, algorithms.py, metrics.py, visualisation.py, diagnostics.py
app/ — streamlit_app.py
tests/ — test_algorithms.py, test_metrics.py, test_datasets.py

## Algorithms
KMeans, MiniBatchKMeans, Agglomerative(Ward), Agglomerative(Complete),
DBSCAN, HDBSCAN, SpectralClustering, GaussianMixture

## Datasets
penguins (seaborn), wine (sklearn), digits 0-4 (sklearn),
credit card 5K (openml), wholesale (openml)

## Makefile Commands
make setup    — first-time install (uv + deps + kernel)
make sync     — fast re-sync after git pull
make lint     — ruff format + ruff check --fix + ty check
make test     — pytest
make notebooks — execute all notebooks (validates < 5 min each)
make run      — streamlit app
make ci       — full pipeline (sync → lint → test → notebooks)

## Conventions
- Docstrings: Google style, 1-line summary only
- Commits: conventional commits, ≤50 char subject
- All code must pass `make lint` before commit
- SpectralClustering: n ≤ 5,000 (O(n²) affinity matrix)
- UMAP: pre-compute once, cache with joblib
- Notebooks: restart-and-run-all must complete < 5 min each

## Token Rules
- Caveman mode: ON (see .claude/skills/caveman.md)
- No restatements of the question
- Code-first, explain only if asked
- Use /compact at each notebook boundary
- Use /clear when switching between notebook and Streamlit work
