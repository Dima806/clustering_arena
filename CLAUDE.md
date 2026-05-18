# Project: clustering_arena

## Identity
Clustering algorithm comparison. 6 notebooks, 1 Streamlit app, src/ library.
9 algorithms × 5 real datasets + circles synthetic. Provocative thesis: K-Means loses on 4/5 real + catastrophically on circles.

## Stack
Python 3.12, sklearn, hdbscan, umap-learn, gudhi, matplotlib, seaborn, streamlit.
Deps: uv (pyproject.toml). No pip.
Constraint: 2-CPU Codespace — n_jobs=2 max, datasets < 50MB combined.

## Structure
notebooks/ — 01_data_exploration, 02_algorithm_comparison, 03_deep_dive_density,
              04_deep_dive_hierarchical, 05_choosing_the_right_algorithm,
              06_topological_clustering
src/ — datasets.py, algorithms.py, metrics.py, visualisation.py, diagnostics.py
app/ — streamlit_app.py (sys.path patched at top for src/ imports)
tests/ — test_algorithms.py, test_metrics.py, test_datasets.py,
          test_diagnostics.py, test_visualisation.py

## Algorithms
KMeans, MiniBatchKMeans, Agglomerative(Ward), Agglomerative(Complete),
DBSCAN, HDBSCAN, SpectralClustering, GaussianMixture, ToMATo(gudhi)

## Datasets
penguins (seaborn), wine (sklearn), digits 0-4 (sklearn),
credit card 5K (openml), wholesale (openml),
circles (sklearn synthetic — two concentric rings, topology benchmark)
Note: NB02 arena runs penguins/wine/digits/circles. Credit card + wholesale require fetch_openml (network, ~minutes).

## Makefile Commands
make setup    — first-time install (uv + deps + kernel)
make sync     — fast re-sync after git pull
make lint     — ruff format + ruff check --fix + ty check
make test     — pytest + coverage report (99% coverage)
make notebooks — execute all 6 notebooks (timeout 600s each)
make run      — streamlit app (port 8501)
make security — bandit SAST + pip-audit CVE scan
make ci       — sync → lint → test (notebooks excluded from CI for speed)

## Key Results (best ARI per dataset, NB02 arena)
penguins: GMM 0.959, ToMATo 0.951, Complete 0.943 — K-Means 0.799 LOSES
wine:     GMM 0.918, K-Means 0.897 — close, GMM wins; ToMATo 0.756 (high-d hurts)
digits:   Ward 0.857, K-Means 0.802 LOSES — Spectral fails (high-d, slow); ToMATo 0.766
circles:  ToMATo 1.000, HDBSCAN 1.000 — K-Means 0.017 CATASTROPHIC FAIL

## Conventions
- Docstrings: Google style, 1-line summary only
- Commits: conventional commits, ≤50 char subject
- All code must pass `make lint` before commit
- SpectralClustering: n ≤ 5,000 (O(n²) affinity matrix)
- UMAP: pre-compute once, cache with joblib in outputs/umap_*.pkl
- Notebooks: each must complete < 10 min (timeout=600s in Makefile)
- Notebook outputs go to /tmp (--output-dir /tmp), no .nbconvert.ipynb artifacts

## Token Rules
- Caveman mode: ON (see .claude/skills/caveman.md)
- No restatements of the question
- Code-first, explain only if asked
- Use /compact at each notebook boundary
- Use /clear when switching between notebook and Streamlit work
