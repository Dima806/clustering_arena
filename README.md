# clustering_arena

**No, K-Means is not always the answer.**

A head-to-head comparison of 9 clustering algorithms on 5 real-world datasets + 1 synthetic topology benchmark, showing exactly where the default algorithm fails — and which alternatives win.

---

## Results (ARI — higher is better)

| Dataset | K-Means | Winner | Winner ARI | Verdict |
|---|---|---|---|---|
| Penguins (non-convex) | 0.799 | GMM | **0.959** | K-Means loses |
| Wine (varying density) | 0.897 | GMM | **0.918** | K-Means loses |
| Digits 0–4 (high-dim) | 0.802 | Ward | **0.857** | K-Means loses |
| Circles (concentric rings) | 0.017 | ToMATo / HDBSCAN | **1.000** | K-Means catastrophically fails |
| Credit card (noisy) | — | DBSCAN | — | requires `fetch_openml` |
| Wholesale (mixed scale) | — | GMM | — | requires `fetch_openml` |

*ARI = Adjusted Rand Index. Best score from a small hyperparameter grid.*

---

## Setup

```bash
git clone <repo>
cd clustering_arena
make setup      # installs uv + all deps + Jupyter kernel
make test       # 45 tests, 99% coverage
make notebooks  # executes all 5 notebooks
make run        # launches Streamlit app at http://localhost:8501
```

Runs fully inside a **2-CPU / 8 GB GitHub Codespace** — no GPU required.

---

## Structure

```
clustering_arena/
├── notebooks/
│   ├── 01_data_exploration.ipynb        # UMAP projections, Hopkins statistic
│   ├── 02_algorithm_comparison.ipynb    # Arena: ARI heatmap, runtime chart (9 algos × 4 datasets)
│   ├── 03_deep_dive_density.ipynb       # DBSCAN vs HDBSCAN, condensed tree
│   ├── 04_deep_dive_hierarchical.ipynb  # Dendrograms, Ward vs Complete
│   ├── 05_choosing_the_right_algorithm.ipynb  # recommend_algorithm(X)
│   └── 06_topological_clustering.ipynb  # ToMATo: rings/spirals vs high-dim failure
├── src/
│   ├── algorithms.py    # 9 algorithm wrappers (incl. ToMATo)
│   ├── datasets.py      # 5 real dataset loaders + circles synthetic + StandardScaler
│   ├── metrics.py       # ARI, Silhouette, Calinski-Harabasz, Davies-Bouldin
│   ├── visualisation.py # UMAP grids, ARI heatmaps, dendrograms
│   └── diagnostics.py   # Hopkins statistic, elbow, k-distance plots
├── app/
│   └── streamlit_app.py # Interactive clustering arena
├── tests/               # 45 tests, 99% coverage
├── outputs/
│   ├── figures/         # Exported PNGs from notebooks
│   ├── results.csv      # ARI × (algorithm × dataset) — arena scoreboard
│   └── umap_*.pkl       # Cached UMAP embeddings (joblib)
├── pyproject.toml       # uv deps
└── Makefile
```

---

## Algorithms

| # | Algorithm | Key strength |
|---|---|---|
| 1 | K-Means | Speed, simplicity |
| 2 | MiniBatchKMeans | Scalability |
| 3 | Agglomerative (Ward) | Hierarchy, high-dimensional |
| 4 | Agglomerative (Complete) | Non-spherical clusters |
| 5 | DBSCAN | Arbitrary shapes, noise detection |
| 6 | HDBSCAN | Varying density, no eps tuning |
| 7 | Spectral Clustering | Non-convex manifolds (n ≤ 5,000) |
| 8 | Gaussian Mixture Model | Soft assignments, elliptical clusters |
| 9 | ToMATo (gudhi) | Topological: rings, spirals, voids (low-dim) |

---

## Makefile

```
make setup     first-time install (uv + deps + kernel)
make sync      fast re-sync after git pull
make lint      ruff format + check + ty typecheck
make test      pytest with coverage report
make notebooks execute all 6 notebooks end-to-end
make run       launch Streamlit app (port 8501)
make security  bandit SAST + pip-audit CVE scan
make ci        sync → lint → test
```

---

## Stack

Python 3.12 · scikit-learn · hdbscan · umap-learn · gudhi · matplotlib · seaborn · streamlit · uv
