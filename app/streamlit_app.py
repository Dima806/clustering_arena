"""Interactive Clustering Arena — compare algorithms side by side."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.algorithms import (
    run_agglomerative_complete,
    run_agglomerative_ward,
    run_dbscan,
    run_gmm,
    run_hdbscan,
    run_kmeans,
    run_minibatch_kmeans,
    run_spectral,
)
from src.datasets import load_digits_subset, load_penguins, load_wine_data
from src.metrics import compute_all_metrics

st.set_page_config(page_title="Clustering Arena", layout="wide")
st.title("Clustering Arena")
st.caption("K-Means is not always the answer — see for yourself.")

DATASET_LOADERS = {
    "Penguins (333 × 4)": load_penguins,
    "Wine (178 × 13)": load_wine_data,
    "Digits 0-4 (~900 × 64)": load_digits_subset,
}

CENTROID_ALGOS = {
    "KMeans",
    "MiniBatchKMeans",
    "Agglomerative Ward",
    "Agglomerative Complete",
    "Spectral",
}

ALGORITHM_FNS = {
    "KMeans": run_kmeans,
    "MiniBatchKMeans": run_minibatch_kmeans,
    "Agglomerative Ward": run_agglomerative_ward,
    "Agglomerative Complete": run_agglomerative_complete,
    "DBSCAN": run_dbscan,
    "HDBSCAN": run_hdbscan,
    "Spectral": run_spectral,
    "GMM": run_gmm,
}


@st.cache_data
def get_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and cache dataset by name."""
    return DATASET_LOADERS[name]()


@st.cache_data
def get_umap_embedding(x: np.ndarray) -> np.ndarray:
    """Compute and cache 2D UMAP embedding."""
    from umap import UMAP

    return UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(x)


with st.sidebar:
    st.header("Dataset")
    dataset_name = st.selectbox("Choose dataset", list(DATASET_LOADERS))

    st.header("Algorithms")
    selected_algos = st.multiselect(
        "Select 2–4 algorithms",
        list(ALGORITHM_FNS),
        default=["KMeans", "HDBSCAN"],
    )

    st.header("Hyperparameters")
    n_clusters = st.slider("n_clusters / n_components", 2, 8, 3)
    eps = st.slider("DBSCAN eps", 0.1, 3.0, 0.5, 0.1)
    min_samples = st.slider("DBSCAN min_samples", 2, 20, 5)
    min_cluster_size = st.slider("HDBSCAN min_cluster_size", 5, 50, 10)

if not selected_algos:
    st.warning("Select at least one algorithm.")
    st.stop()

x, y_true = get_dataset(dataset_name)  # type: ignore[arg-type]
st.write(f"**Shape:** {x.shape} | **True clusters:** {len(np.unique(y_true))}")

with st.spinner("Computing UMAP projection..."):
    embedding = get_umap_embedding(x)

label_sets: dict[str, np.ndarray] = {"Ground truth": y_true}
results: list[dict[str, str | float]] = []

for algo_name in selected_algos:
    fn = ALGORITHM_FNS[algo_name]
    t0 = time.perf_counter()
    if algo_name in CENTROID_ALGOS:
        labels = fn(x, n_clusters=n_clusters)
    elif algo_name == "GMM":
        labels = fn(x, n_components=n_clusters)
    elif algo_name == "DBSCAN":
        labels = fn(x, eps=eps, min_samples=min_samples)
    else:
        labels = fn(x, min_cluster_size=min_cluster_size)
    runtime = time.perf_counter() - t0
    label_sets[algo_name] = labels
    metrics = compute_all_metrics(x, y_true, labels)
    results.append(
        {
            "Algorithm": algo_name,
            "Runtime (s)": f"{runtime:.3f}",
            **{k: f"{v:.3f}" for k, v in metrics.items()},
        }
    )

st.subheader("UMAP Projections")
n_plots = len(label_sets)
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
axes_list: list[plt.Axes] = [axes] if n_plots == 1 else list(axes)
for ax, (name, labels) in zip(axes_list, label_sets.items(), strict=False):
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Metrics")
st.dataframe(pd.DataFrame(results).set_index("Algorithm"), use_container_width=True)
