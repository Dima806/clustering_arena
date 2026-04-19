"""Unified wrappers for all 8 clustering algorithms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture


def run_kmeans(
    x: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Run KMeans clustering."""
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3).fit_predict(x)


def run_minibatch_kmeans(
    x: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Run MiniBatchKMeans clustering."""
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3).fit_predict(
        x
    )


def run_agglomerative_ward(x: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Run Agglomerative clustering with Ward linkage."""
    return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(x)


def run_agglomerative_complete(x: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Run Agglomerative clustering with Complete linkage."""
    return AgglomerativeClustering(n_clusters=n_clusters, linkage="complete").fit_predict(x)


def run_dbscan(x: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """Run DBSCAN clustering."""
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)


def run_hdbscan(x: np.ndarray, min_cluster_size: int = 10) -> np.ndarray:
    """Run HDBSCAN clustering."""
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(x)


def run_spectral(
    x: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Run Spectral Clustering (requires n ≤ 5000 samples)."""
    if len(x) > 5000:
        msg = f"SpectralClustering requires n ≤ 5000, got {len(x)}"
        raise ValueError(msg)
    return SpectralClustering(
        n_clusters=n_clusters, random_state=random_state, n_jobs=2
    ).fit_predict(x)


def run_gmm(
    x: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "full",
    random_state: int = 42,
) -> np.ndarray:
    """Run Gaussian Mixture Model clustering."""
    return GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    ).fit_predict(x)


ALGORITHMS: dict[str, Callable[..., np.ndarray]] = {
    "kmeans": run_kmeans,
    "minibatch_kmeans": run_minibatch_kmeans,
    "agglomerative_ward": run_agglomerative_ward,
    "agglomerative_complete": run_agglomerative_complete,
    "dbscan": run_dbscan,
    "hdbscan": run_hdbscan,
    "spectral": run_spectral,
    "gmm": run_gmm,
}
