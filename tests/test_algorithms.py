"""Tests for all 8 clustering algorithm wrappers on toy data."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.algorithms import (
    run_agglomerative_complete,
    run_agglomerative_ward,
    run_dbscan,
    run_gmm,
    run_hdbscan,
    run_kmeans,
    run_minibatch_kmeans,
    run_spectral,
    run_tomato,
)


@pytest.fixture()
def toy_data() -> np.ndarray:
    x, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
    return x


def test_kmeans_shape(toy_data: np.ndarray) -> None:
    labels = run_kmeans(toy_data, n_clusters=3)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) == 3


def test_minibatch_kmeans_shape(toy_data: np.ndarray) -> None:
    labels = run_minibatch_kmeans(toy_data, n_clusters=3)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) == 3


def test_agglomerative_ward_shape(toy_data: np.ndarray) -> None:
    labels = run_agglomerative_ward(toy_data, n_clusters=3)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) == 3


def test_agglomerative_complete_shape(toy_data: np.ndarray) -> None:
    labels = run_agglomerative_complete(toy_data, n_clusters=3)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) == 3


def test_dbscan_shape(toy_data: np.ndarray) -> None:
    labels = run_dbscan(toy_data, eps=1.0, min_samples=5)
    assert labels.shape == (150,)


def test_hdbscan_shape(toy_data: np.ndarray) -> None:
    labels = run_hdbscan(toy_data, min_cluster_size=5)
    assert labels.shape == (150,)


def test_spectral_shape(toy_data: np.ndarray) -> None:
    labels = run_spectral(toy_data, n_clusters=3)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) == 3


def test_gmm_shape(toy_data: np.ndarray) -> None:
    labels = run_gmm(toy_data, n_components=3)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) == 3


def test_spectral_size_limit() -> None:
    x = np.random.default_rng(0).standard_normal((5001, 2))
    with pytest.raises(ValueError, match="n ≤ 5000"):
        run_spectral(x)


def test_tomato_shape(toy_data: np.ndarray) -> None:
    labels = run_tomato(toy_data, n_clusters=3)
    assert labels.shape == (150,)
    assert len(np.unique(labels)) == 3


def test_tomato_circles() -> None:
    """ToMATo must cleanly separate two concentric rings."""
    from sklearn.datasets import make_circles
    from sklearn.metrics import adjusted_rand_score

    x, y = make_circles(n_samples=200, noise=0.05, factor=0.4, random_state=42)
    labels = run_tomato(x, n_clusters=2, k=10)
    assert adjusted_rand_score(y, labels) > 0.9
