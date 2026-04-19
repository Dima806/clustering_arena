"""Tests for clustering evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from src.metrics import (
    compute_all_metrics,
    compute_ari,
    compute_calinski_harabasz,
    compute_davies_bouldin,
    compute_silhouette,
)


@pytest.fixture()
def clustered_data() -> tuple[np.ndarray, np.ndarray]:
    x, y = make_blobs(n_samples=200, centers=3, cluster_std=0.5, random_state=42)
    return x, y


def test_ari_perfect(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    _, y = clustered_data
    assert compute_ari(y, y) == pytest.approx(1.0)


def test_ari_random(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    _, y = clustered_data
    random_labels = np.zeros(len(y), dtype=int)
    assert compute_ari(y, random_labels) < 0.5


def test_silhouette_range(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = clustered_data
    score = compute_silhouette(x, y)
    assert -1.0 <= score <= 1.0


def test_silhouette_single_cluster(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = clustered_data
    labels = np.zeros(len(x), dtype=int)
    assert compute_silhouette(x, labels) == -1.0


def test_silhouette_all_noise(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = clustered_data
    labels = np.full(len(x), -1, dtype=int)
    assert compute_silhouette(x, labels) == -1.0


def test_calinski_harabasz_positive(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = clustered_data
    assert compute_calinski_harabasz(x, y) > 0


def test_calinski_harabasz_single_cluster(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = clustered_data
    labels = np.zeros(len(x), dtype=int)
    assert compute_calinski_harabasz(x, labels) == 0.0


def test_davies_bouldin_positive(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = clustered_data
    assert compute_davies_bouldin(x, y) > 0


def test_davies_bouldin_single_cluster(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, _ = clustered_data
    labels = np.zeros(len(x), dtype=int)
    assert compute_davies_bouldin(x, labels) == float("inf")


def test_compute_all_metrics_keys(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = clustered_data
    metrics = compute_all_metrics(x, y, y)
    assert set(metrics.keys()) == {"ari", "silhouette", "calinski_harabasz", "davies_bouldin"}


def test_compute_all_metrics_perfect(clustered_data: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = clustered_data
    metrics = compute_all_metrics(x, y, y)
    assert metrics["ari"] == pytest.approx(1.0)
