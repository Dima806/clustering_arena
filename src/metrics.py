"""Evaluation metrics for clustering quality."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def compute_ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand Index: 1.0 = perfect, 0.0 = random."""
    return float(adjusted_rand_score(y_true, y_pred))


def compute_silhouette(x: np.ndarray, y_pred: np.ndarray) -> float:
    """Silhouette Score in [-1, 1]; -1.0 when fewer than 2 non-noise clusters."""
    valid_labels = np.unique(y_pred[y_pred >= 0])
    if len(valid_labels) < 2:
        return -1.0
    mask = y_pred >= 0
    return float(silhouette_score(x[mask], y_pred[mask]))


def compute_calinski_harabasz(x: np.ndarray, y_pred: np.ndarray) -> float:
    """Calinski-Harabasz Index (higher = better); 0.0 when fewer than 2 clusters."""
    valid_labels = np.unique(y_pred[y_pred >= 0])
    if len(valid_labels) < 2:
        return 0.0
    mask = y_pred >= 0
    return float(calinski_harabasz_score(x[mask], y_pred[mask]))


def compute_davies_bouldin(x: np.ndarray, y_pred: np.ndarray) -> float:
    """Davies-Bouldin Index (lower = better); inf when fewer than 2 clusters."""
    valid_labels = np.unique(y_pred[y_pred >= 0])
    if len(valid_labels) < 2:
        return float("inf")
    mask = y_pred >= 0
    return float(davies_bouldin_score(x[mask], y_pred[mask]))


def compute_all_metrics(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute ARI, Silhouette, Calinski-Harabasz, and Davies-Bouldin."""
    return {
        "ari": compute_ari(y_true, y_pred),
        "silhouette": compute_silhouette(x, y_pred),
        "calinski_harabasz": compute_calinski_harabasz(x, y_pred),
        "davies_bouldin": compute_davies_bouldin(x, y_pred),
    }
