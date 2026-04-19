"""Tests for diagnostic utilities."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from sklearn.datasets import make_blobs

matplotlib.use("Agg")

from src.diagnostics import (
    hopkins_statistic,
    plot_elbow,
    plot_kdistance,
    plot_silhouette_scores,
)


@pytest.fixture()
def blobs() -> np.ndarray:
    x, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
    return x


def test_hopkins_clustered_data(blobs: np.ndarray) -> None:
    h = hopkins_statistic(blobs, sample_size=50)
    assert 0.0 <= h <= 1.0
    assert h > 0.6  # well-separated blobs should score high


def test_hopkins_random_data() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (200, 2))
    h = hopkins_statistic(x, sample_size=50)
    assert 0.0 <= h <= 1.0


def test_hopkins_small_sample_size(blobs: np.ndarray) -> None:
    h = hopkins_statistic(blobs, sample_size=5)
    assert 0.0 <= h <= 1.0


def test_plot_elbow_returns_axes(blobs: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    ax = plot_elbow(blobs, k_range=range(2, 5))
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_elbow_with_existing_ax(blobs: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = plot_elbow(blobs, k_range=range(2, 4), ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_silhouette_scores_returns_axes(blobs: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    ax = plot_silhouette_scores(blobs, k_range=range(2, 5))
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_silhouette_scores_with_existing_ax(blobs: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = plot_silhouette_scores(blobs, k_range=range(2, 4), ax=ax)
    assert result is ax
    plt.close("all")


def test_plot_kdistance_returns_axes(blobs: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    ax = plot_kdistance(blobs, k=5)
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_kdistance_with_existing_ax(blobs: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    result = plot_kdistance(blobs, k=3, ax=ax)
    assert result is ax
    plt.close("all")
