"""Tests for visualisation utilities."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from sklearn.datasets import make_blobs

matplotlib.use("Agg")

from src.visualisation import plot_ari_heatmap, plot_dendrogram, plot_umap_grid


@pytest.fixture()
def blobs() -> tuple[np.ndarray, np.ndarray]:
    return make_blobs(n_samples=80, centers=3, cluster_std=0.5, random_state=42)


def test_plot_umap_grid_single_panel() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((50, 2))
    labels = rng.integers(0, 3, 50)
    fig = plot_umap_grid(emb, {"kmeans": labels})
    assert isinstance(fig, Figure)
    plt.close("all")


def test_plot_umap_grid_multiple_panels() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((50, 2))
    labels = rng.integers(0, 3, 50)
    fig = plot_umap_grid(emb, {"kmeans": labels, "hdbscan": labels, "ward": labels})
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 3
    plt.close("all")


def test_plot_ari_heatmap_shape() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    matrix = np.array([[0.8, 0.3, 0.5], [0.6, 0.7, 0.4]])
    fig = plot_ari_heatmap(matrix, algorithms=["kmeans", "ward"], datasets=["a", "b", "c"])
    assert isinstance(fig, Figure)
    plt.close("all")


def test_plot_ari_heatmap_single_cell() -> None:
    import matplotlib.pyplot as plt

    matrix = np.array([[0.75]])
    fig = plot_ari_heatmap(matrix, algorithms=["kmeans"], datasets=["penguins"])
    assert fig is not None
    plt.close("all")


def test_plot_dendrogram_returns_axes(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    x, _ = blobs
    ax = plot_dendrogram(x, method="ward")
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_dendrogram_complete_linkage(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    x, _ = blobs
    ax = plot_dendrogram(x, method="complete")
    assert isinstance(ax, Axes)
    plt.close("all")


def test_plot_dendrogram_with_existing_ax(blobs: tuple[np.ndarray, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    x, _ = blobs
    _, ax = plt.subplots()
    result = plot_dendrogram(x, method="average", ax=ax)
    assert result is ax
    plt.close("all")
