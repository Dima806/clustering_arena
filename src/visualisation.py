"""Visualisation utilities: UMAP grids, ARI heatmaps, dendrograms."""

from __future__ import annotations

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_umap_grid(
    embeddings: np.ndarray,
    label_sets: dict[str, np.ndarray],
    title: str = "Cluster Comparison",
) -> Figure:
    """Plot UMAP embeddings coloured by each label set side by side."""
    import matplotlib.pyplot as plt

    n = len(label_sets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    axes_list: list[Axes] = [axes] if n == 1 else list(axes)
    for ax, (name, labels) in zip(axes_list, label_sets.items(), strict=False):
        scatter = ax.scatter(
            embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_ari_heatmap(
    ari_matrix: np.ndarray,
    algorithms: list[str],
    datasets: list[str],
) -> Figure:
    """Heatmap of ARI scores (rows = algorithms, cols = datasets)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(len(datasets) * 1.5 + 2, len(algorithms) * 0.8 + 1))
    sns.heatmap(
        ari_matrix,
        xticklabels=datasets,
        yticklabels=algorithms,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title("ARI Scores: Algorithms × Datasets")
    plt.tight_layout()
    return fig


def plot_dendrogram(
    x: np.ndarray,
    method: str = "ward",
    ax: Axes | None = None,
) -> Axes:
    """Plot dendrogram for hierarchical clustering comparison."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    z = linkage(x, method=method)
    dendrogram(z, ax=ax, no_labels=True, color_threshold=0.7 * float(z[:, 2].max()))
    ax.set_title(f"Dendrogram ({method} linkage)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Distance")
    return ax
