"""Diagnostic tools: Hopkins statistic, elbow curve, silhouette plot, k-distance."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from sklearn.neighbors import NearestNeighbors


def hopkins_statistic(
    x: np.ndarray,
    sample_size: int = 100,
    random_state: int = 42,
) -> float:
    """Hopkins statistic for clusterability: ~0.5 = random, ~1.0 = strongly clustered."""
    rng = np.random.default_rng(random_state)
    n, d = x.shape
    m = min(sample_size, n - 1)

    idx = rng.choice(n, m, replace=False)
    x_sample = x[idx]
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    x_random = rng.uniform(mins, maxs, (m, d))

    nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree")
    nn.fit(x)

    u_dist, _ = nn.kneighbors(x_random, n_neighbors=1)
    w_dist, _ = nn.kneighbors(x_sample, n_neighbors=2)
    w_dist = w_dist[:, 1]

    u = float(np.sum(u_dist**d))
    w = float(np.sum(w_dist**d))
    return u / (u + w) if (u + w) > 0 else 0.5


def plot_elbow(
    x: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
    ax: Axes | None = None,
) -> Axes:
    """Plot KMeans inertia vs k (elbow curve)."""
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    if k_range is None:
        k_range = range(2, 11)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    inertias = [
        KMeans(n_clusters=k, random_state=random_state, n_init=3).fit(x).inertia_ for k in k_range
    ]
    ax.plot(list(k_range), inertias, "bo-")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Curve")
    return ax


def plot_silhouette_scores(
    x: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
    ax: Axes | None = None,
) -> Axes:
    """Plot silhouette score vs k for KMeans."""
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if k_range is None:
        k_range = range(2, 11)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    scores = [
        silhouette_score(
            x, KMeans(n_clusters=k, random_state=random_state, n_init=3).fit_predict(x)
        )
        for k in k_range
    ]
    ax.plot(list(k_range), scores, "go-")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Scores vs k")
    return ax


def plot_kdistance(
    x: np.ndarray,
    k: int = 5,
    ax: Axes | None = None,
) -> Axes:
    """Plot k-distance graph to guide DBSCAN eps selection."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x)
    distances, _ = nn.kneighbors(x)
    k_distances = np.sort(distances[:, -1])[::-1]

    ax.plot(k_distances)
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"{k}-NN distance")
    ax.set_title(f"k-Distance Plot (k={k}) — elbow ≈ good eps")
    return ax
