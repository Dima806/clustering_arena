"""Dataset loaders and preprocessing for all 5 clustering datasets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml, load_digits, load_wine
from sklearn.preprocessing import StandardScaler


def load_penguins() -> tuple[np.ndarray, np.ndarray]:
    """Load Palmer Penguins dataset (333 samples, 4 features, 3 species)."""
    df = sns.load_dataset("penguins").dropna()
    feature_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    x = StandardScaler().fit_transform(df[feature_cols].to_numpy())
    y = np.array(pd.Categorical(df["species"]).codes, dtype=int)
    return x, y


def load_wine_data() -> tuple[np.ndarray, np.ndarray]:
    """Load Wine dataset (178 samples, 13 features, 3 cultivars)."""
    data = load_wine()
    x = StandardScaler().fit_transform(data.data)
    y = data.target
    return x, y


def load_digits_subset() -> tuple[np.ndarray, np.ndarray]:
    """Load Digits dataset, classes 0-4 (~900 samples, 64 features)."""
    data = load_digits()
    mask = data.target < 5
    x = StandardScaler().fit_transform(data.data[mask])
    y = data.target[mask]
    return x, y


def load_credit_card(
    n_samples: int = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Credit Card Fraud dataset sampled to n_samples rows."""
    result = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
    x_df = result.data.select_dtypes(include=[np.number])
    y_raw = result.target.astype(int)
    rng = np.random.default_rng(random_state)
    n = min(n_samples, len(x_df))
    idx = rng.choice(len(x_df), size=n, replace=False)
    x = StandardScaler().fit_transform(x_df.iloc[idx].to_numpy())
    y = y_raw.iloc[idx].to_numpy()
    return x, y


def load_wholesale() -> tuple[np.ndarray, np.ndarray]:
    """Load Wholesale Customers dataset (440 samples, 6 features, 2 channels)."""
    result = fetch_openml("wholesale-customers", version=1, as_frame=True, parser="auto")
    x_df = result.data.select_dtypes(include=[np.number])
    y = result.target.astype(int).to_numpy()
    x = StandardScaler().fit_transform(x_df.to_numpy())
    return x, y


DATASETS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "penguins": load_penguins,
    "wine": load_wine_data,
    "digits": load_digits_subset,
    "credit_card": load_credit_card,
    "wholesale": load_wholesale,
}
