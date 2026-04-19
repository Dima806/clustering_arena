"""Tests for dataset loaders — only built-in datasets (no network required)."""

from __future__ import annotations

import numpy as np

from src.datasets import load_digits_subset, load_penguins, load_wine_data


def test_penguins_shape() -> None:
    x, y = load_penguins()
    assert x.ndim == 2
    assert x.shape[1] == 4
    assert len(x) == len(y)
    assert len(np.unique(y)) == 3


def test_penguins_scaled() -> None:
    x, _ = load_penguins()
    assert abs(x.mean()) < 0.1
    assert abs(x.std() - 1.0) < 0.1


def test_wine_shape() -> None:
    x, y = load_wine_data()
    assert x.shape == (178, 13)
    assert len(np.unique(y)) == 3


def test_wine_scaled() -> None:
    x, _ = load_wine_data()
    assert abs(x.mean()) < 0.1


def test_digits_subset_shape() -> None:
    x, y = load_digits_subset()
    assert x.shape[1] == 64
    assert set(np.unique(y)) == {0, 1, 2, 3, 4}


def test_digits_subset_scaled() -> None:
    x, _ = load_digits_subset()
    assert abs(x.mean()) < 0.1
