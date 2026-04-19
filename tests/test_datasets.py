"""Tests for dataset loaders — only built-in datasets (no network required)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.datasets import (
    load_credit_card,
    load_digits_subset,
    load_penguins,
    load_wholesale,
    load_wine_data,
)


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


def _make_openml_result(n: int, n_features: int, n_classes: int) -> SimpleNamespace:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        rng.uniform(0, 10, (n, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    target = pd.Series(rng.integers(0, n_classes, n).astype(str))
    return SimpleNamespace(data=data, target=target)


def test_credit_card_shape() -> None:
    fake = _make_openml_result(200, 17, 2)
    with patch("src.datasets.fetch_openml", return_value=fake):
        x, y = load_credit_card(n_samples=100)
    assert x.shape == (100, 17)
    assert len(y) == 100
    assert abs(x.mean()) < 0.5  # scaled


def test_credit_card_respects_n_samples() -> None:
    fake = _make_openml_result(50, 17, 2)
    with patch("src.datasets.fetch_openml", return_value=fake):
        x, y = load_credit_card(n_samples=200)  # capped at 50
    assert len(x) == 50


def test_wholesale_shape() -> None:
    fake = _make_openml_result(440, 6, 2)
    with patch("src.datasets.fetch_openml", return_value=fake):
        x, y = load_wholesale()
    assert x.shape == (440, 6)
    assert len(y) == 440
    assert abs(x.mean()) < 0.5  # scaled
