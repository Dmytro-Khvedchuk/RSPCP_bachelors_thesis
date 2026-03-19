"""Shared fixtures and factory functions for profiling module tests.

Provides synthetic data generators for stationarity testing and
reusable configuration fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def make_stationary_series(
    n: int = 500,
    seed: int = 42,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Generate a stationary (white noise) series.

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        1-D array of i.i.d. standard normal values.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    return rng.standard_normal(n)


def make_unit_root_series(
    n: int = 500,
    seed: int = 42,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Generate a unit root (random walk) series.

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        1-D array of cumulative sum of standard normal values.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    steps: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n)
    return np.cumsum(steps)


def make_stationarity_test_df(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a Pandas DataFrame with known stationary and non-stationary features.

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DataFrame, list of feature column names).
        Contains 'stationary_feat' (white noise) and 'unit_root_feat' (random walk).
    """
    stationary: np.ndarray[tuple[int], np.dtype[np.float64]] = make_stationary_series(n, seed)
    unit_root: np.ndarray[tuple[int], np.dtype[np.float64]] = make_unit_root_series(n, seed + 1)

    df: pd.DataFrame = pd.DataFrame(
        {
            "stationary_feat": stationary,
            "unit_root_feat": unit_root,
        }
    )
    feature_names: list[str] = ["stationary_feat", "unit_root_feat"]
    return df, feature_names
