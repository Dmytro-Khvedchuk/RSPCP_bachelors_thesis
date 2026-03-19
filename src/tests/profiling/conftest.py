"""Shared fixtures and factory functions for profiling module tests.

Provides synthetic data generators for stationarity testing,
distribution analysis, and reusable configuration fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]


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


# ---------------------------------------------------------------------------
# Distribution analysis helpers
# ---------------------------------------------------------------------------


def make_normal_returns(
    n: int = 5000,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate normally distributed returns with realistic crypto-scale volatility.

    Draws from ``N(0, 0.01)`` (mean=0, std=1% per bar).

    Args:
        n: Number of return observations.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of i.i.d. Normal returns.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=0.01, size=n)
    return pd.Series(data, dtype=np.float64, name="log_return")


def make_student_t_returns(
    n: int = 5000,
    nu: float = 5.0,
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate Student-t distributed returns with given degrees of freedom.

    Draws from ``t(nu)`` scaled to realistic return magnitude (scale=0.01).

    Args:
        n: Number of return observations.
        nu: Degrees of freedom (lower = heavier tails).
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of Student-t returns.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_t(df=nu, size=n) * 0.01
    return pd.Series(data, dtype=np.float64, name="log_return")


def make_skewed_returns(
    n: int = 5000,
    skew_direction: str = "left",
    seed: int = 42,
) -> pd.Series:  # type: ignore[type-arg]
    """Generate skewed returns for testing skewness detection.

    Uses ``scipy.stats.skewnorm`` to produce left- or right-skewed samples.

    Args:
        n: Number of return observations.
        skew_direction: ``"left"`` for negative skew, ``"right"`` for positive.
        seed: Random seed for reproducibility.

    Returns:
        Pandas Series of skewed returns.

    Raises:
        ValueError: If *skew_direction* is not ``"left"`` or ``"right"``.
    """
    if skew_direction not in {"left", "right"}:
        msg = f"skew_direction must be 'left' or 'right', got '{skew_direction}'"
        raise ValueError(msg)

    a = -5.0 if skew_direction == "left" else 5.0
    data = stats.skewnorm.rvs(a, size=n, random_state=seed) * 0.01
    return pd.Series(data, dtype=np.float64, name="log_return")
