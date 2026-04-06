"""Shared fixtures and factory functions for the forecasting module tests.

Provides config factories, synthetic data generators, and reusable pytest
fixtures consumed by every test file in this package.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.app.forecasting.domain.value_objects import (
    GARCHConfig,
    GradientBoostingConfig,
    GRUConfig,
    HARRVConfig,
    RidgeConfig,
)


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def make_ridge_config(**overrides: object) -> RidgeConfig:
    """Build a RidgeConfig with sensible test defaults.

    Args:
        **overrides: Keyword arguments forwarded to RidgeConfig.

    Returns:
        Configured RidgeConfig instance.
    """
    defaults: dict[str, object] = {
        "alpha": 1.0,
        "use_huber": False,
        "huber_epsilon": 1.35,
        "random_seed": 42,
    }
    defaults.update(overrides)
    return RidgeConfig(**defaults)  # type: ignore[arg-type]


def make_gb_config(**overrides: object) -> GradientBoostingConfig:
    """Build a GradientBoostingConfig with small n_estimators for speed.

    Args:
        **overrides: Keyword arguments forwarded to GradientBoostingConfig.

    Returns:
        Configured GradientBoostingConfig instance.
    """
    defaults: dict[str, object] = {
        "quantiles": (0.05, 0.25, 0.50, 0.75, 0.95),
        "n_estimators": 20,
        "learning_rate": 0.1,
        "max_depth": 4,
        "min_child_samples": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "apply_isotonic": True,
        "random_seed": 42,
    }
    defaults.update(overrides)
    return GradientBoostingConfig(**defaults)  # type: ignore[arg-type]


def make_gru_config(**overrides: object) -> GRUConfig:
    """Build a GRUConfig with small parameters for fast tests.

    Args:
        **overrides: Keyword arguments forwarded to GRUConfig.

    Returns:
        Configured GRUConfig instance.
    """
    defaults: dict[str, object] = {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.2,
        "sequence_length": 5,
        "learning_rate": 1e-3,
        "n_epochs": 5,
        "batch_size": 8,
        "mc_samples": 3,
        "patience": 3,
        "random_seed": 42,
    }
    defaults.update(overrides)
    return GRUConfig(**defaults)  # type: ignore[arg-type]


def make_har_config(**overrides: object) -> HARRVConfig:
    """Build a HARRVConfig with small lags for fast tests.

    Args:
        **overrides: Keyword arguments forwarded to HARRVConfig.

    Returns:
        Configured HARRVConfig instance.
    """
    defaults: dict[str, object] = {
        "daily_lag": 1,
        "weekly_lag": 3,
        "monthly_lag": 5,
        "fit_intercept": True,
    }
    defaults.update(overrides)
    return HARRVConfig(**defaults)  # type: ignore[arg-type]


def make_garch_config(**overrides: object) -> GARCHConfig:
    """Build a GARCHConfig with test defaults.

    Args:
        **overrides: Keyword arguments forwarded to GARCHConfig.

    Returns:
        Configured GARCHConfig instance.
    """
    defaults: dict[str, object] = {
        "p": 1,
        "q": 1,
        "mean_model": "Constant",
        "ar_order": 1,
        "dist": "t",
        "rescale": True,
    }
    defaults.update(overrides)
    return GARCHConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def make_linear_data(
    n: int = 200,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Generate linear data y = X @ w + noise for regression tests.

    Args:
        n: Number of samples.
        n_features: Number of features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X has shape (n, n_features) and y has shape (n,).
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = rng.standard_normal((n, n_features)).astype(np.float64)
    w: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n_features).astype(np.float64)
    noise: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.normal(0, 0.1, size=n).astype(np.float64)
    y: np.ndarray[tuple[int], np.dtype[np.float64]] = (x @ w + noise).astype(np.float64)
    return x, y


def make_rv_series(
    n: int = 200,
    seed: int = 42,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Generate a synthetic realized volatility series with persistence.

    Creates an AR(1)-like series of positive RV values to simulate
    persistent volatility clustering observed in financial data.

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        1-D array of positive RV-like values with shape (n,).
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    rv: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(n, dtype=np.float64)
    rv[0] = 0.02
    for i in range(1, n):
        # AR(1) with strong persistence + positive noise
        rv[i] = 0.005 + 0.85 * rv[i - 1] + 0.005 * abs(rng.standard_normal())
    return rv


def make_garch_returns(
    n: int = 500,
    seed: int = 42,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Generate synthetic returns with known GARCH(1,1) dynamics.

    Uses the ``arch`` library to simulate a GARCH(1,1) process with
    persistent volatility (omega=0.01, alpha1=0.05, beta1=0.90).

    Args:
        n: Number of observations.
        seed: Random seed for reproducibility.

    Returns:
        1-D array of returns with shape (n,).
    """
    from arch import arch_model

    np.random.seed(seed)  # noqa: NPY002
    am = arch_model(None, mean="Zero", vol="GARCH", p=1, q=1)
    params: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array([0.01, 0.05, 0.90])
    sim = am.simulate(params, nobs=n)
    returns: np.ndarray[tuple[int], np.dtype[np.float64]] = sim["data"].values.astype(np.float64)
    return returns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ridge_config() -> RidgeConfig:
    """Return a default RidgeConfig for tests."""
    return make_ridge_config()


@pytest.fixture
def gb_config() -> GradientBoostingConfig:
    """Return a default GradientBoostingConfig with small n_estimators."""
    return make_gb_config()


@pytest.fixture
def gru_config() -> GRUConfig:
    """Return a default GRUConfig with small parameters."""
    return make_gru_config()


@pytest.fixture
def har_config() -> HARRVConfig:
    """Return a default HARRVConfig with small lags."""
    return make_har_config()


@pytest.fixture
def garch_config() -> GARCHConfig:
    """Return a default GARCHConfig."""
    return make_garch_config()


@pytest.fixture
def linear_data() -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """Return (X, y) linear data for regression tests."""
    return make_linear_data()


@pytest.fixture
def rv_series() -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Return a synthetic persistent RV series."""
    return make_rv_series()


@pytest.fixture
def garch_returns() -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Return synthetic GARCH returns."""
    return make_garch_returns()
