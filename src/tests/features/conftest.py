"""Shared fixtures and factory functions for the features module tests.

Provides OHLCV DataFrame builders, config factories, and reusable pytest
fixtures consumed by every test file in this package.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, UTC

import numpy as np
import polars as pl
import pytest

from src.app.features.domain.value_objects import (
    FeatureConfig,
    IndicatorConfig,
    TargetConfig,
    ValidationConfig,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_ONE_HOUR: timedelta = timedelta(hours=1)

# A small IndicatorConfig with shorter windows so we don't need thousands of rows.
_SMALL_HURST_WINDOW: int = 40
_SMALL_SLOPE_WINDOW: int = 5
_SMALL_OBV_SLOPE_WINDOW: int = 5
_SMALL_VOL_WINDOW: int = 5
_SMALL_AMIHUD_WINDOW: int = 5
_SMALL_RV_WINDOW_1: int = 5
_SMALL_RV_WINDOW_2: int = 10
_SMALL_BOLLINGER_WINDOW: int = 10
_SMALL_ZSCORE_WINDOW: int = 5
_SMALL_GK_WINDOW: int = 5
_SMALL_PARK_WINDOW: int = 5
_SMALL_ATR_PERIOD: int = 5


# ---------------------------------------------------------------------------
# OHLCV DataFrame factory helpers
# ---------------------------------------------------------------------------


def make_ohlcv_df(
    n: int,
    *,
    price_start: float = 100.0,
    price_step: float = 0.0,
    volume: float = 1000.0,
    high_offset: float = 2.0,
    low_offset: float = 2.0,
    open_offset: float = 0.0,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_HOUR,
) -> pl.DataFrame:
    """Build a minimal OHLCV Polars DataFrame.

    Args:
        n: Number of rows.
        price_start: Starting close price.
        price_step: Price increment per row (0 = flat, >0 = uptrend).
        volume: Constant volume per row.
        high_offset: How much above close the high is.
        low_offset: How much below close the low is.
        open_offset: How much above close the open is (negative = open below close).
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    closes: list[float] = [price_start + i * price_step for i in range(n)]
    opens: list[float] = [c + open_offset for c in closes]
    highs: list[float] = [c + high_offset for c in closes]
    lows: list[float] = [c - low_offset for c in closes]
    volumes: list[float] = [volume] * n

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def make_trending_df(
    n: int,
    *,
    direction: float = 1.0,
    price_start: float = 100.0,
    step: float = 1.0,
    volume: float = 1000.0,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_HOUR,
) -> pl.DataFrame:
    """Build a trending OHLCV DataFrame (strictly monotonic prices).

    Args:
        n: Number of rows.
        direction: +1 for uptrend, -1 for downtrend.
        price_start: Starting close price.
        step: Absolute price change per bar.
        volume: Constant volume.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.

    Returns:
        DataFrame with monotonically moving close prices.
    """
    return make_ohlcv_df(
        n,
        price_start=price_start,
        price_step=direction * step,
        volume=volume,
        high_offset=0.5,
        low_offset=0.5,
        base_ts=base_ts,
        interval=interval,
    )


def make_random_walk_df(
    n: int,
    *,
    seed: int = 42,
    price_start: float = 1000.0,
    volatility: float = 5.0,
    volume: float = 1000.0,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_HOUR,
) -> pl.DataFrame:
    """Build a random-walk OHLCV DataFrame.

    Args:
        n: Number of rows.
        seed: NumPy random seed for reproducibility.
        price_start: Starting close price.
        volatility: Per-step standard deviation of returns.
        volume: Constant volume.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.

    Returns:
        DataFrame with random-walk close prices.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    steps: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.normal(0, volatility, size=n)
    closes_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = np.cumsum(steps) + price_start
    closes_arr = np.maximum(closes_arr, 1.0)  # keep prices positive

    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    highs: list[float] = (closes_arr + abs(rng.normal(0, 1, size=n))).tolist()
    lows: list[float] = (closes_arr - abs(rng.normal(0, 1, size=n))).tolist()
    opens: list[float] = closes_arr.tolist()
    closes: list[float] = closes_arr.tolist()
    lows = [min(lo, o, c) for lo, o, c in zip(lows, opens, closes, strict=True)]
    highs = [max(hi, o, c) for hi, o, c in zip(highs, opens, closes, strict=True)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [volume] * n,
        }
    )


def make_mean_reverting_df(
    n: int,
    *,
    seed: int = 42,
    mean: float = 100.0,
    reversion_speed: float = 0.8,
    noise: float = 0.5,
    volume: float = 1000.0,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_HOUR,
) -> pl.DataFrame:
    """Build a mean-reverting OHLCV DataFrame (Ornstein-Uhlenbeck process).

    Args:
        n: Number of rows.
        seed: NumPy random seed.
        mean: Long-run mean price level.
        reversion_speed: Speed of mean reversion (0-1; higher = faster).
        noise: Standard deviation of Gaussian noise.
        volume: Constant volume.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.

    Returns:
        DataFrame with mean-reverting close prices.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    prices: list[float] = [mean]
    for _ in range(n - 1):
        prev: float = prices[-1]
        new_price: float = prev + reversion_speed * (mean - prev) + rng.normal(0, noise)
        prices.append(max(new_price, 1.0))

    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    prices_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(prices)
    highs: list[float] = (prices_arr + 0.3).tolist()
    lows: list[float] = (prices_arr - 0.3).tolist()

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": [volume] * n,
        }
    )


# ---------------------------------------------------------------------------
# Config factories
# ---------------------------------------------------------------------------


def make_small_indicator_config(**overrides: object) -> IndicatorConfig:
    """Build an IndicatorConfig with small windows suitable for short DataFrames.

    Uses windows small enough to work with ~150-row DataFrames without
    requiring hundreds of warmup rows.

    Args:
        **overrides: Additional keyword arguments forwarded to IndicatorConfig.

    Returns:
        IndicatorConfig with small rolling windows.
    """
    defaults: dict[str, object] = {
        "return_horizons": (1, 4),
        "realized_vol_windows": (_SMALL_RV_WINDOW_1, _SMALL_RV_WINDOW_2),
        "garman_klass_window": _SMALL_GK_WINDOW,
        "parkinson_window": _SMALL_PARK_WINDOW,
        "atr_period": _SMALL_ATR_PERIOD,
        "ema_fast_span": 5,
        "ema_slow_span": 10,
        "rsi_period": 7,
        "roc_periods": (1, 4),
        "slope_window": _SMALL_SLOPE_WINDOW,
        "volume_zscore_window": _SMALL_VOL_WINDOW,
        "obv_slope_window": _SMALL_OBV_SLOPE_WINDOW,
        "amihud_window": _SMALL_AMIHUD_WINDOW,
        "hurst_window": _SMALL_HURST_WINDOW,
        "return_zscore_window": _SMALL_ZSCORE_WINDOW,
        "bollinger_window": _SMALL_BOLLINGER_WINDOW,
        "bollinger_num_std": 2.0,
        "clip_lower": -5.0,
        "clip_upper": 5.0,
    }
    defaults.update(overrides)
    return IndicatorConfig(**defaults)  # type: ignore[arg-type]


def make_small_target_config(**overrides: object) -> TargetConfig:
    """Build a TargetConfig with small horizons suitable for short DataFrames.

    Args:
        **overrides: Additional keyword arguments forwarded to TargetConfig.

    Returns:
        TargetConfig with horizons (1, 4) for returns, (2, 4) for vol,
        (1, 4) for z-returns, backward_vol_window=5, and winsorize=False
        by default (to keep existing tests deterministic).
    """
    defaults: dict[str, object] = {
        "forward_return_horizons": (1, 4),
        "forward_vol_horizons": (2, 4),
        "forward_zret_horizons": (1, 4),
        "forward_direction_horizons": (1, 4),
        "backward_vol_window": 5,
        "winsorize": False,
    }
    defaults.update(overrides)
    return TargetConfig(**defaults)  # type: ignore[arg-type]


def make_small_feature_config(**overrides: object) -> FeatureConfig:
    """Build a FeatureConfig with small windows for short DataFrames.

    Args:
        **overrides: Additional keyword arguments forwarded to FeatureConfig.

    Returns:
        FeatureConfig with small indicator and target windows.
    """
    indicator_config: IndicatorConfig = make_small_indicator_config()
    target_config: TargetConfig = make_small_target_config()
    defaults: dict[str, object] = {
        "indicator_config": indicator_config,
        "target_config": target_config,
        "drop_na": True,
        "compute_targets": True,
    }
    defaults.update(overrides)
    return FeatureConfig(**defaults)  # type: ignore[arg-type]


def make_fast_validation_config(**overrides: object) -> ValidationConfig:
    """Build a ValidationConfig with minimal permutations for fast tests.

    Args:
        **overrides: Additional keyword arguments forwarded to ValidationConfig.

    Returns:
        ValidationConfig with n_permutations_mi=10, n_permutations_ridge=10.
    """
    defaults: dict[str, object] = {
        "n_permutations_mi": 100,
        "n_permutations_ridge": 50,
        "n_permutations_stability": 50,
        "alpha": 0.05,
        "stability_threshold": 0.5,
        "target_col": "fwd_logret_1",
        "timestamp_col": "timestamp",
        "temporal_windows": ((2020, 2021), (2021, 2022)),
        "min_window_rows": 10,
        "ridge_alpha": 1.0,
        "random_seed": 42,
        "min_features_kept": 2,
        "min_valid_windows": 1,
        "min_group_features": 2,
        "redundancy_tolerance": 1.1,
    }
    defaults.update(overrides)
    return ValidationConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_df() -> pl.DataFrame:
    """Return a 150-row random-walk OHLCV DataFrame."""
    return make_random_walk_df(150, seed=0)


@pytest.fixture
def large_ohlcv_df() -> pl.DataFrame:
    """Return a 500-row random-walk OHLCV DataFrame for property tests."""
    return make_random_walk_df(500, seed=7)


@pytest.fixture
def default_indicator_config() -> IndicatorConfig:
    """Return a small-window IndicatorConfig suitable for 150-row DataFrames."""
    return make_small_indicator_config()


@pytest.fixture
def default_target_config() -> TargetConfig:
    """Return a TargetConfig with small horizons."""
    return make_small_target_config()


@pytest.fixture
def default_feature_config() -> FeatureConfig:
    """Return a FeatureConfig with small windows for 150-row DataFrames."""
    return make_small_feature_config()


@pytest.fixture
def fast_validation_config() -> ValidationConfig:
    """Return a ValidationConfig with minimal permutations for fast tests."""
    return make_fast_validation_config()


# ---------------------------------------------------------------------------
# Known-value helpers (used in unit tests)
# ---------------------------------------------------------------------------


def _ln2() -> float:
    """Return the natural log of 2."""
    return math.log(2.0)
