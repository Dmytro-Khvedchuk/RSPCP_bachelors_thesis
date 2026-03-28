"""Shared fixtures and factory functions for strategy module tests.

Provides FeatureSet builders with synthetic trending, mean-reverting,
flat, high-volatility, and low-volatility data used across all strategy
test files.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, UTC

import numpy as np
import polars as pl
import pytest

from src.app.features.domain.value_objects import FeatureSet


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_ONE_HOUR: timedelta = timedelta(hours=1)
_N_BARS: int = 100

# Column names expected by default strategy configs
_XOVER_COL: str = "ema_xover_8_21"
_HURST_COL: str = "hurst_100"
_ATR_COL: str = "atr_14"
_RV_COL: str = "rv_24"

# Thresholds used in tests
_HIGH_HURST: float = 0.7
_LOW_HURST: float = 0.3
_TYPICAL_ATR: float = 200.0
_LOW_RV: float = 0.01
_HIGH_RV: float = 1.0


# ---------------------------------------------------------------------------
# Core OHLCV builder
# ---------------------------------------------------------------------------


def _make_ohlcv_base(
    n: int,
    closes: list[float],
    highs: list[float],
    lows: list[float],
    *,
    volume: float = 1000.0,
) -> pl.DataFrame:
    """Build a minimal OHLCV Polars DataFrame from pre-computed price series.

    Args:
        n: Number of rows.
        closes: Close prices (length == n).
        highs: High prices (length == n).
        lows: Low prices (length == n).
        volume: Constant volume for all bars.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    timestamps: list[datetime] = [_BASE_TS + i * _ONE_HOUR for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [volume] * n,
        }
    )


# ---------------------------------------------------------------------------
# FeatureSet factory
# ---------------------------------------------------------------------------


def make_feature_set(
    df: pl.DataFrame,
    *,
    xover_values: list[float] | None = None,
    hurst_values: list[float] | None = None,
    atr_values: list[float] | None = None,
    rv_values: list[float] | None = None,
    extra_feature_cols: tuple[str, ...] = (),
) -> FeatureSet:
    """Build a FeatureSet by attaching synthetic indicator columns to a DataFrame.

    All required indicator columns are added (or overridden) before
    constructing the FeatureSet.  Defaults to sensible neutral values so
    that callers only need to specify the columns relevant to the strategy
    under test.

    Args:
        df: Base OHLCV DataFrame (must include timestamp, open, high, low,
            close, volume).
        xover_values: Values for the ``ema_xover_8_21`` column.  Defaults
            to all-zeros (flat crossover).
        hurst_values: Values for the ``hurst_100`` column.  Defaults to
            all-0.5 (neutral Hurst).
        atr_values: Values for the ``atr_14`` column.  Defaults to
            200.0 per bar.
        rv_values: Values for the ``rv_24`` column.  Defaults to 0.1 per bar.
        extra_feature_cols: Additional column names (already present in df)
            to include in feature_columns.

    Returns:
        Validated FeatureSet.
    """
    n: int = len(df)
    xover: list[float] = xover_values if xover_values is not None else [0.0] * n
    hurst: list[float] = hurst_values if hurst_values is not None else [0.5] * n
    atr: list[float] = atr_values if atr_values is not None else [_TYPICAL_ATR] * n
    rv: list[float] = rv_values if rv_values is not None else [0.1] * n

    enriched: pl.DataFrame = df.with_columns(
        pl.Series(_XOVER_COL, xover),
        pl.Series(_HURST_COL, hurst),
        pl.Series(_ATR_COL, atr),
        pl.Series(_RV_COL, rv),
    )

    feature_cols: tuple[str, ...] = (
        _XOVER_COL,
        _HURST_COL,
        _ATR_COL,
        _RV_COL,
        *extra_feature_cols,
    )

    return FeatureSet(
        df=enriched,
        feature_columns=feature_cols,
        target_columns=(),
        n_rows_raw=n,
        n_rows_clean=n,
    )


# ---------------------------------------------------------------------------
# Synthetic price series builders
# ---------------------------------------------------------------------------


def _trending_up_closes(n: int, *, start: float = 40_000.0, step: float = 100.0) -> list[float]:
    return [start + i * step for i in range(n)]


def _trending_down_closes(n: int, *, start: float = 40_000.0, step: float = 100.0) -> list[float]:
    return [start - i * step for i in range(n)]


def _mean_reverting_closes(n: int, *, seed: int = 42, mean: float = 40_000.0, noise: float = 200.0) -> list[float]:
    rng: np.random.Generator = np.random.default_rng(seed)
    prices: list[float] = [mean]
    for _ in range(n - 1):
        prev: float = prices[-1]
        new_price: float = prev + 0.8 * (mean - prev) + rng.normal(0.0, noise)
        prices.append(max(new_price, 1.0))
    return prices


def _flat_closes(n: int, *, price: float = 40_000.0) -> list[float]:
    return [price] * n


def _oscillating_xover(n: int, *, amplitude: float = 0.6) -> list[float]:
    """Return a sine-wave xover pattern that oscillates above and below zero."""
    return [amplitude * math.sin(2.0 * math.pi * i / 20) for i in range(n)]


# ---------------------------------------------------------------------------
# FeatureSet scenario fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_up_feature_set() -> FeatureSet:
    """Return a FeatureSet with steadily rising prices and positive EMA crossover."""
    n: int = _N_BARS
    closes: list[float] = _trending_up_closes(n)
    highs: list[float] = [c + 200.0 for c in closes]
    lows: list[float] = [c - 200.0 for c in closes]
    df: pl.DataFrame = _make_ohlcv_base(n, closes, highs, lows)
    xover: list[float] = [0.5 + 0.01 * i for i in range(n)]  # strongly positive
    return make_feature_set(df, xover_values=xover, hurst_values=[_HIGH_HURST] * n)


@pytest.fixture
def trending_down_feature_set() -> FeatureSet:
    """Return a FeatureSet with falling prices and negative EMA crossover."""
    n: int = _N_BARS
    closes: list[float] = _trending_down_closes(n)
    highs: list[float] = [c + 200.0 for c in closes]
    lows: list[float] = [max(1.0, c - 200.0) for c in closes]
    df: pl.DataFrame = _make_ohlcv_base(n, closes, highs, lows)
    xover: list[float] = [-0.5 - 0.01 * i for i in range(n)]  # strongly negative
    return make_feature_set(df, xover_values=xover, hurst_values=[_HIGH_HURST] * n)


@pytest.fixture
def mean_reverting_feature_set() -> FeatureSet:
    """Return a FeatureSet with oscillating prices and low Hurst values."""
    n: int = _N_BARS
    closes: list[float] = _mean_reverting_closes(n)
    highs: list[float] = [c + 300.0 for c in closes]
    lows: list[float] = [max(1.0, c - 300.0) for c in closes]
    df: pl.DataFrame = _make_ohlcv_base(n, closes, highs, lows)
    xover: list[float] = _oscillating_xover(n, amplitude=0.6)
    return make_feature_set(df, xover_values=xover, hurst_values=[_LOW_HURST] * n)


@pytest.fixture
def flat_feature_set() -> FeatureSet:
    """Return a FeatureSet with constant prices and zero crossover."""
    n: int = _N_BARS
    closes: list[float] = _flat_closes(n)
    highs: list[float] = [c + 200.0 for c in closes]
    lows: list[float] = [c - 200.0 for c in closes]
    df: pl.DataFrame = _make_ohlcv_base(n, closes, highs, lows)
    return make_feature_set(df, xover_values=[0.0] * n, hurst_values=[0.5] * n)


@pytest.fixture
def high_vol_feature_set() -> FeatureSet:
    """Return a FeatureSet with high realised volatility values."""
    n: int = _N_BARS
    closes: list[float] = _flat_closes(n)
    highs: list[float] = [c + 200.0 for c in closes]
    lows: list[float] = [c - 200.0 for c in closes]
    df: pl.DataFrame = _make_ohlcv_base(n, closes, highs, lows)
    return make_feature_set(df, rv_values=[_HIGH_RV] * n)


@pytest.fixture
def low_vol_feature_set() -> FeatureSet:
    """Return a FeatureSet with near-zero realised volatility values."""
    n: int = _N_BARS
    closes: list[float] = _flat_closes(n)
    highs: list[float] = [c + 200.0 for c in closes]
    lows: list[float] = [c - 200.0 for c in closes]
    df: pl.DataFrame = _make_ohlcv_base(n, closes, highs, lows)
    return make_feature_set(df, rv_values=[_LOW_RV] * n)


def _build_mixed_regime_segments(
    segment: int,
) -> tuple[list[float], list[float], list[float]]:
    """Build concatenated xover, hurst, and close lists for 3-regime data."""
    closes: list[float] = (
        _trending_up_closes(segment, start=40_000.0, step=50.0)
        + _mean_reverting_closes(segment, mean=42_000.0, noise=100.0)
        + _flat_closes(segment, price=42_000.0)
    )
    xover: list[float] = (
        [0.5 + 0.005 * i for i in range(segment)] + _oscillating_xover(segment, amplitude=0.6) + [0.0] * segment
    )
    hurst: list[float] = [_HIGH_HURST] * segment + [_LOW_HURST] * segment + [0.5] * segment
    return closes, xover, hurst


@pytest.fixture
def mixed_regime_feature_set() -> FeatureSet:
    """Return a 120-bar FeatureSet with mixed trend/mean-reversion/flat regimes.

    The first 40 bars are trending up, the next 40 are mean-reverting, and
    the final 40 are flat.  Used for signal diversity tests.
    """
    n: int = 120
    segment: int = 40
    closes, xover, hurst = _build_mixed_regime_segments(segment)

    highs: list[float] = [c + 200.0 for c in closes]
    lows: list[float] = [max(1.0, c - 200.0) for c in closes]
    df: pl.DataFrame = _make_ohlcv_base(n, closes, highs, lows)

    rv: list[float] = [0.1] * n
    return make_feature_set(df, xover_values=xover, hurst_values=hurst, rv_values=rv)
