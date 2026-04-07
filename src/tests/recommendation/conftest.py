"""Shared fixtures and factory functions for recommendation module tests.

Provides synthetic bar DataFrames, signal DataFrames, classifier/regressor
outputs, and helper builders used across label builder and feature builder
test modules.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import polars as pl
import pytest

from src.app.recommendation.application.label_builder import LabelConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
ONE_HOUR: timedelta = timedelta(hours=1)

ASSET_SYMBOL: str = "BTCUSDT"
STRATEGY_NAME: str = "momentum_crossover"


# ---------------------------------------------------------------------------
# Bar factory
# ---------------------------------------------------------------------------


def make_bars(
    n: int,
    *,
    start_price: float = 40_000.0,
    start_time: datetime = BASE_TS,
    interval: timedelta = ONE_HOUR,
    price_step: float = 100.0,
    volume: float = 10.0,
) -> pl.DataFrame:
    """Build a minimal OHLCV Polars DataFrame.

    Prices increase linearly by ``price_step`` per bar to create
    a deterministic uptrend for predictable label computation.
    """
    timestamps = [start_time + i * interval for i in range(n)]
    closes = [start_price + i * price_step for i in range(n)]
    opens = [start_price + i * price_step for i in range(n)]
    highs = [p + 200.0 for p in closes]
    lows = [max(1.0, p - 200.0) for p in closes]
    volumes = [volume] * n

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


# ---------------------------------------------------------------------------
# Signal factory
# ---------------------------------------------------------------------------


def make_signals(
    n: int,
    *,
    sides: list[str] | None = None,
    start_time: datetime = BASE_TS,
    interval: timedelta = ONE_HOUR,
) -> pl.DataFrame:
    """Build a strategy signals DataFrame.

    If ``sides`` is not provided, all signals default to ``"long"``.
    """
    timestamps = [start_time + i * interval for i in range(n)]
    if sides is None:
        sides = ["long"] * n
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "side": sides,
        }
    )


# ---------------------------------------------------------------------------
# Classifier output factory
# ---------------------------------------------------------------------------


def make_classifier_outputs(
    n: int,
    *,
    directions: list[int] | None = None,
    confidences: list[float] | None = None,
    correct: list[bool] | None = None,
    start_time: datetime = BASE_TS,
    interval: timedelta = ONE_HOUR,
) -> pl.DataFrame:
    """Build a classifier output DataFrame."""
    timestamps = [start_time + i * interval for i in range(n)]
    if directions is None:
        directions = [1] * n
    if confidences is None:
        confidences = [0.7] * n

    data = {
        "timestamp": timestamps,
        "clf_direction": directions,
        "clf_confidence": confidences,
    }
    if correct is not None:
        data["clf_correct"] = correct

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Regressor output factory
# ---------------------------------------------------------------------------


def make_regressor_outputs(
    n: int,
    *,
    predicted_returns: list[float] | None = None,
    prediction_stds: list[float] | None = None,
    start_time: datetime = BASE_TS,
    interval: timedelta = ONE_HOUR,
) -> pl.DataFrame:
    """Build a regressor output DataFrame."""
    timestamps = [start_time + i * interval for i in range(n)]
    if predicted_returns is None:
        predicted_returns = [0.005] * n
    if prediction_stds is None:
        prediction_stds = [0.01] * n

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "reg_predicted_return": predicted_returns,
            "reg_prediction_std": prediction_stds,
        }
    )


# ---------------------------------------------------------------------------
# Vol forecast factory
# ---------------------------------------------------------------------------


def make_vol_forecasts(
    n: int,
    *,
    vol_predicted: list[float] | None = None,
    vol_actual: list[float] | None = None,
    start_time: datetime = BASE_TS,
    interval: timedelta = ONE_HOUR,
) -> pl.DataFrame:
    """Build a volatility forecast DataFrame."""
    timestamps = [start_time + i * interval for i in range(n)]
    if vol_predicted is None:
        vol_predicted = [0.02] * n

    data: dict[str, list[object]] = {
        "timestamp": timestamps,
        "vol_predicted": vol_predicted,
    }
    if vol_actual is not None:
        data["vol_actual"] = vol_actual

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Strategy returns factory
# ---------------------------------------------------------------------------


def make_strategy_returns(
    n: int,
    *,
    returns: list[float] | None = None,
    start_time: datetime = BASE_TS,
    interval: timedelta = ONE_HOUR,
) -> pl.DataFrame:
    """Build a historical strategy returns DataFrame."""
    timestamps = [start_time + i * interval for i in range(n)]
    if returns is None:
        returns = [0.001 * (i % 3 - 1) for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_return": returns,
        }
    )


# ---------------------------------------------------------------------------
# Market features factory
# ---------------------------------------------------------------------------


def make_market_features(
    n: int,
    *,
    start_time: datetime = BASE_TS,
    interval: timedelta = ONE_HOUR,
    start_price: float = 40_000.0,
    price_step: float = 100.0,
) -> pl.DataFrame:
    """Build a minimal market features DataFrame."""
    timestamps = [start_time + i * interval for i in range(n)]
    closes = [start_price + i * price_step for i in range(n)]
    volatilities = [0.02 + 0.001 * i for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "close": closes,
            "volatility": volatilities,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_label_config() -> LabelConfig:
    """Return a default LabelConfig with horizon=7 and 10 bps commission."""
    return LabelConfig()


@pytest.fixture
def bars_20() -> pl.DataFrame:
    """Return 20 bars with linearly increasing prices."""
    return make_bars(20)


@pytest.fixture
def long_signals_20() -> pl.DataFrame:
    """Return 20 all-long signals."""
    return make_signals(20)


@pytest.fixture
def market_features_20() -> pl.DataFrame:
    """Return 20 rows of market features."""
    return make_market_features(20)
