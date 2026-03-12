"""Shared fixtures for research module tests.

Provides synthetic OHLCV data, known-distribution return series,
in-memory DuckDB infrastructure, and synthetic bar DataFrames.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.app.system.database.connection import ConnectionManager
from src.app.system.database.settings import DatabaseSettings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_RNG_SEED: int = 42


# ---------------------------------------------------------------------------
# SQL DDL for in-memory tables
# ---------------------------------------------------------------------------

_CREATE_OHLCV_TABLE: str = """
CREATE TABLE IF NOT EXISTS ohlcv (
    asset       VARCHAR        NOT NULL,
    timeframe   VARCHAR        NOT NULL,
    timestamp   TIMESTAMPTZ    NOT NULL,
    open        DECIMAL(18, 8) NOT NULL,
    high        DECIMAL(18, 8) NOT NULL,
    low         DECIMAL(18, 8) NOT NULL,
    close       DECIMAL(18, 8) NOT NULL,
    volume      DOUBLE         NOT NULL,
    PRIMARY KEY (asset, timeframe, timestamp)
);
"""

_CREATE_BARS_TABLE: str = """
CREATE TABLE IF NOT EXISTS aggregated_bars (
    asset           VARCHAR        NOT NULL,
    bar_type        VARCHAR        NOT NULL,
    bar_config_hash VARCHAR(16)    NOT NULL,
    start_ts        TIMESTAMPTZ    NOT NULL,
    end_ts          TIMESTAMPTZ    NOT NULL,
    open            DECIMAL(18, 8) NOT NULL,
    high            DECIMAL(18, 8) NOT NULL,
    low             DECIMAL(18, 8) NOT NULL,
    close           DECIMAL(18, 8) NOT NULL,
    volume          DOUBLE         NOT NULL,
    tick_count      INTEGER        NOT NULL,
    buy_volume      DOUBLE         NOT NULL,
    sell_volume     DOUBLE         NOT NULL,
    vwap            DECIMAL(18, 8) NOT NULL,
    PRIMARY KEY (asset, bar_type, bar_config_hash, start_ts)
);
"""


# ---------------------------------------------------------------------------
# Synthetic OHLCV DataFrame
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    """Return a 1000-row Pandas DataFrame with realistic BTC OHLCV data.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    n: int = 1000
    timestamps: list[datetime] = [_BASE_TS + timedelta(hours=i) for i in range(n)]

    # GBM-like price path
    log_returns: np.ndarray = rng.normal(0.0001, 0.02, size=n)
    prices: np.ndarray = 42000.0 * np.exp(np.cumsum(log_returns))

    # Realistic OHLCV
    high_offset: np.ndarray = np.abs(rng.normal(0, 100, size=n))
    low_offset: np.ndarray = np.abs(rng.normal(0, 100, size=n))
    volumes: np.ndarray = np.abs(rng.normal(100, 30, size=n))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices + high_offset,
            "low": prices - low_offset,
            "close": prices * (1 + rng.normal(0, 0.001, size=n)),
            "volume": volumes,
        }
    )


# ---------------------------------------------------------------------------
# Known-distribution return series
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_returns() -> pd.Series:
    """Return N(0, 0.01) series — known normal distribution.

    Returns:
        500-element Series of normally distributed returns.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    values: np.ndarray = rng.normal(0, 0.01, size=500)
    return pd.Series(values, name="returns")


@pytest.fixture
def fat_tail_returns() -> pd.Series:
    """Return Student-t(3) series — known fat-tailed distribution.

    Returns:
        500-element Series of t-distributed returns (excess kurtosis > 0).
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    values: np.ndarray = rng.standard_t(df=3, size=500) * 0.01
    return pd.Series(values, name="returns")


@pytest.fixture
def ar1_returns() -> pd.Series:
    """Return AR(1) series with phi=0.3 — known serial correlation.

    Returns:
        500-element Series with AR(1) structure.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    n: int = 500
    phi: float = 0.3
    values: np.ndarray = np.zeros(n)
    noise: np.ndarray = rng.normal(0, 0.01, size=n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + noise[i]
    return pd.Series(values, name="returns")


@pytest.fixture
def white_noise_returns() -> pd.Series:
    """Return iid N(0, 0.01) series — no serial correlation.

    Returns:
        500-element Series of iid noise.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    values: np.ndarray = rng.normal(0, 0.01, size=500)
    return pd.Series(values, name="returns")


# ---------------------------------------------------------------------------
# In-memory DuckDB ConnectionManager
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_connection_manager() -> Generator[ConnectionManager]:
    """Create an in-memory DuckDB with both ohlcv and aggregated_bars tables.

    Yields:
        Initialised ConnectionManager pointing at :memory:.
    """
    settings: DatabaseSettings = DatabaseSettings.model_construct(
        path=":memory:",
        read_only=False,
        memory_limit="1GB",
        threads=1,
    )
    cm: ConnectionManager = ConnectionManager(settings=settings)
    cm.initialize()

    from sqlalchemy import text

    with cm.connect() as conn:
        conn.execute(text(_CREATE_OHLCV_TABLE))
        conn.execute(text(_CREATE_BARS_TABLE))
        conn.commit()

    yield cm
    cm.dispose()


# ---------------------------------------------------------------------------
# Synthetic bars DataFrame
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_bars_df() -> pd.DataFrame:
    """Return a 200-row Pandas DataFrame with bar data of varying durations.

    Returns:
        DataFrame with columns matching the aggregated_bars table:
        start_ts, end_ts, open, high, low, close, volume,
        tick_count, buy_volume, sell_volume, vwap.
    """
    rng: np.random.Generator = np.random.default_rng(_RNG_SEED)
    n: int = 200

    # Variable duration bars (10 minutes to 6 hours)
    durations_minutes: np.ndarray = np.abs(rng.normal(60, 30, size=n)).clip(min=10)
    start_times: list[datetime] = []
    current_ts: datetime = _BASE_TS
    for dur in durations_minutes:
        start_times.append(current_ts)
        current_ts += timedelta(minutes=float(dur))

    end_times: list[datetime] = [
        s + timedelta(minutes=float(d)) for s, d in zip(start_times, durations_minutes, strict=True)
    ]

    prices: np.ndarray = 42000.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.005, size=n)))
    high_off: np.ndarray = np.abs(rng.normal(0, 50, size=n))
    low_off: np.ndarray = np.abs(rng.normal(0, 50, size=n))
    volumes: np.ndarray = np.abs(rng.normal(100, 30, size=n))
    tick_counts: np.ndarray = rng.integers(1, 200, size=n)
    buy_ratio: np.ndarray = rng.uniform(0.3, 0.7, size=n)

    return pd.DataFrame(
        {
            "start_ts": start_times,
            "end_ts": end_times,
            "open": prices,
            "high": prices + high_off,
            "low": prices - low_off,
            "close": prices * (1 + rng.normal(0, 0.001, size=n)),
            "volume": volumes,
            "tick_count": tick_counts,
            "buy_volume": volumes * buy_ratio,
            "sell_volume": volumes * (1 - buy_ratio),
            "vwap": prices,
        }
    )
