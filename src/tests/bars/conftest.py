"""Shared fixtures and factory functions for bars module tests.

Provides Polars DataFrame builders, AggregatedBar factories, BarConfig
presets, and the DuckDB infrastructure fixture used by integration tests.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, UTC
from decimal import Decimal

import polars as pl
import pytest

from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.bars.infrastructure.duckdb_repository import DuckDBBarRepository
from src.app.ohlcv.domain.value_objects import Asset
from src.app.system.database.connection import ConnectionManager
from src.app.system.database.settings import DatabaseSettings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_ONE_MINUTE: timedelta = timedelta(minutes=1)

BTC_ASSET: Asset = Asset(symbol="BTCUSDT")
ETH_ASSET: Asset = Asset(symbol="ETHUSDT")


# ---------------------------------------------------------------------------
# DataFrame factory helpers
# ---------------------------------------------------------------------------


def make_trades_df(
    n: int,
    *,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_MINUTE,
    price: float = 42000.0,
    volume: float = 1.0,
    price_step: float = 0.0,
) -> pl.DataFrame:
    """Build a minimal OHLCV Polars DataFrame for aggregator tests.

    Args:
        n: Number of rows to generate.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.
        price: Base close price (also used for open/high/low with small offsets).
        volume: Volume for every row.
        price_step: Price increment per row (for trend simulation).

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    opens: list[float] = [price + i * price_step for i in range(n)]
    closes: list[float] = [price + i * price_step for i in range(n)]
    highs: list[float] = [p + 50.0 for p in closes]
    lows: list[float] = [p - 50.0 for p in closes]
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


def make_bullish_trades_df(
    n: int,
    *,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_MINUTE,
    price: float = 42000.0,
    volume: float = 1.0,
) -> pl.DataFrame:
    """Build a DataFrame where every candle is bullish (close > open).

    All candles have close > open to force buy direction.

    Args:
        n: Number of rows.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.
        price: Base price value.
        volume: Volume for every row.

    Returns:
        DataFrame with bullish candles (close = open + 10).
    """
    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    opens: list[float] = [price] * n
    closes: list[float] = [price + 10.0] * n
    highs: list[float] = [price + 20.0] * n
    lows: list[float] = [price - 5.0] * n
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


def make_bearish_trades_df(
    n: int,
    *,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_MINUTE,
    price: float = 42000.0,
    volume: float = 1.0,
) -> pl.DataFrame:
    """Build a DataFrame where every candle is bearish (close < open).

    Args:
        n: Number of rows.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.
        price: Base price value.
        volume: Volume per row.

    Returns:
        DataFrame with bearish candles (close = open - 10).
    """
    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    opens: list[float] = [price] * n
    closes: list[float] = [price - 10.0] * n
    highs: list[float] = [price + 5.0] * n
    lows: list[float] = [price - 20.0] * n
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


def make_alternating_trades_df(
    n: int,
    *,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_MINUTE,
    price: float = 42000.0,
    volume: float = 1.0,
) -> pl.DataFrame:
    """Build a DataFrame where candles alternate between bullish and bearish.

    Args:
        n: Number of rows.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.
        price: Base price value.
        volume: Volume per row.

    Returns:
        DataFrame alternating between close = open+10 and close = open-10.
    """
    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]
    opens: list[float] = [price] * n
    closes: list[float] = [price + 10.0 if i % 2 == 0 else price - 10.0 for i in range(n)]
    highs: list[float] = [price + 20.0] * n
    lows: list[float] = [price - 20.0] * n
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


def make_varying_volume_df(
    volumes: list[float],
    *,
    base_ts: datetime = _BASE_TS,
    interval: timedelta = _ONE_MINUTE,
    price: float = 42000.0,
) -> pl.DataFrame:
    """Build a DataFrame with explicitly specified per-row volumes.

    Args:
        volumes: Per-row volume values.
        base_ts: Timestamp for the first row.
        interval: Duration between consecutive timestamps.
        price: Price used for all OHLC columns.

    Returns:
        DataFrame with the given volumes.
    """
    n: int = len(volumes)
    timestamps: list[datetime] = [base_ts + i * interval for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [price] * n,
            "high": [price + 50.0] * n,
            "low": [price - 50.0] * n,
            "close": [price] * n,
            "volume": volumes,
        }
    )


# ---------------------------------------------------------------------------
# AggregatedBar factory
# ---------------------------------------------------------------------------


def make_aggregated_bar(
    *,
    asset: Asset = BTC_ASSET,
    bar_type: BarType = BarType.TICK,
    start_ts: datetime = _BASE_TS,
    end_ts: datetime | None = None,
    price: Decimal = Decimal("42000.00"),
    volume: float = 100.0,
    tick_count: int = 10,
) -> AggregatedBar:
    """Build an AggregatedBar with minimal defaults.

    Args:
        asset: Trading-pair symbol.
        bar_type: Bar aggregation type.
        start_ts: Bar start timestamp.
        end_ts: Bar end timestamp.  Defaults to start_ts + 1 hour.
        price: Base price for all OHLC fields and VWAP.
        volume: Total volume.
        tick_count: Number of ticks in the bar.

    Returns:
        Configured AggregatedBar instance.
    """
    if end_ts is None:
        end_ts = start_ts + timedelta(hours=1)

    return AggregatedBar(
        asset=asset,
        bar_type=bar_type,
        start_ts=start_ts,
        end_ts=end_ts,
        open=price,
        high=price + Decimal("100"),
        low=price - Decimal("100"),
        close=price,
        volume=volume,
        tick_count=tick_count,
        buy_volume=volume * 0.5,
        sell_volume=volume * 0.5,
        vwap=price,
    )


# ---------------------------------------------------------------------------
# Infrastructure fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture
def bar_connection_manager() -> Generator[ConnectionManager]:
    """Create an in-memory DuckDB ConnectionManager for bars integration tests.

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
    yield cm
    cm.dispose()


@pytest.fixture
def bar_repository(bar_connection_manager: ConnectionManager) -> DuckDBBarRepository:
    """Return a DuckDBBarRepository backed by in-memory DuckDB with table created.

    Args:
        bar_connection_manager: In-memory connection manager fixture.

    Returns:
        Repository instance with the aggregated_bars table pre-created.
    """
    from sqlalchemy import text

    with bar_connection_manager.connect() as conn:
        conn.execute(text(_CREATE_BARS_TABLE))
        conn.commit()

    return DuckDBBarRepository(connection_manager=bar_connection_manager)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def btc_asset() -> Asset:
    """Return a BTCUSDT Asset fixture."""
    return BTC_ASSET


@pytest.fixture
def eth_asset() -> Asset:
    """Return an ETHUSDT Asset fixture."""
    return ETH_ASSET


@pytest.fixture
def tick_bar_config() -> BarConfig:
    """Return a BarConfig for tick bars with threshold=3."""
    return BarConfig(bar_type=BarType.TICK, threshold=3.0)


@pytest.fixture
def volume_bar_config() -> BarConfig:
    """Return a BarConfig for volume bars with threshold=10.0."""
    return BarConfig(bar_type=BarType.VOLUME, threshold=10.0)


@pytest.fixture
def dollar_bar_config() -> BarConfig:
    """Return a BarConfig for dollar bars with threshold=420000.0."""
    return BarConfig(bar_type=BarType.DOLLAR, threshold=420_000.0)


@pytest.fixture
def imbalance_bar_config() -> BarConfig:
    """Return a BarConfig for tick imbalance bars with threshold=3."""
    return BarConfig(bar_type=BarType.TICK_IMBALANCE, threshold=3.0, ewm_span=10, warmup_period=5)


@pytest.fixture
def run_bar_config() -> BarConfig:
    """Return a BarConfig for tick run bars with threshold=3."""
    return BarConfig(bar_type=BarType.TICK_RUN, threshold=3.0, ewm_span=10, warmup_period=5)
