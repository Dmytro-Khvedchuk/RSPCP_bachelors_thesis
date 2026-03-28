"""Shared fixtures and factory functions for backtest module tests.

Provides bar DataFrames, fake strategies, fake position sizers, and
portfolio snapshot builders used across all backtest test modules.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC

import polars as pl
import pytest

from src.app.backtest.domain.entities import Signal
from src.app.backtest.domain.value_objects import ExecutionConfig, PortfolioSnapshot, Side
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_ONE_HOUR: timedelta = timedelta(hours=1)

BTC_ASSET: Asset = Asset(symbol="BTCUSDT")
ETH_ASSET: Asset = Asset(symbol="ETHUSDT")

INITIAL_CASH: float = 100_000.0


# ---------------------------------------------------------------------------
# Bar factory
# ---------------------------------------------------------------------------


def make_bars(
    n: int,
    *,
    start_price: float = 40_000.0,
    start_time: datetime = _BASE_TS,
    interval_hours: int = 1,
    price_step: float = 0.0,
    volume: float = 10.0,
) -> pl.DataFrame:
    """Build a minimal OHLCV Polars DataFrame for backtest tests.

    Args:
        n: Number of bars to generate.
        start_price: Open price for the first bar.
        start_time: Timestamp for the first bar.
        interval_hours: Hours between consecutive bar timestamps.
        price_step: Price increment per bar (0 for flat prices).
        volume: Volume for every bar.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    interval: timedelta = timedelta(hours=interval_hours)
    timestamps: list[datetime] = [start_time + i * interval for i in range(n)]
    opens: list[float] = [start_price + i * price_step for i in range(n)]
    closes: list[float] = [start_price + i * price_step for i in range(n)]
    highs: list[float] = [p + 200.0 for p in closes]
    lows: list[float] = [max(1.0, p - 200.0) for p in closes]
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


# ---------------------------------------------------------------------------
# Portfolio snapshot factory
# ---------------------------------------------------------------------------


def make_snapshot(
    *,
    equity: float = INITIAL_CASH,
    cash: float = INITIAL_CASH,
    timestamp: datetime = _BASE_TS,
    positions: dict[str, float] | None = None,
) -> PortfolioSnapshot:
    """Build a PortfolioSnapshot for test use.

    Args:
        equity: Total portfolio equity.
        cash: Available cash balance.
        timestamp: Snapshot timestamp.
        positions: Open positions dict (symbol -> signed size).

    Returns:
        Configured PortfolioSnapshot.
    """
    return PortfolioSnapshot(
        timestamp=timestamp,
        equity=equity,
        cash=cash,
        positions=positions or {},
        unrealized_pnl=0.0,
        drawdown=0.0,
    )


# ---------------------------------------------------------------------------
# Fake strategies
# ---------------------------------------------------------------------------


class AlwaysLongStrategy:
    """Fake strategy that emits a LONG signal on every bar without an open position.

    Checks ``portfolio.positions`` to avoid re-entering when already long,
    which would cause the engine to overwrite the existing position and create
    a Trade with entry_time == exit_time on the last bar.
    """

    def __init__(self, asset: Asset) -> None:
        self._asset: Asset = asset

    def on_bar(
        self,
        timestamp: datetime,
        features: pl.DataFrame,  # noqa: ARG002
        portfolio: PortfolioSnapshot,
    ) -> list[Signal]:
        if len(portfolio.positions) > 0:
            return []
        return [
            Signal(
                asset=self._asset,
                side=Side.LONG,
                strength=1.0,
                timestamp=timestamp,
            )
        ]


class NeverTradeStrategy:
    """Fake strategy that never emits any signals."""

    def on_bar(
        self,
        timestamp: datetime,  # noqa: ARG002
        features: pl.DataFrame,  # noqa: ARG002
        portfolio: PortfolioSnapshot,  # noqa: ARG002
    ) -> list[Signal]:
        return []


class SingleSignalStrategy:
    """Fake strategy that emits one signal at bar index 0 only."""

    def __init__(self, asset: Asset, side: Side = Side.LONG) -> None:
        self._asset: Asset = asset
        self._side: Side = side
        self._call_count: int = 0

    def on_bar(
        self,
        timestamp: datetime,
        features: pl.DataFrame,  # noqa: ARG002
        portfolio: PortfolioSnapshot,  # noqa: ARG002
    ) -> list[Signal]:
        self._call_count += 1
        if self._call_count == 1:
            return [
                Signal(
                    asset=self._asset,
                    side=self._side,
                    strength=1.0,
                    timestamp=timestamp,
                )
            ]
        return []


# ---------------------------------------------------------------------------
# Fake position sizer
# ---------------------------------------------------------------------------


class FixedNotionalSizer:
    """Fake sizer that returns a fixed notional regardless of portfolio state."""

    def __init__(self, notional: float = 10_000.0) -> None:
        self._notional: float = notional

    def size(
        self,
        signal: Signal,  # noqa: ARG002
        portfolio: PortfolioSnapshot,  # noqa: ARG002
        volatility: float,  # noqa: ARG002
    ) -> float:
        return self._notional


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def btc_asset() -> Asset:
    """Return a BTCUSDT asset."""
    return BTC_ASSET


@pytest.fixture
def eth_asset() -> Asset:
    """Return an ETHUSDT asset."""
    return ETH_ASSET


@pytest.fixture
def default_config() -> ExecutionConfig:
    """Return a default ExecutionConfig with 10 bps commission."""
    return ExecutionConfig(commission_bps=10.0)


@pytest.fixture
def zero_commission_config() -> ExecutionConfig:
    """Return an ExecutionConfig with zero commission."""
    return ExecutionConfig(commission_bps=0.0)


@pytest.fixture
def flat_bars_5() -> pl.DataFrame:
    """Return 5 flat-price bars at 40_000 each."""
    return make_bars(5, start_price=40_000.0)


@pytest.fixture
def rising_bars_10() -> pl.DataFrame:
    """Return 10 bars with price increasing by 100 per bar."""
    return make_bars(10, start_price=40_000.0, price_step=100.0)


@pytest.fixture
def default_snapshot() -> PortfolioSnapshot:
    """Return a default portfolio snapshot with 100_000 cash/equity."""
    return make_snapshot()
