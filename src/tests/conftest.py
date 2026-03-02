"""Shared test fixtures, factory functions, and constants.

Provides reusable building blocks consumed by every test module in the
project.  Fixtures are auto-discovered by pytest; factory functions and
constants must be imported explicitly.
"""

from __future__ import annotations

from datetime import datetime, UTC
from decimal import Decimal

import pytest

from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DT: datetime = datetime(2024, 1, 1, tzinfo=UTC)
END_DT: datetime = datetime(2024, 6, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def make_asset(symbol: str = "BTCUSDT") -> Asset:
    """Build an ``Asset`` with the given symbol.

    Args:
        symbol: Trading pair symbol.  Defaults to ``"BTCUSDT"``.

    Returns:
        Configured Asset instance.
    """
    return Asset(symbol=symbol)


def make_date_range(
    start: datetime = START_DT,
    end: datetime = END_DT,
) -> DateRange:
    """Build a ``DateRange`` with the given bounds.

    Args:
        start: Range start (inclusive).  Defaults to ``START_DT``.
        end: Range end (exclusive).  Defaults to ``END_DT``.

    Returns:
        Configured DateRange instance.
    """
    return DateRange(start=start, end=end)


def make_candle(
    asset: Asset,
    timeframe: Timeframe,
    timestamp: datetime,
    price: Decimal = Decimal("42000.00"),
) -> OHLCVCandle:
    """Build a minimal valid ``OHLCVCandle`` for test data.

    Args:
        asset: Trading pair.
        timeframe: Candlestick interval.
        timestamp: Candle open time.
        price: Base price used for open/close; high/low are offset by 100.

    Returns:
        Configured OHLCVCandle instance.
    """
    return OHLCVCandle(
        asset=asset,
        timeframe=timeframe,
        timestamp=timestamp,
        open=price,
        high=price + Decimal("100"),
        low=price - Decimal("100"),
        close=price,
        volume=1.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def btc_asset() -> Asset:
    """Return a BTCUSDT asset."""
    return make_asset("BTCUSDT")


@pytest.fixture
def eth_asset() -> Asset:
    """Return an ETHUSDT asset."""
    return make_asset("ETHUSDT")


@pytest.fixture
def date_range() -> DateRange:
    """Return the standard test date range (START_DT to END_DT)."""
    return make_date_range()


@pytest.fixture
def h1_timeframe() -> Timeframe:
    """Return the H1 timeframe."""
    return Timeframe.H1
