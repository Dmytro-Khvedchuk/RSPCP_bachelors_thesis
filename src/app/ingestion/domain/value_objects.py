"""Ingestion domain value objects."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe


class BinanceKlineInterval(StrEnum):
    """Binance kline intervals matching the project's ``Timeframe`` enum.

    Extends the base timeframes with ``M1`` (1-minute) which is needed for
    alternative bar construction in Phase 2.
    """

    M1 = "1m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @classmethod
    def from_timeframe(cls, timeframe: Timeframe) -> BinanceKlineInterval:
        """Convert a domain ``Timeframe`` to its Binance API interval.

        Args:
            timeframe: The domain timeframe to convert.

        Returns:
            The corresponding Binance kline interval.

        Raises:
            ValueError: If the timeframe has no matching interval.
        """
        mapping: dict[Timeframe, BinanceKlineInterval] = {
            Timeframe.H1: cls.H1,
            Timeframe.H4: cls.H4,
            Timeframe.D1: cls.D1,
        }
        result: BinanceKlineInterval | None = mapping.get(timeframe)
        if result is None:
            msg: str = f"No BinanceKlineInterval for timeframe {timeframe!r}"
            raise ValueError(msg)
        return result


class FetchRequest(BaseModel, frozen=True):
    """A single OHLCV fetch job specification.

    Encapsulates which asset, timeframe, and date range to retrieve
    from the exchange API.
    """

    asset: Asset
    timeframe: Timeframe
    date_range: DateRange


TIMEFRAME_INTERVAL_MS: dict[BinanceKlineInterval, int] = {
    BinanceKlineInterval.M1: 60_000,
    BinanceKlineInterval.H1: 60 * 60_000,
    BinanceKlineInterval.H4: 4 * 60 * 60_000,
    BinanceKlineInterval.D1: 24 * 60 * 60_000,
}
"""Millisecond duration for each kline interval, used for pagination cursor advancement."""
