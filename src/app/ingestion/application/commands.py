"""Ingestion command objects — immutable descriptions of ingestion jobs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe


class IngestAssetCommand(BaseModel, frozen=True):
    """Command to ingest OHLCV data for a single asset and timeframe.

    Attributes:
        asset: The trading pair to fetch (e.g. ``BTCUSDT``).
        timeframe: The candlestick interval.
        date_range: The UTC date boundaries to fetch.
    """

    asset: Asset
    timeframe: Timeframe
    date_range: DateRange


class IngestUniverseCommand(BaseModel, frozen=True):
    """Command to ingest OHLCV data for multiple assets and timeframes.

    The Cartesian product of ``assets x timeframes`` is fetched over
    the given date range.

    Attributes:
        assets: Trading pairs to fetch.
        timeframes: Candlestick intervals to fetch per asset.
        date_range: The UTC date boundaries to fetch.
    """

    assets: list[Asset] = Field(min_length=1)
    timeframes: list[Timeframe] = Field(min_length=1)
    date_range: DateRange
