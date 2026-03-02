"""OHLCV repository protocol (domain-layer interface)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.value_objects import Asset, DateRange, TemporalSplit, Timeframe


class IOHLCVRepository(Protocol):
    """Structural interface that any OHLCV data-access implementation must satisfy."""

    def ingest(self, candles: list[OHLCVCandle]) -> int:
        """Persist *candles*, ignoring duplicates.  Return rows written."""
        ...

    def ingest_from_parquet(self, path: Path, asset: Asset, timeframe: Timeframe) -> int:
        """Bulk-load a Parquet file via DuckDB's native reader.  Return rows written."""
        ...

    def query(self, asset: Asset, timeframe: Timeframe, date_range: DateRange) -> list[OHLCVCandle]:
        """Return candles matching the filter, ordered by timestamp."""
        ...

    def query_split(
        self,
        asset: Asset,
        timeframe: Timeframe,
        split: TemporalSplit,
        partition: str,
    ) -> list[OHLCVCandle]:
        """Return candles for a single partition of a temporal split."""
        ...

    def query_cross_asset(
        self,
        assets: list[Asset],
        timeframe: Timeframe,
        date_range: DateRange,
    ) -> dict[str, list[OHLCVCandle]]:
        """Return candles for multiple assets, grouped by symbol."""
        ...

    def get_available_assets(self) -> list[str]:
        """Return distinct asset symbols present in the store."""
        ...

    def get_date_range(self, asset: Asset, timeframe: Timeframe) -> DateRange | None:
        """Return the min/max timestamp range for an asset+timeframe, or *None*."""
        ...

    def count(self) -> int:
        """Return the total number of rows."""
        ...
