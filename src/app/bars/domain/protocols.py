"""Bar domain protocols — structural interfaces for bar aggregation and persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl

from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.ohlcv.domain.value_objects import Asset, DateRange


class IBarAggregator(Protocol):
    """Structural interface for bar aggregation algorithms.

    Implementations receive a Polars DataFrame of trade-level (or 1-minute
    OHLCV) data and produce a list of :class:`AggregatedBar` entities.

    The input DataFrame must contain at least the columns:
    ``timestamp``, ``open``, ``high``, ``low``, ``close``, ``volume``.
    """

    def aggregate(
        self,
        trades: pl.DataFrame,
        *,
        asset: Asset,
        config: BarConfig,
    ) -> list[AggregatedBar]:
        """Aggregate raw data into bars according to the given configuration.

        Args:
            trades: Polars DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``.
            asset: Trading-pair symbol for the resulting bars.
            config: Bar construction parameters (type, threshold, etc.).

        Returns:
            List of aggregated bars ordered by ``start_ts``.
        """
        ...


class IBarRepository(Protocol):
    """Structural interface for aggregated bar persistence.

    Implementations store and retrieve :class:`AggregatedBar` entities,
    keyed by ``(asset, bar_type, bar_config_hash, start_ts)``.
    """

    def ingest(self, bars: list[AggregatedBar], *, config_hash: str) -> int:
        """Persist *bars* with the given configuration hash, ignoring duplicates.

        Args:
            bars: Aggregated bar entities to persist.
            config_hash: Hex digest identifying the :class:`BarConfig` that
                produced these bars.

        Returns:
            Number of rows actually inserted (duplicates are skipped).
        """
        ...

    def query(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
        date_range: DateRange,
    ) -> list[AggregatedBar]:
        """Return bars matching the filter, ordered by ``start_ts``.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash to filter by.
            date_range: UTC date boundaries (``start`` inclusive, ``end`` exclusive).

        Returns:
            Bars ordered chronologically by ``start_ts``.
        """
        ...

    def get_available_configs(self, asset: Asset) -> list[tuple[str, str]]:
        """Return distinct ``(bar_type, config_hash)`` pairs for an asset.

        Args:
            asset: Trading-pair symbol.

        Returns:
            List of ``(bar_type, config_hash)`` tuples.
        """
        ...

    def get_date_range(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
    ) -> DateRange | None:
        """Return the min/max ``start_ts`` range, or *None* if no data exists.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash to filter by.

        Returns:
            The date range or *None* when the store has no matching rows.
        """
        ...

    def count(self) -> int:
        """Return the total number of rows in the bar table.

        Returns:
            Row count.
        """
        ...

    def delete(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
    ) -> int:
        """Delete all bars for a given asset, bar type, and config hash.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash identifying the bars to delete.

        Returns:
            Number of rows deleted.
        """
        ...

    def get_latest_end_ts(
        self,
        asset: Asset,
        bar_type: BarType,
        config_hash: str,
    ) -> datetime | None:
        """Return the latest ``end_ts`` for incremental ingestion, or *None*.

        Args:
            asset: Trading-pair symbol.
            bar_type: Bar aggregation type.
            config_hash: Configuration hash to filter by.

        Returns:
            The most recent ``end_ts`` or *None* when no bars exist.
        """
        ...
