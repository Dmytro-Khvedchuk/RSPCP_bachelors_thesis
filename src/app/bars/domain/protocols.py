"""Bar domain protocols — structural interfaces for bar aggregation."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig
from src.app.ohlcv.domain.value_objects import Asset


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
