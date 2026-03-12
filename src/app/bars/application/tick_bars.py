"""Tick bar aggregator — samples every N input rows."""

from __future__ import annotations

import polars as pl

from src.app.bars.application._aggregation import aggregate_by_metric
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.ohlcv.domain.value_objects import Asset


class TickBarAggregator:
    """Aggregate OHLCV rows into tick bars.

    Each bar contains exactly ``config.threshold`` input rows (the last
    bar may be shorter if the data does not divide evenly).

    Typical default: ``threshold = 1000`` (1 000 one-minute candles per bar).
    """

    def aggregate(  # noqa: PLR6301
        self,
        trades: pl.DataFrame,
        *,
        asset: Asset,
        config: BarConfig,
    ) -> list[AggregatedBar]:
        """Aggregate input rows into tick bars.

        Args:
            trades: Polars DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``.
            asset: Trading-pair symbol for the resulting bars.
            config: Bar configuration; ``threshold`` specifies the number
                of input rows per bar.

        Returns:
            List of tick bars ordered by ``start_ts``.
        """
        return aggregate_by_metric(
            trades,
            asset=asset,
            bar_type=BarType.TICK,
            threshold=config.threshold,
            metric_expr=pl.lit(1),
        )
