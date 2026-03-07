"""Volume bar aggregator — samples when cumulative volume reaches a threshold."""

from __future__ import annotations

import polars as pl

from src.app.bars.application._aggregation import aggregate_by_metric
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.ohlcv.domain.value_objects import Asset


class VolumeBarAggregator:
    """Aggregate OHLCV rows into volume bars.

    A new bar begins each time the cumulative base-asset volume
    reaches ``config.threshold``.  This produces bars that each
    contain approximately the same amount of traded volume,
    sampling more frequently during active periods.
    """

    def aggregate(  # noqa: PLR6301
        self,
        trades: pl.DataFrame,
        *,
        asset: Asset,
        config: BarConfig,
    ) -> list[AggregatedBar]:
        """Aggregate input rows into volume bars.

        Args:
            trades: Polars DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``.
            asset: Trading-pair symbol for the resulting bars.
            config: Bar configuration; ``threshold`` specifies the
                cumulative volume at which a new bar starts.

        Returns:
            List of volume bars ordered by ``start_ts``.
        """
        return aggregate_by_metric(
            trades,
            asset=asset,
            bar_type=BarType.VOLUME,
            threshold=config.threshold,
            metric_expr=pl.col("volume"),
        )
