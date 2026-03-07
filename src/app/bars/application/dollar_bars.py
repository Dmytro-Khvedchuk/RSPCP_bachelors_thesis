"""Dollar bar aggregator — samples when cumulative dollar volume reaches a threshold."""

from __future__ import annotations

import polars as pl

from src.app.bars.application._aggregation import aggregate_by_metric
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.ohlcv.domain.value_objects import Asset


class DollarBarAggregator:
    """Aggregate OHLCV rows into dollar bars.

    A new bar begins each time the cumulative dollar volume
    (``close × volume``) reaches ``config.threshold``.  Dollar bars
    normalise for price level, producing more uniform information
    content per bar than raw volume bars.

    Ref: López de Prado, *Advances in Financial Machine Learning* (2018), §2.3.
    """

    def aggregate(  # noqa: PLR6301
        self,
        trades: pl.DataFrame,
        *,
        asset: Asset,
        config: BarConfig,
    ) -> list[AggregatedBar]:
        """Aggregate input rows into dollar bars.

        Args:
            trades: Polars DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``.
            asset: Trading-pair symbol for the resulting bars.
            config: Bar configuration; ``threshold`` specifies the
                cumulative dollar volume at which a new bar starts.

        Returns:
            List of dollar bars ordered by ``start_ts``.
        """
        return aggregate_by_metric(
            trades,
            asset=asset,
            bar_type=BarType.DOLLAR,
            threshold=config.threshold,
            metric_expr=pl.col("close").cast(pl.Float64) * pl.col("volume"),
        )
