"""Imbalance bar aggregator — tick, volume, and dollar imbalance bars.

Forms a new bar when the cumulative signed imbalance exceeds an adaptive
EMA-based threshold.  During active directional periods, bars are produced
more frequently, capturing information arrival as described in
López de Prado, *Advances in Financial Machine Learning* (2018), §2.3.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import numpy.typing as npt
import polars as pl

from src.app.bars.application._aggregation import build_bar_from_arrays, infer_candle_period, validate_input
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.ohlcv.domain.value_objects import Asset


_IMBALANCE_TYPES: frozenset[BarType] = frozenset(
    {BarType.TICK_IMBALANCE, BarType.VOLUME_IMBALANCE, BarType.DOLLAR_IMBALANCE},
)


class ImbalanceBarAggregator:
    """Aggregate OHLCV rows into imbalance bars.

    Supports tick, volume, and dollar imbalance variants (determined by
    ``config.bar_type``).  Each candle is classified as buy (+1) or
    sell (−1) based on ``close >= open``.  The signed metric accumulates
    until its absolute value exceeds an adaptive threshold:

    * During warmup (first ``config.warmup_period`` bars): threshold is
      fixed at ``config.threshold``.
    * After warmup: threshold is updated via EMA of the absolute
      imbalance observed at each bar completion.
    """

    def aggregate(  # noqa: PLR6301, PLR0914
        self,
        trades: pl.DataFrame,
        *,
        asset: Asset,
        config: BarConfig,
    ) -> list[AggregatedBar]:
        """Aggregate input rows into imbalance bars.

        Args:
            trades: Polars DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``.
            asset: Trading-pair symbol for the resulting bars.
            config: Bar configuration; ``bar_type`` must be one of
                ``TICK_IMBALANCE``, ``VOLUME_IMBALANCE``, or
                ``DOLLAR_IMBALANCE``.

        Returns:
            List of imbalance bars ordered by ``start_ts``.

        Raises:
            ValueError: If ``config.bar_type`` is not an imbalance type
                or the input DataFrame is missing required columns.
        """
        if config.bar_type not in _IMBALANCE_TYPES:
            msg: str = f"Expected imbalance bar type, got {config.bar_type}"
            raise ValueError(msg)

        validate_input(trades)

        if trades.is_empty():
            return []

        # ── extract sorted NumPy arrays ──────────────────────────────
        df: pl.DataFrame = trades.sort("timestamp")
        candle_period: timedelta = infer_candle_period(df)

        timestamps: list[datetime] = df["timestamp"].to_list()
        opens: npt.NDArray[np.float64] = df["open"].cast(pl.Float64).to_numpy()
        highs: npt.NDArray[np.float64] = df["high"].cast(pl.Float64).to_numpy()
        lows: npt.NDArray[np.float64] = df["low"].cast(pl.Float64).to_numpy()
        closes: npt.NDArray[np.float64] = df["close"].cast(pl.Float64).to_numpy()
        volumes: npt.NDArray[np.float64] = df["volume"].cast(pl.Float64).to_numpy()
        n: int = len(df)

        # Direction: +1 if close >= open (buy), -1 otherwise (sell)
        directions: npt.NDArray[np.float64] = np.where(closes >= opens, 1.0, -1.0)

        # Signed metric per row depends on imbalance variant
        signed_metrics: npt.NDArray[np.float64]
        if config.bar_type == BarType.TICK_IMBALANCE:
            signed_metrics = directions
        elif config.bar_type == BarType.VOLUME_IMBALANCE:
            signed_metrics = directions * volumes
        else:  # DOLLAR_IMBALANCE
            signed_metrics = directions * closes * volumes

        # ── sequential bar formation ─────────────────────────────────
        threshold: float = config.threshold
        alpha: float = 2.0 / (config.ewm_span + 1)
        bar_start: int = 0
        cumulative: float = 0.0
        bars_formed: int = 0
        results: list[AggregatedBar] = []

        for i in range(n):
            cumulative += float(signed_metrics[i])

            if abs(cumulative) >= threshold:
                bar: AggregatedBar = build_bar_from_arrays(
                    asset=asset,
                    bar_type=config.bar_type,
                    timestamps=timestamps,
                    opens=opens,
                    highs=highs,
                    lows=lows,
                    closes=closes,
                    volumes=volumes,
                    start_idx=bar_start,
                    end_idx=i,
                    candle_period=candle_period,
                )
                results.append(bar)
                bars_formed += 1

                # Adaptive threshold update after warmup
                if bars_formed >= config.warmup_period:
                    threshold = alpha * abs(cumulative) + (1.0 - alpha) * threshold

                cumulative = 0.0
                bar_start = i + 1

        # Include remaining rows as an incomplete bar
        if bar_start < n:
            tail_bar: AggregatedBar = build_bar_from_arrays(
                asset=asset,
                bar_type=config.bar_type,
                timestamps=timestamps,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                start_idx=bar_start,
                end_idx=n - 1,
                candle_period=candle_period,
            )
            results.append(tail_bar)

        return results
