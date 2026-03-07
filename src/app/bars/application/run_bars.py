"""Run bar aggregator — tick, volume, and dollar run bars.

Forms a new bar when the longest consecutive run of same-direction
candles (measured by count, volume, or dollar volume) exceeds an
adaptive EMA-based expected run length.  Described in López de Prado,
*Advances in Financial Machine Learning* (2018), §2.3.3.
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


_RUN_TYPES: frozenset[BarType] = frozenset(
    {BarType.TICK_RUN, BarType.VOLUME_RUN, BarType.DOLLAR_RUN},
)


class RunBarAggregator:
    """Aggregate OHLCV rows into run bars.

    Supports tick, volume, and dollar run variants (determined by
    ``config.bar_type``).  Within each forming bar the aggregator tracks
    the maximum consecutive buy-run and sell-run.  A bar is completed
    when the dominant run metric exceeds an adaptive threshold:

    * During warmup (first ``config.warmup_period`` bars): threshold is
      fixed at ``config.threshold``.
    * After warmup: threshold is updated via EMA of the dominant run
      value observed at each bar completion.
    """

    def aggregate(  # noqa: PLR6301, PLR0912, PLR0914, PLR0915
        self,
        trades: pl.DataFrame,
        *,
        asset: Asset,
        config: BarConfig,
    ) -> list[AggregatedBar]:
        """Aggregate input rows into run bars.

        Args:
            trades: Polars DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``.
            asset: Trading-pair symbol for the resulting bars.
            config: Bar configuration; ``bar_type`` must be one of
                ``TICK_RUN``, ``VOLUME_RUN``, or ``DOLLAR_RUN``.

        Returns:
            List of run bars ordered by ``start_ts``.

        Raises:
            ValueError: If ``config.bar_type`` is not a run type
                or the input DataFrame is missing required columns.
        """
        if config.bar_type not in _RUN_TYPES:
            msg: str = f"Expected run bar type, got {config.bar_type}"
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

        # Per-row run metric depends on run variant
        run_metrics: npt.NDArray[np.float64]
        if config.bar_type == BarType.TICK_RUN:
            run_metrics = np.ones(n, dtype=np.float64)
        elif config.bar_type == BarType.VOLUME_RUN:
            run_metrics = volumes.copy()
        else:  # DOLLAR_RUN
            run_metrics = closes * volumes

        # ── sequential bar formation ─────────────────────────────────
        threshold: float = config.threshold
        alpha: float = 2.0 / (config.ewm_span + 1)
        bar_start: int = 0
        bars_formed: int = 0
        results: list[AggregatedBar] = []

        # Run tracking state
        current_run_dir: float = 0.0
        current_run_val: float = 0.0
        max_buy_run: float = 0.0
        max_sell_run: float = 0.0

        for i in range(n):
            direction: float = float(directions[i])
            metric: float = float(run_metrics[i])

            # Extend or restart the current run
            if direction == current_run_dir:
                current_run_val += metric
            else:
                current_run_dir = direction
                current_run_val = metric

            # Update max runs for buy / sell within this forming bar
            if direction > 0:
                max_buy_run = max(max_buy_run, current_run_val)
            else:
                max_sell_run = max(max_sell_run, current_run_val)

            dominant_run: float = max(max_buy_run, max_sell_run)

            if dominant_run >= threshold:
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
                    threshold = alpha * dominant_run + (1.0 - alpha) * threshold

                # Reset state for next bar
                bar_start = i + 1
                current_run_dir = 0.0
                current_run_val = 0.0
                max_buy_run = 0.0
                max_sell_run = 0.0

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
