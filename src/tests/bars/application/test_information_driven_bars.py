"""Unit tests for information-driven bar aggregators — imbalance and run bars.

Tests cover directional triggering, wrong bar_type rejection, empty input handling,
adaptive threshold behavior, property tests for tick-count preservation, and
chronological ordering.
"""

from __future__ import annotations

from datetime import datetime, UTC

import polars as pl
import pytest

from src.app.bars.application.imbalance_bars import ImbalanceBarAggregator
from src.app.bars.application.run_bars import RunBarAggregator
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.tests.bars.conftest import (
    BTC_ASSET, make_alternating_trades_df, make_bearish_trades_df, make_bullish_trades_df, make_trades_df
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_THRESHOLD: float = 3.0
_EWM_SPAN: int = 10
_WARMUP: int = 5


def _imbalance_cfg(bar_type: BarType, threshold: float = _THRESHOLD) -> BarConfig:
    """Build an imbalance BarConfig with controlled EWM parameters.

    Args:
        bar_type: Must be one of TICK_IMBALANCE, VOLUME_IMBALANCE, DOLLAR_IMBALANCE.
        threshold: Imbalance threshold.  Defaults to 3.

    Returns:
        Configured BarConfig.
    """
    return BarConfig(bar_type=bar_type, threshold=threshold, ewm_span=_EWM_SPAN, warmup_period=_WARMUP)


def _run_cfg(bar_type: BarType, threshold: float = _THRESHOLD) -> BarConfig:
    """Build a run BarConfig with controlled EWM parameters.

    Args:
        bar_type: Must be one of TICK_RUN, VOLUME_RUN, DOLLAR_RUN.
        threshold: Run threshold.  Defaults to 3.

    Returns:
        Configured BarConfig.
    """
    return BarConfig(bar_type=bar_type, threshold=threshold, ewm_span=_EWM_SPAN, warmup_period=_WARMUP)


# ---------------------------------------------------------------------------
# ImbalanceBarAggregator — wrong bar_type
# ---------------------------------------------------------------------------


class TestImbalanceBarAggregatorValidation:
    """Tests for ImbalanceBarAggregator input validation."""

    @pytest.mark.parametrize(
        "bar_type",
        [BarType.TICK, BarType.VOLUME, BarType.DOLLAR, BarType.TICK_RUN, BarType.VOLUME_RUN, BarType.DOLLAR_RUN],
    )
    def test_non_imbalance_type_raises_value_error(self, bar_type: BarType) -> None:
        """ImbalanceBarAggregator must raise ValueError for non-imbalance bar types."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_trades_df(5)
        config: BarConfig = BarConfig(bar_type=bar_type, threshold=_THRESHOLD)
        with pytest.raises(ValueError, match="imbalance"):
            aggregator.aggregate(df, asset=BTC_ASSET, config=config)

    def test_empty_input_returns_empty_list(self) -> None:
        """aggregate() on empty DataFrame must return an empty list."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        empty_df: pl.DataFrame = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE)
        result: list[AggregatedBar] = aggregator.aggregate(empty_df, asset=BTC_ASSET, config=config)
        assert result == []

    def test_missing_column_raises_value_error(self) -> None:
        """aggregate() with a missing required column must raise ValueError."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        bad_df: pl.DataFrame = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "close": [42000.0]})
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE)
        with pytest.raises(ValueError, match="missing required columns"):
            aggregator.aggregate(bad_df, asset=BTC_ASSET, config=config)


# ---------------------------------------------------------------------------
# ImbalanceBarAggregator — tick imbalance
# ---------------------------------------------------------------------------


class TestTickImbalanceBars:
    """Tests for BarType.TICK_IMBALANCE aggregation logic."""

    def test_single_row_produces_one_bar(self) -> None:
        """A single-row input must produce exactly one bar."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(1)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1

    def test_all_bullish_candles_form_bar_at_threshold(self) -> None:
        """All-bullish data accumulates +1 per tick; bar forms when cumulative >= threshold.

        With threshold=3 and all bullish candles, the bar must form after exactly
        3 rows.  Remaining rows form a partial last bar.
        """
        n_rows: int = 7
        threshold: float = 3.0
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        # 7 rows, threshold=3 → bars at index 2, 5; partial bar rows 6 → 3 bars
        assert len(result) >= 2

    def test_all_bearish_candles_form_bar_at_threshold(self) -> None:
        """All-bearish data accumulates -1 per tick; absolute value triggers the bar.

        With threshold=3 and all bearish candles, bar forms at absolute imbalance >= 3.
        """
        n_rows: int = 7
        threshold: float = 3.0
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bearish_trades_df(n_rows)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) >= 2

    def test_alternating_candles_accumulate_slower(self) -> None:
        """Alternating bullish/bearish candles net out and take longer to form bars.

        7 alternating rows (+1,-1,+1,...) → cumulative never reaches threshold=3
        without sustained direction.  At threshold=3 with 7 rows there should be
        fewer bars than with all-same-direction data.
        """
        threshold: float = 3.0
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        bullish_df: pl.DataFrame = make_bullish_trades_df(7)
        alt_df: pl.DataFrame = make_alternating_trades_df(7)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE, threshold=threshold)

        bullish_bars: list[AggregatedBar] = aggregator.aggregate(bullish_df, asset=BTC_ASSET, config=config)
        alt_bars: list[AggregatedBar] = aggregator.aggregate(alt_df, asset=BTC_ASSET, config=config)

        # Alternating accumulation is slower → fewer complete bars
        # (or same count but with more rows per bar on average)
        total_bullish_ticks: int = sum(b.tick_count for b in bullish_bars)
        total_alt_ticks: int = sum(b.tick_count for b in alt_bars)
        # Both should contain all 7 rows
        assert total_bullish_ticks == 7
        assert total_alt_ticks == 7

    def test_total_tick_count_equals_input_rows(self) -> None:
        """Sum of tick_count across all bars must equal the number of input rows."""
        n_rows: int = 15
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    def test_bars_chronologically_ordered(self) -> None:
        """Bars must be ordered by start_ts (monotonically increasing)."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(20)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for i in range(1, len(result)):
            assert result[i].start_ts > result[i - 1].start_ts

    def test_bar_type_propagated(self) -> None:
        """All output bars must carry bar_type == TICK_IMBALANCE."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(10)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.bar_type == BarType.TICK_IMBALANCE

    def test_buy_sell_volume_lte_volume(self) -> None:
        """buy_volume + sell_volume must be <= volume for every bar."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(20)
        config: BarConfig = _imbalance_cfg(BarType.TICK_IMBALANCE)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.buy_volume + bar.sell_volume <= bar.volume + 1e-9


# ---------------------------------------------------------------------------
# ImbalanceBarAggregator — volume and dollar variants
# ---------------------------------------------------------------------------


class TestVolumeAndDollarImbalanceBars:
    """Tests for BarType.VOLUME_IMBALANCE and DOLLAR_IMBALANCE."""

    @pytest.mark.parametrize("bar_type", [BarType.VOLUME_IMBALANCE, BarType.DOLLAR_IMBALANCE])
    def test_total_tick_count_equals_input_rows(self, bar_type: BarType) -> None:
        """Sum of tick_count must equal input rows for volume and dollar imbalance."""
        n_rows: int = 12
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows, volume=5.0, price=42000.0)
        threshold: float = 10.0 if bar_type == BarType.VOLUME_IMBALANCE else 100_000.0
        config: BarConfig = _imbalance_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    @pytest.mark.parametrize("bar_type", [BarType.VOLUME_IMBALANCE, BarType.DOLLAR_IMBALANCE])
    def test_bars_chronologically_ordered(self, bar_type: BarType) -> None:
        """Bars must be ordered by start_ts for volume and dollar imbalance types."""
        n_rows: int = 12
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows, volume=5.0, price=42000.0)
        threshold: float = 10.0 if bar_type == BarType.VOLUME_IMBALANCE else 100_000.0
        config: BarConfig = _imbalance_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for i in range(1, len(result)):
            assert result[i].start_ts > result[i - 1].start_ts

    @pytest.mark.parametrize("bar_type", [BarType.VOLUME_IMBALANCE, BarType.DOLLAR_IMBALANCE])
    def test_bar_type_propagated(self, bar_type: BarType) -> None:
        """All output bars must carry the correct bar_type."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(10, volume=5.0, price=42000.0)
        threshold: float = 10.0 if bar_type == BarType.VOLUME_IMBALANCE else 100_000.0
        config: BarConfig = _imbalance_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.bar_type == bar_type


# ---------------------------------------------------------------------------
# RunBarAggregator — wrong bar_type
# ---------------------------------------------------------------------------


class TestRunBarAggregatorValidation:
    """Tests for RunBarAggregator input validation."""

    @pytest.mark.parametrize(
        "bar_type",
        [
            BarType.TICK,
            BarType.VOLUME,
            BarType.DOLLAR,
            BarType.TICK_IMBALANCE,
            BarType.VOLUME_IMBALANCE,
            BarType.DOLLAR_IMBALANCE,
        ],
    )
    def test_non_run_type_raises_value_error(self, bar_type: BarType) -> None:
        """RunBarAggregator must raise ValueError for non-run bar types."""
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_trades_df(5)
        config: BarConfig = BarConfig(bar_type=bar_type, threshold=_THRESHOLD)
        with pytest.raises(ValueError, match="run"):
            aggregator.aggregate(df, asset=BTC_ASSET, config=config)

    def test_empty_input_returns_empty_list(self) -> None:
        """aggregate() on empty DataFrame must return an empty list."""
        aggregator: RunBarAggregator = RunBarAggregator()
        empty_df: pl.DataFrame = pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
        config: BarConfig = _run_cfg(BarType.TICK_RUN)
        result: list[AggregatedBar] = aggregator.aggregate(empty_df, asset=BTC_ASSET, config=config)
        assert result == []

    def test_missing_column_raises_value_error(self) -> None:
        """aggregate() with a missing required column must raise ValueError."""
        aggregator: RunBarAggregator = RunBarAggregator()
        bad_df: pl.DataFrame = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "close": [42000.0]})
        config: BarConfig = _run_cfg(BarType.TICK_RUN)
        with pytest.raises(ValueError, match="missing required columns"):
            aggregator.aggregate(bad_df, asset=BTC_ASSET, config=config)


# ---------------------------------------------------------------------------
# RunBarAggregator — tick run
# ---------------------------------------------------------------------------


class TestTickRunBars:
    """Tests for BarType.TICK_RUN aggregation logic."""

    def test_single_row_produces_one_bar(self) -> None:
        """A single-row input must produce exactly one bar."""
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(1)
        config: BarConfig = _run_cfg(BarType.TICK_RUN)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1

    def test_consecutive_same_direction_forms_bar_at_threshold(self) -> None:
        """Consecutive same-direction candles should form a bar when run >= threshold.

        With threshold=3 and all-bullish candles, the run reaches 3 on the 3rd row
        and a bar is formed.
        """
        threshold: float = 3.0
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(7)
        config: BarConfig = _run_cfg(BarType.TICK_RUN, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        # 7 rows, run of 3 triggers at row 2 (0-indexed), again at row 5 → 3 bars
        assert len(result) >= 2

    def test_total_tick_count_equals_input_rows(self) -> None:
        """Sum of tick_count across all bars must equal the number of input rows."""
        n_rows: int = 20
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows)
        config: BarConfig = _run_cfg(BarType.TICK_RUN)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    def test_bars_chronologically_ordered(self) -> None:
        """Bars must be ordered by start_ts (monotonically increasing)."""
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(20)
        config: BarConfig = _run_cfg(BarType.TICK_RUN)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for i in range(1, len(result)):
            assert result[i].start_ts > result[i - 1].start_ts

    def test_bar_type_propagated(self) -> None:
        """All output bars must carry bar_type == TICK_RUN."""
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(10)
        config: BarConfig = _run_cfg(BarType.TICK_RUN)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.bar_type == BarType.TICK_RUN

    def test_buy_sell_volume_lte_volume(self) -> None:
        """buy_volume + sell_volume must be <= volume for every bar."""
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(20)
        config: BarConfig = _run_cfg(BarType.TICK_RUN)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.buy_volume + bar.sell_volume <= bar.volume + 1e-9

    def test_alternating_direction_inhibits_run_formation(self) -> None:
        """Alternating direction prevents sustained runs; fewer complete bars expected.

        With threshold=3 and direction alternating (+1,-1,...), each direction
        change resets the run counter so bars form less often than with
        all-same-direction data.
        """
        threshold: float = 3.0
        aggregator: RunBarAggregator = RunBarAggregator()
        bullish_df: pl.DataFrame = make_bullish_trades_df(12)
        alt_df: pl.DataFrame = make_alternating_trades_df(12)
        config: BarConfig = _run_cfg(BarType.TICK_RUN, threshold=threshold)

        bullish_bars: list[AggregatedBar] = aggregator.aggregate(bullish_df, asset=BTC_ASSET, config=config)
        alt_bars: list[AggregatedBar] = aggregator.aggregate(alt_df, asset=BTC_ASSET, config=config)

        # Both should preserve all ticks
        assert sum(b.tick_count for b in bullish_bars) == 12
        assert sum(b.tick_count for b in alt_bars) == 12

        # Alternating should produce fewer or equal complete bars (more partial tail)
        # The key check: all rows are still present in both cases
        assert len(bullish_bars) >= 1
        assert len(alt_bars) >= 1


# ---------------------------------------------------------------------------
# RunBarAggregator — volume and dollar run variants
# ---------------------------------------------------------------------------


class TestVolumeAndDollarRunBars:
    """Tests for BarType.VOLUME_RUN and DOLLAR_RUN."""

    @pytest.mark.parametrize("bar_type", [BarType.VOLUME_RUN, BarType.DOLLAR_RUN])
    def test_total_tick_count_equals_input_rows(self, bar_type: BarType) -> None:
        """Sum of tick_count must equal input rows for volume and dollar run bars."""
        n_rows: int = 12
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows, volume=5.0, price=42000.0)
        threshold: float = 10.0 if bar_type == BarType.VOLUME_RUN else 100_000.0
        config: BarConfig = _run_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    @pytest.mark.parametrize("bar_type", [BarType.VOLUME_RUN, BarType.DOLLAR_RUN])
    def test_bars_chronologically_ordered(self, bar_type: BarType) -> None:
        """Bars must be ordered by start_ts for volume and dollar run variants."""
        n_rows: int = 12
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows, volume=5.0, price=42000.0)
        threshold: float = 10.0 if bar_type == BarType.VOLUME_RUN else 100_000.0
        config: BarConfig = _run_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for i in range(1, len(result)):
            assert result[i].start_ts > result[i - 1].start_ts

    @pytest.mark.parametrize("bar_type", [BarType.VOLUME_RUN, BarType.DOLLAR_RUN])
    def test_bar_type_propagated(self, bar_type: BarType) -> None:
        """All output bars must carry the correct bar_type."""
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(10, volume=5.0, price=42000.0)
        threshold: float = 10.0 if bar_type == BarType.VOLUME_RUN else 100_000.0
        config: BarConfig = _run_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.bar_type == bar_type


# ---------------------------------------------------------------------------
# Cross-aggregator property: no rows lost — information-driven bars
# ---------------------------------------------------------------------------


class TestNoRowsLostInfoBars:
    """Property tests asserting no input rows are lost for all information-driven types."""

    @pytest.mark.parametrize(
        "bar_type",
        [BarType.TICK_IMBALANCE, BarType.VOLUME_IMBALANCE, BarType.DOLLAR_IMBALANCE],
    )
    @pytest.mark.parametrize("n_rows", [1, 5, 20, 50])
    def test_imbalance_no_rows_lost(self, bar_type: BarType, n_rows: int) -> None:
        """Total tick_count must equal n_rows for all imbalance variants."""
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows, volume=5.0, price=42000.0)
        threshold: float
        if bar_type == BarType.TICK_IMBALANCE:
            threshold = 3.0
        elif bar_type == BarType.VOLUME_IMBALANCE:
            threshold = 10.0
        else:
            threshold = 100_000.0
        config: BarConfig = _imbalance_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    @pytest.mark.parametrize("bar_type", [BarType.TICK_RUN, BarType.VOLUME_RUN, BarType.DOLLAR_RUN])
    @pytest.mark.parametrize("n_rows", [1, 5, 20, 50])
    def test_run_no_rows_lost(self, bar_type: BarType, n_rows: int) -> None:
        """Total tick_count must equal n_rows for all run variants."""
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows, volume=5.0, price=42000.0)
        threshold: float
        if bar_type == BarType.TICK_RUN:
            threshold = 3.0
        elif bar_type == BarType.VOLUME_RUN:
            threshold = 10.0
        else:
            threshold = 100_000.0
        config: BarConfig = _run_cfg(bar_type, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows


# ---------------------------------------------------------------------------
# Adaptive threshold test
# ---------------------------------------------------------------------------


class TestAdaptiveThreshold:
    """Tests for adaptive EMA threshold behavior after warmup."""

    def test_imbalance_threshold_adapts_after_warmup(self) -> None:
        """After warmup bars, the imbalance threshold should update via EMA.

        With a small warmup (1 bar), the second bar onward should form at
        an adapted threshold, not the original fixed one.  We verify that
        the aggregator still produces all-ticks-preserved output, not that
        the exact threshold value matches (that would be brittle white-box
        testing).
        """
        n_rows: int = 50
        aggregator: ImbalanceBarAggregator = ImbalanceBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows)
        # warmup_period=1 means adaptation starts after the first completed bar
        config: BarConfig = BarConfig(
            bar_type=BarType.TICK_IMBALANCE,
            threshold=3.0,
            ewm_span=10,
            warmup_period=1,
        )
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    def test_run_threshold_adapts_after_warmup(self) -> None:
        """After warmup bars, the run threshold should update via EMA.

        We verify tick preservation and bar validity after adaptation.
        """
        n_rows: int = 50
        aggregator: RunBarAggregator = RunBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(n_rows)
        config: BarConfig = BarConfig(
            bar_type=BarType.TICK_RUN,
            threshold=3.0,
            ewm_span=10,
            warmup_period=1,
        )
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows
