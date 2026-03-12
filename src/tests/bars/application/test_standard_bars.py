"""Unit tests for standard bar aggregators — tick, volume, and dollar bars.

Tests cover bar boundary correctness, OHLCV aggregation accuracy, empty input
handling, missing-column validation, single-row edge cases, and the shared
``aggregate_by_metric`` pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC
from decimal import Decimal

import polars as pl
import pytest

from src.app.bars.application.dollar_bars import DollarBarAggregator
from src.app.bars.application.tick_bars import TickBarAggregator
from src.app.bars.application.volume_bars import VolumeBarAggregator
from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarConfig, BarType
from src.app.ohlcv.domain.value_objects import Asset
from src.tests.bars.conftest import BTC_ASSET, make_bullish_trades_df, make_trades_df, make_varying_volume_df


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TICK_THRESHOLD: float = 3.0
_VOLUME_THRESHOLD: float = 10.0
_DOLLAR_THRESHOLD: float = 420_000.0
_PRICE: float = 42000.0


# ---------------------------------------------------------------------------
# TickBarAggregator
# ---------------------------------------------------------------------------


class TestTickBarAggregator:
    """Tests for TickBarAggregator bar boundary and OHLCV correctness."""

    def test_empty_input_returns_empty_list(self) -> None:
        """aggregate() on empty DataFrame must return an empty list."""
        aggregator: TickBarAggregator = TickBarAggregator()
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
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_TICK_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(empty_df, asset=BTC_ASSET, config=config)
        assert result == []

    def test_missing_column_raises_value_error(self) -> None:
        """aggregate() with a missing column must raise ValueError."""
        aggregator: TickBarAggregator = TickBarAggregator()
        bad_df: pl.DataFrame = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "open": [42000.0]})
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_TICK_THRESHOLD)
        with pytest.raises(ValueError, match="missing required columns"):
            aggregator.aggregate(bad_df, asset=BTC_ASSET, config=config)

    def test_single_row_produces_one_bar(self) -> None:
        """A single-row input must produce exactly one bar (the partial last bar)."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(1)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_TICK_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1

    def test_exactly_threshold_rows_produces_one_bar(self) -> None:
        """N rows exactly equal to threshold must produce 1 complete bar."""
        threshold: int = 3
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(threshold)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=float(threshold))
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1

    def test_bars_count_matches_n_over_threshold(self) -> None:
        """6 rows with threshold=3 must produce 2 complete bars."""
        n_rows: int = 6
        threshold: int = 3
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=float(threshold))
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        expected_bars: int = n_rows // threshold
        assert len(result) == expected_bars

    def test_remainder_rows_produce_partial_bar(self) -> None:
        """7 rows with threshold=3 must produce 2 complete bars + 1 partial bar."""
        n_rows: int = 7
        threshold: int = 3
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=float(threshold))
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        expected_bars: int = 3  # 3 + 3 + 1 partial
        assert len(result) == expected_bars

    def test_complete_bars_have_correct_tick_count(self) -> None:
        """Each complete bar must have tick_count == threshold."""
        threshold: int = 3
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(6)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=float(threshold))
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.tick_count == threshold

    def test_total_tick_count_equals_input_rows(self) -> None:
        """Sum of tick_count across all bars must equal the number of input rows."""
        n_rows: int = 10
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_TICK_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total_ticks: int = sum(b.tick_count for b in result)
        assert total_ticks == n_rows

    def test_bars_are_chronologically_ordered(self) -> None:
        """Bars must be ordered by start_ts (monotonically increasing)."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(12)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=_TICK_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for i in range(1, len(result)):
            assert result[i].start_ts > result[i - 1].start_ts

    def test_high_max_within_bar(self) -> None:
        """The high of each bar must equal the maximum high within its input rows."""
        aggregator: TickBarAggregator = TickBarAggregator()
        # 3 rows, each with a distinct price: high = close + 50
        base_ts: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        df: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [
                    base_ts,
                    base_ts + timedelta(minutes=1),
                    base_ts + timedelta(minutes=2),
                ],
                "open": [100.0, 200.0, 300.0],
                "high": [150.0, 250.0, 350.0],
                "low": [50.0, 150.0, 250.0],
                "close": [120.0, 220.0, 320.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1
        assert result[0].high == Decimal("350.0")

    def test_low_min_within_bar(self) -> None:
        """The low of each bar must equal the minimum low within its input rows."""
        aggregator: TickBarAggregator = TickBarAggregator()
        base_ts: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        df: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [
                    base_ts,
                    base_ts + timedelta(minutes=1),
                    base_ts + timedelta(minutes=2),
                ],
                "open": [100.0, 200.0, 300.0],
                "high": [150.0, 250.0, 350.0],
                "low": [50.0, 150.0, 250.0],
                "close": [120.0, 220.0, 320.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1
        assert result[0].low == Decimal("50.0")

    def test_open_is_first_row_open_in_bar(self) -> None:
        """The open of each bar must equal the open of the first row in that bar."""
        aggregator: TickBarAggregator = TickBarAggregator()
        base_ts: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        df: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [
                    base_ts,
                    base_ts + timedelta(minutes=1),
                    base_ts + timedelta(minutes=2),
                ],
                "open": [100.0, 200.0, 300.0],
                "high": [150.0, 250.0, 350.0],
                "low": [50.0, 150.0, 250.0],
                "close": [120.0, 220.0, 320.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1
        assert result[0].open == Decimal("100.0")

    def test_close_is_last_row_close_in_bar(self) -> None:
        """The close of each bar must equal the close of the last row in that bar."""
        aggregator: TickBarAggregator = TickBarAggregator()
        base_ts: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        df: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": [
                    base_ts,
                    base_ts + timedelta(minutes=1),
                    base_ts + timedelta(minutes=2),
                ],
                "open": [100.0, 200.0, 300.0],
                "high": [150.0, 250.0, 350.0],
                "low": [50.0, 150.0, 250.0],
                "close": [120.0, 220.0, 320.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1
        assert result[0].close == Decimal("320.0")

    def test_volume_sum_within_bar(self) -> None:
        """The volume of each bar must equal the sum of volumes within its rows."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_varying_volume_df([2.0, 3.0, 5.0])
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1
        assert result[0].volume == pytest.approx(10.0)

    def test_start_ts_is_first_row_timestamp(self) -> None:
        """start_ts of each bar must equal the timestamp of the first row in that bar."""
        aggregator: TickBarAggregator = TickBarAggregator()
        base_ts: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        df: pl.DataFrame = make_trades_df(3, base_ts=base_ts)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1
        assert result[0].start_ts == base_ts

    def test_end_ts_is_after_last_row_timestamp(self) -> None:
        """end_ts of each bar must be strictly after the timestamp of the last row."""
        aggregator: TickBarAggregator = TickBarAggregator()
        base_ts: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        df: pl.DataFrame = make_trades_df(3, base_ts=base_ts)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1
        last_row_ts: datetime = base_ts + timedelta(minutes=2)
        assert result[0].end_ts > last_row_ts

    def test_asset_propagated_to_bars(self) -> None:
        """The asset symbol must be propagated correctly to all output bars."""
        asset: Asset = Asset(symbol="ETHUSDT")
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(3)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=asset, config=config)
        for bar in result:
            assert bar.asset == asset

    def test_bar_type_is_tick(self) -> None:
        """All output bars must have bar_type == TICK."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(3)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.bar_type == BarType.TICK

    def test_buy_plus_sell_volume_lte_volume(self) -> None:
        """buy_volume + sell_volume must be <= volume for every bar."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(9)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.buy_volume + bar.sell_volume <= bar.volume + 1e-9

    def test_vwap_between_low_and_high(self) -> None:
        """VWAP must be between the bar's low and high (inclusive)."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(6)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.low <= bar.vwap <= bar.high

    @pytest.mark.parametrize(
        ("n_rows", "threshold", "expected_bars"),
        [
            (10, 5, 2),
            (15, 5, 3),
            (6, 2, 3),
            (12, 4, 3),
            (1, 5, 1),
            (5, 5, 1),
        ],
    )
    def test_bar_count_parametrized(
        self,
        n_rows: int,
        threshold: float,
        expected_bars: int,
    ) -> None:
        """Bar count must match floor(n_rows / threshold) or +1 for remainder."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == expected_bars

    def test_large_input_total_tick_count_preserved(self) -> None:
        """1000+ rows: total tick_count must equal number of input rows."""
        n_rows: int = 1001
        threshold: float = 100.0
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total_ticks: int = sum(b.tick_count for b in result)
        assert total_ticks == n_rows


# ---------------------------------------------------------------------------
# VolumeBarAggregator
# ---------------------------------------------------------------------------


class TestVolumeBarAggregator:
    """Tests for VolumeBarAggregator bar boundary and volume threshold correctness."""

    def test_empty_input_returns_empty_list(self) -> None:
        """aggregate() on empty DataFrame must return an empty list."""
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
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
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=_VOLUME_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(empty_df, asset=BTC_ASSET, config=config)
        assert result == []

    def test_missing_column_raises_value_error(self) -> None:
        """aggregate() with a missing column must raise ValueError."""
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        bad_df: pl.DataFrame = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "close": [42000.0]})
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=_VOLUME_THRESHOLD)
        with pytest.raises(ValueError, match="missing required columns"):
            aggregator.aggregate(bad_df, asset=BTC_ASSET, config=config)

    def test_single_row_produces_one_bar(self) -> None:
        """A single-row input must produce exactly one bar."""
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_varying_volume_df([5.0])
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=_VOLUME_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1

    def test_bar_forms_at_cumulative_volume_threshold(self) -> None:
        """A bar must form when cumulative volume first crosses the threshold."""
        # 3 rows × 5.0 volume = 15.0 → threshold 10.0 is crossed at row 2 (cumsum=10)
        volumes: list[float] = [5.0, 5.0, 5.0]
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_varying_volume_df(volumes)
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=10.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        # First bar: rows 0+1 (cumvol=10 at row 1), second bar: row 2 (partial)
        assert len(result) == 2

    def test_total_volume_preserved(self) -> None:
        """Sum of volume across all bars must equal total input volume."""
        volumes: list[float] = [3.0, 4.0, 3.0, 5.0, 2.0, 3.0]
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_varying_volume_df(volumes)
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=_VOLUME_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total_volume: float = sum(b.volume for b in result)
        assert total_volume == pytest.approx(sum(volumes))

    def test_total_tick_count_equals_input_rows(self) -> None:
        """Sum of tick_count across all bars must equal the number of input rows."""
        volumes: list[float] = [2.0, 3.0, 5.0, 4.0, 6.0]
        n_rows: int = len(volumes)
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_varying_volume_df(volumes)
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=_VOLUME_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total_ticks: int = sum(b.tick_count for b in result)
        assert total_ticks == n_rows

    def test_bars_are_chronologically_ordered(self) -> None:
        """Bars must be ordered chronologically by start_ts."""
        volumes: list[float] = [5.0, 5.0, 5.0, 5.0]
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_varying_volume_df(volumes)
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=10.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for i in range(1, len(result)):
            assert result[i].start_ts > result[i - 1].start_ts

    def test_bar_type_is_volume(self) -> None:
        """All output bars must have bar_type == VOLUME."""
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_varying_volume_df([5.0, 5.0, 5.0])
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=10.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.bar_type == BarType.VOLUME

    def test_uniform_volume_rows_correct_bar_boundaries(self) -> None:
        """10 rows each with volume=2.0, threshold=4.0 → 5 bars of 2 rows each."""
        volumes: list[float] = [2.0] * 10
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_varying_volume_df(volumes)
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=4.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 5
        for bar in result:
            assert bar.tick_count == 2


# ---------------------------------------------------------------------------
# DollarBarAggregator
# ---------------------------------------------------------------------------


class TestDollarBarAggregator:
    """Tests for DollarBarAggregator bar boundary and dollar-volume threshold."""

    def test_empty_input_returns_empty_list(self) -> None:
        """aggregate() on empty DataFrame must return an empty list."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
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
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(empty_df, asset=BTC_ASSET, config=config)
        assert result == []

    def test_missing_column_raises_value_error(self) -> None:
        """aggregate() with a missing column must raise ValueError."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
        bad_df: pl.DataFrame = pl.DataFrame({"timestamp": [datetime(2024, 1, 1, tzinfo=UTC)], "volume": [1.0]})
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        with pytest.raises(ValueError, match="missing required columns"):
            aggregator.aggregate(bad_df, asset=BTC_ASSET, config=config)

    def test_single_row_produces_one_bar(self) -> None:
        """A single-row input must produce exactly one bar."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
        df: pl.DataFrame = make_trades_df(1, price=_PRICE)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 1

    def test_dollar_bar_forms_when_dollar_volume_exceeds_threshold(self) -> None:
        """A bar must form when cumulative (close × volume) exceeds the threshold.

        The bar_id formula uses ``floor((cumsum_before) / threshold)``.  Due to
        floating-point precision, a cumsum_before that equals threshold exactly
        may produce ``0.999...`` instead of ``1.0``, so the bar boundary is
        effectively exclusive.

        With price=42000 and volume=5, each row contributes 210000 dollars.
        4 rows produce bar_ids [0,0,0,1] → 2 bars (3-row bar + 1-row partial).
        """
        price: float = 42000.0
        threshold: float = 420_000.0
        aggregator: DollarBarAggregator = DollarBarAggregator()
        # Each row: close=42000, volume=5 → dollar_val = 210000
        # 4 rows: cumsum_before for row 3 = 3*210000 = 630000 → 630000/420000=1.5 → bar_id=1
        df: pl.DataFrame = make_varying_volume_df([5.0, 5.0, 5.0, 5.0], price=price)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=threshold)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        assert len(result) == 2

    def test_total_volume_preserved(self) -> None:
        """Sum of volume across all bars must equal total input volume."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
        df: pl.DataFrame = make_trades_df(6, price=_PRICE, volume=2.0)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total_volume: float = sum(b.volume for b in result)
        assert total_volume == pytest.approx(6 * 2.0)

    def test_total_tick_count_equals_input_rows(self) -> None:
        """Sum of tick_count across all bars must equal the number of input rows."""
        n_rows: int = 8
        aggregator: DollarBarAggregator = DollarBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows, price=_PRICE, volume=2.0)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total_ticks: int = sum(b.tick_count for b in result)
        assert total_ticks == n_rows

    def test_bars_are_chronologically_ordered(self) -> None:
        """Bars must be ordered chronologically by start_ts."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
        df: pl.DataFrame = make_trades_df(9, price=_PRICE, volume=5.0)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for i in range(1, len(result)):
            assert result[i].start_ts > result[i - 1].start_ts

    def test_bar_type_is_dollar(self) -> None:
        """All output bars must have bar_type == DOLLAR."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
        df: pl.DataFrame = make_trades_df(3, price=_PRICE, volume=5.0)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.bar_type == BarType.DOLLAR

    def test_vwap_between_low_and_high(self) -> None:
        """VWAP must lie between the bar's low and high (inclusive)."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
        df: pl.DataFrame = make_bullish_trades_df(6, price=_PRICE, volume=5.0)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        for bar in result:
            assert bar.low <= bar.vwap <= bar.high


# ---------------------------------------------------------------------------
# Cross-aggregator property: no rows lost
# ---------------------------------------------------------------------------


class TestNoRowsLostProperty:
    """Property tests asserting that no input rows are lost or duplicated."""

    @pytest.mark.parametrize(
        "n_rows",
        [1, 2, 3, 7, 10, 50, 100],
    )
    def test_tick_bars_no_rows_lost(self, n_rows: int) -> None:
        """Total tick_count across tick bars must equal n_rows for any input size."""
        aggregator: TickBarAggregator = TickBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows)
        config: BarConfig = BarConfig(bar_type=BarType.TICK, threshold=3.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    @pytest.mark.parametrize(
        "n_rows",
        [1, 2, 3, 7, 10, 50, 100],
    )
    def test_volume_bars_no_rows_lost(self, n_rows: int) -> None:
        """Total tick_count across volume bars must equal n_rows for any input size."""
        aggregator: VolumeBarAggregator = VolumeBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows, volume=1.0)
        config: BarConfig = BarConfig(bar_type=BarType.VOLUME, threshold=5.0)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows

    @pytest.mark.parametrize(
        "n_rows",
        [1, 2, 3, 7, 10, 50, 100],
    )
    def test_dollar_bars_no_rows_lost(self, n_rows: int) -> None:
        """Total tick_count across dollar bars must equal n_rows for any input size."""
        aggregator: DollarBarAggregator = DollarBarAggregator()
        df: pl.DataFrame = make_trades_df(n_rows, price=_PRICE, volume=1.0)
        config: BarConfig = BarConfig(bar_type=BarType.DOLLAR, threshold=_DOLLAR_THRESHOLD)
        result: list[AggregatedBar] = aggregator.aggregate(df, asset=BTC_ASSET, config=config)
        total: int = sum(b.tick_count for b in result)
        assert total == n_rows
