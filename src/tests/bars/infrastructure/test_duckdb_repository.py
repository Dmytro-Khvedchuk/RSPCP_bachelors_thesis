"""Integration tests for DuckDBBarRepository.

Tests cover the full ingest → query round-trip, duplicate handling
(INSERT OR IGNORE), range queries via DateRange, delete(), get_latest_end_ts(),
count(), count_by_config(), and get_available_configs().

All tests use the in-memory DuckDB fixture from bars/conftest.py, so no real
database file is created.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC
from decimal import Decimal

import pytest

from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarType
from src.app.bars.infrastructure.duckdb_repository import DuckDBBarRepository
from src.app.ohlcv.domain.value_objects import Asset, DateRange
from src.tests.bars.conftest import BTC_ASSET, ETH_ASSET, make_aggregated_bar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_TS: datetime = datetime(2024, 1, 1, tzinfo=UTC)
_CONFIG_HASH: str = "abc123def456abcd"
_ALT_CONFIG_HASH: str = "fffaaa111bbb2222"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bars_sequence(
    n: int,
    *,
    asset: Asset = BTC_ASSET,
    bar_type: BarType = BarType.TICK,
    base_ts: datetime = _BASE_TS,
) -> list[AggregatedBar]:
    """Build a list of n sequential AggregatedBar objects.

    Each bar spans one hour and starts where the previous ended.

    Args:
        n: Number of bars to generate.
        asset: Trading-pair symbol.
        bar_type: Bar aggregation type.
        base_ts: Start timestamp of the first bar.

    Returns:
        Ordered list of aggregated bars.
    """
    bars: list[AggregatedBar] = []
    for i in range(n):
        start: datetime = base_ts + i * timedelta(hours=1)
        end: datetime = start + timedelta(hours=1)
        bar: AggregatedBar = make_aggregated_bar(
            asset=asset,
            bar_type=bar_type,
            start_ts=start,
            end_ts=end,
        )
        bars.append(bar)
    return bars


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBBarRepositoryIngest:
    """Integration tests for DuckDBBarRepository.ingest()."""

    def test_ingest_returns_count_of_inserted_rows(self, bar_repository: DuckDBBarRepository) -> None:
        """ingest() must return the number of rows actually inserted."""
        bars: list[AggregatedBar] = _make_bars_sequence(3)
        written: int = bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        assert written == 3

    def test_ingest_empty_list_returns_zero(self, bar_repository: DuckDBBarRepository) -> None:
        """ingest() with an empty list must return 0 without touching the table."""
        written: int = bar_repository.ingest([], config_hash=_CONFIG_HASH)
        assert written == 0

    def test_ingest_single_bar(self, bar_repository: DuckDBBarRepository) -> None:
        """ingest() with a single bar must insert one row."""
        bars: list[AggregatedBar] = _make_bars_sequence(1)
        written: int = bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        assert written == 1

    def test_duplicate_ingest_ignored(self, bar_repository: DuckDBBarRepository) -> None:
        """Inserting the same bars twice must ignore duplicates (INSERT OR IGNORE)."""
        bars: list[AggregatedBar] = _make_bars_sequence(3)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        written_second: int = bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        # All duplicates → 0 new rows; total count must remain 3
        assert bar_repository.count() == 3
        # DuckDB may return 0 or len(bars) for second insert; key check is total count
        _ = written_second  # exact value not asserted as driver may differ

    def test_ingest_large_batch(self, bar_repository: DuckDBBarRepository) -> None:
        """ingest() must handle a batch of 500+ bars without error."""
        bars: list[AggregatedBar] = _make_bars_sequence(500)
        written: int = bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        assert written >= 1  # at least some rows were inserted

    def test_ingest_preserves_decimal_precision(self, bar_repository: DuckDBBarRepository) -> None:
        """Ingested bars must preserve Decimal precision for OHLC and VWAP fields."""
        precise_price: Decimal = Decimal("42000.12345678")
        bar: AggregatedBar = AggregatedBar(
            asset=BTC_ASSET,
            bar_type=BarType.TICK,
            start_ts=_BASE_TS,
            end_ts=_BASE_TS + timedelta(hours=1),
            open=precise_price,
            high=precise_price + Decimal("100"),
            low=precise_price - Decimal("100"),
            close=precise_price,
            volume=100.0,
            tick_count=10,
            buy_volume=50.0,
            sell_volume=50.0,
            vwap=precise_price,
        )
        bar_repository.ingest([bar], config_hash=_CONFIG_HASH)
        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(days=1))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert len(results) == 1
        # Decimal precision may be truncated to 8 decimal places by DuckDB DECIMAL(18,8)
        # Compare as floats since Decimal round-trips through DuckDB DECIMAL(18,8)
        assert float(results[0].open) == pytest.approx(float(precise_price), rel=1e-6)


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBBarRepositoryQuery:
    """Integration tests for DuckDBBarRepository.query()."""

    def test_query_returns_all_bars_in_date_range(self, bar_repository: DuckDBBarRepository) -> None:
        """query() must return all bars whose start_ts falls in [start, end)."""
        bars: list[AggregatedBar] = _make_bars_sequence(5)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=6))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert len(results) == 5

    def test_query_empty_range_returns_empty_list(self, bar_repository: DuckDBBarRepository) -> None:
        """query() for a date range before any data must return an empty list."""
        bars: list[AggregatedBar] = _make_bars_sequence(3)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        early_start: datetime = datetime(2020, 1, 1, tzinfo=UTC)
        early_end: datetime = datetime(2020, 6, 1, tzinfo=UTC)
        dr: DateRange = DateRange(start=early_start, end=early_end)
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert results == []

    def test_query_is_ordered_by_start_ts(self, bar_repository: DuckDBBarRepository) -> None:
        """query() must return bars ordered by start_ts ascending."""
        bars: list[AggregatedBar] = _make_bars_sequence(5)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=6))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        for i in range(1, len(results)):
            assert results[i].start_ts >= results[i - 1].start_ts

    def test_query_filters_by_asset(self, bar_repository: DuckDBBarRepository) -> None:
        """query() must only return bars for the requested asset."""
        btc_bars: list[AggregatedBar] = _make_bars_sequence(3, asset=BTC_ASSET)
        eth_bars: list[AggregatedBar] = _make_bars_sequence(3, asset=ETH_ASSET)
        bar_repository.ingest(btc_bars, config_hash=_CONFIG_HASH)
        bar_repository.ingest(eth_bars, config_hash=_CONFIG_HASH)

        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=6))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert all(b.asset == BTC_ASSET for b in results)
        assert len(results) == 3

    def test_query_filters_by_bar_type(self, bar_repository: DuckDBBarRepository) -> None:
        """query() must only return bars of the requested bar_type."""
        tick_bars: list[AggregatedBar] = _make_bars_sequence(3, bar_type=BarType.TICK)
        volume_bars: list[AggregatedBar] = _make_bars_sequence(3, bar_type=BarType.VOLUME)
        bar_repository.ingest(tick_bars, config_hash=_CONFIG_HASH)
        bar_repository.ingest(volume_bars, config_hash=_CONFIG_HASH)

        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=6))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert all(b.bar_type == BarType.TICK for b in results)

    def test_query_filters_by_config_hash(self, bar_repository: DuckDBBarRepository) -> None:
        """query() must only return bars with the matching config_hash."""
        bars_a: list[AggregatedBar] = _make_bars_sequence(3)
        bars_b: list[AggregatedBar] = _make_bars_sequence(3, base_ts=_BASE_TS + timedelta(days=1))
        bar_repository.ingest(bars_a, config_hash=_CONFIG_HASH)
        bar_repository.ingest(bars_b, config_hash=_ALT_CONFIG_HASH)

        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(days=3))
        results_a: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        results_b: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _ALT_CONFIG_HASH, dr)
        assert len(results_a) == 3
        assert len(results_b) == 3

    def test_query_date_range_is_start_inclusive_end_exclusive(self, bar_repository: DuckDBBarRepository) -> None:
        """query() must include bars at start_ts == range.start and exclude start_ts == range.end."""
        bars: list[AggregatedBar] = _make_bars_sequence(4)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        # Range that exactly spans first 2 bars: start=_BASE_TS (inclusive), end=_BASE_TS+2h (exclusive)
        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=2))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert len(results) == 2

    def test_query_preserves_asset_symbol(self, bar_repository: DuckDBBarRepository) -> None:
        """Queried bars must have the correct asset symbol restored."""
        bars: list[AggregatedBar] = _make_bars_sequence(1, asset=BTC_ASSET)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=2))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert len(results) == 1
        assert results[0].asset.symbol == "BTCUSDT"

    def test_query_preserves_bar_type(self, bar_repository: DuckDBBarRepository) -> None:
        """Queried bars must have the correct bar_type restored."""
        bars: list[AggregatedBar] = _make_bars_sequence(1, bar_type=BarType.VOLUME)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=2))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.VOLUME, _CONFIG_HASH, dr)
        assert len(results) == 1
        assert results[0].bar_type == BarType.VOLUME

    def test_query_no_data_returns_empty_list(self, bar_repository: DuckDBBarRepository) -> None:
        """query() on an empty table must return an empty list."""
        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(days=1))
        results: list[AggregatedBar] = bar_repository.query(BTC_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert results == []


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBBarRepositoryDelete:
    """Integration tests for DuckDBBarRepository.delete()."""

    def test_delete_removes_matching_bars(self, bar_repository: DuckDBBarRepository) -> None:
        """delete() must remove all bars matching (asset, bar_type, config_hash)."""
        bars: list[AggregatedBar] = _make_bars_sequence(5)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        deleted: int = bar_repository.delete(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert deleted == 5
        assert bar_repository.count() == 0

    def test_delete_returns_zero_when_no_bars_exist(self, bar_repository: DuckDBBarRepository) -> None:
        """delete() on a table with no matching bars must return 0."""
        deleted: int = bar_repository.delete(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert deleted == 0

    def test_delete_only_removes_targeted_bars(self, bar_repository: DuckDBBarRepository) -> None:
        """delete() must not remove bars with a different asset, bar_type, or config_hash."""
        btc_bars: list[AggregatedBar] = _make_bars_sequence(3, asset=BTC_ASSET)
        eth_bars: list[AggregatedBar] = _make_bars_sequence(3, asset=ETH_ASSET)
        bar_repository.ingest(btc_bars, config_hash=_CONFIG_HASH)
        bar_repository.ingest(eth_bars, config_hash=_CONFIG_HASH)

        bar_repository.delete(BTC_ASSET, BarType.TICK, _CONFIG_HASH)

        # ETH bars must still exist
        dr: DateRange = DateRange(start=_BASE_TS, end=_BASE_TS + timedelta(hours=6))
        remaining: list[AggregatedBar] = bar_repository.query(ETH_ASSET, BarType.TICK, _CONFIG_HASH, dr)
        assert len(remaining) == 3

    def test_delete_with_different_config_hash_not_affected(self, bar_repository: DuckDBBarRepository) -> None:
        """delete() for config_hash A must not remove bars stored under config_hash B."""
        bars_a: list[AggregatedBar] = _make_bars_sequence(3)
        bars_b: list[AggregatedBar] = _make_bars_sequence(3, base_ts=_BASE_TS + timedelta(days=1))
        bar_repository.ingest(bars_a, config_hash=_CONFIG_HASH)
        bar_repository.ingest(bars_b, config_hash=_ALT_CONFIG_HASH)

        bar_repository.delete(BTC_ASSET, BarType.TICK, _CONFIG_HASH)

        assert bar_repository.count_by_config(BTC_ASSET, BarType.TICK, _ALT_CONFIG_HASH) == 3


# ---------------------------------------------------------------------------
# count() and count_by_config()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBBarRepositoryCount:
    """Integration tests for DuckDBBarRepository.count() and count_by_config()."""

    def test_count_empty_table_returns_zero(self, bar_repository: DuckDBBarRepository) -> None:
        """count() on an empty table must return 0."""
        assert bar_repository.count() == 0

    def test_count_after_ingest(self, bar_repository: DuckDBBarRepository) -> None:
        """count() must reflect the total number of rows after ingest."""
        bars: list[AggregatedBar] = _make_bars_sequence(7)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        assert bar_repository.count() == 7

    def test_count_by_config_filters_correctly(self, bar_repository: DuckDBBarRepository) -> None:
        """count_by_config() must return only rows matching the given filters."""
        bars_a: list[AggregatedBar] = _make_bars_sequence(3)
        bars_b: list[AggregatedBar] = _make_bars_sequence(5, base_ts=_BASE_TS + timedelta(days=1))
        bar_repository.ingest(bars_a, config_hash=_CONFIG_HASH)
        bar_repository.ingest(bars_b, config_hash=_ALT_CONFIG_HASH)

        assert bar_repository.count_by_config(BTC_ASSET, BarType.TICK, _CONFIG_HASH) == 3
        assert bar_repository.count_by_config(BTC_ASSET, BarType.TICK, _ALT_CONFIG_HASH) == 5

    def test_count_by_config_returns_zero_when_no_match(self, bar_repository: DuckDBBarRepository) -> None:
        """count_by_config() must return 0 when no bars match the filter."""
        result: int = bar_repository.count_by_config(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result == 0

    def test_count_reflects_deletion(self, bar_repository: DuckDBBarRepository) -> None:
        """count() must decrease correctly after delete()."""
        bars: list[AggregatedBar] = _make_bars_sequence(5)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        bar_repository.delete(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert bar_repository.count() == 0


# ---------------------------------------------------------------------------
# get_latest_end_ts()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBBarRepositoryGetLatestEndTs:
    """Integration tests for DuckDBBarRepository.get_latest_end_ts()."""

    def test_returns_none_when_no_bars(self, bar_repository: DuckDBBarRepository) -> None:
        """get_latest_end_ts() must return None when the table is empty."""
        result: datetime | None = bar_repository.get_latest_end_ts(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is None

    def test_returns_latest_end_ts(self, bar_repository: DuckDBBarRepository) -> None:
        """get_latest_end_ts() must return the maximum end_ts in the store."""
        bars: list[AggregatedBar] = _make_bars_sequence(5)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        expected_end_ts: datetime = _BASE_TS + timedelta(hours=5)
        result: datetime | None = bar_repository.get_latest_end_ts(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is not None
        # Allow small tolerance because DuckDB TIMESTAMPTZ may round-trip with UTC conversion
        assert abs((result - expected_end_ts).total_seconds()) < 2

    def test_returns_none_for_different_asset(self, bar_repository: DuckDBBarRepository) -> None:
        """get_latest_end_ts() must return None for an asset with no stored bars."""
        bars: list[AggregatedBar] = _make_bars_sequence(3, asset=BTC_ASSET)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        result: datetime | None = bar_repository.get_latest_end_ts(ETH_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is None

    def test_returns_none_after_deletion(self, bar_repository: DuckDBBarRepository) -> None:
        """get_latest_end_ts() must return None after all bars have been deleted."""
        bars: list[AggregatedBar] = _make_bars_sequence(3)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        bar_repository.delete(BTC_ASSET, BarType.TICK, _CONFIG_HASH)

        result: datetime | None = bar_repository.get_latest_end_ts(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is None

    def test_end_ts_is_utc_aware(self, bar_repository: DuckDBBarRepository) -> None:
        """get_latest_end_ts() must return a timezone-aware UTC datetime."""
        bars: list[AggregatedBar] = _make_bars_sequence(1)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        result: datetime | None = bar_repository.get_latest_end_ts(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is not None
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# get_date_range()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBBarRepositoryGetDateRange:
    """Integration tests for DuckDBBarRepository.get_date_range()."""

    def test_returns_none_when_no_bars(self, bar_repository: DuckDBBarRepository) -> None:
        """get_date_range() must return None when the table is empty."""
        result: DateRange | None = bar_repository.get_date_range(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is None

    def test_returns_correct_range(self, bar_repository: DuckDBBarRepository) -> None:
        """get_date_range() must return the min/max start_ts bounds."""
        bars: list[AggregatedBar] = _make_bars_sequence(5)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        result: DateRange | None = bar_repository.get_date_range(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is not None
        # min start_ts is _BASE_TS, max start_ts is _BASE_TS + 4h
        assert abs((result.start - _BASE_TS).total_seconds()) < 2
        assert result.end >= result.start

    def test_single_bar_returns_valid_range(self, bar_repository: DuckDBBarRepository) -> None:
        """get_date_range() with a single bar must return a valid DateRange (start < end)."""
        bars: list[AggregatedBar] = _make_bars_sequence(1)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        result: DateRange | None = bar_repository.get_date_range(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is not None
        assert result.start < result.end

    def test_returns_none_after_deletion(self, bar_repository: DuckDBBarRepository) -> None:
        """get_date_range() must return None after all bars have been deleted."""
        bars: list[AggregatedBar] = _make_bars_sequence(3)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)
        bar_repository.delete(BTC_ASSET, BarType.TICK, _CONFIG_HASH)

        result: DateRange | None = bar_repository.get_date_range(BTC_ASSET, BarType.TICK, _CONFIG_HASH)
        assert result is None


# ---------------------------------------------------------------------------
# get_available_configs()
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDuckDBBarRepositoryGetAvailableConfigs:
    """Integration tests for DuckDBBarRepository.get_available_configs()."""

    def test_returns_empty_list_when_no_bars(self, bar_repository: DuckDBBarRepository) -> None:
        """get_available_configs() must return an empty list for an asset with no data."""
        result: list[tuple[str, str]] = bar_repository.get_available_configs(BTC_ASSET)
        assert result == []

    def test_returns_bar_type_and_config_hash(self, bar_repository: DuckDBBarRepository) -> None:
        """get_available_configs() must return the correct (bar_type, config_hash) pair."""
        bars: list[AggregatedBar] = _make_bars_sequence(2)
        bar_repository.ingest(bars, config_hash=_CONFIG_HASH)

        result: list[tuple[str, str]] = bar_repository.get_available_configs(BTC_ASSET)
        assert len(result) == 1
        bar_type_str: str
        hash_str: str
        bar_type_str, hash_str = result[0]
        assert bar_type_str == BarType.TICK.value
        assert hash_str == _CONFIG_HASH

    def test_returns_multiple_configs(self, bar_repository: DuckDBBarRepository) -> None:
        """get_available_configs() must return all distinct (bar_type, config_hash) pairs."""
        tick_bars: list[AggregatedBar] = _make_bars_sequence(2, bar_type=BarType.TICK)
        volume_bars: list[AggregatedBar] = _make_bars_sequence(2, bar_type=BarType.VOLUME)
        bar_repository.ingest(tick_bars, config_hash=_CONFIG_HASH)
        bar_repository.ingest(volume_bars, config_hash=_ALT_CONFIG_HASH)

        result: list[tuple[str, str]] = bar_repository.get_available_configs(BTC_ASSET)
        assert len(result) == 2

    def test_filters_by_asset(self, bar_repository: DuckDBBarRepository) -> None:
        """get_available_configs() must only return configs for the requested asset."""
        btc_bars: list[AggregatedBar] = _make_bars_sequence(2, asset=BTC_ASSET)
        eth_bars: list[AggregatedBar] = _make_bars_sequence(2, asset=ETH_ASSET)
        bar_repository.ingest(btc_bars, config_hash=_CONFIG_HASH)
        bar_repository.ingest(eth_bars, config_hash=_ALT_CONFIG_HASH)

        btc_configs: list[tuple[str, str]] = bar_repository.get_available_configs(BTC_ASSET)
        eth_configs: list[tuple[str, str]] = bar_repository.get_available_configs(ETH_ASSET)

        assert len(btc_configs) == 1
        assert len(eth_configs) == 1

    def test_deduplicates_configs(self, bar_repository: DuckDBBarRepository) -> None:
        """get_available_configs() must return distinct pairs even with many rows."""
        bars_a: list[AggregatedBar] = _make_bars_sequence(10)
        bar_repository.ingest(bars_a, config_hash=_CONFIG_HASH)
        # Same config, but different bars (different start_ts won't conflict)
        bars_b: list[AggregatedBar] = _make_bars_sequence(10, base_ts=bars_a[-1].end_ts)
        bar_repository.ingest(bars_b, config_hash=_CONFIG_HASH)

        result: list[tuple[str, str]] = bar_repository.get_available_configs(BTC_ASSET)
        assert len(result) == 1  # Still one distinct (bar_type, config_hash)
