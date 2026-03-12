"""Unit tests for bars domain entity AggregatedBar.

Tests cover the invariant model_validator, happy-path construction,
boundary conditions, and Pydantic frozen-model semantics.
"""

from __future__ import annotations

from datetime import datetime, timedelta, UTC
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.app.bars.domain.entities import AggregatedBar
from src.app.bars.domain.value_objects import BarType
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TS_START: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
_TS_END: datetime = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)
_ASSET: Asset = Asset(symbol="BTCUSDT")
_BAR_TYPE: BarType = BarType.TICK
_OPEN: Decimal = Decimal("42000.00")
_HIGH: Decimal = Decimal("42500.00")
_LOW: Decimal = Decimal("41800.00")
_CLOSE: Decimal = Decimal("42200.00")
_VOLUME: float = 150.5
_TICK_COUNT: int = 100
_BUY_VOLUME: float = 80.0
_SELL_VOLUME: float = 70.0
_VWAP: Decimal = Decimal("42100.00")


def _make_bar(
    *,
    asset: Asset = _ASSET,
    bar_type: BarType = _BAR_TYPE,
    start_ts: datetime = _TS_START,
    end_ts: datetime = _TS_END,
    open_: Decimal = _OPEN,
    high: Decimal = _HIGH,
    low: Decimal = _LOW,
    close: Decimal = _CLOSE,
    volume: float = _VOLUME,
    tick_count: int = _TICK_COUNT,
    buy_volume: float = _BUY_VOLUME,
    sell_volume: float = _SELL_VOLUME,
    vwap: Decimal = _VWAP,
) -> AggregatedBar:
    """Build an AggregatedBar with overridable fields.

    Args:
        asset: Trading-pair symbol.
        bar_type: Bar aggregation type.
        start_ts: Bar start timestamp.
        end_ts: Bar end timestamp.
        open_: Open price.
        high: High price.
        low: Low price.
        close: Close price.
        volume: Total volume.
        tick_count: Number of ticks in bar.
        buy_volume: Estimated buy volume.
        sell_volume: Estimated sell volume.
        vwap: Volume-weighted average price.

    Returns:
        Configured AggregatedBar instance.
    """
    return AggregatedBar(
        asset=asset,
        bar_type=bar_type,
        start_ts=start_ts,
        end_ts=end_ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        tick_count=tick_count,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        vwap=vwap,
    )


# ---------------------------------------------------------------------------
# Happy-path construction
# ---------------------------------------------------------------------------


class TestAggregatedBarConstruction:
    """Tests for valid AggregatedBar construction."""

    def test_minimal_valid_bar(self) -> None:
        """AggregatedBar must be constructable with all valid fields."""
        bar: AggregatedBar = _make_bar()
        assert bar.asset == _ASSET
        assert bar.bar_type == _BAR_TYPE
        assert bar.start_ts == _TS_START
        assert bar.end_ts == _TS_END
        assert bar.open == _OPEN
        assert bar.high == _HIGH
        assert bar.low == _LOW
        assert bar.close == _CLOSE
        assert bar.volume == _VOLUME
        assert bar.tick_count == _TICK_COUNT
        assert bar.buy_volume == _BUY_VOLUME
        assert bar.sell_volume == _SELL_VOLUME
        assert bar.vwap == _VWAP

    def test_high_equals_low_is_valid(self) -> None:
        """High == low must be accepted (a candle with no range)."""
        same_price: Decimal = Decimal("42000.00")
        bar: AggregatedBar = _make_bar(high=same_price, low=same_price)
        assert bar.high == bar.low

    def test_zero_volume_is_valid(self) -> None:
        """volume=0 must be accepted (degenerate but not prohibited by invariant)."""
        bar: AggregatedBar = _make_bar(volume=0.0, buy_volume=0.0, sell_volume=0.0)
        assert bar.volume == 0.0

    def test_tick_count_one_is_valid(self) -> None:
        """tick_count=1 is at the lower bound and must be accepted."""
        bar: AggregatedBar = _make_bar(tick_count=1)
        assert bar.tick_count == 1

    def test_buy_sell_sum_exactly_equal_to_volume_is_valid(self) -> None:
        """buy_volume + sell_volume == volume must be accepted."""
        vol: float = 100.0
        bar: AggregatedBar = _make_bar(volume=vol, buy_volume=60.0, sell_volume=40.0)
        assert bar.buy_volume + bar.sell_volume == pytest.approx(vol)

    def test_zero_buy_zero_sell_volume_is_valid(self) -> None:
        """buy_volume=0 and sell_volume=0 must be accepted when volume > 0."""
        bar: AggregatedBar = _make_bar(buy_volume=0.0, sell_volume=0.0)
        assert bar.buy_volume == 0.0
        assert bar.sell_volume == 0.0

    def test_all_bar_types_accepted(self) -> None:
        """AggregatedBar must accept every BarType variant."""
        for bt in BarType:
            bar: AggregatedBar = _make_bar(bar_type=bt)
            assert bar.bar_type == bt

    def test_frozen_prevents_mutation(self) -> None:
        """Attempting to mutate a frozen AggregatedBar field must raise ValidationError."""
        bar: AggregatedBar = _make_bar()
        with pytest.raises(ValidationError):
            bar.tick_count = 999  # type: ignore[misc]

    def test_decimal_precision_preserved(self) -> None:
        """open/high/low/close/vwap as Decimal must preserve exact precision."""
        precise_price: Decimal = Decimal("42000.12345678")
        bar: AggregatedBar = _make_bar(open_=precise_price, high=precise_price, low=precise_price, close=precise_price)
        assert bar.open == precise_price


# ---------------------------------------------------------------------------
# Invariant violations
# ---------------------------------------------------------------------------


class TestAggregatedBarInvariants:
    """Tests for AggregatedBar invariant enforcement."""

    def test_high_less_than_low_raises(self) -> None:
        """High < low must raise ValueError via model_validator."""
        with pytest.raises(ValidationError, match="high"):
            _make_bar(high=Decimal("41000.00"), low=Decimal("42000.00"))

    def test_negative_volume_raises(self) -> None:
        """Volume < 0 must raise ValueError."""
        with pytest.raises(ValidationError, match="volume"):
            _make_bar(volume=-1.0)

    def test_zero_tick_count_raises(self) -> None:
        """tick_count=0 must raise ValueError (ge=1)."""
        with pytest.raises(ValidationError, match="tick_count"):
            _make_bar(tick_count=0)

    def test_negative_tick_count_raises(self) -> None:
        """tick_count=-1 must raise ValueError."""
        with pytest.raises(ValidationError, match="tick_count"):
            _make_bar(tick_count=-1)

    def test_negative_buy_volume_raises(self) -> None:
        """buy_volume < 0 must raise ValueError."""
        with pytest.raises(ValidationError, match="buy_volume"):
            _make_bar(buy_volume=-1.0)

    def test_negative_sell_volume_raises(self) -> None:
        """sell_volume < 0 must raise ValueError."""
        with pytest.raises(ValidationError, match="sell_volume"):
            _make_bar(sell_volume=-1.0)

    def test_buy_plus_sell_exceeds_volume_raises(self) -> None:
        """buy_volume + sell_volume > volume must raise ValueError."""
        with pytest.raises(ValidationError):
            _make_bar(volume=100.0, buy_volume=80.0, sell_volume=30.0)

    def test_start_ts_equal_to_end_ts_raises(self) -> None:
        """start_ts == end_ts must raise ValueError (start must be strictly before end)."""
        same_ts: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValidationError, match="start_ts"):
            _make_bar(start_ts=same_ts, end_ts=same_ts)

    def test_start_ts_after_end_ts_raises(self) -> None:
        """start_ts > end_ts must raise ValueError."""
        start: datetime = datetime(2024, 1, 2, tzinfo=UTC)
        end: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        with pytest.raises(ValidationError, match="start_ts"):
            _make_bar(start_ts=start, end_ts=end)

    def test_buy_sell_exceed_volume_by_tiny_epsilon_raises(self) -> None:
        """Buy + sell exceeding volume by more than epsilon must raise ValueError."""
        epsilon: float = 1e-9
        vol: float = 100.0
        # This should exceed the allowed epsilon tolerance
        with pytest.raises(ValidationError):
            _make_bar(volume=vol, buy_volume=50.0, sell_volume=50.0 + epsilon * 100)

    @pytest.mark.parametrize(
        ("high", "low"),
        [
            (Decimal("42000.00"), Decimal("42001.00")),
            (Decimal("0.01"), Decimal("0.02")),
        ],
    )
    def test_high_below_low_parametrized(self, high: Decimal, low: Decimal) -> None:
        """High < low must always raise regardless of magnitude."""
        with pytest.raises(ValidationError, match="high"):
            _make_bar(high=high, low=low)

    def test_one_second_apart_start_end_is_valid(self) -> None:
        """start_ts and end_ts one second apart must be valid."""
        start: datetime = datetime(2024, 1, 1, tzinfo=UTC)
        end: datetime = start + timedelta(seconds=1)
        bar: AggregatedBar = _make_bar(start_ts=start, end_ts=end)
        assert bar.end_ts > bar.start_ts
