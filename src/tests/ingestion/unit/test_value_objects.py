"""Unit tests for ingestion domain value objects.

Tests cover ``BinanceKlineInterval``, ``FetchRequest``, and
``TIMEFRAME_INTERVAL_MS``.
"""

from __future__ import annotations

from datetime import datetime, UTC

import pytest
from pydantic import ValidationError

from src.app.ingestion.domain.value_objects import BinanceKlineInterval, FetchRequest, TIMEFRAME_INTERVAL_MS
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe
from src.tests.conftest import make_date_range, START_DT


# ---------------------------------------------------------------------------
# Constants (local to this file — _END differs from shared END_DT)
# ---------------------------------------------------------------------------

_END: datetime = datetime(2024, 2, 1, tzinfo=UTC)

_M1_MS: int = 60_000
_H1_MS: int = 3_600_000
_H4_MS: int = 14_400_000
_D1_MS: int = 86_400_000


# ---------------------------------------------------------------------------
# BinanceKlineInterval
# ---------------------------------------------------------------------------


class TestBinanceKlineInterval:
    """Tests for the ``BinanceKlineInterval`` enum."""

    def test_m1_value_matches_binance_api_string(self) -> None:
        """M1 interval value must equal the Binance API string '1m'."""
        assert BinanceKlineInterval.M1 == "1m"

    def test_h1_value_matches_binance_api_string(self) -> None:
        """H1 interval value must equal the Binance API string '1h'."""
        assert BinanceKlineInterval.H1 == "1h"

    def test_h4_value_matches_binance_api_string(self) -> None:
        """H4 interval value must equal the Binance API string '4h'."""
        assert BinanceKlineInterval.H4 == "4h"

    def test_d1_value_matches_binance_api_string(self) -> None:
        """D1 interval value must equal the Binance API string '1d'."""
        assert BinanceKlineInterval.D1 == "1d"

    @pytest.mark.parametrize(
        ("timeframe", "expected"),
        [
            (Timeframe.H1, BinanceKlineInterval.H1),
            (Timeframe.H4, BinanceKlineInterval.H4),
            (Timeframe.D1, BinanceKlineInterval.D1),
        ],
    )
    def test_from_timeframe_maps_each_member_correctly(
        self,
        timeframe: Timeframe,
        expected: BinanceKlineInterval,
    ) -> None:
        """from_timeframe() must return the correct interval for every Timeframe member."""
        result: BinanceKlineInterval = BinanceKlineInterval.from_timeframe(timeframe)
        assert result == expected

    def test_from_timeframe_all_domain_timeframes_covered(self) -> None:
        """Every member of Timeframe must map to a BinanceKlineInterval without raising."""
        for tf in Timeframe:
            result: BinanceKlineInterval = BinanceKlineInterval.from_timeframe(tf)
            assert isinstance(result, BinanceKlineInterval)

    def test_from_timeframe_invalid_value_raises_value_error(self) -> None:
        """from_timeframe() must raise ValueError for a Timeframe it cannot map."""
        with pytest.raises(ValueError, match="No BinanceKlineInterval"):
            _fake: str = "__INVALID__"
            BinanceKlineInterval.from_timeframe(_fake)  # type: ignore[arg-type]

    def test_from_timeframe_returns_strable_value(self) -> None:
        """The returned interval must be usable as a plain string (StrEnum contract)."""
        interval: BinanceKlineInterval = BinanceKlineInterval.from_timeframe(Timeframe.H1)
        assert str(interval) == "1h"


# ---------------------------------------------------------------------------
# FetchRequest
# ---------------------------------------------------------------------------


class TestFetchRequest:
    """Tests for the ``FetchRequest`` value object."""

    def test_construction_with_valid_fields(self) -> None:
        """FetchRequest must be constructable with valid asset, timeframe, and date_range."""
        asset: Asset = Asset(symbol="BTCUSDT")
        timeframe: Timeframe = Timeframe.H1
        dr: DateRange = make_date_range(end=_END)

        request: FetchRequest = FetchRequest(
            asset=asset,
            timeframe=timeframe,
            date_range=dr,
        )

        assert request.asset.symbol == "BTCUSDT"
        assert request.timeframe == Timeframe.H1
        assert request.date_range.start == START_DT
        assert request.date_range.end == _END

    def test_frozen_assignment_raises_validation_error(self) -> None:
        """Mutating a frozen FetchRequest field must raise ValidationError."""
        request: FetchRequest = FetchRequest(
            asset=Asset(symbol="BTCUSDT"),
            timeframe=Timeframe.H1,
            date_range=make_date_range(end=_END),
        )

        with pytest.raises(ValidationError):
            request.asset = Asset(symbol="ETHUSDT")  # type: ignore[misc]

    def test_construction_with_all_timeframes(self) -> None:
        """FetchRequest must accept every supported Timeframe variant."""
        dr: DateRange = make_date_range(end=_END)
        asset: Asset = Asset(symbol="BTCUSDT")

        for tf in Timeframe:
            req: FetchRequest = FetchRequest(
                asset=asset,
                timeframe=tf,
                date_range=dr,
            )
            assert req.timeframe == tf

    def test_missing_required_field_raises_validation_error(self) -> None:
        """Omitting a required field must raise ValidationError."""
        with pytest.raises(ValidationError):
            FetchRequest(  # type: ignore[call-arg]
                asset=Asset(symbol="BTCUSDT"),
                timeframe=Timeframe.H1,
                # date_range deliberately omitted
            )


# ---------------------------------------------------------------------------
# TIMEFRAME_INTERVAL_MS
# ---------------------------------------------------------------------------


class TestTimeframeIntervalMs:
    """Tests for the ``TIMEFRAME_INTERVAL_MS`` mapping constant."""

    def test_all_intervals_have_entry(self) -> None:
        """Every BinanceKlineInterval member must appear as a key in the mapping."""
        for interval in BinanceKlineInterval:
            assert interval in TIMEFRAME_INTERVAL_MS, (
                f"{interval!r} missing from TIMEFRAME_INTERVAL_MS"
            )

    def test_m1_duration_is_60_seconds_in_ms(self) -> None:
        """M1 interval must correspond to exactly 60 000 ms."""
        assert TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.M1] == _M1_MS

    def test_h1_duration_is_3600_seconds_in_ms(self) -> None:
        """H1 interval must correspond to exactly 3 600 000 ms."""
        assert TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1] == _H1_MS

    def test_h4_duration_is_14400_seconds_in_ms(self) -> None:
        """H4 interval must correspond to exactly 14 400 000 ms."""
        assert TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H4] == _H4_MS

    def test_d1_duration_is_86400_seconds_in_ms(self) -> None:
        """D1 interval must correspond to exactly 86 400 000 ms."""
        assert TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.D1] == _D1_MS

    def test_durations_are_strictly_increasing(self) -> None:
        """Shorter intervals must have smaller millisecond values than longer ones."""
        assert _M1_MS < _H1_MS < _H4_MS < _D1_MS

    @pytest.mark.parametrize(
        ("interval", "expected_ms"),
        [
            (BinanceKlineInterval.M1, _M1_MS),
            (BinanceKlineInterval.H1, _H1_MS),
            (BinanceKlineInterval.H4, _H4_MS),
            (BinanceKlineInterval.D1, _D1_MS),
        ],
    )
    def test_duration_parametrized(
        self,
        interval: BinanceKlineInterval,
        expected_ms: int,
    ) -> None:
        """Each interval duration must match its expected millisecond count."""
        assert TIMEFRAME_INTERVAL_MS[interval] == expected_ms

    def test_h1_equals_60_times_m1(self) -> None:
        """H1 duration must be exactly 60x M1 duration."""
        assert TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1] == (
            60 * TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.M1]
        )

    def test_h4_equals_4_times_h1(self) -> None:
        """H4 duration must be exactly 4x H1 duration."""
        assert TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H4] == (
            4 * TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1]
        )

    def test_d1_equals_24_times_h1(self) -> None:
        """D1 duration must be exactly 24x H1 duration."""
        assert TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.D1] == (
            24 * TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1]
        )
