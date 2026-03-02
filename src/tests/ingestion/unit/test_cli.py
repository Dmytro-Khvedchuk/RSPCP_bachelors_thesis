"""Unit tests for the ingestion CLI parser helper functions.

Only the pure helper functions (``_parse_assets``, ``_parse_timeframes``,
``_parse_date``) are tested here.  The Typer ``app`` command itself is not
invoked so that no database connection or Binance API key is required.
"""

from __future__ import annotations

from datetime import datetime, UTC

import pytest
import typer

from src.app.ingestion.cli import _parse_assets, _parse_date, _parse_timeframes
from src.app.ohlcv.domain.value_objects import Asset, Timeframe


# ---------------------------------------------------------------------------
# _parse_assets tests
# ---------------------------------------------------------------------------


class TestParseAssets:
    """Tests for the ``_parse_assets()`` CLI helper."""

    def test_single_valid_symbol_returns_one_asset(self) -> None:
        """A single valid symbol must produce a list containing one Asset."""
        result: list[Asset] = _parse_assets("BTCUSDT")
        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"

    def test_two_valid_symbols_returns_two_assets(self) -> None:
        """Two comma-separated valid symbols must produce a list of two Assets."""
        result: list[Asset] = _parse_assets("BTCUSDT,ETHUSDT")
        assert len(result) == 2
        symbols: list[str] = [a.symbol for a in result]
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    def test_symbols_are_uppercased_automatically(self) -> None:
        """Lowercase or mixed-case symbols must be uppercased before validation."""
        result: list[Asset] = _parse_assets("btcusdt")
        assert result[0].symbol == "BTCUSDT"

    def test_whitespace_around_symbols_is_stripped(self) -> None:
        """Spaces around symbols must be stripped before validation."""
        result: list[Asset] = _parse_assets(" BTCUSDT , ETHUSDT ")
        assert len(result) == 2
        assert result[0].symbol == "BTCUSDT"
        assert result[1].symbol == "ETHUSDT"

    def test_empty_string_raises_bad_parameter(self) -> None:
        """An empty input string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_assets("")

    def test_whitespace_only_string_raises_bad_parameter(self) -> None:
        """A whitespace-only string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_assets("   ")

    def test_invalid_symbol_with_special_chars_raises_bad_parameter(self) -> None:
        """Symbols containing special characters must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_assets("INVALID!")

    def test_symbol_too_short_raises_bad_parameter(self) -> None:
        """A single-character symbol must fail Asset validation and raise BadParameter."""
        with pytest.raises(typer.BadParameter):
            _parse_assets("B")

    def test_symbol_too_long_raises_bad_parameter(self) -> None:
        """A symbol exceeding 20 characters must fail Asset validation and raise BadParameter."""
        long_symbol: str = "A" * 21
        with pytest.raises(typer.BadParameter):
            _parse_assets(long_symbol)

    def test_second_invalid_symbol_still_raises(self) -> None:
        """If any symbol in a comma-separated list is invalid, BadParameter must be raised."""
        with pytest.raises(typer.BadParameter):
            _parse_assets("BTCUSDT,INVALID!")

    @pytest.mark.parametrize(
        "raw",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AB"],
    )
    def test_valid_symbols_are_accepted(self, raw: str) -> None:
        """Various valid symbol formats must all be accepted."""
        result: list[Asset] = _parse_assets(raw)
        assert len(result) == 1
        assert result[0].symbol == raw.upper()


# ---------------------------------------------------------------------------
# _parse_timeframes tests
# ---------------------------------------------------------------------------


class TestParseTimeframes:
    """Tests for the ``_parse_timeframes()`` CLI helper."""

    def test_single_h1_returns_h1_timeframe(self) -> None:
        """Input '1h' must produce a list containing Timeframe.H1."""
        result: list[Timeframe] = _parse_timeframes("1h")
        assert result == [Timeframe.H1]

    def test_single_h4_returns_h4_timeframe(self) -> None:
        """Input '4h' must produce a list containing Timeframe.H4."""
        result: list[Timeframe] = _parse_timeframes("4h")
        assert result == [Timeframe.H4]

    def test_single_d1_returns_d1_timeframe(self) -> None:
        """Input '1d' must produce a list containing Timeframe.D1."""
        result: list[Timeframe] = _parse_timeframes("1d")
        assert result == [Timeframe.D1]

    def test_two_timeframes_returns_two_members(self) -> None:
        """Two comma-separated valid timeframe strings must return two Timeframe members."""
        result: list[Timeframe] = _parse_timeframes("1h,4h")
        assert len(result) == 2
        assert Timeframe.H1 in result
        assert Timeframe.H4 in result

    def test_all_three_timeframes_accepted(self) -> None:
        """All three valid timeframe values together must be accepted."""
        result: list[Timeframe] = _parse_timeframes("1h,4h,1d")
        assert len(result) == 3

    def test_invalid_timeframe_raises_bad_parameter(self) -> None:
        """An unrecognised interval string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_timeframes("invalid")

    def test_minute_interval_not_in_timeframe_raises_bad_parameter(self) -> None:
        """'1m' is not a valid Timeframe and must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_timeframes("1m")

    def test_empty_string_raises_bad_parameter(self) -> None:
        """An empty input string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_timeframes("")

    def test_whitespace_only_raises_bad_parameter(self) -> None:
        """A whitespace-only string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_timeframes("   ")

    def test_second_invalid_in_list_raises_bad_parameter(self) -> None:
        """If any item in a comma-separated list is invalid, BadParameter must be raised."""
        with pytest.raises(typer.BadParameter):
            _parse_timeframes("1h,bad_value")

    def test_whitespace_stripped_from_items(self) -> None:
        """Items padded with spaces must be trimmed before lookup."""
        result: list[Timeframe] = _parse_timeframes(" 1h , 4h ")
        assert len(result) == 2
        assert Timeframe.H1 in result
        assert Timeframe.H4 in result

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1h", Timeframe.H1),
            ("4h", Timeframe.H4),
            ("1d", Timeframe.D1),
        ],
    )
    def test_each_valid_value_maps_correctly(self, raw: str, expected: Timeframe) -> None:
        """Each valid timeframe string must map to the expected Timeframe member."""
        result: list[Timeframe] = _parse_timeframes(raw)
        assert result[0] == expected


# ---------------------------------------------------------------------------
# _parse_date tests
# ---------------------------------------------------------------------------


class TestParseDate:
    """Tests for the ``_parse_date()`` CLI helper."""

    def test_date_only_string_produces_utc_datetime(self) -> None:
        """A 'YYYY-MM-DD' string must produce a UTC-aware datetime at midnight."""
        result: datetime = _parse_date("2020-01-01")
        assert result == datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert result.tzinfo is UTC

    def test_datetime_string_produces_correct_datetime(self) -> None:
        """A 'YYYY-MM-DDTHH:MM:SS' string (naive) must be interpreted as UTC."""
        result: datetime = _parse_date("2024-06-15T12:30:00")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 0
        assert result.tzinfo is UTC

    def test_tz_aware_string_is_accepted(self) -> None:
        """An ISO-8601 string with explicit UTC offset must be accepted."""
        result: datetime = _parse_date("2024-01-01T00:00:00+00:00")
        assert result.tzinfo is not None

    def test_invalid_date_string_raises_bad_parameter(self) -> None:
        """An unparseable string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_date("not-a-date")

    def test_empty_string_raises_bad_parameter(self) -> None:
        """An empty string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_date("")

    def test_completely_invalid_format_raises_bad_parameter(self) -> None:
        """A completely non-ISO string must raise ``typer.BadParameter``."""
        with pytest.raises(typer.BadParameter):
            _parse_date("01/01/2024")

    def test_naive_datetime_gets_utc_timezone_attached(self) -> None:
        """A naive (no tzinfo) parsed datetime must have UTC attached."""
        result: datetime = _parse_date("2023-12-31T23:59:59")
        assert result.tzinfo is UTC

    def test_result_is_timezone_aware(self) -> None:
        """The returned datetime must always carry timezone information."""
        result: datetime = _parse_date("2024-03-01")
        assert result.tzinfo is not None

    @pytest.mark.parametrize(
        ("raw", "expected_year", "expected_month", "expected_day"),
        [
            ("2020-01-01", 2020, 1, 1),
            ("2023-12-31", 2023, 12, 31),
            ("2024-06-15", 2024, 6, 15),
        ],
    )
    def test_date_components_parsed_correctly(
        self,
        raw: str,
        expected_year: int,
        expected_month: int,
        expected_day: int,
    ) -> None:
        """Year, month, and day must be extracted correctly from the ISO string."""
        result: datetime = _parse_date(raw)
        assert result.year == expected_year
        assert result.month == expected_month
        assert result.day == expected_day
