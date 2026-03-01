"""Unit tests for ``BinanceFetcher`` infrastructure implementation.

``_raw_to_candle()`` is tested without mocking (pure static method).
``fetch_ohlcv()`` and related methods are tested with a mocked
``binance.client.Client`` to avoid real network calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.app.ingestion.domain.exceptions import FetchError, RateLimitError
from src.app.ingestion.domain.value_objects import (
    BinanceKlineInterval,
    FetchRequest,
    TIMEFRAME_INTERVAL_MS,
)
from src.app.ingestion.infrastructure.binance_fetcher import BinanceFetcher
from src.app.ingestion.infrastructure.settings import BinanceSettings
from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe
from src.tests.conftest import START_DT, make_asset
from src.tests.ingestion.conftest import (
    KLINE_OPEN_TIME_MS,
    SAMPLE_KLINE_ROW,
    build_kline_batch,
    make_binance_settings,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BTCUSDT: Asset = make_asset("BTCUSDT")
_H1: Timeframe = Timeframe.H1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fetcher_with_mock_client() -> tuple[BinanceFetcher, MagicMock]:
    """Build a BinanceFetcher whose internal Binance Client is replaced with a MagicMock.

    Returns:
        Tuple of (BinanceFetcher instance, the MagicMock replacing the client).
    """
    settings: BinanceSettings = make_binance_settings()
    with patch(
        "src.app.ingestion.infrastructure.binance_fetcher.Client",
        autospec=True,
    ) as mock_client_cls:
        mock_client_instance: MagicMock = MagicMock()
        mock_client_cls.return_value = mock_client_instance
        fetcher: BinanceFetcher = BinanceFetcher(settings)
        # Replace _retryer with a simple passthrough so tests are fast
        fetcher._retryer = None  # type: ignore[assignment]
    return fetcher, mock_client_instance


# ---------------------------------------------------------------------------
# _raw_to_candle tests (static, no mocking)
# ---------------------------------------------------------------------------


class TestRawToCandle:
    """Tests for ``BinanceFetcher._raw_to_candle()`` static method."""

    def test_timestamp_converted_from_ms_to_utc_datetime(self) -> None:
        """Millisecond open_time must be converted to a UTC-aware datetime."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert candle.timestamp == START_DT
        assert candle.timestamp.tzinfo is UTC

    def test_open_price_converted_to_decimal(self) -> None:
        """Open price string must be converted to a Decimal with full precision."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert candle.open == Decimal("42000.00")

    def test_high_price_converted_to_decimal(self) -> None:
        """High price string must be converted to the correct Decimal."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert candle.high == Decimal("42500.00")

    def test_low_price_converted_to_decimal(self) -> None:
        """Low price string must be converted to the correct Decimal."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert candle.low == Decimal("41800.00")

    def test_close_price_converted_to_decimal(self) -> None:
        """Close price string must be converted to the correct Decimal."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert candle.close == Decimal("42200.00")

    def test_volume_converted_to_float(self) -> None:
        """Volume string must be converted to a float."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert isinstance(candle.volume, float)
        assert candle.volume == pytest.approx(150.5)

    def test_asset_is_passed_through_correctly(self) -> None:
        """The asset on the resulting candle must match the input asset."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert candle.asset == _BTCUSDT

    def test_timeframe_is_passed_through_correctly(self) -> None:
        """The timeframe on the resulting candle must match the input timeframe."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, _H1
        )
        assert candle.timeframe == _H1

    @pytest.mark.parametrize(
        ("timeframe",),
        [(Timeframe.H1,), (Timeframe.H4,), (Timeframe.D1,)],
    )
    def test_raw_to_candle_with_all_timeframes(self, timeframe: Timeframe) -> None:
        """_raw_to_candle() must succeed for every supported Timeframe."""
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(
            SAMPLE_KLINE_ROW, _BTCUSDT, timeframe
        )
        assert candle.timeframe == timeframe

    def test_epoch_zero_produces_utc_epoch_datetime(self) -> None:
        """A 0 ms open_time must produce the UTC epoch (1970-01-01 00:00:00)."""
        row: list[Any] = list(SAMPLE_KLINE_ROW)
        row[0] = 0
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(row, _BTCUSDT, _H1)
        assert candle.timestamp == datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_integer_open_time_is_accepted(self) -> None:
        """open_time provided as an integer (not string) must still be parsed."""
        row: list[Any] = list(SAMPLE_KLINE_ROW)
        row[0] = int(KLINE_OPEN_TIME_MS)
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(row, _BTCUSDT, _H1)
        assert candle.timestamp == START_DT

    def test_high_precision_decimal_preserved(self) -> None:
        """Prices with 8 decimal places (Binance format) must not lose precision."""
        row: list[Any] = list(SAMPLE_KLINE_ROW)
        row[1] = "42000.12345678"  # open with 8 decimal places
        row[2] = "42000.12345678"  # high == open (valid since high >= low)
        candle: OHLCVCandle = BinanceFetcher._raw_to_candle(row, _BTCUSDT, _H1)
        assert candle.open == Decimal("42000.12345678")


# ---------------------------------------------------------------------------
# fetch_ohlcv tests (mocked Binance client)
# ---------------------------------------------------------------------------


class TestFetchOhlcv:
    """Tests for ``BinanceFetcher.fetch_ohlcv()`` with a mocked Binance client."""

    def _make_fetch_request(
        self,
        start_ms: int,
        end_ms: int,
        timeframe: Timeframe = Timeframe.H1,
    ) -> FetchRequest:
        """Build a FetchRequest from millisecond timestamps.

        Args:
            start_ms: Range start in milliseconds since epoch.
            end_ms: Range end in milliseconds since epoch.
            timeframe: Candlestick interval.

        Returns:
            Configured FetchRequest.
        """
        start_dt: datetime = datetime.fromtimestamp(start_ms / 1000, tz=UTC)
        end_dt: datetime = datetime.fromtimestamp(end_ms / 1000, tz=UTC)
        return FetchRequest(
            asset=_BTCUSDT,
            timeframe=timeframe,
            date_range=DateRange(start=start_dt, end=end_dt),
        )

    def test_single_batch_returns_correct_candles(self) -> None:
        """When the API returns fewer than batch_size rows, all must be returned."""
        interval_ms: int = TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1]
        start_ms: int = KLINE_OPEN_TIME_MS
        # 5 candles, all within a single batch
        batch: list[list[Any]] = build_kline_batch(start_ms, 5, interval_ms)
        # next batch starts after the last candle (outside the range) -> empty
        end_ms: int = start_ms + 5 * interval_ms

        settings: BinanceSettings = make_binance_settings()

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance
            # First call: return 5 rows; second call (cursor advanced past end) never reached
            mock_instance.get_klines.return_value = batch

            fetcher: BinanceFetcher = BinanceFetcher(settings)

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms)
        candles: list[OHLCVCandle] = fetcher.fetch_ohlcv(request)

        assert len(candles) == 5
        assert candles[0].timestamp == datetime.fromtimestamp(start_ms / 1000, tz=UTC)

    def test_empty_api_response_returns_empty_list(self) -> None:
        """When the API returns an empty list, fetch_ohlcv() must return an empty list."""
        start_ms: int = KLINE_OPEN_TIME_MS
        end_ms: int = start_ms + TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1] * 10

        settings: BinanceSettings = make_binance_settings()

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.get_klines.return_value = []

            fetcher: BinanceFetcher = BinanceFetcher(settings)

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms)
        candles: list[OHLCVCandle] = fetcher.fetch_ohlcv(request)

        assert candles == []
        # get_klines must have been called exactly once (then loop breaks on empty)
        mock_instance.get_klines.assert_called_once()

    def test_pagination_stitches_multiple_batches(self) -> None:
        """When API returns full batch_size rows, fetch_ohlcv() must request another batch.

        The loop condition is ``current_start < end_ms``.  After consuming batch2,
        ``next_start = batch2_start + batch_size * interval_ms``.  We set
        ``end_ms = next_start + 1`` so that the loop must issue a third API call
        (which returns empty) before terminating.
        """
        interval_ms: int = TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1]
        start_ms: int = KLINE_OPEN_TIME_MS
        batch_size: int = 3  # small batch so we can test pagination without many rows

        settings: BinanceSettings = BinanceSettings.model_construct(
            api_key="test_key",
            secret_key="test_secret",
            max_retries=1,
            retry_min_wait=0,
            retry_max_wait=1,
            batch_size=batch_size,
        )

        # First batch: rows 0-2 (3 rows)
        batch1: list[list[Any]] = build_kline_batch(start_ms, batch_size, interval_ms)
        # Second batch: rows 3-5 (3 rows)
        batch2_start: int = start_ms + batch_size * interval_ms
        batch2: list[list[Any]] = build_kline_batch(batch2_start, batch_size, interval_ms)
        # After batch2, next_start = batch2_start + batch_size * interval_ms.
        # Set end_ms 1 ms beyond that so the loop must attempt a third call.
        third_start: int = batch2_start + batch_size * interval_ms
        end_ms: int = third_start + 1  # forces a third iteration that returns empty

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.get_klines.side_effect = [batch1, batch2, []]

            fetcher: BinanceFetcher = BinanceFetcher(settings)

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms)
        candles: list[OHLCVCandle] = fetcher.fetch_ohlcv(request)

        assert len(candles) == batch_size * 2
        assert mock_instance.get_klines.call_count == 3

    def test_non_advancing_cursor_breaks_loop(self) -> None:
        """If last_open_time + interval_ms <= current_start, the loop must break."""
        interval_ms: int = TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1]
        start_ms: int = KLINE_OPEN_TIME_MS
        end_ms: int = start_ms + 10 * interval_ms

        # Return a row whose open_time + interval_ms equals start_ms (non-advancing)
        stale_row: list[Any] = list(SAMPLE_KLINE_ROW)
        # open_time such that open_time + interval_ms == start_ms
        stale_row[0] = start_ms - interval_ms

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.get_klines.return_value = [stale_row]

            fetcher: BinanceFetcher = BinanceFetcher(settings=make_binance_settings())

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms)
        candles: list[OHLCVCandle] = fetcher.fetch_ohlcv(request)

        # One candle was converted before the stale cursor check
        assert len(candles) == 1
        # Loop must not have called get_klines again after detecting stale cursor
        mock_instance.get_klines.assert_called_once()

    def test_rate_limit_error_raised_on_http_429(self) -> None:
        """A BinanceAPIException with status_code 429 must raise RateLimitError."""
        from binance.exceptions import BinanceAPIException  # type: ignore[import-untyped]

        start_ms: int = KLINE_OPEN_TIME_MS
        end_ms: int = start_ms + TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1] * 2

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance

            # Simulate a BinanceAPIException with status_code 429
            rate_limit_exc: BinanceAPIException = BinanceAPIException(
                response=MagicMock(status_code=429),
                status_code=429,
                text="Too many requests",
            )
            mock_instance.get_klines.side_effect = rate_limit_exc

            fetcher: BinanceFetcher = BinanceFetcher(settings=make_binance_settings())

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms)

        with pytest.raises(RateLimitError):
            fetcher.fetch_ohlcv(request)

    def test_retryable_error_exhausts_retries_propagates_original_exception(self) -> None:
        """After max_retries attempts, the underlying transient exception must propagate.

        ``BinanceFetcher`` uses ``tenacity`` with ``reraise=True``, which means the
        original exception is re-raised after retries are exhausted rather than being
        wrapped in ``tenacity.RetryError``.  Callers should expect the original
        exception type (e.g. ``ConnectionError``) to propagate from
        ``_fetch_klines_batch`` when all retries fail.
        """
        from requests.exceptions import ConnectionError as RequestsConnectionError
        from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_none

        start_ms: int = KLINE_OPEN_TIME_MS
        end_ms: int = start_ms + TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1] * 2

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.get_klines.side_effect = RequestsConnectionError("timeout")

            settings: BinanceSettings = make_binance_settings()
            fetcher: BinanceFetcher = BinanceFetcher(settings=settings)

        # Replace the retryer with one that retries twice with no wait, reraise=True
        fetcher._retryer = Retrying(  # type: ignore[assignment]
            stop=stop_after_attempt(2),
            wait=wait_none(),
            retry=retry_if_exception_type(RequestsConnectionError),
            reraise=True,
        )

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms)

        # With reraise=True, tenacity re-raises the original ConnectionError
        # (not RetryError), so the underlying exception propagates to the caller.
        with pytest.raises(RequestsConnectionError):
            fetcher.fetch_ohlcv(request)

        # Confirm all retry attempts were made
        assert mock_instance.get_klines.call_count == 2

    def test_correct_symbol_passed_to_api(self) -> None:
        """fetch_ohlcv() must pass the asset symbol to the Binance client."""
        interval_ms: int = TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1]
        start_ms: int = KLINE_OPEN_TIME_MS
        end_ms: int = start_ms + interval_ms

        batch: list[list[Any]] = build_kline_batch(start_ms, 1, interval_ms)

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.get_klines.return_value = batch

            fetcher: BinanceFetcher = BinanceFetcher(settings=make_binance_settings())

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms)
        fetcher.fetch_ohlcv(request)

        call_kwargs: dict[str, Any] = mock_instance.get_klines.call_args.kwargs
        assert call_kwargs["symbol"] == "BTCUSDT"

    def test_correct_interval_passed_to_api(self) -> None:
        """fetch_ohlcv() must pass the correct Binance interval string to the API."""
        interval_ms: int = TIMEFRAME_INTERVAL_MS[BinanceKlineInterval.H1]
        start_ms: int = KLINE_OPEN_TIME_MS
        end_ms: int = start_ms + interval_ms

        batch: list[list[Any]] = build_kline_batch(start_ms, 1, interval_ms)

        with patch(
            "src.app.ingestion.infrastructure.binance_fetcher.Client",
            autospec=True,
        ) as mock_cls:
            mock_instance: MagicMock = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.get_klines.return_value = batch

            fetcher: BinanceFetcher = BinanceFetcher(settings=make_binance_settings())

        request: FetchRequest = self._make_fetch_request(start_ms, end_ms, Timeframe.H1)
        fetcher.fetch_ohlcv(request)

        call_kwargs: dict[str, Any] = mock_instance.get_klines.call_args.kwargs
        assert call_kwargs["interval"] == "1h"
