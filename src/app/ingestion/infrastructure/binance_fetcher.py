"""Binance API client implementing the ``IMarketDataFetcher`` protocol."""

from __future__ import annotations

from datetime import datetime, UTC
from decimal import Decimal
from typing import Any

from binance.client import Client  # type: ignore[import-untyped]
from binance.exceptions import BinanceAPIException, BinanceRequestException  # type: ignore[import-untyped]
from loguru import logger
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, Timeout
from tenacity import retry_if_exception_type, RetryError, Retrying, stop_after_attempt, wait_exponential

from src.app.ingestion.domain.exceptions import FetchError, RateLimitError
from src.app.ingestion.domain.value_objects import BinanceKlineInterval, FetchRequest, TIMEFRAME_INTERVAL_MS
from src.app.ingestion.infrastructure.settings import BinanceSettings
from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.value_objects import Asset, Timeframe


_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    BinanceAPIException,
    BinanceRequestException,
    Timeout,
    RequestsConnectionError,
    HTTPError,
)
"""Exception types that trigger automatic retry with exponential backoff."""

_RATE_LIMIT_HTTP_CODE: int = 429
"""HTTP status code indicating Binance rate limit exhaustion."""


class BinanceFetcher:
    """Fetches OHLCV candle data from the Binance REST API.

    Implements the ``IMarketDataFetcher`` protocol.  Handles paginated
    fetching (up to ``batch_size`` klines per request), exponential-backoff
    retries on transient errors, and conversion of raw Binance responses
    into domain ``OHLCVCandle`` entities.

    Args:
        settings: Binance API connection and retry configuration.
    """

    def __init__(self, settings: BinanceSettings) -> None:
        """Initialise the fetcher with Binance API credentials and retry config.

        Args:
            settings: Binance API connection and retry configuration.
        """
        self._settings: BinanceSettings = settings
        self._client: Client = Client(
            api_key=settings.api_key,
            api_secret=settings.secret_key,
        )
        self._retryer: Retrying = Retrying(  # type: ignore[type-arg]
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=settings.retry_min_wait,
                max=settings.retry_max_wait,
            ),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            reraise=True,
        )
        logger.info("BinanceFetcher initialised.")

    def fetch_ohlcv(self, request: FetchRequest) -> list[OHLCVCandle]:
        """Fetch OHLCV candles for the given request.

        Paginates through the Binance klines endpoint in batches of
        ``batch_size``, advancing the cursor by the interval duration
        after each batch.

        Args:
            request: Specifies the asset, timeframe, and date range to fetch.

        Returns:
            A list of ``OHLCVCandle`` entities ordered by timestamp.

        """
        interval: BinanceKlineInterval = BinanceKlineInterval.from_timeframe(request.timeframe)
        interval_ms: int = TIMEFRAME_INTERVAL_MS[interval]

        start_ms: int = int(request.date_range.start.timestamp() * 1000)
        end_ms: int = int(request.date_range.end.timestamp() * 1000)

        logger.info(
            "Fetching OHLCV | asset={} timeframe={} start={} end={}",
            request.asset.symbol,
            request.timeframe.value,
            request.date_range.start.isoformat(),
            request.date_range.end.isoformat(),
        )

        candles: list[OHLCVCandle] = []
        current_start: int = start_ms

        while current_start < end_ms:
            batch: list[list[Any]] = self._fetch_klines_batch(
                symbol=request.asset.symbol,
                interval=interval,
                start_time=current_start,
            )

            if len(batch) == 0:
                logger.warning(
                    "No kline data returned | asset={} timeframe={} cursor={}",
                    request.asset.symbol,
                    interval.value,
                    current_start,
                )
                break

            candles.extend(
                self._raw_to_candle(row, request.asset, request.timeframe)
                for row in batch
            )

            last_open_time: int = int(batch[-1][0])
            next_start: int = last_open_time + interval_ms

            if next_start <= current_start:
                logger.error(
                    "Non-advancing kline cursor | asset={} timeframe={} current={} next={}",
                    request.asset.symbol,
                    interval.value,
                    current_start,
                    next_start,
                )
                break

            current_start = next_start

        logger.info(
            "Fetched {} candles | asset={} timeframe={}",
            len(candles),
            request.asset.symbol,
            request.timeframe.value,
        )
        return candles

    def _fetch_klines_batch(
        self,
        symbol: str,
        interval: BinanceKlineInterval,
        start_time: int,
    ) -> list[list[Any]]:
        """Fetch a single batch of klines from the Binance API.

        Uses ``tenacity`` retry with exponential backoff on transient errors.
        Rate-limit responses (HTTP 429) are detected and raised as
        ``RateLimitError`` without retry.

        Args:
            symbol: Trading pair symbol (e.g. ``BTCUSDT``).
            interval: Binance kline interval.
            start_time: Start timestamp in milliseconds since epoch.

        Returns:
            Raw kline records as returned by the Binance API.

        Raises:
            FetchError: If all retry attempts are exhausted.
        """
        data: list[list[Any]] = []
        try:
            for attempt in self._retryer:  # type: ignore[union-attr]
                with attempt:
                    data = self._call_klines_api(symbol, interval, start_time)
        except RetryError as exc:
            msg: str = f"All {self._settings.max_retries} retries exhausted for {symbol}"
            raise FetchError(msg) from exc
        return data

    def _call_klines_api(
        self,
        symbol: str,
        interval: BinanceKlineInterval,
        start_time: int,
    ) -> list[list[Any]]:
        """Execute a single Binance klines API call.

        Checks for rate-limit responses and raises ``RateLimitError``
        (which is **not** retryable) so it propagates immediately.

        Args:
            symbol: Trading pair symbol (e.g. ``BTCUSDT``).
            interval: Binance kline interval.
            start_time: Start timestamp in milliseconds since epoch.

        Returns:
            Raw kline records from the Binance API.

        Raises:
            RateLimitError: If the API returns HTTP 429.
            BinanceAPIException: On other Binance API errors (retryable).
        """
        try:
            result: list[list[Any]] = self._client.get_klines(  # type: ignore[assignment]
                symbol=symbol,
                interval=interval.value,
                startTime=start_time,
                limit=self._settings.batch_size,
            )
        except BinanceAPIException as exc:
            if exc.status_code == _RATE_LIMIT_HTTP_CODE:  # type: ignore[union-attr]
                msg: str = f"Binance rate limit hit for {symbol}"
                raise RateLimitError(msg) from exc
            logger.warning(
                "Binance API error | symbol={} | {}",
                symbol,
                exc,
            )
            raise
        except (*_RETRYABLE_EXCEPTIONS,) as exc:
            logger.warning(
                "Retryable network error | symbol={} | {}",
                symbol,
                exc,
            )
            raise
        return result  # type: ignore[return-value]

    @staticmethod
    def _raw_to_candle(row: list[Any], asset: Asset, timeframe: Timeframe) -> OHLCVCandle:
        """Convert a raw Binance kline row to an ``OHLCVCandle`` entity.

        Binance kline response indices:
            ``[0]`` open_time (ms), ``[1]`` open, ``[2]`` high,
            ``[3]`` low, ``[4]`` close, ``[5]`` volume.

        Args:
            row: A single raw kline record from the Binance API.
            asset: The trading pair asset.
            timeframe: The candlestick timeframe.

        Returns:
            A validated ``OHLCVCandle`` domain entity.
        """
        open_time_ms: int = int(row[0])
        timestamp: datetime = datetime.fromtimestamp(open_time_ms / 1000, tz=UTC)

        return OHLCVCandle(
            asset=asset,
            timeframe=timeframe,
            timestamp=timestamp,
            open=Decimal(str(row[1])),
            high=Decimal(str(row[2])),
            low=Decimal(str(row[3])),
            close=Decimal(str(row[4])),
            volume=float(row[5]),
        )
