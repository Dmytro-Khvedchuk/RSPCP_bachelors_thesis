"""Ingestion-specific test fixtures, fakes, and builders.

Provides Protocol-compatible fake implementations, Binance kline builders,
settings helpers, and fixtures used across all ingestion test sub-packages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.app.ingestion.domain.value_objects import FetchRequest
from src.app.ingestion.infrastructure.settings import BinanceSettings
from src.app.ohlcv.domain.entities import OHLCVCandle
from src.app.ohlcv.domain.value_objects import Asset, DateRange, Timeframe, TemporalSplit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KLINE_OPEN_TIME_MS: int = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC

SAMPLE_KLINE_ROW: list[Any] = [
    KLINE_OPEN_TIME_MS,
    "42000.00",  # open
    "42500.00",  # high
    "41800.00",  # low
    "42200.00",  # close
    "150.50000000",  # volume
    1_704_070_799_999,  # close_time_ms
    "6321234.50",  # quote_asset_volume
    1234,  # number_of_trades
    "75.25",  # taker_buy_base
    "3160617.25",  # taker_buy_quote
    "0",  # ignore
]

FAKE_API_KEY: str = "fake_binance_api_key_12345"
FAKE_SECRET_KEY: str = "fake_binance_secret_key_67890"


# ---------------------------------------------------------------------------
# BinanceSettings helpers
# ---------------------------------------------------------------------------


class BinanceSettingsNoEnvFile(BaseSettings):
    """BinanceSettings variant that disables ``.env`` file loading.

    Used for validation tests where we need to assert that missing required
    fields raise ``ValidationError`` — without the real ``.env`` silently
    providing those values.
    """

    model_config = SettingsConfigDict(
        env_prefix="BINANCE_",
        env_file=None,  # no .env fallback
        extra="ignore",
    )

    api_key: str = Field(description="Binance API key.")
    secret_key: str = Field(description="Binance secret key.")
    max_retries: int = Field(default=5, ge=1)
    retry_min_wait: int = Field(default=1, ge=1)
    retry_max_wait: int = Field(default=10, ge=1)
    batch_size: int = Field(default=1000, ge=1, le=1000)


def make_binance_settings() -> BinanceSettings:
    """Return minimal ``BinanceSettings`` without reading from environment.

    Returns:
        BinanceSettings constructed via ``model_construct`` (no validation).
    """
    return BinanceSettings.model_construct(
        api_key="test_api_key",
        secret_key="test_secret_key",
        max_retries=1,
        retry_min_wait=0,
        retry_max_wait=1,
        batch_size=1000,
    )


# ---------------------------------------------------------------------------
# Kline builders
# ---------------------------------------------------------------------------


def build_kline_batch(
    start_ms: int,
    count: int,
    interval_ms: int,
) -> list[list[Any]]:
    """Build ``count`` synthetic kline rows starting from ``start_ms``.

    Args:
        start_ms: First row open_time in milliseconds.
        count: Number of rows to generate.
        interval_ms: Gap between consecutive rows in milliseconds.

    Returns:
        List of raw kline rows matching the Binance API format.
    """
    rows: list[list[Any]] = []
    for i in range(count):
        open_ms: int = start_ms + i * interval_ms
        row: list[Any] = [
            open_ms,
            "42000.00",
            "42500.00",
            "41800.00",
            "42200.00",
            "1.0",
            open_ms + interval_ms - 1,
            "42000.00",
            10,
            "0.5",
            "21000.00",
            "0",
        ]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Fakes (Protocol-compatible)
# ---------------------------------------------------------------------------


class FakeMarketDataFetcher:
    """In-memory fake implementation of ``IMarketDataFetcher``.

    Returns a pre-configured list of candles for every call to
    ``fetch_ohlcv()``.  Tracks all received requests for assertion.
    """

    def __init__(self, candles_to_return: list[OHLCVCandle] | None = None) -> None:
        """Initialise the fake with a fixed response payload.

        Args:
            candles_to_return: Candles returned by every ``fetch_ohlcv`` call.
                Defaults to an empty list.
        """
        self._response: list[OHLCVCandle] = candles_to_return or []
        self.calls: list[FetchRequest] = []

    def fetch_ohlcv(self, request: FetchRequest) -> list[OHLCVCandle]:
        """Record the request and return the pre-configured response.

        Args:
            request: The fetch specification.

        Returns:
            Pre-configured candle list.
        """
        self.calls.append(request)
        return list(self._response)


class FakeOHLCVRepository:
    """In-memory fake implementation of ``IOHLCVRepository``.

    Tracks ingested candles and simulates ``ingest()`` and
    ``get_date_range()`` without touching any database.
    """

    def __init__(self, existing_date_range: DateRange | None = None) -> None:
        """Initialise the fake with optional pre-existing data.

        Args:
            existing_date_range: Simulated existing date range returned by
                ``get_date_range()``.  If *None*, behaves as if no data exists.
        """
        self._existing_date_range: DateRange | None = existing_date_range
        self.ingested: list[OHLCVCandle] = []
        self.ingest_call_count: int = 0

    def ingest(self, candles: list[OHLCVCandle]) -> int:
        """Store candles and return the count written.

        Args:
            candles: Candles to persist.

        Returns:
            Number of candles stored.
        """
        self.ingest_call_count += 1
        self.ingested.extend(candles)
        return len(candles)

    def ingest_from_parquet(self, path: Path, asset: Asset, timeframe: Timeframe) -> int:
        """Not used in these tests — raises to surface accidental calls.

        Args:
            path: Parquet file path.
            asset: Trading pair.
            timeframe: Candlestick interval.

        Returns:
            Always 0 in this fake.
        """
        return 0

    def query(
        self,
        asset: Asset,
        timeframe: Timeframe,
        date_range: DateRange,
    ) -> list[OHLCVCandle]:
        """Return an empty list — not exercised by service tests.

        Args:
            asset: Trading pair.
            timeframe: Candlestick interval.
            date_range: Query bounds.

        Returns:
            Empty list.
        """
        return []

    def query_split(
        self,
        asset: Asset,
        timeframe: Timeframe,
        split: TemporalSplit,
        partition: str,
    ) -> list[OHLCVCandle]:
        """Return an empty list — not exercised by service tests.

        Args:
            asset: Trading pair.
            timeframe: Candlestick interval.
            split: Temporal split configuration.
            partition: Which partition to query.

        Returns:
            Empty list.
        """
        return []

    def query_cross_asset(
        self,
        assets: list[Asset],
        timeframe: Timeframe,
        date_range: DateRange,
    ) -> dict[str, list[OHLCVCandle]]:
        """Return empty dict — not exercised by service tests.

        Args:
            assets: Trading pairs.
            timeframe: Candlestick interval.
            date_range: Query bounds.

        Returns:
            Empty dict.
        """
        return {}

    def get_available_assets(self) -> list[str]:
        """Return an empty list — not exercised by service tests.

        Returns:
            Empty list.
        """
        return []

    def get_date_range(self, asset: Asset, timeframe: Timeframe) -> DateRange | None:
        """Return the simulated existing date range.

        Args:
            asset: Trading pair.
            timeframe: Candlestick interval.

        Returns:
            Pre-configured date range or None.
        """
        return self._existing_date_range

    def count(self) -> int:
        """Return total stored candle count.

        Returns:
            Number of ingested candles.
        """
        return len(self.ingested)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binance_settings() -> BinanceSettings:
    """Return minimal BinanceSettings constructed without environment."""
    return make_binance_settings()


@pytest.fixture
def fake_fetcher() -> FakeMarketDataFetcher:
    """Return a FakeMarketDataFetcher with empty response."""
    return FakeMarketDataFetcher()


@pytest.fixture
def fake_repository() -> FakeOHLCVRepository:
    """Return a FakeOHLCVRepository with no pre-existing data."""
    return FakeOHLCVRepository()


@pytest.fixture
def set_binance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject mandatory Binance environment variables for settings tests."""
    monkeypatch.setenv("BINANCE_API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("BINANCE_SECRET_KEY", FAKE_SECRET_KEY)
