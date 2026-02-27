"""Ingestion domain protocols (interfaces)."""

from __future__ import annotations

from typing import Protocol

from src.app.ingestion.domain.value_objects import FetchRequest
from src.app.ohlcv.domain.entities import OHLCVCandle


class IMarketDataFetcher(Protocol):
    """Structural interface for fetching OHLCV market data from an exchange.

    Implementations are responsible for pagination, retries, and converting
    raw exchange responses into domain ``OHLCVCandle`` entities.
    """

    def fetch_ohlcv(self, request: FetchRequest) -> list[OHLCVCandle]:
        """Fetch OHLCV candles for the given request.

        Args:
            request: Specifies the asset, timeframe, and date range to fetch.

        Returns:
            A list of ``OHLCVCandle`` entities ordered by timestamp.

        Raises:
            FetchError: If the exchange API calls fail after retries.
            RateLimitError: If the exchange rate limit is exhausted.
        """
        ...
