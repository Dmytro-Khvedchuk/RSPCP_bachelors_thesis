"""Ingestion domain exceptions."""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion-related errors."""


class FetchError(IngestionError):
    """Raised when Binance API calls fail after all retries are exhausted."""


class RateLimitError(IngestionError):
    """Raised when the Binance API rate limit is exhausted."""
