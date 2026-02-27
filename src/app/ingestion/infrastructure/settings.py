"""Binance API settings loaded from environment / ``.env``."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BinanceSettings(BaseSettings):
    """Configuration for the Binance exchange API client.

    Every field is read from an environment variable with the ``BINANCE_``
    prefix (e.g. ``BINANCE_API_KEY``, ``BINANCE_API_SECRET``).
    """

    model_config = SettingsConfigDict(
        env_prefix="BINANCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(description="Binance API key for authentication.")
    api_secret: str = Field(description="Binance API secret for authentication.")
    max_retries: int = Field(default=5, ge=1, description="Maximum number of retry attempts per API call.")
    retry_min_wait: int = Field(default=1, ge=1, description="Minimum wait time in seconds between retries.")
    retry_max_wait: int = Field(default=10, ge=1, description="Maximum wait time in seconds between retries.")
    batch_size: int = Field(default=1000, ge=1, le=1000, description="Number of klines per API request.")
