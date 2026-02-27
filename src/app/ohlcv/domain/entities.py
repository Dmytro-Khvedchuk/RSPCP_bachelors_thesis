"""OHLCV domain entity."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, model_validator

from src.app.ohlcv.domain.value_objects import Asset, Timeframe


class OHLCVCandle(BaseModel, frozen=True):
    """A single OHLCV candlestick — the core domain entity.

    Prices are stored as :class:`~decimal.Decimal` to preserve the 8-decimal
    precision used by Binance.  Volume remains a plain *float* because
    sub-satoshi precision is unnecessary there.
    """

    asset: Asset
    timeframe: Timeframe
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: float

    @model_validator(mode="after")
    def _validate_invariants(self) -> Self:
        if self.high < self.low:
            msg = f"high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        if self.volume < 0:
            msg = f"volume must be >= 0, got {self.volume}"
            raise ValueError(msg)
        return self
