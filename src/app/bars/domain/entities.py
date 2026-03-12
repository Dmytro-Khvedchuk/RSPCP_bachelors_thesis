"""Bar domain entity — the aggregated bar."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import BaseModel, model_validator

from src.app.bars.domain.value_objects import BarType
from src.app.ohlcv.domain.value_objects import Asset


class AggregatedBar(BaseModel, frozen=True):
    """A single aggregated bar produced by any bar-construction algorithm.

    Contains standard OHLCV fields plus micro-structure metadata
    (tick count, buy/sell volume split, VWAP) that downstream models
    and feature engineering stages consume.

    Prices are stored as :class:`~decimal.Decimal` to preserve
    exchange-level precision.  Volumes remain plain *float* because
    sub-satoshi precision is unnecessary there.

    Invariants:
        * ``high >= low``.
        * ``volume >= 0``.
        * ``tick_count >= 1``.
        * ``buy_volume >= 0`` and ``sell_volume >= 0``.
        * ``buy_volume + sell_volume <= volume`` (within floating-point tolerance).
        * ``start_ts < end_ts``.
    """

    asset: Asset
    bar_type: BarType
    start_ts: datetime
    end_ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: float
    tick_count: int
    buy_volume: float
    sell_volume: float
    vwap: Decimal

    @model_validator(mode="after")
    def _validate_invariants(self) -> Self:
        """Enforce bar invariants.

        Returns:
            Validated instance.

        Raises:
            ValueError: If any invariant is violated.
        """
        if self.high < self.low:
            msg: str = f"high ({self.high}) must be >= low ({self.low})"
            raise ValueError(msg)
        if self.volume < 0:
            msg = f"volume must be >= 0, got {self.volume}"
            raise ValueError(msg)
        if self.tick_count < 1:
            msg = f"tick_count must be >= 1, got {self.tick_count}"
            raise ValueError(msg)
        if self.buy_volume < 0:
            msg = f"buy_volume must be >= 0, got {self.buy_volume}"
            raise ValueError(msg)
        if self.sell_volume < 0:
            msg = f"sell_volume must be >= 0, got {self.sell_volume}"
            raise ValueError(msg)
        total_directional: float = self.buy_volume + self.sell_volume
        epsilon: float = 1e-9
        if total_directional > self.volume + epsilon:
            msg = (
                f"buy_volume ({self.buy_volume}) + sell_volume ({self.sell_volume}) = "
                f"{total_directional} must not exceed volume ({self.volume})"
            )
            raise ValueError(msg)
        if self.start_ts >= self.end_ts:
            msg = f"start_ts ({self.start_ts}) must be before end_ts ({self.end_ts})"
            raise ValueError(msg)
        return self
