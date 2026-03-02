"""OHLCV domain value objects."""

from __future__ import annotations

from datetime import datetime, UTC
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import field_validator, model_validator


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------


class Asset(BaseModel, frozen=True):
    """A validated trading-pair symbol (e.g. ``BTCUSDT``)."""

    symbol: Annotated[str, PydanticField(pattern=r"^[A-Z0-9]{2,20}$")]

    def __str__(self) -> str:
        """Return the raw symbol string."""
        return self.symbol


# ---------------------------------------------------------------------------
# Timeframe
# ---------------------------------------------------------------------------


class Timeframe(StrEnum):
    """Supported candlestick timeframes."""

    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


# ---------------------------------------------------------------------------
# DateRange
# ---------------------------------------------------------------------------


class DateRange(BaseModel, frozen=True):
    """An inclusive UTC date-range.

    Both bounds must be timezone-aware (UTC) and ``start`` must precede
    ``end``.
    """

    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _must_be_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            msg = "DateRange bounds must be timezone-aware (UTC)"
            raise ValueError(msg)
        if v.tzinfo != UTC:
            msg = "DateRange bounds must be in UTC"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _start_before_end(self) -> Self:
        if self.start >= self.end:
            msg = f"start ({self.start}) must be before end ({self.end})"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# TemporalSplit
# ---------------------------------------------------------------------------


class TemporalSplit(BaseModel, frozen=True):
    """Train / validation / test temporal split.

    Construction enforces strict chronological ordering so that data leakage
    between partitions is impossible by design.
    """

    train: DateRange
    validation: DateRange
    test: DateRange

    @model_validator(mode="after")
    def _strict_ordering(self) -> Self:
        if self.train.end > self.validation.start:
            msg = f"Train end ({self.train.end}) must not exceed validation start ({self.validation.start})"
            raise ValueError(msg)
        if self.validation.end > self.test.start:
            msg = f"Validation end ({self.validation.end}) must not exceed test start ({self.test.start})"
            raise ValueError(msg)
        return self

    def get_range(self, partition: str) -> DateRange:
        """Return the :class:`DateRange` for *partition* (train/validation/test).

        Args:
            partition: One of ``"train"``, ``"validation"``, or ``"test"``.

        Returns:
            The date range corresponding to the requested partition.

        Raises:
            ValueError: If *partition* is not one of the three valid names.
        """
        mapping = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        if partition not in mapping:
            msg = f"Unknown partition '{partition}' — choose from train, validation, test"
            raise ValueError(msg)
        return mapping[partition]
