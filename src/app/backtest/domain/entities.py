"""Backtest domain entities — signals, positions, trades, equity curves."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import model_validator

from src.app.backtest.domain.value_objects import Side, TradeResult
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


class Signal(BaseModel, frozen=True):
    """A directional trading signal emitted by a strategy.

    Strength ranges from 0 (no conviction) to 1 (maximum conviction)
    and is consumed by the position sizer to determine trade size.

    Invariants:
        * ``strength`` must be in the closed interval ``[0, 1]``.
    """

    asset: Asset
    side: Side
    strength: Annotated[
        float,
        PydanticField(ge=0.0, le=1.0, description="Signal conviction from 0.0 to 1.0"),
    ]
    timestamp: datetime


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


class Position(BaseModel, frozen=False):
    """A mutable open position tracked by the backtest engine.

    Unlike :class:`Trade`, a position is *live* — its unrealised P&L
    is updated on every bar.  Optional stop-loss and take-profit
    levels are checked by the execution layer.

    Invariants:
        * ``size`` must be positive.
        * ``entry_price`` must be positive.
    """

    asset: Asset
    side: Side
    size: Annotated[float, PydanticField(gt=0, description="Position size in base units")]
    entry_price: Annotated[float, PydanticField(gt=0, description="Average entry price")]
    entry_time: datetime
    unrealized_pnl: Annotated[float, PydanticField(default=0.0)]
    stop_loss: Annotated[float | None, PydanticField(default=None)]
    take_profit: Annotated[float | None, PydanticField(default=None)]


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------


class Trade(BaseModel, frozen=True):
    """Immutable record of a completed trade lifecycle (entry to exit).

    Produced when a :class:`Position` is closed.  Contains full P&L
    accounting including commissions.

    Invariants:
        * ``exit_time`` must be strictly after ``entry_time``.
    """

    asset: Asset
    side: Side
    size: Annotated[float, PydanticField(gt=0, description="Trade size in base units")]
    entry_price: Annotated[float, PydanticField(gt=0, description="Entry fill price")]
    exit_price: Annotated[float, PydanticField(gt=0, description="Exit fill price")]
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    net_pnl: float
    commission_paid: Annotated[
        float,
        PydanticField(ge=0, description="Total commission paid for the round trip"),
    ]

    @model_validator(mode="after")
    def _exit_after_entry(self) -> Self:
        """Ensure exit time is strictly after entry time.

        Returns:
            Validated instance.

        Raises:
            ValueError: If ``exit_time`` is not after ``entry_time``.
        """
        if self.exit_time <= self.entry_time:
            msg: str = f"exit_time ({self.exit_time}) must be after entry_time ({self.entry_time})"
            raise ValueError(msg)
        return self

    def to_result(self) -> TradeResult:
        """Convert to a :class:`TradeResult` value object.

        Strips the :class:`Asset` entity reference, producing a
        lightweight record suitable for aggregation and serialisation.

        Returns:
            An equivalent :class:`TradeResult`.
        """
        return TradeResult(
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            side=self.side,
            size=self.size,
            entry_time=self.entry_time,
            exit_time=self.exit_time,
            gross_pnl=self.gross_pnl,
            net_pnl=self.net_pnl,
            commission_paid=self.commission_paid,
        )


# ---------------------------------------------------------------------------
# EquityCurve
# ---------------------------------------------------------------------------


class EquityCurve(BaseModel, frozen=True):
    """Time-indexed equity series produced by a backtest run.

    Stores aligned lists of timestamps and equity values.  Downstream
    consumers use this to compute Sharpe ratio, drawdown, and other
    performance statistics.

    Invariants:
        * ``timestamps`` and ``values`` must have equal length.
        * ``timestamps`` must be monotonically increasing.
    """

    timestamps: list[datetime]
    values: list[float]

    @model_validator(mode="after")
    def _validate_alignment_and_ordering(self) -> Self:
        """Ensure timestamps and values are aligned and chronologically ordered.

        Returns:
            Validated instance.

        Raises:
            ValueError: If lengths differ or timestamps are not monotonically
                increasing.
        """
        if len(self.timestamps) != len(self.values):
            msg: str = f"timestamps length ({len(self.timestamps)}) must equal values length ({len(self.values)})"
            raise ValueError(msg)
        for i in range(1, len(self.timestamps)):
            if self.timestamps[i] <= self.timestamps[i - 1]:
                msg = (
                    f"timestamps must be monotonically increasing, but "
                    f"index {i} ({self.timestamps[i]}) <= index {i - 1} "
                    f"({self.timestamps[i - 1]})"
                )
                raise ValueError(msg)
        return self
