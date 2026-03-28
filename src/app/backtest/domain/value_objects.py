"""Backtest domain value objects — execution config, trade results, portfolio snapshots."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import model_validator


# ---------------------------------------------------------------------------
# Side
# ---------------------------------------------------------------------------


class Side(StrEnum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


# ---------------------------------------------------------------------------
# ExecutionConfig
# ---------------------------------------------------------------------------


class ExecutionConfig(BaseModel, frozen=True):
    """Execution-cost configuration for backtest simulations.

    Encapsulates commission assumptions and cost-sweep parameters used by
    the backtest engine to evaluate strategy robustness across different
    fee regimes.

    Attributes:
        commission_bps: Default commission in basis points (1 bp = 0.01%).
        asset_cost_multiplier: Per-asset cost multipliers that override the
            default commission.  Keys are asset symbols (e.g. ``"BTCUSDT"``).
        min_trade_count: Minimum number of trades required for a backtest
            result to be considered statistically meaningful.
        cost_sweep_bps: Commission levels (in bps) to sweep over when
            evaluating cost sensitivity.
    """

    commission_bps: Annotated[
        float,
        PydanticField(default=10, ge=0, description="Default commission in basis points"),
    ]

    asset_cost_multiplier: Annotated[
        dict[str, float],
        PydanticField(
            default_factory=dict,
            description="Per-asset cost multipliers overriding the default commission",
        ),
    ]

    min_trade_count: Annotated[
        int,
        PydanticField(
            default=30,
            gt=0,
            description="Minimum trades for statistical significance",
        ),
    ]

    cost_sweep_bps: Annotated[
        list[float],
        PydanticField(
            default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 30.0],
            description="Commission levels (bps) for cost-sensitivity sweep",
        ),
    ]


# ---------------------------------------------------------------------------
# TradeResult
# ---------------------------------------------------------------------------


class TradeResult(BaseModel, frozen=True):
    """Immutable record of a completed trade.

    Captures entry/exit prices, timing, direction, size, and the
    gross/net P&L after commissions.  Used as the atomic unit for
    strategy performance aggregation.

    Invariants:
        * ``exit_time`` must be strictly after ``entry_time``.
        * ``size`` must be positive.
        * ``commission_paid`` must be non-negative.
    """

    entry_price: Annotated[float, PydanticField(gt=0, description="Entry fill price")]
    exit_price: Annotated[float, PydanticField(gt=0, description="Exit fill price")]
    side: Side
    size: Annotated[float, PydanticField(gt=0, description="Position size in base asset units")]
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


# ---------------------------------------------------------------------------
# PortfolioSnapshot
# ---------------------------------------------------------------------------


class PortfolioSnapshot(BaseModel, frozen=True):
    """Point-in-time snapshot of portfolio state.

    Captures equity, cash, open positions, unrealised P&L, and
    drawdown at a single timestamp.  The backtest engine emits one
    snapshot per bar to build the equity curve.

    Invariants:
        * ``equity`` must be non-negative.
        * ``drawdown`` must be non-positive (zero or negative).
    """

    timestamp: datetime
    equity: Annotated[float, PydanticField(ge=0, description="Total portfolio equity")]
    cash: float
    positions: Annotated[
        dict[str, float],
        PydanticField(
            default_factory=dict,
            description="Open positions: asset symbol to signed size",
        ),
    ]
    unrealized_pnl: Annotated[float, PydanticField(default=0.0)]
    drawdown: Annotated[float, PydanticField(default=0.0, le=0)]
