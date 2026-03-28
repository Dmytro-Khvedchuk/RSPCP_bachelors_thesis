"""Backtest domain protocols — structural interfaces for strategies and position sizing."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl

from src.app.backtest.domain.entities import Signal
from src.app.backtest.domain.value_objects import PortfolioSnapshot


class IStrategy(Protocol):
    """Structural interface for trading strategies.

    Implementations receive a bar timestamp, a feature DataFrame, and
    the current portfolio state, returning zero or more :class:`Signal`
    objects that the backtest engine routes to the position sizer.
    """

    def on_bar(
        self,
        timestamp: datetime,
        features: pl.DataFrame,
        portfolio: PortfolioSnapshot,
    ) -> list[Signal]:
        """Generate trading signals for the current bar.

        Args:
            timestamp: Bar timestamp (UTC).
            features: Polars DataFrame containing feature columns for the
                current bar.  Implementations must not look ahead — only
                rows up to and including *timestamp* are valid.
            portfolio: Current portfolio state snapshot.

        Returns:
            List of signals (may be empty if no action is warranted).
        """
        ...


class IPositionSizer(Protocol):
    """Structural interface for position sizing algorithms.

    Implementations translate a :class:`Signal` into a concrete position
    size, accounting for portfolio state and current volatility.
    """

    def size(
        self,
        signal: Signal,
        portfolio: PortfolioSnapshot,
        volatility: float,
    ) -> float:
        """Compute position size for a given signal.

        Args:
            signal: Trading signal with direction and conviction strength.
            portfolio: Current portfolio state (equity, cash, positions).
            volatility: Current asset volatility estimate (e.g. realised vol
                over a trailing window).

        Returns:
            Position size in base asset units.  Zero means no trade.
        """
        ...
