"""Backtest execution engine — sequential bar-by-bar simulation with next-bar fills."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import polars as pl
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.backtest.domain.entities import EquityCurve, Position, Signal, Trade
from src.app.backtest.domain.protocols import IPositionSizer, IStrategy
from src.app.backtest.domain.value_objects import ExecutionConfig, PortfolioSnapshot, Side
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"},
)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


class BacktestResult(BaseModel, frozen=True):
    """Immutable result of a complete backtest run.

    Bundles the equity curve, completed trades, per-bar portfolio
    snapshots, and the execution configuration used.

    Attributes:
        equity_curve: Time-indexed equity series.
        trades: List of all completed trades.
        snapshots: Per-bar portfolio snapshots.
        config: Execution configuration used for this run.
    """

    equity_curve: EquityCurve
    trades: list[Trade]
    snapshots: list[PortfolioSnapshot]
    config: ExecutionConfig


# ---------------------------------------------------------------------------
# _EngineState — mutable bookkeeping for a single backtest run
# ---------------------------------------------------------------------------


class _EngineState(BaseModel, frozen=False):
    """Mutable state bundle for a single backtest run."""

    cash: float
    open_positions: dict[str, Position] = {}
    trades: list[Trade] = []
    snapshots: list[PortfolioSnapshot] = []
    equity_timestamps: list[datetime] = []
    equity_values: list[float] = []
    peak_equity: float
    pending_signals: list[Signal] = []
    prev_timestamp: Annotated[datetime | None, PydanticField(default=None)]


# ---------------------------------------------------------------------------
# _BarRow — pre-extracted OHLC for a single bar
# ---------------------------------------------------------------------------


class _BarRow(BaseModel, frozen=True):
    """Pre-extracted OHLC fields for a single bar."""

    index: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# ExecutionEngine
# ---------------------------------------------------------------------------


class ExecutionEngine:
    """Sequential bar-by-bar backtest engine with next-bar fill semantics.

    Signals generated at bar *t* are filled at bar *t + 1*'s open price,
    ensuring that no future information leaks into execution decisions.
    Commission costs, stop-loss / take-profit levels, and staleness
    detection are handled automatically.

    Position sizers return **notional USD amounts** which the engine
    converts to base-asset units by dividing by the fill price.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        strategy: IStrategy,
        sizer: IPositionSizer,
    ) -> None:
        """Initialise the execution engine.

        Args:
            config: Execution-cost and sweep configuration.
            strategy: Trading strategy producing signals per bar.
            sizer: Position sizer translating signals to notional sizes.
        """
        self._config: ExecutionConfig = config
        self._strategy: IStrategy = strategy
        self._sizer: IPositionSizer = sizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0914
        self,
        bars: pl.DataFrame,
        asset: Asset,
        initial_cash: float = 100_000.0,
    ) -> BacktestResult:
        """Execute the backtest over *bars*.

        Args:
            bars: Polars DataFrame with columns ``timestamp``, ``open``,
                ``high``, ``low``, ``close``, ``volume``, sorted by
                ``timestamp``.
            asset: Asset being backtested.
            initial_cash: Starting portfolio cash in USD.

        Returns:
            A :class:`BacktestResult` containing the equity curve,
            trades, snapshots, and configuration.
        """
        _validate_bars(bars)

        staleness_threshold: float = _compute_staleness_threshold(bars)
        state: _EngineState = _EngineState(cash=initial_cash, peak_equity=initial_cash)  # ty: ignore[missing-argument]

        col_ts: list[datetime] = bars.get_column("timestamp").to_list()
        col_open: list[float] = bars.get_column("open").to_list()
        col_high: list[float] = bars.get_column("high").to_list()
        col_low: list[float] = bars.get_column("low").to_list()
        col_close: list[float] = bars.get_column("close").to_list()
        bar_count: int = len(bars)

        for i in range(bar_count):
            bar: _BarRow = _BarRow(
                index=i,
                timestamp=col_ts[i],
                open=col_open[i],
                high=col_high[i],
                low=col_low[i],
                close=col_close[i],
            )
            self._process_bar(bar, bars, asset, state, staleness_threshold)

        self._liquidate_remaining(state, col_close, col_ts, bar_count, asset)

        equity_curve: EquityCurve = EquityCurve(
            timestamps=state.equity_timestamps,
            values=state.equity_values,
        )
        return BacktestResult(
            equity_curve=equity_curve,
            trades=state.trades,
            snapshots=state.snapshots,
            config=self._config,
        )

    # ------------------------------------------------------------------
    # Per-bar processing
    # ------------------------------------------------------------------

    def _process_bar(
        self,
        bar: _BarRow,
        bars: pl.DataFrame,
        asset: Asset,
        state: _EngineState,
        staleness_threshold: float,
    ) -> None:
        """Process a single bar: fill signals, update P&L, check SL/TP, snapshot.

        Args:
            bar: Current bar OHLC data.
            bars: Full bars DataFrame (for slicing features).
            asset: Asset being backtested.
            state: Mutable engine state.
            staleness_threshold: Maximum allowed gap in seconds.
        """
        # Staleness check
        if state.prev_timestamp is not None and staleness_threshold > 0.0:
            gap: float = (bar.timestamp - state.prev_timestamp).total_seconds()
            if gap > staleness_threshold:
                logger.debug(
                    "Stale bar skipped at {}: gap {:.0f}s > threshold {:.0f}s",
                    bar.timestamp,
                    gap,
                    staleness_threshold,
                )
                state.prev_timestamp = bar.timestamp
                return

        # Fill pending signals at current bar's OPEN
        snapshot_for_sizing: PortfolioSnapshot = _build_snapshot(
            timestamp=bar.timestamp,
            cash=state.cash,
            open_positions=state.open_positions,
            peak_equity=state.peak_equity,
        )
        self._fill_pending_signals(bar.open, bar.timestamp, asset, state, snapshot_for_sizing)
        state.pending_signals = []

        # Update unrealised P&L at close
        for position in state.open_positions.values():
            position.unrealized_pnl = _unrealised_pnl(position, bar.close)

        # Check stop-loss / take-profit
        self._check_exits(bar, asset, state)

        # Portfolio snapshot
        snapshot: PortfolioSnapshot = _build_snapshot(
            timestamp=bar.timestamp,
            cash=state.cash,
            open_positions=state.open_positions,
            peak_equity=state.peak_equity,
        )
        state.peak_equity = max(state.peak_equity, snapshot.equity)
        state.snapshots.append(snapshot)
        state.equity_timestamps.append(bar.timestamp)
        state.equity_values.append(snapshot.equity)

        # Generate new signals (for next bar — no lookahead)
        features_up_to_now: pl.DataFrame = bars.slice(0, bar.index + 1)
        state.pending_signals = self._strategy.on_bar(bar.timestamp, features_up_to_now, snapshot)

        state.prev_timestamp = bar.timestamp

    # ------------------------------------------------------------------
    # Signal filling
    # ------------------------------------------------------------------

    def _fill_pending_signals(
        self,
        open_price: float,
        timestamp: datetime,
        asset: Asset,
        state: _EngineState,
        snapshot: PortfolioSnapshot,
    ) -> None:
        """Fill queued signals at the current bar's open price.

        If a signal's direction opposes an existing position, the
        existing position is closed first.

        Args:
            open_price: Current bar's open price (fill price).
            timestamp: Current bar timestamp.
            asset: Asset being traded.
            state: Mutable engine state (cash, positions, trades).
            snapshot: Portfolio snapshot used for sizing.
        """
        for signal in state.pending_signals:
            notional: float = self._sizer.size(signal, snapshot, 0.0)
            if notional <= 0.0:
                continue

            position_size: float = notional / open_price
            if position_size <= 0.0:
                continue

            effective_bps: float = self._effective_bps(asset)
            entry_commission: float = open_price * position_size * effective_bps / 10_000

            # Close opposite position first if present
            sym: str = asset.symbol
            existing: Position | None = state.open_positions.get(sym)
            if existing is not None and existing.side != signal.side:
                trade, cash_returned = self._close_position(
                    position=existing,
                    exit_price=open_price,
                    exit_time=timestamp,
                    asset=asset,
                )
                state.trades.append(trade)
                state.cash += cash_returned
                del state.open_positions[sym]

            # Check sufficient cash
            total_cost: float = open_price * position_size + entry_commission
            if state.cash < total_cost:
                logger.debug(
                    "Insufficient cash for {} {} entry: need {:.2f}, have {:.2f}",
                    signal.side.value,
                    asset.symbol,
                    total_cost,
                    state.cash,
                )
                continue

            # Deduct cost and open position
            state.cash -= total_cost
            new_position: Position = Position(  # ty: ignore[missing-argument]
                asset=asset,
                side=signal.side,
                size=position_size,
                entry_price=open_price,
                entry_time=timestamp,
            )
            state.open_positions[asset.symbol] = new_position

    # ------------------------------------------------------------------
    # SL / TP checking
    # ------------------------------------------------------------------

    def _check_exits(self, bar: _BarRow, asset: Asset, state: _EngineState) -> None:
        """Check and execute stop-loss / take-profit exits.

        Args:
            bar: Current bar OHLC data.
            asset: Asset being traded.
            state: Mutable engine state.
        """
        symbols_to_close: list[str] = []
        for sym, position in state.open_positions.items():
            exit_price: float | None = _check_sl_tp(position, bar.high, bar.low)
            if exit_price is not None:
                trade, cash_returned = self._close_position(
                    position=position,
                    exit_price=exit_price,
                    exit_time=bar.timestamp,
                    asset=asset,
                )
                state.trades.append(trade)
                state.cash += cash_returned
                symbols_to_close.append(sym)

        for sym in symbols_to_close:
            del state.open_positions[sym]

    # ------------------------------------------------------------------
    # End-of-backtest liquidation
    # ------------------------------------------------------------------

    def _liquidate_remaining(
        self,
        state: _EngineState,
        col_close: list[float],
        col_ts: list[datetime],
        bar_count: int,
        asset: Asset,
    ) -> None:
        """Close all remaining positions at the last bar's close.

        Args:
            state: Mutable engine state.
            col_close: Pre-extracted close prices.
            col_ts: Pre-extracted timestamps.
            bar_count: Number of bars.
            asset: Asset being traded.
        """
        if not state.open_positions or bar_count == 0:
            return

        last_close: float = col_close[bar_count - 1]
        last_ts: datetime = col_ts[bar_count - 1]

        for position in state.open_positions.values():
            trade, cash_returned = self._close_position(
                position=position,
                exit_price=last_close,
                exit_time=last_ts,
                asset=asset,
            )
            state.trades.append(trade)
            state.cash += cash_returned

        state.open_positions.clear()

        if state.snapshots:
            final: PortfolioSnapshot = PortfolioSnapshot(
                timestamp=last_ts,
                equity=state.cash,
                cash=state.cash,
                positions={},
                unrealized_pnl=0.0,
                drawdown=_drawdown(state.cash, state.peak_equity),
            )
            state.snapshots[-1] = final
            state.equity_values[-1] = state.cash

    # ------------------------------------------------------------------
    # Position closing
    # ------------------------------------------------------------------

    def _close_position(
        self,
        *,
        position: Position,
        exit_price: float,
        exit_time: datetime,
        asset: Asset,
    ) -> tuple[Trade, float]:
        """Close a position and compute P&L.

        At entry the engine deducted ``entry_price * size + entry_commission``
        from cash.  At exit we return the collateral plus gross P&L minus
        exit commission.

        Args:
            position: The position to close.
            exit_price: Price at which the position is closed.
            exit_time: Timestamp of the exit.
            asset: Asset of the position.

        Returns:
            A ``(trade, cash_returned)`` tuple where *cash_returned* is
            the amount to add back to the cash balance.
        """
        gross: float = _gross_pnl(position, exit_price)
        effective_bps: float = self._effective_bps(asset)
        entry_comm: float = position.entry_price * position.size * effective_bps / 10_000
        exit_comm: float = exit_price * position.size * effective_bps / 10_000
        total_comm: float = entry_comm + exit_comm
        net: float = gross - total_comm

        trade: Trade = Trade(
            asset=position.asset,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            gross_pnl=gross,
            net_pnl=net,
            commission_paid=total_comm,
        )

        cash_returned: float = position.entry_price * position.size + gross - exit_comm
        return trade, cash_returned

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_bps(self, asset: Asset) -> float:
        """Return the effective commission in bps for *asset*.

        Args:
            asset: Asset to look up.

        Returns:
            Commission in basis points, adjusted by the per-asset
            multiplier.
        """
        multiplier: float = self._config.asset_cost_multiplier.get(asset.symbol, 1.0)
        return self._config.commission_bps * multiplier


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _validate_bars(bars: pl.DataFrame) -> None:
    """Validate the input bars DataFrame.

    Args:
        bars: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or the DataFrame
            is empty.
    """
    if len(bars) == 0:
        msg: str = "bars DataFrame must not be empty"
        raise ValueError(msg)
    missing: frozenset[str] = _REQUIRED_COLUMNS - set(bars.columns)
    if missing:
        msg = f"bars DataFrame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def _compute_staleness_threshold(bars: pl.DataFrame) -> float:
    """Compute the staleness threshold as 2x the median bar duration.

    Args:
        bars: Bars DataFrame with a ``timestamp`` column.

    Returns:
        Staleness threshold in seconds.  Returns ``0.0`` when fewer
        than two bars are present.
    """
    timestamps: list[datetime] = bars.get_column("timestamp").to_list()
    bar_count: int = len(timestamps)
    if bar_count < 2:  # noqa: PLR2004
        return 0.0
    durations: list[float] = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, bar_count)]
    sorted_durations: list[float] = sorted(durations)
    median_duration: float = sorted_durations[len(sorted_durations) // 2]
    staleness_multiplier: float = 2.0
    return median_duration * staleness_multiplier


def _unrealised_pnl(position: Position, mark_price: float) -> float:
    """Compute unrealised P&L for *position* at *mark_price*.

    Args:
        position: Open position.
        mark_price: Current market price.

    Returns:
        Unrealised profit or loss.
    """
    if position.side == Side.LONG:
        return (mark_price - position.entry_price) * position.size
    return (position.entry_price - mark_price) * position.size


def _gross_pnl(position: Position, exit_price: float) -> float:
    """Compute gross P&L for closing *position* at *exit_price*.

    Args:
        position: Position being closed.
        exit_price: Exit fill price.

    Returns:
        Gross profit or loss (before commissions).
    """
    if position.side == Side.LONG:
        return (exit_price - position.entry_price) * position.size
    return (position.entry_price - exit_price) * position.size


def _check_sl_tp(
    position: Position,
    high_price: float,
    low_price: float,
) -> float | None:
    """Check stop-loss and take-profit levels against bar extremes.

    Stop-loss is checked first to model worst-case execution.

    Args:
        position: Open position with optional SL/TP levels.
        high_price: Bar high.
        low_price: Bar low.

    Returns:
        Exit price if a level was triggered, otherwise *None*.
    """
    sl: float | None = position.stop_loss
    tp: float | None = position.take_profit

    if position.side == Side.LONG:
        if sl is not None and low_price <= sl:
            return sl
        if tp is not None and high_price >= tp:
            return tp
    else:
        if sl is not None and high_price >= sl:
            return sl
        if tp is not None and low_price <= tp:
            return tp

    return None


def _build_snapshot(
    *,
    timestamp: datetime,
    cash: float,
    open_positions: dict[str, Position],
    peak_equity: float,
) -> PortfolioSnapshot:
    """Build a :class:`PortfolioSnapshot` from current engine state.

    Equity is computed as ``cash + sum(entry_price * size) +
    total_unrealised`` for all open positions — equivalent to
    ``cash + sum(market_value)`` where market value depends on side.

    Args:
        timestamp: Snapshot timestamp.
        cash: Current cash balance.
        open_positions: Open positions keyed by symbol.
        peak_equity: Highest equity observed so far.

    Returns:
        A frozen :class:`PortfolioSnapshot`.
    """
    total_unrealised: float = sum(p.unrealized_pnl for p in open_positions.values())
    collateral: float = sum(p.entry_price * p.size for p in open_positions.values())
    equity: float = cash + collateral + total_unrealised
    positions_dict: dict[str, float] = {
        sym: (p.size if p.side == Side.LONG else -p.size) for sym, p in open_positions.items()
    }

    return PortfolioSnapshot(
        timestamp=timestamp,
        equity=max(0.0, equity),
        cash=cash,
        positions=positions_dict,
        unrealized_pnl=total_unrealised,
        drawdown=_drawdown(equity, peak_equity),
    )


def _drawdown(equity: float, peak_equity: float) -> float:
    """Compute drawdown as a non-positive fraction.

    Args:
        equity: Current equity.
        peak_equity: Peak equity to date.

    Returns:
        Drawdown value (``<= 0``).
    """
    if peak_equity <= 0.0:
        return 0.0
    dd: float = (equity - peak_equity) / peak_equity
    return min(dd, 0.0)
