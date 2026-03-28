"""Walk-forward evaluation framework — expanding and rolling window backtest runner."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Protocol

import polars as pl
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField

from src.app.backtest.application.execution import BacktestResult, ExecutionEngine
from src.app.backtest.application.metrics import BacktestMetrics, compute_metrics
from src.app.backtest.domain.entities import EquityCurve, Trade
from src.app.backtest.domain.protocols import IPositionSizer, IStrategy
from src.app.backtest.domain.value_objects import ExecutionConfig
from src.app.ohlcv.domain.value_objects import Asset


# ---------------------------------------------------------------------------
# WindowMode
# ---------------------------------------------------------------------------


class WindowMode(StrEnum):
    """Walk-forward window expansion mode.

    ``EXPANDING`` — the training window always starts at bar 0;
    ``ROLLING`` — the training window slides forward, keeping a fixed
    number of bars.
    """

    EXPANDING = "expanding"
    ROLLING = "rolling"


# ---------------------------------------------------------------------------
# WalkForwardConfig
# ---------------------------------------------------------------------------


class WalkForwardConfig(BaseModel, frozen=True):
    """Configuration for walk-forward evaluation.

    Attributes:
        mode: Window expansion mode (expanding or rolling).
        train_bars: Minimum number of bars in the training window.
        test_bars: Number of bars in each test (out-of-sample) window.
        step_bars: Number of bars to advance between windows.
            Defaults to ``test_bars`` (non-overlapping test windows).
    """

    mode: Annotated[
        WindowMode,
        PydanticField(default=WindowMode.EXPANDING, description="Window expansion mode"),
    ]
    train_bars: Annotated[
        int,
        PydanticField(gt=0, description="Minimum number of bars in the training window"),
    ]
    test_bars: Annotated[
        int,
        PydanticField(gt=0, description="Number of bars in each test window"),
    ]
    step_bars: Annotated[
        int | None,
        PydanticField(
            default=None,
            gt=0,
            description="Bars to advance per step; defaults to test_bars",
        ),
    ]

    @property
    def effective_step_bars(self) -> int:
        """Return the step size, defaulting to ``test_bars`` when unset.

        Returns:
            Effective step size in bars.
        """
        return self.step_bars if self.step_bars is not None else self.test_bars


# ---------------------------------------------------------------------------
# IStrategyFactory protocol
# ---------------------------------------------------------------------------


class IStrategyFactory(Protocol):
    """Factory that trains a strategy on historical bars and returns a fitted instance.

    Walk-forward evaluation calls this once per window: the factory
    receives the training slice and must return an ``IStrategy`` ready
    to receive ``on_bar()`` calls on unseen test data.
    """

    def __call__(self, train_bars: pl.DataFrame) -> IStrategy:
        """Create a fitted strategy from training data.

        Args:
            train_bars: Polars DataFrame of the training window,
                sorted by ``timestamp``.

        Returns:
            A fitted :class:`IStrategy` instance.
        """
        ...


# ---------------------------------------------------------------------------
# WindowResult
# ---------------------------------------------------------------------------


class WindowResult(BaseModel, frozen=True):
    """Result for a single walk-forward window.

    Attributes:
        window_index: Zero-based window ordinal.
        train_start: Timestamp of the first bar in the training window.
        train_end: Timestamp of the last bar in the training window.
        test_start: Timestamp of the first bar in the test window.
        test_end: Timestamp of the last bar in the test window.
        backtest_result: Raw execution result on the test window.
        metrics: Performance metrics computed on the test window.
    """

    window_index: Annotated[int, PydanticField(ge=0)]
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    backtest_result: BacktestResult
    metrics: BacktestMetrics


# ---------------------------------------------------------------------------
# WalkForwardResult
# ---------------------------------------------------------------------------


class WalkForwardResult(BaseModel, frozen=True):
    """Aggregate result of a complete walk-forward evaluation.

    Attributes:
        windows: Per-window results in chronological order.
        aggregate_metrics: Metrics computed on the chained out-of-sample
            equity curve spanning all windows.
        config: Walk-forward configuration used.
    """

    windows: list[WindowResult]
    aggregate_metrics: BacktestMetrics
    config: WalkForwardConfig


# ---------------------------------------------------------------------------
# _WindowSpec — internal window boundaries
# ---------------------------------------------------------------------------


class _WindowSpec(BaseModel, frozen=True):
    """Internal specification of a single walk-forward window's bar indices."""

    window_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int


# ---------------------------------------------------------------------------
# WalkForwardRunner
# ---------------------------------------------------------------------------


class WalkForwardRunner:
    """Walk-forward evaluation runner.

    Iterates over expanding or rolling windows, training a fresh
    strategy per window and running the backtest engine on the
    subsequent test slice.  Consecutive test windows are chained:
    the final equity of window *N* becomes the initial cash of
    window *N + 1*, producing a realistic out-of-sample equity
    curve.

    No lookahead is possible because the strategy factory only
    receives bars from the training window, and the execution
    engine processes test bars sequentially.
    """

    def __init__(
        self,
        config: WalkForwardConfig,
        execution_config: ExecutionConfig,
        strategy_factory: IStrategyFactory,
        sizer: IPositionSizer,
    ) -> None:
        """Initialise the walk-forward runner.

        Args:
            config: Walk-forward window configuration.
            execution_config: Execution-cost configuration passed to
                the backtest engine.
            strategy_factory: Callable that takes training bars and
                returns a fitted ``IStrategy``.
            sizer: Position sizer shared across all windows.
        """
        self._config: WalkForwardConfig = config
        self._execution_config: ExecutionConfig = execution_config
        self._strategy_factory: IStrategyFactory = strategy_factory
        self._sizer: IPositionSizer = sizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0914
        self,
        bars: pl.DataFrame,
        asset: Asset,
        initial_cash: float = 100_000.0,
        *,
        min_trade_count: int = 30,
        risk_free_rate: float = 0.0,
    ) -> WalkForwardResult:
        """Execute the full walk-forward evaluation.

        Args:
            bars: Polars DataFrame with OHLCV columns, sorted by
                ``timestamp``.
            asset: Asset being backtested.
            initial_cash: Starting portfolio cash in USD.
            min_trade_count: Minimum trades for trade-level metrics.
            risk_free_rate: Annualised risk-free rate for Sharpe
                computation.

        Returns:
            A :class:`WalkForwardResult` containing per-window and
            aggregate metrics.

        Raises:
            ValueError: If the bar count is insufficient for even one
                walk-forward window.
        """
        total_bars: int = len(bars)
        window_specs: list[_WindowSpec] = _generate_window_specs(
            total_bars=total_bars,
            config=self._config,
        )

        if len(window_specs) == 0:
            msg: str = (
                f"Insufficient bars ({total_bars}) for walk-forward: "
                f"need at least {self._config.train_bars + self._config.test_bars} bars "
                f"(train_bars={self._config.train_bars}, test_bars={self._config.test_bars})"
            )
            raise ValueError(msg)

        logger.info(
            "Walk-forward: {} windows, mode={}, bars={}",
            len(window_specs),
            self._config.mode.value,
            total_bars,
        )

        # Extract timestamp column once for logging
        col_ts: list[datetime] = bars.get_column("timestamp").to_list()

        window_results: list[WindowResult] = []
        running_cash: float = initial_cash

        for spec in window_specs:
            train_slice: pl.DataFrame = bars.slice(spec.train_start_idx, spec.train_end_idx - spec.train_start_idx)
            test_slice: pl.DataFrame = bars.slice(spec.test_start_idx, spec.test_end_idx - spec.test_start_idx)

            logger.debug(
                "Window {}: train [{} .. {}] ({} bars), test [{} .. {}] ({} bars), cash={:.2f}",
                spec.window_index,
                col_ts[spec.train_start_idx],
                col_ts[spec.train_end_idx - 1],
                spec.train_end_idx - spec.train_start_idx,
                col_ts[spec.test_start_idx],
                col_ts[spec.test_end_idx - 1],
                spec.test_end_idx - spec.test_start_idx,
                running_cash,
            )

            # Train strategy on training window (no test data visible)
            strategy: IStrategy = self._strategy_factory(train_slice)

            # Run backtest on test window
            engine: ExecutionEngine = ExecutionEngine(
                config=self._execution_config,
                strategy=strategy,
                sizer=self._sizer,
            )
            backtest_result: BacktestResult = engine.run(
                bars=test_slice,
                asset=asset,
                initial_cash=running_cash,
            )

            # Compute per-window metrics
            metrics: BacktestMetrics = compute_metrics(
                equity_curve=backtest_result.equity_curve,
                trades=backtest_result.trades,
                min_trade_count=min_trade_count,
                risk_free_rate=risk_free_rate,
            )

            window_result: WindowResult = WindowResult(
                window_index=spec.window_index,
                train_start=col_ts[spec.train_start_idx],
                train_end=col_ts[spec.train_end_idx - 1],
                test_start=col_ts[spec.test_start_idx],
                test_end=col_ts[spec.test_end_idx - 1],
                backtest_result=backtest_result,
                metrics=metrics,
            )
            window_results.append(window_result)

            # Chain equity: final equity of this window becomes initial
            # cash for the next window
            final_equity: float = _final_equity(backtest_result)
            running_cash = final_equity

            logger.debug(
                "Window {} done: total_return={}, n_trades={}, final_equity={:.2f}",
                spec.window_index,
                f"{metrics.total_return:.4f}" if metrics.total_return is not None else "N/A",
                metrics.n_trades,
                final_equity,
            )

        # Aggregate: build a single chained equity curve from all windows
        aggregate_equity: EquityCurve = _chain_equity_curves(window_results)
        all_trades: list[Trade] = _collect_trades(window_results)
        aggregate_metrics: BacktestMetrics = compute_metrics(
            equity_curve=aggregate_equity,
            trades=all_trades,
            min_trade_count=min_trade_count,
            risk_free_rate=risk_free_rate,
        )

        logger.info(
            "Walk-forward complete: {} windows, aggregate total_return={}, n_trades={}",
            len(window_results),
            f"{aggregate_metrics.total_return:.4f}" if aggregate_metrics.total_return is not None else "N/A",
            aggregate_metrics.n_trades,
        )

        return WalkForwardResult(
            windows=window_results,
            aggregate_metrics=aggregate_metrics,
            config=self._config,
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _generate_window_specs(
    *,
    total_bars: int,
    config: WalkForwardConfig,
) -> list[_WindowSpec]:
    """Generate walk-forward window specifications.

    For *expanding* mode the training window always starts at index 0.
    For *rolling* mode the training window slides forward to maintain
    exactly ``train_bars`` bars.

    Args:
        total_bars: Total number of bars in the dataset.
        config: Walk-forward configuration.

    Returns:
        List of ``_WindowSpec`` objects in chronological order.
        Empty if the dataset is too small for a single window.
    """
    step: int = config.effective_step_bars
    train_bars: int = config.train_bars
    test_bars: int = config.test_bars

    specs: list[_WindowSpec] = []
    window_idx: int = 0

    # First test window starts right after the initial training window
    test_start: int = train_bars

    while test_start + test_bars <= total_bars:
        test_end: int = test_start + test_bars

        if config.mode == WindowMode.EXPANDING:
            train_start: int = 0
        else:
            # Rolling: keep exactly train_bars before test_start
            train_start = test_start - train_bars

        train_end: int = test_start

        specs.append(
            _WindowSpec(
                window_index=window_idx,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
            ),
        )

        window_idx += 1
        test_start += step

    return specs


def _final_equity(backtest_result: BacktestResult) -> float:
    """Extract the final equity value from a backtest result.

    Falls back to the last equity curve value.  If the equity curve
    is empty (should not happen with valid inputs), returns ``0.0``.

    Args:
        backtest_result: Result from a single-window backtest run.

    Returns:
        Final portfolio equity.
    """
    values: list[float] = backtest_result.equity_curve.values
    if len(values) == 0:
        return 0.0
    return values[-1]


def _chain_equity_curves(window_results: list[WindowResult]) -> EquityCurve:
    """Chain per-window equity curves into a single aggregate curve.

    Each window's equity curve is already at the correct absolute
    level because we chain ``initial_cash`` across windows.  We
    simply concatenate the timestamps and values.

    When consecutive windows share a boundary timestamp, the first
    window's last point is dropped to avoid a duplicate (and to
    satisfy the monotonically-increasing timestamp invariant).

    Args:
        window_results: Per-window results in chronological order.

    Returns:
        A single :class:`EquityCurve` spanning all test windows.

    Raises:
        ValueError: If no window results are provided.
    """
    if len(window_results) == 0:
        msg: str = "Cannot chain equity curves: no window results"
        raise ValueError(msg)

    all_timestamps: list[datetime] = []
    all_values: list[float] = []

    for i, wr in enumerate(window_results):
        ec: EquityCurve = wr.backtest_result.equity_curve
        ts_list: list[datetime] = ec.timestamps
        val_list: list[float] = ec.values

        if len(ts_list) == 0:
            continue

        if i > 0 and len(all_timestamps) > 0:
            # Drop points from this window that would violate monotonicity
            start_idx: int = 0
            last_ts: datetime = all_timestamps[-1]
            while start_idx < len(ts_list) and ts_list[start_idx] <= last_ts:
                start_idx += 1
            ts_list = ts_list[start_idx:]
            val_list = val_list[start_idx:]

        all_timestamps.extend(ts_list)
        all_values.extend(val_list)

    return EquityCurve(timestamps=all_timestamps, values=all_values)


def _collect_trades(window_results: list[WindowResult]) -> list[Trade]:
    """Collect all trades from all windows into a single list.

    Args:
        window_results: Per-window results in chronological order.

    Returns:
        Flat list of all trades across windows.
    """
    trades: list[Trade] = []
    for wr in window_results:
        trades.extend(wr.backtest_result.trades)
    return trades
