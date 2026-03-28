"""Cost-sensitivity sweep — run backtest at multiple fee levels."""

from __future__ import annotations

import polars as pl

from src.app.backtest.application.execution import BacktestResult, ExecutionEngine
from src.app.backtest.domain.protocols import IPositionSizer, IStrategy
from src.app.backtest.domain.value_objects import ExecutionConfig
from src.app.ohlcv.domain.value_objects import Asset


def run_with_cost_sweep(  # noqa: PLR0913
    strategy: IStrategy,
    sizer: IPositionSizer,
    bars: pl.DataFrame,
    asset: Asset,
    *,
    fees: list[float] | None = None,
    initial_cash: float = 100_000.0,
    base_config: ExecutionConfig | None = None,
) -> dict[float, BacktestResult]:
    """Run backtest at multiple fee levels to evaluate cost sensitivity.

    Iterates over a list of commission levels (in basis points) and
    returns one :class:`BacktestResult` per level.  This enables quick
    assessment of whether a strategy's edge survives realistic and
    adversarial transaction-cost assumptions.

    Args:
        strategy: Trading strategy implementing :class:`IStrategy`.
        sizer: Position sizer implementing :class:`IPositionSizer`.
        bars: Polars DataFrame with OHLCV columns sorted by timestamp.
        asset: Asset being backtested.
        fees: Commission levels in basis points.  Defaults to
            ``[5.0, 10.0, 15.0, 20.0, 30.0]``.
        initial_cash: Starting portfolio cash.
        base_config: Optional base configuration whose non-commission
            fields are inherited.  If *None*, defaults are used.

    Returns:
        Mapping from fee level (bps) to the corresponding backtest
        result.
    """
    if fees is None:
        fees = [5.0, 10.0, 15.0, 20.0, 30.0]

    # Inherit non-commission fields from base_config when provided
    asset_cost_multiplier: dict[str, float] = base_config.asset_cost_multiplier if base_config else {}
    min_trade_count: int = base_config.min_trade_count if base_config else 30
    cost_sweep_bps: list[float] = base_config.cost_sweep_bps if base_config else fees

    results: dict[float, BacktestResult] = {}
    for fee_bps in fees:
        config: ExecutionConfig = ExecutionConfig(
            commission_bps=fee_bps,
            asset_cost_multiplier=asset_cost_multiplier,
            min_trade_count=min_trade_count,
            cost_sweep_bps=cost_sweep_bps,
        )
        engine: ExecutionEngine = ExecutionEngine(config=config, strategy=strategy, sizer=sizer)
        results[fee_bps] = engine.run(bars=bars, asset=asset, initial_cash=initial_cash)

    return results
