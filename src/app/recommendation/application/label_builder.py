"""Label builder — fixed-horizon strategy return labels for recommender training."""

from __future__ import annotations

from typing import Annotated

import polars as pl
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

_REQUIRED_BAR_COLUMNS: frozenset[str] = frozenset({"timestamp", "close"})
_REQUIRED_SIGNAL_COLUMNS: frozenset[str] = frozenset({"timestamp", "side"})

_VALID_SIDES: frozenset[str] = frozenset({"long", "short", "flat"})


# ---------------------------------------------------------------------------
# LabelConfig
# ---------------------------------------------------------------------------


class LabelConfig(BaseModel, frozen=True):
    """Configuration for fixed-horizon strategy return label computation.

    Controls the look-ahead horizon, transaction cost deduction, and
    minimum data requirements for label validity.

    Attributes:
        label_horizon: Number of bars to look ahead for return computation.
        commission_bps: Transaction costs in basis points (1 bp = 0.01 percent).
            Applied as a round-trip cost (entry + exit).
        min_bars_for_label: Minimum bars required in the forward window for a
            label to be emitted.  Defaults to ``label_horizon``.
    """

    label_horizon: Annotated[
        int,
        PydanticField(
            default=7,
            gt=0,
            description="Number of bars to look ahead for return computation",
        ),
    ]

    commission_bps: Annotated[
        float,
        PydanticField(
            default=10.0,
            ge=0.0,
            description="Round-trip transaction cost in basis points",
        ),
    ]

    min_bars_for_label: Annotated[
        int | None,
        PydanticField(
            default=None,
            ge=1,
            description="Minimum bars in forward window; defaults to label_horizon",
        ),
    ]

    @property
    def effective_min_bars(self) -> int:
        """Return the effective minimum bars threshold.

        Returns:
            ``min_bars_for_label`` if set, otherwise ``label_horizon``.
        """
        if self.min_bars_for_label is not None:
            return self.min_bars_for_label
        return self.label_horizon


# ---------------------------------------------------------------------------
# LabelBuilder
# ---------------------------------------------------------------------------


class LabelBuilder:
    """Computes fixed-horizon strategy return labels for recommender training.

    For each decision point where the strategy has a directional signal
    (long or short), the builder computes the net return over
    ``[t, t + label_horizon]`` after subtracting round-trip transaction
    costs.  Flat signals are excluded since there is no trade to evaluate.

    This implements Khandani et al. (2010) fixed-horizon labeling,
    generalized for the meta-labeling framework (Lopez de Prado, 2018).

    Attributes:
        config: Label computation configuration.
    """

    def __init__(self, config: LabelConfig | None = None) -> None:
        """Initialise the label builder.

        Args:
            config: Label configuration.  Uses defaults if not provided.
        """
        self._config: LabelConfig = config or LabelConfig()  # ty: ignore[missing-argument]

    @property
    def config(self) -> LabelConfig:
        """Return the label configuration.

        Returns:
            Frozen ``LabelConfig`` instance.
        """
        return self._config

    def build_labels(  # noqa: PLR0914
        self,
        bars: pl.DataFrame,
        signals: pl.DataFrame,
        asset_symbol: str,
        strategy_name: str,
    ) -> pl.DataFrame:
        """Compute fixed-horizon strategy return labels at decision points.

        For each bar *t* where ``side`` is ``"long"`` or ``"short"``:

        1. Forward return = ``close[t + H] / close[t] - 1``
        2. Directional return = forward return if long, negated if short
        3. Net return = directional return - round-trip cost

        Bars where ``t + H`` exceeds data length are dropped.

        Args:
            bars: OHLCV bars with at least ``timestamp`` and ``close``
                columns, sorted by ``timestamp``.
            signals: Strategy signals with ``timestamp`` and ``side``
                columns.  ``side`` must be one of ``"long"``, ``"short"``,
                or ``"flat"``.
            asset_symbol: Asset identifier (e.g. ``"BTCUSDT"``).
            strategy_name: Strategy identifier (e.g. ``"momentum_crossover"``).

        Returns:
            Polars DataFrame with columns: ``timestamp``, ``strategy_return``,
            ``asset``, ``strategy``, ``side``, ``horizon``.
        """
        _validate_bars(bars)
        _validate_signals(signals)

        active: pl.DataFrame = self._merge_and_filter(bars, signals, asset_symbol, strategy_name)
        if len(active) == 0:
            return _empty_label_frame()

        return self._compute_labels(bars, active, asset_symbol, strategy_name)

    @staticmethod
    def _merge_and_filter(
        bars: pl.DataFrame,
        signals: pl.DataFrame,
        asset_symbol: str,
        strategy_name: str,
    ) -> pl.DataFrame:
        """Merge bars with signals and filter to directional signals only.

        Args:
            bars: OHLCV bars DataFrame.
            signals: Strategy signals DataFrame.
            asset_symbol: Asset identifier for logging.
            strategy_name: Strategy identifier for logging.

        Returns:
            Filtered DataFrame with only ``"long"`` / ``"short"`` rows,
            or an empty DataFrame if no matches.
        """
        merged: pl.DataFrame = bars.select("timestamp", "close").join(
            signals.select("timestamp", "side"),
            on="timestamp",
            how="inner",
        )

        if len(merged) == 0:
            logger.warning(
                "No matching timestamps between bars and signals for {} / {}",
                asset_symbol,
                strategy_name,
            )
            return merged.clear()

        active: pl.DataFrame = merged.filter(pl.col("side").is_in(["long", "short"]))

        if len(active) == 0:
            logger.info(
                "All signals are flat for {} / {} — no labels to compute",
                asset_symbol,
                strategy_name,
            )

        return active

    def _compute_labels(
        self,
        bars: pl.DataFrame,
        active: pl.DataFrame,
        asset_symbol: str,
        strategy_name: str,
    ) -> pl.DataFrame:
        """Compute net strategy returns for each active decision point.

        Args:
            bars: Full OHLCV bars DataFrame (for forward close lookup).
            active: Filtered DataFrame of directional signals with close prices.
            asset_symbol: Asset identifier for metadata columns.
            strategy_name: Strategy identifier for metadata columns.

        Returns:
            Label DataFrame, or empty label frame if no valid labels remain.
        """
        horizon: int = self._config.label_horizon
        min_bars: int = self._config.effective_min_bars
        round_trip_cost: float = 2.0 * self._config.commission_bps / 10_000

        n_rows: int = len(bars)
        close_col: list[float] = bars.get_column("close").to_list()
        ts_col: list[object] = bars.get_column("timestamp").to_list()

        ts_to_idx: dict[object, int] = {ts_col[i]: i for i in range(n_rows)}

        result_timestamps: list[object] = []
        result_returns: list[float] = []
        result_sides: list[str] = []

        active_timestamps: list[object] = active.get_column("timestamp").to_list()
        active_sides: list[str] = active.get_column("side").to_list()
        active_closes: list[float] = active.get_column("close").to_list()

        for i in range(len(active)):
            net_return: float | None = _compute_single_label(
                bar_idx=ts_to_idx.get(active_timestamps[i]),
                close_t=active_closes[i],
                close_col=close_col,
                n_rows=n_rows,
                horizon=horizon,
                min_bars=min_bars,
                round_trip_cost=round_trip_cost,
                side=active_sides[i],
            )
            if net_return is not None:
                result_timestamps.append(active_timestamps[i])
                result_returns.append(net_return)
                result_sides.append(active_sides[i])

        if len(result_timestamps) == 0:
            logger.info(
                "No labels after horizon filtering for {} / {} (horizon={}, bars={})",
                asset_symbol,
                strategy_name,
                horizon,
                n_rows,
            )
            return _empty_label_frame()

        labels: pl.DataFrame = pl.DataFrame(
            {
                "timestamp": result_timestamps,
                "strategy_return": result_returns,
                "asset": [asset_symbol] * len(result_timestamps),
                "strategy": [strategy_name] * len(result_timestamps),
                "side": result_sides,
                "horizon": [horizon] * len(result_timestamps),
            }
        )

        logger.info(
            "Built {} labels for {} / {} (horizon={}, mean_return={:.6f})",
            len(labels),
            asset_symbol,
            strategy_name,
            horizon,
            labels.get_column("strategy_return").mean(),
        )

        return labels


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _compute_single_label(  # noqa: PLR0913
    *,
    bar_idx: int | None,
    close_t: float,
    close_col: list[float],
    n_rows: int,
    horizon: int,
    min_bars: int,
    round_trip_cost: float,
    side: str,
) -> float | None:
    """Compute the net strategy return for a single decision point.

    Args:
        bar_idx: Index of this bar in the full bars array, or ``None``.
        close_t: Close price at decision time.
        close_col: Full close price list from the bars DataFrame.
        n_rows: Total number of bars.
        horizon: Look-ahead horizon in bars.
        min_bars: Minimum forward bars required.
        round_trip_cost: Pre-computed round-trip transaction cost.
        side: Trade direction (``"long"`` or ``"short"``).

    Returns:
        Net strategy return, or ``None`` if the label cannot be computed.
    """
    if bar_idx is None:
        return None

    future_idx: int = bar_idx + horizon
    remaining_bars: int = n_rows - bar_idx - 1
    if remaining_bars < min_bars or future_idx >= n_rows:
        return None

    if close_t == 0.0:
        return None

    forward_return: float = close_col[future_idx] / close_t - 1.0
    directional_return: float = forward_return if side == "long" else -forward_return
    return directional_return - round_trip_cost


def _validate_bars(bars: pl.DataFrame) -> None:
    """Validate the bars DataFrame has required columns and is non-empty.

    Args:
        bars: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if len(bars) == 0:
        msg: str = "bars DataFrame must not be empty"
        raise ValueError(msg)
    missing: frozenset[str] = _REQUIRED_BAR_COLUMNS - set(bars.columns)
    if missing:
        msg = f"bars DataFrame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def _validate_signals(signals: pl.DataFrame) -> None:
    """Validate the signals DataFrame has required columns and is non-empty.

    Args:
        signals: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or DataFrame is empty.
    """
    if len(signals) == 0:
        msg: str = "signals DataFrame must not be empty"
        raise ValueError(msg)
    missing: frozenset[str] = _REQUIRED_SIGNAL_COLUMNS - set(signals.columns)
    if missing:
        msg = f"signals DataFrame is missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def _empty_label_frame() -> pl.DataFrame:
    """Return an empty DataFrame with the label schema.

    Returns:
        Empty Polars DataFrame with the correct column names and types.
    """
    return pl.DataFrame(
        {
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "strategy_return": pl.Series([], dtype=pl.Float64),
            "asset": pl.Series([], dtype=pl.String),
            "strategy": pl.Series([], dtype=pl.String),
            "side": pl.Series([], dtype=pl.String),
            "horizon": pl.Series([], dtype=pl.Int64),
        }
    )
