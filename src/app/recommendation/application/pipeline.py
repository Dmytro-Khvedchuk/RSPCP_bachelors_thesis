"""Walk-forward training pipeline for the ML recommendation system.

Orchestrates multi-layer walk-forward evaluation with strict temporal
purging between layers.  L1 (classifier/regressor) OOS predictions become
L2 (recommender) features, ensuring no information leakage across layers.
"""

from __future__ import annotations

from typing import Annotated, Self

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from pydantic import model_validator

from src.app.recommendation.application.feature_builder import RecommenderFeatureBuilder
from src.app.recommendation.application.label_builder import LabelBuilder
from src.app.recommendation.domain.protocols import IRecommender
from src.app.recommendation.domain.value_objects import Recommendation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel, frozen=True):
    """Configuration for the walk-forward recommender pipeline.

    Controls window count, train/test split ratio, purge/embargo bars,
    and label computation parameters.

    Attributes:
        n_windows: Number of walk-forward evaluation windows.
        train_ratio: Fraction of total data reserved for the initial
            training set.  Must be in ``(0.0, 1.0)``.
        purge_bars: Number of bars to discard between train and test
            to prevent information leakage from overlapping labels.
        embargo_bars: Additional bars after the purge gap before the
            test period begins.
        label_horizon: Fixed-horizon label horizon passed to
            :class:`LabelBuilder`.
        commission_bps: Transaction cost in basis points for label
            computation.
    """

    n_windows: Annotated[
        int,
        PydanticField(
            default=5,
            gt=0,
            description="Number of walk-forward evaluation windows",
        ),
    ]

    train_ratio: Annotated[
        float,
        PydanticField(
            default=0.6,
            gt=0.0,
            lt=1.0,
            description="Fraction of data for the initial training set",
        ),
    ]

    purge_bars: Annotated[
        int,
        PydanticField(
            default=10,
            ge=0,
            description="Bars to purge between train and test periods",
        ),
    ]

    embargo_bars: Annotated[
        int,
        PydanticField(
            default=5,
            ge=0,
            description="Additional embargo bars after purge",
        ),
    ]

    label_horizon: Annotated[
        int,
        PydanticField(
            default=7,
            gt=0,
            description="Fixed-horizon label horizon for LabelBuilder",
        ),
    ]

    commission_bps: Annotated[
        float,
        PydanticField(
            default=10.0,
            ge=0.0,
            description="Transaction cost in basis points for labels",
        ),
    ]

    @model_validator(mode="after")
    def _validate_purge_embargo(self) -> Self:
        """Ensure purge + embargo is non-negative (always true, but explicit).

        Returns:
            Validated instance.

        Raises:
            ValueError: If purge_bars + embargo_bars is negative.
        """
        total_gap: int = self.purge_bars + self.embargo_bars
        if total_gap < 0:
            msg: str = f"purge_bars + embargo_bars must be >= 0, got {total_gap}"
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Result value objects
# ---------------------------------------------------------------------------


class WindowResult(BaseModel, frozen=True):
    """Result from a single walk-forward window.

    Contains the train/test split boundaries, predictions from the
    primary recommender and all baselines, and realised test labels.

    Attributes:
        window_index: Zero-based window number.
        train_start_idx: Inclusive start index of the training period.
        train_end_idx: Exclusive end index of the training period.
        test_start_idx: Inclusive start index of the test period
            (after purge + embargo).
        test_end_idx: Exclusive end index of the test period.
        n_train_samples: Number of training samples after feature/label
            alignment and NaN removal.
        n_test_samples: Number of test samples.
        primary_predictions: Recommendations from the primary model.
        baseline_predictions: Mapping of baseline name to its recommendations.
        test_labels: Realised strategy returns on the test period.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    n_train_samples: int
    n_test_samples: int
    primary_predictions: list[Recommendation]
    baseline_predictions: dict[str, list[Recommendation]]
    test_labels: list[float]


class PipelineResult(BaseModel, frozen=True):
    """Aggregated result across all walk-forward windows.

    Attributes:
        window_results: Per-window results.
        all_primary_predictions: Concatenated primary predictions across
            all windows (OOS only).
        all_test_labels: Concatenated realised labels across all windows.
        n_windows_completed: Number of windows that produced valid results.
        n_windows_skipped: Number of windows skipped due to insufficient data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    window_results: list[WindowResult]
    all_primary_predictions: list[Recommendation]
    all_test_labels: list[float]
    n_windows_completed: int
    n_windows_skipped: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class RecommenderPipeline:
    """Walk-forward training pipeline with multi-layer temporal purging.

    Orchestrates expanding-window walk-forward evaluation where:

    * **L1** (classifier/regressor) OOS predictions are pre-computed and
      passed in as DataFrames, ensuring they are genuinely out-of-sample.
    * **L2** (recommender) trains on features assembled from L1 outputs
      plus market state, with labels from realised strategy returns.
    * **L3** (evaluation) predicts on the held-out test window and
      collects realised returns for metric computation.

    The pipeline uses an **expanding (anchored) window** approach:
    training always starts from index 0 and grows with each window.

    Attributes:
        config: Pipeline configuration.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        primary: IRecommender,
        baselines: list[IRecommender],
        baseline_names: list[str],
        config: PipelineConfig,
        feature_builder: RecommenderFeatureBuilder,
        label_builder: LabelBuilder,
    ) -> None:
        """Initialise the walk-forward pipeline.

        Args:
            primary: Primary recommender model (e.g. GradientBoostingRecommender).
            baselines: Baseline recommender models for comparison.
            baseline_names: Human-readable names for each baseline, in the
                same order as ``baselines``.
            config: Pipeline configuration.
            feature_builder: Assembles recommender feature vectors.
            label_builder: Computes fixed-horizon strategy return labels.

        Raises:
            ValueError: If ``baselines`` and ``baseline_names`` have
                different lengths.
        """
        if len(baselines) != len(baseline_names):
            msg: str = (
                f"baselines and baseline_names must have the same length, "
                f"got {len(baselines)} and {len(baseline_names)}"
            )
            raise ValueError(msg)

        self._primary: IRecommender = primary
        self._baselines: list[IRecommender] = baselines
        self._baseline_names: list[str] = baseline_names
        self._config: PipelineConfig = config
        self._feature_builder: RecommenderFeatureBuilder = feature_builder
        self._label_builder: LabelBuilder = label_builder

    @property
    def config(self) -> PipelineConfig:
        """Return the pipeline configuration.

        Returns:
            Frozen ``PipelineConfig`` instance.
        """
        return self._config

    def run(  # noqa: PLR0913, PLR0917
        self,
        bars: pl.DataFrame,
        signals: pl.DataFrame,
        market_features: pl.DataFrame,
        classifier_outputs: pl.DataFrame | None = None,
        regressor_outputs: pl.DataFrame | None = None,
        vol_forecasts: pl.DataFrame | None = None,
        strategy_returns: pl.DataFrame | None = None,
        btc_returns: pl.DataFrame | None = None,
        universe_returns: pl.DataFrame | None = None,
        *,
        asset_symbol: str = "UNKNOWN",
        strategy_name: str = "default",
    ) -> PipelineResult:
        """Execute the walk-forward pipeline across all windows.

        For each expanding window:

        1. Slice all input DataFrames to the train and test periods.
        2. Build recommender features (L2) on the train period.
        3. Compute labels on the train period.
        4. Train the primary and baseline recommenders.
        5. Build features and predict on the test period.
        6. Collect realised labels on the test period.

        Args:
            bars: OHLCV bars with ``timestamp`` and ``close``.
            signals: Strategy signals with ``timestamp`` and ``side``.
            market_features: Phase 4 features with ``timestamp``.
            classifier_outputs: Pre-computed L1 OOS classifier predictions.
            regressor_outputs: Pre-computed L1 OOS regressor predictions.
            vol_forecasts: Pre-computed L1 OOS volatility forecasts.
            strategy_returns: Historical strategy returns.
            btc_returns: BTC return series for cross-asset features.
            universe_returns: Universe mean return series.
            asset_symbol: Asset identifier for labeling and logging.
            strategy_name: Strategy identifier for labeling.

        Returns:
            Aggregated pipeline result across all windows.
        """
        _validate_run_inputs(bars, signals, market_features)

        n_total: int = len(bars)
        splits: list[tuple[int, int, int, int]] = _compute_splits(
            n_total=n_total,
            n_windows=self._config.n_windows,
            train_ratio=self._config.train_ratio,
            purge_bars=self._config.purge_bars,
            embargo_bars=self._config.embargo_bars,
        )

        logger.info(
            "Starting walk-forward pipeline: {} windows, {} total bars, asset={}",
            len(splits),
            n_total,
            asset_symbol,
        )

        window_results: list[WindowResult] = []
        all_primary: list[Recommendation] = []
        all_labels: list[float] = []
        n_skipped: int = 0

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
            result: WindowResult | None = self._run_single_window(
                window_index=w_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                bars=bars,
                signals=signals,
                market_features=market_features,
                classifier_outputs=classifier_outputs,
                regressor_outputs=regressor_outputs,
                vol_forecasts=vol_forecasts,
                strategy_returns=strategy_returns,
                btc_returns=btc_returns,
                universe_returns=universe_returns,
                asset_symbol=asset_symbol,
                strategy_name=strategy_name,
            )
            if result is None:
                n_skipped += 1
                continue

            window_results.append(result)
            all_primary.extend(result.primary_predictions)
            all_labels.extend(result.test_labels)

        n_completed: int = len(window_results)
        logger.info(
            "Walk-forward pipeline complete: {}/{} windows succeeded, {} total OOS predictions",
            n_completed,
            len(splits),
            len(all_primary),
        )

        return PipelineResult(
            window_results=window_results,
            all_primary_predictions=all_primary,
            all_test_labels=all_labels,
            n_windows_completed=n_completed,
            n_windows_skipped=n_skipped,
        )

    # ------------------------------------------------------------------
    # Private: single window execution
    # ------------------------------------------------------------------

    def _run_single_window(  # noqa: PLR0913, PLR0917, PLR0914
        self,
        *,
        window_index: int,
        train_start: int,
        train_end: int,
        test_start: int,
        test_end: int,
        bars: pl.DataFrame,
        signals: pl.DataFrame,
        market_features: pl.DataFrame,
        classifier_outputs: pl.DataFrame | None,
        regressor_outputs: pl.DataFrame | None,
        vol_forecasts: pl.DataFrame | None,
        strategy_returns: pl.DataFrame | None,
        btc_returns: pl.DataFrame | None,
        universe_returns: pl.DataFrame | None,
        asset_symbol: str,
        strategy_name: str,
    ) -> WindowResult | None:
        """Execute a single walk-forward window.

        Args:
            window_index: Zero-based window number.
            train_start: Inclusive start index of training period.
            train_end: Exclusive end index of training period.
            test_start: Inclusive start index of test period.
            test_end: Exclusive end index of test period.
            bars: Full OHLCV bars DataFrame.
            signals: Full strategy signals DataFrame.
            market_features: Full market features DataFrame.
            classifier_outputs: Full classifier outputs (or None).
            regressor_outputs: Full regressor outputs (or None).
            vol_forecasts: Full volatility forecasts (or None).
            strategy_returns: Full strategy returns (or None).
            btc_returns: Full BTC returns (or None).
            universe_returns: Full universe returns (or None).
            asset_symbol: Asset identifier.
            strategy_name: Strategy identifier.

        Returns:
            WindowResult if successful, None if the window was skipped
            due to insufficient data.
        """
        # Slice all DataFrames to train and test periods via row indices
        train_bars: pl.DataFrame = bars.slice(train_start, train_end - train_start)
        test_bars: pl.DataFrame = bars.slice(test_start, test_end - test_start)
        train_signals: pl.DataFrame = signals.slice(train_start, train_end - train_start)
        test_signals: pl.DataFrame = signals.slice(test_start, test_end - test_start)
        train_mf: pl.DataFrame = market_features.slice(train_start, train_end - train_start)
        test_mf: pl.DataFrame = market_features.slice(test_start, test_end - test_start)

        train_clf: pl.DataFrame | None = _safe_slice(classifier_outputs, train_start, train_end)
        test_clf: pl.DataFrame | None = _safe_slice(classifier_outputs, test_start, test_end)
        train_reg: pl.DataFrame | None = _safe_slice(regressor_outputs, train_start, train_end)
        test_reg: pl.DataFrame | None = _safe_slice(regressor_outputs, test_start, test_end)
        train_vol: pl.DataFrame | None = _safe_slice(vol_forecasts, train_start, train_end)
        test_vol: pl.DataFrame | None = _safe_slice(vol_forecasts, test_start, test_end)
        train_strat: pl.DataFrame | None = _safe_slice(strategy_returns, train_start, train_end)
        test_strat: pl.DataFrame | None = _safe_slice(strategy_returns, test_start, test_end)
        train_btc: pl.DataFrame | None = _safe_slice(btc_returns, train_start, train_end)
        test_btc: pl.DataFrame | None = _safe_slice(btc_returns, test_start, test_end)
        train_uni: pl.DataFrame | None = _safe_slice(universe_returns, train_start, train_end)
        test_uni: pl.DataFrame | None = _safe_slice(universe_returns, test_start, test_end)

        # --- Build TRAIN features ---
        train_features: pl.DataFrame = self._feature_builder.build_features(
            train_mf,
            classifier_outputs=train_clf,
            regressor_outputs=train_reg,
            vol_forecasts=train_vol,
            strategy_returns=train_strat,
            btc_returns=train_btc,
            universe_returns=train_uni,
            asset_symbol=asset_symbol,
        )

        # --- Build TRAIN labels ---
        train_labels: pl.DataFrame = self._label_builder.build_labels(
            train_bars,
            train_signals,
            asset_symbol,
            strategy_name,
        )

        if len(train_labels) == 0:
            logger.warning("Window {} skipped: no train labels produced", window_index)
            return None

        # --- Align features and labels on timestamp ---
        x_train_arr: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        y_train_arr: np.ndarray[tuple[int], np.dtype[np.float64]]
        feature_cols: list[str]
        x_train_arr, y_train_arr, feature_cols = _align_features_labels(train_features, train_labels)

        if len(y_train_arr) < 2:  # noqa: PLR2004
            logger.warning(
                "Window {} skipped: only {} aligned train samples (need >= 2)",
                window_index,
                len(y_train_arr),
            )
            return None

        # --- Fit primary recommender ---
        self._primary.fit(x_train_arr, y_train_arr)

        # --- Fit baselines ---
        for baseline in self._baselines:
            baseline.fit(x_train_arr, y_train_arr)

        # --- Build TEST features ---
        test_features: pl.DataFrame = self._feature_builder.build_features(
            test_mf,
            classifier_outputs=test_clf,
            regressor_outputs=test_reg,
            vol_forecasts=test_vol,
            strategy_returns=test_strat,
            btc_returns=test_btc,
            universe_returns=test_uni,
            asset_symbol=asset_symbol,
        )

        # --- Build TEST labels for evaluation ---
        test_labels_df: pl.DataFrame = self._label_builder.build_labels(
            test_bars,
            test_signals,
            asset_symbol,
            strategy_name,
        )

        # --- Align test features and labels ---
        x_test_arr: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        y_test_arr: np.ndarray[tuple[int], np.dtype[np.float64]]
        x_test_arr, y_test_arr, _ = _align_features_labels(test_features, test_labels_df, expected_cols=feature_cols)

        if len(y_test_arr) == 0:
            logger.warning("Window {} skipped: no aligned test samples", window_index)
            return None

        # --- Predict ---
        primary_recs: list[Recommendation] = self._primary.predict(x_test_arr)

        baseline_preds: dict[str, list[Recommendation]] = {}
        for name, baseline in zip(self._baseline_names, self._baselines, strict=True):
            baseline_recs: list[Recommendation] = baseline.predict(x_test_arr)
            baseline_preds[name] = baseline_recs

        test_labels_list: list[float] = y_test_arr.tolist()

        logger.info(
            "Window {}: train=[{}:{}) test=[{}:{}) | {} train, {} test samples",
            window_index,
            train_start,
            train_end,
            test_start,
            test_end,
            len(y_train_arr),
            len(y_test_arr),
        )

        return WindowResult(
            window_index=window_index,
            train_start_idx=train_start,
            train_end_idx=train_end,
            test_start_idx=test_start,
            test_end_idx=test_end,
            n_train_samples=len(y_train_arr),
            n_test_samples=len(y_test_arr),
            primary_predictions=primary_recs,
            baseline_predictions=baseline_preds,
            test_labels=test_labels_list,
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _validate_run_inputs(
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    market_features: pl.DataFrame,
) -> None:
    """Validate required input DataFrames.

    Args:
        bars: OHLCV bars DataFrame.
        signals: Strategy signals DataFrame.
        market_features: Market features DataFrame.

    Raises:
        ValueError: If any required DataFrame is empty or missing required columns.
    """
    if len(bars) == 0:
        msg: str = "bars DataFrame must not be empty"
        raise ValueError(msg)
    if "timestamp" not in bars.columns or "close" not in bars.columns:
        msg = "bars DataFrame must contain 'timestamp' and 'close' columns"
        raise ValueError(msg)
    if len(signals) == 0:
        msg = "signals DataFrame must not be empty"
        raise ValueError(msg)
    if "timestamp" not in signals.columns or "side" not in signals.columns:
        msg = "signals DataFrame must contain 'timestamp' and 'side' columns"
        raise ValueError(msg)
    if len(market_features) == 0:
        msg = "market_features DataFrame must not be empty"
        raise ValueError(msg)
    if "timestamp" not in market_features.columns:
        msg = "market_features DataFrame must contain a 'timestamp' column"
        raise ValueError(msg)


def _compute_splits(
    *,
    n_total: int,
    n_windows: int,
    train_ratio: float,
    purge_bars: int,
    embargo_bars: int,
) -> list[tuple[int, int, int, int]]:
    """Compute expanding-window train/test split boundaries.

    Uses anchored (expanding) windows where training always starts at
    index 0.  The initial split point is at ``train_ratio * n_total``.
    Subsequent split points are evenly spaced in the remaining data,
    creating ``n_windows`` evaluation windows.

    Each window:
    - Train:  ``[0, split_i)``
    - Purge:  ``[split_i, split_i + purge + embargo)``  (discarded)
    - Test:   ``[split_i + purge + embargo, split_{i+1})``

    Args:
        n_total: Total number of bars.
        n_windows: Number of walk-forward windows.
        train_ratio: Fraction for the initial training set.
        purge_bars: Bars to purge between train and test.
        embargo_bars: Additional embargo bars.

    Returns:
        List of ``(train_start, train_end, test_start, test_end)`` tuples.

    Raises:
        ValueError: If the data is too small for the requested configuration.
    """
    initial_split: int = int(n_total * train_ratio)
    total_gap: int = purge_bars + embargo_bars

    if initial_split < 2:  # noqa: PLR2004
        msg: str = (
            f"Insufficient data: initial training set would have {initial_split} bars "
            f"(need >= 2). Total bars={n_total}, train_ratio={train_ratio}"
        )
        raise ValueError(msg)

    remaining: int = n_total - initial_split
    if remaining <= total_gap:
        msg = (
            f"Insufficient data after initial split: {remaining} bars remaining, "
            f"but purge+embargo requires {total_gap} bars. "
            f"Total bars={n_total}, initial_split={initial_split}"
        )
        raise ValueError(msg)

    # Compute split points evenly in [initial_split, n_total]
    # split_points[0] = initial_split, split_points[-1] = n_total
    split_points: list[int] = []
    for i in range(n_windows + 1):
        point: float = initial_split + i * (n_total - initial_split) / n_windows
        split_points.append(int(round(point)))

    splits: list[tuple[int, int, int, int]] = []
    for w in range(n_windows):
        train_start: int = 0
        train_end: int = split_points[w]
        test_start: int = split_points[w] + total_gap
        test_end: int = split_points[w + 1]

        # Ensure test_start does not exceed test_end
        if test_start >= test_end:
            logger.warning(
                "Window {} has no test data after purge+embargo (test_start={} >= test_end={}), skipping",
                w,
                test_start,
                test_end,
            )
            continue

        # Ensure a minimum training set
        if train_end < 2:  # noqa: PLR2004
            logger.warning(
                "Window {} has insufficient training data (train_end={}), skipping",
                w,
                train_end,
            )
            continue

        splits.append((train_start, train_end, test_start, test_end))

    if len(splits) == 0:
        msg = (
            f"No valid walk-forward windows could be constructed. "
            f"n_total={n_total}, n_windows={n_windows}, "
            f"train_ratio={train_ratio}, purge={purge_bars}, embargo={embargo_bars}"
        )
        raise ValueError(msg)

    return splits


def _safe_slice(
    df: pl.DataFrame | None,
    start: int,
    end: int,
) -> pl.DataFrame | None:
    """Slice a DataFrame by row indices, returning None if input is None.

    Args:
        df: DataFrame to slice, or None.
        start: Inclusive start row index.
        end: Exclusive end row index.

    Returns:
        Sliced DataFrame, or None.
    """
    if df is None:
        return None
    return df.slice(start, end - start)


def _align_features_labels(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    expected_cols: list[str] | None = None,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    list[str],
]:
    """Align features and labels on timestamp, returning numpy arrays.

    Performs an inner join on ``timestamp``, drops rows with any NaN
    in the feature columns, and converts to numpy for model consumption.

    Args:
        features: Feature DataFrame with ``timestamp`` + feature columns.
        labels: Label DataFrame with ``timestamp`` and ``strategy_return``.
        expected_cols: If provided, select only these feature columns
            (ensures test features match train feature columns).

    Returns:
        Tuple of ``(x_array, y_array, feature_column_names)``.
    """
    if len(labels) == 0 or len(features) == 0:
        empty_x: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.empty((0, 0), dtype=np.float64)
        empty_y: np.ndarray[tuple[int], np.dtype[np.float64]] = np.empty(0, dtype=np.float64)
        feature_cols: list[str] = expected_cols if expected_cols is not None else []
        return empty_x, empty_y, feature_cols

    # Inner join on timestamp
    aligned: pl.DataFrame = features.join(
        labels.select("timestamp", "strategy_return"),
        on="timestamp",
        how="inner",
    )

    if len(aligned) == 0:
        empty_x = np.empty((0, 0), dtype=np.float64)
        empty_y = np.empty(0, dtype=np.float64)
        feature_cols = expected_cols if expected_cols is not None else []
        return empty_x, empty_y, feature_cols

    # Determine feature columns (everything except timestamp and strategy_return)
    all_cols: list[str] = aligned.columns
    feature_cols = [c for c in all_cols if c not in {"timestamp", "strategy_return"}]

    if expected_cols is not None:
        # Use only the expected columns, in the expected order
        available: set[str] = set(feature_cols)
        feature_cols = [c for c in expected_cols if c in available]
        # Fill missing expected columns with zeros
        missing_cols: list[str] = [c for c in expected_cols if c not in available]
        if missing_cols:
            fill_exprs: list[pl.Expr] = [pl.lit(0.0).alias(c) for c in missing_cols]
            aligned = aligned.with_columns(fill_exprs)
            feature_cols = expected_cols  # Restore full expected order

    if len(feature_cols) == 0:
        empty_x = np.empty((len(aligned), 0), dtype=np.float64)
        y_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = (
            aligned.get_column("strategy_return").to_numpy().astype(np.float64)
        )
        return empty_x, y_arr, feature_cols

    # Drop rows where the label is null, then fill remaining feature nulls
    # with 0.0.  Rolling features (permutation entropy, rolling Sharpe, etc.)
    # produce null at window edges — this is expected and 0.0 is a neutral
    # fill for tree-based models.
    feature_df: pl.DataFrame = aligned.select(feature_cols + ["strategy_return"])
    label_present: pl.DataFrame = feature_df.filter(pl.col("strategy_return").is_not_null())

    if len(label_present) == 0:
        empty_x = np.empty((0, len(feature_cols)), dtype=np.float64)
        empty_y = np.empty(0, dtype=np.float64)
        return empty_x, empty_y, feature_cols

    filled: pl.DataFrame = label_present.with_columns([pl.col(c).fill_null(0.0) for c in feature_cols])

    x_arr: np.ndarray[tuple[int, int], np.dtype[np.float64]] = (
        filled.select(feature_cols).to_numpy().astype(np.float64)
    )
    y_arr = filled.get_column("strategy_return").to_numpy().astype(np.float64)

    return x_arr, y_arr, feature_cols
