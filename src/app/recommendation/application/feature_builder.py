"""Recommender feature builder — assembles feature vectors for the ML recommendation system."""

from __future__ import annotations

from typing import Annotated

import polars as pl
from loguru import logger
from pydantic import BaseModel
from pydantic import Field as PydanticField


# ---------------------------------------------------------------------------
# RecommenderFeatureConfig
# ---------------------------------------------------------------------------


class RecommenderFeatureConfig(BaseModel, frozen=True):
    """Configuration for recommender feature assembly.

    Controls rolling window sizes, regime thresholds, and reference
    volatility for MI significance proxy.

    Attributes:
        rolling_window: Window size for rolling statistics (accuracy,
            Sharpe, win rate, cross-correlation).
        vol_regime_threshold: Standard deviations above mean volatility
            to classify as HIGH regime.
        mi_reference_vol: Reference realised volatility from the
            2022--2023 crypto winter period.  When rolling vol exceeds
            this value, the MI significance proxy fires.  ``None``
            disables this feature.
    """

    rolling_window: Annotated[
        int,
        PydanticField(
            default=20,
            gt=1,
            description="Window for rolling statistics (accuracy, Sharpe, win rate)",
        ),
    ]

    vol_regime_threshold: Annotated[
        float,
        PydanticField(
            default=1.0,
            ge=0.0,
            description="Std devs above mean vol for HIGH regime classification",
        ),
    ]

    mi_reference_vol: Annotated[
        float | None,
        PydanticField(
            default=None,
            description="Reference vol from 2022-2023 for MI significance proxy",
        ),
    ]

    perm_entropy_dim: Annotated[
        int,
        PydanticField(
            default=3,
            ge=2,
            le=7,
            description="Embedding dimension for rolling permutation entropy (m)",
        ),
    ]

    perm_entropy_delay: Annotated[
        int,
        PydanticField(
            default=1,
            ge=1,
            description="Time delay for rolling permutation entropy (tau)",
        ),
    ]


# ---------------------------------------------------------------------------
# RecommenderFeatureBuilder
# ---------------------------------------------------------------------------


class RecommenderFeatureBuilder:
    """Assembles feature vectors for the ML recommendation system.

    Combines market state, classifier output, regressor output, volatility
    forecasts, regime indicators, cross-asset features, and historical
    strategy performance into a single feature DataFrame aligned by
    ``timestamp``.

    Each upstream input is optional: when ``None``, the corresponding
    feature group is silently skipped (graceful degradation).  This
    enables incremental development — the recommender can start with
    market features alone and gain signal as upstream models come online.

    Attributes:
        config: Feature assembly configuration.
    """

    def __init__(self, config: RecommenderFeatureConfig | None = None) -> None:
        """Initialise the feature builder.

        Args:
            config: Feature configuration.  Uses defaults if not provided.
        """
        self._config: RecommenderFeatureConfig = config or RecommenderFeatureConfig()  # ty: ignore[missing-argument]

    @property
    def config(self) -> RecommenderFeatureConfig:
        """Return the feature configuration.

        Returns:
            Frozen ``RecommenderFeatureConfig`` instance.
        """
        return self._config

    def build_features(  # noqa: PLR0913, PLR0917
        self,
        market_features: pl.DataFrame,
        classifier_outputs: pl.DataFrame | None = None,
        regressor_outputs: pl.DataFrame | None = None,
        vol_forecasts: pl.DataFrame | None = None,
        strategy_returns: pl.DataFrame | None = None,
        btc_returns: pl.DataFrame | None = None,
        universe_returns: pl.DataFrame | None = None,
        *,
        asset_symbol: str | None = None,
    ) -> pl.DataFrame:
        """Assemble the full recommender feature vector.

        Starts with market features as the backbone and progressively
        joins additional feature groups when their upstream data is
        available.  All joins are on ``timestamp``.

        Args:
            market_features: Phase 4 feature matrix with ``timestamp`` and
                market indicators (volatility, momentum, Hurst, etc.).
            classifier_outputs: Direction forecasts with ``timestamp``,
                ``clf_direction`` (+1/-1), ``clf_confidence``, and
                optionally ``clf_correct`` (bool for rolling accuracy).
            regressor_outputs: Return forecasts with ``timestamp``,
                ``reg_predicted_return``, ``reg_prediction_std``, and
                optionally ``reg_quantile_spread``, ``reg_ci_width``.
            vol_forecasts: Volatility forecasts with ``timestamp``,
                ``vol_predicted``, and optionally ``vol_actual`` (for
                error trend computation).
            strategy_returns: Historical strategy returns with
                ``timestamp`` and ``strategy_return`` (for rolling Sharpe
                and win rate).
            btc_returns: BTC return series with ``timestamp`` and
                ``btc_return`` (for cross-asset features on altcoins).
            universe_returns: Universe-wide return series with
                ``timestamp`` and ``universe_mean_return`` (for relative
                strength computation).
            asset_symbol: Asset symbol (e.g. ``"BTCUSDT"``).  When the
                symbol contains ``"BTC"``, BTC-specific cross-asset
                features (lagged return, beta, cross-correlation) are
                skipped to avoid self-reference.

        Returns:
            Polars DataFrame with ``timestamp`` and all assembled feature
            columns, one row per decision point.
        """
        _validate_market_features(market_features)

        result: pl.DataFrame = market_features

        # Each feature group is assembled independently and joined
        if classifier_outputs is not None:
            clf_features: pl.DataFrame = self._build_classifier_features(classifier_outputs)
            result = result.join(clf_features, on="timestamp", how="left")

        if regressor_outputs is not None:
            reg_features: pl.DataFrame = self._build_regressor_features(regressor_outputs)
            result = result.join(reg_features, on="timestamp", how="left")

        if vol_forecasts is not None:
            vol_features: pl.DataFrame = self._build_vol_features(vol_forecasts)
            result = result.join(vol_features, on="timestamp", how="left")

        # Combined forecast features require both classifier and regressor
        if classifier_outputs is not None and regressor_outputs is not None:
            combined: pl.DataFrame = self._build_combined_features(result)
            # Combined features are computed in-place from already-joined columns
            result = combined

        # Regime features (computed from volatility if available)
        regime: pl.DataFrame = self._build_regime_features(result)
        result = regime

        # Cross-asset features
        if btc_returns is not None or universe_returns is not None:
            cross: pl.DataFrame = self._build_cross_asset_features(
                result, btc_returns, universe_returns, asset_symbol=asset_symbol
            )
            result = cross

        # Historical strategy features
        if strategy_returns is not None:
            strat: pl.DataFrame = self._build_strategy_features(strategy_returns)
            result = result.join(strat, on="timestamp", how="left")

        n_features: int = len(result.columns) - 1  # exclude timestamp
        logger.info(
            "Assembled {} recommender features across {} rows",
            n_features,
            len(result),
        )

        return result

    # ------------------------------------------------------------------
    # Private feature group builders
    # ------------------------------------------------------------------

    def _build_classifier_features(self, clf_df: pl.DataFrame) -> pl.DataFrame:
        """Build classifier feature columns.

        Expects ``timestamp``, ``clf_direction``, ``clf_confidence``,
        and optionally ``clf_correct`` (bool) for rolling accuracy.

        Args:
            clf_df: Classifier output DataFrame.

        Returns:
            DataFrame with ``timestamp`` and classifier feature columns.
        """
        cols: list[pl.Expr] = [pl.col("timestamp")]

        if "clf_direction" in clf_df.columns:
            cols.append(pl.col("clf_direction"))

        if "clf_confidence" in clf_df.columns:
            cols.append(pl.col("clf_confidence"))

        # Rolling accuracy from clf_correct boolean column
        if "clf_correct" in clf_df.columns:
            window: int = self._config.rolling_window
            cols.append(
                pl.col("clf_correct")
                .cast(pl.Float64)
                .rolling_mean(window_size=window, min_samples=1)
                .alias("clf_rolling_accuracy")
            )

        result: pl.DataFrame = clf_df.select(cols)
        return result

    @staticmethod
    def _build_regressor_features(reg_df: pl.DataFrame) -> pl.DataFrame:
        """Build regressor feature columns.

        Expects ``timestamp``, ``reg_predicted_return``,
        ``reg_prediction_std``, and optionally ``reg_quantile_spread``,
        ``reg_ci_width``.

        Args:
            reg_df: Regressor output DataFrame.

        Returns:
            DataFrame with ``timestamp`` and regressor feature columns.
        """
        passthrough_cols: list[str] = [
            "reg_predicted_return",
            "reg_prediction_std",
            "reg_quantile_spread",
            "reg_ci_width",
        ]
        cols: list[pl.Expr] = [pl.col("timestamp")]
        cols.extend(pl.col(col_name) for col_name in passthrough_cols if col_name in reg_df.columns)

        result: pl.DataFrame = reg_df.select(cols)
        return result

    def _build_vol_features(self, vol_df: pl.DataFrame) -> pl.DataFrame:
        """Build volatility forecast feature columns.

        Expects ``timestamp``, ``vol_predicted``, and optionally
        ``vol_actual`` for error trend and QLIKE residual computation.

        The QLIKE loss is defined as::

            QLIKE(σ²_pred, σ²_actual) = σ²_pred / σ²_actual
                                        - ln(σ²_pred / σ²_actual) - 1

        We compute this point-wise and then take a rolling mean to
        produce ``vol_qlike_residual``.  Values near zero indicate a
        well-calibrated variance forecast; positive values indicate
        systematic over- or under-prediction.

        Args:
            vol_df: Volatility forecast DataFrame.

        Returns:
            DataFrame with ``timestamp`` and vol feature columns.
        """
        cols: list[pl.Expr] = [pl.col("timestamp")]

        if "vol_predicted" in vol_df.columns:
            cols.append(pl.col("vol_predicted"))

        has_both: bool = "vol_predicted" in vol_df.columns and "vol_actual" in vol_df.columns
        if has_both:
            window: int = self._config.rolling_window

            # Vol error trend: rolling mean of (predicted - actual) forecast error
            cols.append(
                (pl.col("vol_predicted") - pl.col("vol_actual"))
                .rolling_mean(window_size=window, min_samples=1)
                .alias("vol_error_trend")
            )

            # QLIKE residual: predicted/actual - ln(predicted/actual) - 1
            # Guard against vol_actual <= 0 to avoid division by zero / log domain error.
            # When vol_actual is non-positive, the ratio is undefined — emit null.
            safe_ratio: pl.Expr = pl.when(pl.col("vol_actual") > 0.0).then(
                pl.col("vol_predicted") / pl.col("vol_actual")
            )
            qlike_pointwise: pl.Expr = safe_ratio - safe_ratio.log() - 1.0
            cols.append(qlike_pointwise.rolling_mean(window_size=window, min_samples=1).alias("vol_qlike_residual"))

        result: pl.DataFrame = vol_df.select(cols)
        return result

    @staticmethod
    def _build_combined_features(df: pl.DataFrame) -> pl.DataFrame:
        """Build combined forecast features from already-joined classifier and regressor columns.

        Requires ``clf_direction`` and ``reg_predicted_return`` to be
        present in ``df``.  Computes forecast agreement and conviction
        score.

        Args:
            df: DataFrame with both classifier and regressor columns joined.

        Returns:
            DataFrame with additional combined feature columns.
        """
        exprs: list[pl.Expr] = []

        has_direction: bool = "clf_direction" in df.columns
        has_return: bool = "reg_predicted_return" in df.columns

        if has_direction and has_return:
            # Forecast agreement: classifier direction matches regressor sign
            exprs.append(
                pl.when(
                    (pl.col("clf_direction") > 0) & (pl.col("reg_predicted_return") > 0)
                    | (pl.col("clf_direction") < 0) & (pl.col("reg_predicted_return") < 0)
                )
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("forecast_agreement")
            )

        has_confidence: bool = "clf_confidence" in df.columns
        if has_return and has_confidence:
            # Conviction score: |predicted_return| * classifier_confidence
            exprs.append((pl.col("reg_predicted_return").abs() * pl.col("clf_confidence")).alias("conviction_score"))

        if exprs:
            result: pl.DataFrame = df.with_columns(exprs)
            return result

        return df

    def _build_regime_features(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Build regime indicator features.

        Computes volatility regime (HIGH/LOW), MI significance proxy
        from ``vol_predicted``, and rolling permutation entropy from
        ``close`` prices if present in ``df``.

        Permutation entropy (Bandt & Pompe, 2002) measures the
        complexity / randomness of a time series.  High values (near 1)
        indicate random-walk-like behaviour; low values indicate
        structured (predictable) regimes.

        Args:
            df: Current feature DataFrame (may contain ``vol_predicted``
                and/or ``close``).

        Returns:
            DataFrame with additional regime columns.
        """
        exprs: list[pl.Expr] = []

        # Volatility regime: HIGH if vol > mean + threshold * std
        if "vol_predicted" in df.columns:
            threshold: float = self._config.vol_regime_threshold
            vol_mean: float | None = df.get_column("vol_predicted").mean()  # ty: ignore[invalid-assignment]
            vol_std: float | None = df.get_column("vol_predicted").std()  # ty: ignore[invalid-assignment]
            if vol_mean is not None and vol_std is not None and vol_std > 0.0:
                cutoff: float = vol_mean + threshold * vol_std
                exprs.append(
                    pl.when(pl.col("vol_predicted") > cutoff).then(pl.lit(1)).otherwise(pl.lit(0)).alias("vol_regime")
                )

        # MI significance proxy: rolling vol > reference vol from 2022-2023
        ref_vol: float | None = self._config.mi_reference_vol
        if ref_vol is not None and "vol_predicted" in df.columns:
            exprs.append(
                pl.when(pl.col("vol_predicted") > ref_vol)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("mi_significant_regime")
            )

        # Apply expression-based features first
        result: pl.DataFrame = df.with_columns(exprs) if exprs else df

        # Rolling permutation entropy (requires close prices for returns)
        if "close" in result.columns:
            pe_values: pl.Series = _compute_rolling_permutation_entropy(
                result.get_column("close"),
                window=self._config.rolling_window,
                m=self._config.perm_entropy_dim,
                delay=self._config.perm_entropy_delay,
            )
            result = result.with_columns(pe_values.alias("rolling_perm_entropy"))

        return result

    def _build_cross_asset_features(
        self,
        df: pl.DataFrame,
        btc_returns: pl.DataFrame | None,
        universe_returns: pl.DataFrame | None,
        *,
        asset_symbol: str | None = None,
    ) -> pl.DataFrame:
        """Build cross-asset feature columns.

        Computes relative strength vs universe mean and BTC-related
        features for altcoins.  BTC-specific features (lagged return,
        rolling correlation, beta) are skipped when ``asset_symbol``
        contains ``"BTC"`` to avoid self-reference.

        Args:
            df: Current feature DataFrame.
            btc_returns: BTC return series with ``timestamp`` and
                ``btc_return``.
            universe_returns: Universe mean return series with
                ``timestamp`` and ``universe_mean_return``.
            asset_symbol: Asset symbol.  When it contains ``"BTC"``,
                BTC-specific cross-asset features are suppressed.

        Returns:
            DataFrame with additional cross-asset columns.
        """
        result: pl.DataFrame = df

        # Determine whether this asset IS BTC — skip self-referencing features
        is_btc: bool = asset_symbol is not None and "BTC" in asset_symbol.upper()

        # BTC-related features: only for altcoins (not BTC itself)
        if not is_btc and btc_returns is not None and "btc_return" in btc_returns.columns:
            # BTC lagged return (lag 1) — Granger causality confirmed in RC2
            btc_lagged: pl.DataFrame = btc_returns.select(
                pl.col("timestamp"),
                pl.col("btc_return").shift(1).alias("btc_lagged_return"),
            )
            result = result.join(btc_lagged, on="timestamp", how="left")

            # Rolling cross-correlation and beta require asset returns
            if "close" in result.columns:
                window: int = self._config.rolling_window
                # Join raw BTC returns for correlation / beta computation
                result = result.join(
                    btc_returns.select("timestamp", "btc_return"),
                    on="timestamp",
                    how="left",
                    suffix="_raw",
                )
                # Compute simple return for the asset if not already present
                if "asset_return" not in result.columns:
                    result = result.with_columns(
                        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("_asset_return_tmp")
                    )
                    asset_ret_col: str = "_asset_return_tmp"
                else:
                    asset_ret_col = "asset_return"

                # Rolling cross-correlation with BTC
                result = result.with_columns(
                    pl.rolling_corr(
                        pl.col(asset_ret_col),
                        pl.col("btc_return"),
                        window_size=window,
                        min_samples=2,
                    ).alias("rolling_cross_corr")
                )

                # Beta to BTC: cov(asset, BTC) / var(BTC) over rolling window
                btc_rolling_var: pl.Expr = pl.col("btc_return").rolling_var(window_size=window, min_samples=2)
                rolling_cov: pl.Expr = _rolling_cov_expr(
                    pl.col(asset_ret_col),
                    pl.col("btc_return"),
                    window_size=window,
                    min_samples=2,
                )
                result = result.with_columns(
                    pl.when(btc_rolling_var > 0.0)
                    .then(rolling_cov / btc_rolling_var)
                    .otherwise(pl.lit(0.0))
                    .alias("btc_beta")
                )

                # Clean up temporary columns
                drop_cols: list[str] = [c for c in ["_asset_return_tmp", "btc_return"] if c in result.columns]
                if drop_cols:
                    result = result.drop(drop_cols)

        # Relative strength: asset return vs universe mean
        if universe_returns is not None and "universe_mean_return" in universe_returns.columns:
            result = result.join(
                universe_returns.select("timestamp", "universe_mean_return"),
                on="timestamp",
                how="left",
            )
            if "close" in result.columns:
                result = result.with_columns(
                    ((pl.col("close") / pl.col("close").shift(1) - 1.0) - pl.col("universe_mean_return")).alias(
                        "relative_strength"
                    )
                )
                result = result.drop("universe_mean_return")

        return result

    def _build_strategy_features(self, strategy_df: pl.DataFrame) -> pl.DataFrame:
        """Build historical strategy performance features.

        Computes rolling Sharpe ratio and rolling win rate from
        historical strategy returns.

        Args:
            strategy_df: Historical strategy returns with ``timestamp``
                and ``strategy_return`` columns.

        Returns:
            DataFrame with ``timestamp``, ``rolling_strategy_sharpe``,
            and ``rolling_win_rate``.
        """
        window: int = self._config.rolling_window
        cols: list[pl.Expr] = [pl.col("timestamp")]

        if "strategy_return" in strategy_df.columns:
            # Rolling Sharpe: mean / std (annualization omitted — relative comparison)
            rolling_mean: pl.Expr = pl.col("strategy_return").rolling_mean(window_size=window, min_samples=2)
            rolling_std: pl.Expr = pl.col("strategy_return").rolling_std(window_size=window, min_samples=2)
            cols.append(
                pl.when(rolling_std > 0.0)
                .then(rolling_mean / rolling_std)
                .otherwise(pl.lit(0.0))
                .alias("rolling_strategy_sharpe")
            )

            # Rolling win rate: fraction of positive returns
            cols.append(
                pl.col("strategy_return")
                .gt(0.0)
                .cast(pl.Float64)
                .rolling_mean(window_size=window, min_samples=1)
                .alias("rolling_win_rate")
            )

        result: pl.DataFrame = strategy_df.select(cols)
        return result


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _validate_market_features(df: pl.DataFrame) -> None:
    """Validate the market features DataFrame.

    Args:
        df: Market features DataFrame.

    Raises:
        ValueError: If ``timestamp`` column is missing or DataFrame is empty.
    """
    if len(df) == 0:
        msg: str = "market_features DataFrame must not be empty"
        raise ValueError(msg)
    if "timestamp" not in df.columns:
        msg = "market_features DataFrame must contain a 'timestamp' column"
        raise ValueError(msg)


def _permutation_entropy(values: list[float], m: int, delay: int) -> float | None:
    """Compute normalised permutation entropy for a single window.

    Implements Bandt & Pompe (2002) with normalisation to ``[0, 1]``.
    A value of 1.0 means maximum complexity (uniform distribution of
    ordinal patterns); 0.0 means perfectly deterministic.

    Args:
        values: Sequence of observations (length >= ``m * delay``).
        m: Embedding dimension (number of elements in each pattern).
        delay: Time delay between successive elements in a pattern.

    Returns:
        Normalised permutation entropy in ``[0, 1]``, or ``None`` when
        the window is too short to form any pattern.
    """
    import math  # noqa: PLC0415

    n: int = len(values)
    required_length: int = (m - 1) * delay + 1
    if n < required_length:
        return None

    # Count ordinal patterns
    pattern_counts: dict[tuple[int, ...], int] = {}
    n_patterns: int = 0
    for i in range(n - required_length + 1):
        # Extract m values at positions i, i+delay, i+2*delay, ...
        window_vals: list[float] = [values[i + j * delay] for j in range(m)]
        # Convert to rank pattern (argsort of argsort gives ranks)
        indexed: list[tuple[float, int]] = sorted((v, idx) for idx, v in enumerate(window_vals))
        pattern: tuple[int, ...] = tuple(rank for rank, (_val, _orig_idx) in enumerate(indexed))
        # Re-index: we want the rank at each original position
        rank_at_pos: list[int] = [0] * m
        for rank_val, (_sort_val, orig_idx) in enumerate(indexed):
            rank_at_pos[orig_idx] = rank_val
        pattern = tuple(rank_at_pos)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        n_patterns += 1

    if n_patterns == 0:
        return None

    # Shannon entropy normalised by log(m!)
    max_entropy: float = math.log(math.factorial(m))
    if max_entropy == 0.0:
        return None

    entropy: float = 0.0
    for count in pattern_counts.values():
        prob: float = count / n_patterns
        if prob > 0.0:
            entropy -= prob * math.log(prob)

    normalised: float = entropy / max_entropy
    return normalised


def _compute_rolling_permutation_entropy(
    close_series: pl.Series,
    window: int,
    m: int,
    delay: int,
) -> pl.Series:
    """Compute rolling permutation entropy over a price series.

    First converts prices to simple returns, then applies a rolling
    window of ``_permutation_entropy`` to each window of returns.

    Args:
        close_series: Close price series.
        window: Rolling window size.
        m: Embedding dimension for permutation entropy.
        delay: Time delay for permutation entropy.

    Returns:
        Float64 Series of rolling permutation entropy values.  The
        first ``window - 1`` entries (plus the first return entry)
        will be ``None``.
    """
    import numpy as np  # noqa: PLC0415

    # Convert close prices to simple returns
    close_np: np.ndarray[tuple[int], np.dtype[np.float64]] = close_series.to_numpy()
    n: int = len(close_np)

    if n < 2:  # noqa: PLR2004
        return pl.Series("rolling_perm_entropy", [None] * n, dtype=pl.Float64)

    returns: np.ndarray[tuple[int], np.dtype[np.float64]] = np.diff(close_np) / close_np[:-1]

    # Compute rolling PE over returns
    n_returns: int = len(returns)
    pe_values: list[float | None] = [None]  # first close has no return

    for i in range(n_returns):
        if i < window - 1:
            pe_values.append(None)
        else:
            window_start: int = i - window + 1
            window_slice: list[float] = returns[window_start : i + 1].tolist()
            pe_val: float | None = _permutation_entropy(window_slice, m=m, delay=delay)
            pe_values.append(pe_val)

    result: pl.Series = pl.Series("rolling_perm_entropy", pe_values, dtype=pl.Float64)
    return result


def _rolling_cov_expr(
    col_a: pl.Expr,
    col_b: pl.Expr,
    window_size: int,
    min_samples: int,
) -> pl.Expr:
    """Build a Polars expression for rolling covariance.

    Uses the identity: ``cov(A, B) = E[AB] - E[A]E[B]``.

    Args:
        col_a: First column expression.
        col_b: Second column expression.
        window_size: Rolling window size.
        min_samples: Minimum non-null samples required.

    Returns:
        Polars expression computing rolling covariance.
    """
    mean_ab: pl.Expr = (col_a * col_b).rolling_mean(window_size=window_size, min_samples=min_samples)
    mean_a: pl.Expr = col_a.rolling_mean(window_size=window_size, min_samples=min_samples)
    mean_b: pl.Expr = col_b.rolling_mean(window_size=window_size, min_samples=min_samples)
    cov_expr: pl.Expr = mean_ab - mean_a * mean_b
    return cov_expr
