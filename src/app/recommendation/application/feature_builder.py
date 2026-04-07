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
            cross: pl.DataFrame = self._build_cross_asset_features(result, btc_returns, universe_returns)
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
        ``vol_actual`` for error trend computation.

        Args:
            vol_df: Volatility forecast DataFrame.

        Returns:
            DataFrame with ``timestamp`` and vol feature columns.
        """
        cols: list[pl.Expr] = [pl.col("timestamp")]

        if "vol_predicted" in vol_df.columns:
            cols.append(pl.col("vol_predicted"))

        # Vol error trend: rolling mean of (predicted - actual) forecast error
        if "vol_predicted" in vol_df.columns and "vol_actual" in vol_df.columns:
            window: int = self._config.rolling_window
            cols.append(
                (pl.col("vol_predicted") - pl.col("vol_actual"))
                .rolling_mean(window_size=window, min_samples=1)
                .alias("vol_error_trend")
            )

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

        Computes volatility regime (HIGH/LOW) and MI significance proxy
        from ``vol_predicted`` if already present in ``df``.

        Args:
            df: Current feature DataFrame (may contain ``vol_predicted``).

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

        if exprs:
            result: pl.DataFrame = df.with_columns(exprs)
            return result

        return df

    def _build_cross_asset_features(
        self,
        df: pl.DataFrame,
        btc_returns: pl.DataFrame | None,
        universe_returns: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """Build cross-asset feature columns.

        Computes relative strength vs universe mean and BTC-related
        features for altcoins.

        Args:
            df: Current feature DataFrame.
            btc_returns: BTC return series with ``timestamp`` and
                ``btc_return``.
            universe_returns: Universe mean return series with
                ``timestamp`` and ``universe_mean_return``.

        Returns:
            DataFrame with additional cross-asset columns.
        """
        result: pl.DataFrame = df

        # BTC lagged return (lag 1) — Granger causality confirmed in RC2
        if btc_returns is not None and "btc_return" in btc_returns.columns:
            btc_lagged: pl.DataFrame = btc_returns.select(
                pl.col("timestamp"),
                pl.col("btc_return").shift(1).alias("btc_lagged_return"),
            )
            result = result.join(btc_lagged, on="timestamp", how="left")

            # Rolling cross-correlation with BTC
            if "close" in result.columns:
                window: int = self._config.rolling_window
                # We compute rolling correlation between asset returns and BTC returns
                # First join raw BTC returns for correlation computation
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

                result = result.with_columns(
                    pl.rolling_corr(
                        pl.col(asset_ret_col),
                        pl.col("btc_return"),
                        window_size=window,
                        min_samples=2,
                    ).alias("rolling_cross_corr")
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
