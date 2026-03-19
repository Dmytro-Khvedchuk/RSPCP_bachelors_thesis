"""Property-based tests for the compute_all_indicators orchestrator.

Verifies global properties that must hold for all indicators across all
non-null rows after the warmup period: finiteness, correct shape,
clipping bounds, and determinism.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from src.app.features.application.indicators import compute_all_indicators
from src.app.features.domain.value_objects import IndicatorConfig

from src.tests.features.conftest import (
    make_random_walk_df,
    make_small_indicator_config,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _feature_cols(df: pl.DataFrame, config: IndicatorConfig) -> list[str]:
    """Return the list of feature columns (non-OHLCV columns).

    Args:
        df: DataFrame output from compute_all_indicators.
        config: Config used for computation (not inspected here, kept for
            symmetry).

    Returns:
        List of column names that are indicator features.
    """
    ohlcv: set[str] = {"timestamp", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in ohlcv]


def _expected_feature_count(config: IndicatorConfig) -> int:  # noqa: PLR0914
    """Compute the expected number of feature columns for the given config.

    Args:
        config: IndicatorConfig to count expected columns for.

    Returns:
        Total expected feature column count.
    """
    n_returns: int = len(config.return_horizons)
    n_rv: int = len(config.realized_vol_windows)
    n_gk: int = 1
    n_park: int = 1
    n_atr: int = 1
    n_ema_xover: int = 1
    n_rsi: int = 1
    n_roc: int = len(config.roc_periods)
    n_vol_zscore: int = 1
    n_amihud: int = 1
    n_ret_zscore: int = 1
    n_bbpctb: int = 1
    n_bbwidth: int = 1
    n_slope: int = 1
    n_obv_slope: int = 1
    n_hurst: int = 1

    return (
        n_returns
        + n_rv
        + n_gk
        + n_park
        + n_atr
        + n_ema_xover
        + n_rsi
        + n_roc
        + n_vol_zscore
        + n_amihud
        + n_ret_zscore
        + n_bbpctb
        + n_bbwidth
        + n_slope
        + n_obv_slope
        + n_hurst
    )


class TestPropertyFinite:
    """Property tests: all feature values must be finite after warmup."""

    def test_all_features_finite_after_warmup(self) -> None:
        """No NaN or inf values should appear in feature columns after warmup.

        Warmup period is estimated as the hurst_window since that is the
        longest rolling window in compute_all_indicators.
        """
        n: int = 300
        df: pl.DataFrame = make_random_walk_df(n, seed=20)
        config: IndicatorConfig = make_small_indicator_config()
        result: pl.DataFrame = compute_all_indicators(df, config)

        warmup: int = config.hurst_window + 5  # small safety margin
        result_after_warmup: pl.DataFrame = result.slice(warmup)

        feat_cols: list[str] = _feature_cols(result_after_warmup, config)
        for col in feat_cols:
            series: pl.Series = result_after_warmup[col]
            null_count: int = series.null_count()
            assert null_count == 0, f"Column '{col}' has {null_count} nulls after warmup"
            arr: list[float | None] = series.to_list()
            non_null_arr: list[float] = [v for v in arr if v is not None]
            for val in non_null_arr:
                assert math.isfinite(val), f"Column '{col}' has non-finite value {val}"

    def test_all_features_finite_large_df(self) -> None:
        """Same finiteness guarantee holds on a large (500-row) DataFrame."""
        n: int = 500
        df: pl.DataFrame = make_random_walk_df(n, seed=21)
        config: IndicatorConfig = make_small_indicator_config()
        result: pl.DataFrame = compute_all_indicators(df, config)

        warmup: int = config.hurst_window + 5
        result_after_warmup: pl.DataFrame = result.slice(warmup)

        feat_cols: list[str] = _feature_cols(result_after_warmup, config)
        for col in feat_cols:
            null_count: int = result_after_warmup[col].null_count()
            assert null_count == 0, f"Column '{col}' has {null_count} nulls after warmup (large df)"


class TestPropertyShape:
    """Property tests: feature count and row count must match config."""

    def test_feature_count_matches_config(self) -> None:
        """Number of feature columns must equal the expected count from config."""
        df: pl.DataFrame = make_random_walk_df(150, seed=22)
        config: IndicatorConfig = make_small_indicator_config()
        result: pl.DataFrame = compute_all_indicators(df, config)
        feat_cols: list[str] = _feature_cols(result, config)
        expected_count: int = _expected_feature_count(config)
        assert len(feat_cols) == expected_count

    def test_features_shape_row_count_preserved(self) -> None:
        """compute_all_indicators must not drop or add rows."""
        n: int = 200
        df: pl.DataFrame = make_random_walk_df(n, seed=23)
        config: IndicatorConfig = make_small_indicator_config()
        result: pl.DataFrame = compute_all_indicators(df, config)
        assert len(result) == n

    def test_feature_count_changes_with_config(self) -> None:
        """Adding an extra return horizon increases the feature count by 1."""
        df: pl.DataFrame = make_random_walk_df(150, seed=24)
        config_base: IndicatorConfig = make_small_indicator_config()
        config_extra: IndicatorConfig = make_small_indicator_config(return_horizons=(1, 4, 8))

        result_base: pl.DataFrame = compute_all_indicators(df, config_base)
        result_extra: pl.DataFrame = compute_all_indicators(df, config_extra)

        n_base: int = len(_feature_cols(result_base, config_base))
        n_extra: int = len(_feature_cols(result_extra, config_extra))
        assert n_extra == n_base + 1


class TestPropertyClipping:
    """Property tests: all feature values must respect clip bounds."""

    def test_features_clipped_within_bounds(self) -> None:
        """All feature column values must be within [clip_lower, clip_upper]."""
        df: pl.DataFrame = make_random_walk_df(200, seed=25)
        lo: float = -3.0
        hi: float = 3.0
        config: IndicatorConfig = make_small_indicator_config(clip_lower=lo, clip_upper=hi)
        result: pl.DataFrame = compute_all_indicators(df, config)

        feat_cols: list[str] = _feature_cols(result, config)
        for col in feat_cols:
            series: pl.Series = result[col].drop_nulls()
            if len(series) == 0:
                continue
            col_min: float = float(series.min())  # type: ignore[arg-type]
            col_max: float = float(series.max())  # type: ignore[arg-type]
            assert col_min >= lo - 1e-6, f"{col} min {col_min} violates clip_lower={lo}"
            assert col_max <= hi + 1e-6, f"{col} max {col_max} violates clip_upper={hi}"

    def test_features_clipped_narrow_bounds(self) -> None:
        """Features with clip_lower=-1.0, clip_upper=1.0 must all be in [-1, 1]."""
        df: pl.DataFrame = make_random_walk_df(200, seed=26)
        lo: float = -1.0
        hi: float = 1.0
        config: IndicatorConfig = make_small_indicator_config(clip_lower=lo, clip_upper=hi)
        result: pl.DataFrame = compute_all_indicators(df, config)

        feat_cols: list[str] = _feature_cols(result, config)
        for col in feat_cols:
            series: pl.Series = result[col].drop_nulls()
            if len(series) == 0:
                continue
            col_min: float = float(series.min())  # type: ignore[arg-type]
            col_max: float = float(series.max())  # type: ignore[arg-type]
            assert col_min >= lo - 1e-6, f"{col}: min {col_min} < {lo}"
            assert col_max <= hi + 1e-6, f"{col}: max {col_max} > {hi}"


class TestPropertyDeterminism:
    """Property tests: indicator computation must be deterministic."""

    def test_deterministic_output_same_input(self) -> None:
        """Same input DataFrame always produces identical feature values."""
        df: pl.DataFrame = make_random_walk_df(150, seed=27)
        config: IndicatorConfig = make_small_indicator_config()

        result_1: pl.DataFrame = compute_all_indicators(df, config)
        result_2: pl.DataFrame = compute_all_indicators(df, config)

        feat_cols: list[str] = _feature_cols(result_1, config)
        for col in feat_cols:
            vals_1: list[float | None] = result_1[col].to_list()
            vals_2: list[float | None] = result_2[col].to_list()
            assert vals_1 == pytest.approx(vals_2, rel=1e-9, nan_ok=True), f"Column '{col}' is not deterministic"

    def test_different_seeds_produce_different_results(self) -> None:
        """Different random walks should produce different indicator values."""
        config: IndicatorConfig = make_small_indicator_config()
        df_a: pl.DataFrame = make_random_walk_df(150, seed=28)
        df_b: pl.DataFrame = make_random_walk_df(150, seed=99)

        result_a: pl.DataFrame = compute_all_indicators(df_a, config)
        result_b: pl.DataFrame = compute_all_indicators(df_b, config)

        # At least the close-based features must differ
        col: str = "logret_1"
        vals_a: list[float | None] = result_a[col].to_list()
        vals_b: list[float | None] = result_b[col].to_list()
        assert vals_a != vals_b
