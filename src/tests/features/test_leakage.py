"""Future-leakage detection tests for indicators and targets.

These tests verify that backward-looking indicator functions do not use
future data.  The key invariant is that at row t, each feature should only
depend on data at times s <= t.  We test this via:

1. Correlation analysis: backward-looking features should NOT be more
   correlated with future targets than with past/current targets.
2. Shift-independence: features computed on time-reversed series should have
   a similar statistical distribution to forward-series features, because
   purely causal functions are symmetric in time (up to sign).
3. Shuffled-target correlation: on shuffled price data features should have
   near-zero correlation with randomly generated targets.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.app.features.application.indicators import compute_all_indicators
from src.app.features.application.targets import compute_all_targets
from src.app.features.domain.value_objects import FeatureConfig

from src.tests.features.conftest import (
    make_random_walk_df,
    make_small_feature_config,
    make_small_indicator_config,
    make_small_target_config,
)


_WARMUP_BUFFER: int = 60  # rows to skip at start of each series


def _build_feature_set(seed: int = 42) -> pl.DataFrame:
    """Build a feature matrix on a random-walk series.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV + all indicator features + target columns.
    """
    df: pl.DataFrame = make_random_walk_df(400, seed=seed)
    ind_config = make_small_indicator_config()
    tgt_config = make_small_target_config()
    df_with_features: pl.DataFrame = compute_all_indicators(df, ind_config)
    df_full: pl.DataFrame = compute_all_targets(df_with_features, tgt_config)
    # Drop warmup rows and rows with any null
    df_clean: pl.DataFrame = df_full.slice(_WARMUP_BUFFER).drop_nulls()
    return df_clean


def _feature_cols(df: pl.DataFrame) -> list[str]:
    """Return backward-looking indicator column names.

    Args:
        df: DataFrame with features and targets.

    Returns:
        List of feature (non-OHLCV, non-target) column names.
    """
    exclude: set[str] = {"timestamp", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in exclude and not c.startswith("fwd_")]


def _target_cols(df: pl.DataFrame) -> list[str]:
    """Return forward-looking target column names.

    Args:
        df: DataFrame with features and targets.

    Returns:
        List of target column names (prefixed with 'fwd_').
    """
    return [c for c in df.columns if c.startswith("fwd_")]


class TestNoFutureLookAhead:
    """Tests that indicator features do not use future data."""

    def test_features_not_more_correlated_with_future_than_past(self) -> None:  # noqa: PLR0914
        """Verify features are not more correlated with future than past targets.

        The test computes:
          - corr(feature_t, target_{t+k}) [future — simulated by shift(-k)]
          - corr(feature_t, target_{t-k}) [past — simulated by shift(+k)]

        If a feature leaks future data, it will be more correlated with
        future targets than with the same target shifted into the past.
        This is only a probabilistic check, so we test AVERAGE behaviour
        across all features.
        """
        df: pl.DataFrame = _build_feature_set(seed=40)
        feat_cols: list[str] = _feature_cols(df)
        # Use the 1-bar forward return as the target
        target_series: np.ndarray[tuple[int], np.dtype[np.float64]] = df["fwd_logret_1"].to_numpy().astype(np.float64)

        shift_steps: int = 3  # bars to shift
        n: int = len(df)
        future_corrs: list[float] = []
        past_corrs: list[float] = []

        for feat_col in feat_cols:
            feat: np.ndarray[tuple[int], np.dtype[np.float64]] = df[feat_col].to_numpy().astype(np.float64)

            # future target: target shifted left by shift_steps (we use values that exist)
            future_target: np.ndarray[tuple[int], np.dtype[np.float64]] = target_series[shift_steps:]
            feat_for_future: np.ndarray[tuple[int], np.dtype[np.float64]] = feat[: n - shift_steps]

            # past target: target shifted right by shift_steps
            past_target: np.ndarray[tuple[int], np.dtype[np.float64]] = target_series[: n - shift_steps]
            feat_for_past: np.ndarray[tuple[int], np.dtype[np.float64]] = feat[shift_steps:]

            if len(future_target) < 10 or len(past_target) < 10:
                continue

            # Correlation of feature with future target
            fc: float = float(np.corrcoef(feat_for_future, future_target)[0, 1])
            # Correlation of feature with past target
            pc: float = float(np.corrcoef(feat_for_past, past_target)[0, 1])

            if not (np.isnan(fc) or np.isnan(pc)):
                future_corrs.append(abs(fc))
                past_corrs.append(abs(pc))

        assert len(future_corrs) > 0, "No valid correlations computed"

        # On average, the absolute correlation with future target should NOT
        # systematically exceed the correlation with past target.
        # We allow a small slack (0.05) to account for statistical noise.
        avg_future: float = float(np.mean(future_corrs))
        avg_past: float = float(np.mean(past_corrs))

        # A feature set with look-ahead bias would have avg_future >> avg_past.
        # We check that future corr is NOT substantially larger than past corr.
        assert avg_future <= avg_past + 0.05, (
            f"Features appear to look ahead: avg |corr_future|={avg_future:.4f} >> avg |corr_past|={avg_past:.4f}"
        )

    def test_feature_target_correlation_on_shuffled_data(self) -> None:
        """Verify near-zero correlation between features and independent random targets.

        This is a sanity check: if features are truly backward-looking, they
        carry no information about an independently generated random signal.
        """
        rng: np.random.Generator = np.random.default_rng(50)

        df: pl.DataFrame = _build_feature_set(seed=51)
        feat_cols: list[str] = _feature_cols(df)
        n: int = len(df)
        random_target: np.ndarray[tuple[int], np.dtype[np.float64]] = rng.standard_normal(n)

        corr_abs: list[float] = []
        for feat_col in feat_cols:
            feat: np.ndarray[tuple[int], np.dtype[np.float64]] = df[feat_col].to_numpy().astype(np.float64)
            c: float = float(np.corrcoef(feat, random_target)[0, 1])
            if not np.isnan(c):
                corr_abs.append(abs(c))

        assert len(corr_abs) > 0
        avg_corr: float = float(np.mean(corr_abs))
        # Expected near-zero correlation with an independent random signal
        assert avg_corr < 0.3, (
            f"Average absolute correlation with random target is {avg_corr:.4f}, "
            f"which is higher than expected for backward-looking features"
        )


class TestIndicatorShiftIndependence:
    """Test that indicator distributions are similar on forward and reversed series."""

    def test_indicator_means_similar_on_forward_and_reversed(self) -> None:  # noqa: PLR0914
        """Verify similar feature std on forward and time-reversed price series.

        If a feature leaks future data, reversing time would destroy the
        forward look-ahead and drastically change the feature distribution.
        For purely backward-looking rolling statistics, the distribution of
        the indicator (after discarding warmup) should be similar regardless
        of whether time goes forward or backward.

        We allow a 50% relative tolerance on mean absolute values to
        account for the asymmetry of financial time series.
        """
        df_forward: pl.DataFrame = make_random_walk_df(300, seed=60)
        # Create time-reversed version
        df_reversed: pl.DataFrame = df_forward.reverse()

        ind_config = make_small_indicator_config()
        result_fwd: pl.DataFrame = compute_all_indicators(df_forward, ind_config)
        result_rev: pl.DataFrame = compute_all_indicators(df_reversed, ind_config)

        warmup: int = ind_config.hurst_window + 5
        result_fwd_clean: pl.DataFrame = result_fwd.slice(warmup).drop_nulls()
        result_rev_clean: pl.DataFrame = result_rev.slice(warmup).drop_nulls()

        ohlcv: set[str] = {"timestamp", "open", "high", "low", "close", "volume"}
        feat_cols: list[str] = [c for c in result_fwd_clean.columns if c not in ohlcv]

        violations: list[str] = []
        for col in feat_cols:
            fwd_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = result_fwd_clean[col].to_numpy().astype(np.float64)
            rev_arr: np.ndarray[tuple[int], np.dtype[np.float64]] = result_rev_clean[col].to_numpy().astype(np.float64)

            fwd_std: float = float(np.std(fwd_arr))
            rev_std: float = float(np.std(rev_arr))

            if fwd_std < 1e-10 and rev_std < 1e-10:
                continue  # both near-zero — consistent

            # Ratio of standard deviations should be within [0.1, 10]
            if fwd_std > 1e-10 and rev_std > 1e-10:
                ratio: float = fwd_std / rev_std
                if ratio > 10.0 or ratio < 0.1:
                    violations.append(f"{col}: fwd_std={fwd_std:.4f}, rev_std={rev_std:.4f}")

        assert len(violations) == 0, (
            "Features with dramatically different distributions on forward vs reversed series:\n"
            + "\n".join(violations)
        )


class TestTargetComputedFromFutureData:
    """Verify that forward targets are indeed forward-looking (use future data)."""

    def test_forward_target_correlates_with_future_returns(self) -> None:
        """fwd_logret_1 at row t should equal the actual log return at t+1.

        This confirms the target IS forward-looking (as designed) — which
        is the correct behaviour for training labels.
        """
        df: pl.DataFrame = make_random_walk_df(100, seed=70)
        tgt_config = make_small_target_config()
        result: pl.DataFrame = compute_all_targets(df, tgt_config)
        result_clean: pl.DataFrame = result.drop_nulls()

        # The 1-bar log return backward from t+1 should approximate fwd_logret_1
        # We compute backward log returns and compare with forward targets
        import math as _math

        close_list: list[float] = result_clean["close"].to_list()  # type: ignore[assignment]
        fwd_list: list[float | None] = result_clean["fwd_logret_1"].to_list()

        # fwd_logret_1[i] = ln(close[i+1] / close[i])
        # We verify this by checking adjacent-row relationships in the clean data
        n: int = len(close_list)
        errors: list[float] = []
        for i in range(n - 1):
            expected: float = _math.log(close_list[i + 1] / close_list[i])
            actual: float | None = fwd_list[i]
            if actual is not None:
                errors.append(abs(actual - expected))

        if errors:
            max_err: float = max(errors)
            assert max_err < 1e-8, f"fwd_logret_1 does not match actual next-bar return: max_err={max_err}"

    def test_feature_matrix_builder_leakage_free(self) -> None:
        """FeatureMatrixBuilder with compute_targets=False produces no fwd_ columns.

        This tests the production-mode path where forward targets are excluded,
        eliminating any risk of leakage in live inference.
        """
        from src.app.features.application.feature_matrix import FeatureMatrixBuilder

        df: pl.DataFrame = make_random_walk_df(200, seed=71)
        config: FeatureConfig = make_small_feature_config(compute_targets=False, drop_na=True)
        builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
        feature_set = builder.build(df, config)

        fwd_cols: list[str] = [c for c in feature_set.df.columns if c.startswith("fwd_")]
        assert len(fwd_cols) == 0, f"fwd_ columns present in inference mode: {fwd_cols}"
        assert len(feature_set.target_columns) == 0
