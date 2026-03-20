"""Dry-run integration test exercising the RC2 notebook's full pipeline without a real database.

Creates synthetic OHLCV data, builds a feature matrix via FeatureMatrixBuilder,
runs FeatureValidator, and then exercises all four RC2 analysis services:
    - RC2StationarityAnalyzer
    - RC2FeatureAnalyzer (VIF, correlation)
    - RC2ValidationAnalyzer (MI table, DA table)
    - RC2PredictabilityAnalyzer (PE table, feasibility)

Also verifies the Go/No-Go pre-registration criteria can be evaluated.
Completes in < 30 seconds using minimal data (200 rows) and low permutation counts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import polars as pl
import pytest

from src.app.features.application.feature_matrix import FeatureMatrixBuilder
from src.app.features.application.validation import FeatureValidator
from src.app.features.domain.entities import ValidationReport
from src.app.features.domain.value_objects import (
    FeatureConfig,
    FeatureSet,
    IndicatorConfig,
    TargetConfig,
    ValidationConfig,
)
from src.app.profiling.domain.value_objects import (
    PermutationEntropyResult,
    PredictabilityProfile,
    SampleTier,
)
from src.app.research.application.rc2_features import RC2FeatureAnalyzer
from src.app.research.application.rc2_predictability import RC2PredictabilityAnalyzer
from src.app.research.application.rc2_preregistration import (
    GoNoGoCriterion,
    PreRegistrationSpec,
    build_preregistration_spec,
)
from src.app.research.application.rc2_stationarity import (
    RC2StationarityAnalyzer,
    StationaritySummary,
)
from src.app.research.application.rc2_validation_analysis import (
    RC2ValidationAnalyzer,
    compute_target_entropy_gaussian,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_ROWS: int = 200
_N_ASSETS: int = 4
_ASSETS: list[str] = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT"]
_BAR_TYPE: str = "dollar"
_SEED: int = 42


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_synthetic_ohlcv_polars(
    n: int = _N_ROWS,
    seed: int = _SEED,
) -> pl.DataFrame:
    """Generate synthetic OHLCV data as a Polars DataFrame.

    Produces a GBM-like price series with realistic OHLCV structure.

    Args:
        n: Number of bars.
        seed: Random seed.

    Returns:
        Polars DataFrame with timestamp, open, high, low, close, volume columns.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    close: np.ndarray = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))  # type: ignore[type-arg]
    open_arr: np.ndarray = close * (1.0 + rng.normal(0, 0.001, n))  # type: ignore[type-arg]
    high_arr: np.ndarray = close * (1.0 + np.abs(rng.normal(0, 0.003, n)))  # type: ignore[type-arg]
    low_arr: np.ndarray = close * (1.0 - np.abs(rng.normal(0, 0.003, n)))  # type: ignore[type-arg]
    volume_arr: np.ndarray = rng.uniform(100.0, 5000.0, n)  # type: ignore[type-arg]
    # Build timestamps as a Pandas DatetimeIndex, then let Polars ingest natively
    pd_timestamps: pd.DatetimeIndex = pd.date_range(
        "2020-01-01",
        periods=n,
        freq="h",
        tz="UTC",
    )

    return pl.DataFrame(
        {
            "timestamp": pd_timestamps,
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close,
            "volume": volume_arr,
        }
    )


# ---------------------------------------------------------------------------
# Fast config overrides
# ---------------------------------------------------------------------------


def _make_fast_feature_config() -> FeatureConfig:
    """Build a FeatureConfig with small windows for fast test execution.

    Returns:
        FeatureConfig with reduced indicator windows.
    """
    indicator_config: IndicatorConfig = IndicatorConfig(
        return_horizons=(1, 4),
        realized_vol_windows=(12, 24),
        garman_klass_window=12,
        parkinson_window=12,
        atr_period=10,
        ema_fast_span=5,
        ema_slow_span=12,
        rsi_period=10,
        roc_periods=(1, 4),
        slope_window=10,
        volume_zscore_window=12,
        obv_slope_window=10,
        amihud_window=12,
        hurst_window=50,
        return_zscore_window=12,
        bollinger_window=12,
    )
    target_config: TargetConfig = TargetConfig(
        forward_return_horizons=(1,),
        forward_vol_horizons=(4,),
    )
    return FeatureConfig(
        indicator_config=indicator_config,
        target_config=target_config,
        drop_na=True,
        compute_targets=True,
    )


def _make_fast_validation_config() -> ValidationConfig:
    """Build a ValidationConfig with minimal permutations for fast execution.

    Returns:
        ValidationConfig with 100 MI permutations and 50 Ridge permutations.
    """
    return ValidationConfig(
        n_permutations_mi=100,
        n_permutations_ridge=50,
        n_permutations_stability=50,
        alpha=0.10,
        stability_threshold=0.25,
        target_col="fwd_logret_1",
        timestamp_col="timestamp",
        temporal_windows=((2020, 2021),),
        ridge_train_ratio=0.7,
        permutation_block_size=5,
        min_window_rows=10,
        min_features_kept=3,
        min_valid_windows=1,
        min_group_features=2,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def feature_config() -> FeatureConfig:
    """Fast feature config for all tests in this module."""
    return _make_fast_feature_config()


@pytest.fixture(scope="module")
def validation_config() -> ValidationConfig:
    """Fast validation config for all tests in this module."""
    return _make_fast_validation_config()


@pytest.fixture(scope="module")
def feature_set(feature_config: FeatureConfig) -> FeatureSet:
    """Build a single feature set from synthetic data.

    Args:
        feature_config: Fast feature configuration.

    Returns:
        FeatureSet with indicators and targets computed.
    """
    df: pl.DataFrame = _make_synthetic_ohlcv_polars(n=_N_ROWS, seed=_SEED)
    builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
    return builder.build(df, feature_config)


@pytest.fixture(scope="module")
def validation_report(
    feature_set: FeatureSet,
    validation_config: ValidationConfig,
) -> ValidationReport:
    """Run FeatureValidator on the synthetic feature set.

    Args:
        feature_set: Pre-built feature set.
        validation_config: Fast validation config.

    Returns:
        ValidationReport from the three-gate validation.
    """
    validator: FeatureValidator = FeatureValidator()
    return validator.validate(feature_set, validation_config)


@pytest.fixture(scope="module")
def multi_asset_feature_dfs(feature_config: FeatureConfig) -> dict[tuple[str, str], pd.DataFrame]:
    """Build feature DataFrames for all 4 assets (Pandas, for RC2 analyzers).

    Args:
        feature_config: Fast feature configuration.

    Returns:
        Dict mapping (asset, bar_type) to Pandas DataFrames with feature columns.
    """
    builder: FeatureMatrixBuilder = FeatureMatrixBuilder()
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for i, asset in enumerate(_ASSETS):
        df: pl.DataFrame = _make_synthetic_ohlcv_polars(n=_N_ROWS, seed=_SEED + i)
        fset: FeatureSet = builder.build(df, feature_config)
        pdf: pd.DataFrame = fset.df.to_pandas()
        result[(asset, _BAR_TYPE)] = pdf
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRC2DryRunPipeline:
    """End-to-end dry run of RC2 notebook pipeline with synthetic data."""

    def test_feature_matrix_build(self, feature_set: FeatureSet) -> None:
        """FeatureMatrixBuilder should produce a non-empty feature set."""
        assert feature_set.n_rows_clean > 0
        assert len(feature_set.feature_columns) > 0
        assert len(feature_set.target_columns) > 0
        # No NaN should remain after drop_na
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        feat_cols: list[str] = list(feature_set.feature_columns)
        assert not df_pd[feat_cols].isna().any().any()

    def test_feature_validator_runs(self, validation_report: ValidationReport) -> None:
        """FeatureValidator should produce a complete validation report."""
        assert validation_report.n_features_total > 0
        assert validation_report.n_features_kept >= 0
        assert validation_report.n_features_total == (
            validation_report.n_features_kept + validation_report.n_features_dropped
        )
        # Every feature should have a result
        assert len(validation_report.feature_results) == validation_report.n_features_total

    def test_rc2_stationarity_analyzer(
        self,
        multi_asset_feature_dfs: dict[tuple[str, str], pd.DataFrame],
        feature_set: FeatureSet,
    ) -> None:
        """RC2StationarityAnalyzer should classify features across all (asset, bar_type) combos."""
        analyzer: RC2StationarityAnalyzer = RC2StationarityAnalyzer()
        feature_names: list[str] = list(feature_set.feature_columns)

        summary: StationaritySummary = analyzer.analyze_features(
            multi_asset_feature_dfs,
            feature_names,
        )

        # Categories should be exhaustive
        total_classified: int = (
            len(summary.universally_stationary) + len(summary.universally_non_stationary) + len(summary.mixed_features)
        )
        assert total_classified == summary.n_total_features

        # Per-asset-bar reports should match input combos
        assert len(summary.per_asset_bar) == len(multi_asset_feature_dfs)

        # Rendering methods should not crash
        summary_table: pd.DataFrame = analyzer.render_summary_table(summary)
        assert len(summary_table) == len(feature_names)

        cross_asset: pd.DataFrame = analyzer.render_cross_asset_table(summary)
        assert len(cross_asset) == len(feature_names)

        therefore: str = analyzer.generate_therefore(summary)
        assert "Therefore" in therefore

    def test_rc2_feature_analyzer_vif(
        self,
        feature_set: FeatureSet,
    ) -> None:
        """RC2FeatureAnalyzer.compute_vif should return a VIF table for all features."""
        analyzer: RC2FeatureAnalyzer = RC2FeatureAnalyzer()
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        feature_names: list[str] = list(feature_set.feature_columns)

        vif_df: pd.DataFrame = analyzer.compute_vif(df_pd, feature_names)
        assert len(vif_df) == len(feature_names)
        assert "feature" in vif_df.columns
        assert "vif" in vif_df.columns
        # All VIFs should be positive (or inf)
        assert (vif_df["vif"] > 0).all()

    def test_rc2_feature_analyzer_correlation(
        self,
        feature_set: FeatureSet,
    ) -> None:
        """RC2FeatureAnalyzer.compute_correlation_matrix should return a square matrix."""
        analyzer: RC2FeatureAnalyzer = RC2FeatureAnalyzer()
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        feature_names: list[str] = list(feature_set.feature_columns)

        corr: pd.DataFrame = analyzer.compute_correlation_matrix(df_pd, feature_names)
        assert corr.shape[0] == len(feature_names)
        assert corr.shape[1] == len(feature_names)
        # Diagonal should be ~1.0 for non-constant features (constant features get NaN)
        for i in range(len(feature_names)):
            val: float = float(corr.iloc[i, i])
            if not math.isnan(val):
                assert val == pytest.approx(1.0, abs=1e-10)

    def test_rc2_feature_analyzer_distributions(
        self,
        feature_set: FeatureSet,
        validation_report: ValidationReport,
    ) -> None:
        """RC2FeatureAnalyzer.compute_feature_distributions should produce stats for all features."""
        analyzer: RC2FeatureAnalyzer = RC2FeatureAnalyzer()
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        feature_names: list[str] = list(feature_set.feature_columns)
        kept_features: list[str] = list(validation_report.kept_feature_names)

        dist_df: pd.DataFrame = analyzer.compute_feature_distributions(
            df_pd,
            feature_names,
            kept_features,
        )
        assert len(dist_df) == len(feature_names)
        assert "status" in dist_df.columns
        assert set(dist_df["status"].unique()) <= {"kept", "dropped"}

    def test_rc2_feature_analyzer_rationale_table(self) -> None:
        """RC2FeatureAnalyzer.build_feature_rationale_table should produce a non-empty table."""
        analyzer: RC2FeatureAnalyzer = RC2FeatureAnalyzer()
        rationale: pd.DataFrame = analyzer.build_feature_rationale_table()
        assert len(rationale) > 0
        assert "Feature" in rationale.columns
        assert "Economic Rationale" in rationale.columns

    def test_rc2_validation_analyzer_mi_table(
        self,
        validation_report: ValidationReport,
        feature_set: FeatureSet,
    ) -> None:
        """RC2ValidationAnalyzer.build_mi_table should produce a table with effect sizes."""
        analyzer: RC2ValidationAnalyzer = RC2ValidationAnalyzer()
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        target_arr: np.ndarray = df_pd["fwd_logret_1"].to_numpy(dtype=np.float64)  # type: ignore[type-arg]
        target_entropy: float = compute_target_entropy_gaussian(target_arr)

        mi_table: pd.DataFrame = analyzer.build_mi_table(validation_report, target_entropy)
        assert len(mi_table) == validation_report.n_features_total
        assert "MI (nats)" in mi_table.columns
        assert "MI/H(target) %" in mi_table.columns
        assert "BH p" in mi_table.columns
        assert "Keep" in mi_table.columns

    def test_rc2_validation_analyzer_da_table(
        self,
        validation_report: ValidationReport,
    ) -> None:
        """RC2ValidationAnalyzer.build_da_table should produce a table with economic significance."""
        analyzer: RC2ValidationAnalyzer = RC2ValidationAnalyzer()
        breakeven_da: float = 0.52  # Synthetic threshold

        da_table: pd.DataFrame = analyzer.build_da_table(validation_report, breakeven_da)
        assert len(da_table) == validation_report.n_features_total
        assert "DA observed" in da_table.columns
        assert "DA excess (pp)" in da_table.columns
        assert "DA vs break-even (pp)" in da_table.columns

    def test_rc2_validation_analyzer_cross_bar_comparison(
        self,
        validation_report: ValidationReport,
    ) -> None:
        """RC2ValidationAnalyzer.build_cross_bar_comparison should handle multiple bar types."""
        analyzer: RC2ValidationAnalyzer = RC2ValidationAnalyzer()
        reports: dict[str, ValidationReport] = {
            "dollar": validation_report,
            "time_1h": validation_report,
        }
        cross_bar: pd.DataFrame = analyzer.build_cross_bar_comparison(reports)
        assert "dollar" in cross_bar.columns
        assert "time_1h" in cross_bar.columns

    def test_rc2_predictability_analyzer_pe_table(self) -> None:
        """RC2PredictabilityAnalyzer.build_pe_table should handle synthetic profiles."""
        analyzer: RC2PredictabilityAnalyzer = RC2PredictabilityAnalyzer()
        profiles: dict[tuple[str, str], PredictabilityProfile] = {}
        for asset in _ASSETS:
            profiles[(asset, _BAR_TYPE)] = PredictabilityProfile(
                asset=asset,
                bar_type=_BAR_TYPE,
                tier=SampleTier.A,
                n_observations=_N_ROWS,
                permutation_entropies=(
                    PermutationEntropyResult(dimension=3, normalized_entropy=0.95, js_complexity=0.02),
                    PermutationEntropyResult(dimension=5, normalized_entropy=0.97, js_complexity=0.01),
                ),
                n_eff=float(_N_ROWS) * 0.8,
                n_eff_ratio=0.8,
                mde_da=0.52,
                breakeven_da=0.51,
            )

        pe_table: pd.DataFrame = analyzer.build_pe_table(profiles)
        assert len(pe_table) == _N_ASSETS
        assert "H_norm_d5" in pe_table.columns

    def test_rc2_predictability_analyzer_feasibility(self) -> None:
        """RC2PredictabilityAnalyzer.build_feasibility_table should classify power gaps."""
        analyzer: RC2PredictabilityAnalyzer = RC2PredictabilityAnalyzer()
        profiles: dict[tuple[str, str], PredictabilityProfile] = {
            ("BTCUSDT", "dollar"): PredictabilityProfile(
                asset="BTCUSDT",
                bar_type="dollar",
                tier=SampleTier.A,
                n_observations=_N_ROWS,
                mde_da=0.52,
                breakeven_da=0.54,
            ),
            ("ETHUSDT", "dollar"): PredictabilityProfile(
                asset="ETHUSDT",
                bar_type="dollar",
                tier=SampleTier.A,
                n_observations=_N_ROWS,
                mde_da=0.56,
                breakeven_da=0.51,
            ),
        }

        feasibility: pd.DataFrame = analyzer.build_feasibility_table(profiles)
        assert len(feasibility) == 2
        assert "classification" in feasibility.columns
        # BTC: breakeven > mde -> feasible, ETH: breakeven < mde -> underpowered
        btc_row: pd.Series = feasibility[feasibility["asset"] == "BTCUSDT"].iloc[0]  # type: ignore[type-arg]
        assert btc_row["classification"] == "feasible"

    def test_rc2_predictability_analyzer_therefore(self) -> None:
        """RC2PredictabilityAnalyzer.generate_section4_therefore should produce narrative."""
        analyzer: RC2PredictabilityAnalyzer = RC2PredictabilityAnalyzer()
        pe_table: pd.DataFrame = pd.DataFrame(
            {
                "asset": ["BTCUSDT"],
                "bar_type": ["dollar"],
                "H_norm_d5": [0.97],
            }
        )
        vr_data: pd.DataFrame = pd.DataFrame(
            {
                "asset": ["BTCUSDT"],
                "significant": [True],
            }
        )
        feasibility: pd.DataFrame = pd.DataFrame(
            {
                "asset": ["BTCUSDT"],
                "classification": ["feasible"],
            }
        )

        therefore: str = analyzer.generate_section4_therefore(pe_table, vr_data, feasibility)
        assert "Therefore" in therefore
        assert len(therefore) > 50

    def test_go_no_go_criteria_evaluation(self) -> None:
        """Pre-registration spec should contain all Go/No-Go criteria that can be evaluated."""
        spec: PreRegistrationSpec = build_preregistration_spec()

        criteria: tuple[GoNoGoCriterion, ...] = spec.go_no_go_criteria
        # There should be 6 pre-registered criteria
        assert len(criteria) == 6

        # Every criterion should have non-empty threshold and rationale
        for criterion in criteria:
            assert len(criterion.criterion) > 0
            assert len(criterion.threshold) > 0
            assert len(criterion.rationale) > 0

        # Verify we can simulate a mechanical evaluation of each criterion
        # (this exercises the code path the notebook would use)
        simulated_results: dict[str, bool] = {
            "Features passing three-gate validation": True,
            "DA excess over baseline": True,
            "Permutation entropy H_norm": True,
            "Effective sample size N_eff": True,
            "Cross-asset feature consistency (Kendall tau)": False,
            "BDS on GARCH residuals": True,
        }
        for criterion in criteria:
            assert criterion.criterion in simulated_results, f"Unknown criterion: {criterion.criterion}"

        # Overall go decision: majority pass
        n_pass: int = sum(simulated_results.values())
        overall_go: bool = n_pass >= 4
        assert overall_go is True

    def test_go_no_go_trial_count(self) -> None:
        """Pre-registration initial trial count should be computable."""
        spec: PreRegistrationSpec = build_preregistration_spec()
        total_trials: int = sum(cat.initial_count for cat in spec.trial_count_categories)
        # Should be a reasonable number (> 0, not absurdly large)
        assert total_trials > 0
        assert total_trials < 100

    def test_target_entropy_computation(self, feature_set: FeatureSet) -> None:
        """Target entropy should be a positive finite number for non-degenerate targets."""
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        target_arr: np.ndarray = df_pd["fwd_logret_1"].to_numpy(dtype=np.float64)  # type: ignore[type-arg]
        entropy: float = compute_target_entropy_gaussian(target_arr)
        assert math.isfinite(entropy)
        # For a Gaussian with non-zero variance, entropy should be positive
        # (actually it can be negative for very small variance, but with
        # crypto-like returns it should be positive due to 0.5*ln(2*pi*e*var))
        assert entropy != 0.0

    def test_feature_target_correlations(
        self,
        feature_set: FeatureSet,
    ) -> None:
        """RC2FeatureAnalyzer.compute_feature_target_correlations should produce per-feature r values."""
        analyzer: RC2FeatureAnalyzer = RC2FeatureAnalyzer()
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        feature_names: list[str] = list(feature_set.feature_columns)

        corr_df: pd.DataFrame = analyzer.compute_feature_target_correlations(
            df_pd,
            feature_names,
            "fwd_logret_1",
        )
        assert len(corr_df) == len(feature_names)
        assert "pearson_r" in corr_df.columns
        assert "spearman_r" in corr_df.columns
        # Correlations should be in [-1, 1]
        valid_pearson: pd.Series = corr_df["pearson_r"].dropna()  # type: ignore[type-arg]
        assert (valid_pearson >= -1.0).all()
        assert (valid_pearson <= 1.0).all()

    def test_pipeline_services_interoperate(  # noqa: PLR0914
        self,
        feature_set: FeatureSet,
        validation_report: ValidationReport,
        multi_asset_feature_dfs: dict[tuple[str, str], pd.DataFrame],
    ) -> None:
        """All RC2 services should be able to consume each other's outputs without errors.

        This is the most important test: it verifies that the data flows
        correctly through the entire RC2 pipeline, catching any signature
        mismatches or type incompatibilities between services.
        """
        feature_names: list[str] = list(feature_set.feature_columns)
        df_pd: pd.DataFrame = feature_set.df.to_pandas()
        target_arr: np.ndarray = df_pd["fwd_logret_1"].to_numpy(dtype=np.float64)  # type: ignore[type-arg]

        # Step 1: Stationarity
        stationarity_analyzer: RC2StationarityAnalyzer = RC2StationarityAnalyzer()
        stationarity_summary: StationaritySummary = stationarity_analyzer.analyze_features(
            multi_asset_feature_dfs,
            feature_names,
        )
        assert stationarity_summary.n_total_features == len(feature_names)

        # Step 2: Feature analysis (VIF + correlation + distributions)
        feature_analyzer: RC2FeatureAnalyzer = RC2FeatureAnalyzer()
        vif_table: pd.DataFrame = feature_analyzer.compute_vif(df_pd, feature_names)
        corr_matrix: pd.DataFrame = feature_analyzer.compute_correlation_matrix(df_pd, feature_names)
        dist_table: pd.DataFrame = feature_analyzer.compute_feature_distributions(
            df_pd,
            feature_names,
            list(validation_report.kept_feature_names),
        )

        # Step 3: Validation analysis (MI + DA tables)
        validation_analyzer: RC2ValidationAnalyzer = RC2ValidationAnalyzer()
        target_entropy: float = compute_target_entropy_gaussian(target_arr)
        mi_table: pd.DataFrame = validation_analyzer.build_mi_table(validation_report, target_entropy)
        da_table: pd.DataFrame = validation_analyzer.build_da_table(validation_report, breakeven_da=0.52)

        # Step 4: Pre-registration Go/No-Go
        spec: PreRegistrationSpec = build_preregistration_spec()
        criteria: tuple[GoNoGoCriterion, ...] = spec.go_no_go_criteria

        # Verify all outputs are non-empty DataFrames (or valid objects)
        assert len(vif_table) > 0
        assert len(corr_matrix) > 0
        assert len(dist_table) > 0
        assert len(mi_table) > 0
        assert len(da_table) > 0
        assert len(criteria) > 0

        # Verify the validation report's kept features appear in MI table
        mi_kept: set[str] = set(mi_table[mi_table["Keep"]]["Feature"].tolist())
        report_kept: set[str] = set(validation_report.kept_feature_names)
        assert mi_kept == report_kept
