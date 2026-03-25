"""Tests for RC2 Section 3 validation analysis service.

Covers MI table building, DA table building, stability heatmap data,
cross-bar-type comparison, multi-horizon comparison, holdout retention,
horizon summary, and the standalone target entropy helper.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.app.features.domain.entities import (
    FeatureValidationResult,
    ValidationReport,
)
from src.app.research.application.rc2_validation_analysis import (
    RC2ValidationAnalyzer,
    compute_target_entropy_gaussian,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_feature_result(
    feature_name: str = "feat_a",
    mi_score: float = 0.05,
    mi_pvalue: float = 0.01,
    fdr_corrected_p: float = 0.02,
    mi_significant: bool = True,
    directional_accuracy: float = 0.55,
    da_null_mean: float = 0.50,
    da_pvalue: float = 0.01,
    da_beats_null: bool = True,
    dc_mae: float = 0.01,
    stability_score: float = 0.75,
    is_stable: bool = True,
    group: str = "returns",
    keep: bool = True,
) -> FeatureValidationResult:
    return FeatureValidationResult(
        feature_name=feature_name,
        mi_score=mi_score,
        mi_pvalue=mi_pvalue,
        fdr_corrected_p=fdr_corrected_p,
        mi_significant=mi_significant,
        directional_accuracy=directional_accuracy,
        da_null_mean=da_null_mean,
        da_pvalue=da_pvalue,
        da_beats_null=da_beats_null,
        dc_mae=dc_mae,
        dc_mae_null_mean=float("nan"),
        stability_score=stability_score,
        is_stable=is_stable,
        group=group,
        keep=keep,
    )


def _make_report(
    features: list[FeatureValidationResult] | None = None,
) -> ValidationReport:
    if features is None:
        features = [
            _make_feature_result("feat_a", mi_score=0.10, directional_accuracy=0.55, keep=True),
            _make_feature_result("feat_b", mi_score=0.05, directional_accuracy=0.52, keep=True),
            _make_feature_result(
                "feat_c",
                mi_score=0.01,
                mi_pvalue=0.30,
                fdr_corrected_p=0.40,
                mi_significant=False,
                directional_accuracy=0.49,
                da_beats_null=False,
                keep=False,
            ),
        ]
    kept = [f for f in features if f.keep]
    dropped = [f for f in features if not f.keep]
    return ValidationReport(
        feature_results=tuple(features),
        interaction_results=(),
        n_features_total=len(features),
        n_features_kept=len(kept),
        n_features_dropped=len(dropped),
        kept_feature_names=tuple(sorted(f.feature_name for f in kept)),
        dropped_feature_names=tuple(sorted(f.feature_name for f in dropped)),
        fallback_triggered=False,
        stability_skipped=False,
    )


@pytest.fixture
def analyzer() -> RC2ValidationAnalyzer:
    return RC2ValidationAnalyzer()


@pytest.fixture
def basic_report() -> ValidationReport:
    return _make_report()


# ---------------------------------------------------------------------------
# MI table tests
# ---------------------------------------------------------------------------


class TestBuildMITable:
    def test_columns_present(self, analyzer: RC2ValidationAnalyzer, basic_report: ValidationReport) -> None:
        df = analyzer.build_mi_table(basic_report, target_entropy=1.0)
        expected_cols = {"Feature", "Group", "MI (nats)", "Raw p", "BH p", "Significant", "MI/H(target) %", "Keep"}
        assert set(df.columns) == expected_cols

    def test_sorted_by_mi_descending(self, analyzer: RC2ValidationAnalyzer, basic_report: ValidationReport) -> None:
        df = analyzer.build_mi_table(basic_report, target_entropy=1.0)
        mi_values = df["MI (nats)"].tolist()
        assert mi_values == sorted(mi_values, reverse=True)

    def test_effect_size_computation(self, analyzer: RC2ValidationAnalyzer) -> None:
        report = _make_report([_make_feature_result("f1", mi_score=0.10)])
        df = analyzer.build_mi_table(report, target_entropy=2.0)
        expected_pct = (0.10 / 2.0) * 100.0
        assert abs(df.iloc[0]["MI/H(target) %"] - round(expected_pct, 3)) < 1e-6

    def test_zero_entropy_no_crash(self, analyzer: RC2ValidationAnalyzer) -> None:
        report = _make_report([_make_feature_result("f1", mi_score=0.10)])
        df = analyzer.build_mi_table(report, target_entropy=0.0)
        # Should not divide by zero; effect size will be very large but not inf
        assert len(df) == 1
        assert df.iloc[0]["MI/H(target) %"] > 0

    def test_row_count_matches_features(self, analyzer: RC2ValidationAnalyzer, basic_report: ValidationReport) -> None:
        df = analyzer.build_mi_table(basic_report, target_entropy=1.0)
        assert len(df) == basic_report.n_features_total

    def test_single_feature(self, analyzer: RC2ValidationAnalyzer) -> None:
        report = _make_report([_make_feature_result("only_one")])
        df = analyzer.build_mi_table(report, target_entropy=1.0)
        assert len(df) == 1
        assert df.iloc[0]["Feature"] == "only_one"


# ---------------------------------------------------------------------------
# DA table tests
# ---------------------------------------------------------------------------


class TestBuildDATable:
    def test_columns_present(self, analyzer: RC2ValidationAnalyzer, basic_report: ValidationReport) -> None:
        df = analyzer.build_da_table(basic_report, breakeven_da=0.52)
        expected_cols = {
            "Feature",
            "Group",
            "DA observed",
            "DA null",
            "DA excess (pp)",
            "DA vs break-even (pp)",
            "p",
            "Beats null",
            "Keep",
        }
        assert set(df.columns) == expected_cols

    def test_da_excess_computation(self, analyzer: RC2ValidationAnalyzer) -> None:
        report = _make_report([_make_feature_result("f1", directional_accuracy=0.55)])
        df = analyzer.build_da_table(report, breakeven_da=0.52)
        assert abs(df.iloc[0]["DA excess (pp)"] - 5.0) < 0.01  # (0.55 - 0.50) * 100
        assert abs(df.iloc[0]["DA vs break-even (pp)"] - 3.0) < 0.01  # (0.55 - 0.52) * 100

    def test_negative_da_vs_breakeven(self, analyzer: RC2ValidationAnalyzer) -> None:
        report = _make_report([_make_feature_result("f1", directional_accuracy=0.50)])
        df = analyzer.build_da_table(report, breakeven_da=0.52)
        assert df.iloc[0]["DA vs break-even (pp)"] < 0

    def test_sorted_by_excess_descending(
        self, analyzer: RC2ValidationAnalyzer, basic_report: ValidationReport
    ) -> None:
        df = analyzer.build_da_table(basic_report, breakeven_da=0.52)
        excess_values = df["DA excess (pp)"].tolist()
        assert excess_values == sorted(excess_values, reverse=True)


# ---------------------------------------------------------------------------
# Stability heatmap tests
# ---------------------------------------------------------------------------


class TestBuildStabilityHeatmapData:
    def test_basic_shape(self, analyzer: RC2ValidationAnalyzer) -> None:
        r1 = _make_report(
            [
                _make_feature_result("f1", mi_significant=True),
                _make_feature_result("f2", mi_significant=False),
            ]
        )
        r2 = _make_report(
            [
                _make_feature_result("f1", mi_significant=False),
                _make_feature_result("f2", mi_significant=True),
            ]
        )
        df = analyzer.build_stability_heatmap_data({"2020-2021": r1, "2021-2022": r2})
        assert df.shape == (2, 2)
        assert list(df.columns) == ["2020-2021", "2021-2022"]

    def test_values_are_binary(self, analyzer: RC2ValidationAnalyzer) -> None:
        report = _make_report([_make_feature_result("f1", mi_significant=True)])
        df = analyzer.build_stability_heatmap_data({"w1": report})
        assert set(df.values.flatten().tolist()).issubset({0, 1})

    def test_empty_input(self, analyzer: RC2ValidationAnalyzer) -> None:
        df = analyzer.build_stability_heatmap_data({})
        assert df.empty

    def test_correct_values(self, analyzer: RC2ValidationAnalyzer) -> None:
        report = _make_report(
            [
                _make_feature_result("f1", mi_significant=True),
                _make_feature_result("f2", mi_significant=False),
            ]
        )
        df = analyzer.build_stability_heatmap_data({"w1": report})
        assert df.loc["f1", "w1"] == 1
        assert df.loc["f2", "w1"] == 0


# ---------------------------------------------------------------------------
# Cross-bar comparison tests
# ---------------------------------------------------------------------------


class TestBuildCrossBarComparison:
    def test_basic_shape(self, analyzer: RC2ValidationAnalyzer) -> None:
        dollar_report = _make_report([_make_feature_result("f1", mi_score=0.10)])
        volume_report = _make_report([_make_feature_result("f1", mi_score=0.08)])
        df = analyzer.build_cross_bar_comparison({"dollar": dollar_report, "volume": volume_report})
        assert df.shape == (1, 2)
        assert list(df.columns) == ["dollar", "volume"]

    def test_missing_feature_gets_nan(self, analyzer: RC2ValidationAnalyzer) -> None:
        dollar_report = _make_report(
            [
                _make_feature_result("f1", mi_score=0.10),
                _make_feature_result("f2", mi_score=0.05),
            ]
        )
        volume_report = _make_report([_make_feature_result("f1", mi_score=0.08)])
        df = analyzer.build_cross_bar_comparison({"dollar": dollar_report, "volume": volume_report})
        assert df.shape == (2, 2)
        assert pd.isna(df.loc["f2", "volume"])

    def test_empty_input(self, analyzer: RC2ValidationAnalyzer) -> None:
        df = analyzer.build_cross_bar_comparison({})
        assert df.empty


# ---------------------------------------------------------------------------
# Multi-horizon comparison tests
# ---------------------------------------------------------------------------


class TestBuildMultiHorizonComparison:
    def test_basic_shape(self, analyzer: RC2ValidationAnalyzer) -> None:
        r1 = _make_report([_make_feature_result("f1"), _make_feature_result("f2")])
        r4 = _make_report([_make_feature_result("f1"), _make_feature_result("f2")])
        df = analyzer.build_multi_horizon_comparison({"fwd_logret_1": r1, "fwd_logret_4": r4})
        assert len(df) == 2  # 2 features
        assert isinstance(df.columns, pd.MultiIndex)
        # Two horizons x 5 metrics = 10 columns
        assert len(df.columns) == 10

    def test_empty_input(self, analyzer: RC2ValidationAnalyzer) -> None:
        df = analyzer.build_multi_horizon_comparison({})
        assert df.empty

    def test_horizon_labels_in_columns(self, analyzer: RC2ValidationAnalyzer) -> None:
        r1 = _make_report([_make_feature_result("f1")])
        df = analyzer.build_multi_horizon_comparison({"fwd_logret_1": r1})
        horizons = df.columns.get_level_values(0).unique().tolist()
        assert "fwd_logret_1" in horizons


# ---------------------------------------------------------------------------
# Holdout retention tests
# ---------------------------------------------------------------------------


class TestComputeHoldoutRetention:
    def test_full_retention(self, analyzer: RC2ValidationAnalyzer) -> None:
        train = _make_report([_make_feature_result("f1", keep=True)])
        holdout = _make_report([_make_feature_result("f1", keep=True)])
        df = analyzer.compute_holdout_retention(train, holdout)
        assert len(df) == 1
        assert bool(df.iloc[0]["Retained"]) is True

    def test_lost_in_holdout(self, analyzer: RC2ValidationAnalyzer) -> None:
        train = _make_report([_make_feature_result("f1", keep=True)])
        holdout = _make_report([_make_feature_result("f1", keep=False)])
        df = analyzer.compute_holdout_retention(train, holdout)
        assert bool(df.iloc[0]["Retained"]) is False

    def test_not_kept_in_train(self, analyzer: RC2ValidationAnalyzer) -> None:
        train = _make_report([_make_feature_result("f1", keep=False)])
        holdout = _make_report([_make_feature_result("f1", keep=True)])
        df = analyzer.compute_holdout_retention(train, holdout)
        assert bool(df.iloc[0]["Retained"]) is False

    def test_columns_present(self, analyzer: RC2ValidationAnalyzer) -> None:
        train = _make_report([_make_feature_result("f1")])
        holdout = _make_report([_make_feature_result("f1")])
        df = analyzer.compute_holdout_retention(train, holdout)
        expected_cols = {
            "Feature",
            "Train MI sig",
            "Holdout MI sig",
            "Train DA beats",
            "Holdout DA beats",
            "Train Keep",
            "Holdout Keep",
            "Retained",
        }
        assert set(df.columns) == expected_cols


# ---------------------------------------------------------------------------
# Horizon summary tests
# ---------------------------------------------------------------------------


class TestComputeHorizonSummary:
    def test_basic(self, analyzer: RC2ValidationAnalyzer) -> None:
        r1 = _make_report()
        df = analyzer.compute_horizon_summary({"fwd_logret_1": r1})
        assert len(df) == 1
        assert df.iloc[0]["Horizon"] == "fwd_logret_1"
        assert df.iloc[0]["N features"] == 3

    def test_multiple_horizons(self, analyzer: RC2ValidationAnalyzer) -> None:
        r1 = _make_report()
        r4 = _make_report()
        df = analyzer.compute_horizon_summary({"fwd_logret_1": r1, "fwd_logret_4": r4})
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Target entropy tests
# ---------------------------------------------------------------------------


class TestComputeTargetEntropyGaussian:
    def test_standard_normal(self) -> None:
        # H(N(0,1)) = 0.5 * log(2*pi*e) ~= 1.4189
        expected = 0.5 * math.log(2 * math.pi * math.e)
        result = compute_target_entropy_gaussian(np.random.default_rng(42).normal(0, 1, 100000))
        assert abs(result - expected) < 0.05  # sample variance close to 1

    def test_zero_variance(self) -> None:
        result = compute_target_entropy_gaussian(np.ones(100))
        assert result == 0.0

    def test_positive_result(self) -> None:
        result = compute_target_entropy_gaussian(np.random.default_rng(42).normal(0, 0.5, 1000))
        assert result > 0.0
