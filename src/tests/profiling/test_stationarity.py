"""Tests for StationarityScreener against known synthetic data.

Uses white noise (stationary) and random walk (unit root) series to
verify correct classification by the joint ADF + KPSS test.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.app.profiling.application.stationarity import StationarityScreener
from src.app.profiling.domain.value_objects import (
    StationarityReport,
    StationarityTestResult,
)

from src.tests.profiling.conftest import (
    make_stationarity_test_df,
    make_stationary_series,
    make_unit_root_series,
)


class TestStationarityTestResult:
    """Tests for StationarityTestResult construction."""

    def test_valid_result_construction(self) -> None:
        """A valid StationarityTestResult should be constructable."""
        result = StationarityTestResult(
            feature_name="test_feat",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            kpss_statistic=0.2,
            kpss_pvalue=0.1,
            is_stationary=True,
            classification="stationary",
            suggested_transformation=None,
        )
        assert result.feature_name == "test_feat"
        assert result.is_stationary is True

    def test_result_is_frozen(self) -> None:
        """StationarityTestResult must be immutable."""
        from pydantic import ValidationError

        result = StationarityTestResult(
            feature_name="test_feat",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            kpss_statistic=0.2,
            kpss_pvalue=0.1,
            is_stationary=True,
            classification="stationary",
            suggested_transformation=None,
        )
        with pytest.raises(ValidationError):
            result.is_stationary = False  # type: ignore[misc]


class TestStationarityReport:
    """Tests for StationarityReport construction."""

    def test_report_construction(self) -> None:
        """A valid StationarityReport should be constructable."""
        result = StationarityTestResult(
            feature_name="feat1",
            adf_statistic=-4.0,
            adf_pvalue=0.001,
            kpss_statistic=0.1,
            kpss_pvalue=0.5,
            is_stationary=True,
            classification="stationary",
            suggested_transformation=None,
        )
        report = StationarityReport(
            results=(result,),
            n_stationary=1,
            n_non_stationary=0,
            asset="BTCUSDT",
            bar_type="dollar",
        )
        assert report.n_stationary == 1
        assert len(report.results) == 1


class TestStationarityScreener:
    """Integration tests for StationarityScreener.screen."""

    def test_white_noise_classified_as_stationary(self) -> None:
        """White noise series should be classified as stationary."""
        series = make_stationary_series(1000, seed=42)
        df = pd.DataFrame({"white_noise": series})
        screener = StationarityScreener()
        report = screener.screen(
            df,
            feature_names=["white_noise"],
            asset="BTCUSDT",
            bar_type="dollar",
        )
        assert report.n_stationary == 1
        assert report.results[0].classification == "stationary"
        assert report.results[0].is_stationary is True

    def test_random_walk_classified_as_non_stationary(self) -> None:
        """Random walk series should be classified as unit_root."""
        series = make_unit_root_series(1000, seed=42)
        df = pd.DataFrame({"random_walk": series})
        screener = StationarityScreener()
        report = screener.screen(
            df,
            feature_names=["random_walk"],
            asset="BTCUSDT",
            bar_type="dollar",
        )
        assert report.results[0].is_stationary is False
        assert report.results[0].classification in {"unit_root", "trend_stationary"}

    def test_mixed_features_counted_correctly(self) -> None:
        """Report counts should match the number of stationary/non-stationary features."""
        df, feature_names = make_stationarity_test_df(1000, seed=42)
        screener = StationarityScreener()
        report = screener.screen(
            df,
            feature_names=feature_names,
            asset="ETHUSDT",
            bar_type="volume",
        )
        assert report.n_stationary + report.n_non_stationary == len(feature_names)
        assert len(report.results) == len(feature_names)

    def test_report_metadata(self) -> None:
        """Report should carry asset and bar_type metadata."""
        df, feature_names = make_stationarity_test_df(500, seed=10)
        screener = StationarityScreener()
        report = screener.screen(
            df,
            feature_names=feature_names,
            asset="SOLUSDT",
            bar_type="dollar_imbalance",
        )
        assert report.asset == "SOLUSDT"
        assert report.bar_type == "dollar_imbalance"

    def test_nan_in_feature_raises_value_error(self) -> None:
        """NaN in a feature column should raise ValueError."""
        df = pd.DataFrame({"bad_feat": [1.0, 2.0, float("nan"), 4.0]})
        screener = StationarityScreener()
        with pytest.raises(ValueError, match="NaN or inf"):
            screener.screen(df, feature_names=["bad_feat"], asset="X", bar_type="Y")

    def test_inf_in_feature_raises_value_error(self) -> None:
        """Inf in a feature column should raise ValueError."""
        df = pd.DataFrame({"bad_feat": [1.0, 2.0, float("inf"), 4.0]})
        screener = StationarityScreener()
        with pytest.raises(ValueError, match="NaN or inf"):
            screener.screen(df, feature_names=["bad_feat"], asset="X", bar_type="Y")

    def test_suggested_transformation_atr_prefix(self) -> None:
        """Non-stationary features with 'atr_' prefix should suggest 'pct_atr'."""
        series = make_unit_root_series(1000, seed=99)
        df = pd.DataFrame({"atr_14": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["atr_14"], asset="X", bar_type="Y")
        if not report.results[0].is_stationary:
            assert report.results[0].suggested_transformation == "pct_atr"

    def test_suggested_transformation_amihud_prefix(self) -> None:
        """Non-stationary features with 'amihud_' prefix should suggest 'rolling_zscore'."""
        series = make_unit_root_series(1000, seed=100)
        df = pd.DataFrame({"amihud_24": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["amihud_24"], asset="X", bar_type="Y")
        if not report.results[0].is_stationary:
            assert report.results[0].suggested_transformation == "rolling_zscore"

    def test_suggested_transformation_hurst_prefix(self) -> None:
        """Non-stationary features with 'hurst_' prefix should suggest 'first_difference'."""
        series = make_unit_root_series(1000, seed=101)
        df = pd.DataFrame({"hurst_100": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["hurst_100"], asset="X", bar_type="Y")
        if not report.results[0].is_stationary:
            assert report.results[0].suggested_transformation == "first_difference"

    def test_no_transformation_for_stationary_feature(self) -> None:
        """Stationary features should have no suggested transformation."""
        series = make_stationary_series(1000, seed=200)
        df = pd.DataFrame({"stationary": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["stationary"], asset="X", bar_type="Y")
        if report.results[0].is_stationary:
            assert report.results[0].suggested_transformation is None

    def test_no_transformation_for_unknown_prefix(self) -> None:
        """Non-stationary features with unknown prefix should have no suggestion."""
        series = make_unit_root_series(1000, seed=300)
        df = pd.DataFrame({"unknown_feat": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["unknown_feat"], asset="X", bar_type="Y")
        if not report.results[0].is_stationary:
            assert report.results[0].suggested_transformation is None

    def test_adf_pvalue_in_range(self) -> None:
        """ADF p-value should always be in [0, 1]."""
        series = make_stationary_series(500, seed=42)
        df = pd.DataFrame({"feat": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["feat"], asset="X", bar_type="Y")
        assert 0.0 <= report.results[0].adf_pvalue <= 1.0

    def test_kpss_pvalue_in_range(self) -> None:
        """KPSS p-value should always be in [0, 1] (clamped)."""
        series = make_stationary_series(500, seed=42)
        df = pd.DataFrame({"feat": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["feat"], asset="X", bar_type="Y")
        assert 0.0 <= report.results[0].kpss_pvalue <= 1.0

    def test_custom_alpha(self) -> None:
        """Custom alpha should be respected without errors."""
        df, feature_names = make_stationarity_test_df(500, seed=42)
        screener = StationarityScreener()
        report = screener.screen(
            df,
            feature_names=feature_names,
            asset="BTCUSDT",
            bar_type="dollar",
            alpha=0.01,
        )
        assert len(report.results) == 2


# ---------------------------------------------------------------------------
# Explicit ADF and KPSS p-value tests (spec requirement)
# ---------------------------------------------------------------------------


class TestADFAndKPSSPValues:
    """Tests for individual ADF and KPSS p-value fields on canonical series.

    These tests verify the spec requirement that the screener correctly
    distinguishes between ADF and KPSS decisions on known series.
    """

    def test_random_walk_adf_fails_to_reject(self) -> None:
        """Random walk: ADF should fail to reject the unit root null (large p-value)."""
        series = make_unit_root_series(n=1000, seed=42)
        df = pd.DataFrame({"rw": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["rw"], asset="BTCUSDT", bar_type="dollar")

        result = report.results[0]
        # ADF null is unit root; fail to reject => large p-value (> 0.05)
        assert result.adf_pvalue > 0.05, f"ADF p-value={result.adf_pvalue:.4f} should be > 0.05 for a random walk"

    def test_white_noise_adf_rejects_unit_root(self) -> None:
        """White noise: ADF should reject the unit root null (small p-value)."""
        series = make_stationary_series(n=1000, seed=42)
        df = pd.DataFrame({"wn": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["wn"], asset="BTCUSDT", bar_type="dollar")

        result = report.results[0]
        # ADF null is unit root; reject => small p-value (< 0.05)
        assert result.adf_pvalue < 0.05, f"ADF p-value={result.adf_pvalue:.4f} should be < 0.05 for white noise"

    def test_white_noise_kpss_does_not_reject_stationarity(self) -> None:
        """White noise: KPSS should not reject the stationarity null (large p-value)."""
        series = make_stationary_series(n=1000, seed=42)
        df = pd.DataFrame({"wn": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["wn"], asset="BTCUSDT", bar_type="dollar")

        result = report.results[0]
        # KPSS null is stationarity; fail to reject => large p-value (> 0.05)
        assert result.kpss_pvalue > 0.05, f"KPSS p-value={result.kpss_pvalue:.4f} should be > 0.05 for white noise"

    def test_random_walk_kpss_rejects_stationarity(self) -> None:
        """Random walk: KPSS should reject the stationarity null (small p-value)."""
        series = make_unit_root_series(n=1000, seed=42)
        df = pd.DataFrame({"rw": series})
        screener = StationarityScreener()
        report = screener.screen(df, feature_names=["rw"], asset="BTCUSDT", bar_type="dollar")

        result = report.results[0]
        # KPSS null is stationarity; reject => small p-value (< 0.05)
        assert result.kpss_pvalue < 0.05, f"KPSS p-value={result.kpss_pvalue:.4f} should be < 0.05 for a random walk"
