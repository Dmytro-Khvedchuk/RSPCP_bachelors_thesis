"""Tests for RC2 Section 4 predictability analysis service.

Covers PE table building, complexity-entropy data extraction, R5 comparison,
VR profile data, N_eff table, feasibility table, Therefore paragraph
generation, and edge cases (empty profiles, missing PE data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.app.profiling.domain.value_objects import (
    AutocorrelationProfile,
    LjungBoxResult,
    PermutationEntropyResult,
    PredictabilityProfile,
    SampleTier,
    VarianceRatioResult,
)
from src.app.research.application.rc2_predictability import (
    RC2PredictabilityAnalyzer,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_pe_results(
    dimensions: tuple[int, ...] = (3, 4, 5, 6),
    h_norm_base: float = 0.95,
    c_base: float = 0.05,
) -> tuple[PermutationEntropyResult, ...]:
    """Create mock permutation entropy results for multiple dimensions."""
    return tuple(
        PermutationEntropyResult(
            dimension=d,
            normalized_entropy=min(h_norm_base + d * 0.01, 1.0),
            js_complexity=max(c_base - d * 0.005, 0.0),
        )
        for d in dimensions
    )


def _make_predictability_profile(
    asset: str = "BTCUSDT",
    bar_type: str = "dollar",
    tier: SampleTier = SampleTier.A,
    n_obs: int = 5000,
    pe_results: tuple[PermutationEntropyResult, ...] | None = None,
    n_eff: float | None = 3000.0,
    n_eff_ratio: float | None = 0.60,
    mde_da: float | None = 0.523,
    breakeven_da: float | None = 0.625,
) -> PredictabilityProfile:
    """Create a mock PredictabilityProfile with sensible defaults."""
    if pe_results is None:
        pe_results = _make_pe_results()
    return PredictabilityProfile(
        asset=asset,
        bar_type=bar_type,
        tier=tier,
        n_observations=n_obs,
        permutation_entropies=pe_results,
        n_eff=n_eff,
        n_eff_ratio=n_eff_ratio,
        mde_da=mde_da,
        breakeven_da=breakeven_da,
    )


def _make_vr_result(
    horizon_days: float = 1.0,
    q: int = 24,
    vr: float = 0.95,
    z: float = -2.1,
    p: float = 0.036,
    sig: bool = True,
) -> VarianceRatioResult:
    """Create a mock VarianceRatioResult."""
    return VarianceRatioResult(
        calendar_horizon_days=horizon_days,
        bar_count_q=q,
        variance_ratio=vr,
        z_statistic=z,
        p_value=p,
        significant=sig,
    )


def _make_autocorrelation_profile(
    asset: str = "BTCUSDT",
    bar_type: str = "dollar",
    vr_results: tuple[VarianceRatioResult, ...] | None = None,
) -> AutocorrelationProfile:
    """Create a mock AutocorrelationProfile with VR results."""
    n = 100
    if vr_results is None:
        vr_results = (
            _make_vr_result(1.0, 24, 0.95, -2.1, 0.036, True),
            _make_vr_result(7.0, 168, 1.03, 1.5, 0.134, False),
        )
    return AutocorrelationProfile(
        asset=asset,
        bar_type=bar_type,
        tier=SampleTier.A,
        n_observations=5000,
        acf_values=np.zeros(n),
        pacf_values=np.zeros(n),
        acf_squared_values=np.zeros(n),
        pacf_squared_values=np.zeros(n),
        ljung_box_returns=(LjungBoxResult(lag=5, q_statistic=10.0, p_value=0.05, significant=True),),
        ljung_box_squared=(LjungBoxResult(lag=5, q_statistic=50.0, p_value=0.001, significant=True),),
        has_serial_correlation=True,
        has_volatility_clustering=True,
        vr_results=vr_results,
    )


@pytest.fixture
def analyzer() -> RC2PredictabilityAnalyzer:
    """Return a fresh analyzer instance."""
    return RC2PredictabilityAnalyzer()


@pytest.fixture
def pred_profiles() -> dict[tuple[str, str], PredictabilityProfile]:
    """Return a standard set of predictability profiles for testing."""
    return {
        ("BTCUSDT", "dollar"): _make_predictability_profile("BTCUSDT", "dollar"),
        ("BTCUSDT", "volume"): _make_predictability_profile("BTCUSDT", "volume", n_obs=3000, n_eff=1800.0),
        ("ETHUSDT", "dollar"): _make_predictability_profile("ETHUSDT", "dollar", n_obs=4500, n_eff=2700.0),
    }


@pytest.fixture
def acf_profiles() -> dict[tuple[str, str], AutocorrelationProfile]:
    """Return a standard set of autocorrelation profiles for testing."""
    return {
        ("BTCUSDT", "dollar"): _make_autocorrelation_profile("BTCUSDT", "dollar"),
        ("ETHUSDT", "dollar"): _make_autocorrelation_profile("ETHUSDT", "dollar"),
    }


# ---------------------------------------------------------------------------
# TestBuildPETable
# ---------------------------------------------------------------------------


class TestBuildPETable:
    """Tests for the PE table builder."""

    def test_shape_and_columns(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
    ) -> None:
        """PE table should have one row per profile and H_norm/C columns per dimension."""
        df = analyzer.build_pe_table(pred_profiles)
        assert len(df) == 3
        assert "asset" in df.columns
        assert "bar_type" in df.columns
        assert "H_norm_d5" in df.columns
        assert "C_d5" in df.columns

    def test_sorted_output(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
    ) -> None:
        """Table should be sorted by (asset, bar_type)."""
        df = analyzer.build_pe_table(pred_profiles)
        assets = df["asset"].tolist()
        bar_types = df["bar_type"].tolist()
        # BTCUSDT should come before ETHUSDT
        assert assets[0] == "BTCUSDT"
        assert assets[-1] == "ETHUSDT"
        # Within BTCUSDT, dollar should come before volume
        btc_bars = [bt for a, bt in zip(assets, bar_types, strict=True) if a == "BTCUSDT"]
        assert btc_bars == sorted(btc_bars)

    def test_empty_profiles(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Empty profiles dict should return empty DataFrame."""
        df = analyzer.build_pe_table({})
        assert df.empty

    def test_profile_without_pe(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Profile with None permutation_entropies still produces a row."""
        # Tier C profiles have None PE results
        tier_c_profile = PredictabilityProfile(
            asset="BTCUSDT",
            bar_type="dollar",
            tier=SampleTier.C,
            n_observations=100,
            permutation_entropies=None,
        )
        df = analyzer.build_pe_table({("BTCUSDT", "dollar"): tier_c_profile})
        assert len(df) == 1
        assert "H_norm_d5" not in df.columns


# ---------------------------------------------------------------------------
# TestBuildComplexityEntropyData
# ---------------------------------------------------------------------------


class TestBuildComplexityEntropyData:
    """Tests for the complexity-entropy plane data builder."""

    def test_correct_dimension_filtering(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
    ) -> None:
        """Should only include rows for the specified dimension."""
        df = analyzer.build_complexity_entropy_data(pred_profiles, dimension=5)
        assert len(df) == 3
        assert set(df.columns) == {"asset", "bar_type", "H_norm", "C"}

    def test_non_existent_dimension(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
    ) -> None:
        """Dimension not present in profiles should return empty DataFrame."""
        df = analyzer.build_complexity_entropy_data(pred_profiles, dimension=99)
        assert df.empty

    def test_empty_profiles(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Empty profiles should return empty DataFrame."""
        df = analyzer.build_complexity_entropy_data({}, dimension=5)
        assert df.empty


# ---------------------------------------------------------------------------
# TestCompareWithR5
# ---------------------------------------------------------------------------


class TestCompareWithR5:
    """Tests for R5 comparison logic."""

    def test_known_assets_have_r5_reference(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
    ) -> None:
        """BTC and ETH should have R5 reference values."""
        results = analyzer.compare_with_r5(pred_profiles, dimension=5)
        btc_results = [r for r in results if r.asset == "BTCUSDT"]
        assert len(btc_results) >= 1
        for r in btc_results:
            assert r.r5_h_norm is not None
            assert r.delta_h is not None
            assert r.is_below_r5 is not None

    def test_unknown_asset_no_r5(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """SOL is not in R5; fields should be None."""
        profiles = {
            ("SOLUSDT", "dollar"): _make_predictability_profile("SOLUSDT", "dollar"),
        }
        results = analyzer.compare_with_r5(profiles, dimension=5)
        assert len(results) == 1
        assert results[0].r5_h_norm is None
        assert results[0].delta_h is None
        assert results[0].is_below_r5 is None

    def test_delta_computation(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Delta should be our_h_norm - r5_h_norm."""
        pe = _make_pe_results(dimensions=(5,), h_norm_base=0.97, c_base=0.03)
        profiles = {
            ("BTCUSDT", "dollar"): _make_predictability_profile("BTCUSDT", "dollar", pe_results=pe),
        }
        results = analyzer.compare_with_r5(profiles, dimension=5)
        assert len(results) == 1
        expected_h = 0.97 + 5 * 0.01  # = 1.0 (clamped)
        expected_h = min(expected_h, 1.0)
        expected_delta = expected_h - 0.985
        assert results[0].delta_h == pytest.approx(expected_delta, abs=1e-10)

    def test_is_below_r5_flag(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """is_below_r5 should be True when our H_norm < R5 reference."""
        low_pe = (PermutationEntropyResult(dimension=5, normalized_entropy=0.970, js_complexity=0.03),)
        profiles = {
            ("BTCUSDT", "dollar"): _make_predictability_profile("BTCUSDT", "dollar", pe_results=low_pe),
        }
        results = analyzer.compare_with_r5(profiles, dimension=5)
        assert results[0].is_below_r5 is True

    def test_skip_missing_pe_dimension(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Profile without the specified dimension is skipped."""
        pe = _make_pe_results(dimensions=(3, 4))  # no d=5
        profiles = {
            ("BTCUSDT", "dollar"): _make_predictability_profile(
                pe_results=pe,
            ),
        }
        results = analyzer.compare_with_r5(profiles, dimension=5)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# TestBuildVRProfileData
# ---------------------------------------------------------------------------


class TestBuildVRProfileData:
    """Tests for VR profile data extraction."""

    def test_extracts_vr_results(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        acf_profiles: dict[tuple[str, str], AutocorrelationProfile],
    ) -> None:
        """Should extract all VR results from profiles."""
        df = analyzer.build_vr_profile_data(acf_profiles)
        # 2 profiles x 2 VR results each = 4 rows
        assert len(df) == 4
        expected_cols = {
            "asset",
            "bar_type",
            "horizon_days",
            "bar_count_q",
            "vr",
            "z_stat",
            "p_value",
            "significant",
        }
        assert set(df.columns) == expected_cols

    def test_no_vr_results(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Profiles without VR results produce empty DataFrame."""
        prof_no_vr = AutocorrelationProfile(
            asset="BTCUSDT",
            bar_type="dollar",
            tier=SampleTier.C,
            n_observations=100,
            acf_values=np.zeros(10),
            pacf_values=np.zeros(10),
            acf_squared_values=np.zeros(10),
            pacf_squared_values=np.zeros(10),
            ljung_box_returns=(),
            ljung_box_squared=(),
            has_serial_correlation=False,
            has_volatility_clustering=False,
            vr_results=None,
        )
        df = analyzer.build_vr_profile_data({("BTCUSDT", "dollar"): prof_no_vr})
        assert df.empty

    def test_significant_flag_preserved(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Significant flag should match original VR result."""
        vr_sig = _make_vr_result(1.0, 24, 0.95, -2.5, 0.01, True)
        vr_not = _make_vr_result(7.0, 168, 1.01, 0.5, 0.60, False)
        prof = _make_autocorrelation_profile(vr_results=(vr_sig, vr_not))
        df = analyzer.build_vr_profile_data({("BTCUSDT", "dollar"): prof})
        sig_rows = df[df["significant"]]
        not_sig_rows = df[~df["significant"]]
        assert len(sig_rows) == 1
        assert len(not_sig_rows) == 1


# ---------------------------------------------------------------------------
# TestBuildNEffTable
# ---------------------------------------------------------------------------


class TestBuildNEffTable:
    """Tests for the N_eff summary table."""

    def test_all_fields_populated(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
    ) -> None:
        """All expected columns should be present."""
        df = analyzer.build_neff_table(pred_profiles)
        expected_cols = {"asset", "bar_type", "n_obs", "n_eff", "n_eff_ratio", "mde_da", "breakeven_da", "tier"}
        assert expected_cols == set(df.columns)
        assert len(df) == 3

    def test_sorted_output(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
    ) -> None:
        """Table should be sorted by (asset, bar_type)."""
        df = analyzer.build_neff_table(pred_profiles)
        assert df.iloc[0]["asset"] == "BTCUSDT"
        assert df.iloc[-1]["asset"] == "ETHUSDT"

    def test_empty_profiles(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Empty profiles dict should return empty DataFrame."""
        df = analyzer.build_neff_table({})
        assert df.empty


# ---------------------------------------------------------------------------
# TestBuildFeasibilityTable
# ---------------------------------------------------------------------------


class TestBuildFeasibilityTable:
    """Tests for the feasibility gap table."""

    def test_gap_computation(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Gap should be (breakeven - mde) * 100 in percentage points."""
        profiles = {
            ("BTCUSDT", "dollar"): _make_predictability_profile(
                mde_da=0.523,
                breakeven_da=0.625,
            ),
        }
        df = analyzer.build_feasibility_table(profiles)
        assert len(df) == 1
        expected_gap = (0.625 - 0.523) * 100
        assert df.iloc[0]["gap_pp"] == pytest.approx(expected_gap, abs=0.01)
        assert df.iloc[0]["classification"] == "feasible"

    def test_marginal_classification(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Near-zero gap should classify as marginal."""
        profiles = {
            ("BTCUSDT", "dollar"): _make_predictability_profile(
                mde_da=0.560,
                breakeven_da=0.555,
            ),
        }
        df = analyzer.build_feasibility_table(profiles)
        assert df.iloc[0]["classification"] == "marginal"

    def test_underpowered_classification(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Large negative gap should classify as underpowered."""
        profiles = {
            ("BTCUSDT", "dollar"): _make_predictability_profile(
                mde_da=0.580,
                breakeven_da=0.520,
            ),
        }
        df = analyzer.build_feasibility_table(profiles)
        assert df.iloc[0]["classification"] == "underpowered"

    def test_skips_none_values(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Profiles with None mde_da or breakeven_da are skipped."""
        tier_c = PredictabilityProfile(
            asset="BTCUSDT",
            bar_type="dollar",
            tier=SampleTier.C,
            n_observations=50,
            mde_da=None,
            breakeven_da=None,
        )
        df = analyzer.build_feasibility_table({("BTCUSDT", "dollar"): tier_c})
        assert df.empty


# ---------------------------------------------------------------------------
# TestGenerateSection4Therefore
# ---------------------------------------------------------------------------


class TestGenerateSection4Therefore:
    """Tests for the Therefore paragraph generator."""

    def test_contains_key_phrases(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
        acf_profiles: dict[tuple[str, str], AutocorrelationProfile],
    ) -> None:
        """Therefore should contain PE, VR, and R5 references."""
        pe_table = analyzer.build_pe_table(pred_profiles)
        vr_data = analyzer.build_vr_profile_data(acf_profiles)
        feasibility = analyzer.build_feasibility_table(pred_profiles)
        text = analyzer.generate_section4_therefore(pe_table, vr_data, feasibility)
        assert "**Therefore:**" in text
        assert "Permutation entropy" in text
        assert "Variance ratio" in text
        assert "R5" in text
        assert "Brownian" in text

    def test_includes_feasibility_count(
        self,
        analyzer: RC2PredictabilityAnalyzer,
        pred_profiles: dict[tuple[str, str], PredictabilityProfile],
        acf_profiles: dict[tuple[str, str], AutocorrelationProfile],
    ) -> None:
        """Therefore should report the number of feasible combinations."""
        pe_table = analyzer.build_pe_table(pred_profiles)
        vr_data = analyzer.build_vr_profile_data(acf_profiles)
        feasibility = analyzer.build_feasibility_table(pred_profiles)
        text = analyzer.generate_section4_therefore(pe_table, vr_data, feasibility)
        # All 3 combos have mde_da=0.523 < breakeven_da=0.625 => feasible
        assert "3/3" in text

    def test_empty_inputs(
        self,
        analyzer: RC2PredictabilityAnalyzer,
    ) -> None:
        """Empty DataFrames should still produce a valid paragraph."""
        text = analyzer.generate_section4_therefore(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert "**Therefore:**" in text
        assert "R5" in text
        # Should NOT crash with KeyError
        assert "Permutation entropy" not in text  # no data to summarize
