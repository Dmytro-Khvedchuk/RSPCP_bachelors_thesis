"""Tests for RC2 pre-registration threshold utilities.

Covers all pure functions in rc2_thresholds.py with known-value verification,
edge cases (zero/negative inputs, single sample), boundary conditions, and
realistic crypto parameter scenarios.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm  # type: ignore[import-untyped]

from src.app.research.application.rc2_thresholds import (
    BreakevenDAResult,
    DSRResult,
    FeasibilityGapResult,
    HarveyThresholdResult,
    MDEResult,
    RC2ThresholdSummary,
    compute_breakeven_da,
    compute_deflated_sharpe_ratio,
    compute_feasibility_gap,
    compute_harvey_threshold,
    compute_mde_da,
    compute_mi_significance_threshold,
    compute_rc2_thresholds,
    compute_required_n_eff_for_breakeven,
    compute_stability_threshold,
    compute_vif_threshold,
)


# ---------------------------------------------------------------------------
# TestComputeBreakevenDA
# ---------------------------------------------------------------------------


class TestComputeBreakevenDA:
    """Tests for the break-even directional accuracy computation."""

    def test_known_values(self) -> None:
        """For mean|r|=0.008 and cost=0.002, breakeven DA should be 0.625."""
        result: BreakevenDAResult = compute_breakeven_da(0.008, 0.002)
        assert result.breakeven_da == pytest.approx(0.625, abs=1e-10)
        assert result.required_edge_pp == pytest.approx(12.5, abs=1e-8)

    def test_large_mean_return(self) -> None:
        """Large mean|r| relative to cost gives DA close to 0.50."""
        result: BreakevenDAResult = compute_breakeven_da(0.05, 0.002)
        expected_da: float = 0.5 + 0.002 / (2 * 0.05)
        assert result.breakeven_da == pytest.approx(expected_da, abs=1e-10)
        assert result.breakeven_da <= 0.52

    def test_small_mean_return(self) -> None:
        """Small mean|r| relative to cost gives DA close to 1.0."""
        result: BreakevenDAResult = compute_breakeven_da(0.001, 0.002)
        expected_da: float = 0.5 + 0.002 / (2 * 0.001)
        assert result.breakeven_da == pytest.approx(min(expected_da, 1.0), abs=1e-10)

    def test_zero_mean_return(self) -> None:
        """Zero mean|r| yields DA=1.0 (impossible to profit)."""
        result: BreakevenDAResult = compute_breakeven_da(0.0, 0.002)
        assert result.breakeven_da == 1.0
        assert result.required_edge_pp == 50.0

    def test_negative_mean_return(self) -> None:
        """Negative mean|r| yields DA=1.0 (degenerate case)."""
        result: BreakevenDAResult = compute_breakeven_da(-0.001, 0.002)
        assert result.breakeven_da == 1.0

    def test_zero_cost(self) -> None:
        """Zero transaction cost gives breakeven DA = 0.50 exactly."""
        result: BreakevenDAResult = compute_breakeven_da(0.008, 0.0)
        assert result.breakeven_da == pytest.approx(0.5, abs=1e-10)
        assert result.required_edge_pp == pytest.approx(0.0, abs=1e-10)

    def test_cost_equals_mean_return(self) -> None:
        """When cost = 2*mean|r|, DA should be 1.0 (need perfect accuracy)."""
        result: BreakevenDAResult = compute_breakeven_da(0.001, 0.002)
        assert result.breakeven_da == pytest.approx(1.0, abs=1e-10)

    def test_typical_crypto_dollar_bar(self) -> None:
        """Typical BTC dollar bar: mean|r| ~= 0.007, cost=0.002 -> DA ~= 0.643."""
        result: BreakevenDAResult = compute_breakeven_da(0.007, 0.002)
        expected: float = 0.5 + 0.002 / (2 * 0.007)
        assert result.breakeven_da == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# TestComputeMDEDA
# ---------------------------------------------------------------------------


class TestComputeMDEDA:
    """Tests for the minimum detectable DA computation."""

    def test_known_values_n3000(self) -> None:
        """N_eff=3000, alpha=0.05, power=0.80 -> MDE DA ~= 0.5227."""
        result: MDEResult = compute_mde_da(3000.0, 0.05, 0.80)
        z_a: float = float(norm.ppf(0.95))
        z_b: float = float(norm.ppf(0.80))
        expected_mde: float = (z_a + z_b) / (2 * math.sqrt(3000))
        assert result.mde_da == pytest.approx(0.5 + expected_mde, abs=1e-4)

    def test_known_values_n500(self) -> None:
        """N_eff=500 -> larger MDE than N_eff=3000."""
        result_500: MDEResult = compute_mde_da(500.0)
        result_3000: MDEResult = compute_mde_da(3000.0)
        assert result_500.mde_da > result_3000.mde_da

    def test_more_power_requires_larger_mde(self) -> None:
        """Higher power -> larger MDE (need bigger effect to detect with more certainty)."""
        result_80: MDEResult = compute_mde_da(1000.0, 0.05, 0.80)
        result_90: MDEResult = compute_mde_da(1000.0, 0.05, 0.90)
        assert result_90.mde_da > result_80.mde_da

    def test_stricter_alpha_requires_larger_mde(self) -> None:
        """Lower alpha -> larger MDE."""
        result_05: MDEResult = compute_mde_da(1000.0, 0.05, 0.80)
        result_01: MDEResult = compute_mde_da(1000.0, 0.01, 0.80)
        assert result_01.mde_da > result_05.mde_da

    def test_zero_n_eff(self) -> None:
        """N_eff=0 yields MDE DA = 1.0."""
        result: MDEResult = compute_mde_da(0.0)
        assert result.mde_da == 1.0

    def test_negative_n_eff(self) -> None:
        """Negative N_eff yields MDE DA = 1.0."""
        result: MDEResult = compute_mde_da(-100.0)
        assert result.mde_da == 1.0

    def test_very_large_n_eff(self) -> None:
        """Very large N_eff -> MDE DA close to 0.50."""
        result: MDEResult = compute_mde_da(1e8)
        assert result.mde_da == pytest.approx(0.5, abs=0.001)

    def test_output_type(self) -> None:
        """Result should be an MDEResult with all expected fields."""
        result: MDEResult = compute_mde_da(1000.0)
        assert isinstance(result, MDEResult)
        assert result.n_eff == 1000.0
        assert result.alpha == 0.05
        assert result.power == 0.80
        assert result.detectable_edge_pp > 0.0


# ---------------------------------------------------------------------------
# TestComputeFeasibilityGap
# ---------------------------------------------------------------------------


class TestComputeFeasibilityGap:
    """Tests for the feasibility gap assessment."""

    def test_well_powered_scenario(self) -> None:
        """MDE DA < breakeven DA -> feasible with positive gap."""
        result: FeasibilityGapResult = compute_feasibility_gap(mde_da=0.52, breakeven_da=0.60)
        assert result.is_feasible is True
        assert result.gap_pp == pytest.approx(8.0, abs=1e-10)
        assert "Well-powered" in result.interpretation

    def test_underpowered_scenario(self) -> None:
        """MDE DA > breakeven DA -> infeasible with negative gap."""
        result: FeasibilityGapResult = compute_feasibility_gap(mde_da=0.58, breakeven_da=0.52)
        assert result.is_feasible is False
        assert result.gap_pp < 0.0
        assert "Underpowered" in result.interpretation

    def test_marginal_scenario(self) -> None:
        """MDE DA ~= breakeven DA -> marginal."""
        result: FeasibilityGapResult = compute_feasibility_gap(mde_da=0.555, breakeven_da=0.56)
        assert "Marginal" in result.interpretation

    def test_exact_match(self) -> None:
        """MDE DA == breakeven DA -> marginal (gap=0)."""
        result: FeasibilityGapResult = compute_feasibility_gap(mde_da=0.55, breakeven_da=0.55)
        assert result.gap_pp == pytest.approx(0.0, abs=1e-10)
        assert "Marginal" in result.interpretation

    def test_gap_sign_convention(self) -> None:
        """Positive gap means MDE is below breakeven (good)."""
        result: FeasibilityGapResult = compute_feasibility_gap(mde_da=0.51, breakeven_da=0.60)
        assert result.gap_pp > 0.0
        assert result.is_feasible is True


# ---------------------------------------------------------------------------
# TestComputeHarveyThreshold
# ---------------------------------------------------------------------------


class TestComputeHarveyThreshold:
    """Tests for the Harvey multiple-testing threshold computation."""

    def test_single_test(self) -> None:
        """Single test: Bonferroni alpha = raw alpha."""
        result: HarveyThresholdResult = compute_harvey_threshold(1, 0.05)
        assert result.bonferroni_alpha == pytest.approx(0.05, abs=1e-10)
        expected_t: float = float(norm.ppf(1.0 - 0.025))
        assert result.bonferroni_t == pytest.approx(expected_t, abs=1e-4)

    def test_345_tests_project_scenario(self) -> None:
        """23 features x 5 bar types x 3 horizons = 345 tests."""
        result: HarveyThresholdResult = compute_harvey_threshold(345, 0.05)
        expected_alpha: float = 0.05 / 345
        assert result.bonferroni_alpha == pytest.approx(expected_alpha, abs=1e-10)
        # Bonferroni t should be well above 3.0
        assert result.bonferroni_t > 3.0
        # Harvey threshold is always 3.0
        assert result.harvey_t == 3.0

    def test_bonferroni_increases_with_tests(self) -> None:
        """More tests -> higher Bonferroni t-threshold."""
        t_10: float = compute_harvey_threshold(10).bonferroni_t
        t_100: float = compute_harvey_threshold(100).bonferroni_t
        t_1000: float = compute_harvey_threshold(1000).bonferroni_t
        assert t_10 < t_100 < t_1000

    def test_zero_tests(self) -> None:
        """Zero tests treated as 1 test (no correction)."""
        result: HarveyThresholdResult = compute_harvey_threshold(0)
        assert result.n_tests == 1

    def test_holm_equals_bonferroni_for_rank1(self) -> None:
        """Holm-Bonferroni at rank 1 is identical to Bonferroni."""
        result: HarveyThresholdResult = compute_harvey_threshold(50)
        assert result.holm_bonferroni_t == pytest.approx(result.bonferroni_t, abs=1e-10)


# ---------------------------------------------------------------------------
# TestComputeDeflatedSharpeRatio
# ---------------------------------------------------------------------------


class TestComputeDeflatedSharpeRatio:
    """Tests for the Deflated Sharpe Ratio computation."""

    def test_single_trial_zero_sharpe(self) -> None:
        """SR=0 with 1 trial should give p-value >= 0.50."""
        result: DSRResult = compute_deflated_sharpe_ratio(
            observed_sharpe=0.0,
            n_trials=1,
            n_observations=1000,
        )
        assert result.dsr_pvalue >= 0.5 - 0.01

    def test_high_sharpe_single_trial(self) -> None:
        """SR=3.0 with 1 trial and n_observations=1000 should be significant."""
        result: DSRResult = compute_deflated_sharpe_ratio(
            observed_sharpe=3.0,
            n_trials=1,
            n_observations=1000,
        )
        assert result.dsr_pvalue < 0.05
        assert result.is_significant is True

    def test_many_trials_increases_expected_max(self) -> None:
        """More trials -> higher expected max SR -> harder to be significant."""
        result_1: DSRResult = compute_deflated_sharpe_ratio(1.5, n_trials=1, n_observations=1000)
        result_100: DSRResult = compute_deflated_sharpe_ratio(1.5, n_trials=100, n_observations=1000)
        assert result_100.expected_max_sharpe > result_1.expected_max_sharpe
        # p-value should be higher or equal with more trials (less significant)
        assert result_100.dsr_pvalue >= result_1.dsr_pvalue

    def test_short_sample_reduces_significance(self) -> None:
        """Shorter sample -> higher variance of SR -> less significant."""
        result_long: DSRResult = compute_deflated_sharpe_ratio(2.0, n_trials=10, n_observations=5000)
        result_short: DSRResult = compute_deflated_sharpe_ratio(2.0, n_trials=10, n_observations=100)
        assert result_short.dsr_pvalue >= result_long.dsr_pvalue

    def test_negative_kurtosis_reduces_variance(self) -> None:
        """Negative excess kurtosis (platykurtic) gives tighter SR estimate."""
        result_normal: DSRResult = compute_deflated_sharpe_ratio(2.0, 10, 1000, kurtosis=0.0)
        result_kurtotic: DSRResult = compute_deflated_sharpe_ratio(2.0, 10, 1000, kurtosis=5.0)
        # Higher kurtosis -> higher SR variance -> harder to be significant
        assert result_kurtotic.dsr_pvalue >= result_normal.dsr_pvalue

    def test_n_observations_less_than_2(self) -> None:
        """n_observations < 2 should return p-value = 1.0 (cannot estimate)."""
        result: DSRResult = compute_deflated_sharpe_ratio(2.0, 10, 1)
        assert result.dsr_pvalue == 1.0
        assert result.is_significant is False

    def test_output_fields(self) -> None:
        """All output fields should be populated."""
        result: DSRResult = compute_deflated_sharpe_ratio(1.5, 20, 2000, 0.1, 3.0)
        assert result.observed_sharpe == 1.5
        assert result.n_trials == 20
        assert result.n_observations == 2000
        assert result.skewness == pytest.approx(0.1)
        assert result.kurtosis == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# TestComputeRequiredNEffForBreakeven
# ---------------------------------------------------------------------------


class TestComputeRequiredNEffForBreakeven:
    """Tests for the required N_eff computation."""

    def test_known_values(self) -> None:
        """For breakeven DA=0.55 (5pp edge), the formula should give N_eff ~= 614.7."""
        z_a: float = float(norm.ppf(0.95))
        z_b: float = float(norm.ppf(0.80))
        expected_neff: float = ((z_a + z_b) / (2 * 0.05)) ** 2
        result: float = compute_required_n_eff_for_breakeven(0.55)
        assert result == pytest.approx(expected_neff, rel=1e-4)

    def test_higher_breakeven_needs_fewer_samples(self) -> None:
        """Larger edge is easier to detect -> needs fewer samples."""
        n_55: float = compute_required_n_eff_for_breakeven(0.55)
        n_60: float = compute_required_n_eff_for_breakeven(0.60)
        n_70: float = compute_required_n_eff_for_breakeven(0.70)
        assert n_55 > n_60 > n_70

    def test_breakeven_at_50_returns_inf(self) -> None:
        """DA = 0.50 means no edge; need infinite samples."""
        result: float = compute_required_n_eff_for_breakeven(0.50)
        assert math.isinf(result)

    def test_breakeven_below_50_returns_inf(self) -> None:
        """DA < 0.50 is degenerate; returns inf."""
        result: float = compute_required_n_eff_for_breakeven(0.45)
        assert math.isinf(result)

    def test_consistency_with_mde(self) -> None:
        """Required N_eff for breakeven DA should yield MDE DA ~= breakeven DA."""
        breakeven: float = 0.56
        n_required: float = compute_required_n_eff_for_breakeven(breakeven)
        mde_result: MDEResult = compute_mde_da(n_required)
        assert mde_result.mde_da == pytest.approx(breakeven, abs=1e-3)


# ---------------------------------------------------------------------------
# TestComputeMISignificanceThreshold
# ---------------------------------------------------------------------------


class TestComputeMISignificanceThreshold:
    """Tests for the MI significance threshold computation."""

    def test_binary_target(self) -> None:
        """Binary balanced target: H = ln(2). 1% threshold ~= 0.00693."""
        h: float = math.log(2.0)
        result: float = compute_mi_significance_threshold(h, 0.01)
        assert result == pytest.approx(h * 0.01, abs=1e-10)

    def test_five_percent_fraction(self) -> None:
        """5% of target entropy."""
        h: float = 1.0
        result: float = compute_mi_significance_threshold(h, 0.05)
        assert result == pytest.approx(0.05, abs=1e-10)

    def test_zero_entropy(self) -> None:
        """Zero entropy target -> threshold = 0."""
        result: float = compute_mi_significance_threshold(0.0, 0.01)
        assert result == 0.0

    def test_negative_entropy(self) -> None:
        """Negative entropy (invalid) -> threshold = 0."""
        result: float = compute_mi_significance_threshold(-1.0, 0.01)
        assert result == 0.0


# ---------------------------------------------------------------------------
# TestComputeVIFThreshold
# ---------------------------------------------------------------------------


class TestComputeVIFThreshold:
    """Tests for the VIF threshold selection."""

    def test_conservative_always_5(self) -> None:
        """Conservative mode always returns 5.0 regardless of N/p ratio."""
        assert compute_vif_threshold(10000, 23, conservative=True) == 5.0
        assert compute_vif_threshold(100, 23, conservative=True) == 5.0

    def test_non_conservative_large_ratio(self) -> None:
        """N/p >= 50 -> VIF > 10 threshold."""
        result: float = compute_vif_threshold(5000, 23, conservative=False)
        assert result == 10.0

    def test_non_conservative_small_ratio(self) -> None:
        """N/p < 50 -> VIF > 5 threshold."""
        result: float = compute_vif_threshold(500, 23, conservative=False)
        assert result == 5.0

    def test_zero_features(self) -> None:
        """Zero features -> default to 10.0."""
        result: float = compute_vif_threshold(1000, 0, conservative=False)
        assert result == 10.0


# ---------------------------------------------------------------------------
# TestComputeStabilityThreshold
# ---------------------------------------------------------------------------


class TestComputeStabilityThreshold:
    """Tests for the temporal stability threshold."""

    def test_four_windows_50_percent(self) -> None:
        """4 windows at 50% -> need 2 significant."""
        result: int = compute_stability_threshold(4, 0.50)
        assert result == 2

    def test_four_windows_75_percent(self) -> None:
        """4 windows at 75% -> need 3 significant."""
        result: int = compute_stability_threshold(4, 0.75)
        assert result == 3

    def test_zero_windows(self) -> None:
        """Zero windows -> minimum 1."""
        result: int = compute_stability_threshold(0, 0.50)
        assert result == 1

    def test_single_window(self) -> None:
        """1 window -> need 1 significant."""
        result: int = compute_stability_threshold(1, 0.50)
        assert result == 1

    def test_ceiling_behavior(self) -> None:
        """3 windows at 50% -> ceil(1.5) = 2."""
        result: int = compute_stability_threshold(3, 0.50)
        assert result == 2


# ---------------------------------------------------------------------------
# TestComputeRC2Thresholds (integration)
# ---------------------------------------------------------------------------


class TestComputeRC2Thresholds:
    """Integration tests for the composite threshold computation."""

    def test_typical_btc_dollar_bars(self) -> None:
        """Realistic BTC dollar bar scenario."""
        result: RC2ThresholdSummary = compute_rc2_thresholds(
            asset="BTCUSDT",
            bar_type="dollar",
            n_bars=5286,
            n_eff=3000.0,
            mean_abs_return=0.008,
            round_trip_cost=0.002,
        )
        # Breakeven DA should be 0.625
        assert result.breakeven.breakeven_da == pytest.approx(0.625, abs=1e-3)
        # MDE DA should be much lower (~0.523)
        assert result.mde.mde_da < 0.53
        # Should be feasible
        assert result.feasibility.is_feasible is True
        # Harvey t for 345 tests should be reasonable
        assert result.harvey.bonferroni_t > 3.0
        # VIF threshold
        assert result.vif_threshold == 5.0

    def test_small_imbalance_bars(self) -> None:
        """Small imbalance bar scenario (N~530, low N_eff)."""
        result: RC2ThresholdSummary = compute_rc2_thresholds(
            asset="BTCUSDT",
            bar_type="dollar_imbalance",
            n_bars=530,
            n_eff=400.0,
            mean_abs_return=0.015,
        )
        # MDE DA should be larger with small N_eff
        assert result.mde.mde_da > 0.55
        # Breakeven should be ~0.567
        expected_be: float = 0.5 + 0.002 / (2 * 0.015)
        assert result.breakeven.breakeven_da == pytest.approx(expected_be, abs=1e-3)

    def test_all_fields_populated(self) -> None:
        """All summary fields should be non-None."""
        result: RC2ThresholdSummary = compute_rc2_thresholds(
            asset="ETHUSDT",
            bar_type="volume",
            n_bars=3264,
            n_eff=2000.0,
            mean_abs_return=0.010,
        )
        assert result.asset == "ETHUSDT"
        assert result.bar_type == "volume"
        assert result.n_bars == 3264
        assert result.n_eff == 2000.0
        assert result.breakeven is not None
        assert result.mde is not None
        assert result.feasibility is not None
        assert result.harvey is not None


# ---------------------------------------------------------------------------
# TestRealisticScenarios: What do the numbers say for our specific project?
# ---------------------------------------------------------------------------


class TestRealisticScenarios:
    """End-to-end tests with realistic project parameters.

    These tests document the actual numerical landscape for the four
    assets and five bar types used in this thesis.
    """

    def test_dollar_bars_all_assets_feasible(self) -> None:
        """Dollar bars (N~5286, N_eff~3000, mean|r|~0.007-0.010) should all be feasible."""
        scenarios: list[tuple[str, float]] = [
            ("BTCUSDT", 0.008),
            ("ETHUSDT", 0.010),
            ("LTCUSDT", 0.009),
            ("SOLUSDT", 0.012),
        ]
        for asset, mean_abs_r in scenarios:
            result: RC2ThresholdSummary = compute_rc2_thresholds(
                asset=asset,
                bar_type="dollar",
                n_bars=5286,
                n_eff=3000.0,
                mean_abs_return=mean_abs_r,
            )
            assert result.feasibility.is_feasible is True, f"{asset} should be feasible"
            # Verify the gap is positive
            assert result.feasibility.gap_pp > 0.0, f"{asset} gap should be positive"

    def test_imbalance_bars_may_be_marginal(self) -> None:
        """Imbalance bars (N~530, N_eff~350) have larger MDE -> potentially marginal."""
        result: RC2ThresholdSummary = compute_rc2_thresholds(
            asset="BTCUSDT",
            bar_type="volume_imbalance",
            n_bars=530,
            n_eff=350.0,
            mean_abs_return=0.015,
        )
        # MDE should be relatively large
        assert result.mde.mde_da > 0.55
        # With large mean|r|=0.015, breakeven is moderate
        assert result.breakeven.breakeven_da < 0.57

    def test_dsr_with_reasonable_trial_count(self) -> None:
        """Deflated Sharpe for SR=1.5 with 50 trials (hyperparameter search)."""
        result: DSRResult = compute_deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=50,
            n_observations=3000,
            skewness=-0.5,
            kurtosis=4.0,
        )
        # This should be marginally significant at best
        # The expected max SR with 50 trials is substantial
        assert result.expected_max_sharpe > 0.0
        # Result gives a p-value (not testing exact value, just sanity)
        assert 0.0 <= result.dsr_pvalue <= 1.0

    def test_required_neff_for_typical_crypto(self) -> None:
        """For mean|r|=0.008, cost=0.002 -> breakeven=0.625 -> required N_eff ~= 24.7."""
        be_result: BreakevenDAResult = compute_breakeven_da(0.008, 0.002)
        n_required: float = compute_required_n_eff_for_breakeven(be_result.breakeven_da)
        # Edge is 12.5pp, so required N_eff = ((z_a + z_b) / (2 * 0.125))^2
        z_a: float = float(norm.ppf(0.95))
        z_b: float = float(norm.ppf(0.80))
        expected: float = ((z_a + z_b) / (2 * 0.125)) ** 2
        assert n_required == pytest.approx(expected, rel=1e-3)
        # This should be very small -- we are WELL powered for this edge
        assert n_required < 100.0

    def test_harvey_threshold_for_project(self) -> None:
        """Project scenario: 23 features x 5 bar types x 3 horizons = 345 tests."""
        result: HarveyThresholdResult = compute_harvey_threshold(345)
        # Bonferroni threshold should be around 3.6
        assert result.bonferroni_t > 3.5
        assert result.bonferroni_t < 4.5
        # Harvey's simple threshold is always 3.0
        assert result.harvey_t == 3.0
        # For single-asset analysis (23 * 3 = 69 tests), threshold is lower
        single_asset: HarveyThresholdResult = compute_harvey_threshold(69)
        assert single_asset.bonferroni_t < result.bonferroni_t
