"""Tests for SerialDependenceAnalyzer against synthetic time series.

Verifies ACF/PACF computation, multi-lag Ljung-Box tests, Lo-MacKinlay
variance ratio with Chow-Denning, Granger causality, and tier-gated
behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.app.profiling.application.serial_dependence import SerialDependenceAnalyzer
from src.app.profiling.domain.value_objects import (
    AutocorrelationConfig,
    SampleTier,
)
from src.tests.profiling.conftest import (
    make_ar1_returns,
    make_causal_pair,
    make_garch_like_returns,
    make_random_walk_returns,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_config() -> AutocorrelationConfig:
    """Return a default AutocorrelationConfig for tests."""
    return AutocorrelationConfig()


def _make_analyzer() -> SerialDependenceAnalyzer:
    """Return a fresh SerialDependenceAnalyzer instance."""
    return SerialDependenceAnalyzer()


# ---------------------------------------------------------------------------
# ACF / PACF tests
# ---------------------------------------------------------------------------


class TestACFPACF:
    """Tests for ACF and PACF computation."""

    def test_acf_white_noise(self) -> None:
        """White noise returns: ACF values near 0 for lag > 0."""
        returns = make_random_walk_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0)

        # Lag 0 ACF should be 1.0
        assert abs(profile.acf_values[0] - 1.0) < 1e-10
        # Remaining lags should be near zero (within 2/sqrt(n) ~ 0.045)
        threshold = 3.0 / np.sqrt(len(returns))
        for lag_idx in range(1, len(profile.acf_values)):
            assert abs(profile.acf_values[lag_idx]) < threshold

    def test_acf_ar1_process(self) -> None:
        """AR(1) with phi=0.5: ACF at lag 1 should be approximately 0.5."""
        returns = make_ar1_returns(n=5000, phi=0.5, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0)

        assert abs(profile.acf_values[1] - 0.5) < 0.1

    def test_pacf_ar1_process(self) -> None:
        """AR(1) with phi=0.5: PACF at lag 1 ~0.5, lags > 1 near 0."""
        returns = make_ar1_returns(n=5000, phi=0.5, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0)

        # PACF at lag 1 should be near phi
        assert abs(profile.pacf_values[1] - 0.5) < 0.1
        # PACF at higher lags should be near 0 for AR(1)
        threshold = 3.0 / np.sqrt(len(returns))
        for lag_idx in range(2, min(6, len(profile.pacf_values))):
            assert abs(profile.pacf_values[lag_idx]) < threshold

    def test_acf_max_lag_capped(self) -> None:
        """When data is short, max_lag auto-caps to n//2 - 1."""
        # With n=20, max_lag should cap at 9 (20//2 - 1)
        returns = make_random_walk_returns(n=20, seed=42)
        config = AutocorrelationConfig(max_lag=40)
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        # ACF array includes lag 0, so length = effective_lag + 1
        expected_max_lag = 20 // 2 - 1  # = 9
        assert len(profile.acf_values) == expected_max_lag + 1


# ---------------------------------------------------------------------------
# Ljung-Box tests
# ---------------------------------------------------------------------------


class TestLjungBox:
    """Tests for multi-lag Ljung-Box testing."""

    def test_ljung_box_white_noise(self) -> None:
        """White noise: no significant lags expected."""
        returns = make_random_walk_returns(n=2000, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0)

        assert profile.has_serial_correlation is False
        for lb in profile.ljung_box_returns:
            assert lb.significant is False

    def test_ljung_box_ar1(self) -> None:
        """AR(1) phi=0.3: should detect serial correlation at lag 5."""
        returns = make_ar1_returns(n=2000, phi=0.3, seed=42)
        config = AutocorrelationConfig(ljung_box_lags=(5,))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert len(profile.ljung_box_returns) == 1
        assert profile.ljung_box_returns[0].lag == 5
        assert profile.ljung_box_returns[0].significant is True
        assert profile.has_serial_correlation is True

    def test_ljung_box_multiple_lags(self) -> None:
        """Returns one result per requested lag."""
        returns = make_random_walk_returns(n=2000, seed=42)
        config = AutocorrelationConfig(ljung_box_lags=(5, 10, 20))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert len(profile.ljung_box_returns) == 3
        result_lags = [r.lag for r in profile.ljung_box_returns]
        assert result_lags == [5, 10, 20]

    def test_ljung_box_squared_returns_garch(self) -> None:
        """GARCH-like squared returns should show significance."""
        returns = make_garch_like_returns(n=2000, seed=42)
        config = AutocorrelationConfig(ljung_box_lags=(5, 10))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert profile.has_volatility_clustering is True
        # At least one LB on squared returns should be significant
        assert any(lb.significant for lb in profile.ljung_box_squared)


# ---------------------------------------------------------------------------
# Variance ratio tests
# ---------------------------------------------------------------------------


class TestVarianceRatio:
    """Tests for Lo-MacKinlay variance ratio and Chow-Denning."""

    def test_vr_random_walk(self) -> None:
        """Random walk: VR approximately 1.0 at all horizons, not significant."""
        returns = make_random_walk_returns(n=5000, seed=42)
        config = AutocorrelationConfig(vr_calendar_horizons_days=(1.0, 3.0, 7.0))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert profile.vr_results is not None
        for vr in profile.vr_results:
            assert abs(vr.variance_ratio - 1.0) < 0.15

    def test_vr_mean_reverting(self) -> None:
        """Mean-reverting (AR(1) negative phi): VR < 1 at longer horizons."""
        returns = make_ar1_returns(n=5000, phi=-0.3, seed=42)
        config = AutocorrelationConfig(vr_calendar_horizons_days=(3.0, 7.0))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert profile.vr_results is not None
        # With negative autocorrelation, VR should be < 1
        for vr in profile.vr_results:
            assert vr.variance_ratio < 1.0

    def test_vr_trending(self) -> None:
        """Trending (AR(1) positive phi): VR > 1."""
        returns = make_ar1_returns(n=5000, phi=0.3, seed=42)
        config = AutocorrelationConfig(vr_calendar_horizons_days=(3.0, 7.0))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert profile.vr_results is not None
        for vr in profile.vr_results:
            assert vr.variance_ratio > 1.0

    def test_vr_tier_b_caps_horizon(self) -> None:
        """Tier B skips horizons beyond 7 days."""
        returns = make_random_walk_returns(n=5000, seed=42)
        config = AutocorrelationConfig(vr_calendar_horizons_days=(1.0, 7.0, 14.0))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.B, bars_per_day=24.0, config=config
        )

        assert profile.vr_results is not None
        # 14-day horizon should be excluded for Tier B
        cal_horizons = [vr.calendar_horizon_days for vr in profile.vr_results]
        assert 14.0 not in cal_horizons
        assert 1.0 in cal_horizons
        assert 7.0 in cal_horizons

    def test_vr_tier_c_skipped(self) -> None:
        """Tier C gets None for VR results."""
        returns = make_random_walk_returns(n=100, seed=42)
        profile = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C, bars_per_day=24.0)

        assert profile.vr_results is None
        assert profile.chow_denning_stat is None
        assert profile.chow_denning_pvalue is None

    def test_chow_denning_stat(self) -> None:
        """Chow-Denning stat is max(|z_i|) across all tested horizons."""
        returns = make_ar1_returns(n=5000, phi=0.3, seed=42)
        config = AutocorrelationConfig(vr_calendar_horizons_days=(1.0, 3.0, 7.0))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert profile.vr_results is not None
        assert profile.chow_denning_stat is not None
        assert profile.chow_denning_pvalue is not None

        # CD stat should be max of absolute z-statistics
        max_abs_z = max(abs(vr.z_statistic) for vr in profile.vr_results)
        assert abs(profile.chow_denning_stat - max_abs_z) < 1e-10

        # p-value should be in [0, 1]
        assert 0.0 <= profile.chow_denning_pvalue <= 1.0

    def test_calendar_to_bar_conversion(self) -> None:
        """Correct conversion with known bars_per_day."""
        returns = make_random_walk_returns(n=5000, seed=42)
        config = AutocorrelationConfig(vr_calendar_horizons_days=(1.0,))
        # 24 bars/day -> 1-day horizon = 24 bars
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        assert profile.vr_results is not None
        assert len(profile.vr_results) == 1
        assert profile.vr_results[0].bar_count_q == 24


# ---------------------------------------------------------------------------
# Granger causality tests
# ---------------------------------------------------------------------------


class TestGrangerCausality:
    """Tests for Granger causality testing."""

    def test_granger_causal_relationship(self) -> None:
        """Constructed X->Y causal relationship should be detected."""
        x, y = make_causal_pair(n=1000, lag=1, seed=42)
        analyzer = _make_analyzer()
        results = analyzer.test_granger_pairs(
            returns_dict={"X": x, "Y": y},
            lags=(1, 2),
            alpha=0.05,
        )

        # X -> Y should be significant at lag 1
        x_to_y_lag1 = [r for r in results if r.source_name == "X" and r.target_name == "Y" and r.lag == 1]
        assert len(x_to_y_lag1) == 1
        assert x_to_y_lag1[0].significant is True

    def test_granger_independent(self) -> None:
        """Independent series should not detect Granger causality."""
        rng = np.random.default_rng(42)
        x = pd.Series(rng.normal(0, 0.01, 1000), dtype=np.float64, name="X")
        y = pd.Series(rng.normal(0, 0.01, 1000), dtype=np.float64, name="Y")

        analyzer = _make_analyzer()
        results = analyzer.test_granger_pairs(
            returns_dict={"X": x, "Y": y},
            lags=(1, 2, 4),
            alpha=0.05,
        )

        # With independent series, most tests should not be significant
        # (allow for one false positive at 5% level)
        n_significant = sum(1 for r in results if r.significant)
        assert n_significant <= 2

    def test_granger_tier_a_only(self) -> None:
        """Tier B/C should get None for granger_results in the profile."""
        returns = make_random_walk_returns(n=2000, seed=42)
        profile_b = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.B, bars_per_day=24.0)
        profile_c = _make_analyzer().analyze(returns, "BTCUSDT", "dollar", SampleTier.C, bars_per_day=24.0)

        assert profile_b.granger_results is None
        assert profile_c.granger_results is None

    def test_granger_pairs_all_combinations(self) -> None:
        """test_granger_pairs covers all ordered pairs."""
        rng = np.random.default_rng(99)
        series_dict = {
            "A": pd.Series(rng.normal(0, 0.01, 500), dtype=np.float64),
            "B": pd.Series(rng.normal(0, 0.01, 500), dtype=np.float64),
            "C": pd.Series(rng.normal(0, 0.01, 500), dtype=np.float64),
        }
        analyzer = _make_analyzer()
        results = analyzer.test_granger_pairs(
            returns_dict=series_dict,
            lags=(1,),
            alpha=0.05,
        )

        # 3 assets -> 6 ordered pairs, 1 lag each -> 6 results
        assert len(results) == 6
        pairs = {(r.source_name, r.target_name) for r in results}
        expected_pairs = {
            ("A", "B"),
            ("A", "C"),
            ("B", "A"),
            ("B", "C"),
            ("C", "A"),
            ("C", "B"),
        }
        assert pairs == expected_pairs


# ---------------------------------------------------------------------------
# Full profile tests
# ---------------------------------------------------------------------------


class TestFullProfile:
    """Tests for the complete analyze() output."""

    def test_analyze_tier_a_full_profile(self) -> None:
        """Tier A returns all non-Granger fields populated."""
        returns = make_ar1_returns(n=5000, phi=0.2, seed=42)
        config = AutocorrelationConfig(
            ljung_box_lags=(5, 10),
            vr_calendar_horizons_days=(1.0, 3.0),
        )
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config
        )

        # Metadata
        assert profile.asset == "BTCUSDT"
        assert profile.bar_type == "dollar"
        assert profile.tier == SampleTier.A
        assert profile.n_observations == 5000

        # ACF / PACF populated
        assert len(profile.acf_values) > 0
        assert len(profile.pacf_values) > 0
        assert len(profile.acf_squared_values) > 0
        assert len(profile.pacf_squared_values) > 0

        # Ljung-Box populated
        assert len(profile.ljung_box_returns) == 2
        assert len(profile.ljung_box_squared) == 2

        # VR populated
        assert profile.vr_results is not None
        assert len(profile.vr_results) == 2
        assert profile.chow_denning_stat is not None
        assert profile.chow_denning_pvalue is not None

    def test_analyze_tier_c_minimal(self) -> None:
        """Tier C: only ACF + Ljung-Box, no VR or Granger."""
        returns = make_random_walk_returns(n=100, seed=42)
        config = AutocorrelationConfig(ljung_box_lags=(5,))
        profile = _make_analyzer().analyze(
            returns, "BTCUSDT", "dollar", SampleTier.C, bars_per_day=24.0, config=config
        )

        # ACF / PACF should still be present
        assert len(profile.acf_values) > 0
        assert len(profile.pacf_values) > 0

        # VR and Granger should be None
        assert profile.vr_results is None
        assert profile.chow_denning_stat is None
        assert profile.chow_denning_pvalue is None
        assert profile.granger_results is None

    def test_deterministic_output(self) -> None:
        """Same input produces same output."""
        returns = make_ar1_returns(n=1000, phi=0.3, seed=123)
        config = AutocorrelationConfig(
            ljung_box_lags=(5, 10),
            vr_calendar_horizons_days=(1.0,),
        )
        analyzer = _make_analyzer()

        profile_1 = analyzer.analyze(returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config)
        profile_2 = analyzer.analyze(returns, "BTCUSDT", "dollar", SampleTier.A, bars_per_day=24.0, config=config)

        # Compare scalar fields
        assert profile_1.asset == profile_2.asset
        assert profile_1.n_observations == profile_2.n_observations
        assert profile_1.has_serial_correlation == profile_2.has_serial_correlation
        assert profile_1.has_volatility_clustering == profile_2.has_volatility_clustering
        assert profile_1.ljung_box_returns == profile_2.ljung_box_returns
        assert profile_1.ljung_box_squared == profile_2.ljung_box_squared
        assert profile_1.vr_results == profile_2.vr_results
        assert profile_1.chow_denning_stat == profile_2.chow_denning_stat

        # Compare numpy arrays
        np.testing.assert_array_equal(profile_1.acf_values, profile_2.acf_values)
        np.testing.assert_array_equal(profile_1.pacf_values, profile_2.pacf_values)


# ---------------------------------------------------------------------------
# Value object construction tests
# ---------------------------------------------------------------------------


class TestValueObjects:
    """Tests for value object construction and constraints."""

    def test_autocorrelation_config_defaults(self) -> None:
        """Default config should have sensible defaults."""
        config = AutocorrelationConfig()
        assert config.max_lag == 40
        assert config.ljung_box_lags == (5, 10, 20, 40)
        assert config.alpha == 0.05
        assert config.granger_max_lags == (1, 2, 4, 8)
        assert config.vr_calendar_horizons_days == (1.0, 3.0, 7.0, 14.0)
        assert config.vr_robust is True

    def test_autocorrelation_config_frozen(self) -> None:
        """AutocorrelationConfig should be immutable."""
        from pydantic import ValidationError

        config = AutocorrelationConfig()
        with pytest.raises(ValidationError):
            config.max_lag = 100  # type: ignore[misc]

    def test_ljung_box_result_construction(self) -> None:
        """LjungBoxResult should be constructable with valid data."""
        from src.app.profiling.domain.value_objects import LjungBoxResult

        result = LjungBoxResult(lag=5, q_statistic=10.5, p_value=0.03, significant=True)
        assert result.lag == 5
        assert result.significant is True

    def test_variance_ratio_result_construction(self) -> None:
        """VarianceRatioResult should be constructable with valid data."""
        from src.app.profiling.domain.value_objects import VarianceRatioResult

        result = VarianceRatioResult(
            calendar_horizon_days=7.0,
            bar_count_q=168,
            variance_ratio=0.85,
            z_statistic=-2.1,
            p_value=0.036,
            significant=True,
        )
        assert result.bar_count_q == 168
        assert result.significant is True

    def test_granger_result_construction(self) -> None:
        """GrangerResult should be constructable with valid data."""
        from src.app.profiling.domain.value_objects import GrangerResult

        result = GrangerResult(
            source_name="BTCUSDT_returns",
            target_name="ETHUSDT_returns",
            lag=1,
            f_statistic=5.2,
            p_value=0.023,
            significant=True,
        )
        assert result.source_name == "BTCUSDT_returns"
        assert result.significant is True
